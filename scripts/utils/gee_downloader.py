import xarray as xr
import ee
import geemap
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import os
import time
from shapely.geometry import Polygon

start = time.time()


# Initialize the Earth Engine API
try:
    ee.Initialize()
except Exception as e:
    # If initialization fails, authenticate
    print("Initializing Earth Engine...")
    ee.Authenticate()
    ee.Initialize()

# Define function to extract the acquisition date from Sentinel-2 product name
def get_date_from_product(s2_product):
    date_str = s2_product[11:19]  # Extract 'YYYYMMDD'
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    return f"{year}-{month:02d}-{day:02d}"

# Function to create a bounding box in lat-lon from (lat, lon) coordinates
def create_bounding_box(lat_centroid, lon_centroid, deg=0.09):
    # Define the size of the bounding box in degrees (approximately 10km)
    size = deg / 2  # Half the size to create a square bounding box
    
    # Calculate the corners of the bounding box
    coordinates = [
        [lon_centroid - size, lat_centroid - size],  # Bottom-left
        [lon_centroid + size, lat_centroid - size],  # Bottom-right
        [lon_centroid + size, lat_centroid + size],  # Top-right
        [lon_centroid - size, lat_centroid + size],  # Top-left
        [lon_centroid - size, lat_centroid - size]   # Closing the polygon
    ]
    
    # Create the bounding box geometry for Earth Engine
    #ee_bbox = ee.Geometry.Polygon(coordinates)

    return coordinates#ee_bbox

def fetch_and_process(collection, date, bands, bounding_box):
    """Fetch images from the specified collection and process them."""
    images = collection.filterDate(date, date.advance(1, 'day')).filterBounds(bounding_box)
    count = images.size().getInfo()

    # Select the bands you need and filter for those that contain them
    # We filter them here as band order can vary slightly between images (auxillary bands: QA10, QA20, QA60, MSK_CLASSI_*)
    # Without this step, the download of these heterogeneous images would fail
    images = images.select(bands)

    # Slightly convoluted; but basically sometimes the bounding box overlaps with several Sentinel-2 tiles
    # If this is the case (i.e., count > 1), then mosaic them into a single ee.Image object
    # In the other case, we simply return the single image
    if count > 0:
        # Get the native CRS from the first image in the collection
        crs = images.first().select(0).projection().crs()

        if count > 1:
            print("Mosaicking multiple images.") # TODO: Check if the mosaic + reprojection is necessary without downloading the images
            mosaic_image = images.mosaic()

            # Reproject the mosaic back to native CRS
            # TODO: For some reason download_ee_image() does not work with this activated
            #mosaic_image = mosaic_image.reproject(crs=crs)

            # Reproject the mosaic back to the original CRS
            return mosaic_image, crs
        else:
            print(f"Found single image: {images.first().get('system:id').getInfo()}")
            return images.first(), crs
    return None, None

def calculate_ndi(image, band1='B2', band2='B8', ndi_name='NDI_B2B8'):
    """
    Calculate the Normalized Difference Index (NDI) between two bands and return it as a separate image.
    Args:
        image: The Sentinel-2 image.
        band1: The first band for the NDI calculation (e.g., 'B2').
        band2: The second band for the NDI calculation (e.g., 'B8').
        ndi_name: Name of the resulting NDI band.
    Returns:
        An Earth Engine image containing only the NDI band.
    """
    ndi = image.expression(
        "(B8 - B2) / (B8 + B2)",
        {'B8': image.select(band2), 'B2': image.select(band1)}
    ).rename(ndi_name)

    # Mask invalid values
    #ndi = ndi.updateMask(ndi.neq(None).And(ndi.gt(-1)).And(ndi.lt(1)))

    return ndi

def calculate_fdi(image, nir_band='B8', re2_band='B6', swir1_band='B11', fdi_name='FDI'):
    """
    Calculate the Floating Debris Index (FDI) using Sentinel-2 bands.
    
    Args:
        image: The Sentinel-2 image.
        nir_band: Near Infrared (NIR) band (e.g., 'B8').
        re2_band: Red Edge 2 band (e.g., 'B6').
        swir1_band: Shortwave Infrared 1 (SWIR1) band (e.g., 'B11').
        fdi_name: Name of the resulting FDI band.
    
    Returns:
        An Earth Engine image containing only the FDI band.
    """
    # Convert DN to reflectance by dividing by 10,000
    nir = image.select(nir_band).divide(10000)   # B8 (NIR)
    re2 = image.select(re2_band).divide(10000)  # B6 (Red Edge)
    swir1 = image.select(swir1_band).divide(10000)  # B11 (SWIR1)
    
    # Central wavelengths for Sentinel-2
    lambda_nir = 832.8
    lambda_swir1 = 1613.7
    lambda_red = 664.6

    # Calculate NIR prime
    nir_prime = image.expression(
        "re2 + (swir1 - re2) * ((lambda_nir - lambda_red) / (lambda_swir1 - lambda_red)) * 10",
        {
            're2': re2,#image.select(re2_band),
            'swir1': swir1,#image.select(swir1_band),
            'lambda_nir': lambda_nir,
            'lambda_red': lambda_red,
            'lambda_swir1': lambda_swir1
        }
    )

    # Calculate FDI
    fdi = nir.subtract(nir_prime).rename(fdi_name)

    return fdi

def download_single_s2img(s2_product, lat_centroid, lon_centroid, out_dir):
    """
    Downloads a single Sentinel-2 image, checking if the image already exists locally.
    If L2A data is unavailable, it falls back to L1C data.
    
    Args:
        s2_product (str): Sentinel-2 product name in .SAFE format.
        lat_centroid (float): Latitude of the center point.
        lon_centroid (float): Longitude of the center point.
        out_dir (str): Directory to save the downloaded image.
    
    Returns:
        None. The image is saved as a GeoTIFF file in the specified directory.
    """
    # Generate the output filename by cleaning the product name
    file_name = s2_product.replace(".SAFE", ".tif")  # Clean the filename

    # Build potential file paths for L2A and L1C data
    l2a_path = os.path.join(out_dir, f"L2A_{file_name}")
    l1c_path = os.path.join(out_dir, f"L1C_{file_name}")

    # Skip download if either file already exists
    if os.path.exists(l2a_path) or os.path.exists(l1c_path):
        print(f"File '{file_name}' already exists. Skipping download.")
        return  # Skip processing if either file exists

    print('\n' + '#'*40)
    print(f"Processing file: '{file_name}'")
    
    # Load Sentinel-2 image collections (L2A and L1C)
    sentinel2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') # L2A data
    sentinel2_toa = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')   # L1C data

    # Extract the acquisition date from the product name
    date = ee.Date(get_date_from_product(s2_product))

    # Create the bounding box (lat-lon; EPSG:4326)
    coordinates = create_bounding_box(lat_centroid, lon_centroid, deg=0.225) # ~25km
    bounding_box = ee.Geometry.Polygon(coordinates)

    # Save the bounding box as a shapefile (GeoDataFrame)
    bbox_gdf = gpd.GeoDataFrame(
        {'geometry': [Polygon(coordinates)]},
        crs="EPSG:4326"
    )
    bbox_gdf.to_file(os.path.join(out_dir, 'bounding_box.shp')) # Save as .shp as this is required for Planet Explorer

    # Save the centroid as a GeoPackage
    centroid_gdf = gpd.GeoDataFrame(
        {'geometry': [Point(lon_centroid, lat_centroid)]},
        crs="EPSG:4326"
    )
    centroid_gdf.to_file(os.path.join(out_dir, 'centroid.gpkg'), driver='GPKG')

    # Attempt to get fetch and process the L2A image
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    sentinel2, crs = fetch_and_process(sentinel2_sr, date, bands, bounding_box)
    prefix = 'L2A' # Default to L2A prefix for exporting

    # If L2A is not available, try fetching the L1C image
    if sentinel2 is None:
        sentinel2, crs = fetch_and_process(sentinel2_toa, date, bands, bounding_box)
        prefix = 'L1C'

    # If an image was found, export it to the output directory the Sentinel-2 tile
    if sentinel2:
        # Define the filename and export path
        os.makedirs(out_dir, exist_ok=True)
        
        # Calculate NDI_B2B8 and Floating Debris Index
        ndi = calculate_ndi(sentinel2)
        fdi = calculate_fdi(sentinel2)
        
        sentinel2_with_indices  = sentinel2.addBands([ndi, fdi])

        # We re-use the name given in the NetCDF as 's2_product' for quick matching between the filaments and the newly downloaded .tif files
        # Since we also include L2A data now (default is L1C in original predictions), we append the processing level as prefix
        # Files go from 'S2A_MSIL1C_20160217T111122_N0201_R137_T29SQA_20160217T111843.SAFE' => 'L1C_S2A_MSIL1C_20160217T111122_N0201_R137_T29SQA_20160217T111843.tif'
        export_path = os.path.join(out_dir, f"{prefix}_{file_name}")

        geemap.download_ee_image(sentinel2_with_indices,
                               filename=export_path,
                               scale=10, # Set pixel size to 10m
                               region=bounding_box,
                               crs=crs,
                               overwrite=True,
                               )
        
    print('#'*40)

def download_multiple_s2img(start_date, end_date, lat_centroid, lon_centroid, out_dir="data/plp_tiles"):
    """
    Downloads multiple Sentinel-2 L2A images within a date range, skipping any already existing files.
    
    Args:
        start_date (str): Start date for the image search in 'YYYY-MM-DD' format.
        end_date (str): End date for the image search in 'YYYY-MM-DD' format.
        lat_centroid (float): Latitude of the center point.
        lon_centroid (float): Longitude of the center point.
        out_dir (str): Directory to save the downloaded images.
    
    Returns:
        None. All images are saved as GeoTIFF files in the specified directory.
    """
    # Create the bounding box for the area of interest (approx. 10km)
    bounding_box = ee.Geometry.Polygon(create_bounding_box(lat_centroid, lon_centroid))

    # Load the Sentinel-2 L2A image collection and filter by date + location
    l2a_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterDate(start_date, end_date) \
                        .filterBounds(bounding_box)
    
    # Load the Sentinel-2 L1C image collection as fallback option
    l1c_collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                    .filterDate(start_date, end_date) \
                    .filterBounds(bounding_box)

    # Get the number of images in the L2A collection
    image_count = l2a_collection.size().getInfo()

    # Fallback to L1C if no L2A images are available
    if image_count == 0:
        print(f"No L2A images found between {start_date} and {end_date}. Attempting to download L1C images.")
        image_count = l1c_collection.size().getInfo()
        
        print(f"Found {image_count} L1C images between {start_date} and {end_date}.")
        image_collection = l1c_collection
        prefix = "L1C"
    else:
        print(f"Found {image_count} L2A images between {start_date} and {end_date}.")
        image_collection = l2a_collection
        prefix = "L2A"

    # Iterate over each image in the filtered ImageCollection
    images = image_collection.toList(image_count)

    # Loop through each image and download it
    for i in range(image_count):
        image = ee.Image(images.get(i))

        # Select Sentinel-2 L2A bands
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        image = image.select(bands)

        # Extract the acquisition date for naming the file
        acquisition_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        export_path = os.path.join(out_dir, f"{prefix}_{acquisition_date}.tif")

        # Check if the image has already been downloaded
        #if os.path.exists(export_path):
        #    print(f"File '{export_path}' already exists. Skipping.")
        #    continue
        print(f"Processing image for date: {acquisition_date}")

        # Create the output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        
        # Get native CRS for export
        crs = image.select(0).projection().crs().getInfo()

        # Export the image as a GeoTIFF file
        geemap.download_ee_image(image,
                        filename=export_path,
                        scale=10, # Set pixel size to 10m
                        region=bounding_box,
                        crs=crs,
                        overwrite=True
                        )
        print(f"Image exported: {export_path}")
    print("All images downloaded")

