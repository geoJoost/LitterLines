import sys, os # TODO: Remove. Kept for debugging
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

import os
import numpy as np
import geemap
import ee
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pyproj
import time
import glob

# Custom modules
from utils.gee_downloader import get_date_from_product, fetch_and_process, create_bounding_box, download_single_s2img

start = time.time()

def reflectance_per_filament(s2_product, lat_centroid, lon_centroid, x_centroid, y_centroid, x_locations, y_locations, export_path):
    print('\n' + '#'*40)
    print(f"Processing product: {export_path}")
    
    # Find Sentinel-2 data for the given date
    sentinel2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')# L2A data
    sentinel2_toa = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')  # L1C data

    # Extract the acquisition date
    date = get_date_from_product(s2_product)

    # Create the bounding box (lat-lon; EPSG:4326)
    bounding_box = create_bounding_box(lat_centroid, lon_centroid)

    # Attempt to get L2A images first
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    sentinel2, crs = fetch_and_process(sentinel2_sr, date, bands, bounding_box)
    prefix = 'L2A' # Default to L2A prefix for exporting

    # Fallback to L1C if no L2A images are found
    if sentinel2 is None:
        sentinel2, crs = fetch_and_process(sentinel2_toa, date, bands, bounding_box)
        prefix = 'L1C'

    if sentinel2:
        """ Cartesian coordinates into geographic coordinates"""
        # TODO: This whole part needs to be redone. Points are not in the correct location, yet

        # Assumed pixel size (10 meters per pixel)
        pixel_size_m = 10

        # Create a transformer from EPSG:4326 (Geographic) to EPSG:32629 (UTM)
        geo_to_utm = pyproj.Transformer.from_crs("EPSG:4326", crs.getInfo(), always_xy=True)

        # Get UTM coordinates of the geographic centroid
        utm_x_centroid, utm_y_centroid = geo_to_utm.transform(lon_centroid, lat_centroid)

        # Calculate the pixel offsets in meters
        x_offsets = (x_locations - x_centroid) * pixel_size_m  # Offset in meters
        y_offsets = (y_locations - y_centroid) * pixel_size_m  # Offset in meters

        # Compute the UTM coordinates for the points after the clockwise rotation
        utm_x_real = utm_x_centroid + y_offsets    # New x = Old y
        utm_y_real = utm_y_centroid - x_offsets    # New y = -Old x (inverted)

        # Create a list of valid points and create geometry for GeoDataFrame
        valid_points = [(utm_x, utm_y) for utm_x, utm_y in zip(utm_x_real, utm_y_real) if not np.isnan(utm_x) and not np.isnan(utm_y)]
        geometry = [Point(utm_x, utm_y) for utm_x, utm_y in valid_points]

        # Create GeoDataFrame and save
        gdf = gpd.GeoDataFrame(geometry=geometry, crs=crs.getInfo())  # Specify UTM as the CRS

        # Attach X and Y columns for better transferability from final CSV to original NetCDF
        gdf['pixel_x'] = x_locations[~np.isnan(x_locations)]
        gdf['pixel_y'] = y_locations[~np.isnan(y_locations)]

        # Save the resulting points as a GeoPackage
        # TODO: Kept for debugging
        s2_name = s2_product.decode('utf-8').replace('.SAFE', '')  # Clean the filename
        gdf.to_file(f"data/test/{s2_name}_points.gpkg", driver="GPKG")
        print("Geographic points successfully projected, rotated 90 degrees clockwise, and saved in UTM CRS.")
        #return

        """ Extract spectral signatures from Sentinel-2 bands """
        # Create a list of Features with latitude, longitude, and s2_product
        features = []
        for idx, row in gdf.to_crs('EPSG:4326').iterrows():
            geom = row.geometry
            # Create a point geometry
            ee_point = ee.Geometry.Point([geom.x, geom.y])
            
            # Create a feature with properties
            feature = ee.Feature(ee_point, {
                's2_product': s2_product.decode('utf-8'), # Sentinel-2 tile as defined by NetCDF
                'atm_level': prefix, # L2A / L1C
                'pixel_x':  row['pixel_x'], # X defined by NetCDF; included for cross-referencing
                'pixel_y': row['pixel_y'], # Y defined by NetCDF; included for cross-referencing
                'latitude': geom.y,  # Latitude
                'longitude': geom.x  # Longitude
            })
            
            features.append(feature)

        # Create a FeatureCollection from the list of features
        ee_points = ee.FeatureCollection(features)

        # Export all pixels overlapping with points for each band
        geemap.extract_values_to_points(ee_points, 
                                        sentinel2, 
                                        export_path,
                                        scale=10, # 10m pixel resolution
                                        crs=crs.getInfo() # File should still be in UTM, just to be sure
                                        )
    print('#' * 40)
    print(f"Script finished in {round(time.time() - start, 4)} seconds")

def process_cozar_data(netcdf_path, csv_dir, out_dir):
    """
    Process the Cozar dataset and save reflectance data for each filament as CSV.
    Merge the CSVs, reorder the columns, and save as a single CSV.
    
    Parameters:
    netcdf_path (str): Path to the NetCDF file.
    csv_dir (str): Directory to save individual filament CSV files.
    out_dir (str): Directory to save the final merged CSV file.
    """
    # Initialize the Earth Engine API
    try:
        ee.Initialize()
    except Exception as e:
        # If initialization fails, authenticate
        print("Initializing Earth Engine...")
        ee.Authenticate()
        ee.Initialize()

    # Load the Cozar dataset
    cozar_data = xr.open_dataset(netcdf_path)

    # Loop over each filament and process it
    for i in range(len(cozar_data['s2_product'])):
        # Debugging: stop after processing a few filaments
        if i > 5:
            print('Processing completed for a few filaments.')
            break
        
        # Extract product name and clean it for filename purposes
        s2_product = cozar_data['s2_product'][i].item()

        # Create the export path
        file_name = s2_product.decode('utf-8').replace(".SAFE", ".csv")
        export_path = os.path.join(csv_dir, file_name)

        # Check if the CSV file for this product already exists
        if os.path.exists(export_path):
            print(f"'{file_name}' already exists. Skipping processing.")
            continue  # Skip to the next filament if the file exists

        # Extract latitude and longitude centroids
        lat_centroid = cozar_data['lat_centroid'][i].values
        lon_centroid = cozar_data['lon_centroid'][i].values
        
        # Extract XY centroids
        x_centroid = cozar_data['x_centroid'][i].values
        y_centroid = cozar_data['y_centroid'][i].values
        
        # Extract XY locations of filament pixels
        x_locations = cozar_data['pixel_x'][i].values
        y_locations = cozar_data['pixel_y'][i].values

        # Download GeoTIFF for debugging
        download_single_s2img(s2_product, lat_centroid, lon_centroid, "data/test")
        
        # Process the filament reflectance
        reflectance_per_filament(s2_product, lat_centroid, lon_centroid, 
                                 x_centroid, y_centroid, 
                                 x_locations, y_locations, 
                                 export_path)

    # Read and merge all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    df = pd.concat([pd.read_csv(file) for file in csv_files])

    # Define desired column order for readability
    order = [
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12',
        'system:index', 's2_product', 'atm_level', 'pixel_x', 'pixel_y', 'latitude', 'longitude'
    ]
    df = df[order]

    # Save the merged CSV
    os.makedirs(out_dir, exist_ok=True)
    output_csv = os.path.join(out_dir, "cozar_reflectance.csv")

    df.to_csv(output_csv, index=False)
    print(f"Reordered CSV saved as {output_csv}")

    return df