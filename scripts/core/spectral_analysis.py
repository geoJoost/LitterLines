import sys, os # TODO: Remove. Kept for debugging
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

import os
import geopandas as gpd
import glob
import ee
import geemap
import pandas as pd
import numpy as np
from spectral import spectral_angles
import ast


# Custom modules
from utils.gee_downloader import download_multiple_s2img, create_bounding_box, fetch_and_process
import os

def get_plp_scenes(shapefile_path, out_dir):
    # Load the polygon annotations using geopandas
    gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    """ Download Sentinel-2 images for the PLP2021 targets """
    # We need centroids for generating bounding boxes of approx. 10km for downloading the GeoTIFF's
    point = gdf.iloc[0].geometry
    lat_centroid = point.y
    lon_centroid = point.x

    # Download GeoTIFF's for debugging / annotations
    start_date = "2022-06-16"
    end_date = "2022-10-10" # +1 day after experiments finish
    #download_multiple_s2img(start_date, end_date, lat_centroid, lon_centroid, out_dir)

    """ Extract spectral data from the PLP2021 scenes """
    # Load Sentinel-2 image collections (L2A and L1C)
    #sentinel2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')# L2A data
    sentinel2_toa = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')  # L1C data

    # Parameters for extracting spectral information
    l2a_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    bounding_box = ee.Geometry.Polygon(create_bounding_box(lat_centroid, lon_centroid))

    # Base directory for temporary CSV exports
    root = "data/tmp"
    os.makedirs(root, exist_ok=True)

    for target in gdf['type'].unique():
        csv_path = f"data/processed/{target}_reflectance.csv"
        if os.path.exists(csv_path):
            print(f"Processed file for {target} already exists. Skipping download and extraction.")
            continue  # Skip the current target if the processed file already exists

        # Continue processing
        gdf_target = gdf[gdf['type'] == target]
        csv_files = [] # Track CSVs for appending later

        # Iterate through each date for the current PLP target
        for date in gdf_target['date'].unique():
            # Convert date to YYYY-MM-DD format for Earth Engine
            acquisition_date = pd.to_datetime(str(date)).strftime('%Y-%m-%d')
            ee_date = ee.Date(acquisition_date)

            # Filter Sentinel-2 collection by date and bands
            sentinel2, crs = fetch_and_process(sentinel2_toa, ee_date, l2a_bands, bounding_box)

            # Retrieve only the single point associated with the current target and date
            gdf_date = gdf_target[gdf_target['date'] == date]
            ee_points = geemap.gdf_to_ee(gdf_date)

            # Define an adaptive export path using target and date
            export_path = os.path.join(root, f"{target}_{acquisition_date}.csv")
            csv_files.append(export_path)

            geemap.extract_values_to_points(ee_points,
                                sentinel2,
                                export_path,
                                scale=10, # Set pixel size to 10m
                                crs=sentinel2.select(0).projection().crs().getInfo(), # Use native CRS
            )
        # After the loop, concatenate all export CSVs for this target
        target_reflectance = pd.concat([pd.read_csv(file) for file in csv_files])
        
        # Convert all DN into reflectance values
        target_reflectance[l2a_bands] = target_reflectance[l2a_bands] / 10000
        target_reflectance.to_csv(csv_path, index=False)

def spectral_similarity(pix_reflectance, target_files):
    """
    Calculate the spectral similarity for a single pixel spectrum against known target spectra.
    
    Parameters:
    pixel_spectrum (numpy array): 1D array containing the spectral data for the pixel across all bands.
    l2a_bands (list of str): List of band names to be used for the computation.
    target_files (dict): Dictionary mapping target names to file paths of their reference spectra.
    
    Returns:
    dict: A dictionary where each key is a target name and each value is the spectral angle for that target.
    """

    # Ensure pixel_spectrum has correct shape for compatibility with 3D calculations
    pixel_data = pix_reflectance[np.newaxis, np.newaxis, :]  # Shape: (1, 1, B)
    
    # Load and compute mean reflectance vectors for each target
    mean_reflectance_vectors = []
    target_names = []
    
    for target, file_path in target_files.items():
        target_df = pd.read_csv(file_path)
        gee_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

        # Divide by 10,000 to get proper reflectance
        scaled_df = target_df[gee_bands]
        mean_reflectance = scaled_df.mean().values  # Compute mean reflectance across bands
        
        mean_reflectance_vectors.append(mean_reflectance)
        target_names.append(target)

    # Convert mean reflectance vectors to a numpy array for spectral angle calculation
    members = np.array(mean_reflectance_vectors)  # Shape: (N_targets, B)
    
    # Calculate spectral angles between pixel data and each target mean reflectance vector
    angles = spectral_angles(pixel_data, members).squeeze()
    degrees = np.degrees(angles) # Convert radians to degrees
    return degrees

def print_spectral_angles(file_path, top_n=10, spatial_overlap='yes'):
    """
    Filters and prints the top-N lowest spectral angles along with relevant metadata for each target column.
    Optionally filters by spatial overlap.

    Args:
    - file_path: Path to the CSV file containing spectral data.
    - top_n: Number of observations with the lowest spectral angles to print.
    - spatial_overlap: Either 'yes' to include overlapping images, or 'no' to exclude them.
    """
    # Load the data
    df = pd.read_csv(file_path)

    # List of spectral angle columns
    spectral_angle_columns = ['sam_HDPE', 'sam_PVC', 'sam_HDPE_BF', 'sam_water']

    # Mapping sensor IDs to readable format
    sensor_mapping = {
        'PS2': 'Dove-C (PS2)',
        'PS2.SD': 'Dove-R (PS2.SD)',
        'PSB.SD': 'SuperDove (PSB.SD)'
    }

    # Filter based on spatial overlap
    if spatial_overlap == 'yes':
        df = df[df['potential_overlap'] == 'yes']

    for column in spectral_angle_columns:
        print(f"\n########## Top {top_n} Lowest Spectral Angles for {column} ##########\n")
        
        # Get the top-N rows with the lowest spectral angles for this column
        top_rows = df.nsmallest(top_n, column)

        # Iterate through the rows and print details in a numbered list
        for idx, row in enumerate(top_rows.itertuples(), start=1):
            print(f"#{idx}")
            
            # Convert the spectral angle to percentage and format it
            spectral_angle = getattr(row, column)  # Convert to percentage
            
            # Print variables
            print(f"Spectral Angle ({column}): {spectral_angle:.2f}°")
            print(f"Date: {row.date}")
            print(f"Coordinates: {row.lat_centroid}, {row.lon_centroid}")
            print(f"Sentinel-2 product: {row.s2_product}")
            print(f"Potential overlap: {row.potential_overlap}")
            print(f"Image IDs: {row.image_ids}")
            
            # Map each sensor ID from the list to its readable format
            sensor_ids = ast.literal_eval(getattr(row, 'sensor_ids'))  # Get the string representation of sensor_ids
            mapped_sensor_ids = [sensor_mapping.get(sensor) for sensor in sensor_ids]
            print(f"PS-sensors: {', '.join(mapped_sensor_ids)}")
            
            print("-" * 50)  # Separator for readability
