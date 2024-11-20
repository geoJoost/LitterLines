import os
import json
import shutil
import ast
import pandas as pd

# Custom modules
from utils.gee_downloader import download_single_s2img

def backup_geopackage(folder_path, folder_name):
    """
    Creates a backup of existing geopackages to prevent accidental loss.

    Args:
    - folder_path: Path to the current folder.
    - folder_name: Name of the folder.
    """
    geopackage_path = os.path.join(folder_path, f"{folder_name}.gpkg")
    backup_folder = os.path.join(folder_path, 'backup')
    if os.path.exists(geopackage_path):
        os.makedirs(backup_folder, exist_ok=True)
        shutil.copy(geopackage_path, os.path.join(backup_folder, f"{folder_name}.gpkg"))
        print(f"Backup created for {geopackage_path} in {backup_folder}")

def prepare_annotation_folder(base_path, target, rank, s2_product, metadata):
    """
    Prepares the folder structure for annotations and backs up existing annotation files.

    Args:
    - base_path: Base directory for storing annotation folders.
    - target: The target material (e.g., HDPE, PVC).
    - rank: Rank of the spectral angle.
    - s2_product: Sentinel-2 product name.
    - metadata: Metadata dictionary to store in the folder.
    """
    # Clean the target name and Sentinel-2 product
    target_cleaned = target.replace('sam_', '')  # Remove 'sam_' from the target name
    s2_cleaned = s2_product.replace('.SAFE', '')  # Remove '.SAFE' from the product name

    # Folder name based on rank, target, and cleaned product
    folder_name = f"{target_cleaned}_{rank:02}_{s2_cleaned}"
    folder_path = os.path.join(base_path, folder_name)

    # Backup any existing geopackages before starting
    backup_geopackage(base_path, folder_name)

    # Create the folder if it doesn't already exist
    os.makedirs(folder_path, exist_ok=True)

    # Save metadata as a JSON file if it doesn't already exist
    metadata_path = os.path.join(folder_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    # Ensure annotations subfolder exists
    annotations_folder = os.path.join(folder_path, 'annotations')
    os.makedirs(annotations_folder, exist_ok=True)

    # Check if a geopackage already exists; log and skip if found
    geopackage_path = os.path.join(folder_path, f"{folder_name}.gpkg")
    if os.path.exists(geopackage_path):
        print(f"Geopackage exists: {geopackage_path}. Skipping creation.")

    return folder_path

def print_spectral_angles(file_path, top_n=10, spatial_overlap='yes', base_path='data/annotations'):
    """
    Filters and prints the top-N lowest spectral angles along with relevant metadata for each target column.
    Optionally filters by spatial overlap, prepares folders, and ensures no file overwriting.

    Args:
    - file_path: Path to the CSV file containing spectral data.
    - top_n: Number of observations with the lowest spectral angles to print.
    - spatial_overlap: Either 'yes' to include overlapping images, or 'no' to exclude them.
    - base_path: Path where annotation folders will be created.
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Print the column names to verify what columns exist in the DataFrame
    print(df.columns)

    # List of spectral angle columns
    spectral_angle_columns = ['sam_HDPE', 'sam_PVC', 'sam_HDPE_BF']

    # Mapping sensor IDs to readable format
    sensor_mapping = {
        'PS2': 'Dove-C (PS2)',
        'PS2.SD': 'Dove-R (PS2.SD)',
        'PSB.SD': 'SuperDove (PSB.SD)'
    }

    # Filter based on spatial overlap
    if spatial_overlap == 'yes':
        df = df[df['potential_overlap'] == 'yes']

    # Sort the data by spectral angle (lowest first) and process the top-N
    for column in spectral_angle_columns:
        print(f"\n########## Top {top_n} Lowest Spectral Angles for {column} ##########\n")
        
        # Get the top-N rows with the lowest spectral angles for this column
        top_rows = df.nsmallest(top_n, column)

        # Iterate through the rows and print details in a numbered list
        for idx, row in enumerate(top_rows.itertuples(), start=1):
            print(f"#{idx}")
            
            # Convert the spectral angle to percentage and format it
            spectral_angle = getattr(row, column) * 100  # Convert to percentage
            
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
            
            # Prepare metadata
            metadata = {
                'date': row.date,
                'coordinates': {'lat': row.lat_centroid, 'lon': row.lon_centroid},
                'sensor_ids': mapped_sensor_ids,
                's2_product': row.s2_product,
                'image_ids': row.image_ids,
                'spectral_angle': spectral_angle
            }

            # Prepare the folder for this annotation using the spectral angle column as the target
            folder_path = prepare_annotation_folder(base_path, column, idx, row.s2_product, metadata)

            # Call the function to download the Sentinel-2 image
            download_single_s2img(row.s2_product, row.lat_centroid, row.lon_centroid, folder_path)
            
            print(f"Folder prepared at: {folder_path}")
            print("-" * 50)  # Separator for readability


print_spectral_angles('data/processed/cozar_reflectance.csv', top_n=10, spatial_overlap='yes')