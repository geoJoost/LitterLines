import os
import json
import shutil
import ast
import pandas as pd

# Custom modules
from utils.gee_downloader import download_single_s2img

def backup_existing_geopackage(base_folder, annotation_folder_name):
    """
    Creates a backup of an existing geopackage file in the 'backup' subfolder.

    Args:
    - base_folder: Base folder where the annotation folder is located.
    - annotation_folder_name: Name of the annotation folder.
    """
    geopackage_path = os.path.join(base_folder, f"{annotation_folder_name}.gpkg")
    backup_folder = os.path.join(base_folder, 'backup')
    
    if os.path.exists(geopackage_path):
        os.makedirs(backup_folder, exist_ok=True)
        backup_path = os.path.join(backup_folder, f"{annotation_folder_name}.gpkg")
        shutil.copy(geopackage_path, backup_path)
        print(f"Backup created: {backup_path}")

def prepare_annotation_folder(base_dir, target_material, rank, s2_product_name, metadata):
    """
    Prepares the folder structure for storing annotations and metadata.

    Args:
    - base_dir: Directory to store annotation folders.
    - target_material: Material type (e.g., HDPE, PVC).
    - rank: Rank of the spectral angle.
    - s2_product_name: Sentinel-2 product name.
    - metadata: Metadata dictionary to save in the folder.

    Returns:
    - folder_path: Path to the created annotation folder.
    """
    # Clean up input names
    target_cleaned = target_material.replace('sam_', '')  # Remove prefix 'sam_'
    s2_cleaned = s2_product_name.replace('.SAFE', '')  # Remove '.SAFE' extension

    # Define annotation folder name and path
    annotation_folder_name = f"{target_cleaned}_{rank:02}_{s2_cleaned}"
    annotation_folder_path = os.path.join(base_dir, annotation_folder_name)

    # Backup any existing geopackage
    backup_existing_geopackage(base_dir, annotation_folder_name)

    # Create the folder structure
    os.makedirs(annotation_folder_path, exist_ok=True)
    annotations_subfolder = os.path.join(annotation_folder_path, 'annotations')
    os.makedirs(annotations_subfolder, exist_ok=True)

    # Save metadata if not already present
    metadata_path = os.path.join(annotation_folder_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    # Check if a geopackage already exists
    geopackage_path = os.path.join(annotation_folder_path, f"{annotation_folder_name}.gpkg")
    if os.path.exists(geopackage_path):
        print(f"Geopackage exists: {geopackage_path}. Skipping creation.")

    return annotation_folder_path

def process_spectral_angles(csv_file, top_n=10, include_overlap=True, annotations_base_dir='data/annotations'):
    """
    Processes a CSV of spectral angles, prepares annotation folders for the top-N targets, and downloads Sentinel-2 images.

    Args:
    - csv_file: Path to the CSV file containing spectral data.
    - top_n: Number of top-ranked spectral angles to process.
    - include_overlap: Whether to include records with spatial overlap.
    - annotations_base_dir: Base directory to store annotation folders.
    """
    # Load the spectral angles CSV
    spectral_data = pd.read_csv(csv_file)

    # Display available columns for context
    print("CSV columns:", spectral_data.columns)

    # Define spectral angle targets
    target_columns = ['sam_HDPE', 'sam_PVC', 'sam_HDPE_BF']

    # Map sensor IDs to human-readable names
    sensor_id_map = {
        'PS2': 'Dove-C (PS2)',
        'PS2.SD': 'Dove-R (PS2.SD)',
        'PSB.SD': 'SuperDove (PSB.SD)'
    }

    # Filter data based on overlap condition
    if include_overlap:
        spectral_data = spectral_data[spectral_data['potential_overlap'] == 'yes']

    # Process each spectral angle target
    for target_column in target_columns:
        print(f"\n### Processing Top {top_n} Spectral Angles for {target_column} ###\n")
        
        # Get top-N rows with the lowest spectral angles for the current target
        top_rows = spectral_data.nsmallest(top_n, target_column)

        for idx, row in enumerate(top_rows.itertuples(), start=1):
            print(f"#{idx} Processing {target_column}")

            # Retrieve spectral angle and associated metadata
            spectral_angle = getattr(row, target_column)
            date = row.date
            lat, lon = row.lat_centroid, row.lon_centroid
            s2_product = row.s2_product
            image_ids = row.image_ids

            # Map sensor IDs to readable names
            sensor_ids = ast.literal_eval(getattr(row, 'sensor_ids'))
            mapped_sensors = [sensor_id_map.get(sensor, sensor) for sensor in sensor_ids]

            # Create metadata dictionary
            metadata = {
                'date': date,
                'coordinates': {'latitude': lat, 'longitude': lon},
                'sensor_ids': mapped_sensors,
                's2_product': s2_product,
                'image_ids': image_ids,
                'spectral_angle': spectral_angle
            }

            # Prepare annotation folder
            folder_path = prepare_annotation_folder(
                base_dir=annotations_base_dir,
                target_material=target_column,
                rank=idx,
                s2_product_name=s2_product,
                metadata=metadata
            )

            # Download the Sentinel-2 image
            download_single_s2img(s2_product, lat, lon, folder_path)
            print(f"Annotation folder ready at: {folder_path}")
            print("-" * 50)  # Separator for readability
