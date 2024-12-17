import os
import numpy as np
import xarray as xr
import dask.array as da
import rioxarray as rxr
import matplotlib.pyplot as plt
import time
import pandas as pd

# Custom modules
from utils.visualization_functions import *
from core.extract_cozar_reflectance import process_cozar_data
from core.spectral_analysis import get_plp_scenes, spectral_similarity, print_spectral_angles
from utils.planet_functions import planet_query
from utils.gee_downloader import get_date_from_product
from utils.create_water_samples import generate_water_points
from utils.organise_annotation_folders import process_spectral_angles

# Define start time to measure how long the script takes to complete
start = time.time()

def process_cozar_predictions(netcdf_path, plot_figures=True):
    # TODO: Kept for archival purposes; or in case geocorrection becomes feasible
    #csv_dir = "data/test"
    #out_dir = "data/processed"
    #cozar_reflectance = process_cozar_data(netcdf_path, csv_dir, out_dir)
    
    # Acquire and process Plastic Litter Project using L1C reflectance
    # We acquire reflectance for the following targets: HDPE, wood, HDPE+wood for PLP2021
    # and PVC, HDPE-Clean, HDPE-BioFouled for PLP2022
    tif_dir = "data/plp_tiles"
    plp2021 = get_plp_scenes('data/raw/PLP2021_targets.shp', tif_dir)
    plp2022 = get_plp_scenes('data/raw/PLP2022_targets.shp', tif_dir)
    
    # Create a dataset of randomly generated points in the Gulf of Gera
    generate_water_points(output_path="data/raw/water_px.shp")
    get_plp_scenes('data/raw/water_px.shp', tif_dir)

    # Check if the file already exists
    output_file = 'data/processed/cozar_reflectance.csv'
    if not os.path.exists(output_file):
        # Read in the predictions
        cozar_data = xr.open_dataset(netcdf_path)

        # Extract spectral bands for filtering; output should correspond to TOA reflectance with L2A bands
        spectral_bands = cozar_data.attrs['spectral_bands'].split(', ')
        l2a_bands = [band for band in spectral_bands if band != "B10"]

        # Placeholder for the final DataFrame
        columns = [
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12',
            'lat_centroid', 'lon_centroid', 'date', 'sam_HDPE', 'sam_PVC', 'sam_HDPE_BF', 'sam_water',
            's2_product', 'potential_overlap', 'image_ids', 'sensor_ids', 'image_types'
        ]
        final_data = []
        
        # Process each marine litter windrow filament
        for i in range(len(cozar_data['s2_product'])):
            # TODO - Debugging: stop after processing a few filaments
            #if i > 100:
            #    print('Processing completed for a few filaments.')
            #    break
            
            # Extract product name and metadata
            s2_product = cozar_data['s2_product'][i].item().decode()
            lat_centroid = cozar_data['lat_centroid'][i].values
            lon_centroid = cozar_data['lon_centroid'][i].values
            date = get_date_from_product(s2_product)

            # Query for spatial-temporal overlap between Sentinel-2 and PlanetScope
            potential_overlap, image_ids, sensor_ids, types = planet_query(lat_centroid, lon_centroid, date)
            
            # Process each pixel within the current filament
            pixel_spec = cozar_data['pixel_spec'][i].values  # L1C reflectance
            pixel_spec = pixel_spec[~np.isnan(pixel_spec).any(axis=1)]  # Remove rows that contain any NaN values
            pixel_spec = pixel_spec[:, [l2a_bands.index(band) for band in l2a_bands]]  # Exclude B10 from analysis
            
            # Define target files for the spectral angle computation
            target_files = {
                "HDPE": "data/processed/HDPE_reflectance.csv",  # HDPE from PLP2021
                "PVC": "data/processed/PVC_reflectance.csv",  # PVC from PLP2022
                "HDPE-BF": "data/processed/HDPE-BF_reflectance.csv",  # Biofouled HDPE from PLP2022
                "Water": "data/processed/Water_reflectance.csv" # Randomly selected water pixels (n=205) in 2021/2022
            }

            for pix_reflectance in pixel_spec:
                # Compute spectral angle (SAM) for this pixel
                sam_results = spectral_similarity(pix_reflectance, target_files)

                # Append the data row for the pixel
                final_data.append([
                    *pix_reflectance,  # Reflectance for bands B1-B12
                    lat_centroid, 
                    lon_centroid, 
                    date, 
                    sam_results[0],  # SAM for HDPE
                    sam_results[1],  # SAM for PVC
                    sam_results[2],  # SAM for HDPE-BF
                    sam_results[3],  # SAM for water
                    s2_product,
                    potential_overlap, 
                    image_ids, 
                    sensor_ids,
                    types
                ])

        # Convert to DataFrame
        final_df = pd.DataFrame(final_data, columns=columns)

        # Save to a CSV file or other format
        final_df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print(f"The file {output_file} already exists. Skipping processing.")

    # Plotting functions
    if plot_figures:
        print('Printing figures')
        
        # Define spectral bands for Sentinel-2; differ from original as they do not include zeros (i.e., B04 => B4)
        l2a_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

        # Plot histogram of Cozar predictions. We sample 10% of the >36 million pixels
        #plot_histogram(pixel_spec, l1_bands, output_dir='doc/figures', sample_fraction=0.10)

        # Boxplot
        #plot_boxplot(cozar_reflectance, l2a_bands, output_dir='doc/figures')
        
        # Plot spectral signatures of the PLP targets
        plot_individual_spectra(output_file, l2a_bands, output_dir='doc/figures', use_wavelengths=True)
        
        # Plot 1000 random spectral signatures from the Cozar dataset
        plot_random_spectra(output_file, l2a_bands, output_dir='doc/figures', use_wavelengths=True, n_random=1000)

        # Spectral signature line graph
        plot_line_graph(output_file, l2a_bands, output_dir='doc/figures', use_wavelengths=True)
        plot_line_graph(output_file, l2a_bands, output_dir='doc/figures', use_wavelengths=False)
        
        # SAM's and top-5 MLW's
        plot_spectral_angle_signatures(output_file, l2a_bands, top_n=5)
        
        # Scatterplot of MLW + targets using NDVI-FDI
        plot_scatter(output_file, output_dir='doc/figures')

        # PCA
        plot_pca(output_file, l2a_bands, output_dir="doc/figures")
        
        # Plot PlanetScope statistics
        plot_planetscope_acquisitions(output_file)
        
        # T-SNE plot with Spectral Angle Mapping
        # TODO: This needs a revamp including: Spectral Angle Mapping as distance metric, and tuning of perplexity value
        #plot_tsne(cozar_reflectance, l2a_bands, output_dir='doc/figures')
    
    # Output the potential PlanetScope scenes
    print_spectral_angles(output_file, top_n=10, spatial_overlap='yes')

    # Prepare annotation folders by:
    ## Downloading Sentinel-2 data (~25km)
    ## Computing the FDI and NDI_B2B8
    ## Preparing metadata
    process_spectral_angles(csv_file=output_file, top_n=40, include_overlap=True)

    # 
    

   




process_cozar_predictions(netcdf_path="data/raw/WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc", plot_figures=True)

print(f"Script finished in {round(time.time() - start, 2)} seconds")
