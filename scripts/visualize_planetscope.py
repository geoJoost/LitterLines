import os
import rasterio
import numpy as np

# Custom modules
from utils.compute_spectral_indices import scene_to_ndi


def compute_ndi(input_tif):
    """ Computes NDI for the given TIF file and saves it as a new TIF. """
    
    # Define output path based on the parent directory of the input TIF
    input_dir = os.path.dirname(input_tif)
    parent_dir = os.path.dirname(input_dir)
    input_basename = os.path.basename(input_tif)
    output_filename = os.path.splitext(input_basename)[0] + '_NDI_B2B8.tif'
    output_tif = os.path.join(parent_dir, output_filename)
    
    # Open the input raster file
    with rasterio.open(input_tif) as dataset:
        # Read the bands into an array
        raster = dataset.read()  # This will read all bands into a 3D numpy array
        band_list = [f'B{band}' for band in range(1, dataset.count + 1)]
        
        # Calculate NDI using blue (B1) and NIR (B4)
        ndi_result = scene_to_ndi(raster, band_list, b1_idx='B1', b2_idx='B4')

        # Save the NDI result to a new TIF file
        profile = dataset.profile
        profile.update(dtype=rasterio.float32, compress='deflate')  # Use float32 for NDI and apply compression

        with rasterio.open(output_tif, 'w', **profile) as ndi_dataset:
            ndi_dataset.write(ndi_result, 1)  # Write the NDI result as a new band
            
    print(f"NDI file saved as {output_tif}")

# Example usage
input_tif = '/misc/rs1/jvandalen/TRACEv2/data/annotations/HDPE_BF_05_S2A_MSIL1C_20210527T095031_N0300_R079_T33TVF_20210527T104619/S2A_MSIL1C_20210527T095031_N0300_R079_T33TVF_20210527T104619_psscene_analytic_udm2/PSScene/20210527_092942_101f_3B_AnalyticMS_clip.tif' 
compute_ndi(input_tif)