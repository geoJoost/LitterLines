import os
import numpy as np
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
from utils.visualization_functions import plot_histogram

def process_cozar_predictions(file_path, plot_hist=True):
    np.random.seed(42)
    # Read in the predictions
    cozar_data = xr.open_dataset(file_path)

    # First we get all the spectral reflectance values (L1C; TOA)
    pixel_spec = cozar_data['pixel_spec']

    """ Plot simple histogram """
    if plot_hist:
        l1_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
        plot_histogram(pixel_spec, l1_bands, output_dir='doc/figures', sample_fraction=0.10) # Dataset has 36 million pixels, we sample 10%
    
    """ Acquire and process Plastic Litter Project (PLP) 2021 """


    """ Plot box-plot to compare reflectance values """


    """ Plot 2D feature space """


    """ Extract Cozar predictions closest to PLP2021 data, and acquire Sentinel-2 ID's """


    print('...')



process_cozar_predictions(file_path="data/raw/WASP_LW_SENT2_MED_L1C_B_201506_202109_10m_6y_NRT_v1.0.nc", plot=False)