import numpy as np

def scene_to_ndi(raster, band_list, b1_idx='B4', b2_idx='B8'):
    """ Calculate Normalized Difference Index """
    # Read in the desired bands based on all multispectral bands given
    b1 = raster[band_list.index(b1_idx)].astype(float) #=> for instance use "B8" to sample from (n_channels, n_pixels)
    b2 = raster[band_list.index(b2_idx)].astype(float)

    # Calculate NDI while ignoring nodata values
    with np.errstate(divide='ignore', invalid='ignore'):
        ndi = (b1 - b2) / (b1 + b2)

    return ndi

def scene_to_fdi(raster, band_list, b1_idx='B8', b2_idx='B6', b3_idx='B11'):
    """ 
    Calculate the Floating Debris Index (FDI) 
    using three spectral bands (NIR, Red Edge, SWIR1).
    """

    # Placeholder for the central wavelengths
    # TODO: Current are hard-coded to Sentinel-2A
    lambda_nir = 832.8
    lambda_swir1 = 1613.7
    lambda_red = 664.6

    # Read in the desired bands based on the band indices
    nir = raster[band_list.index(b1_idx)]    # B8 (NIR)
    red2 = raster[band_list.index(b2_idx)]   # B6 (Red Edge)
    swir1 = raster[band_list.index(b3_idx)]  # B11 (SWIR1)

    # Calculate NIR prime
    nir_prime = red2 + (swir1 - red2) * (lambda_nir - lambda_red) / (lambda_swir1 - lambda_red)

    # Calculate FDI
    fdi = nir - nir_prime

    return fdi

