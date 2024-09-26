import os
import numpy as np
import matplotlib.pyplot as plt
from utils.compute_spectral_indices import scene_to_ndi, scene_to_fdi

def plot_histogram(pixel_spec, l1_bands, output_dir, sample_fraction=0.10):
    """
    Plots histograms for spectral bands, NDVI, NDI(B2,B8), and FDI.
    
    Parameters:
    - pixel_spec: xarray.DataArray containing spectral reflectance data.
    - l1_bands: List of band names corresponding to spectral bands.
    - output_dir: Directory path to save the output PNG file.
    - sample_fraction: Fraction of data to sample for plotting (default is 10%).
    """
    np.random.seed(42)  # Set seed for reproducibility
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Flatten and stack the pixel data (remove spatial component)
    reflectance_flat = pixel_spec.stack(pixels=("n_filaments", "n_max_pixels_fil")).values  # (13, n_pixels)

    # Get the number of bands and pixels
    n_bands, n_pixels = reflectance_flat.shape

    # Calculate the number of pixels to sample
    n_sample_pixels = int(n_pixels * sample_fraction)

    # Randomly sample the data
    sample_indices = np.random.choice(n_pixels, n_sample_pixels, replace=False)
    sampled_reflectance = reflectance_flat[:, sample_indices]

    # Create a grid of subplots (4x4 grid)
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()

    # Plot histograms for the spectral bands
    for i in range(n_bands):
        axes[i].hist(sampled_reflectance[i], bins=50, color='blue', alpha=0.7)
        axes[i].set_title(f"Histogram for {l1_bands[i]}")
        axes[i].set_xlabel('Reflectance [%]')
        axes[i].set_ylabel('Frequency')

    # Generate and plot histogram for NDVI
    ndvi = scene_to_ndi(sampled_reflectance, l1_bands, b1_idx='B4', b2_idx='B8')
    axes[n_bands].hist(ndvi, bins=50, color='green', alpha=0.7)
    axes[n_bands].set_title("Histogram for NDVI")
    axes[n_bands].set_xlabel('NDVI [-]')
    axes[n_bands].set_ylabel('Frequency')

    # Generate and plot histogram for NDI_B2, B8
    ndib2b8 = scene_to_ndi(sampled_reflectance, l1_bands, b1_idx='B2', b2_idx='B8')
    axes[n_bands + 1].hist(ndib2b8, bins=50, color='green', alpha=0.7)
    axes[n_bands + 1].set_title(r"Histogram for NDI$_{B2, B8}$")
    axes[n_bands + 1].set_xlabel(r'NDI$_{B2, B8}$ [-]')
    axes[n_bands + 1].set_ylabel('Frequency')

    # Generate and plot histogram for FDI
    fdi = scene_to_fdi(sampled_reflectance, l1_bands, b1_idx='B8', b2_idx='B6', b3_idx='B11')
    axes[n_bands + 2].hist(fdi, bins=50, color='green', alpha=0.7)
    axes[n_bands + 2].set_title(r"Histogram for FDI")
    axes[n_bands + 2].set_xlabel(r'FDI [-]')
    axes[n_bands + 2].set_ylabel('Frequency')

    # Remove unused subplots (if any)
    for j in range(n_bands + 3, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    # Save the combined plot into a single PNG
    output_path = os.path.join(output_dir, 'combined_histograms.png')
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory

    print(f"Histograms saved at: {output_path}")
