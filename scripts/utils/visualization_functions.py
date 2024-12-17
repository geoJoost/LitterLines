import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Custom modules
from utils.compute_spectral_indices import scene_to_ndi, scene_to_fdi

def plot_histogram(pixel_spec, l2a_bands, output_dir, sample_fraction=0.10):
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

    """ Cozar Predictions """
    # Flatten and stack the pixel data (remove spatial component)
    reflectance_flat = pixel_spec.stack(pixels=("n_filaments", "n_max_pixels_fil")).values  # (13, n_pixels)

    # Get the number of bands and pixels
    n_bands, n_pixels = reflectance_flat.shape

    # Calculate the number of pixels to sample
    n_sample_pixels = int(n_pixels * sample_fraction)

    # Randomly sample the data
    sample_indices = np.random.choice(n_pixels, n_sample_pixels, replace=False)
    sampled_reflectance = reflectance_flat[:, sample_indices]

    """ Plotting """
    # Create a grid of subplots (4x4 grid)
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()

    # Plot histograms for the spectral bands
    for i in range(n_bands):
        axes[i].hist(sampled_reflectance[i], bins=50, color='blue', alpha=0.7)
        axes[i].set_title(f"Histogram for {l2a_bands[i]}")
        axes[i].set_xlabel('Reflectance [%]')
        axes[i].set_ylabel('Frequency')

    # Generate and plot histogram for NDVI
    ndvi = scene_to_ndi(sampled_reflectance, l2a_bands, b1_idx='B4', b2_idx='B8')
    axes[n_bands].hist(ndvi, bins=50, color='green', alpha=0.7)
    axes[n_bands].set_title("Histogram for NDVI")
    axes[n_bands].set_xlabel('NDVI [-]')
    axes[n_bands].set_ylabel('Frequency')

    # Generate and plot histogram for NDI_B2, B8
    ndib2b8 = scene_to_ndi(sampled_reflectance, l2a_bands, b1_idx='B2', b2_idx='B8')
    axes[n_bands + 1].hist(ndib2b8, bins=50, color='green', alpha=0.7)
    axes[n_bands + 1].set_title(r"Histogram for NDI$_{B2, B8}$")
    axes[n_bands + 1].set_xlabel(r'NDI$_{B2, B8}$ [-]')
    axes[n_bands + 1].set_ylabel('Frequency')

    # Generate and plot histogram for FDI
    fdi = scene_to_fdi(sampled_reflectance, l2a_bands, b1_idx='B8', b2_idx='B6', b3_idx='B11')
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
    output_path = os.path.join(output_dir, 'cozar_histogram.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Histograms saved at: {output_path}")

def plot_boxplot(cozar_dataset, l2a_bands, output_dir):
    """
    Plots boxplots for spectral bands, NDVI, NDI(B2,B8), and FDI.
    
    Parameters:
    - cozar_dataset: DataFrame containing spectral reflectance data.
    - plp2021: Numpy array containing the PLP2021 spectral reflectance data.
    - l2a_bands: List of band names corresponding to spectral bands.
    - output_dir: Directory path to save the output PNG file.
    """
    np.random.seed(42)  # Set seed for reproducibility

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    """ Cozar Predictions """
    # Split the cozar dataset into L1C and L2A based on 'atm_level' column
    cozar_L1C = cozar_dataset[cozar_dataset['atm_level'] == 'L1C']
    cozar_L2A = cozar_dataset[cozar_dataset['atm_level'] == 'L2A']

    # Extract only the band columns (B1, B2, ..., B12) from both L1C and L2A
    cozar_L1C_reflectance = cozar_L1C[l2a_bands]
    cozar_L2A_reflectance = cozar_L2A[l2a_bands]

    """ PLP2021 """
    # Get the number of bands
    assert cozar_L1C_reflectance.shape[1] == plp2021.shape[0], "Number of bands must match for both datasets."

    """ Descriptive statistics & data cleaning """
    # Ensure zero values are filtered from both datasets
    filtered_plp2021 = [band_data[band_data != 0] for band_data in plp2021]
    filtered_cozar_L1C = [cozar_L1C_reflectance[band][cozar_L1C_reflectance[band] != 0].values for band in l2a_bands]
    filtered_cozar_L2A = [cozar_L2A_reflectance[band][cozar_L2A_reflectance[band] != 0].values for band in l2a_bands]

    # Print statistics side by side for each band
    print(f"{'Band':<5} | {'PLP2021 Mean':<15} | {'Cozar L1C Mean':<15} | {'Cozar L2A Mean':<15} | "
          f"{'PLP2021 Std Dev':<15} | {'Cozar L1C Std Dev':<15} | {'Cozar L2A Std Dev':<15} | "
          f"{'PLP2021 Min':<15} | {'Cozar L1C Min':<15} | {'Cozar L2A Min':<15} | "
          f"{'PLP2021 Max':<15} | {'Cozar L1C Max':<15} | {'Cozar L2A Max':<15}")
    print("-" * 175)

    for i, band in enumerate(l2a_bands):
        # Compute statistics for PLP2021
        plp_mean = np.mean(filtered_plp2021[i])
        plp_std = np.std(filtered_plp2021[i])
        plp_min = np.min(filtered_plp2021[i])
        plp_max = np.max(filtered_plp2021[i])

        # Compute statistics for Cozar L1C
        cozar_L1C_mean = np.mean(filtered_cozar_L1C[i])
        cozar_L1C_std = np.std(filtered_cozar_L1C[i])
        cozar_L1C_min = np.min(filtered_cozar_L1C[i])
        cozar_L1C_max = np.max(filtered_cozar_L1C[i])

        # Compute statistics for Cozar L2A
        cozar_L2A_mean = np.mean(filtered_cozar_L2A[i])
        cozar_L2A_std = np.std(filtered_cozar_L2A[i])
        cozar_L2A_min = np.min(filtered_cozar_L2A[i])
        cozar_L2A_max = np.max(filtered_cozar_L2A[i])

        # Print the statistics side by side for the current band
        print(f"{i+1:<5} | {plp_mean:<15.4f} | {cozar_L1C_mean:<15.4f} | {cozar_L2A_mean:<15.4f} | "
              f"{plp_std:<15.4f} | {cozar_L1C_std:<15.4f} | {cozar_L2A_std:<15.4f} | "
              f"{plp_min:<15.4f} | {cozar_L1C_min:<15.4f} | {cozar_L2A_min:<15.4f} | "
              f"{plp_max:<15.4f} | {cozar_L1C_max:<15.4f} | {cozar_L2A_max:<15.4f}")

    """ Plotting """
    # Create the boxplot
    fig, ax = plt.subplots(figsize=(12, 6))

    # X positions for the bands
    positions = np.arange(1, len(l2a_bands) + 1)
    width = 0.2  # Adjust width for three boxplots per band

    # Colors for each dataset
    colors = ['#0d3b66', '#faf0ca', '#ffa07a']  # PLP2021, Cozar L1C, Cozar L2A
    outline_color = '#000000'  # Black outline color for the boxes
    median_color = '#cc2936'

    # Plot each band separately for the three datasets
    for i, band in enumerate(l2a_bands):
        # PLP2021 data (shifted to the left)
        ax.boxplot(
            filtered_plp2021[i],
            positions=[positions[i] - width],
            widths=width,
            patch_artist=True,  # Allows coloring
            boxprops=dict(facecolor=colors[0], edgecolor=outline_color, linewidth=1),
            medianprops=dict(color=median_color),
            flierprops=dict(marker='o', markerfacecolor=colors[0], markersize=4, alpha=0.4),
            showfliers=True
        )

        # Cozar L1C Reflectance data (centered)
        ax.boxplot(
            filtered_cozar_L1C[i] / 10000,
            positions=[positions[i]],
            widths=width,
            patch_artist=True,  # Allows coloring
            boxprops=dict(facecolor=colors[1], edgecolor=outline_color, linewidth=1),
            medianprops=dict(color=median_color),
            flierprops=dict(marker='o', markerfacecolor=colors[1], markersize=4, alpha=0.4),
            showfliers=True
        )

        # Cozar L2A Reflectance data (shifted to the right)
        ax.boxplot(
            filtered_cozar_L2A[i] / 10000,
            positions=[positions[i] + width],
            widths=width,
            patch_artist=True,  # Allows coloring
            boxprops=dict(facecolor=colors[2], edgecolor=outline_color, linewidth=1),
            medianprops=dict(color=median_color),
            flierprops=dict(marker='o', markerfacecolor=colors[2], markersize=4, alpha=0.4),
            showfliers=True
        )

    # Customizing the X-ticks and labels
    ax.set_xticks(positions)
    ax.set_xticklabels(l2a_bands)

    # Add labels and title
    ax.set_ylabel('Reflectance [%]')
    ax.set_title('Sentinel-2 reflectance comparison with PLP2021')

    # Adding a custom legend
    plp2021_count = int(np.count_nonzero(filtered_plp2021) / len(l2a_bands))  # Number of pixels
    cozar_L1C_count = int(np.mean([np.count_nonzero(band) for band in filtered_cozar_L1C]) / len(l2a_bands))
    cozar_L2A_count = int(np.mean([np.count_nonzero(band) for band in filtered_cozar_L2A]) / len(l2a_bands))

    legend_elements = [
        Patch(facecolor=colors[0], edgecolor=colors[0], label=f'PLP2021 (n={plp2021_count})'),
        Patch(facecolor=colors[1], edgecolor=colors[1], label=f'Cozar L1C (n={cozar_L1C_count})'),
        Patch(facecolor=colors[2], edgecolor=colors[2], label=f'Cozar L2A (n={cozar_L2A_count})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Improve layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{output_dir}/s2_boxplots.png", dpi=300)
    plt.close()

    print("Boxplot printed")

def plot_line_graph(cozar_csv, l2a_bands, output_dir, use_wavelengths=False):
    """
    Plots line graphs with MLW as the baseline signature in each subplot. 
    All other spectral signatures, including HDPE, are compared to MLW.

    Parameters:
    - cozar_csv: Path to the CSV file containing spectral reflectance data (reference data).
    - l2a_bands: List of band names corresponding to spectral bands.
    - output_dir: Directory path to save the output PNG file.
    - use_wavelengths: If True, use wavelengths (in nm) for the X-axis. Otherwise, use band names.
    """
    # Mapping from bands to wavelengths
    wavelengths = {
        "B1": 442, "B2": 492, "B3": 559, "B4": 664,
        "B5": 704, "B6": 740, "B7": 782, "B8": 832,
        "B8A": 864, "B9": 945, "B11": 1613, "B12": 2202
    }
    x_axis = [wavelengths[band] for band in l2a_bands] if use_wavelengths else l2a_bands
    x_label = 'Wavelength (nm)' if use_wavelengths else 'Bands'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    """ Load and Process Data """
    # Load the Cozar dataset (MLW)
    cozar_dataset = pd.read_csv(cozar_csv)
    cozar_L1C = cozar_dataset[l2a_bands]

    # Calculate the mean and standard deviation for MLW
    mlw_means = cozar_L1C.mean()
    mlw_stds = cozar_L1C.std()
    mlw_n = len(cozar_L1C)

    # Load target datasets
    targets = ["HDPE", "PVC", "HDPE-BF", "HDPE-C", "wood", "Water"]
    output_files = {
        "HDPE": "data/processed/HDPE_reflectance.csv",
        "PVC": "data/processed/PVC_reflectance.csv",
        "HDPE-BF": "data/processed/HDPE-BF_reflectance.csv",
        "HDPE-C": "data/processed/HDPE-C_reflectance.csv",
        "wood": "data/processed/wood_reflectance.csv",
        "Water": "data/processed/Water_reflectance.csv"
    }
    target_means = {}
    target_stds = {}
    target_n = {}

    for label in targets:
        data = pd.read_csv(output_files[label])
        target_means[label] = data[l2a_bands].mean()
        target_stds[label] = data[l2a_bands].std()
        target_n[label] = len(data)

    """ Plotting Subplots """
    # Define colors for each target
    colors = {
        "HDPE": '#ff7d00',
        "PVC": '#15616d',
        "HDPE-BF": '#001524',
        "HDPE-C": '#8a3d62',
        "wood": '#6c4f3d',
        "Water": '#6495ED'  # Blue for Water
    }

    # Define markers for each target
    markers = {
        "HDPE": 'x',       # 'x' marker
        "PVC": 's',        # Square marker
        "HDPE-BF": '^',    # Triangle marker
        "HDPE-C": 'D',     # Diamond marker 
        "wood": 'H',       # Hexagon marker
        "Water": 'o'       # Circle marker for Water
    }

    # Adjust the number of subplots
    ncols = 2
    nrows = (len(targets) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4))
    axes = axes.flatten()  # Flatten for easy iteration

    for i, label in enumerate(targets):
        ax = axes[i]

        # Plot MLW (baseline signature) in each subplot
        ax.plot(x_axis, mlw_means, label=f"MLW (n={mlw_n})", color='#6494aa', linewidth=2, marker='o', ms=5)
        ax.fill_between(x_axis, mlw_means - mlw_stds, mlw_means + mlw_stds, color='#6494aa', alpha=0.2)

        # Plot the target dataset in the current subplot
        ax.plot(x_axis, target_means[label], label=f"{label} (n={target_n[label]})", 
                color=colors[label], linewidth=2, marker=markers[label], markersize=5)
        ax.fill_between(x_axis, target_means[label] - target_stds[label], 
                        target_means[label] + target_stds[label], color=colors[label], alpha=0.2)

        # Customize subplot
        ax.set_title(f'{label}')
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_xlabel(x_label)
        if i % ncols == 0:
            ax.set_ylabel('Reflectance')

        # Add legend
        ax.legend(loc='upper right')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Improve layout
    plt.tight_layout()

    # Save the figure
    output_filename = "s2_linegraph_wl.png" if use_wavelengths else "s2_linegraph.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Spectral reflectance subplot graph (vs MLW) saved to {output_path}")

def plot_individual_spectra(cozar_csv, l2a_bands, output_dir, use_wavelengths=False):
    """
    Plots individual spectral signatures from the target datasets.

    Parameters:
    - cozar_csv: Path to the CSV file containing spectral reflectance data (reference data).
    - l2a_bands: List of band names corresponding to spectral bands.
    - output_dir: Directory path to save the output PNG file.
    - use_wavelengths: If True, use wavelengths (in nm) for the X-axis. Otherwise, use band names.
    """
    # Mapping from bands to wavelengths
    wavelengths = {
        "B1": 442, "B2": 492, "B3": 559, "B4": 664,
        "B5": 704, "B6": 740, "B7": 782, "B8": 832,
        "B8A": 864, "B9": 945, "B11": 1613, "B12": 2202
    }
    x_axis = [wavelengths[band] for band in l2a_bands] if use_wavelengths else l2a_bands
    x_label = 'Wavelength (nm)' if use_wavelengths else 'Bands'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    """ Load and Process Data """
    # Load the Cozar dataset (MLW)
    cozar_dataset = pd.read_csv(cozar_csv)
    cozar_L1C = cozar_dataset[l2a_bands]

    # Load target datasets
    targets = ["HDPE", "PVC", "HDPE-BF", "HDPE-C", "HDPE+Wood", "wood", "Water"]
    output_files = {
        "HDPE": "data/processed/HDPE_reflectance.csv",
        "PVC": "data/processed/PVC_reflectance.csv",
        "HDPE-BF": "data/processed/HDPE-BF_reflectance.csv",
        "HDPE-C": "data/processed/HDPE-C_reflectance.csv",
        "HDPE+Wood": "data/processed/HDPE+Wood_reflectance.csv", 
        "wood": "data/processed/wood_reflectance.csv",
        "Water": "data/processed/Water_reflectance.csv"
    }

    # Define colors for each target
    colors = {
        "HDPE": '#ff7d00',
        "PVC": '#15616d',
        "HDPE-BF": '#001524',
        "HDPE-C": '#8a3d62',
        "HDPE+Wood": '#b8854a',
        "wood": '#6c4f3d',
        "Water": '#6495ED'
    }

    # Define markers for each target
    markers = {
        "HDPE": 'x',
        "PVC": 's',
        "HDPE-BF": '^',
        "HDPE-C": 'D',
        "HDPE+Wood": 'p',
        "wood": 'H',
        "Water": 'o' 
    }

    # Adjust the number of subplots
    ncols = 2
    nrows = (len(targets) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4))
    axes = axes.flatten()  # Flatten for easy iteration

    for i, label in enumerate(targets):
        ax = axes[i]

        # Load the target data
        data = pd.read_csv(output_files[label])

        # Plot all individual spectra from the target dataset
        for index, row in data.iterrows():
            ax.plot(x_axis, row[l2a_bands], color=colors[label], linewidth=1, marker=markers[label], markersize=3, alpha=0.5)

        # Customize subplot
        ax.set_title(f'{label}')
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_xlabel(x_label)
        if i % ncols == 0:
            ax.set_ylabel('Reflectance')

        # Add legend (optional, but will show last plotted marker)
        ax.legend([label], loc='upper right')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Improve layout
    plt.tight_layout()

    # Save the figure
    output_filename = "s2_plp_spectra.png" if use_wavelengths else "s2_plp_spectra_bands.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Spectral reflectance individual signature graph saved to {output_path}")

def plot_random_spectra(cozar_csv, l2a_bands, output_dir, use_wavelengths=True, n_random=1000):
    """
    Plots random individual spectral signatures from the Cozar dataset and the mean spectrum.

    Parameters:
    - cozar_csv: Path to the Cozar dataset CSV file.
    - l2a_bands: List of band names corresponding to spectral bands.
    - output_dir: Directory path to save the output PNG file.
    - use_wavelengths: If True, use wavelengths (in nm) for the X-axis. Otherwise, use band names.
    """
    # Mapping from bands to wavelengths
    wavelengths = {
        "B1": 442, "B2": 492, "B3": 559, "B4": 664,
        "B5": 704, "B6": 740, "B7": 782, "B8": 832,
        "B8A": 864, "B9": 945, "B11": 1613, "B12": 2202
    }
    
    # Determine the x-axis values (either wavelengths or band names)
    x_axis = [wavelengths[band] for band in l2a_bands] if use_wavelengths else l2a_bands
    x_label = 'Wavelength (nm)' if use_wavelengths else 'Bands'

    # Load the Cozar dataset
    cozar_dataset = pd.read_csv(cozar_csv)
    
    # Randomly select 1000 rows from the dataset
    random_rows = cozar_dataset.sample(n=1000, random_state=42)

    # Calculate the mean spectrum across all rows
    mean_spectrum = cozar_dataset[l2a_bands].mean()

    # Plot individual spectra (randomly selected 1000 rows)
    plt.figure(figsize=(12, 6))
    
    for _, row in random_rows.iterrows():
        plt.plot(x_axis, row[l2a_bands], color='lightgray', alpha=0.5, linewidth=1)

    # Plot the mean spectrum
    plt.plot(x_axis, mean_spectrum, color='#6494aa', linewidth=2, marker='o', ms=5, label='Mean')


    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel('Reflectance')
    plt.title('Random spectral signatures from MLW-predictions (n=1000)')
    
    # Add a single label for the random lines in the legend
    plt.plot([], [], color='lightgray', alpha=0.5, linewidth=1, label='MLW (L1C)')
    plt.legend()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_path = os.path.join(output_dir, 's2_linegraph_cozar.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Line graph saved to {output_path}")

def plot_scatter(cozar_csv, output_dir):
    """
    Plots a scatter plot of NDVI vs. FDI for the Cozar dataset and comparison targets (HDPE, PVC, HDPE-BF, Water).

    Parameters:
    - cozar_csv: Path to the Cozar dataset CSV file.
    - output_dir: Directory where the scatter plot image will be saved.
    """
    # Load Cozar dataset
    cozar_data = pd.read_csv(cozar_csv)

    # Paths to comparison datasets
    target_files = {
        "HDPE": "data/processed/HDPE_reflectance.csv",
        "PVC": "data/processed/PVC_reflectance.csv",
        "HDPE-BF": "data/processed/HDPE-BF_reflectance.csv",
        "Water": "data/processed/Water_reflectance.csv"  # Added Water dataset
    }

    # Load target datasets
    targets = {}
    for target, path in target_files.items():
        targets[target] = pd.read_csv(path)

    # Helper functions for NDVI and FDI calculation
    def calculate_ndvi(data):
        return (data['B8'] - data['B4']) / (data['B8'] + data['B4'])

    def calculate_fdi(data):
        lambda_nir = 832
        lambda_swir1 = 1613
        lambda_red = 664
        nir_prime = data['B4'] + (data['B11'] - data['B4']) * (lambda_nir - lambda_red) / (lambda_swir1 - lambda_red)
        return data['B8'] - nir_prime

    # Calculate NDVI and FDI for Cozar dataset
    cozar_data['NDVI'] = calculate_ndvi(cozar_data)
    cozar_data['FDI'] = calculate_fdi(cozar_data)

    # Calculate NDVI and FDI for target datasets
    for target in targets:
        targets[target]['NDVI'] = calculate_ndvi(targets[target])
        targets[target]['FDI'] = calculate_fdi(targets[target])

    # Create scatter plot
    plt.figure(figsize=(10, 6))

    # Scatter for Cozar with custom marker
    plt.scatter(cozar_data['NDVI'], cozar_data['FDI'], alpha=0.2, label='MLW', color='#ccc5b9', marker='o')

    # Define marker styles for each target
    markers = {
        "HDPE": 'x',  # 'x' marker
        "PVC": 's',   # Square marker
        "HDPE-BF": '^',  # Triangle marker
        "Water": 'o'  # Circle marker for Water
    }

    # Define colors for each target
    colors = {
        "HDPE": '#ff7d00', 
        "PVC": '#15616d', 
        "HDPE-BF": '#001524',
        "Water": '#6495ED'  # Color for Water
    }

    # Scatter for each target with different markers and colors
    for target, data in targets.items():
        plt.scatter(data['NDVI'], data['FDI'], alpha=1.0, marker=markers[target], label=target, color=colors[target])

    # Add labels, legend, and grid
    plt.xlabel('NDVI')
    plt.ylabel('FDI')
    plt.title('Scatter Plot of NDVI vs FDI')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, zorder=2)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save and show the plot
    output_path = os.path.join(output_dir, "s2_scatter.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Scatter plot saved at {output_path}")

def plot_pca(output_file, l2a_bands, output_dir):
    """
    Generates and plots a PCA visualization for the spectral data and comparison targets (HDPE, PVC, HDPE-BF, Water).
    
    Parameters:
    - output_file: Path to the Cozar dataset CSV file.
    - l2a_bands: List of band names corresponding to spectral bands (e.g., ['B1', 'B2', ..., 'B12']).
    - output_dir: Directory where the PCA plot image will be saved.
    """
    # Load Cozar dataset
    cozar_data = pd.read_csv(output_file)

    # Paths to comparison datasets
    target_files = {
        "HDPE": "data/processed/HDPE_reflectance.csv",
        "PVC": "data/processed/PVC_reflectance.csv",
        "HDPE-BF": "data/processed/HDPE-BF_reflectance.csv",
        "Water": "data/processed/Water_reflectance.csv"  # Added Water dataset
    }

    # Load target datasets
    targets = {}
    for target, path in target_files.items():
        targets[target] = pd.read_csv(path)

    # Extract relevant bands from the datasets
    data = cozar_data[l2a_bands].copy()

    # Add target datasets
    for target, target_data in targets.items():
        target_data = target_data[l2a_bands]
        data = pd.concat([data, target_data], ignore_index=True)

    # Perform PCA
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(data)

    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2', 'PC3'])

    # Assign labels to the PCA DataFrame
    labels = ['Cozar'] * len(cozar_data)  # Cozar dataset label
    for target in target_files.keys():
        labels += [target] * len(targets[target])  # Add target labels
    
    pca_df['Label'] = labels

    # Define colors and markers
    colors = {
        "HDPE": '#ff7d00',
        "PVC": '#15616d',
        "HDPE-BF": '#001524',
        "Water": '#6495ED',  # Color for Water
        "Cozar": '#ccc5b9'  # Color for Cozar dataset
    }

    markers = {
        "HDPE": 'x',       # 'x' marker for HDPE
        "PVC": 's',        # Square marker for PVC
        "HDPE-BF": '^',    # Triangle marker for HDPE-BF
        "Water": 'o',      # Circle marker for Water
        "Cozar": 'o'       # Circle marker for Cozar
    }

    # Create the plot with two subplots (PC1 vs PC2, PC2 vs PC3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot PC1 vs PC2 on the first subplot
    cozar_subset = pca_df[pca_df['Label'] == 'Cozar']
    axes[0].scatter(cozar_subset['PC1'], cozar_subset['PC2'], label='MLW', color=colors['Cozar'], marker=markers['Cozar'], alpha=0.4, s=100)

    for label in target_files.keys():
        subset = pca_df[pca_df['Label'] == label]
        axes[0].scatter(subset['PC1'], subset['PC2'], label=label, color=colors[label], marker=markers[label], alpha=1.0, s=100)

    axes[0].set_title('PCA: PC1 vs PC2')
    axes[0].grid(True, linestyle='--', alpha=0.7, zorder=-1)  # Grid below the scatter points

    # Plot PC2 vs PC3 on the second subplot
    cozar_subset = pca_df[pca_df['Label'] == 'Cozar']
    axes[1].scatter(cozar_subset['PC2'], cozar_subset['PC3'], label='MLW', color=colors['Cozar'], marker=markers['Cozar'], alpha=0.4, s=100)

    for label in target_files.keys():
        subset = pca_df[pca_df['Label'] == label]
        axes[1].scatter(subset['PC2'], subset['PC3'], label=label, color=colors[label], marker=markers[label], alpha=1.0, s=100)

    axes[1].set_title('PCA: PC2 vs PC3')
    axes[1].grid(True, linestyle='--', alpha=0.7, zorder=-1)  # Grid below the scatter points

    # Add legend to the second subplot (PC2 vs PC3)
    axes[1].legend(title='Targets', loc='upper right')

    # Customize the layout and appearance
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, "s2_pca.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"PCA plot saved at {output_path}")

def plot_planetscope_acquisitions(file_path):
    """
    Plots a bar chart showing the number of potential images per year for each sensor
    from a DataFrame, excluding rows where potential_overlap is 'no'.
    """
    
    # Read data from the CSV
    df = pd.read_csv(file_path)

    # Filter out rows where potential_overlap is 'no'
    df = df[df['potential_overlap'] == 'yes']
    
    # If no valid data is left after filtering, print a message and return
    if df.empty:
        print("No valid data available for plotting.")
        return
    
    # Handle columns where the values are in list format (image_ids, sensor_ids, image_types)
    df['image_ids'] = df['image_ids'].apply(ast.literal_eval)  # Convert the string of list back to a list
    df['sensor_ids'] = df['sensor_ids'].apply(ast.literal_eval)
    df['image_types'] = df['image_types'].apply(ast.literal_eval)
    
    # Check if all lists are of equal length for each row
    mismatched_lengths = df[df.apply(lambda row: len(row['image_ids']) != len(row['sensor_ids']) or len(row['sensor_ids']) != len(row['image_types']), axis=1)]
    
    if not mismatched_lengths.empty:
        print(f"Warning: Found {len(mismatched_lengths)} rows with mismatched list lengths. These rows will be excluded.")
        df = df[df.apply(lambda row: len(row['image_ids']) == len(row['sensor_ids']) == len(row['image_types']), axis=1)]
    
    # Exploding the lists so that each row corresponds to a single image
    df_exploded = df.explode(['image_ids', 'sensor_ids', 'image_types']).reset_index(drop=True)
    
    # Rename 'acquisition_date' to 'date'
    df_exploded['date'] = pd.to_datetime(df_exploded['date'], errors='coerce')
    
    # Drop rows where 'date' could not be converted
    df_exploded = df_exploded.dropna(subset=['date'])
    
    # Extract the year from the date
    df_exploded['year'] = df_exploded['date'].dt.year
    
    # Group by year and sensor_id to count the number of unique image_ids
    image_count_by_year_and_sensor = df_exploded.groupby(['year', 'sensor_ids'])['image_ids'].nunique().reset_index(name='unique_image_count')
    
    # If there's no data after grouping, print a message and return
    if image_count_by_year_and_sensor.empty:
        print("No valid image data to plot.")
        return

    # Flatten the sensor_id lists for the plot, converting them to a more readable format
    sensor_mapping = {
        'PS2': 'Dove-C',
        'PS2.SD': 'Dove-R',
        'PSB.SD': 'SuperDove'
    }
    
    # Map sensor IDs to descriptive names
    image_count_by_year_and_sensor['sensor_ids'] = image_count_by_year_and_sensor['sensor_ids'].map(sensor_mapping).fillna(image_count_by_year_and_sensor['sensor_ids'])
    
    # Custom color palette
    custom_colors = {
        'Dove-C': '#15616d',   
        'Dove-R': '#ff7d00',   
        'SuperDove': '#001524'
    }
    
    # Styling with Seaborn
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Bar plot with individual bars for each sensor
    sns.barplot(
        x='year', 
        y='unique_image_count', 
        hue='sensor_ids', 
        data=image_count_by_year_and_sensor, 
        palette=custom_colors
    )

    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add title and labels
    plt.title('Potential Marine Litter Windrow Scenes from PlanetScope', fontsize=14)
    plt.ylabel('Frequency of sensor occurrences', fontsize=12)
    plt.xlabel('')  # Remove 'Year' from X-axis label
    plt.xticks(rotation=45, fontsize=10)  # Rotate x-ticks for better readability
    plt.legend(title='PS-sensor', fontsize=10)

    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig("doc/figures/planetscope_acquisitions.png")
    print('Plot saved as "doc/figures/planetscope_acquisitions.png"')

def plot_spectral_angle_signatures(file_path, l2a_bands, top_n=5):
    """
    Plots the distribution of spectral angles and the spectral signatures of the top-N lowest angles.
    
    Args:
    - file_path: Path to the CSV file containing spectral data.
    - l2a_bands: List of spectral band columns to use for spectral signatures (e.g., ['B1', ..., 'B12']).
    - top_n: Number of observations with the lowest spectral angles to visualize.
    
    Returns:
    - A combined figure with spectral angle distribution and spectral signatures.
    """
    # Wavelengths dictionary for converting bands
    wavelengths = {
        "B1": 442, "B2": 492, "B3": 559, "B4": 664,
        "B5": 704, "B6": 740, "B7": 782, "B8": 832,
        "B8A": 864, "B9": 945, "B11": 1613, "B12": 2202
    }
    x_axis = [wavelengths[band] for band in l2a_bands]

    # Load the data
    df = pd.read_csv(file_path)

    # Filter relevant columns
    spectral_angles = ['sam_HDPE', 'sam_PVC', 'sam_HDPE_BF', 'sam_water']
    rename_mapping = {'sam_HDPE': 'HDPE', 'sam_PVC': 'PVC', 'sam_HDPE_BF': 'HDPE-BF', 'sam_water': 'Water'}

    # Melt the spectral angles into a single column for easier plotting
    df_melted = df.melt(id_vars=l2a_bands + ['date'], 
                        value_vars=spectral_angles, 
                        var_name='target', 
                        value_name='spectral_angle')

    # Apply renaming to the target column
    df_melted['target'] = df_melted['target'].map(rename_mapping)

    # Drop rows with NaN spectral angles
    df_melted = df_melted.dropna(subset=['spectral_angle'])

    # Top-N lowest spectral angles
    top_spectral = df_melted.nsmallest(top_n, 'spectral_angle')

    # Define custom colors
    colors = {
        "HDPE": '#ff7d00',
        "PVC": '#15616d',
        "HDPE-BF": '#001524',
        "Water": '#6495ED',
        "Cozar": '#ccc5b9'
    }
    
    # Custom color palette for seaborn
    palette = {key: colors[key] for key in df_melted['target'].unique() if key in colors}

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})

    # Top plot: Spectral angle distribution
    sns.histplot(data=df_melted, x='spectral_angle', hue='target', ax=axes[0], kde=True, palette=palette)
    axes[0].set_title('Spectral Angle Distribution')
    axes[0].set_xlabel('Spectral Angle (°)') 
    axes[0].set_ylabel('Frequency')
    axes[0].grid(visible=True, linestyle='--', alpha=0.7)

    # Bottom plot: Spectral signatures of top-N
    for _, row in top_spectral.iterrows():
        band_values = row[l2a_bands].values
        axes[1].plot(x_axis, band_values, marker='o', label=f"{row['target']}: {row['spectral_angle']:.1f}°", color=colors.get(row['target'], '#000000'))

    axes[1].set_title(f'Spectral Signatures of Top-{top_n} Observations by Lowest Spectral Angle')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Reflectance')
    axes[1].legend(title='Target: Spectral Angle (°)', loc='upper right')
    axes[1].grid(visible=True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("doc/figures/spectral_angle_signatures.png")
    print('Plot saved as "doc/figures/spectral_angle_signatures.png"')

def plot_tsne(cozar_dataset, l2a_bands, output_dir, perplexity=30, learning_rate=200, n_iter=1000):
    """
    Generates and plots a t-SNE visualization for the spectral data, including L1C, L2A, and other datasets (HDPE, Wood, HDPE+Wood).
    
    Parameters:
    - cozar_dataset: DataFrame containing spectral reflectance data with 'atm_level' column.
    - l2a_bands: List of band names corresponding to spectral bands (e.g., ['B1', 'B2', ..., 'B12']).
    - output_files: Dictionary containing paths to additional datasets, e.g., {'HDPE': path_to_hdpe, 'Wood': path_to_wood, 'HDPE+Wood': path_to_hdpe_wood}.
    - output_dir: Directory path to save the output PNG file.
    - perplexity: t-SNE perplexity parameter, which affects the balance between local and global aspects of the data.
    - learning_rate: t-SNE learning rate parameter.
    - n_iter: Maximum number of iterations for optimization.
    """
    # Paths to the three CSV files
    output_files = {
        "HDPE": "data/processed/HDPE_reflectance.csv",
        "PVC": "data/processed/PVC_reflectance.csv",
        "HDPE-BF": "data/processed/HDPE-BF_reflectance.csv",
        "HDPE-C": "data/processed/HDPE-C_reflectance.csv",
        "HDPE+Wood": "data/processed/HDPE+Wood_reflectance.csv",
        "wood": "data/processed/wood_reflectance.csv"
    }

    # Colors for the datasets
    colors = {
        "HDPE": '#6494aa',
        "LW (L1C)": '#a63d40',  # Litter windrow (L1C)
        "LW (L2A)": '#e9b872',  # Litter windrow (L2A)
        "PVC": '#6a5acd',
        "HDPE-BF": '#4682b4',
        "HDPE-C": '#ff6347',
        "HDPE+Wood": '#904ca9',
        "wood": '#90a959'
    }


    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    """ Prepare Cozar data (L1C, L2A) """
    cozar_L1C = cozar_dataset[cozar_dataset['atm_level'] == 'L1C'][l2a_bands]
    cozar_L2A = cozar_dataset[cozar_dataset['atm_level'] == 'L2A'][l2a_bands]

    # Labels for L1C and L2A
    data = pd.concat([cozar_L1C, cozar_L2A])
    labels = ['LW (L1C)'] * len(cozar_L1C) + ['LW (L2A)'] * len(cozar_L2A)

    """ Add HDPE, Wood, HDPE+Wood datasets """
    for label, output_file in output_files.items():
        # Load the dataset
        additional_data = pd.read_csv(output_file)

        # Extract the bands and append to the data
        data = pd.concat([data, additional_data[l2a_bands]])
        labels += [label] * len(additional_data)

    """ Perform t-SNE """
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, max_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(data)

    """ Plot the t-SNE results """
    tsne_df = pd.DataFrame(tsne_results, columns=['Dim 1', 'Dim 2'])
    tsne_df['Label'] = labels

    # Use seaborn to create a scatter plot with different colors for each dataset
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Dim 1', y='Dim 2', 
                    hue='Label', 
                    palette=colors,
                    data=tsne_df, 
                    s=100, alpha=0.7)

    # Customize plot
    plt.title('t-SNE plot of spectral reflectance')
    plt.legend(loc='best', title='Class')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set Y-axis to 0 as negative values are not possible
    #plt.ylim(0)

    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, "s2_tsne_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"t-SNE plot saved at {output_path}")