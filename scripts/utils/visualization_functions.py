import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import torch

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



def plot_spectral_analysis(file_path, l2a_bands, output_dir, use_wavelengths=False, top_n=10):
    """
    Plots line graphs with MLW as the baseline signature in each subplot. 
    All other spectral signatures, including HDPE, are compared to MLW.
    Also adds a third row with a large spectral angle distribution plot spanning all three columns.

    Parameters:
    - cozar_csv: Path to the CSV file containing spectral reflectance data (reference data).
    - l2a_bands: List of band names corresponding to spectral bands.
    - output_dir: Directory path to save the output PNG file.
    - use_wavelengths: If True, use wavelengths (in nm) for the X-axis. Otherwise, use band names.
    - file_path: Path to the CSV file for spectral angle data (needed for the large figure).
    - top_n: Number of top N lowest spectral angles to highlight.
    """
    # Mapping from bands to wavelengths
    wavelengths = {
        "B1": 442, "B2": 492, "B3": 559, "B4": 664,
        "B5": 704, "B6": 740, "B7": 782, "B8": 832,
        "B8A": 864, "B9": 945, "B11": 1613, "B12": 2202
    }
    x_axis = [wavelengths[band] for band in l2a_bands] if use_wavelengths else l2a_bands
    x_label = 'Wavelength [nm]' if use_wavelengths else 'Bands'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    """ Load and Process Data """
    # Load the Cozar dataset (MLW)
    cozar_dataset = pd.read_csv(file_path)
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

    """ Create Figure with GridSpec """
    # Create a 3-row grid with equal height for each row
    fig = plt.figure(figsize=(6, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])  # Equal height for all rows

    # 1. First two rows: Create subplots for each target
    axes = []  # List to hold the axes for the first 6 subplots.
    for i, label in enumerate(targets):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        axes.append(ax)
        # Plot MLW (baseline signature) in each subplot
        ax.plot(x_axis, mlw_means, label=f"MLW (n={mlw_n})", color='#6494aa',
                linewidth=2, marker='o', ms=5)
        ax.fill_between(x_axis, mlw_means - mlw_stds, mlw_means + mlw_stds,
                        color='#6494aa', alpha=0.2)

        # Plot the target dataset in the current subplot
        ax.plot(x_axis, target_means[label], label=f"{label} (n={target_n[label]})", 
                color={'HDPE': '#ff7d00', 'PVC': '#15616d', 'HDPE-BF': '#001524',
                       'HDPE-C': '#8a3d62', 'wood': '#6c4f3d', 'Water': '#6495ED'}[label],
                linewidth=2, marker={'HDPE': 'x', 'PVC': 's', 'HDPE-BF': '^',
                                       'HDPE-C': 'D', 'wood': 'H', 'Water': 'o'}[label],
                markersize=5)
        ax.fill_between(x_axis, target_means[label] - target_stds[label],
                        target_means[label] + target_stds[label],
                        color={'HDPE': '#ff7d00', 'PVC': '#15616d', 'HDPE-BF': '#001524',
                               'HDPE-C': '#8a3d62', 'wood': '#6c4f3d', 'Water': '#6495ED'}[label],
                        alpha=0.2)

        # Customize subplot
        #ax.set_title(f'{label}')
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', linewidth=0.5)
        if i >= 3:
            ax.set_xlabel(x_label)
        if i % 3 == 0:
            ax.set_ylabel('TOA reflectance')
        ax.legend(loc='upper right')

    # 2. Third row: Create a large axis spanning all 3 columns for spectral angle distribution.
    ax_angle = fig.add_subplot(gs[2, :])

    # (x_axis is redefined here if needed)
    x_axis = [wavelengths[band] for band in l2a_bands]

    # Load the data for spectral angles
    df = pd.read_csv(file_path)

    # Filter relevant columns and melt the data for easier plotting
    spectral_angles = ['sam_HDPE', 'sam_PVC', 'sam_HDPE_BF', 'sam_water']
    rename_mapping = {'sam_HDPE': 'HDPE', 'sam_PVC': 'PVC',
                        'sam_HDPE_BF': 'HDPE-BF', 'sam_water': 'Water'}
    df_melted = df.melt(id_vars=l2a_bands + ['date'], 
                        value_vars=spectral_angles, 
                        var_name='Target', 
                        value_name='spectral_angle')
    df_melted['Target'] = df_melted['Target'].map(rename_mapping)
    df_melted = df_melted.dropna(subset=['spectral_angle'])

    # Top-N lowest spectral angles (if needed)
    top_spectral = df_melted.nsmallest(top_n, 'spectral_angle')

    # Define custom colors for the spectral angle plot
    angle_colors = {
        "HDPE": '#ff7d00',
        "PVC": '#15616d',
        "HDPE-BF": '#001524',
        "Water": '#6495ED',
        "Cozar": '#ccc5b9'
    }
    palette = {key: angle_colors[key] for key in df_melted['Target'].unique() if key in angle_colors}

    # Plot the spectral angle distribution on the large axis
    sns.histplot(data=df_melted, x='spectral_angle', hue='Target', ax=ax_angle, kde=True, palette=palette)
    #ax_angle.set_title('Spectral Angle Distribution')
    ax_angle.set_xlabel('Spectral angle [°]')
    ax_angle.set_ylabel('Frequency')
    ax_angle.grid(True, linestyle='--', alpha=0.7)

    # Set identical Y-limits for all subplots
    all_y_values = [ax.get_ylim() for ax in axes]
    y_min = min([y[0] for y in all_y_values])
    y_max = max([y[1] for y in all_y_values])
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    output_filename = "spectral_analysis"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(f"{output_path}.png", dpi=300)
    plt.savefig(f"{output_path}.pdf", dpi=300)
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
    x_label = 'Wavelength [nm]' if use_wavelengths else 'Bands'

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
    x_label = 'Wavelength [nm]' if use_wavelengths else 'Bands'

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
    axes[1].set_xlabel('Wavelength [nm]')
    axes[1].set_ylabel('Reflectance')
    axes[1].legend(title='Target: Spectral Angle (°)', loc='upper right')
    axes[1].grid(visible=True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("doc/figures/spectral_angle_signatures.png")
    print('Plot saved as "doc/figures/spectral_angle_signatures.png"')

def plot_featurespace(test_loader, train_loader, val_loader):
    """Generate scatter plots comparing RAI and NDVI between the test region and each training/validation region."""
    def compute_indices(image_batch):
        """ Compute NDVI and RAI for a batch of images. """
        blue, green, red, nir = image_batch[:, 0], image_batch[:, 1], image_batch[:, 2], image_batch[:, 3]
        epsilon = 1e-10
        ndvi = (nir - red) / (nir + red + epsilon)
        rai = (nir - blue) / (nir + blue + epsilon)
        return ndvi, rai
    
    # Extract test data
    test_images, test_masks, test_region_ids = next(iter(test_loader))
    test_ndvi, test_rai = compute_indices(test_images)

    # Extract only pixels belonging to MLWs (masked regions)
    test_ndvi_select = test_ndvi[test_masks.bool()]
    test_rai_select = test_rai[test_masks.bool()]
    
    # Combine train and val loaders
    combined_loaders = {"Train": train_loader, "Validation": val_loader}
    
    # Store unique regions
    unique_regions = {}

    for loader_name, loader in combined_loaders.items():
        for images, masks, region_ids in loader:
            for i in range(images.shape[0]):  # Loop over batch elements
                region_id = region_ids[i]
                
                if region_id not in unique_regions:
                    unique_regions[region_id] = {
                        "ndvi": [], 
                        "rai": [], 
                        "name": f"{loader_name} Region {region_id}",
                        "region": region_id
                    }
                
                # Compute indices for the specific image
                ndvi, rai = compute_indices(images[i].unsqueeze(0))  # Keep batch dimension
                ndvi = ndvi.squeeze(0)
                rai = rai.squeeze(0)
                
                # Extract only pixels belonging to MLWs
                ndvi_select = ndvi[masks[i].bool()]
                rai_select = rai[masks[i].bool()]

                # Store data separately per region
                unique_regions[region_id]["ndvi"].append(ndvi_select)
                unique_regions[region_id]["rai"].append(rai_select)
    
    # Create subplots (adjusting rows and columns to fit 13 subplots without extra grids)
    num_regions = len(unique_regions)
    num_plots = num_regions + 1  # One extra plot for the test data

    # Calculate number of rows and columns to fit all plots without empty grids
    ncols = (num_plots + 2) // 3  # At least 3 rows, as you have 13 plots in total
    nrows = (num_plots + ncols - 1) // ncols  # Adjust rows based on number of columns

    fig, axes = plt.subplots(nrows, ncols, figsize=(0.8 * num_plots, 0.6 * num_plots), sharex=True, sharey=True, constrained_layout=True)

    # Flatten the axes for easy iteration
    axes = axes.flatten()

    # Ensure axes is iterable if there's only one region
    if num_regions == 1:
        axes = [axes]

    # Remove extra axes if any (this happens if the subplot grid is larger than needed)
    for ax in axes[num_plots:]:
        ax.axis('off')  # Hide axes that are not used

    # Convert test dataset to NumPy for processing
    test_ndvi_np = test_ndvi_select.numpy()
    test_rai_np = test_rai_select.numpy()
    test_points = np.column_stack((test_ndvi_np, test_rai_np))  # Combine into (x, y) points

    # Compute the convex hull (if there are enough points)
    hull = None
    if len(test_points) > 2:
        hull = ConvexHull(test_points)
        hull_vertices = test_points[hull.vertices]  # Get the boundary points

    # Plot the test data on the first subplot
    ax = axes[0]
    ax.scatter(test_ndvi_np, test_rai_np, alpha=0.3, s=3, c="#F2CC8F", label="Validated MLW")
    ax.set_xlabel("NDVI", fontsize=8)
    ax.set_ylabel("RAI", fontsize=8)
    handles = [mpatches.Patch(facecolor="#F2CC8F", label="Validated MLW")]
    ax.legend(handles=handles, loc='upper left', fontsize=8)

    # Loop through the remaining regions and plot training data
    for idx, (ax, (region_id, data)) in enumerate(zip(axes[1:], unique_regions.items())):
        # Concatenate NDVI/RAI values for this region
        ndvi_concat = torch.cat(data["ndvi"]).numpy()
        rai_concat = torch.cat(data["rai"]).numpy()

        # Plot the training data
        ax.scatter(ndvi_concat, rai_concat, alpha=0.3, s=3, c="#3D405B")

        # Plot the convex hull outline of the test dataset
        if hull is not None:
            ax.plot(hull_vertices[:, 0], hull_vertices[:, 1], "--", color="#F2CC8F", linewidth=1.5)

        ax.set_xlabel("NDVI", fontsize=8)
        ax.set_ylabel("RAI", fontsize=8)

        # Add legend for the training data, reformat from '20201202_Egypt' to '2020-12-02, Egypt'
        label = f"{data['region'][:4]}-{data['region'][4:6]}-{data['region'][6:8]}, {data['region'][9:]}"
        handles = [mpatches.Patch(facecolor="#3D405B", label=label)]
        ax.legend(handles=handles, loc='upper left', fontsize=8)

    # Save the figure
    plt.savefig('doc/figures/litterlines_distribution.png', dpi=600, bbox_inches='tight')
    plt.savefig('doc/figures/litterlines_distribution.pdf')
    plt.close()

def plot_litterlines_patches(images, masks, region_ids):
    num_patches = 5  # Number of patches to visualize

    # Convert tensors to numpy arrays
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    region_ids = np.array(region_ids)  # Convert to NumPy array for easy filtering

    # Get unique regions and one index per region
    unique_regions, unique_indices = np.unique(region_ids, return_index=True)

    # Select up to `num_patches` unique regions
    selected_indices = unique_indices[:num_patches]

    # Define colormap for mask visualization
    binary_cmap = ListedColormap(['#FCF3EE', '#68000D'])

    # Custom colormap for NDVI/RAI to handle 0.0 as black
    cmap_ndvi = plt.cm.coolwarm
    cmap_ndvi.set_bad(color='black')  # Masked values appear black

    cmap_rai = plt.cm.coolwarm
    cmap_rai.set_bad(color='black')  # Masked values appear black

    # Create a 4x4 grid for visualization
    fig, axes = plt.subplots(num_patches, 4, figsize=(6.5, 7), constrained_layout=True)

    for i, idx in enumerate(selected_indices):
        image_patch = images[idx]
        mask_patch = masks[idx]
        region_id = region_ids[idx]

        blue, green, red, nir = image_patch
        rgb = np.stack([red, green, blue])

        # Normalize RGB channels
        vmin_rgb, vmax_rgb = 0, 0.15
        rgb_normalized = np.clip((rgb - vmin_rgb) / (vmax_rgb - vmin_rgb), 0, 1)
        rgb_image = rgb_normalized.transpose(1, 2, 0)

        # Compute indices
        epsilon = 1e-10
        ndvi = (nir - red) / (nir + red + epsilon)
        rai = (nir - blue) / (nir + blue + epsilon)

        vmin_ndvi, vmax_ndvi = -0.20, 0.05
        vmin_rai, vmax_rai = -0.5, 0

        # Mask 0.0 values
        ndvi_masked = np.ma.masked_where(ndvi == 0.0, ndvi)
        rai_masked = np.ma.masked_where(rai == 0.0, rai)

        axes[i, 0].imshow(rgb_image, vmin=0, vmax=1)
        axes[i, 0].axis('off')
        axes[0, 0].set_title("RGB", fontsize=11)

        im1 = axes[i, 1].imshow(ndvi_masked, cmap=cmap_ndvi, vmin=vmin_ndvi, vmax=vmax_ndvi)
        axes[0, 1].set_title("NDVI", fontsize=11)
        axes[i, 1].axis('off')

        im2 = axes[i, 2].imshow(rai_masked, cmap=cmap_rai, vmin=vmin_rai, vmax=vmax_rai)
        axes[0, 2].set_title("RAI", fontsize=11)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(mask_patch, cmap=binary_cmap)
        axes[0, 3].set_title("Label", fontsize=11)
        axes[i, 3].axis('off')

        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        #axes[i, 0].annotate(region_id, xy=(-0.1, 0.5), xycoords='axes fraction',
        #                    fontsize=10, ha='right', va='center')

    plt.savefig('doc/figures/litterlines_patches.png', dpi=600, bbox_inches='tight')
    plt.savefig('doc/figures/litterlines_patches.pdf')
    plt.close()
    
def plot_signatures(test_loader, train_loader, val_loader):
    """Generate line plots comparing R, G, B, and NIR values between the test region and each training/validation region."""
    def extract_channel_means(image_batch, mask_batch):
        """ Compute mean values for R, G, B, and NIR channels for masked regions. """
        mask_batch = mask_batch.unsqueeze(1)  # Add a channel dimension → Shape becomes [140, 1, 256, 256]
        masked_images = image_batch * mask_batch  # Element-wise multiplication to retain only masked pixels
        valid_pixels = mask_batch.sum(dim=(2, 3))  # Count valid pixels per channel
        
        # Avoid division by zero
        valid_pixels[valid_pixels == 0] = 1
        
        mean_values = masked_images.sum(dim=(2, 3)) / valid_pixels  # Compute mean for each channel
        return mean_values.mean(dim=0).numpy()  # Average across batch
    
    # Extract test data
    test_images, test_masks, test_region_ids = next(iter(test_loader))
    test_means = extract_channel_means(test_images, test_masks)

    # Combine train and val loaders
    combined_loaders = {"Train": train_loader, "Validation": val_loader}

    unique_regions = {}

    for loader_name, loader in combined_loaders.items():
        for images, masks, region_ids in loader:
            for i in range(images.shape[0]):  # Loop over batch elements
                region_id = region_ids[i]
                
                if region_id not in unique_regions:
                    unique_regions[region_id] = {
                        "means": [],
                        "name": f"{loader_name} Region {region_id}",
                        "region": region_id
                    }
                
                means = extract_channel_means(images[i].unsqueeze(0), masks[i].unsqueeze(0))
                unique_regions[region_id]["means"].append(means)

    # Create subplots
    num_regions = len(unique_regions)
    num_plots = num_regions + 1  # Include test region plot

    ncols = (num_plots + 2) // 3  # At least 3 rows
    nrows = (num_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(0.8 * num_plots, 0.6 * num_plots), sharex=True, sharey=True, constrained_layout=True)

    axes = axes.flatten()
    for ax in axes[num_plots:]:
        ax.axis('off')

    x_labels = ["B", "G", "R", "NIR"]
    x_ticks = np.arange(len(x_labels))

    # Plot test region
    axes[0].plot(x_ticks, test_means, marker='x', linestyle='-', color="#F2CC8F", label="Validated MLW \n(Kikaki et al., 2020)")
    
    # Compute standard deviation for test region (assuming you're using the full batch)
    test_stds = np.std(test_images.numpy(), axis=(0, 2, 3))  # Calculate std for R, G, B, NIR over batch
    axes[0].fill_between(x_ticks, test_means - test_stds, test_means + test_stds, color="#F2CC8F", alpha=0.3)
    
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(x_labels)
    axes[0].legend(fontsize=8)

    # Plot training/validation regions
    for idx, (ax, (region_id, data)) in enumerate(zip(axes[1:], unique_regions.items())):
        region_means = np.mean(np.stack(data["means"]), axis=0)  # Compute mean across all images
        region_stds = np.std(np.stack(data["means"]), axis=0)  # Compute standard deviation across batch
        
        # Plot the means
        label = f"{data['region'][:4]}-{data['region'][4:6]}-{data['region'][6:8]}, {data['region'][9:]}"
        ax.plot(x_ticks, region_means, marker='o', linestyle='-', color="#3D405B", label=label)
        
        # Plot the standard deviation as shaded area (±1 standard deviation)
        ax.fill_between(x_ticks, region_means - region_stds, region_means + region_stds, 
                        color="#3D405B", alpha=0.3)  # Adjust alpha for transparency
        
        # Plot test region for comparison
        ax.plot(x_ticks, test_means, marker='x', linestyle='-', color="#F2CC8F")  # Test data
        ax.fill_between(x_ticks, test_means - test_stds, test_means + test_stds, color="#F2CC8F", alpha=0.3)
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.legend(fontsize=8, loc='upper right')

    # Set the Y-axis label for the outer plots (left and bottom axes)
    for idx, ax in enumerate(axes):
        if idx % ncols == 0: # Check for leftmost columns
            ax.set_ylabel('TOA reflectance [-]')#, fontsize=8)

    # Save the plot
    plt.savefig('doc/figures/litterlines_signatures.png', dpi=600, bbox_inches='tight')
    plt.savefig('doc/figures/litterlines_signatures.pdf')
    plt.close()