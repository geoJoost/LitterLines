import os
import glob
import torch
import geopandas as gpd
import rasterio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from rasterio.mask import mask
from rasterio.features import rasterize
from shapely.geometry import mapping
import matplotlib.pyplot as plt

import rasterio.windows

# Custom imports
from dataloader_utils import flag_noisy_label, flag_nir_displacement, parse_reflectance_coefficients, split_linestring

class LitterLinesDataset(Dataset):
    def __init__(self, root_dir, transform=None, patch_size=256):
        """
        Args:
            root_dir (string): Directory with all the scene folders (e.g., /data/PS-LitterLines).
            transform (callable, optional): Optional transform to be applied on an image.
            patch_size (int): Size of patches to be used for segmentation.
        """
        self.root_dir = root_dir
        self.geojson_path = os.path.join(root_dir, "mlw_annotations_20250121.gpkg")
        self.transform = transform
        self.patch_size = patch_size
        
        # Read the GeoPackage with litter lines into a GeoDataFrame
        self.litter_data = gpd.read_file(self.geojson_path).to_crs(epsg=4326)

        # Get all the available scene folders in the dataset
        self.scene_folders = glob.glob(os.path.join(self.root_dir, '*/*/'))[:1] # TODO: Remove hard-coding. Used for debugging

    def __len__(self):
        return len(self.scene_folders)

    def __getitem__(self, idx):
        # Select a scene
        scene_folder = self.scene_folders[idx]
        #scene_folder = 'data/PS-LitterLines/20181107_Italy/1003' #/20201202_075858_57_2264_3B_AnalyticMS_clip.tif'
        
        # Load the corresponding PlanetScope scenes
        image_paths = glob.glob(os.path.join(scene_folder, "*AnalyticMS*.tif"))
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No matching image files found in {scene_folder}")
        
        # Parse reflectance coefficients from the associated XML file
        xml_paths = glob.glob(os.path.join(scene_folder, "*metadata*.xml"))
        if len(xml_paths) == 0:
            raise FileNotFoundError(f"No XML metadata file found in {scene_folder}")

        # Initialize lists
        all_patches, all_masks = [], []

        # Iterate over each individual PlanetScope scene
        for image_path, xml_path in zip(image_paths, xml_paths):
            # Get reflectance coefficients for converting TOAR into TOA
            reflectance_coefficients = parse_reflectance_coefficients(xml_path)

            # Get raster attributes for creating image patches
            with rasterio.open(image_path) as src:
                raster_transform = src.transform 
                crs = src.crs
            
            # Extract PlanetScope ID from the image path
            #scene_name = '20210430_082512_56_242d' # TODO: Remove this line. Used for 4-channel displacement
            scene_name = os.path.basename(image_path).split("_3B")[0] # Get PS filenames like '20201202_075858_57_2264'
            scene_annotations = self.litter_data[self.litter_data['ps_product'] == scene_name]

            if scene_annotations.empty:
                print(f"No annotations found for '{image_path}'")
                continue
            print(f"Working on images from '{image_path}'")	
        
            # Generate patches and masks
            patches, masks = self._create_patches_and_masks(image_path, raster_transform, crs, scene_annotations, reflectance_coefficients)

            print(f"Number of patches: {len(patches)} for PlanetScope scene: {scene_name}\n")
            
            # Apply transforms if any
            if self.transform:
                patches = [self.transform(patch) for patch in patches]
                masks = [self.transform(mask) for mask in masks]

            # Append patches and masks for the current scene to the lists
            all_patches.extend(patches)
            all_masks.extend(masks)
        
        # Append the regionID to use in train-validation-test split later, to prevent spatial autocorrelation
        region_id = image_path.split("/")[2]
        #region_ids = [region_id] * len(all_patches)

        return region_id, all_patches, all_masks
    
    def _create_patches_and_masks(self, image_path, transform, crs, annotations, reflectance_coefficients):
        """
        Create 256x256px patches around annotations and corresponding masks.
        """
        patch_size = self.patch_size
        patches = []
        masks = []

        # Window size in geographic coordinates
        window_size = patch_size * transform[0]  # Transform[0] gets pixel size in meters
        annotations = annotations.to_crs(crs) # EPSG:4326 into local CRS

        # To prevent loss of annotations, make sure the linestrings do not exceed patch size
        # If this occurs, break up the linestring into equal segments

        # Splint individual linestrings into segments < window_size to make sure all annotations are captured within the labels
        annotations['temp_geometry'] = annotations['geometry'].apply(
            lambda geom: split_linestring(geom, max_length=window_size)
        )
        # Explode the new geometries and replace the original data
        annotations = annotations.explode('temp_geometry', ignore_index=True)
        annotations['geometry'] = annotations['temp_geometry']

        assert annotations['geometry'].length.max() <= window_size, "Found segments exceeding window size"

        # Iterate through each linestring in the annotations
        with rasterio.open(image_path) as src:
            for _, row in annotations.iterrows():
                geom = row['geometry']
                #print(f"Linestring length: {geom.length:.2f}")
                if not geom.is_valid:
                    print(f"Linestring in PS-scene '{row['ps_product']}' with ID #{row.name} is invalid")
                    continue

                # For each linestring, get the centroid to create a window (=patch)
                x, y = geom.centroid.x, geom.centroid.y
                window = rasterio.windows.from_bounds(
                    x - window_size / 2, y - window_size / 2,
                    x + window_size / 2, y + window_size / 2,
                    transform
                )
                    
                # Read the image data within the window
                patch = src.read(window=window).astype(np.float32)
                
                # Ensure the patch is of the correct size
                if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                    continue
               
                # Convert TOAR to TOA reflectance in-place
                for band_idx in range(patch.shape[0]):
                    coefficient = reflectance_coefficients.get(band_idx + 1)
                    patch[band_idx] *= coefficient
                              
                def quick_viz(patch):
                    import matplotlib.pyplot as plt
                    import numpy as np

                    # Reorganizing channels from BGR to RGB
                    bgr = patch[:3]
                    rgb = bgr[::-1]

                    # Normalize RGB for visualization using 2%-98% percentiles
                    vmin_rgb, vmax_rgb = np.percentile(rgb, [1, 99])
                    rgb_normalized = np.clip((rgb.transpose(1, 2, 0) - vmin_rgb) / (vmax_rgb - vmin_rgb), 0, 1)

                    # Prepare individual channels for visualization
                    channels = ['Blue', 'Green', 'Red', 'NIR']
                    colors = ['blue', 'green', 'red', 'purple'] 
                    patch_normalized = []
                    for i in range(4):
                        vmin, vmax = np.percentile(patch[i], [1, 99])
                        patch_normalized.append(np.clip((patch[i] - vmin) / (vmax - vmin), 0, 1))

                    # Create a figure with 3 rows and 2 columns
                    fig, axes = plt.subplots(3, 2, figsize=(6, 10))

                    # Plot 1: RGB visualization
                    axes[0, 0].imshow(rgb_normalized)
                    axes[0, 0].axis('off')
                    axes[0, 0].set_title("True-colour (1%-99%)", fontsize=11)

                    # Plot 2: Histogram
                    for i, channel in enumerate(channels):
                        axes[0, 1].hist(
                            patch[i].flatten(), bins=256,# range=(0, 1),
                            color=colors[i], alpha=0.6, label=channel
                        )
                    axes[0, 1].set_title("Histogram of TOA reflectance")
                    #axes[0, 1].set_xlabel("Reflectance")
                    #axes[0, 1].set_ylabel("Frequency")

                    axes[0, 1].legend(loc='upper right')
                    axes[0, 1].grid(axis='y', alpha=0.75)

                    # Plot individual channels with cross gridlines
                    for i in range(4):
                        ax = axes[1 + i // 2, i % 2]  # Map to the second and third rows
                        ax.imshow(patch_normalized[i])#, cmap=colors[i])
                        ax.axhline(patch.shape[1] // 2, color='white', linestyle='--', alpha=0.7)  # Horizontal cross
                        ax.axvline(patch.shape[2] // 2, color='white', linestyle='--', alpha=0.7)  # Vertical cross
                        ax.set_title(f"{channels[i]}", fontsize=11)
                        ax.axis('off')

                    # Adjust layout and save the figure
                    plt.tight_layout()
                    plt.savefig(f"doc/debug/rgb_patch.png", bbox_inches='tight')
                    plt.close()
                    print('Plotted figure')
                #quick_viz(patch)

                # Construct Shapely window for retrieving all annotations within the newly created patch
                from shapely.geometry import box
                window_bounds = (
                    x - window_size / 2, y - window_size / 2,
                    x + window_size / 2, y + window_size / 2
                )

                # Clip the annotations to image patch
                annotations_clip = gpd.clip(annotations, box(*window_bounds))

                # In the annotations, we have three labels:
                ## 0: Not used, background
                ## 1: Debris targets / suspected plastics
                ## 2: Organic debris / mucilage
                ## 3: Misc targets, such as wakes, clouds, and sensor noise
                # We use the third type for hard negative mining, therefore, no positive values should be generated in the final mask
                if int(row['type']) == 3:
                    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
                else:
                    # For all other types, rasterize the annotations and create binary masks
                    window_transform = src.window_transform(window) # Derive the transform for the current window
                    line_raster = rasterize(
                        [(geom, 1) for geom in annotations_clip['geometry']],
                        out_shape=(patch_size, patch_size),
                        transform=window_transform,
                        fill=0,
                        all_touched=True,
                        dtype=np.uint8
                    )

                    # Generate the corresponding mask
                    mask = self._generate_mask(patch, line_raster)
                
                # Re-order from CHW to HWC for self.transform()
                patch = patch.transpose(1, 2, 0)

                # Flag 1: Check whether positive labels exceed >20% of all pixels in the label
                # If this is the case, it is likely that the Otsu threshold has failed for this label 
                # And therefore, it generated noisy and unusable segmentation mask / label               
                if flag_noisy_label(mask, threshold_percentage=20):
                    print(f"Generated label in PS-scene '{row['ps_product']}' with ID #{row.name} is invalid.")
                    continue

                # Flag 2: Check for NIR displacement in the dataset.
                # The NIR band is considered displaced if its peak TOA reflectance is more than 20 pixels away from the RGB peaks.
                #if flag_nir_displacement(patch, geom, window_transform, patch_size, pixel_size=src.res[0], threshold=20):
                    print(f"Spectral misalignment detected: The NIR band in the PS-scene '{row['ps_product']}' with ID #{row.name} is displaced beyond the acceptable threshold.")
                    continue

                patches.append(patch)
                masks.append(mask)

        #print(f"Shape of patches: {len(patches)}")
        #print(f"Shape of masks: {len(masks)}")
        return patches, masks

    def _generate_mask(self, patch, mask, buffersize_water=3, water_seed_probability=0.90, object_seed_probability=0.2, rw_beta=10):
        # TODO: Choose either Otsu segments or the random walker. Make this clear in the code
        # TODO: Handle image borders better. Right now the Otsu thresholding is shifted by it
        """
        Refines a coarse label mask given a patch using Otsu thresholding on VNIR data and Random Walker.
        """
        from skimage.filters import gaussian, threshold_otsu
        from skimage.morphology import disk, dilation
        from skimage.segmentation import random_walker
        import numpy as np

        out_shape = patch.shape[1:]

        # Create the water mask
        mask_water = dilation(mask, footprint=disk(buffersize_water)) == 0

        # Generate water seeds
        random_seeds = np.random.rand(*out_shape) > water_seed_probability
        seeds_water = random_seeds * mask_water
       
        # Extract VNIR bands
        blue, green, red, nir = patch
        
        # Create a NaN mask for patches on the scene edges
        nan_mask = np.isnan(red) | np.isnan(green) | np.isnan(red) | np.isnan(nir)
        
        # Switch between NDI and band multiplication
        use_ndi = False
        if use_ndi:
            # Best performance was found utilizing red, instead of green (i.e., NDWI)
            # We also invert the NDI so debris objects are bright in the resulting image
            single_channel = -(red - nir) / (red + nir + 1e-10) # Add small constant to prevent division by zero
        else: # Band multiplication
            single_channel = blue * green * red * nir

        # Apply Gaussian blur
        blur = True
        if blur:
            blurred_channel = gaussian(single_channel, sigma=2)
            single_channel = np.where(nan_mask, np.nan, blurred_channel) # Restore NaN's so they dont influence the Otsu threshold

        # Normalize single_channel into RGB range [0, 255]
        vmin, vmax = np.nanpercentile(single_channel, [0, 100])
        
        single_channel = 255 * (single_channel - vmin) / (vmax - vmin)
        single_channel[np.isnan(single_channel)] = np.nan  # Restore NaN's

        # Otsu thresholding
        thresh = threshold_otsu(single_channel[~nan_mask]) # Otsu threshold with NaN exclusion
        otsu_segments = single_channel > thresh
        
        # Mask lines
        mask_lines = (~mask_water)

        # Generate seeds for random walker
        if (otsu_segments * mask_lines).sum() > 0:
            seeds_lines = otsu_segments * mask_lines * (np.random.rand(*out_shape) > object_seed_probability)
        else:
            seeds_lines = mask_lines * (np.random.rand(*out_shape) > object_seed_probability)

        # Random walker algorithm to refine the Otsu segments
        refinement = True
        if refinement:
            # Apply random walker segmentation if seeds are present
            if seeds_lines.sum() > 0:
                markers = seeds_lines * 1 + seeds_water * 2
                labels = random_walker(otsu_segments, markers, beta=rw_beta, mode='bf', return_full_prob=False) == 1
                #labels = random_walker(otsu_segments, markers, beta=rw_beta, mode='cg_j', return_full_prob=False) == 1
                #print("Refined the labels")
            else:
                print("Could not refine sample, returning original mask")
                labels = mask
                #markers = None
        else:
            labels = otsu_segments

        # Visualization
        def quick_viz(single_channel, otsu_thresh, otsu_segments, final_mask, seeds_lines, seeds_water, mask):
            """
            Visualizes NDWI single channel, histogram with Otsu threshold, 
            Otsu segments, final mask, seed points, and overlay the original mask.
            Adjusts for potential y-axis flipping between coordinates and image.
            """
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            # Plot NDWI (single channel)
            axes[0, 0].imshow(single_channel, cmap="gray")
            axes[0, 0].set_title("Inverse NDWI")
            axes[0, 0].axis("off")

            # Plot histogram with Otsu threshold
            axes[0, 1].hist(single_channel[~np.isnan(single_channel)].ravel(), bins=256, color='gray', alpha=0.7)
            axes[0, 1].axvline(otsu_thresh, color='red', linestyle='--', label=f'Otsu Threshold: {otsu_thresh:.2f}')
            #axes[0, 1].axvline(otsu_thresh[0], color='red', linestyle='--', label=f'Otsu Threshold: {otsu_thresh[0]:.2f}')
            #axes[0, 1].axvline(otsu_thresh[1], color='blue', linestyle='--', label=f'Otsu Threshold: {otsu_thresh[1]:.2f}')

            axes[0, 1].set_title("Histogram with Otsu threshold")
            axes[0, 1].set_xlabel("Inverse NDWI")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].legend()

            # Plot Otsu segments
            axes[1, 0].imshow(otsu_segments, cmap="gray")
            axes[1, 0].set_title("Otsu segments")
            axes[1, 0].axis("off")

            # Overlay the seed points (seeds_lines and seeds_water) on Otsu segments
            # Flip the y-coordinates for correct alignment with imshow
            #axes[1, 0].scatter(*np.flip(np.where(seeds_lines), axis=0), color='red', label='Object Seeds', s=5, marker='o', alpha=0.7)
            #axes[1, 0].scatter(*np.flip(np.where(seeds_water), axis=0), color='blue', label='Water Seeds', s=5, marker='x', alpha=0.7)

            # Overlay the original mask (only positive values = 1) on Otsu segments in orange
            positive_mask_coords = np.where(mask == 1)
            axes[1, 0].scatter(*np.flip(positive_mask_coords, axis=0), color='orange', label='Line annotations', s=5, marker='o', alpha=0.7)
            axes[1, 0].legend(loc='upper right')

            # Plot final mask
            axes[1, 1].imshow(final_mask, cmap="gray")
            axes[1, 1].scatter(*np.flip(positive_mask_coords, axis=0), color='orange', label='Line annotations', s=5, marker='o', alpha=0.7)
            axes[1, 1].set_title("Final mask")
            axes[1, 1].axis("off")

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"doc/debug/otsu_mask.png")
            plt.close()
        quick_viz(single_channel, thresh, otsu_segments, labels, seeds_lines, seeds_water, mask)

        return labels

import random
from collections import defaultdict

def split_dataset_by_region(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Splits the dataset by unique region IDs to avoid spatial autocorrelation.

    Args:
        dataset (LitterLinesDataset): The dataset to split.
        train_ratio (float): Fraction of regions for training.
        val_ratio (float): Fraction for validation.
        test_ratio (float): Fraction for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    """
    random.seed(seed)

    # Group scene indices by region ID
    region_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        region_id, _, _ = dataset[idx]
        region_to_indices[region_id].append(idx)

    # Shuffle regions randomly
    all_regions = list(region_to_indices.keys())
    random.shuffle(all_regions)

    # Split regions into train, val, test
    num_regions = len(all_regions)
    train_count = int(num_regions * train_ratio)
    val_count = int(num_regions * val_ratio)

    train_regions = set(all_regions[:train_count])
    val_regions = set(all_regions[train_count:train_count + val_count])
    test_regions = set(all_regions[train_count + val_count:])

    # Create subsets
    train_indices = [idx for region in train_regions for idx in region_to_indices[region]]
    val_indices = [idx for region in val_regions for idx in region_to_indices[region]]
    test_indices = [idx for region in test_regions for idx in region_to_indices[region]]

    return {
        "train": torch.utils.data.Subset(dataset, train_indices),
        "val": torch.utils.data.Subset(dataset, val_indices),
        "test": torch.utils.data.Subset(dataset, test_indices),
    }


def custom_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    # Separate patches and masks
    all_region_ids, all_patches, all_masks = [], [], []

    for ids, patches, masks in batch:
        if len(patches) == 0 or len(masks) == 0:
            continue  # Skip empty returns
        all_patches.extend(patches)
        all_masks.extend(masks)
        all_region_ids.extend(ids)

    # Pad and stack patches and masks
    padded_region_ids = pad_sequence(all_region_ids, batch_first=True, padding_value='ID')
    padded_patches = pad_sequence(all_patches, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(all_masks, batch_first=True, padding_value=0)

    return padded_region_ids, padded_patches, padded_masks


# Usage example
root_dir = "data/PS-LitterLines"
transform = transforms.Compose([transforms.ToTensor()])
dataset = LitterLinesDataset(root_dir=root_dir, transform=transform)
splits = split_dataset_by_region(dataset)

# DataLoader for batching
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=24, shuffle=False, collate_fn=custom_collate_fn) # TODO: Set shuffle to True

# Iterate through the DataLoader
for batch_idx, (region, images, masks) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    print(f" - Region batch shape: {region.shape}")
    print(f" - Image batch shape: {images.shape}")
    print(f" - Mask batch shape: {masks.shape}")
    break  # Stop after one batch for debugging
