import os
import glob
import random
import numpy as np
import torch
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from rasterio.features import rasterize
import rasterio.windows
from skimage.restoration import denoise_bilateral


# Custom imports
from core.dataloader_utils import flag_noisy_label, flag_nir_displacement, parse_reflectance_coefficients, split_linestring

class LitterLinesDataset(Dataset):
    def __init__(self, root_dir, transform=None, patch_size=256):
        """
        Args:
            root_dir (string): Directory with all the scene folders (e.g., /data/PS-LitterLines).
            transform (callable, optional): Optional transform to be applied on an image.
            patch_size (int): Size of patches to be used for segmentation.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = patch_size

        # Read the GeoPackage with litter lines into a GeoDataFrame
        self.geojson_path = os.path.join(root_dir, "mlw_annotations.gpkg")
        self.litter_data = gpd.read_file(self.geojson_path).to_crs(epsg=4326)

        # Store a list of all image-metadata pairs
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Precomputes file paths and their corresponding region IDs."""
        scene_folders = glob.glob(os.path.join(self.root_dir, '*/*/'))#[:4] #TODO: Remove. Kept for debugging
        #scene_folders = ['data/LitterLines/20171009_Kikaki/103a/', 'data/LitterLines/20171009_Kikaki/103c/', 'data/LitterLines/20171009_Kikaki/0f25/']
        
        for scene_folder in scene_folders:
            # Get image paths and metadata XMLs
            image_paths = glob.glob(os.path.join(scene_folder, "*AnalyticMS*.tif"))
            xml_paths = glob.glob(os.path.join(scene_folder, "*metadata*.xml"))

            if not image_paths or not xml_paths:
                continue  # Skip if no valid images or metadata

            # Extract region ID from path
            region_id = os.path.normpath(scene_folder).split(os.sep)[2]  # Example: '20180824_Syria'
            
            for image_path, xml_path in zip(image_paths, xml_paths):
                self.samples.append((image_path, xml_path, region_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """ Returns all patches and masks for a given scene, with caching"""
        image_path, xml_path, region_id = self.samples[idx]
        scene_name = os.path.basename(image_path).split("_3B")[0] # Extract scene-name
        
        # Define cache filename
        cache_dir = os.path.join("data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_patches = os.path.join(cache_dir, f"{region_id}_{scene_name}_patches.pt")
        cache_file_masks = os.path.join(cache_dir, f"{region_id}_{scene_name}_masks.pt")
        
        if os.path.exists(cache_file_patches):
            print(f"[INFO] LitterLines is pre-processed. Retrieving dataset from cache.")
            patches = torch.load(cache_file_patches)
            masks = torch.load(cache_file_masks)
        else:
            # Get reflectance coefficients for TOAR to TOA
            reflectance_coefficients = parse_reflectance_coefficients(xml_path)
            
            # Find associated MLW-annotations
            scene_annotations = self.litter_data[self.litter_data['ps_product'] == scene_name]
            if scene_annotations.empty:
                print(f"[INFO] Skipping {region_id} with ID {scene_name}. No annotations found")
                return torch.empty(0), torch.empty(0), None # Return empty tensors and None for region_id for filtering the data
            print(f"[INFO] Processing {region_id} with ID: {scene_name}")
                
            # Generate patches and masks
            patches, masks = self._create_patches_and_masks(image_path, scene_annotations, reflectance_coefficients)
            
            # TODO: Remove at some point. Kept for debugging
            #total_true_pixels = sum(arr.sum() for arr in masks)
            #total_pixels = sum(arr.size for arr in masks)
            #print(f"[INFO] {region_id} with ID {scene_name} has {total_true_pixels} pixels of MLWs, out of {total_pixels}")

            # Convert to tensor
            patches = torch.tensor(np.array(patches), dtype=torch.float32) # (N, C, 256, 256)
            masks = torch.tensor(np.array(masks), dtype=torch.int8) # (N, 1, 256, 256)
            
            torch.save(patches, cache_file_patches)
            torch.save(masks, cache_file_masks)

        # Apply transform if available
        if self.transform:
            print(f"[INFO] Applying transform to images.")
            patches = torch.stack([self.transform(patch) for patch in patches])
            masks = torch.stack([self.transform(mask) for mask in masks])
        
        return patches, masks, region_id

    def _create_patches_and_masks(self, image_path, annotations, reflectance_coefficients):
        """
        Create 256x256px patches around annotations and corresponding masks.
        """
        patch_size = self.patch_size
        patches = []
        masks = []

        # Open the image file to get metadata
        with rasterio.open(image_path) as src:
            transform = src.transform
            crs = src.crs

            # Window size in geographic coordinates
            window_size = patch_size * transform[0]  # Transform[0] gets pixel size in meters
            annotations = annotations.to_crs(crs) # EPSG:4326 into local CRS

            # To prevent loss of annotations, split linestrings into smaller segments when necessary
            annotations['temp_geometry'] = annotations['geometry'].apply(
                lambda geom: split_linestring(geom, max_length=window_size)
            )
            annotations = annotations.explode('temp_geometry', ignore_index=True)
            annotations['geometry'] = annotations['temp_geometry']

            assert annotations['geometry'].length.max() <= window_size, "Found segments exceeding window size"

            # Iterate through each linestring in the annotations
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
                    #print('Plotted figure')
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
                    mask = self._generate_mask(patch, line_raster).astype(np.int8)
                
                # Re-order from CHW to HWC for self.transform()
                #patch = patch.transpose(1, 2, 0)

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

        """
        Refines a coarse label mask given a patch using Otsu thresholding on VNIR data and Random Walker.
        """
        from skimage.filters import threshold_otsu
        from skimage.morphology import disk, dilation
        from skimage.segmentation import random_walker
        import numpy as np
                
        # Slight improvements in thresholded results, but slows down processing significantly
        # for band_idx in range(patch.shape[0]):
        #    valid_mask = patch[band_idx] > 0  # Identify valid (non-zero) pixels
        #    filtered_band = denoise_bilateral(patch[band_idx], sigma_color=None, sigma_spatial=5)
        #    patch[band_idx][valid_mask] = filtered_band[valid_mask]  # Apply only to valid pixels

        # Extract VNIR bands
        blue, green, red, nir = patch

        # Best performance was found utilizing using the Rotation-Absorption Index
        # We also invert the NDI so debris objects are bright in the resulting image
        with np.errstate(divide='ignore', invalid='ignore'): # Ignore NoData (=0.0) values outside the scene      
            single_channel = (blue - nir) / (blue + nir) *-1
           
        # Remove NoData values (np.nan) for values outside the scene
        valid_pixels = single_channel[~np.isnan(single_channel)]  # Extract only valid pixels
        
        # Otsu threshold
        thresh = threshold_otsu(valid_pixels)  # Otsu threshold
        q95 = np.percentile(valid_pixels, 95) # IQR95 threshold
        otsu_segments = single_channel > q95
        
        # Create the water mask
        mask_water = dilation(mask, footprint=disk(buffersize_water)) == 0

        # Generate water seeds
        out_shape = patch.shape[1:] # (256, 256)
        random_seeds = np.random.rand(*out_shape) > water_seed_probability
        seeds_water = random_seeds * mask_water
        
        # Mask lines
        mask_lines = (~mask_water)

        # Generate seeds for random walker
        if (otsu_segments * mask_lines).sum() > 0:
            seeds_lines = otsu_segments * mask_lines * (np.random.rand(*out_shape) > object_seed_probability)
        else:
            seeds_lines = mask_lines * (np.random.rand(*out_shape) > object_seed_probability)

        # Apply random walker segmentation if seeds are present
        if seeds_lines.sum() > 0:
            markers = seeds_lines * 1 + seeds_water * 2
            labels = random_walker(patch, markers, beta=rw_beta, mode='bf', return_full_prob=False, channel_axis=0) == 1
            #print("Refined the labels")
        else:
            print("Could not refine sample, returning original mask")
            labels = mask
            #markers = None

        # Visualization
        def quick_viz(single_channel, valid_pixels_rescaled, otsu_thresh, q95, otsu_segments, final_mask, seeds_lines, seeds_water, mask):
            """
            Visualizes NDWI single channel, histogram with Otsu threshold, 
            Otsu segments, final mask, seed points, and overlay the original mask.
            Adjusts for potential y-axis flipping between coordinates and image.
            """
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            # Plot single channel visualization
            axes[0, 0].imshow(single_channel, cmap="gray")
            axes[0, 0].set_title("Inverse Rotation-Absorption Index")
            axes[0, 0].axis("off")

            # Plot histogram with Otsu threshold
            axes[0, 1].hist(valid_pixels_rescaled, bins=256, color='gray', alpha=0.7)
            axes[0, 1].axvline(otsu_thresh, color='red', linestyle='--', label=f'Otsu Threshold: {otsu_thresh:.2f}')
            axes[0, 1].axvline(q95, color='blue', linestyle='--', label=f'IQR95 Threshold: {q95:.2f}')


            axes[0, 1].set_title("Histogram with thresholds")
            axes[0, 1].set_xlabel("Inverse RAI")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].legend()

            # Plot Otsu segments
            axes[1, 0].imshow(otsu_segments, cmap="gray")
            axes[1, 0].set_title("Segments")
            axes[1, 0].axis("off")

            # Overlay the seed points (seeds_lines and seeds_water) on Otsu segments
            # Flip the y-coordinates for correct alignment with imshow
            axes[1, 0].scatter(*np.flip(np.where(seeds_lines), axis=0), color='red', label='Object Seeds', s=5, marker='o', alpha=0.7)
            axes[1, 0].scatter(*np.flip(np.where(seeds_water), axis=0), color='blue', label='Water Seeds', s=5, marker='x', alpha=0.7)

            # Overlay the original mask (only positive values = 1) on Otsu segments in orange
            positive_mask_coords = np.where(mask == 1)
            axes[1, 0].scatter(*np.flip(positive_mask_coords, axis=0), color='orange', label='Line annotations', s=5, marker='o', alpha=0.7)
            axes[1, 0].legend(loc='upper right')

            # Plot final mask
            axes[1, 1].imshow(final_mask, cmap="gray")
            axes[1, 1].scatter(*np.flip(positive_mask_coords, axis=0), color='orange', label='Line annotations', s=5, marker='o', alpha=0.7)
            axes[1, 1].set_title("Final label")
            axes[1, 1].axis("off")

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"doc/debug/otsu_mask.png")
            plt.close()
        #quick_viz(single_channel, valid_pixels, thresh, q95, otsu_segments, labels, seeds_lines, seeds_water, mask)

        return labels

class DatasetManager:
    """
    Manages dataset splitting, loading, and preprocessing for the LitterLines dataset.
    """

    def __init__(self, dataset, train_ratio=0.7, val_ratio=0.2, seed=42, batch_size=8):
        """
        Initializes the DatasetManager with dataset and split configurations.

        Args:
            dataset (LitterLinesDataset): The dataset to split.
            train_ratio (float): Fraction of regions for training.
            val_ratio (float): Fraction for validation.
            seed (int): Random seed for reproducibility.
            batch_size (int): Batch size for DataLoaders.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.splits = self.split_dataset_by_region(dataset, train_ratio, val_ratio, seed)

    def split_dataset_by_region(self, dataset, train_ratio, val_ratio, seed):
        """
        Splits the dataset by unique region IDs to avoid spatial autocorrelation.

        Args:
            dataset (LitterLinesDataset): The dataset to split.
            train_ratio (float): Fraction of regions for training.
            val_ratio (float): Fraction for validation.
            seed (int): Random seed for reproducibility.

        Returns:
            dict: {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        """
        random.seed(seed)

        # Group scene indices by region ID
        region_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, _, region_id = dataset[idx]
            if region_id is None:  # Skip empty samples
                continue
            region_to_indices[region_id].append(idx)
            
        # Debug: Print total number of regions detected
        print(f"Total unique regions detected: {len(region_to_indices)}")
            
        # Ensure Kikaki dataset is used exclusively for testing
        test_indices = region_to_indices.get('20171009_Kikaki', [])
        if '20171009_Kikaki' in region_to_indices:
            del region_to_indices['20171009_Kikaki']

        # Shuffle regions randomly
        all_regions = list(region_to_indices.keys())
        random.shuffle(all_regions)

        # Split regions into train, val, test
        num_regions = len(all_regions)
        train_count = round(num_regions * train_ratio)
        val_count = num_regions - train_count # Ensure all regions are used

        train_regions = set(all_regions[:train_count])
        val_regions = set(all_regions[train_count:])
        #test_regions = set(all_regions[train_count + val_count:])

        # Create subsets
        train_indices = [idx for region in train_regions for idx in region_to_indices[region]]
        val_indices = [idx for region in val_regions for idx in region_to_indices[region]]
        
        return {
            "train": Subset(dataset, train_indices),
            "val": Subset(dataset, val_indices),
            "test": Subset(dataset, test_indices),
        }

    @staticmethod
    def custom_collate_fn(batch):
        """
        Custom collate function to handle variable-sized batches.

        Args:
            batch (list): List of (patches, masks, region_id) tuples.

        Returns:
            tuple: Stacked patches, masks, and repeated region IDs.
        """
        all_patches, all_masks, all_region_ids = [], [], []

        for patches, masks, region_id in batch:
            if len(patches) == 0 or len(masks) == 0:
                continue  # Skip empty returns
            all_patches.extend(patches)
            all_masks.extend(masks)

            # Repeat the region_id to match the number of patches/masks
            all_region_ids.extend([region_id] * len(patches))

        # Stack patches and masks into batches
        stacked_patches = torch.stack(all_patches)  # (N, C, H, W)
        stacked_masks = torch.stack(all_masks)  # (N, H, W)

        return stacked_patches, stacked_masks, all_region_ids

    def get_dataloader(self, split, shuffle=True):
        """
        Returns a DataLoader for a given split.

        Args:
            split (str): One of "train", "val", or "test".
            shuffle (bool): Whether to shuffle data (default: True for train).

        Returns:
            DataLoader: The PyTorch DataLoader for the given split.
        """
        if split not in self.splits:
            raise ValueError(f"Invalid split name: {split}. Choose from 'train', 'val', or 'test'.")

        return DataLoader(self.splits[split], batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.custom_collate_fn)