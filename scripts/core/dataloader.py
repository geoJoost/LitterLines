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
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Custom imports
from .dataloader_utils import flag_noisy_label, flag_nir_displacement, parse_reflectance_coefficients, split_linestring
# CHANGE: It was "from core.dataloader_utils..."

# ===== Albumentations Transform Pipeline for Training =====
def get_train_augmentation_normal():
    """Returns Albumentations augmentation pipeline with normal (default) intensity."""
    return A.Compose([
        # Orientation transforms
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        
        # Geometric transforms: moderate shift/scale/rotate
        A.Affine(
            scale=(0.9, 1.1),   # up to 10% zoom in/out
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # up to 5% translation
            rotate=(-15, 15),   # rotate by ±15 degrees
            p=0.4
        ),
        
        # Noise & blur: moderate levels
        A.GaussNoise(
            std_range=(0.1, 0.2),      # Range as *fraction* of max value (e.g., 0.1–0.2 × 255 for uint8)
            mean_range=(0.0, 0.0),     # Centered noise, can be changed if needed
            per_channel=True,          # Per-channel noise
            noise_scale_factor=1,      # 1 means pixel-wise, <1 means coarser/faster
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),            # Blur with 3x3 or 5x5 kernel
        
        # Color/Intensity: moderate shifts
        A.RandomBrightnessContrast(
            brightness_limit=0.25, contrast_limit=0.2, p=0.5   # brightness ±25%, contrast ±20%
        ),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),           # gamma 0.85–1.15 (±15%)
        
        # Occlusion: moderate, creates holes that cover up fractions of the patch
        A.CoarseDropout(
            num_holes_range=(1, 4),            # (min, max) number of holes per image
            hole_height_range=(0.05, 0.2),     # fraction of image height (or int in px)
            hole_width_range=(0.05, 0.2),      # fraction of image width (or int in px)
            fill=0,                            # fill value (0 for black, or "random", etc.)
            fill_mask=None,                    # for mask, None = no mask occlusion
            p=0.2
        ),

        ToTensorV2()
    ])

# === Mild augmentations (gentle) ===
def get_train_augmentation_low():
    """Returns Albumentations augmentation pipeline with mild intensity for images & masks."""
    
    return A.Compose([
        # Spatial flips/rotations (same probability as normal, but inherently mild effect)
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),

        # Mild geometric transforms: small shifts, scales, rotations
        A.Affine(
            scale=(0.98, 1.02),                   # zoom in/out (1.0 means no change)
            translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},  # shift ±1%
            rotate=(-5, 5),                       # rotate ±5 degrees
            shear=0,                              # no shear
            p=0.4
        ),
        
        # Mild noise & blur
        A.GaussNoise(
            std_range=(0.02, 0.08),   # 2%–8% noise
            mean_range=(0.0, 0.0),
            per_channel=True,
            noise_scale_factor=1,
            p=0.2,
        ),
        A.GaussianBlur(blur_limit=3, p=0.2),                 # minimal blur (3x3 kernel only)
        
        # Mild color/intensity shifts
        A.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.5   # slight brightness/contrast change (±10%)
        ),
        A.RandomGamma(gamma_limit=(95, 105), p=0.3),          # minor gamma variation (0.95–1.05)
        
        # Occlusion simulation: very mild
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.03, 0.07),
            hole_width_range=(0.03, 0.07),
            fill=0,
            fill_mask=None,
            p=0.1
        ),

        # Convert to tensor
        ToTensorV2()
    ])

# === Intense augmentations (aggressive) ===
def get_train_augmentation_high():
    """Returns Albumentations augmentation pipeline with high (intense) augmentation levels."""
    return A.Compose([
        # Orientation flips/rotations (same probabilities)
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        
        # Geometric transforms: larger shifts/scales/rotations
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            rotate=(-45, 45),
            shear=(-10, 10), # Only in high
            p=0.4,
        ),
        
        # Noise & blur: higher levels
        A.GaussNoise(
            std_range=(0.15, 0.3),    # 15%–30% noise
            mean_range=(-0.05, 0.05),
            per_channel=True,
            noise_scale_factor=1,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(5, 7), p=0.2),           # blur with 5x5 or 7x7 kernel
        
        # Color/Intensity: strong shifts
        A.RandomBrightnessContrast(
            brightness_limit=0.4, contrast_limit=0.3, p=0.5   # brightness ±40%, contrast ±30%
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),          # gamma 0.70–1.30 (wider gamma change)
        
        # Occlusion: more/larger occlusions, simulating heavy cloud/glint
        A.CoarseDropout(
            num_holes_range=(2, 8),
            hole_height_range=(0.10, 0.30),
            hole_width_range=(0.10, 0.30),
            fill=0,
            fill_mask=None,
            p=0.4
        ),

        ToTensorV2()
    ])

# === Very Intense augmentations (very aggressive) ===
def get_train_augmentation_very_high():
    """Albumentations augmentation pipeline with *very high* augmentation levels."""
    return A.Compose([
        # Orientation flips/rotations (same probabilities)
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        
        # Geometric transforms: extreme shifts/scales/rotations/shears
        A.Affine(
            scale=(0.65, 1.35),  # 65% to 135% scaling (very strong zoom in/out)
            translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},  # up to 25% translation
            rotate=(-90, 90),    # full right/left tilt
            shear=(-25, 25),     # more skewing
            p=0.4,
        ),
        
        # Noise & blur: even higher noise/blur
        A.GaussNoise(
            std_range=(0.25, 0.5),      # 25%–50% noise
            mean_range=(-0.08, 0.08),   # broader mean
            per_channel=True,
            noise_scale_factor=1,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(7, 11), p=0.2),  # bigger blur
        
        # Color/Intensity: max perturbations
        A.RandomBrightnessContrast(
            brightness_limit=0.6, contrast_limit=0.5, p=0.5   # up to ±60% brightness, ±50% contrast
        ),
        A.RandomGamma(gamma_limit=(40, 160), p=0.3),  # gamma 0.4–1.6
        
        # Occlusion: max coverage, many & large
        A.CoarseDropout(
            num_holes_range=(5, 16),
            hole_height_range=(0.15, 0.40),
            hole_width_range=(0.15, 0.40),
            fill=0,
            fill_mask=None,
            p=0.4
        ),

        ToTensorV2()
    ])

def print_augmentation_pipeline(augmentation):
    """
    Print only the essential, reproducible parameters for each transform in an Albumentations Compose pipeline.
    """
    print("\n=== TRAIN AUGMENTATION PIPELINE USED ===")
    if hasattr(augmentation, 'transforms'):
        for t in augmentation.transforms:
            params = {}
            # Loop through __init__ arguments, if available in __dict__
            init_keys = t.__init__.__code__.co_varnames[1:t.__init__.__code__.co_argcount]
            for k in init_keys:
                if k in t.__dict__:
                    v = t.__dict__[k]
                    if isinstance(v, (int, float, str, bool, tuple, list, dict)):
                        params[k] = v
            # Always add probability p if present
            if 'p' in t.__dict__:
                params['p'] = t.__dict__['p']
            print(f"  {type(t).__name__}: {params}")
    else:
        print("  [No transforms found in pipeline]")
    print("========================================\n")

def get_val_augmentation():
    """
    Only convert to tensor for validation/test (no augmentation).
    """
    return A.Compose([
        ToTensorV2()
    ])

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

def to_tensor(x, dtype=torch.float):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).type(dtype)
    return x.type(dtype)

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
        scene_folders = glob.glob(os.path.join(self.root_dir, '*/*/'))
        for scene_folder in scene_folders:
            image_paths = glob.glob(os.path.join(scene_folder, "*AnalyticMS*.tif"))
            xml_paths = glob.glob(os.path.join(scene_folder, "*metadata*.xml"))
            if not image_paths or not xml_paths:
                continue  # Skip if no valid images or metadata

            # Extract region ID from folder name
            import re
            parent_of_parent = os.path.basename(os.path.dirname(os.path.dirname(scene_folder)))
            match = re.search(r'\d{8}_(\w+)', parent_of_parent)
            region_id = match.group(1).upper() if match else parent_of_parent.upper()
            for image_path, xml_path in zip(image_paths, xml_paths):
                self.samples.append((image_path, xml_path, region_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns all patches and masks for a given scene, with caching and quality checks."""
        image_path, xml_path, region_id = self.samples[idx]
        scene_name = os.path.basename(image_path).split("_3B")[0] # Extract scene-name

        # Define cache filename. 
        # WARNING: If you change quality checks (e.g. NIR filter, noisy labels), 
        # you must clear this cache so new filtering takes effect!
        cache_dir = os.path.join("data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_patches = os.path.join(cache_dir, f"{region_id}_{scene_name}_patches.pt")
        cache_file_masks = os.path.join(cache_dir, f"{region_id}_{scene_name}_masks.pt")

        if os.path.exists(cache_file_patches):
            print(f"[INFO] LitterLines is pre-processed. Retrieving dataset from cache.")
            patches = torch.load(cache_file_patches)
            masks = torch.load(cache_file_masks)
        else:
            # Parse reflectance coefficients for conversion
            reflectance_coefficients = parse_reflectance_coefficients(xml_path)

            # Filter annotations to current scene
            scene_annotations = self.litter_data[self.litter_data['ps_product'] == scene_name]
            if scene_annotations.empty:
                print(f"[INFO] Skipping {region_id} with ID {scene_name}. No annotations found")
                return torch.empty(0), torch.empty(0), None # Return empty tensors and None for region_id

            print(f"[INFO] Processing {region_id} with ID: {scene_name}")

            # Main patch and mask creation (calls all filtering functions)
            patches, masks = self._create_patches_and_masks(
                image_path, scene_annotations, reflectance_coefficients
            )

            # Save filtered, final patches to cache (guarantees only "good" data is reused)
            patches = torch.tensor(np.array(patches), dtype=torch.float32) # (N, C, 256, 256)
            masks = torch.tensor(np.array(masks), dtype=torch.int8)        # (N, 1, 256, 256)
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
        Create 256x256px patches around annotations and corresponding masks,
        while performing all quality checks (e.g. noisy label, NIR displacement).
        """
        patch_size = self.patch_size
        patches = []
        masks = []

        with rasterio.open(image_path) as src:
            transform = src.transform
            crs = src.crs
            window_size = patch_size * transform[0]
            annotations = annotations.to_crs(crs)

            # Split long linestrings for robust patching
            annotations['temp_geometry'] = annotations['geometry'].apply(
                lambda geom: split_linestring(geom, max_length=window_size)
            )
            annotations = annotations.explode('temp_geometry', ignore_index=True)
            annotations['geometry'] = annotations['temp_geometry']
            assert annotations['geometry'].length.max() <= window_size, "Found segments exceeding window size"

            for _, row in annotations.iterrows():
                geom = row['geometry']
                if not geom.is_valid:
                    print(f"Linestring in PS-scene '{row['ps_product']}' with ID #{row.name} is invalid")
                    continue

                # Center window on linestring centroid
                x, y = geom.centroid.x, geom.centroid.y
                window = rasterio.windows.from_bounds(
                    x - window_size / 2, y - window_size / 2,
                    x + window_size / 2, y + window_size / 2,
                    transform
                )
                patch = src.read(window=window).astype(np.float32)
                if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                    continue

                # Convert to reflectance in-place
                for band_idx in range(patch.shape[0]):
                    coefficient = reflectance_coefficients.get(band_idx + 1)
                    patch[band_idx] *= coefficient

                # Construct patch label
                from shapely.geometry import box
                window_bounds = (
                    x - window_size / 2, y - window_size / 2,
                    x + window_size / 2, y + window_size / 2
                )
                annotations_clip = gpd.clip(annotations, box(*window_bounds))
                if int(row['type']) == 3:
                    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
                else:
                    window_transform = src.window_transform(window)
                    line_raster = rasterize(
                        [(geom, 1) for geom in annotations_clip['geometry']],
                        out_shape=(patch_size, patch_size),
                        transform=window_transform,
                        fill=0,
                        all_touched=True,
                        dtype=np.uint8
                    )
                    mask = self._generate_mask(patch, line_raster).astype(np.int8)

                # ===== Quality Check 1: Noisy Label Filter =====
                if flag_noisy_label(mask, threshold_percentage=20):
                    print(f"Generated label in PS-scene '{row['ps_product']}' with ID #{row.name} is invalid (noisy label >20%).")
                    continue

                # ===== Quality Check 2: NIR Displacement Filter =====
                # The NIR band is considered displaced if its peak reflectance is >20 pixels away from RGB peaks.
                print(f"[DEBUG _create_patches_and_masks] patch.shape = {patch.shape}, patch_size = {patch_size}") # DEBUG

                if flag_nir_displacement(patch, geom, window_transform, patch_size, pixel_size=src.res[0], threshold=20):
                    print(f"Spectral misalignment detected: NIR band in PS-scene '{row['ps_product']}' with ID #{row.name} is displaced beyond threshold. Skipping patch.")
                    continue

                # If patch passes all quality checks, add it
                print(f"PATCH DEBUG: patch.shape = {patch.shape} for {image_path}") # DEBUG
                patches.append(patch)
                masks.append(mask)

        return patches, masks

    def _generate_mask(self, patch, mask, buffersize_water=3, water_seed_probability=0.90, object_seed_probability=0.2, rw_beta=10):
        """
        Refines a coarse label mask given a patch using Otsu thresholding on VNIR data and Random Walker.
        (No structural changes here, left as is.)
        """
        from skimage.filters import threshold_otsu
        from skimage.morphology import disk, dilation
        from skimage.segmentation import random_walker
        import numpy as np

        blue, green, red, nir = patch
        with np.errstate(divide='ignore', invalid='ignore'):
            single_channel = (blue - nir) / (blue + nir) * -1
        valid_pixels = single_channel[~np.isnan(single_channel)]
        thresh = threshold_otsu(valid_pixels)
        q95 = np.percentile(valid_pixels, 95)
        otsu_segments = single_channel > q95

        mask_water = dilation(mask, footprint=disk(buffersize_water)) == 0
        out_shape = patch.shape[1:]
        random_seeds = np.random.rand(*out_shape) > water_seed_probability
        seeds_water = random_seeds * mask_water
        mask_lines = (~mask_water)
        if (otsu_segments * mask_lines).sum() > 0:
            seeds_lines = otsu_segments * mask_lines * (np.random.rand(*out_shape) > object_seed_probability)
        else:
            seeds_lines = mask_lines * (np.random.rand(*out_shape) > object_seed_probability)
        if seeds_lines.sum() > 0:
            markers = seeds_lines * 1 + seeds_water * 2
            labels = random_walker(patch, markers, beta=rw_beta, mode='bf', return_full_prob=False, channel_axis=0) == 1
        else:
            print("Could not refine sample, returning original mask")
            labels = mask
        return labels
    
### NEW CLASS bc of error with the patches in the test_loader.py ###
class LitterLinesPatchDataset(Dataset):
    def __init__(self, litterlines_path, transform=None):
        self.transform = transform
        self.patch_mask_regionid_triples = []  # list of (patch, mask, region_id)
        
        # Load all regions (scenes)
        dataset = LitterLinesDataset(litterlines_path, transform=None)
        for idx in range(len(dataset)):
            patches, masks, region_id = dataset[idx]
            if patches.nelement() == 0:
                continue  # skip empty
            for i in range(patches.shape[0]):
                patch = patches[i].numpy()        # (4, 256, 256) - convert to numpy for Albumentations
                mask = masks[i].numpy()           # (1, 256, 256)
                # Squeeze channel dim if mask has (1, H, W)
                if mask.shape[0] == 1:
                    mask = mask[0]
                # Apply augmentation if provided
                if self.transform:
                    # Albumentations expects channels first for mask, and channels_last for image!
                    # So transpose patch to (H, W, C) for augmentations, then transpose back
                    patch = np.transpose(patch, (1, 2, 0))  # (H, W, C)
                    augmented = self.transform(image=patch, mask=mask)
                    patch = np.transpose(augmented['image'], (2, 0, 1))  # Back to (C, H, W)
                    mask = augmented['mask']
                    # Ensure mask has shape (1, H, W)
                    mask = np.expand_dims(mask, axis=0)
                patch = to_tensor(patch, dtype=torch.float)
                mask = to_tensor(mask, dtype=torch.float)
                self.patch_mask_regionid_triples.append((patch, mask, region_id))

        print(f"[INFO] Loaded {len(self.patch_mask_regionid_triples)} patches in total.")
    
    def __len__(self):
        return len(self.patch_mask_regionid_triples)
    
    def __getitem__(self, idx):
        patch, mask, region_id = self.patch_mask_regionid_triples[idx]
        return patch, mask, region_id
### END OF NEW CLASS ######################

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