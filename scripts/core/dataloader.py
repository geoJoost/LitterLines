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
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import rasterio.windows

class LitterSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, patch_size=256):
        """
        Args:
            root_dir (string): Directory with all the scene folders (e.g., /data/PS-LitterLines).
            transform (callable, optional): Optional transform to be applied on an image.
            patch_size (int): Size of patches to be used for segmentation.
        """
        self.root_dir = root_dir
        self.geojson_path = os.path.join(root_dir, "mlw_annotations_20241211.gpkg")
        self.transform = transform
        self.patch_size = patch_size
        
        # Read the GeoPackage with litter lines into a GeoDataFrame
        self.litter_data = gpd.read_file(self.geojson_path).to_crs(epsg=4326)

        # Get all the available scene folders in the dataset
        self.scene_folders = glob.glob(os.path.join(self.root_dir, '*/*/'))

    def __len__(self):
        return len(self.scene_folders)

    def __getitem__(self, idx):
        # Select a scene folder
        scene_folder = self.scene_folders[idx]
        scene_folder = 'data/PS-LitterLines/20180222_Italy/0e20' #/20201202_075858_57_2264_3B_AnalyticMS_clip.tif'
        
        # Load the corresponding image
        image_path = glob.glob(os.path.join(scene_folder, "*AnalyticMS*.tif"))
                
        if len(image_path) == 0:
            raise FileNotFoundError(f"No matching image files found in {scene_folder}")
        image_path = image_path[0] # TODO: Replace; check if sort is needed
        
        # Parse reflectance coefficients from the associated XML file
        # These are required to convert TOAR into TOA reflectance
        xml_path = glob.glob(os.path.join(scene_folder, "*metadata*.xml"))
        if len(xml_path) == 0:
            raise FileNotFoundError(f"No XML metadata file found in {scene_folder}")
        xml_path = xml_path[0] # TODO: Replace; check if sort is needed
        reflectance_coefficients = self._parse_reflectance_coefficients(xml_path)

        with rasterio.open(image_path) as src:
            full_image = src.read()
            transform = src.transform  # Affine transform for georeferencing
            crs = src.crs

        # Filter annotations for the current scene
        scene_name = os.path.basename(image_path).split("_3B")[0] # Get PS filenames like '20201202_075858_57_2264'
        scene_annotations = self.litter_data[self.litter_data['ps_product'] == scene_name]

        if scene_annotations.empty:
            print(f"No annotations found for '{image_path}'")
            return [], []#None, None
        print(f"Working on images from '{image_path}'")	
     
        # Generate patches and masks
        patches, masks = self._create_patches_and_masks(image_path, transform, crs, scene_annotations, reflectance_coefficients)      
        
        # Apply transforms if any
        if self.transform:
            patches = [self.transform(patch) for patch in patches]
            masks = [self.transform(mask) for mask in masks]

        return patches, masks
    
    def _parse_reflectance_coefficients(self, xml_file):
        """
        Parse the XML file to extract reflectance coefficients for each band.
        """
        root = ET.parse(xml_file).getroot()

        # Define namespace for parsing
        namespaces = {'ps': 'http://schemas.planet.com/ps/v1/planet_product_metadata_geocorrected_level'}

        coefficients = {}
        for band_metadata in root.findall('.//ps:bandSpecificMetadata', namespaces):
            band_number = int(band_metadata.find('ps:bandNumber', namespaces).text)
            reflectance_coefficient = float(
                band_metadata.find('ps:reflectanceCoefficient', namespaces).text
            )
            coefficients[band_number] = reflectance_coefficient

        return coefficients

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
        def split_linestring(linestring, max_length):
            """
            Splits a LineString into smaller LineStrings, each with a maximum length.
            """
            from shapely.ops import split
            from shapely.geometry import MultiPoint

            #print(f"Linestring length: {linestring.length:.2f} m")
            if linestring.length <= max_length:
                return [linestring]
            
            # Adjust max_length by 50 meters to ensure segments do not exceed the window_size
            max_length -= 50
            
            # For two-point lines, mostly the ones drawn for PlanetScope sensor noise (i.e., stripe noise)
            # We densify the lines to add additional points which can be used for splitting the linestring
            from shapely import segmentize
            linestring = segmentize(linestring, max_segment_length=50) # Add vertices every 50m
                       
            # Collect the existing vertices from the LineString
            coords = list(linestring.coords)
            
            # Create splitting points based on along-line distance (path distance)
            from shapely.geometry import Point
            split_points = []
            running_length = 0.0
            for i in range(1, len(coords)):
                # Get points p1 and p2
                p1 = Point(coords[i-1])
                p2 = Point(coords[i])

                # Get the along-line distance between p1 and p2
                # TODO: Perhaps change this to Euclidean distance instead
                segment_length = linestring.project(p2) - linestring.project(p1)
                running_length += segment_length

                if running_length >= max_length:
                    split_points.append(p2)  # Add the point where the segment exceeds max_length
                    running_length = 0  # Reset running length after splitting
            
            # Split the LineString at the calculated points
            split_segments = split(linestring, MultiPoint(split_points))
            
            # If the result is a GeometryCollection, extract individual LineStrings
            split_segments_viz = [geom for geom in split_segments.geoms]
            
            # Visualization of the result in debug console
            fig, ax = plt.subplots(figsize=(10, 5))

            # Plot the original line
            x, y = linestring.xy
            ax.plot(x, y, label="Original LineString", color='gray', linewidth=5)

            # Plot the split segments
            for segment in split_segments_viz:
                x, y = segment.xy
                ax.plot(x, y, label="Split Segment")

            # Set up labels and legend
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title(f"Split LineString (max length = {max_length} meters)")
            ax.legend()
            
            plt.tight_layout()
            plt.savefig("doc/debug/linesplit.png")
            plt.close()

            return [geom for geom in split_segments.geoms]

        # Splint individual linestrings into segments < window_size to make sure all annotations are captured within the labels
        annotations['geometry'] = annotations['geometry'].apply(
            lambda geom: split_linestring(geom, max_length=window_size)
        ) # .explode() turns GDF into into regular DataFrame
        annotations = gpd.GeoDataFrame(annotations.explode(ignore_index=True), geometry='geometry')        
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

                    # Assuming `patch` is already defined
                    # Reorganizing channels from BGR to RGB
                    bgr = patch[:3]
                    rgb = bgr[::-1]

                    # Normalize RGB for visualization using 2%-98% percentiles
                    vmin_rgb, vmax_rgb = np.percentile(rgb, [2, 98])
                    rgb_normalized = np.clip((rgb.transpose(1, 2, 0) - vmin_rgb) / (vmax_rgb - vmin_rgb), 0, 1)

                    # Prepare individual channels for visualization
                    channels = ['Blue', 'Green', 'Red', 'NIR']
                    colors = ['blue', 'green', 'red', 'purple'] 
                    patch_normalized = []
                    for i in range(4):
                        vmin, vmax = np.percentile(patch[i], [2, 98])
                        patch_normalized.append(np.clip((patch[i] - vmin) / (vmax - vmin), 0, 1))

                    # Create a figure with 3 rows and 2 columns
                    fig, axes = plt.subplots(3, 2, figsize=(6, 10))

                    # Plot 1: RGB visualization
                    axes[0, 0].imshow(rgb_normalized)
                    axes[0, 0].axis('off')
                    axes[0, 0].set_title("True-colour (2%-98%)", fontsize=12)

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
                        ax.set_title(f"{channels[i]} (2%-98%)")
                        ax.axis('off')

                    # Adjust layout and save the figure
                    plt.tight_layout()
                    plt.savefig(f"doc/debug/rgb_patch.png", bbox_inches='tight')
                    plt.close()
                quick_viz(patch)

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

                patches.append(patch)
                masks.append(mask)

        print(f"Shape of patches: {len(patches)}")
        print(f"Shape of masks: {len(masks)}")
        return patches, masks

    def _generate_mask(self, patch, mask, buffersize_water=3, water_seed_probability=0.90, object_seed_probability=0.2, rw_beta=10):
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

        print(f"Number of NaN values: {np.isnan(patch).sum()}")
        
        # Extract VNIR bands
        blue, green, red, nir = patch
        
        # Create a NaN mask for patches on the scene edges
        nan_mask = np.isnan(red) | np.isnan(green) | np.isnan(red) | np.isnan(nir)
        
        # Switch between NDI and band multiplication
        use_ndi = True
        if use_ndi:
            # Best performance was found utilizing red, instead of green (i.e., NDWI)
            # We also invert the NDI so debris objects are bright in the resulting image
            single_channel = -(red - nir) / (red + nir + 1e-10) # Add small constant to prevent division by zero
        else: # Band multiplication
            single_channel = red * nir

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
        
        from skimage.filters import threshold_multiotsu
        #thresh = threshold_multiotsu(single_channel, 3)
        
        # Select the brightest region (highest single_channel values)
        #brightest_threshold = thresh[-1]
        #otsu_segments = single_channel > brightest_threshold

        """
        ### check that suitable thresh has been found. if not, repeat Otsu
        medi = np.nanmedian(single_channel)
        perc = 5
        factor = 2
        v = True

        while thresh < medi + (medi / factor):
            if v:
                print ("   --> Repeating Otsu ...")
            blur_scaled_min_cut = single_channel
            blur_scaled_min_cut = np.where(blur_scaled_min_cut < np.percentile(blur_scaled_min_cut, perc), np.percentile(blur_scaled_min_cut, perc), blur_scaled_min_cut).astype(np.uint8)
            # plt.figure()
            # plt.imshow(blur_scaled_min_cut)
            # plt.show()
            thresh = threshold_otsu(blur_scaled_min_cut)
            otsu_segments = blur_scaled_min_cut > thresh
            perc += 5
            if perc > 95:
                if factor < 5:
                    factor+=1
                    perc = 5
                else:
                    break
        if v:
            print("Thresh:", thresh)
        """
        # Mask out NaN areas
        #otsu_segments = np.where(np.isnan(single_channel), 0, otsu_segments)

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

def custom_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    # Separate patches and masks
    all_patches, all_masks = [], []

    for patches, masks in batch:
        if len(patches) == 0 or len(masks) == 0:
            continue  # Skip empty returns
        all_patches.extend(patches)
        all_masks.extend(masks)

    # Pad and stack patches and masks
    padded_patches = pad_sequence(all_patches, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(all_masks, batch_first=True, padding_value=0)

    return padded_patches, padded_masks

# Usage example
root_dir = "data/PS-LitterLines"
transform = transforms.Compose([transforms.ToTensor()])
dataset = LitterSegmentationDataset(root_dir=root_dir, transform=transform)

# DataLoader for batching
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=custom_collate_fn) # TODO: Set shuffle to True

# Iterate through the DataLoader
for batch_idx, (images, masks) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    print(f" - Image batch shape: {images.shape}")
    print(f" - Mask batch shape: {masks.shape}")
    break  # Stop after one batch for debugging
