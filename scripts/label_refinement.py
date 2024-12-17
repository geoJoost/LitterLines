import os
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from skimage.morphology import dilation, disk
from skimage.filters import threshold_otsu
from skimage.segmentation import random_walker
from shapely.geometry import box



def visualize_debugging_steps(single_channel, otsu_segments, mask_lines, seeds_lines, seeds_water, labels):
    """
    Visualize key steps in the mask refinement process.
    
    Args:
        single_channel (np.ndarray): The scaled NIR image used for Otsu thresholding.
        otsu_segments (np.ndarray): Binary mask from Otsu thresholding.
        mask_lines (np.ndarray): The rasterized line mask.
        seeds_lines (np.ndarray): Seeds generated from Otsu and line masks.
        seeds_water (np.ndarray): Seeds representing water/background.
        labels (np.ndarray): Final refined mask after random walker segmentation.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax = axes.ravel()

    # Step #1: Scaled single channel
    ax[0].imshow(single_channel, cmap='gray')
    ax[0].set_title("#1: Visualization")
    ax[0].axis("off")

    # Step #2: Otsu segments
    ax[1].imshow(otsu_segments, cmap='gray')
    ax[1].set_title("#2: Otsu thresholded segments")
    ax[1].axis("off")

    # Step #3: Mask lines
    ax[2].imshow(mask_lines, cmap='gray')
    ax[2].set_title("#3: Rasterized Line Mask")
    ax[2].axis("off")

    # Step #4: Seeds from lines
    ax[3].imshow(seeds_lines, cmap='gray')
    ax[3].set_title("#4: MLW Seeds")
    ax[3].axis("off")

    # Step #5: Seeds from water
    ax[4].imshow(seeds_water, cmap='gray')
    ax[4].set_title("#5: Water Seeds")
    ax[4].axis("off")

    # Step #6: Final refined mask
    ax[5].imshow(labels, cmap='gray')
    ax[5].set_title("#6: Final Refined Mask")
    ax[5].axis("off")

    plt.tight_layout()
    #plt.show()
    plt.savefig('doc/debug/planetscope_otsu.png')
    print('...')

def clip_around_line(image, line_raster, transform, buffer=50):
    """
    Clips the raster and line mask to a smaller area around the line for debugging.
    
    Args:
        image (np.ndarray): The full multispectral image array.
        line_raster (np.ndarray): The rasterized line array.
        transform (Affine): The affine transformation of the raster.
        buffer (int): Pixel buffer around the line for clipping.
    
    Returns:
        (np.ndarray, np.ndarray, Affine): Clipped image, clipped line raster, and new transform.
    """
    # Find the bounding box of the line
    rows, cols = np.where(line_raster > 0)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("No lines found in the rasterized data.")

    min_row, max_row = max(0, rows.min() - buffer), min(line_raster.shape[0], rows.max() + buffer)
    min_col, max_col = max(0, cols.min() - buffer), min(line_raster.shape[1], cols.max() + buffer)

    # Clip the image and line raster
    clipped_image = image[:, min_row:max_row, min_col:max_col]
    clipped_line_raster = line_raster[min_row:max_row, min_col:max_col]

    # Update the transform
    new_transform = transform * rasterio.Affine.translation(min_col, min_row)

    return clipped_image, clipped_line_raster, new_transform

def refine_masks(image, mask, buffersize_water=3, water_seed_probability=0.90, object_seed_probability=0.2, rw_beta=10, return_all=False):
    """
    Refines a coarse label mask given an image using Otsu thresholding on VNIR data and Random Walker.
    """
    out_shape = image.shape[1:]

    # Create the water mask
    mask_lines = mask
    mask_water = dilation(mask_lines, footprint=disk(buffersize_water)) == 0

    # Generate water seeds
    random_seeds = np.random.rand(*out_shape) > water_seed_probability
    seeds_water = random_seeds * mask_water

    # Stack VNIR channels 
    blue = image[0] * 3.028566418433337e-05
    green = image[1] * 3.284715506911182e-05
    red = image[2] * 3.953608604523683e-05
    nir = image[3] * 6.257568332099102e-05

    # Calculate visualization for marine litter windrows
    #single_channel = blue * green * red

    # Calculate NDWI
    single_channel = np.where(
        (green + nir) != 0.0, # Avoid division by zero
        -(green - nir) / (green + nir + 1e-10),  # Small constant to avoid division by zero
        np.nan # Set NDWI to NaN where invalid
    )

    # Apply Gaussian blur to reduce noise
    from skimage.filters import gaussian
    # Apply Gaussian blur to reduce noise while handling NaN
    # Replace NaN with a neutral value before applying the blur, then restore NaN
    nan_mask = np.isnan(single_channel)
    single_channel_no_nan = np.where(nan_mask, 0, single_channel)  # Replace NaN with 0 temporarily
    blurred_channel = gaussian(single_channel_no_nan, sigma=2)  # Apply Gaussian blur
    single_channel = np.where(nan_mask, np.nan, blurred_channel)
    
    # Normalize the single_channel using the 2-98 percentile range, while keeping NaN
    vmin, vmax = np.nanpercentile(single_channel, [2, 98])
    single_channel = np.where(
        ~np.isnan(single_channel),  # Only normalize non-NaN values
        255 * np.clip((single_channel - vmin) / (vmax - vmin), 0, 1),
        np.nan  # Preserve NaN
    )
    
    # Otsu thresholding
    thresh = threshold_otsu(single_channel[~nan_mask])
    otsu_segments = single_channel > thresh
    print(f"Otsu threshold: {thresh}")

    # Mask out the NaN areas (i.e., land and NoData)
    otsu_segments = np.where(np.isnan(single_channel), 0, otsu_segments)

    # Visualization
    def quick_viz(single_channel, otsu_thresh):
        import matplotlib.pyplot as plt
        """
        Visualize the single channel image alongside its histogram with a given Otsu threshold.
        
        Args:
            single_channel (np.ndarray): The image to visualize.
            otsu_thresh (float): The Otsu threshold value to display on the histogram.
        """
        # Create figure and axes
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Show the single_channel image
        ax[0].imshow(single_channel, cmap="gray")
        ax[0].set_title("NDWI")
        ax[0].axis("off")

        # Show the histogram
        ax[1].hist(single_channel[~np.isnan(single_channel)].ravel(), bins=256, color='gray', alpha=0.7)
        ax[1].axvline(otsu_thresh, color='red', linestyle='--', label=f'Otsu Threshold: {otsu_thresh:.2f}')
        ax[1].set_title("Histogram with Otsu threshold")
        ax[1].set_xlabel("TOA reflectance")
        ax[1].set_ylabel("Frequency")
        ax[1].legend()

        # Display the plots
        plt.tight_layout()
        plt.savefig("doc/debug/otsu.png")
        plt.close()
    quick_viz(single_channel, thresh)

    mask_lines = (~mask_water)
    
    # Generate seeds for random walker
    #object_seed_probability = 0.05
    if (otsu_segments * mask_lines).sum() > 0:
        seeds_lines = otsu_segments * mask_lines * (np.random.rand(*out_shape) > object_seed_probability)
    else:
        seeds_lines = mask_lines * (np.random.rand(*out_shape) > object_seed_probability)

    # Apply random walker segmentation if seeds are present
    #if seeds_lines.sum() > 0:
    #    markers = seeds_lines * 1 + seeds_water * 2
    #    #labels = random_walker(otsu_segments, markers, beta=rw_beta, mode='bf', return_full_prob=False) == 1
    #    labels = random_walker(otsu_segments, markers, beta=rw_beta, mode='cg_j', return_full_prob=False) == 1
    #    print("Refined the labels")
    #else:
    #    print("Could not refine sample, returning original mask")
    #    labels = mask
    #    markers = None
    # Function to apply on chunks
    def process_chunk(chunk, **kwargs):
        """Applies random walker to a chunk of data."""
        # Split the Otsu segments and markers
        otsu = chunk[:, :, 0]
        markers = chunk[:, :, 1]

        if np.any(markers > 0):
            labels_chunk = random_walker(otsu, markers, **kwargs) == 1
        else:
            labels_chunk = np.zeros_like(otsu, dtype=bool)
        return labels_chunk

    # Apply random walker in parallel
    from skimage.util import apply_parallel
    if seeds_lines.sum() > 0:
        # First calculate the proper markers
        markers = seeds_lines * 1 + seeds_water * 2

        # To guarantee proper chunking and parallelization, we attach it as additional dimension
        stack = np.dstack((otsu_segments, markers))

        labels = apply_parallel(
            process_chunk,
            stack,#otsu_segments,
            chunks=(1024, 1024),  # Define chunk size
            depth=10,  # Depth for overlap
            dtype=np.uint8, 
            mode='nearest',  # Instead of 'reflect' since Otsu returns binary values
            #extra_arguments=(seeds_lines, seeds_water),
            extra_keywords={'beta': rw_beta, 'mode': 'bf'},
            compute= True, # Return numpy array
            channel_axis = stack.shape[2]
        )
        print("Refined the labels")
    else:
        print("Could not refine sample, returning original mask")
        labels = mask

    # Debugging Visualization
    #visualize_debugging_steps(single_channel, otsu_segments, mask_lines, seeds_lines, seeds_water, labels)

    #if return_all:
    #    return labels, otsu_segments, markers, single_channel, mask_lines
    #else:
    #    return labels, single_channel, otsu_segments
    return labels, single_channel, otsu_segments

def process_scene(tif_path, sceneid, gkpg_path, buffersize_water=3, water_seed_probability=0.90, object_seed_probability=0.2, rw_beta=10, return_all=False, land_shapefile=None):
    """
    Processes a scene by refining masks based on VNIR data and rasterized geometries.
    
    Args:
        tif_path (str): Path to the multispectral .tif file.
        sceneid (str): PlanetScope scene ID to filter rows in the GeoPackage.
        gkpg_path (str): Path to the GeoPackage containing vector data.
        buffersize_water (int): Buffer size for water dilation.
        water_seed_probability (float): Probability threshold for water seeds.
        object_seed_probability (float): Probability threshold for object seeds.
        rw_beta (float): Beta parameter for the random walker algorithm.
        return_all (bool): If True, returns additional outputs for debugging.
    
    Returns:
        Saves refined raster mask as a GeoTIFF and returns the refined mask.
    """

    # Load the multispectral image
    with rasterio.open(tif_path) as src:
        image = src.read()  # Read all bands
        transform = src.transform
        crs = src.crs  # Coordinate reference system
        image_shape = (src.height, src.width)
        image_bounds = src.bounds

    # Load the GeoPackage and filter by sceneid
    gdf = gpd.read_file(gkpg_path).to_crs(crs)
    gdf_filtered = gdf[gdf['ps_product'] == sceneid]

    # Rasterize the line geometry
    line_raster = rasterize(
        [(geom, 1) for geom in gdf_filtered.geometry],
        out_shape=image_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )

    # Load and rasterize the land mask
    if land_shapefile:
        # Load the global land shapefile from OpenStreetMap (EPSG:4326)
        land_gdf = gpd.read_file(land_shapefile)

        # Create a bounding box from the raster bounds 
        # Reproject into EPSG:4326 to prevent shapely errors with NaN/Inf numbers
        bbox = gpd.GeoDataFrame(geometry=[box(*image_bounds)], crs=crs).to_crs(land_gdf.crs)

        # Clip the land geometries to the bounding box
        # Reproject back into local CRS (e.g, EPSG:32635)
        land_gdf_clipped = gpd.overlay(land_gdf, bbox, how="intersection").to_crs(crs)

        # Rasterize the land shapefile to match the image's resolution and transform
        land_raster = rasterize(
            [(geom, 1) for geom in land_gdf_clipped['geometry']],
            out_shape=image.shape[1:],
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
        # Mask out land areas by setting NaN where land is present
        land_mask = land_raster == 1  # Land is 1, background is 0

        # Set image values covered by the land mask to NaN
        image = np.where(land_mask, np.nan, image)

    # Refine the mask using the clipped image and rasterized line
    refined_mask, single_channel, otsu_segments = refine_masks(
        image=image, #clipped_image,
        mask=line_raster, #clipped_line_raster,
        buffersize_water=buffersize_water,
        water_seed_probability=water_seed_probability,
        object_seed_probability=object_seed_probability,
        rw_beta=rw_beta,
        return_all=return_all,
    )

    # Handle output directory and filename
    output_dir = "data/test"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sceneid}_refined.tif")

    print("Saving rasters")
    
    # Single-channel visualization
    otsu_rgb_path = os.path.join(output_dir, f"{sceneid}_rgb.tif")
    with rasterio.open(
        otsu_rgb_path,
        "w",
        driver="GTiff",
        height=single_channel.shape[0],
        width=single_channel.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(single_channel, 1)
    print(f"Saved Otsu input to {otsu_rgb_path}")
    
    # Otsu segments
    otsu_path = os.path.join(output_dir, f"{sceneid}_otsu.tif")
    with rasterio.open(
        otsu_path,
        "w",
        driver="GTiff",
        height=otsu_segments.shape[0],
        width=otsu_segments.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(otsu_segments, 1)
    print(f"Saved Otsu input to {otsu_rgb_path}")

    # Save clipped line raster
    clipped_lines_path = os.path.join(output_dir, f"{sceneid}_clipped_lines.tif")
    with rasterio.open(
        clipped_lines_path,
        "w",
        driver="GTiff",
        height=line_raster.shape[0],
        width=line_raster.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(line_raster, 1)

    # Save refined mask (if applicable)
    if isinstance(refined_mask, tuple):
        refined_labels = refined_mask[0]
    else:
        refined_labels = refined_mask

    refined_clipped_path = os.path.join(output_dir, f"{sceneid}_refined_clipped.tif")
    with rasterio.open(
        refined_clipped_path,
        "w",
        driver="GTiff",
        height=refined_labels.shape[0],
        width=refined_labels.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(refined_labels.astype(rasterio.uint8), 1)

    print(f"Clipped line raster saved to {clipped_lines_path}")
    print(f"Refined clipped mask saved to {refined_clipped_path}")
    return refined_mask

process_scene(
    tif_path='data/PS-LitterLines/20210430_Greece/242d/20210430_082512_56_242d_3B_AnalyticMS.tif',
    sceneid="20210430_082512_56_242d", 
    gkpg_path="/misc/rs1/jvandalen/TRACEv2/data/annotations/mlw_annotations.gpkg",
    buffersize_water=5,
    water_seed_probability=0.85,
    object_seed_probability=0.25,
    rw_beta=12,
    return_all=False,
    land_shapefile="data/raw/land-polygons-split-4326/land_polygons.shp"
)
