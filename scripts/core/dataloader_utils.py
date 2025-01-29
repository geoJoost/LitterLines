import numpy as np
from shapely.geometry import LineString, Point
from shapely.affinity import translate, rotate
from rasterio.features import rasterize
import matplotlib.pyplot as plt

def flag_noisy_label(mask, threshold_percentage):
    """
    Check if the number of True values in a binary mask exceeds a given percentage.

    Parameters:
        mask (np.ndarray): A binary mask with True/False values.
        threshold_percentage (float): The threshold percentage (0-100) to check against.

    Returns:
        bool: True if the percentage of True values exceeds the threshold, otherwise False.
    """
    # Calculate the total number of elements in the label
    total_elements = mask.size

    # Calculate the number of True values in the label array
    true_count = np.sum(mask)

    # Calculate the percentage of True values
    true_percentage = (true_count / total_elements) * 100

    # Check if the percentage exceeds the threshold
    return true_percentage > threshold_percentage

def flag_nir_displacement(patch, geom, window_transform, patch_size, pixel_size, threshold=20):
    def create_perp_line(linestring, distance):
        """
        Creates a single perpendicular line at the centroid of the given LineString.
        
        Parameters:
            linestring (shapely.geometry.LineString): The input line geometry.
            distance (float): The distance to offset the perpendicular line on both sides of the centroid.
        
        Returns:
            shapely.geometry.LineString: A perpendicular LineString centered at the LineString's centroid.
        """
        # Retrieve the centroid of the input line
        centroid = linestring.centroid

        # Offset points to the left and right of the centroid
        point_left = translate(centroid, xoff=-distance)
        point_right = translate(centroid, xoff=distance)

        # Create a perpendicular line using the offset points
        perp_line = LineString([point_left, point_right])

        # Rotate the original line by 90 degrees around the centroid
        #rotated_left = rotate(linestring, 90)#, origin='centroid')
        #rotated_right = rotate(linestring, -90)

        # Translate the rotated line to the left and right by the specified distance
        #point_left = translate(rotated_left.centroid, xoff=-distance yoff=0)
        #point_right = translate(rotated_right.centroid, xoff=distance)

        #perp_line = LineString([point_left, point_right])

        return perp_line
    
    
    def quick_viz(patch, values, perp_mask, pixel_size):
        # Reorganizing the patch from (256, 256, 4) to (4, 256, 256)
        patch = patch.transpose(2, 0, 1)
        
        # Remove NoData areas when patches are on the border of the image
        patch[patch == 0.0] = np.nan
        
        # Reorganizing channels from BGR to RGB
        bgr = patch[:3]
        rgb = bgr[::-1]  # Reverse the order to get RGB from BGR

        # Compute vmin and vmax for each channel individually
        vmin_rgb = np.nanpercentile(rgb, 0, axis=(1, 2))
        vmax_rgb = np.nanpercentile(rgb, 100, axis=(1, 2))
        
        # Normalize each channel individually
        rgb_normalized = np.empty_like(rgb) # Pre-allocate memory
        for i in range(3):  # Loop over the channels (Blue, Green, Red)
            rgb_normalized[i] = np.clip((rgb[i] - vmin_rgb[i]) / (vmax_rgb[i] - vmin_rgb[i]), 0, 1)
        rgb_normalized = rgb_normalized.transpose(1, 2, 0)
        
        # Prepare individual channels for visualization
        channels = ['Blue', 'Green', 'Red', 'NIR']
        colors = ['blue', 'green', 'red', 'purple'] 
        patch_normalized = []
        for i in range(4):
            vmin, vmax = np.nanpercentile(patch[i], [1, 99])
            patch_normalized.append(np.clip((patch[i] - vmin) / (vmax - vmin), 0, 1))

        # Create a figure with 2 rows and 5 columns using gridspec
        fig = plt.figure(figsize=(16, 5))
        spec = fig.add_gridspec(2, 5, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 1])

        # First row: True-colour and individual channel plots
        ax0 = fig.add_subplot(spec[0, 0])  # True-color plot
        ax0.imshow(rgb_normalized)
        ax0.axis('off')
        ax0.set_title("True-colour", fontsize=11)

        # Individual channel plots
        for i in range(4):
            ax = fig.add_subplot(spec[0, 1 + i])  # Individual channel plots
            ax.imshow(patch_normalized[i])
            ax.axhline(patch.shape[1] // 2, color='white', linestyle='--', alpha=0.7)  # Horizontal cross
            ax.axvline(patch.shape[2] // 2, color='white', linestyle='--', alpha=0.7)  # Vertical cross
            ax.set_title(f"{channels[i]} (1%-99%)", fontsize=11)
            ax.axis('off')

        # Second row: Histograms and TOA reflectance along line
        # Plot 1: Histograms of TOA reflectance for all channels (spanning 2 columns)
        ax1 = fig.add_subplot(spec[1, 0:2])  # Histogram spanning 2 columns
        for i, channel in enumerate(channels):
            ax1.hist(
                patch[i].flatten(), bins=256,  # range=(0, 1),
                color=colors[i], alpha=0.6, label=channel
            )
        ax1.set_title("Histogram of TOA reflectance")
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.75)

        # Plot 2: TOA Reflectance Along Line (spanning 3 columns)
        ax2 = fig.add_subplot(spec[1, 2:5])  # TOA Reflectance spanning 3 columns
        num_values = values[0].shape[0]
        distances = np.arange(num_values) * pixel_size  # Distance in meters

        for i, (val, channel) in enumerate(zip(values, channels)):
            ax2.plot(
                distances,  # Use the distance for the X-axis
                val, 
                label=channel, 
                color=colors[i], 
                alpha=0.7, 
                marker='o', 
                markersize=4
            )
        ax2.set_title("Mean TOA reflectance along perpendicular line")
        ax2.set_xlabel("Distance (m)")
        ax2.set_ylabel("Reflectance")
        ax2.legend(loc='upper right')
        ax2.grid(axis='both', alpha=0.5)

        # Plot the perpendicular line if the rasterized line (perp_mask) exists
        if perp_mask is not None:
            # Get the coordinates of the perpendicular line (y, x) from the mask
            line_coords = np.where(perp_mask == 1)  # Returns indices (y, x)
            y_coords, x_coords = line_coords
            
            # Plot the perpendicular line on the true-color plot
            #ax0.plot(x_coords, y_coords, color='yellow', lw=2, label='Line (6px)')
            #ax0.legend(loc='upper left')

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"doc/debug/rgb_patch.png", bbox_inches='tight')
        plt.close()
    
    # Create perpendicular line
    perp_line = create_perp_line(geom, distance=1000)

    # To smooth out the data, create a buffer to make it 5-pixels wide
    buffer_distance = 5 * pixel_size / 2 # Radius is half
    perp_line = perp_line.buffer(buffer_distance)

    # Rasterize the perpendicular line
    # This also removes parts of the line outside the patch
    perp_mask = rasterize(
            [perp_line],
            out_shape=(patch_size, patch_size),
            transform=window_transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
    line_mask = perp_mask == 1

    # Initialize variables
    values, peaks = [], []
    distances = np.arange(patch_size) * pixel_size

    # Iterate over bands
    for i in range(patch.shape[2]):
        # Extract the band values for the line
        band_values = patch[:, :, i][line_mask]
        
        # Reshape into (approximately num_columns, 256) where num_columns is the width and 256 is the length
        # The exact number of rows depends on the raster alignment
        reshaped_values = band_values.reshape(-1, patch_size)  # Ensure correct number of columns
        
        # Average across the pixels (rows) to get 1D array of length corresponding to patch size
        column_averages = np.mean(reshaped_values, axis=0)
        values.append(column_averages)

        # Find the peak value and its location
        max_value = np.max(column_averages)
        max_idx = np.argmax(column_averages)
        max_distance = distances[max_idx]

        # Store the peak info (value, column, distance)
        peaks.append((max_value, max_idx, max_distance))
        print(f"Band {i} - Max TOA reflectance: {max_value} at {max_distance} meters / index position #{max_idx}")
    
    # Visually inspect the TOA reflectance in spatial graphs 
    #quick_viz(patch, values, perp_mask, pixel_size)
    
    # Compare TOA reflectance peaks to determine if the NIR band is displaced by more than the specified threshold.
    for rgb_idx in [0, 1, 2]: # RGB indices
        # Calculate the difference in peak locations between the RGB bands and the NIR band (index=3).
        distance_diff = abs(peaks[rgb_idx][1] - peaks[3][1]) # index=1, is for the max_idx variable
        print(f"Difference in NIR peak compared to RGB peaks is: {distance_diff}px")

        if distance_diff > threshold: # Default 20px
            return True
    return False

def parse_reflectance_coefficients(xml_file):
    import xml.etree.ElementTree as ET
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
    
    return [geom for geom in split_segments.geoms]