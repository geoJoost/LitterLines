import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.affinity import rotate


def visualize_debug(linestring, perp_lines):
    """
    Visualize the input LineString and all perpendicular lines.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the original LineString
    x, y = linestring.xy
    ax.plot(x, y, label="Original LineString", color="blue", linewidth=2)

    # Plot all perpendicular lines
    for idx, perp_line in enumerate(perp_lines):
        x_perp, y_perp = perp_line.xy
        ax.plot(x_perp, y_perp, label=f"Perpendicular Line {idx + 1}", color="orange", linewidth=1)

    # Add legend and labels
    ax.legend()
    ax.set_title("Debug Visualization of Perpendicular Lines")
    ax.set_xlabel("X Coordinates")
    ax.set_ylabel("Y Coordinates")
    ax.grid(True)

    # Save the figure
    #plt.show()
    plt.savefig("doc/debug/perp_line.png")
    plt.close()
    print("Saved plot")

def process_lines_extract_spectral(fid_list, tif_filepath):
    """
    Processes input linestrings to create perpendicular lines, extract spectral data,
    and generate plots of spectral profiles and an RGB overlay.
    
    Parameters:
        fid_list (list): List of unique line IDs.
        tif_filepath (str): Filepath to the .tif file.
    """
    # Open the PlanetScope file
    with rasterio.open(tif_filepath) as src:
        raster = src.read()
        bounds = src.bounds
        transform = src.transform
        raster_crs = src.crs

    # Load lines the MLW annotations
    gdf = gpd.read_file("data/PS-LitterLines/mlw_annotations_20241211.gpkg").to_crs(raster_crs)
    selected_lines = gdf[gdf.index.isin(fid_list)]

    def giveline(linestring, distance, interval):
        """
        Generate multiple perpendicular lines at regular intervals along the input LineString.
        
        Parameters:
            linestring (shapely.geometry.LineString): Input LineString geometry.
            distance (float): Distance to shift the lines.
            interval (float): Interval (in meters) at which to create perpendicular lines.
            
        Returns:
            List[LineString]: List of perpendicular LineStrings at regular intervals.
        """
        from shapely.affinity import translate

        # Calculate the total number of intervals along the linestring
        num_intervals = int(linestring.length // interval)
        print(f"Number of intervals: {num_intervals}")

        # List to store all perpendicular lines
        perp_lines = []

        # If the linestring is too short, return a perpendicular line based on the centroid
        if num_intervals == 0:
            point_on_line = linestring.centroid
            point_left = translate(point_on_line, xoff=-distance)
            point_right = translate(point_on_line, xoff=distance)
            perp_lines.append(LineString([point_left, point_right]))
        else:
            # Iterate through each interval to calculate perpendicular lines
            for i in range(1, num_intervals + 1):
                # Get the point along the line at the current interval
                point_on_line = linestring.interpolate(i * interval)

                # Move the current point leftwards and rightwards by the distance
                point_left = translate(point_on_line, xoff=-distance)
                point_right = translate(point_on_line, xoff=distance)

                # Create a perpendicular line connecting the two shifted points
                perp_line = LineString([point_left, point_right])
                
                # Add the perpendicular line to the list
                perp_lines.append(perp_line)

        visualize_debug(linestring, perp_lines)

        return perp_lines
    
    def extract_along_line(raster_path, lines_gdf, bands=[1, 2, 3, 4]):
        """
        Extract all raster pixel values along each perpendicular line.

        Parameters:
            raster_path (str): Path to the raster file.
            lines_gdf (GeoDataFrame): GeoDataFrame containing perpendicular lines.
            bands (list): List of raster bands to extract data from.

        Returns:
            list: A list of dictionaries containing distances and reflectance values for each line.
        """
        from rasterio.features import rasterize
        results = []
        # TODO: Remove the with statement and replace with direct raster input
        with rasterio.open(raster_path) as src:
            cell_size = int(src.res[0])
            raster_transform = src.transform
            raster_crs = src.crs
            raster_shape = (src.height, src.width)
            
            # For each line in the GeoDataFrame
            for _, row in lines_gdf.iterrows():
                line = row['perpline']  # Get the LineString geometry
                
                # Rasterize the current line
                line_raster = rasterize(
                        [(line, 1)],
                        out_shape=raster_shape,
                        transform=raster_transform,
                        fill=0,
                        all_touched=True,
                        dtype=np.uint8
                    )
                
                # Get pixel indices for lines
                rows, cols = np.where(line_raster == 1)
                
                # Extract values for each VNIR band
                values = []
                for band in bands:
                    band_data = src.read(band)
                    band_values = band_data[rows, cols]
                    values.append(band_values)
                
                num_points = int(line.length // src.res[0]) + 1  # Number of points based on raster resolution
                distances = np.linspace(0, line.length, num_points)  # Distances along the line
                
                # Store results for this line
                results.append({
                    'distances': distances,
                    'values': np.array(values),  # Shape: (num_bands, num_points)
                })
        
        return results


    def plot_histograms(results, raster_path, bands_labels=["Blue", "Green", "Red", "NIR"]):
        """
        Plot histograms for reflectance values along each line and overlay the perpendicular lines on the raster image.

        Parameters:
            results (list): Results returned by extract_raster_along_line_full.
            raster_file (str): Path to the raster file for the RGB image.
            bands_labels (list): Labels for each raster band.
        """
        # Read the raster image for plotting
        with rasterio.open(raster_path) as src:
            # Assuming that bands 1, 2, and 3 are RGB
            blue = src.read(1)
            green = src.read(2)
            red = src.read(3)
            rgb = np.dstack([red, green, blue])

        # Create a figure with two subplots: one for the raster and one for histograms
        fig, axes = plt.subplots(len(results), 2, figsize=(16, 6 * len(results)))
        
        # Ensure axes is iterable even for single line case
        if len(results) == 1:
            axes = np.expand_dims(axes, axis=0)
        
        for i, result in enumerate(results):
            distances = result['distances']
            values = result['values']  # Shape: (num_bands, num_points)
            
            distances = np.arange(0, len(values[0])) # TODO: Replace / remove

            # Plot the raster image with the perpendicular line(s)
            ax_rgb = axes[i, 0]
            ax_rgb.imshow(rgb)
            ax_rgb.set_title(f"RGB Image with Perpendicular Line {i+1}")
            ax_rgb.set_xlabel("X (meters)")
            ax_rgb.set_ylabel("Y (meters)")

            # Assuming you have the perpendicular line as a GeoSeries in your result
            # If result['perpline'] is a list of LineStrings, you may need to loop over them
            #for line in result['perpline']:  # Assuming the line is stored in result['perpline']
            #    ax_rgb.plot(
            #        [point[0] for point in line.coords],  # X coordinates of the line
            #        [point[1] for point in line.coords],  # Y coordinates of the line
            #        color='red', linewidth=2, label=f"Line {i+1}"
            #    )
            #ax_rgb.legend()

            # Plot the spectral reflectance histograms
            ax_hist = axes[i, 1]
            for band_idx, band_values in enumerate(values):
                ax_hist.plot(distances, band_values, label=bands_labels[band_idx])

            ax_hist.set_title(f"Spectral Reflectance Along Perpendicular Line {i+1}")
            ax_hist.set_xlabel("Distance Along Line (meters)")
            ax_hist.set_ylabel("Reflectance")
            ax_hist.legend()
            ax_hist.grid(True)

        # Adjust layout
        plt.tight_layout()
        # Save the figure
        plt.savefig("doc/debug/perpline_hist.png")
        plt.show()


    # Create a column of perpendicular lines
    selected_lines['perpline'] = selected_lines.apply(lambda x: giveline(linestring=x.geometry, distance=1000, interval=20), axis=1)

    # Filter out NaN rows and create new GeoDataFrame
    perpendicular_lines = gpd.GeoDataFrame(selected_lines.explode(column='perpline', ignore_index=True), geometry='perpline', crs=raster_crs)

    # Extract TOA reflectance along the perpendicular lines
    results = extract_along_line(tif_filepath, perpendicular_lines)

    # Plot histograms
    plot_histograms(results, tif_filepath)

    print('...')






process_lines_extract_spectral([79, 81, 83, 84, 87], "data/PS-LitterLines/20210430_Greece/242d/20210430_082512_56_242d_3B_AnalyticMS.tif")
