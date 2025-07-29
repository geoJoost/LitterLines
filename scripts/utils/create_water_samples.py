import random
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd

def generate_water_points(output_path, seed=42):
    """
    Generate a shapefile with random points within a predefined bounding box based on unique dates read
    directly from .csv files.

    Parameters:
        output_path (str): Path to save the generated shapefile.
        seed (int): Seed for reproducibility of random points.

    Returns:
        None: Saves the shapefile at the specified location.
    """
    random.seed(seed)  # Seed for reproducibility
    
    # Predefined input files
    output_files = {
        "HDPE": "data/processed/HDPE_reflectance.csv",
        "PVC": "data/processed/PVC_reflectance.csv",
        "HDPE-BF": "data/processed/HDPE-BF_reflectance.csv",
        "HDPE-C": "data/processed/HDPE-C_reflectance.csv",
        "HDPE+Wood": "data/processed/HDPE+Wood_reflectance.csv",
        "wood": "data/processed/wood_reflectance.csv",
    }

    # Predefined bounding box generated from Copernicus Browser
    bounding_box = {
        "type": "Polygon",
        "coordinates": [
            [
                [26.513143, 39.040522],
                [26.513143, 39.043888],
                [26.525803, 39.043888],
                [26.525803, 39.040522],
                [26.513143, 39.040522],
            ]
        ],
    }

    # Extract all the unique dates covered by PLP2021 and PLP2022
    unique_dates = []
    for label, file_path in output_files.items():
        df = pd.read_csv(file_path)
        # Convert dates to strings and add them to the list
        unique_dates.extend(df["date"].astype(str).unique())

    # Remove duplicates by converting to a set and back to a list
    unique_dates = list(set(unique_dates))
    
    def generate_random_points(bbox, num_points, seed=42):
        """
        Generate random points within a bounding box.
        
        Parameters:
            bbox (dict): GeoJSON-like dictionary representing the bounding box as a Polygon.
            num_points (int): Number of random points to generate.

        Returns:
            list of shapely.geometry.Point: Randomly generated points.
        """
        #random.seed(seed)
        
        polygon = Polygon(bbox["coordinates"][0])
        min_x, min_y, max_x, max_y = polygon.bounds
        points = []

        while len(points) < num_points:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            point = Point(x, y)
            if polygon.contains(point):
                points.append(point)

        return points

    # Generate points for each date
    all_points = []
    all_dates = []
    all_types = []
    for date in unique_dates:
        points = generate_random_points(bounding_box, 5)
        all_points.extend(points)
        all_dates.extend([date] * len(points))
        all_types.extend(["Water"] * len(points))  # Add "Water" as type for all points

    # Create GeoDataFrame
    geo_df = gpd.GeoDataFrame(
        {"date": all_dates, "type": all_types},
        geometry=all_points,
        crs="EPSG:4326",
    )

    # Save to shapefile
    geo_df.to_file(output_path)

    print(f"Shapefile saved to {output_path}")