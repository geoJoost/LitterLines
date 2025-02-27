import os
import time
import requests
from requests.auth import HTTPBasicAuth

# Custom modules
from utils import credentials
from utils.gee_downloader import create_bounding_box

'''

Using Planet API to order images 

This script utilizes the Planet API to search for and order satellite images based on specified criteria such as geographic location,
acquisition date, and cloud cover.

'''

def planet_query(lat_centroid, lon_centroid, date):
    # Put your API key here
    os.environ['PL_API_KEY'] = credentials.PLANET_KEY

    # Setup the API Key from the `PL_API_KEY` environment variable
    PLANET_API_KEY = os.getenv('PL_API_KEY')

    BASE_URL = "https://api.planet.com/data/v1"

    # Setup a session
    session = requests.Session()

    # Authenticate session with username and password, pass in an empty string for the password
    session.auth = (PLANET_API_KEY, "")

    ####### CREATE FILTERS ##############
    # Set up the geometry, date, and cloud cover filters 
    bounding_box = create_bounding_box(lat_centroid, lon_centroid)

    geojson_geometry = {
        'type': 'Polygon',
        'coordinates': [bounding_box]
    }
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": geojson_geometry
    }
    date_range_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": f"{date}T00:00:00.000Z",
            "lte": f"{date}T23:59:00.000Z"
        }
    }
    cloud_cover_filter = {  # TODO: Unused for now, can be reactivated for Adriatic Sea
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {
            "lte": 0.6
        }
    }

    # Combine filters
    combined_filter = {
        "type": "AndFilter",
        "config": [geometry_filter, date_range_filter]  # , cloud_cover_filter]
    }

    ####### SEARCHING ITEMS AND ASSETS #######
    item_type = "PSScene"
    search_request = {
        "item_types": [item_type],
        "filter": combined_filter
    }

    # Retry Logic
    retries = 4
    for attempt in range(retries + 1):
        try:
            # Fire off the POST request
            search_result = requests.post(
                'https://api.planet.com/data/v1/quick-search',
                auth=HTTPBasicAuth(PLANET_API_KEY, ''),
                json=search_request
            )

            if search_result.status_code != 200:
                raise ValueError(f"Unexpected status code: {search_result.status_code}")

            geojson = search_result.json()  # Parse the response as JSON

            # Initialize static variables for export
            potential_overlap = 'no'  # Initialize potential_overlap to 'no'

            # Check if there are any features in the geojson response
            if not geojson.get('features'):
                print(f"No PlanetScope images found on {date}. Exiting function.")
                return potential_overlap, [], [], []  # Return 'no' and an empty list if no features are found
            print(f"Processing PlanetScope images on {date}")

            # Extract variables for future processing
            image_ids = [feature['id'] for feature in geojson['features']]  # PS-ids
            sensor_ids = [feature['properties']['instrument'] for feature in geojson['features']]  # PS-sensor => PS2, PS2.SD or PSB.SD
            types = [] # List to store the image types as 3-bands or 4-bands (3b/4b). 8-band products are available from SuperDove (PSB.SD) but not stored for now
            
            # For loop to check for the presence of the "ortho_analytic_4b" asset for each image ID
            for image_id in image_ids:
                id_url = f'https://api.planet.com/data/v1/item-types/{item_type}/items/{image_id}/assets'
                result = requests.get(id_url, auth=HTTPBasicAuth(PLANET_API_KEY, ''))

                # Check whether 3-bands or 4-bands images are available. 8-bands are not stored but available in sensor_ids
                if "ortho_analytic_4b" in result.json():
                    types.append('4b')
                    potential_overlap = 'yes'
                    print(f"Image {image_id}: Found 4-band PlanetScope image")
                    
                elif "ortho_analytic_3b" in result.json():
                    types.append('3b')
                    potential_overlap = 'yes'
                    print(f"Image {image_id}: Found 3-band PlanetScope image")
                    
                else:
                    print(f"Image {image_id}: No valid asset found")
                    types.append(None)  # Append None or a placeholder for missing types

            # Return potential_overlap and activated_image_ids
            return potential_overlap, image_ids, sensor_ids, types

        except (requests.JSONDecodeError, ValueError) as e:
            # Log the error and retry
            print(f"Attempt {attempt + 1} failed: {e}")
            with open("error_log.txt", "a") as error_log:
                error_log.write(f"Failed query: lat={lat_centroid}, lon={lon_centroid}, date={date}\n")
                error_log.write(f"Error: {str(e)}\n")
                if 'search_result' in locals():
                    error_log.write(f"Response: {search_result.status_code} - {search_result.text[:500]}\n")

            if attempt < retries:
                print("Retrying...")
            else:
                print("All attempts failed. Logging and continuing.")
                return 'no', [], [], []

        except requests.exceptions.HTTPError as e:
            # Handle 502 Bad Gateway error
            if e.response.status_code == 502:
                print("Received 502 error. Retrying after 1 minute...")
                time.sleep(60)  # Wait for 60 seconds before retrying
                if attempt < retries:
                    print("Retrying...")
                else:
                    print("All attempts failed. Logging and continuing.")
                    with open("error_log.txt", "a") as error_log:
                        error_log.write(f"Failed query after retries: lat={lat_centroid}, lon={lon_centroid}, date={date}\n")
                        error_log.write(f"Error: {str(e)}\n")
                    return 'no', [], [], []

        except Exception as e:
            # Log unexpected exceptions
            print(f"Unexpected error: {e}")
            with open("error_log.txt", "a") as error_log:
                error_log.write(f"Unexpected error for query: lat={lat_centroid}, lon={lon_centroid}, date={date}\n")
                error_log.write(f"Error: {str(e)}\n")
            return 'no', [], [], []