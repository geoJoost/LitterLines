import os
import glob
from osgeo import gdal
from eodag import EODataAccessGateway
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Custom modules
from utils import credentials


def create_tif_stack(safe_dir, output_file, clip_bbox):
    """
    Converts Sentinel-2 .SAFE files into a single multi-band GeoTIFF.
    
    Args:
        safe_dir (str): The path to the .SAFE directory.
        output_file (str): The path to the output multi-band TIFF file.
    """
    # Set PROJ_LIB to ensure GDAL can find the necessary PROJ database files
    os.environ["PROJ_LIB"] = credentials.PROJ_PATH
    gdal.UseExceptions()

    # Construct the path to the bands (usually located under the "GRANULE" subdirectory)
    granule_dir = os.path.join(safe_dir, "GRANULE")
    
    # Find all JP2 files in the "IMG_DATA" folder (typically under GRANULE/*/IMG_DATA)
    jp2_files = glob.glob(os.path.join(granule_dir, '*', 'IMG_DATA', '*.jp2'))

    # Crucial step: sort the .jp2 files in the correct order to prevent mixed results in the final TIFF
    valid_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'] # L1C bands
    sorted_files = sorted(
        [jp2 for jp2 in jp2_files if "_TCI" not in jp2], # Exclude TCI
        key=lambda x: valid_bands.index(os.path.basename(x).split("_")[-1].replace('.jp2', ''))
    )

    # TODO: Optional is to include atmospheric correction; for now, we can do without as no analysis of the Sentinel-2 data is done

    # Use GDAL to stack the band files into a single multi-band GeoTIFF
    vrt = gdal.BuildVRT('', sorted_files, separate=True)

    # Apply clipping directly to the VRT since entire Sentinel-2 tiles are downloaded by EODAG
    vrt_crs = gdal.Open(sorted_files[0]).GetProjection() # Retrieve CRS from original JP2 files
    bbox_gdf = clip_bbox.to_crs(vrt_crs)
    bbox_coords = bbox_gdf.total_bounds

    # Use gdal.Warp to clip the VRT
    vrt_clipped = gdal.Warp('',
                                vrt,
                                outputBounds=bbox_coords,
                                format='VRT') # Keep as VRT in memory


    # Translate the VRT to a multi-band GeoTIFF
    gdal.Translate(output_file, 
                   vrt_clipped, # TODO: Fix compression of ZSTD. Alternatives are LZW and DEFLATE but they slow down reading speed
                   #creationOptions=["COMPRESS=ZSTD", "PREDICTOR=2", "ZLEVEL=1"], # See https://kokoalberti.com/articles/geotiff-compression-optimization-guide/
                   ) 

def download_s2(search_criteria, clip_bbox, download_dir):
    # Sign-in credentials to PEPS
    os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = credentials.PEPS_USERNAME
    os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = credentials.PEPS_PASSWORD 
    os.environ["EODAG__COP_DATASPACE__AUTH__CREDENTIALS__USERNAME"] = credentials.COP_USERNAME
    os.environ["EODAG__COP_DATASPACE__AUTH__CREDENTIALS__PASSWORD"] = credentials.COP_PASSWORD
    
    # Initialize folders for downloading and processing
    safe_dir = os.path.join(download_dir, "safe_tiles")  # .SAFE files
    tif_dir = os.path.join(download_dir, "tif_tiles")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(safe_dir, exist_ok=True) # .SAFE files
    os.makedirs(tif_dir, exist_ok=True) # .tif files

    # Initialize EODataAccessGateway instance
    dag = EODataAccessGateway()

    # Search for products using the defined criteria
    print("Searching for Sentinel-2 data...")
    products = dag.search_all(**search_criteria)

    # Guarantee that the product entirely contains the search area
    # Note: It does not reduce the number of products for PLP2021
    filtered_products = products.filter_overlap(
        contains=True,
        geometry=search_criteria['geom']
    )

    # If no products are found, handle it
    if not filtered_products:
        print("No products found for the given search criteria.")
        return
    print(f"Found {len(filtered_products)} Sentinel-2 tiles")

    # Download the first N products
    N = 1
    for product in products:#products[:N]:
        product_title = product.properties['title'] #=> S2A_MSIL1C_20230909T103631_N0509_R008_T31TCG_20230909T141300

        # From the Copernicus Dataspace, L1C files are available at two processing levels: N0300/N0301 N0500
        # As this effectively doubles the number of images, we select only the newest processing level of v05.00
        if 'N0500' not in product_title:
            #print(f"Skipping product '{product_title}' due to incorrect processing level.")
            continue
        
        # Make file-paths for indexing and/or saving
        safe_filepath = os.path.join(safe_dir, product_title, f"{product_title}.SAFE") # Copernicus has double-nested SAFE files
        tif_filepath = os.path.join(tif_dir, f"{product_title}.tif")

        # Download SAFE files
        if not os.path.exists(safe_filepath):

            # Before downloading, print a quicklook for the user to make sure its the correct image
            plotting = False 
            if plotting:
                # This line takes care of downloading the quicklook
                quicklook_path = product.get_quicklook()

                fig = plt.figure(figsize=(10, 8))
                # Read the quicklook image
                img = mpimg.imread(quicklook_path)

                # Display the quicklook image
                ax = fig.add_subplot(1, 1, 1)  # Single subplot for the quicklook
                ax.set_title(f"Quicklook for {product_title}")
                plt.imshow(img)
                plt.axis('off')  # Hide axes if preferred
                
                plt.tight_layout()
                plt.savefig("doc/figures/quicklook.png")

            # Download the product using EODAG, and store in safe_tiles directory
            print(f"\nDownloading product '{product_title}'.")
            dag.download(product, 
                         outputs_prefix=safe_dir,
                         wait=1, # Wait one minute before retrying if initial download fails
                         timeout=60 # If download fails, maximum time before stop retrying to download
                         )
        else:
            print(f"\nSAFE file already exists: '{product_title}.SAFE'. Skipping download.")

        # Process TIFF files
        if not os.path.exists(tif_filepath):
            print(f"TIFF file not found. Processing .SAFE file to create '{product_title}.tif'.")
            
            # Create multi-band TIFF file
            create_tif_stack(safe_filepath, tif_filepath, clip_bbox)
        else:
            print(f"TIFF file already exists: '{product_title}.tif'. Skipping download")

    print("Download script completed.")