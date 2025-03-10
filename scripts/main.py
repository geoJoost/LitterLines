import pandas as pd
import torch

# Custom modules
from core.dataloader import LitterLinesDataset, DatasetManager
from utils.visualization_functions import plot_featurespace, plot_litterlines_patches, plot_signatures

def process_litterlines(root_dir, plot_figures=True):
    # Define LitterLines dataset
    transform = None  # Placeholder for transforms if needed
    dataset = LitterLinesDataset(root_dir=root_dir, transform=transform)

    # Initialize DatasetManager
    manager = DatasetManager(dataset, train_ratio=0.8, val_ratio=0.2, batch_size=30) # TODO: Fine-tune the batch_size variable
    print("DatasetManger called")
    
    # Get DataLoaders
    # Training/validation datasets are random selection from the LitterLines dataset
    train_loader = manager.get_dataloader("train") # (80%)
    val_loader = manager.get_dataloader("val", shuffle=False) # (20%)
    test_loader = manager.get_dataloader("test", shuffle=False) # Five PlanetScope scenes from Kikaki et al. (2020) taken on 2017-10-09
    
    if plot_figures:
        # This visualization function plots five patches (256px) from the LitterLines dataset in RGB, NDVI, RAI, and the label
        images, masks, region_ids = next(iter(train_loader))
        #plot_litterlines_patches(images, masks, region_ids)
        
        # This function plots the data for each unique region using NDVI-RAI, compared to validated MLWs from Kikaki et al. (2020)
        #plot_featurespace(test_loader, train_loader, val_loader)
        
        # This function is similar to the previous, but for spectral signatures instead
        plot_signatures(test_loader, train_loader, val_loader)
        

    
# Processing of the created dataset for dataset validation, and future future model development
process_litterlines(root_dir="data/LitterLines", plot_figures=True)

