import pandas as pd
import torch

# Custom modules
from core.dataloader import LitterLinesDataset, DatasetManager
from utils.visualization_functions import plot_featurespace, plot_litterlines_patches

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
        plot_litterlines_patches(images, masks, region_ids)
        
        # This function plots the data for each unique region using NDVI-RAI, compared to validated MLWs from Kikaki et al. (2020)
        plot_featurespace(test_loader, train_loader, val_loader)
    

    # Just for testing, GPT code for generating a RF model + small tuning for hyperpameters
    # Results were poor (IoU ~2.5%) due to massive imbalance in the dataset   
    def train_random_forest(train_loader, val_loader, n_positive=70000, n_negative=70000):
        import numpy as np
        import torch
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, jaccard_score
        
        np.random.seed(42)
        
        # Step 1: Prepare training data
        train_images, train_masks, region_ids = next(iter(train_loader))
        
        # Initialize empty lists for training features (X_train) and labels (y_train)
        X_train = []
        y_train = []
        
        # Step 2: Process each image to select target (1) and non-target (0) pixels
        for image, mask in zip(train_images, train_masks):
            # Flatten the image and mask
            image_flat = image.view(-1, 4).numpy()  # Flatten image to (num_pixels, 4) where 4 corresponds to VNIR bands
            mask_flat = mask.view(-1).numpy()  # Flatten the mask to (num_pixels,)
            
            # Get indices for target pixels (1) and non-target pixels (0)
            target_indices = np.where(mask_flat == 1)[0]
            non_target_indices = np.where(mask_flat == 0)[0]
            
            # Calculate number of pixels to sample from each class
            n_pos_to_sample = min(n_positive // len(train_images), len(target_indices))
            n_neg_to_sample = min(n_negative // len(train_images), len(non_target_indices))
            
            # Randomly select target and non-target pixels
            positive_pixels = np.random.choice(target_indices, n_pos_to_sample, replace=False)
            negative_pixels = np.random.choice(non_target_indices, n_neg_to_sample, replace=False)
            
            # Append selected pixels to the training data
            X_train.append(image_flat[positive_pixels])  # Features of positive pixels
            X_train.append(image_flat[negative_pixels])  # Features of negative pixels
            y_train.append(mask_flat[positive_pixels])  # Labels of positive pixels (1)
            y_train.append(mask_flat[negative_pixels])  # Labels of negative pixels (0)
        
        # Convert lists to numpy arrays
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        
        # GPT code for Random Forets model, hyperparameter search using RandomizedSearchCV
        # Performance was incredibly poor (IoU ~2.5%), due to massive imbalance in the dataset
        clf = RandomForestClassifier(
            n_estimators=300, 
            criterion='gini',
            min_samples_split=10,
            min_samples_leaf=50,
            max_features='sqrt',
            max_depth=None,
            oob_score=True,  # Helps estimate generalization
            class_weight={0:1, 1:10}, #'balanced_subsample',  # For imbalanced classes
            random_state=42,
            n_jobs=-1  # Parallel processing
        )

        clf.fit(X_train, y_train)  # Fit the classifier to the selected pixels

        # Step 4: Evaluate the model on the validation set
        val_images, val_masks, _ = next(iter(val_loader))

        # Flatten the validation images and masks
        X_val = val_images.view(val_images.size(0), -1, 4).numpy()  # Shape: (n_val_patches, 4 * 256 * 256)
        y_val = val_masks.view(val_masks.size(0), -1).numpy().astype(int)  # Shape: (n_val_patches, 256 * 256)

        # Flatten validation masks to 1D
        y_val_flat = y_val.flatten()

        # Step 5: Make predictions
        y_pred = clf.predict(X_val.reshape(-1, 4))  # Predict the class for each pixel

        # Step 6: Calculate metrics using sklearn
        accuracy = accuracy_score(y_val_flat, y_pred)
        f1 = f1_score(y_val_flat, y_pred, average='binary')  # Binary F1-score
        iou = jaccard_score(y_val_flat, y_pred, average='binary')  # Binary IoU (Jaccard Index)

        print(f"Validation accuracy: {accuracy * 100:.2f}%")
        print(f"F1-score: {f1:.4f}")
        print(f"IoU (Jaccard): {iou * 100:.2f}%")
        print('..')

    # Call the function to train and validate the model
    train_random_forest(train_loader, val_loader, n_positive=70000, n_negative=70000)

    
# Processing of the created dataset for dataset validation, and future future model development
process_litterlines(root_dir="data/LitterLines", plot_figures=False)

