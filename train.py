import os
import sys
import time
import glob
import wandb
import torch
import argparse
import itertools
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam
import matplotlib.pyplot as plt
from types import SimpleNamespace
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from scripts.segmentation_model import SegmentationModel
from scripts.core.dataloader import get_train_augmentation_normal, get_train_augmentation_low, get_train_augmentation_high, get_train_augmentation_very_high, get_val_augmentation, print_augmentation_pipeline

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model for marine litter windrows")
    parser.add_argument('--model_num', type=int, default=1,
                        help='Number of model in a certain configuration (for naming purposes)')
    parser.add_argument('--AUGMENTATIONS', type=str, default="high", choices=["none", "low", "normal", "high", "vhigh"],
                        help='Level of augmentations to use: low, normal, high, vhigh, none')
    parser.add_argument('--START_FROM_CHECKPOINT', type=str2bool, default=True,
                        help='Start from pretrained checkpoint/weights (True/False)')
    parser.add_argument('--BAND_MAPPING', type=str2bool, default=True,
                        help='Use band mapping for 1st conv weights (True/False)')
    parser.add_argument('--FIRST_CONV_WARMUP_EPOCHS', type=int, default=0,
                        help='Number of epochs for warm-up training of the 1st convolution')
    parser.add_argument('--BATCH_SIZE', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience (number of epochs)')
    return parser.parse_args()

args = parse_args()

# === TUNE HYPERPARAMETERS BASED ON HARDWARE AND NEEDS === #
model_num = args.model_num                          # Number of model in a certain configuration. For naming purposes.
AUGMENTATIONS = args.AUGMENTATIONS                  # Level of augmentations (low, normal, high, vhigh, none)
START_FROM_CHECKPOINT = args.START_FROM_CHECKPOINT  # Set False to start from scratch (if False, OOM likely)
BAND_MAPPING = args.BAND_MAPPING                    # Set True to implement partial weight transfer to the 1st convolution through band mapping
FIRST_CONV_WARMUP_EPOCHS = args.FIRST_CONV_WARMUP_EPOCHS  # Number of epochs for warm-up training of the 1st convolution
BATCH_SIZE = args.BATCH_SIZE                        # adjust for your hardware
early_stopping_patience = args.early_stopping_patience  # Early stop after n epochs without val_loss improvement

#Default values, uncomment to change disregarding flags
# model_num = 1                 # Number of model in a certain configuration. For naming purposes.
# AUGMENTATIONS = "high"        # Level of augmentations (low, normal, high, vhigh, none)
# START_FROM_CHECKPOINT = True  # Set False to start from scratch (if False, OOM likely)
# BAND_MAPPING = True           # Set True to implement partial weight transfer to the 1st convolution through band mapping
# FIRST_CONV_WARMUP_EPOCHS = 0  # Number of epochs for warm-up training of the 1st convolution
# BATCH_SIZE = 64               # adjust for your hardware
# early_stopping_patience = 20  # Early stop after n epochs without val_loss improvement

NUM_EPOCHS = 150                                    # increase for more training margine
NUM_WORKERS = 8                                     # adjust based on OS, number of CPU cores and utilization
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5                                 # increase if overfitting
IMAGE_SIZE = 256
HR_ONLY = True                                      # this for 4-channel input
LITTERLINES_PATH = r"/mnt/guanabana/raid/home/marin066/LitterLines dataset"
# ======================================================== #

# --- Split ratios ---
train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.10

# Note: The augmentation parameters are in the dataloader.py

# Folder for storing all best model weights
MODELS_DIR = "Models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("Script started")

# For naming:
bm     = "BMap"                              if BAND_MAPPING and START_FROM_CHECKPOINT                  else ""
freeze = f"Freeze{FIRST_CONV_WARMUP_EPOCHS}" if FIRST_CONV_WARMUP_EPOCHS != 0 and START_FROM_CHECKPOINT else ""
ckpt   = "Ckpt_"                             if START_FROM_CHECKPOINT                                   else ""

# Weights & Biases initialization
wandb.init(
    project="litterlines-segmentation",
    name=f"Model{model_num}_AUG{AUGMENTATIONS}{bm}{freeze}{ckpt}bs{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_sr{train_ratio}/{val_ratio}/{test_ratio}",
    config={
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": early_stopping_patience,
        "image_size": IMAGE_SIZE,
        "model": "unet",
    }
)

if START_FROM_CHECKPOINT:
    # --- Adjusting Sentinel-2 checkpoint (12 bands) for PlanetScope (4 bands) ---
    checkpoint_path = Path(r"/mnt/guanabana/raid/home/marin066/LitterLines_repo/weights/unet++3/epoch=43-val_loss=0.56-auroc=0.986.ckpt")
    model = SegmentationModel.load_from_checkpoint(checkpoint_path)
    # Replace first conv to accept 4 channels (PlanetScope)
    model.model.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Alters 1st convolution of segmentation model to only accept 4 chanels
    print("Initializing from checkpoint.")
    if BAND_MAPPING:
        # ---- Custom Partial Weight Transfer for First Conv through Band Mapping ----
        # 1. Access the original conv1 weights from the checkpoint
        orig_conv1_weight = model.__class__.load_from_checkpoint(checkpoint_path).model.encoder.conv1.weight.data  # [out_c, 12, k, k]

        # 2. Access the new conv1 layer (4 input channels)
        new_conv1 = model.model.encoder.conv1  # [out_c, 4, k, k]

        # 3. Sentinel-2 12-band → PlanetScope 4-band mapping
        band_map = [3, 2, 1, 7]  # [B4, B3, B2, B8] in S2  (R, G, B, NIR)

        # 4. Transfer weights
        with torch.no_grad():
            for new_idx, old_idx in enumerate(band_map):
                new_conv1.weight[:, new_idx, :, :] = orig_conv1_weight[:, old_idx, :, :]
            if new_conv1.bias is not None:
                new_conv1.bias.zero_()  # Usually zero, since most convs don't use bias

        print("Sentinel-2 checkpoint weights for corresponding bands transferred to PlanetScope's 1st convolution (partial weight transfer).")
else:
    # Start from scratch, without checkpoint
    args = SimpleNamespace(
        model="unet",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        hr_only=HR_ONLY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
    )
    print("Initializing with random weights (no checkpoint).")
    model = SegmentationModel(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {str(device).upper()}")
model = model.to(device)

use_amp = device.type == 'cuda' # Automatic Mixed Precision (AMP) on CUDA
if use_amp:
    scaler = torch.amp.GradScaler()
else:
    scaler = None

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Add repo root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.core.dataloader import LitterLinesPatchDataset

print("Loading dataset...")
base_dataset = LitterLinesPatchDataset(LITTERLINES_PATH)
print(f"\nDataset loaded. Total patches: {len(base_dataset)}")

# ---- PATCH-LEVEL REGION-AWARE SPLIT (Optimal brute-force split by patch count) ----
# 1. Extract region_ids for each patch
print("Building patch-region mapping...")
region_ids = []
for i in range(len(base_dataset)):
    try:
        patch, mask, region_id = base_dataset[i]
    except Exception:
        raise RuntimeError("Your LitterLinesPatchDataset must return (patch, mask, region_id) for region-aware splitting!")
    region_ids.append(region_id)

# 2. Map region_id -> list of patch indices
region_to_patch_idxs = defaultdict(list)
for idx, region_id in enumerate(region_ids):
    region_to_patch_idxs[region_id].append(idx)

all_regions = list(region_to_patch_idxs.keys())
n_regions = len(all_regions)


total_patches = sum(len(patch_idxs) for patch_idxs in region_to_patch_idxs.values())
target_train = int(round(train_ratio * total_patches))
target_val = int(round(val_ratio * total_patches))
target_test = total_patches - target_train - target_val  # Use remainder for test

# --- Decide number of regions per split (for 8 regions, e.g. 5/2/1) ---
train_regions_n = int(round(train_ratio * n_regions))
val_regions_n = int(round(val_ratio * n_regions))
test_regions_n = n_regions - train_regions_n - val_regions_n

# --- Brute-force search for region split with patch counts closest to desired ratios (KIKAKI is fixed to test dataset) ---
print("Finding optimal region split...")

best_score = float('inf')
best_split = None

region_indices = list(range(n_regions))
region_list = all_regions

# ---- Find KIKAKI region index, error if not found ----
kikaki_name = "KIKAKI"
assert kikaki_name in region_list, f"Region '{kikaki_name}' not found in your data!"
kikaki_idx = region_list.index(kikaki_name)

# ---- Remove KIKAKI from available indices for brute-force ----
remaining_indices = [i for i in region_indices if i != kikaki_idx]

# We must assign KIKAKI to test set, so test_regions_n will be at least 1
test_regions_n_adj = test_regions_n - 1  # We'll add KIKAKI manually after

# For small n_regions, this is very fast.
for train_idxs in itertools.combinations(remaining_indices, train_regions_n):
    remaining1 = set(remaining_indices) - set(train_idxs)
    for val_idxs in itertools.combinations(remaining1, val_regions_n):
        test_idxs = list(remaining1 - set(val_idxs))
        # Add KIKAKI to test
        final_test_idxs = test_idxs + [kikaki_idx]
        # Get region names
        train_regions = [region_list[i] for i in train_idxs]
        val_regions = [region_list[i] for i in val_idxs]
        test_regions = [region_list[i] for i in final_test_idxs]
        # Compute patch counts
        n_train_patches = sum(len(region_to_patch_idxs[r]) for r in train_regions)
        n_val_patches   = sum(len(region_to_patch_idxs[r]) for r in val_regions)
        n_test_patches  = sum(len(region_to_patch_idxs[r]) for r in test_regions)
        # Score: sum of absolute patch differences from targets
        score = abs(n_train_patches - target_train) + abs(n_val_patches - target_val) + abs(n_test_patches - target_test)
        if score < best_score:
            best_score = score
            best_split = (train_regions, val_regions, test_regions)
            # Early exit if perfect
            if score == 0:
                break

train_regions, val_regions, test_regions = best_split

# Build patch indices from regions
train_idxs = [idx for region in train_regions for idx in region_to_patch_idxs[region]]
val_idxs   = [idx for region in val_regions   for idx in region_to_patch_idxs[region]]
test_idxs  = [idx for region in test_regions  for idx in region_to_patch_idxs[region]]

print(f"Optimal split found (score={best_score})")
print(f"Targets: train={target_train}, val={target_val}, test={target_test} patches")
print(f"Actual : train={len(train_idxs)}, val={len(val_idxs)}, test={len(test_idxs)}")
print(f"Region counts: train={len(train_regions)}, val={len(val_regions)}, test={len(test_regions)}")
print(f"Regions assigned to train: {train_regions}")
print(f"Regions assigned to val:   {val_regions}")
print(f"Regions assigned to test:  {test_regions}\n")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

def to_tensor(x, dtype=torch.float):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).type(dtype)
    return x.type(dtype)

# 3. Wrapper that creates patch-level subsets and applies augmentation/transform per sample
class PatchDataset(Dataset):
    def __init__(self, base_dataset, idxs, transform=None):
        self.base_dataset = base_dataset
        self.idxs = idxs
        self.transform = transform

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, i):
        patch, mask, _ = self.base_dataset[self.idxs[i]] # Only return (patch, mask) for training
        # For Albumentations, convert to numpy and (H,W,C) format
        patch_np = patch.numpy()                # (4, 256, 256)
        mask_np = mask.numpy()                  # (1, 256, 256)
        # Remove mask channel for transform
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        if self.transform:
            patch_np = np.transpose(patch_np, (1, 2, 0))   # (256, 256, 4)
            augmented = self.transform(image=patch_np, mask=mask_np)
            # ToTensorV2() outputs tensors, not numpy arrays!
            patch = augmented['image']     # shape: (4, 256, 256), torch.Tensor
            mask = augmented['mask']       # shape: (1, 256, 256), torch.Tensor
        else:
            # Still convert to float tensors for DataLoader/model
            patch = torch.from_numpy(patch_np).float()
            mask = torch.from_numpy(mask_np).float()
        return patch, mask

# Instantiate datasets with correct transforms
if AUGMENTATIONS.lower() == "normal":
    train_dataset = PatchDataset(base_dataset, train_idxs, transform=get_train_augmentation_normal())
elif AUGMENTATIONS.lower() == "low":
    train_dataset = PatchDataset(base_dataset, train_idxs, transform=get_train_augmentation_low())
elif AUGMENTATIONS.lower() == "high":
    train_dataset = PatchDataset(base_dataset, train_idxs, transform=get_train_augmentation_high())
elif AUGMENTATIONS.lower() == "vhigh":
    train_dataset = PatchDataset(base_dataset, train_idxs, transform=get_train_augmentation_very_high())
else: train_dataset = PatchDataset(base_dataset, train_idxs)
print("Augmentation level:", AUGMENTATIONS.upper())

# Print train augmentation pipeline for reproducibility and logs
if AUGMENTATIONS.lower() == "normal":
    print_augmentation_pipeline(get_train_augmentation_normal())
elif AUGMENTATIONS.lower() == "low":
    print_augmentation_pipeline(get_train_augmentation_low())
elif AUGMENTATIONS.lower() == "high":
    print_augmentation_pipeline(get_train_augmentation_high())
elif AUGMENTATIONS.lower() == "vhigh":
    print_augmentation_pipeline(get_train_augmentation_very_high())

val_dataset = PatchDataset(base_dataset, val_idxs, transform=get_val_augmentation())
test_dataset = PatchDataset(base_dataset, test_idxs, transform=get_val_augmentation())

# 4. DataLoaders (no custom collate, standard batch size)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print("Dataloaders created.", end=' ')

# --- TRAINING LOOP ---
best_val_loss = float('inf')
best_train_loss = float('inf')
best_epoch = -1
epochs_no_improve = 0
bar_format = '{l_bar}{bar} | Batch {n_fmt}/{total_fmt} [Elapsed: {elapsed}s, ETA: {remaining}s, ~{rate_fmt}]'

train_losses = []
val_losses = []
epoch_times = []

print("Starting training loop...")

try:
        # ========== WARM-UP 1st CONV LAYER ONLY ==========
    if FIRST_CONV_WARMUP_EPOCHS != 0 and START_FROM_CHECKPOINT:
        # Freeze all encoder layers except the first conv
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        for param in model.model.encoder.conv1.parameters():
            param.requires_grad = True
        print(f"Encoder frozen except for first convolution. Starting warm-up training for {FIRST_CONV_WARMUP_EPOCHS} epochs...")

        for epoch in range(FIRST_CONV_WARMUP_EPOCHS):
            model.train()
            running_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"[Warm-up] Epoch {epoch+1}", leave=False, bar_format=bar_format):
                images = images.to(device)
                masks = masks.to(device)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                optimizer.zero_grad()
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, masks.float())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks.float())
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * images.size(0)
            avg_loss = running_loss / len(train_loader.dataset)
            print(f"Warm-up Epoch [{epoch+1}/{FIRST_CONV_WARMUP_EPOCHS}], Loss: {avg_loss:.4f}")

        # Unfreeze all encoder layers for full training
        for param in model.model.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen. Proceeding to full training...")

    # --- Full training start ---
    for epoch in range(NUM_EPOCHS):
        print(f"\nStarting epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}", leave=False, bar_format=bar_format):
            images = images.to(device)
            masks = masks.to(device)
            if masks.ndim == 3: # (B, H, W)
                masks = masks.unsqueeze(1) # --> (B, 1, H, W)
            optimizer.zero_grad()
            # ----- AMP logic -----
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images) # outputs shape: (B, 1, H, W), masks: (B, 1, H, W) or (B, H, W)
                loss = criterion(outputs, masks.float())
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # --- VALIDATION LOOP ---
        print(f"Running validation for epoch {epoch+1}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}", leave=False, bar_format=bar_format):
                images = images.to(device)
                masks = masks.to(device)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, masks.float())
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # --- wandb log epoch metrics ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_loss
        })
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
           # Remove previous best models from Models folder
            for file in glob.glob(os.path.join(MODELS_DIR, f"Model{model_num}_AUG{AUGMENTATIONS}{bm}{freeze}{ckpt}bs{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_sr{train_ratio}-{val_ratio}-{test_ratio}_epoch*.pth")):
                os.remove(file)
            # Save the new best model in Models folder
            print("Saving new best model...")
            model_path = os.path.join(
                MODELS_DIR,
                f"Model{model_num}_AUG{AUGMENTATIONS}{bm}{freeze}{ckpt}bs{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_sr{train_ratio}-{val_ratio}-{test_ratio}_epoch{epoch+1}.pth"
            )
            torch.save(model.state_dict(), model_path)

            best_val_loss = val_loss
            best_train_loss = epoch_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0 # reset to 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping triggered: no improvement in validation loss for {early_stopping_patience} consecutive epochs (stopped at epoch {epoch+1}).")
                break

        # Store epoch duration
        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)

except KeyboardInterrupt:
    print("\n*** KeyboardInterrupt detected! Training stopped early by user. Creating summary... ***\n")

# --- TEST SET EVALUATION ---
def compute_iou(preds, targets, threshold=0.5, eps=1e-7):
    # preds, targets: (B, 1, H, W)
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou

def compute_dice(preds, targets, threshold=0.5, eps=1e-7):
    # preds, targets: (B, 1, H, W)
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    total = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2 * intersection + eps) / (total + eps)
    return dice

# Load best model weights for fair test
if best_epoch > 0:
    best_model_path = os.path.join(
        MODELS_DIR,
        f"Model{model_num}_AUG{AUGMENTATIONS}{bm}{freeze}{ckpt}bs{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_sr{train_ratio}-{val_ratio}-{test_ratio}_epoch{best_epoch}.pth"
    )
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    test_loss = 0.0
    ious = []
    dices = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, masks.float())
            test_loss += loss.item() * images.size(0)
            # Compute IoU & Dice for the batch
            ious.append(compute_iou(outputs, masks).cpu())
            dices.append(compute_dice(outputs, masks).cpu())
    test_loss /= len(test_loader.dataset)
    mean_iou = torch.cat(ious).mean().item()
    mean_dice = torch.cat(dices).mean().item()
else:
    test_loss = float('nan')
    mean_iou = float('nan')
    mean_dice = float('nan')
    print("Warning: No best model was saved, skipping test set evaluation.")

# --- LEARNING CURVE PLOT ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve.png")  # Save to file as well
plt.show()

# --- wandb log test metrics & learning curve image ---
wandb.log({"learning_curve": wandb.Image("learning_curve.png"),
    "test_loss": test_loss,
    "test_iou": mean_iou,
    "test_dice": mean_dice,
    "best_val_loss": best_val_loss
})
plt.show()

# --- FINAL SUMMARY ---
num_train = len(train_dataset)
num_val = len(val_dataset)
num_test = len(test_dataset)
num_batches = len(train_loader)
num_val_batches = len(val_loader)
num_test_batches = len(test_loader)
avg_epoch_time = np.mean(epoch_times) if epoch_times else float('nan')

print("\n=== Training complete! Summary ===")
print(f"Total epochs:                 {len(train_losses)}")
print(f"Total patches:                {len(base_dataset)}")
print(f"Train patches:                {num_train}")
print(f"Validation patches:           {num_val}")
print(f"Test patches:                 {num_test}")
print(f"Average time per epoch:       {avg_epoch_time:.2f}s")
print(f"Training batches/epoch:       {num_batches}")
print(f"Validation batches/epoch:     {num_val_batches}")
print(f"Test batches/epoch:           {num_test_batches}")

if best_epoch > 0:
    print(f"Best model: epoch {best_epoch} | train_loss={best_train_loss:.4f} | val_loss={best_val_loss:.4f} | Test → loss: {test_loss:.4f}, IoU:  {mean_iou:.4f}, Dice: {mean_dice:.4f}")
    print(f"Learning curve saved to learning_curve.png")
    wandb.save(model_path)  # Log model checkpoint to wandb

else:
    print("No best model was saved (possible issue in training/validation loss tracking).")

# Finish wandb run
wandb.finish()