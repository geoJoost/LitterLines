import os
import glob
import torch
import shutil
import rasterio
import itertools
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score
from scripts.core.ensemble_model import EnsembleModel
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -- Import your modules/classes (update these paths if necessary!) --
from scripts.segmentation_model import SegmentationModel
from scripts.core.dataloader import LitterLinesPatchDataset

# === DO NOT FORGET TO MATCH THE Split Ratios === #

# ----------- USER CONFIG -----------
# --- Split Ratios ---
train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.1

# ====== HYPERPARAMETERS ====== #
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
IMAGE_SIZE = 256
HR_ONLY = True
START_FROM_CHECKPOINT = True
# ============================= # 

EnsembleVis = False # Set True to get an ensemble of the probabilites and standard deviations from the models of a folder
EnsembleFolder = "BestConfigModels"
ProbVis = False # Set True to get a probability visualization (along with rgb, ground truth mask and predicted mask) of a specific model (CHECKPOINT_PATH)
CHECKPOINT_PATH = r"BestConfigModels/Model2_AUGlowBMapCkpt_bs64_lr0.001_wd1e-05_sr0.75-0.15-0.1_epoch70.pth"
ScenePred = True # Set True for whole scene prediction
LITTERLINES_PATH = "/mnt/guanabana/raid/home/marin066/LitterLines dataset"  # Your dataset path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BATCHES = 99 # to limit the number of batches
THRESHOLD = 0.5
ckpt = "_ckpt" if START_FROM_CHECKPOINT else "" # for naming SAVE_DIR
SAVE_DIR = f"ModelVis_bs{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}{ckpt}"
# -----------------------------------

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint file not found at {CHECKPOINT_PATH}")

# Delete any previous ModelVis_* directory in current directory
for old_dir in glob.glob("ModelVis_*"):
    if os.path.isdir(old_dir):
        print(f"Deleting previous visualization directory: {old_dir}")
        shutil.rmtree(old_dir)

os.makedirs(SAVE_DIR, exist_ok=True)

# --- 1. Load Model (PyTorch Lightning) ---
print(f"Loading model checkpoint from: {CHECKPOINT_PATH}")

args = SimpleNamespace(
    model="unet",
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    hr_only=HR_ONLY,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
)

# --- Load Model for Inference (must match training time architecture!) ---
import segmentation_models_pytorch as smp
from scripts.segmentation_model import SegmentationModel

base_model = smp.UnetPlusPlus(
    encoder_name="resnet34",        # Use same encoder as training!
    encoder_weights=None,
    in_channels=4,
    classes=1,
)

model = SegmentationModel(args)
model.model = base_model

if START_FROM_CHECKPOINT:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

# --- 2. Load Dataset (must return patch, mask, region_id) ---
print("Loading dataset...")
base_dataset = LitterLinesPatchDataset(LITTERLINES_PATH)
print(f"\nTotal patches in dataset: {len(base_dataset)}")

# --- 3. Region split (test set selection) ---

# Assume base_dataset is already loaded (as in training)
region_ids = []
for i in range(len(base_dataset)):
    patch, mask, region_id = base_dataset[i]
    region_ids.append(region_id)

region_to_patch_idxs = defaultdict(list)
for idx, region_id in enumerate(region_ids):
    region_to_patch_idxs[region_id].append(idx)

# --- Use SAME split logic as in training ---
all_regions = list(region_to_patch_idxs.keys())  # Use sorted(region_to_patch_idxs.keys()) if you want alphabetical!
n_regions = len(all_regions)

total_patches = sum(len(patch_idxs) for patch_idxs in region_to_patch_idxs.values())
target_train = int(round(train_ratio * total_patches))
target_val = int(round(val_ratio * total_patches))
target_test = total_patches - target_train - target_val

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

print(f"Train/val/test patch counts: {len(train_idxs)}, {len(val_idxs)}, {len(test_idxs)}")
print(f"Test regions: {test_regions}")
print(f"Expecting {len(test_idxs)} test patches.")

class PatchDataset(Dataset):
    def __init__(self, base_dataset, idxs):
        self.base_dataset = base_dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        # Unpack all three values, including region_id
        patch, mask, region_id = self.base_dataset[self.idxs[i]]
        return patch, mask, region_id

test_dataset = PatchDataset(base_dataset, test_idxs)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Test dataset size: {len(test_dataset)}")
print(f"Running visualize_predictions: num batches in loader: {len(test_loader)}")

# --- 4. Visualization Function ---
def safe_squeeze(img):
    """Squeeze singleton dims; if shape is (C,H,W) with C=1, make (H,W)."""
    img = np.squeeze(img)
    if img.ndim == 2:
        return img
    elif img.ndim == 3 and img.shape[0] == 1:
        return img[0]
    else:
        # fallback: try full squeeze or error
        return img

# --- PROPABILITY VISUALIZATION ---
def visualize_predictions_with_probs(model=model, dataloader=test_loader, device=DEVICE, num_batches=NUM_BATCHES, threshold=THRESHOLD, save_dir=SAVE_DIR):
    """
    Visualizes predictions for a batch of images from a dataloader.
    For each patch, displays/saves:
      - Input (RGB satellite image)
      - Ground Truth Mask
      - Predicted Probability Map
      - Predicted Binary Mask (using given threshold)
    """
    # Firstly calculates and prints test metrics
    print("\n=== TEST SET METRICS ===")
    print(f"Test Loss:  {test_loss:.4f}")
    print(f"Mean IoU:   {mean_iou:.4f}")
    print(f"Mean Dice:  {mean_dice:.4f}")
    print("========================\n")

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks, region_ids) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)           # (B, 1, H, W)
            preds = probs > threshold

            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            probs = probs.cpu().numpy()
            preds = preds.cpu().numpy()

            batch_size = images.shape[0]
            for i in range(batch_size):
                fig = plt.figure(figsize=(7, 7))
                gs = fig.add_gridspec(2, 2)

                axs = [
                    fig.add_subplot(gs[0, 0]),
                    fig.add_subplot(gs[0, 1]),
                    fig.add_subplot(gs[1, 0]),
                    fig.add_subplot(gs[1, 1]),
                ]

                # Adjust left/right and top/bottom independently
                fig.subplots_adjust(
                    left=0.03,   # ← left padding (smaller = tighter)
                    right=0.97,  # → right padding
                    top=0.95,    # ↑ top padding
                    bottom=0.05, # ↓ bottom padding
                    wspace=0.1,  # horizontal space between subplots
                    hspace=0.2   # vertical space between subplots
)
                # --- INPUT PATCH (RGB composite, per-channel normalization) ---
                bgr = images[i, :3, :, :]  # (3, H, W)
                rgb = bgr[::-1, :, :]      # Convert BGR to RGB
                rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)

                # Normalize each channel individually
                rgb_norm = np.zeros_like(rgb)
                for c in range(3):
                    channel = rgb[..., c]
                    rgb_norm[..., c] = (channel - np.min(channel)) / (np.ptp(channel) + 1e-8)

                axs[0].imshow(rgb_norm)
                axs[0].set_title("Input Patch (RGB, norm)")
                axs[0].axis('off')

                # 2. Ground Truth Mask
                mask_img = masks[i][0] if masks[i].ndim == 3 else masks[i]
                axs[1].imshow(mask_img, cmap='gray', vmin=0, vmax=1)
                axs[1].set_title("Ground Truth Mask")
                axs[1].axis('off')

                # 3. Probability Heatmap
                # Plot predicted probabilities (bottom-left)
                im2 = axs[2].imshow(probs[i][0], cmap='viridis', vmin=0, vmax=1)
                axs[2].set_title("Predicted Probabilities")
                axs[2].axis('off')

                # Add colorbar below the subplot using make_axes_locatable
                divider = make_axes_locatable(axs[2])
                cax = divider.append_axes("bottom", size="3%", pad=0.05)  # adjust pad if needed
                cbar = plt.colorbar(im2, cax=cax, orientation='horizontal')
                cbar.set_ticks([0.0, 0.5, 1.0])
                cbar.ax.tick_params(labelsize=9)

                # 4. Predicted Binary Mask based on threshold
                axs[3].imshow(preds[i][0], cmap='gray', vmin=0, vmax=1)
                axs[3].set_title(f"Predicted Mask (>{threshold:.2f})")
                axs[3].axis('off')
                
                if save_dir is not None:
                    region_id = region_ids[i]
                    fname = os.path.join(save_dir, f"batch{batch_idx}_patch{i}_{region_id}.png")
                    fig.savefig(fname, dpi=300)
                    print(f"Saved: {fname}")
                    plt.close()
                else:
                    plt.show()

            if batch_idx + 1 >= num_batches:
                break

# Ensemble probabilities and standard deviation
def visualize_ensemble_predictions_with_uncertainty(model=model, dataloader=test_loader, device=DEVICE, num_batches=NUM_BATCHES, save_dir=SAVE_DIR, threshold=THRESHOLD):
    """
    Visualizes ensemble predictions with uncertainty.
    For each patch, shows:
      - Input Patch (RGB)
      - Binary Predicted Mask from Ensemble Mean
      - Mean Probability Heatmap
      - Std Dev Heatmap (Uncertainty)
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks, region_ids) in enumerate(dataloader):
            images = images.to(device)
            mean_logits, std_logits = model(images)

            mean_probs = torch.sigmoid(mean_logits)
            preds = (mean_probs > threshold).float()

            images = images.cpu().numpy()
            mean_probs = mean_probs.cpu().numpy()
            std_logits = std_logits.cpu().numpy()
            preds = preds.cpu().numpy()

            batch_size = images.shape[0]
            for i in range(batch_size):
                fig = plt.figure(figsize=(8, 8))
                gs = fig.add_gridspec(2, 2)

                axs = [
                    fig.add_subplot(gs[0, 0]),
                    fig.add_subplot(gs[0, 1]),
                    fig.add_subplot(gs[1, 0]),
                    fig.add_subplot(gs[1, 1]),
                ]

                # 1. Input Patch (RGB normalized)
                bgr = images[i, :3, :, :]
                rgb = bgr[::-1, :, :]
                rgb = np.transpose(rgb, (1, 2, 0))
                rgb_norm = np.zeros_like(rgb)
                for c in range(3):
                    ch = rgb[..., c]
                    rgb_norm[..., c] = (ch - ch.min()) / (np.ptp(ch) + 1e-8)
                axs[0].imshow(rgb_norm)
                axs[0].set_title("Input Patch (RGB)")
                axs[0].axis("off")

                # 2. Predicted Binary Mask
                axs[1].imshow(preds[i][0], cmap="gray", vmin=0, vmax=1)
                axs[1].set_title(f"Predicted Mask (>{threshold:.2f})")
                axs[1].axis("off")

                # 3. Mean Probability Heatmap
                im2 = axs[2].imshow(mean_probs[i][0], cmap="viridis", vmin=0, vmax=1)
                axs[2].set_title("Mean Probability")
                axs[2].axis("off")
                divider2 = make_axes_locatable(axs[2])
                cax2 = divider2.append_axes("bottom", size="4%", pad=0.05)
                plt.colorbar(im2, cax=cax2, orientation="horizontal")

                # 4. Std Dev Heatmap
                im3 = axs[3].imshow(std_logits[i][0], cmap="magma", vmin=0)
                axs[3].set_title("Std Devs")
                axs[3].axis("off")
                divider3 = make_axes_locatable(axs[3])
                cax3 = divider3.append_axes("bottom", size="4%", pad=0.05)
                plt.colorbar(im3, cax=cax3, orientation="horizontal")

                plt.tight_layout()
                if save_dir is not None:
                    region_id = region_ids[i]
                    fname = os.path.join(save_dir, f"ensemble4x_batch{batch_idx}_patch{i}_{region_id}.png")
                    plt.savefig(fname, dpi=300)
                    print(f"Saved: {fname}")
                    plt.close()
                else:
                    plt.show()

            if batch_idx + 1 >= num_batches:
                break
    # Print metrics again
    print("\n=== TEST SET METRICS ===")
    print(f"Test Loss:  {test_loss:.4f}")
    print(f"Mean IoU:   {mean_iou:.4f}")
    print(f"Mean Dice:  {mean_dice:.4f}")
    print("========================")

# ================================ Run metrics ================================
import torch.nn as nn

def compute_iou(preds, targets, threshold=THRESHOLD, eps=1e-7):
    """
    Computes IoU for a batch of predicted and target masks.
    """
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou

def compute_dice(preds, targets, threshold=THRESHOLD, eps=1e-7):
    """
    Computes Dice coefficient for a batch of predicted and target masks.
    """
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    total = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2 * intersection + eps) / (total + eps)
    return dice

def compute_map(preds, targets):
    """
    Computes mean average precision (mAP) per image over the batch.

    Returns:
        List of APs (one per image).
    """
    preds = torch.sigmoid(preds).cpu().numpy().reshape(preds.shape[0], -1)
    targets = (targets > 0.5).cpu().numpy().reshape(targets.shape[0], -1)
    aps = []
    for p, t in zip(preds, targets):
        if np.sum(t) == 0:
            continue  # skip empty targets to avoid undefined AP
        ap = average_precision_score(t, p)
        aps.append(ap)
    return aps

# --- TEST SET EVALUATION (as in trainer) ---
criterion = nn.BCEWithLogitsLoss()
model.eval()
test_loss = 0.0
ious = []
dices = []
maps_all = []

with torch.no_grad():
    for images, masks, region_ids in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, masks.float())
        test_loss += loss.item() * images.size(0)
        ious.append(compute_iou(outputs, masks).cpu())
        dices.append(compute_dice(outputs, masks).cpu())
        maps = compute_map(outputs, masks)
        maps_all.extend(maps)

test_loss /= len(test_loader.dataset)
mean_iou = torch.cat(ious).mean().item()
mean_dice = torch.cat(dices).mean().item()

if EnsembleVis == True:
    model = EnsembleModel(models_dir=EnsembleFolder, args=args, device=DEVICE)

        # ------------------------------ EVALUATE INDIVIDUAL MODELS IN ENSEMBLE ------------------------------
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    print("\nEvaluating individual models in ensemble...\n")
    num_models = len(model.models)
    losses, ious_all, dices_all, maps_all = [], [], [], []

    for i, submodel in enumerate(model.models):
        submodel.eval()
        sub_loss = 0.0
        ious = []
        dices = []
        maps = []

        with torch.no_grad():
            for images, masks, region_ids in test_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                outputs = submodel(images)
                loss = criterion(outputs, masks.float())
                sub_loss += loss.item() * images.size(0)
                ious.append(compute_iou(outputs, masks).cpu())
                dices.append(compute_dice(outputs, masks).cpu())
                batch_maps = compute_map(outputs, masks)
                maps.extend(batch_maps)

        avg_loss = sub_loss / len(test_loader.dataset)
        avg_iou = torch.cat(ious).mean().item()
        avg_dice = torch.cat(dices).mean().item()
        avg_map = np.mean(maps) if maps else float("nan")
        print(f"[Model {i+1:02d}] Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f} | mAP: {avg_map:.4f}")
        losses.append(avg_loss)
        ious_all.append(avg_iou)
        dices_all.append(avg_dice)
        maps_all.append(avg_map)

    mean_loss = np.mean(losses)
    mean_iou = np.mean(ious_all)
    mean_dice = np.mean(dices_all)
    mean_map = np.mean(maps_all) if maps_all else float("nan")

    print("\n=== MEAN METRICS ACROSS ENSEMBLE MODELS ===")
    print(f"Avg Loss:  {mean_loss:.4f}")
    print(f"Avg IoU:   {mean_iou:.4f}")
    print(f"Avg Dice:  {mean_dice:.4f}")
    print(f"Avg mAP:   {mean_map:.4f}")
    print("===========================================\n")

    # =============================================================================

# Scene Predictor
if ScenePred == True:
    def predict_scene_and_visualize(
        model,
        base_dataset,
        scene_path="/mnt/guanabana/raid/home/marin066/LitterLines dataset/20171009_Kikaki/0f25/20171009_153515_0f25_3B_AnalyticMS.tif",
        patch_size=256,
        device="cuda",
        threshold=THRESHOLD,
        save_path="SceneVis/scene_output.png"
    ):
        import rasterio

        model = EnsembleModel(models_dir=EnsembleFolder, args=args, device=DEVICE)

        with rasterio.open(scene_path) as src:
            full_img = src.read().astype(np.float32)  # (C, H, W)
            original_height, original_width = full_img.shape[1], full_img.shape[2]

        # Pad image to fit patch grid
        c, h, w = full_img.shape
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        full_img = np.pad(full_img, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        _, h, w = full_img.shape

        n_patches_h = h // patch_size
        n_patches_w = w // patch_size

        # Prepare empty output maps
        mean_probs_map = np.zeros((h, w), dtype=np.float32)
        std_probs_map = np.zeros((h, w), dtype=np.float32)
        binary_mask_map = np.zeros((h, w), dtype=np.uint8)
        full_gt_mask = np.zeros((h, w), dtype=np.uint8)

        model.eval()
        with torch.no_grad():
            patch_idx = 0
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    y = i * patch_size
                    x = j * patch_size

                    patch = full_img[:, y:y+patch_size, x:x+patch_size]
                    patch_tensor = torch.tensor(patch).unsqueeze(0).to(device)

                    mean_logits, std_logits = model(patch_tensor)
                    mean_prob = torch.sigmoid(mean_logits).cpu().squeeze().numpy()
                    std_map = std_logits.cpu().squeeze().numpy()
                    binary_pred = (mean_prob > threshold).astype(np.uint8)

                    mean_probs_map[y:y+patch_size, x:x+patch_size] = mean_prob
                    std_probs_map[y:y+patch_size, x:x+patch_size] = std_map
                    binary_mask_map[y:y+patch_size, x:x+patch_size] = binary_pred

                    # Load corresponding ground truth patch from dataset
                    if patch_idx < len(base_dataset):
                        _, mask_tensor, _ = base_dataset[patch_idx]
                        mask_np = mask_tensor.squeeze().numpy().astype(np.uint8)
                        full_gt_mask[y:y+patch_size, x:x+patch_size] = mask_np
                    patch_idx += 1

        # Remove padding to match original dimensions
        mean_probs_map = mean_probs_map[:original_height, :original_width]
        std_probs_map = std_probs_map[:original_height, :original_width]
        binary_mask_map = binary_mask_map[:original_height, :original_width]
        full_gt_mask = full_gt_mask[:original_height, :original_width]
        rgb = np.transpose(full_img[[2, 1, 0], :, :], (1, 2, 0))[:original_height, :original_width, :]

        # Normalize RGB for display
        rgb_norm = np.zeros_like(rgb)
        for c in range(3):
            channel = rgb[..., c]
            rgb_norm[..., c] = (channel - np.min(channel)) / (np.ptp(channel) + 1e-8)

        # Plot results in 2-row grid: 2 plots on top, 3 below
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        axs = axs.flatten()

        axs[0].imshow(rgb_norm)
        axs[0].set_title("Full Scene (RGB)")
        axs[0].axis("off")

        axs[1].imshow(full_gt_mask, cmap="gray", vmin=0, vmax=1)
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        axs[2].imshow(binary_mask_map, cmap="gray", vmin=0, vmax=1)
        axs[2].set_title(f"Predicted Mask (>{threshold:.2f})")
        axs[2].axis("off")

        im2 = axs[3].imshow(mean_probs_map, cmap="viridis", vmin=0, vmax=1)
        axs[3].set_title("Mean Probabilities")
        axs[3].axis("off")
        divider2 = make_axes_locatable(axs[3])
        cax2 = divider2.append_axes("bottom", size="4%", pad=0.05)
        plt.colorbar(im2, cax=cax2, orientation="horizontal")

        im3 = axs[4].imshow(std_probs_map, cmap="magma", vmin=0)
        axs[4].set_title("Per-pixel Std Dev (Uncertainty)")
        axs[4].axis("off")
        divider3 = make_axes_locatable(axs[4])
        cax3 = divider3.append_axes("bottom", size="4%", pad=0.05)
        plt.colorbar(im3, cax=cax3, orientation="horizontal")

        axs[5].axis("off")  # empty subplot for symmetry

        fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.06, wspace=0.1, hspace=0.25)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved scene visualization to: {save_path}")
            plt.close()
        else:
            plt.show()

# --- 5. Run Visualization Functions ---
print("Visualizing predictions from test set...")

if EnsembleVis == True:
    visualize_ensemble_predictions_with_uncertainty(model=model)
if ProbVis == True:
    visualize_predictions_with_probs()
if ScenePred == True:
    predict_scene_and_visualize(model, base_dataset=base_dataset, device=DEVICE)

print("Done!")