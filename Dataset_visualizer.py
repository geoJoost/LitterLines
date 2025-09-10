import sys
import os
from torch.utils.data import DataLoader
from scripts.core.dataloader import LitterLinesDataset
import matplotlib.pyplot as plt
import numpy as np

# Add repo root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

Litterlinespath = r"/mnt/guanabana/raid/home/marin066/LitterLines dataset"  # Adjust if needed

dataset = LitterLinesDataset(Litterlinespath)
dataloader = DataLoader(dataset, batch_size=4)

i = 0 # Change dataset number (i) for different images from the dataset
idx = 0 # Change idx to see different patches in your batch.

patch, mask, regionid = dataset[i]

print(f"Patch shape: {patch.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Region ID: {regionid}")

def visualize_patch_and_mask(patch, mask, idx=idx, save_path=None):
    # patch: [N, 4, 256, 256], mask: [N, 256, 256]
    patch_np = patch[idx, :3].detach().cpu().numpy()  # (3, 256, 256)
    mask_np = mask[idx].detach().cpu().numpy()        # (256, 256)
    img = np.moveaxis(patch_np, 0, -1)                # (256, 256, 3)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img / img.max())
    plt.title("Patch (RGB)")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Mask")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    plt.show()

# Region visualization
def visualize_region_patches(dataset, region_name="KIKAKI", max_patches=20, save_dir="KIKAKI_visuals"):
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for idx in range(len(dataset)):
        patch, mask, regionid = dataset[idx]
        if regionid == region_name:
            save_path = os.path.join(save_dir, f"{regionid}_patch_{idx:03d}.png")
            visualize_patch_and_mask(patch, mask, save_path)
            count += 1
            if count >= max_patches:
                break
    if count == 0:
        print(f"No patches found for region '{region_name}'")
    else:
        print(f"Visualized {count} patches for region '{region_name}'.")

# Usage:
visualize_region_patches(dataset, region_name="KIKAKI", max_patches=20)

# Call this after you load patch and mask
# visualize_patch_and_mask(patch, mask, save_path= f"{regionid}{i:02d}{idx:02d}_patch_mask.png") # :02d means padded to two digits
