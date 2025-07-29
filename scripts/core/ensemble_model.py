import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from scripts.segmentation_model import SegmentationModel

class EnsembleModel(nn.Module):
    def __init__(self, models_dir, args, device):
        """
        Loads all model checkpoints from a directory and wraps them in an ensemble.

        Args:
            models_dir (str): Directory containing .pth model checkpoints.
            args (SimpleNamespace): Arguments used to construct base model.
            device (torch.device): CPU or CUDA.
        """
        super().__init__()
        self.device = device
        self.models = nn.ModuleList()

        checkpoints = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".pth")]
        if not checkpoints:
            raise ValueError(f"No .pth checkpoints found in {models_dir}")

        print(f"Found {len(checkpoints)} model checkpoints in {models_dir}")

        for ckpt_path in checkpoints:
            base_model = smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=4,
                classes=1,
            )
            model = SegmentationModel(args)
            model.model = base_model
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            self.models.append(model.to(device))
            print(f"Loaded: {ckpt_path}")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input batch of shape (B, C, H, W)

        Returns:
            mean_output (torch.Tensor): Mean of model outputs (B, 1, H, W)
            std_output (torch.Tensor): Standard deviation of model outputs (B, 1, H, W)
        """
        with torch.no_grad():
            outputs = [model(x) for model in self.models]
            stacked = torch.stack(outputs, dim=0)  # (N_models, B, 1, H, W)
            mean_output = stacked.mean(dim=0)
            std_output = stacked.std(dim=0, unbiased=False)
            return mean_output, std_output