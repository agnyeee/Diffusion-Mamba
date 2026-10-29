"""Stage diffusion features and trainable classifier integration."""
import torch
from torch import nn

class FeatureFusion(nn.Module):
    """Concatenate channel-first aligned denoiser feature maps."""
    def forward(self, *features):
        if not features:
            raise ValueError("at least one diffusion feature map is required")
        spatial = features[0].shape[-2:]
        if any(x.ndim != 4 or x.shape[-2:] != spatial for x in features):
            raise ValueError("all diffusion feature maps must be BCHW and spatially aligned")
        return torch.cat(features, dim=1)
