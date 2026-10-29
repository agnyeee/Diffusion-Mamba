"""Diffusion feature + local spectral-spatial Mamba classifier."""
import torch
from torch import nn
from cpstg import CPSTG

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError("mamba-ssm is required for the paper's Mamba classifier; install a build compatible with the DGX PyTorch/CUDA stack (see README).") from exc

class DiffMambaClassifier(nn.Module):
    def __init__(self, feature_channels, bands=200, hidden=128, classes=16, patch_size=9, mamba_layers=1, mamba_state=16):
        super().__init__()
        if feature_channels < 1:
            raise ValueError("feature_channels must be positive")
        self.patch_size, self.bands, self.feature_channels = patch_size, bands, feature_channels
        self.spectral_embed = nn.Sequential(nn.Conv2d(bands, hidden, 1), nn.BatchNorm2d(hidden), nn.GELU())
        self.diffusion_embed = nn.Sequential(nn.Conv2d(feature_channels, hidden, 1), nn.BatchNorm2d(hidden), nn.GELU())
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
                                      batch_first=True, norm_first=True), num_layers=1)
        # Dynamic positional embedding approximating the paper's depthwise
        # convolution after projecting high-dimensional diffusion features.
        self.dynamic_position = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.mamba = nn.ModuleList([Mamba(d_model=hidden, d_state=mamba_state, d_conv=4, expand=2) for _ in range(mamba_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(mamba_layers)])
        self.cpstg = CPSTG(hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, classes))

    def forward(self, spectral_patch, diffusion_patch):
        if spectral_patch.ndim != 4 or diffusion_patch.ndim != 4:
            raise ValueError("classifier inputs must be BCHW tensors")
        if spectral_patch.shape[1] != self.bands or diffusion_patch.shape[1] != self.feature_channels:
            raise ValueError(f"expected {self.bands} spectral and {self.feature_channels} diffusion channels")
        if spectral_patch.shape[0] != diffusion_patch.shape[0] or spectral_patch.shape[-2:] != diffusion_patch.shape[-2:]:
            raise ValueError("spectral and diffusion patches must have equal batch and spatial dimensions")
        if spectral_patch.shape[-2:] != (self.patch_size, self.patch_size):
            raise ValueError(f"expected {self.patch_size}x{self.patch_size} patches")
        b, _, h, w = spectral_patch.shape
        spectral_tokens = self.spectral_embed(spectral_patch).flatten(2).transpose(1, 2)
        spectral_tokens = self.transformer(spectral_tokens)
        local = self.diffusion_embed(diffusion_patch)
        local = local + self.dynamic_position(local)
        tokens = local.flatten(2).transpose(1, 2)
        for block, norm in zip(self.mamba, self.norms):
            tokens = tokens + block(norm(tokens))
        center = tokens[:, (h // 2) * w + w // 2]
        semantic = self.cpstg(tokens, center)
        global_feature = tokens.mean(dim=1)
        spatial_semantic = spectral_tokens.mean(dim=1)
        return self.classifier(torch.cat((semantic, global_feature, spatial_semantic), dim=-1))
