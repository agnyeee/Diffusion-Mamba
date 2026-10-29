"""Central settings for the Indian Pines Diff-Mamba pipeline.

Paths are resolved from the repository root, independent of the current working
directory. Development settings are intentionally small and never selected
implicitly for a full training run.
"""
from dataclasses import dataclass
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Config:
    dataset_path: Path = ROOT / "data" / "Indian_30_split.mat"
    dataset_name: str = "Indian Pines (Indian_30_split)"
    bands: int = 200
    classes: int = 16
    height: int = 145
    width: int = 145
    diffusion_patch: int = 64
    classification_patch: int = 9
    diffusion_steps: int = 500
    diffusion_epochs: int = 20000
    diffusion_lr: float = 1e-4
    # A100 40 GB workstation target; paper reports 20 on an 80 GB A100.
    diffusion_batch_size: int = 2
    diffusion_patches_per_epoch: int = 512
    uvit_depth: int = 6
    uvit_heads: int = 8
    classifier_epochs: int = 30
    classifier_lr: float = 1e-3
    classifier_batch_size: int = 64
    classifier_hidden: int = 128
    mamba_layers: int = 1
    mamba_state: int = 16
    feature_timesteps: tuple = (5, 10, 50, 100, 200, 400)
    feature_stages: tuple = (0, 1, 2)
    seed: int = 2025
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = ROOT / "outputs" / "checkpoints"
    feature_dir: Path = ROOT / "outputs" / "features"
    result_dir: Path = ROOT / "outputs" / "results"

CFG = Config()

def training_settings(mode: str) -> dict:
    """Return explicit development or full settings; reject ambiguous modes."""
    if mode == "development":
        return {"diffusion_epochs": 2, "diffusion_batch_size": 1, "diffusion_patches_per_epoch": 4,
                "classifier_epochs": 2, "classifier_batch_size": 8}
    if mode == "full":
        return {"diffusion_epochs": CFG.diffusion_epochs,
                "diffusion_batch_size": CFG.diffusion_batch_size,
                "diffusion_patches_per_epoch": CFG.diffusion_patches_per_epoch,
                "classifier_epochs": CFG.classifier_epochs,
                "classifier_batch_size": CFG.classifier_batch_size}
    raise ValueError("mode must be explicitly 'development' or 'full'")
