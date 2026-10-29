"""Indian Pines MAT loading and coordinate-preserving patch datasets."""
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from config import CFG

CLASS_NAMES = ("Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers")


def load_indian_pines(path=CFG.dataset_path):
    """Load input/TR/TE and validate the supplied split without changing it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Indian Pines MAT file not found: {path}")
    try:
        mat = sio.loadmat(path)
    except Exception as exc:
        raise ValueError(f"Could not read MAT file {path}: {exc}") from exc
    missing = {"input", "TR", "TE"} - mat.keys()
    if missing:
        raise KeyError(f"MAT file {path} is missing variables: {sorted(missing)}")
    cube = np.asarray(mat["input"], dtype=np.float32)
    train, test = np.asarray(mat["TR"]), np.asarray(mat["TE"])
    if cube.shape != (CFG.height, CFG.width, CFG.bands):
        raise ValueError(f"input must have shape {(CFG.height, CFG.width, CFG.bands)}, got {cube.shape}")
    if train.shape != (CFG.height, CFG.width) or test.shape != train.shape:
        raise ValueError(f"TR and TE must both have shape {(CFG.height, CFG.width)}; got {train.shape} and {test.shape}")
    for name, labels in (("TR", train), ("TE", test)):
        if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
            raise ValueError(f"{name} labels must be finite numeric values")
        if np.any(labels < 0) or np.any(labels > CFG.classes) or not np.equal(labels, np.floor(labels)).all():
            raise ValueError(f"{name} labels must be integers in [0, {CFG.classes}]")
    if np.any((train > 0) & (test > 0)):
        raise ValueError("TR and TE overlap; refusing supervised split leakage")
    # Per-band min-max scaling on the full unlabeled cube is transductive and
    # label-free; the same scaled cube is used by diffusion and classification.
    lo = cube.min(axis=(0, 1), keepdims=True)
    hi = cube.max(axis=(0, 1), keepdims=True)
    span = hi - lo
    cube = np.divide(cube - lo, span, out=np.zeros_like(cube), where=span > 0)
    cube = np.where(span > 0, cube * 2.0 - 1.0, 0.0)
    return cube, train.astype(np.int64), test.astype(np.int64)


def pad_cube(cube, pad):
    if pad <= 0:
        return cube
    mode = "reflect" if cube.shape[0] > pad and cube.shape[1] > pad else "edge"
    return np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode=mode)


def center_patches(cube, coordinates, size=CFG.classification_patch):
    """Extract HWC patches centered on (row,col), preserving coordinate order."""
    if size < 1 or size % 2 != 1:
        raise ValueError("classification patch size must be a positive odd integer")
    margin = size // 2
    padded = pad_cube(cube, margin)
    if len(coordinates) == 0:
        return np.empty((0, size, size, cube.shape[-1]), dtype=np.float32)
    patches = np.stack([padded[r:r + size, c:c + size] for r, c in coordinates])
    return np.ascontiguousarray(patches, dtype=np.float32)


def diffusion_tiles(cube, patch_size=CFG.diffusion_patch):
    """Cover full scene with non-overlapping tiles; return crop metadata."""
    h, w, bands = cube.shape
    rows, cols = (h + patch_size - 1) // patch_size, (w + patch_size - 1) // patch_size
    out_h, out_w = rows * patch_size, cols * patch_size
    bottom, right = out_h - h, out_w - w
    padded = np.pad(cube, ((0, bottom), (0, right), (0, 0)), mode="reflect")
    tiles = np.stack([padded[r:r+patch_size, c:c+patch_size] for r in range(0, out_h, patch_size) for c in range(0, out_w, patch_size)])
    # UViT convention: (N,1,bands,H,W).
    tiles = tiles.transpose(0, 3, 1, 2)[:, None]
    return np.ascontiguousarray(tiles), (rows, cols, h, w)


class ArrayDataset(Dataset):
    def __init__(self, values):
        self.values = torch.from_numpy(np.asarray(values, dtype=np.float32))
    def __len__(self):
        return len(self.values)
    def __getitem__(self, index):
        return self.values[index]


class CoordinateDataset(Dataset):
    def __init__(self, cube, labels, coordinates, feature_maps=None, patch_size=CFG.classification_patch):
        self.coords = np.asarray(coordinates, dtype=np.int64)
        self.patches = torch.from_numpy(center_patches(cube, self.coords, patch_size).transpose(0, 3, 1, 2))
        self.labels = torch.from_numpy(labels[self.coords[:, 0], self.coords[:, 1]].astype(np.int64) - 1)
        self.features = None
        self.padded_features = None
        if feature_maps is not None:
            # Keep diffusion maps in half precision on host memory until batching.
            combined = np.concatenate(feature_maps, axis=-1).astype(np.float16, copy=False)
            self.padded_features = torch.from_numpy(np.pad(combined, ((patch_size // 2, patch_size // 2),
                (patch_size // 2, patch_size // 2), (0, 0)), mode="edge")).permute(2, 0, 1).contiguous()
    def __len__(self):
        return len(self.coords)
    def __getitem__(self, index):
        r, c = self.coords[index]
        if self.features is None:
            return self.patches[index], self.labels[index]
        # Feature patch extracted from a spatially padded feature map.
        margin = self.patches.shape[-1] // 2
        fm = self.padded_features
        feature_patch = fm[:, r:r+2*margin+1, c:c+2*margin+1]
        return self.patches[index], feature_patch, self.labels[index]


def get_split_coordinates(labels):
    return np.argwhere(labels > 0)
class RandomDiffusionPatchDataset(Dataset):
    """On-demand unlabeled spatial crops; avoids materializing overlapping 64x64 patches."""
    def __init__(self, cube, patch_size=CFG.diffusion_patch, samples_per_epoch=512):
        if patch_size < 1 or samples_per_epoch < 1:
            raise ValueError("patch_size and samples_per_epoch must be positive")
        self.cube = np.asarray(cube, dtype=np.float32)
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.padded = pad_cube(self.cube, self.patch_size // 2)
    def __len__(self):
        return self.samples_per_epoch
    def __getitem__(self, index):
        h, w = self.cube.shape[:2]
        row = np.random.randint(0, h)
        col = np.random.randint(0, w)
        p = self.patch_size
        patch = self.padded[row:row+p, col:col+p]
        return torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)[None], dtype=np.float32))
