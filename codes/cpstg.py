"""Center Pixel-Driven Semantic Token Generator (paper Eqs. 13-20).

Features are weighted using a Gaussian over flattened spatial index distance
and a second Gaussian over feature distance from the center token. Their product
is normalized and used for the weighted semantic-token sum.
"""
import torch

class CPSTG(nn.Module):
    def __init__(self, dim, heads=None):
        super().__init__()
        if dim < 1: raise ValueError("CPSTG feature dimension must be positive")
        self.dim = dim

    def forward(self, tokens, center=None):
        """Return (B,D) semantic tokens for a row-major (B,H*W,D) sequence."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"CPSTG expects (B,N,{self.dim}), got {tuple(tokens.shape)}")
        b, n, d = tokens.shape
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError("CPSTG requires a square row-major patch sequence")
        center_index = n // 2
        center_feature = tokens[:, center_index:center_index + 1] if center is None else center.unsqueeze(1)
        if center_feature.shape != (b, 1, d):
            raise ValueError(f"center token must have shape (B,{d})")
        idx = torch.arange(n, device=tokens.device, dtype=tokens.dtype)
        mu_idx = (idx - center_index).abs().mean().clamp_min(torch.finfo(tokens.dtype).eps)
        weight_idx = torch.exp(-0.5 * ((idx - center_index) / mu_idx).square())
        weight_idx = weight_idx / weight_idx.sum().clamp_min(torch.finfo(tokens.dtype).eps)
        distance = torch.linalg.vector_norm(tokens - center_feature, ord=2, dim=-1)
        mu_feature = distance.mean(dim=1, keepdim=True).clamp_min(torch.finfo(tokens.dtype).eps)
        weight_feature = torch.exp(-0.5 * (distance / mu_feature).square())
        weight_feature = weight_feature / weight_feature.sum(dim=1, keepdim=True).clamp_min(torch.finfo(tokens.dtype).eps)
        weights = torch.softmax(weight_idx.unsqueeze(0) * weight_feature, dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)
