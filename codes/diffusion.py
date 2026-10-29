"""DDPM noise schedule and stable tensor-batched sampling helpers."""
import torch
import torch.nn.functional as F

class Diffusion:
    def __init__(self, T=500, beta_start=1e-4, beta_end=2e-2):
        if T < 2:
            raise ValueError("diffusion timesteps T must be at least 2")
        self.T = int(T)
        self.betas = torch.linspace(beta_start, beta_end, self.T, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.rsqrt(self.alphas)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).clamp_min(1e-20).sqrt()
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod).clamp_min(1e-20)).clamp_min(0.0)

    @staticmethod
    def _extract(values, t, x):
        values = values.to(device=x.device, dtype=x.dtype)
        t = t.to(device=x.device, dtype=torch.long)
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"timestep tensor must have shape ({x.shape[0]},), got {tuple(t.shape)}")
        if torch.any((t < 0) | (t >= values.numel())):
            raise ValueError("timestep index is outside the configured diffusion schedule")
        return values.gather(0, t).view(x.shape[0], *([1] * (x.ndim - 1)))

    def forward_diffusion_sample(self, x0, t, device=None, noise=None):
        if x0.ndim != 5:
            raise ValueError("diffusion input must be (batch, channel, bands, height, width)")
        x0 = x0.to(device=device or x0.device)
        noise = torch.randn_like(x0) if noise is None else noise.to(x0)
        a = self._extract(self.sqrt_alphas_cumprod, t, x0)
        b = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0)
        return a * x0 + b * noise, noise

    def get_loss(self, model, x0, t):
        noisy, noise = self.forward_diffusion_sample(x0, t)
        prediction = model(noisy, t)
        if prediction.shape != noise.shape:
            raise ValueError(f"noise prediction shape {tuple(prediction.shape)} != target {tuple(noise.shape)}")
        # Standard DDPM epsilon objective is mean squared error.
        return F.mse_loss(prediction.float(), noise.float()), noisy, noise, prediction

    @torch.no_grad()
    def sample_timestep(self, model, x, t):
        beta = self._extract(self.betas, t, x)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x)
        sqrt_recip = self._extract(self.sqrt_recip_alphas, t, x)
        mean = sqrt_recip * (x - beta * model(x, t) / sqrt_one_minus.clamp_min(1e-12))
        variance = self._extract(self.posterior_variance, t, x)
        noise = torch.randn_like(x)
        nonzero = (t != 0).to(x.dtype).view(x.shape[0], *([1] * (x.ndim - 1)))
        return mean + nonzero * variance.sqrt() * noise

    @torch.no_grad()
    def reconstruct(self, model, xt, tempT=None, num=5, from_noise=False, shape=None):
        if num <= 0:
            raise ValueError("num must be positive")
        device = next(model.parameters()).device
        if from_noise:
            if shape is None:
                raise ValueError("shape is required when sampling from noise")
            x = torch.randn(shape, device=device)
        else:
            if xt is None:
                raise ValueError("xt is required unless from_noise=True")
            x = xt.to(device)
        last = self.T - 1 if tempT is None else int(torch.as_tensor(tempT).max().item())
        if not 0 <= last < self.T:
            raise ValueError(f"tempT must be in [0,{self.T - 1}]")
        indices = set(torch.linspace(0, last, min(num, last + 1), dtype=torch.long).tolist())
        snapshots = []
        for i in range(last, -1, -1):
            t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
            x = self.sample_timestep(model, x, t)
            if i in indices or i == 0:
                snapshots.append(x.detach().cpu())
        return list(range(last, -1, -1)), snapshots
