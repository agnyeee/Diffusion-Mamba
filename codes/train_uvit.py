"""Train the UViT epsilon-prediction diffusion denoiser."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import CFG, training_settings
from data import load_indian_pines, RandomDiffusionPatchDataset
from diffusion import Diffusion
from uvit3d import UViT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "full"), default="full")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--patches-per-epoch", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    settings = training_settings(args.mode)
    epochs = args.epochs or settings["diffusion_epochs"]
    batch_size = args.batch_size or settings["diffusion_batch_size"]
    patches_per_epoch = args.patches_per_epoch or settings["diffusion_patches_per_epoch"]
    if epochs <= 0 or batch_size <= 0 or patches_per_epoch <= 0:
        raise ValueError("epochs, batch size, and patches per epoch must be positive")
    random.seed(CFG.seed); np.random.seed(CFG.seed); torch.manual_seed(CFG.seed)
    device = torch.device(CFG.device)
    cube, _, _ = load_indian_pines()
    patch_dataset = RandomDiffusionPatchDataset(cube, CFG.diffusion_patch, patches_per_epoch)
    loader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    model = UViT(in_chans=1, num_spe=CFG.bands, patch_size=CFG.diffusion_patch,
                 depth=CFG.uvit_depth, num_heads=CFG.uvit_heads).to(device)
    diffusion = Diffusion(CFG.diffusion_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.diffusion_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = 0
    if args.resume:
        if not args.resume.is_file(): raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
    CFG.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"mode": args.mode, "epochs": epochs, "batch_size": batch_size, "patches_per_epoch": patches_per_epoch, "learning_rate": CFG.diffusion_lr,
                  "timesteps": CFG.diffusion_steps, "diffusion_patch": CFG.diffusion_patch, "bands": CFG.bands,
                  "uvit_depth": CFG.uvit_depth, "uvit_heads": CFG.uvit_heads,
                  "seed": CFG.seed, "device": str(device), "amp": bool(args.amp)}
    (CFG.checkpoint_dir / "diffusion_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, epochs):
        model.train(); losses=[]
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            t = torch.randint(0, diffusion.T, (batch.shape[0],), device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                loss, *_ = diffusion.get_loss(model, batch, t)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses))
        print(f"epoch {epoch + 1}/{epochs}: mse={mean_loss:.7f}")
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": run_config, "loss": mean_loss}
        torch.save(state, CFG.checkpoint_dir / "uvit3d_latest.pt")
        if (epoch + 1) % 100 == 0 or epoch + 1 == epochs:
            torch.save(state, CFG.checkpoint_dir / f"uvit3d_epoch_{epoch + 1:05d}.pt")

if __name__ == "__main__": main()
