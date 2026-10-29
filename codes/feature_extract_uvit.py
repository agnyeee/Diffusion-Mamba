"""Extract frozen UViT feature maps at configured diffusion timesteps."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from config import CFG
from data import load_indian_pines, diffusion_tiles
from diffusion import Diffusion
from uvit3d import UViT

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CFG.checkpoint_dir / "uvit3d_latest.pt")
    parser.add_argument("--batch-size", type=int, default=1)
    args=parser.parse_args()
    if not args.checkpoint.is_file(): raise FileNotFoundError(f"diffusion checkpoint not found: {args.checkpoint}")
    if args.batch_size < 1: raise ValueError("batch-size must be positive")
    device=torch.device(CFG.device)
    torch.manual_seed(CFG.seed)
    cube,_,_=load_indian_pines(); tiles, (rows,cols,h,w)=diffusion_tiles(cube)
    state=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=UViT(in_chans=1,num_spe=CFG.bands,patch_size=CFG.diffusion_patch,
               depth=CFG.uvit_depth,num_heads=CFG.uvit_heads).to(device)
    model.load_state_dict(state.get("model",state)); model.eval()
    diffusion=Diffusion(CFG.diffusion_steps)
    CFG.feature_dir.mkdir(parents=True,exist_ok=True)
    stage_count=len(model.out_blocks)
    for timestep in CFG.feature_timesteps:
        if timestep >= diffusion.T: raise ValueError(f"feature timestep {timestep} must be below {diffusion.T}")
        stage_buffers=[[] for _ in range(stage_count)]
        for start in range(0,len(tiles),args.batch_size):
            x=torch.from_numpy(tiles[start:start+args.batch_size]).to(device)
            t=torch.full((len(x),),timestep,device=device,dtype=torch.long)
            noisy,_=diffusion.forward_diffusion_sample(x,t)
            with torch.inference_mode():
                _=model(noisy,t,feature=True)
                stages=model.return_features()
            if len(stages)!=stage_count: raise RuntimeError("UViT feature stage count mismatch")
            for idx,feat in enumerate(stages):
                fmap=feat.reshape(len(x),CFG.diffusion_patch,CFG.diffusion_patch,-1).cpu().numpy()
                stage_buffers[idx].append(fmap)
        for stage_idx,parts in enumerate(stage_buffers):
            tiled=np.concatenate(parts,axis=0).reshape(rows,cols,CFG.diffusion_patch,CFG.diffusion_patch,-1)
            scene=tiled.transpose(0,2,1,3,4).reshape(rows*CFG.diffusion_patch,cols*CFG.diffusion_patch,-1)[:h,:w]
            out=CFG.feature_dir / f"t{timestep}_stage{stage_idx}.npy"
            np.save(out,scene.astype(np.float16))
            print(f"saved {out} shape={scene.shape}")
    metadata={"height":h,"width":w,"timesteps":list(CFG.feature_timesteps),"stages":stage_count,"storage_dtype":"float16"}
    (CFG.feature_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

if __name__ == "__main__": main()
