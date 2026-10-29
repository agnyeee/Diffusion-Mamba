"""Generate full-scene 145x145 predictions and a labeled color map."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import CFG
from data import load_indian_pines, CoordinateDataset, CLASS_NAMES
from train_classifier import load_features
from mamba_classifier import DiffMambaClassifier

def main():
    cube,_,_=load_indian_pines(); features=load_features(); coords=np.indices((CFG.height,CFG.width)).reshape(2,-1).T
    unlabeled=np.zeros((CFG.height,CFG.width),dtype=np.int64)
    dataset=CoordinateDataset(cube,unlabeled,coords,features)
    loader=DataLoader(dataset,batch_size=CFG.classifier_batch_size,shuffle=False,num_workers=0)
    path=CFG.checkpoint_dir/"classifier_best.pt"
    if not path.is_file(): raise FileNotFoundError(f"classifier checkpoint not found: {path}")
    state=torch.load(path,map_location=CFG.device,weights_only=False); cc=state["config"]
    model=DiffMambaClassifier(cc["feature_channels"],CFG.bands,CFG.classifier_hidden,CFG.classes,CFG.classification_patch,CFG.mamba_layers,CFG.mamba_state).to(CFG.device); model.load_state_dict(state["model"]); model.eval()
    result=np.zeros(CFG.height*CFG.width,dtype=np.uint8)
    offset=0
    with torch.inference_mode():
        for x,f,_ in loader:
            pred=model(x.to(CFG.device),f.to(device=CFG.device,dtype=torch.float32)).argmax(1).cpu().numpy().astype(np.uint8)+1
            result[offset:offset+len(pred)]=pred; offset+=len(pred)
    result=result.reshape(CFG.height,CFG.width)
    CFG.result_dir.mkdir(parents=True,exist_ok=True); np.save(CFG.result_dir/"classification_map.npy",result)
    cmap=plt.get_cmap("tab20",CFG.classes)
    fig,ax=plt.subplots(figsize=(9,8)); image=ax.imshow(result,vmin=1,vmax=CFG.classes,cmap=cmap,interpolation="nearest")
    ticks=np.arange(CFG.classes)+1; bar=fig.colorbar(image,ax=ax,ticks=ticks,fraction=.046,pad=.04); bar.ax.set_yticklabels([f"{i+1}: {name}" for i,name in enumerate(CLASS_NAMES)],fontsize=8)
    ax.set_title("Indian Pines predicted classes"); ax.set_xlabel("Column"); ax.set_ylabel("Row"); fig.tight_layout(); fig.savefig(CFG.result_dir/"classification_map.png",dpi=200); plt.close(fig)
    print(f"saved {CFG.height}x{CFG.width} prediction map to {CFG.result_dir}")

if __name__=="__main__": main()
