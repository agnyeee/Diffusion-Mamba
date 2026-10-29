"""Train the supervised Mamba + CPSTG classifier from TR pixels only."""
import argparse, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from config import CFG, training_settings
from data import load_indian_pines, CoordinateDataset, get_split_coordinates
from mamba_classifier import DiffMambaClassifier

def load_features():
    maps=[]
    for t in CFG.feature_timesteps:
        for stage in CFG.feature_stages:
            path=CFG.feature_dir/f"t{t}_stage{stage}.npy"
            if not path.is_file(): raise FileNotFoundError(f"required diffusion feature missing: {path}; run feature_extract_uvit.py first")
            arr=np.load(path,mmap_mode="r")
            if arr.ndim!=3 or arr.shape[:2]!=(CFG.height,CFG.width): raise ValueError(f"feature map {path} must have shape (145,145,C), got {arr.shape}")
            maps.append(arr)
    return maps

def stratified_validation_indices(labels, coords, fraction=0.2, seed=CFG.seed):
    rng=np.random.default_rng(seed); train_ids=[]; val_ids=[]
    values=labels[coords[:,0],coords[:,1]]
    for cls in np.unique(values):
        ids=np.flatnonzero(values==cls); rng.shuffle(ids)
        n_val=min(max(1,int(round(len(ids)*fraction))),max(0,len(ids)-1)) if len(ids)>1 else 0
        val_ids.extend(ids[:n_val]); train_ids.extend(ids[n_val:])
    if not train_ids: raise ValueError("training split is empty after validation selection")
    return np.asarray(train_ids),np.asarray(val_ids)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("development","full"),default="full"); p.add_argument("--epochs",type=int); p.add_argument("--batch-size",type=int); p.add_argument("--amp",action="store_true"); p.add_argument("--resume",type=str); args=p.parse_args()
    settings=training_settings(args.mode); epochs=args.epochs or settings["classifier_epochs"]; bs=args.batch_size or settings["classifier_batch_size"]
    if epochs<1 or bs<1: raise ValueError("epochs and batch-size must be positive")
    random.seed(CFG.seed); np.random.seed(CFG.seed); torch.manual_seed(CFG.seed); device=torch.device(CFG.device)
    cube,tr,_=load_indian_pines(); coords=get_split_coordinates(tr); features=load_features()
    dataset=CoordinateDataset(cube,tr,coords,features)
    train_ids,val_ids=stratified_validation_indices(tr,coords)
    train_loader=DataLoader(Subset(dataset,train_ids),batch_size=bs,shuffle=True,num_workers=0,pin_memory=device.type=="cuda")
    val_loader=DataLoader(Subset(dataset,val_ids),batch_size=bs,shuffle=False,num_workers=0,pin_memory=device.type=="cuda") if len(val_ids) else None
    feature_channels=sum(x.shape[-1] for x in features)
    model=DiffMambaClassifier(feature_channels,CFG.bands,CFG.classifier_hidden,CFG.classes,CFG.classification_patch,CFG.mamba_layers,CFG.mamba_state).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=CFG.classifier_lr); criterion=torch.nn.CrossEntropyLoss()
    scaler=torch.amp.GradScaler("cuda",enabled=args.amp and device.type=="cuda")
    start=0; best=float("inf")
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); start=state["epoch"]+1; best=state.get("best_val_loss",best)
    CFG.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    conf={"mode":args.mode,"epochs":epochs,"batch_size":bs,"classes":CFG.classes,"mamba_layers":CFG.mamba_layers,"mamba_state":CFG.mamba_state,"hidden":CFG.classifier_hidden,"feature_timesteps":list(CFG.feature_timesteps),"feature_stages":list(CFG.feature_stages),"feature_channels":feature_channels,"train_count":len(train_ids),"validation_count":len(val_ids),"validation_source":"stratified holdout of TR only","amp":bool(args.amp)}
    (CFG.checkpoint_dir/"classifier_config.json").write_text(json.dumps(conf,indent=2),encoding="utf-8")
    for epoch in range(start,epochs):
        model.train(); total=0.; count=0
        for spectral,feat,y in train_loader:
            spectral=spectral.to(device,non_blocking=True); feat=feat.to(device=device,dtype=torch.float32,non_blocking=True); y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=args.amp and device.type=="cuda"):
                logits=model(spectral,feat); loss=criterion(logits,y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total+=float(loss.detach())*len(y); count+=len(y)
        train_loss=total/max(count,1); model.eval(); val_loss=0.; val_count=0; val_correct=0
        if val_loader:
            with torch.inference_mode():
                for spectral,feat,y in val_loader:
                    spectral=spectral.to(device); feat=feat.to(device=device,dtype=torch.float32); y=y.to(device)
                    logits=model(spectral,feat); val_loss+=float(criterion(logits,y))*len(y); val_count+=len(y); val_correct+=int((logits.argmax(1)==y).sum())
            val_loss/=max(val_count,1); val_acc=val_correct/max(val_count,1)
        else: val_loss=train_loss; val_acc=float("nan")
        print(f"epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_accuracy={val_acc:.4f}")
        state={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":conf,"best_val_loss":min(best,val_loss)}
        torch.save(state,CFG.checkpoint_dir/"classifier_latest.pt")
        if val_loss<best:
            best=val_loss; state["best_val_loss"]=best; torch.save(state,CFG.checkpoint_dir/"classifier_best.pt")

if __name__=="__main__": main()
