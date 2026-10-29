"""Evaluate the trained classifier exclusively on the supplied TE labels."""
import csv, json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from torch.utils.data import DataLoader
from config import CFG
from data import load_indian_pines, CoordinateDataset, get_split_coordinates, CLASS_NAMES
from train_classifier import load_features
from mamba_classifier import DiffMambaClassifier

def main():
    cube,tr,te=load_indian_pines(); coords=get_split_coordinates(te); features=load_features()
    dataset=CoordinateDataset(cube,te,coords,features)
    loader=DataLoader(dataset,batch_size=CFG.classifier_batch_size,shuffle=False,num_workers=0)
    state_path=CFG.checkpoint_dir/"classifier_best.pt"
    if not state_path.is_file(): raise FileNotFoundError(f"classifier checkpoint not found: {state_path}")
    state=torch.load(state_path,map_location=CFG.device,weights_only=False); cc=state["config"]
    model=DiffMambaClassifier(cc["feature_channels"],CFG.bands,CFG.classifier_hidden,CFG.classes,CFG.classification_patch,CFG.mamba_layers,CFG.mamba_state).to(CFG.device); model.load_state_dict(state["model"]); model.eval()
    actual=[]; predicted=[]
    with torch.inference_mode():
        for x,f,y in loader:
            logits=model(x.to(CFG.device),f.to(device=CFG.device,dtype=torch.float32)); predicted.extend(logits.argmax(1).cpu().numpy().tolist()); actual.extend(y.numpy().tolist())
    actual=np.asarray(actual); predicted=np.asarray(predicted); matrix=confusion_matrix(actual,predicted,labels=np.arange(CFG.classes)); counts=matrix.sum(1)
    per_class=np.divide(matrix.diagonal(),counts,out=np.full(CFG.classes,np.nan),where=counts>0); oa=float(np.mean(actual==predicted)); aa=float(np.nanmean(per_class)); kappa=float(cohen_kappa_score(actual,predicted,labels=np.arange(CFG.classes)))
    CFG.result_dir.mkdir(parents=True,exist_ok=True)
    report={"split":"TE","sample_count":len(actual),"overall_accuracy":oa,"average_accuracy":aa,"cohen_kappa":kappa,"per_class_accuracy":{CLASS_NAMES[i]:None if np.isnan(per_class[i]) else float(per_class[i]) for i in range(CFG.classes)},"confusion_matrix":matrix.tolist()}
    (CFG.result_dir/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (CFG.result_dir/"per_class_accuracy.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f); writer.writerow(["class_id","class_name","support","accuracy"])
        for i,name in enumerate(CLASS_NAMES): writer.writerow([i+1,name,int(counts[i]),"" if np.isnan(per_class[i]) else float(per_class[i])])
    np.savetxt(CFG.result_dir/"confusion_matrix.csv",matrix,delimiter=",",fmt="%d")
    print(json.dumps({"OA":oa,"AA":aa,"Kappa":kappa,"test_pixels":len(actual)},indent=2))

if __name__=="__main__": main()
