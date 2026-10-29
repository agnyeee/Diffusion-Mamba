"""Inspect and validate the Indian_30_split.mat metadata (no model execution)."""
import json
from data import load_indian_pines
from config import CFG

def main():
    cube, tr, te = load_indian_pines()
    report = {"dataset": CFG.dataset_name, "path": str(CFG.dataset_path), "input_shape": list(cube.shape), "TR_shape": list(tr.shape), "TE_shape": list(te.shape), "training_pixels": int((tr > 0).sum()), "test_pixels": int((te > 0).sum()), "classes_in_TR": sorted(int(x) for x in set(tr.ravel()) - {0}), "classes_in_TE": sorted(int(x) for x in set(te.ravel()) - {0}), "normalization": "per-band min-max to [-1,1] over label-free full cube"}
    print(json.dumps(report, indent=2))
    if report["training_pixels"] != 444 or report["test_pixels"] != 9805:
        raise ValueError(f"Provided split counts differ from expected 444/9805: {report['training_pixels']}/{report['test_pixels']}")
    return report

if __name__ == "__main__": main()
