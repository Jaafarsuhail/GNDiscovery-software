"""Verify the released GNRetain inference model, reference and example predictions."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import platform

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app_core"))
from gnretain221_prediction_core import load_feature_panel, load_model_artifact, load_model_metadata, predict_from_features
from gnretain_source_feature_generator import batch_generate_runtime_source_features

EXPECTED_SHA256 = "7b9bfb19ab866eb624d0d16273f9ac3af1c0ffa9b20d2f79f1b11d316e8b9258"
EXPECTED_PROBS_BY_PLATFORM = {
    "Linux": [0.9047532268646592, 0.924851425558533, 0.08511242417117204],
    "Windows": [0.9040012047333228, 0.9145038376370949, 0.08511242417117204],
}
REQUIRED_REFERENCE_FILES = [
    "GNRetain221_Support_Assets/metadata/GNRetain221_training_rows.csv",
    "GNRetain221_Support_Assets/ad_reference/diagnostic_descriptor_reference.csv",
    "GNRetain221_Support_Assets/ad_reference/diagnostic_descriptor_quantiles.csv",
    "GNRetain221_Support_Assets/ad_reference/diagnostic_pca_reference.csv",
    "GNRetain221_Support_Assets/ad_reference/focused_descriptor_reference.csv",
    "GNRetain221_Support_Assets/ad_reference/focused_descriptor_quantiles.csv",
    "GNRetain221_Support_Assets/ad_reference/focused_pca_reference.csv",
]

def main():
    model_path = ROOT / "GNRetain221_Runtime_Model/model/GNRetain221_LightGBM_5seed_ensemble.joblib"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, f"GNRetain estimator changed: {digest}"
    meta = load_model_metadata(str(ROOT))
    artifact = load_model_artifact(str(ROOT))
    ordered = load_feature_panel(str(ROOT))
    assert (int(meta["training_n"]), int(meta["n_features"]), int(meta["ensemble_members"])) == (5452,221,5)
    assert len(ordered)==221 and list(artifact["features"]) == ordered
    assert len(artifact["models"])==5 and list(artifact["seeds"])==[42,52,62,72,82]
    assert np.isclose(float(artifact["threshold"]), .575, rtol=0, atol=1e-12)
    assert np.isclose(float(meta["threshold"]), .575, rtol=0, atol=1e-12)
    missing=[x for x in REQUIRED_REFERENCE_FILES if not (ROOT/x).is_file()]
    if missing:
        raise RuntimeError("First rebuild the 5,452 retention reference from original Supplementary Table S3. Missing: "+", ".join(missing))
    ref = pd.read_csv(ROOT/REQUIRED_REFERENCE_FILES[0])
    assert len(ref)==5452, f"Expected 5,452 GNRetain training references, found {len(ref)}"
    source_smiles=[
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
        "CCO",
        "invalid_$$",
    ]
    X,reports=batch_generate_runtime_source_features(source_smiles,project_root=ROOT)
    assert X.shape==(3,389)
    assert reports["status"].tolist()==["ok","ok","ok","failed"]
    result=predict_from_features(X,project_root=ROOT)
    vals=result["gnretention__proba_retention_positive"].astype(float).to_numpy()
    platform_name = platform.system()
    expected = EXPECTED_PROBS_BY_PLATFORM.get(platform_name)
    if expected is not None:
        assert np.allclose(vals,expected,rtol=0,atol=2e-6), f"Frozen model predictions differ on {platform_name}: {vals}"
    else:
        print(f"Note: no frozen probability fixture for {platform_name}; model and class-label checks still apply")
    assert result["gnretention__threshold"].eq(.575).all()
    binary=result["gnretention__binary_class"].to_numpy(dtype=int)
    assert np.array_equal(binary,(vals>=.575).astype(int))
    assert result["gnretention__class_label"].tolist()==["RP","RP","RN"]
    print(json.dumps({"status":"PASS","gnretain_training_compounds":5452,"features":221,"models":5,"threshold":.575,"retention_reference_rows":len(ref),"reference_predictions":vals.tolist()},indent=2))

if __name__=="__main__":main()
