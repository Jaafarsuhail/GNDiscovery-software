"""Validate distributable GNDiscovery source bundle without the external S3 workbook."""
from pathlib import Path
import json,hashlib,sys

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED=(
    "GNRetain221_Support_Assets/metadata/GNRetain221_training_rows.csv",
    "GNRetain221_Support_Assets/ad_reference/diagnostic_descriptor_reference.csv",
    "GNRetain221_Support_Assets/ad_reference/diagnostic_descriptor_quantiles.csv",
    "GNRetain221_Support_Assets/ad_reference/diagnostic_pca_reference.csv",
    "GNRetain221_Support_Assets/ad_reference/focused_descriptor_reference.csv",
    "GNRetain221_Support_Assets/ad_reference/focused_descriptor_quantiles.csv",
    "GNRetain221_Support_Assets/ad_reference/focused_pca_reference.csv",
)
EXPECTED_MODEL_HASH="7b9bfb19ab866eb624d0d16273f9ac3af1c0ffa9b20d2f79f1b11d316e8b9258"

def main():
    if any((ROOT/p).exists() for p in EXCLUDED):
        raise AssertionError("GNRetain reference rows were included in the distribution: "+", ".join(p for p in EXCLUDED if (ROOT/p).exists()))
    saved=json.loads((ROOT/"PUBLIC_PACKAGE_SHA256.json").read_text(encoding="utf8"))
    actual={str(p.relative_to(ROOT)).replace("\\","/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob("*") if p.is_file() and not any(part in (".git","__pycache__") for part in p.relative_to(ROOT).parts) and p.name!="PUBLIC_PACKAGE_SHA256.json"}
    if saved!=actual:
        missing=sorted(set(saved)-set(actual));extra=sorted(set(actual)-set(saved))
        mismatched=sorted(p for p in set(actual)&set(saved) if actual[p]!=saved[p])
        raise AssertionError(f"Manifest mismatch: missing={missing}, extra={extra}, changed={mismatched}")
    meta=json.loads((ROOT/"GNRetain221_Runtime_Model/model/GNRetain221_model_metadata.json").read_text())
    assert (meta['training_n'],meta['n_features'],meta['ensemble_members'],meta['threshold'])==(5452,221,5,0.575)
    assert actual['GNRetain221_Runtime_Model/model/GNRetain221_LightGBM_5seed_ensemble.joblib']==EXPECTED_MODEL_HASH
    assert (ROOT/"scripts/rehydrate_gnretain_from_s3.py").exists()
    assert (ROOT/"data_manifests/gnretain_s3_row_ids.txt").exists()
    assert (ROOT/"LICENSE").exists() and (ROOT/"DATA_LICENSE.md").exists()
    print(f"PASS: {len(saved)} tracked payload files checked; seven retention reference files excluded; GNRetain=5452/221/5/0.575")

if __name__=="__main__": main()
