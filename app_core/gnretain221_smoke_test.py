from __future__ import annotations

import argparse
import json
import py_compile

import numpy as np
import pandas as pd

from focused_visualization import focused_query_values, transform_query_to_focused_pca
from gnretain_source_feature_generator import (
    EXPECTED_SOURCE_FEATURE_COUNT,
    batch_generate_runtime_source_features,
    check_source_schema_compatibility,
    find_project_root,
)
from gnretain221_ad_diagnostics import applicability_domain_summary, molecular_diagnostics, simple_markdown_report
from gnretain221_prediction_core import (
    EXPECTED_MODEL_FEATURE_COUNT,
    inspect_model_bundles,
    load_feature_panel,
    load_model_artifact,
    load_model_metadata,
    predict_from_features,
)

AMOXICILLIN = "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def add_result(rows, name, function):
    print(f"RUN {name}", flush=True)
    try:
        details = function()
        rows.append({"test": name, "status": "passed", "details": str(details)[:1500]})
        print(f"PASS {name}", flush=True)
    except Exception as exc:
        rows.append({"test": name, "status": "failed", "details": f"{type(exc).__name__}: {exc}"})
        print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    root = find_project_root(args.root)
    rows: list[dict] = []
    prepared: dict = {}

    def source_schema_test():
        check = check_source_schema_compatibility(root)
        assert check["source_schema_order_valid"]
        assert int(check["source_feature_count"]) == EXPECTED_SOURCE_FEATURE_COUNT
        return check

    def source_generation_preflight():
        smiles = [ASPIRIN, AMOXICILLIN, "CCO", "this_is_not_smiles"]
        X, reports = batch_generate_runtime_source_features(smiles, root)
        assert X.shape == (3, EXPECTED_SOURCE_FEATURE_COUNT)
        assert reports["status"].tolist() == ["ok", "ok", "ok", "failed"]
        prepared["X"] = X
        prepared["reports"] = reports
        return {"source_shape": X.shape, "statuses": reports["status"].tolist()}

    def model_artifact_test():
        panel = load_feature_panel(str(root))
        artifact = load_model_artifact(str(root))
        metadata = load_model_metadata(str(root))
        assert len(panel) == EXPECTED_MODEL_FEATURE_COUNT
        assert list(artifact["features"]) == panel == list(metadata["features"])
        assert np.isclose(float(artifact["threshold"]), float(metadata["threshold"]))
        bundles = inspect_model_bundles(root)
        assert len(bundles) == 1 and bundles["status"].eq("ok").all()
        return {"model_features": len(panel), "threshold": float(artifact["threshold"])}

    def prediction_from_runtime_features_test():
        X = prepared["X"]
        pred = predict_from_features(X, root)
        assert len(pred) == 3
        probabilities = pred["gnretention__proba_retention_positive"].astype(float)
        assert probabilities.between(0, 1).all()
        prepared["pred"] = pred
        return {
            "aspirin_probability": float(probabilities.iloc[0]),
            "amoxicillin_probability": float(probabilities.iloc[1]),
            "ethanol_probability": float(probabilities.iloc[2]),
        }

    def ad_test():
        X_amox = prepared["X"].iloc[[1]].reset_index(drop=True)
        runtime_smiles = prepared["reports"].iloc[1]["runtime_smiles"]
        summary, neighbors, flags = applicability_domain_summary(X_amox, runtime_smiles, project_root=root)
        assert len(flags) == EXPECTED_MODEL_FEATURE_COUNT
        assert flags["feature"].tolist() == load_feature_panel(str(root))
        assert int(summary["ad_features_evaluated"]) == EXPECTED_MODEL_FEATURE_COUNT
        assert len(neighbors) > 0
        prepared["amox_diag"] = molecular_diagnostics(runtime_smiles)
        return {"summary": summary, "flag_counts": flags["severity"].value_counts().to_dict()}

    def focused_visualization_test():
        X_amox = prepared["X"].iloc[[1]].reset_index(drop=True)
        diag = prepared["amox_diag"]
        ref_dir = root / "GNRetain221_Support_Assets" / "ad_reference"
        meta = json.loads((ref_dir / "focused_pca_metadata.json").read_text(encoding="utf-8"))
        values = focused_query_values(X_amox, diag)
        coords = transform_query_to_focused_pca(X_amox, diag, meta)
        assert set(meta["properties"]).issubset(values)
        assert coords is not None and np.all(np.isfinite(coords))
        focused_ref = pd.read_csv(ref_dir / "focused_descriptor_reference.csv")
        focused_pca = pd.read_csv(ref_dir / "focused_pca_reference.csv")
        assert len(focused_ref) == 5452 and len(focused_pca) == 5452
        return {"query_values": values, "focused_coordinates": coords, "reference_rows": len(focused_ref)}

    def batch_alignment_test():
        reports = prepared["reports"]
        assert reports["status"].tolist() == ["ok", "ok", "ok", "failed"]
        assert len(prepared["pred"]) == 3
        return {"valid_prediction_rows": 3, "failed_input_rows": 1}

    def report_test():
        report = simple_markdown_report({"probability": 0.5, "ad_features_evaluated": 221})
        assert report.startswith("# GNRetain Single-Molecule Prediction Report")
        assert "ad_features_evaluated" in report
        return report.splitlines()[0]

    def reference_integrity_test():
        ref_dir = root / "GNRetain221_Support_Assets" / "ad_reference"
        ranges = pd.read_csv(ref_dir / "GNRetain221_feature_range_quantiles.csv")
        policy = json.loads((ref_dir / "GNRetain221_ad_policy.json").read_text(encoding="utf-8"))
        assert len(ranges) == 221
        assert ranges["feature"].astype(str).tolist() == load_feature_panel(str(root))
        assert int(policy["n_reference"]) == 5452
        assert "221-feature" in policy["feature_space"]
        return {"range_rows": len(ranges), "reference_n": policy["n_reference"]}

    def ui_compile_test():
        app_file = root / "streamlit_app" / "gndiscovery_app.py"
        py_compile.compile(str(app_file), doraise=True)
        source = app_file.read_text(encoding="utf-8")
        required = [
            "GNDiscovery: E. coli",
            "Choose a prediction workflow",
            "E. coli ΔtolC-associated retention",
            "E. coli ATCC 25922 MIC activity",
            "Integrated Assessment",
            "Support / Applicability",
            "Prediction Support",
            "Retention Analogues",
            "Training Analogues",
            "render_retention_analogues",
            "render_mic_analogues",
            "Accuracy for this support profile",
        ]
        missing = [text for text in required if text not in source]
        assert not missing, f"Missing UI elements: {missing}"
        forbidden = [
            "Phase09_Motif389_Runtime_Model",
            "motif389_ad_diagnostics",
            "motif389_feature_generator",
            "ChEMBL active chemical space",
            "Simple mode",
            "Advanced mode",
        ]
        present = [text for text in forbidden if text in source]
        assert not present, f"Legacy app identifiers remain: {present}"
        return {"compiled": True, "required_ui_strings": len(required)}

    for name, function in [
        ("source_schema_389", source_schema_test),
        ("source_feature_generation_preflight", source_generation_preflight),
        ("model_schema_artifact_221", model_artifact_test),
        ("prediction_from_runtime_features", prediction_from_runtime_features_test),
        ("applicability_domain_221_features", ad_test),
        ("focused_ionization_solubility_visualization", focused_visualization_test),
        ("batch_invalid_row_alignment", batch_alignment_test),
        ("reference_integrity", reference_integrity_test),
        ("markdown_report", report_test),
        ("streamlit_ui_compile", ui_compile_test),
    ]:
        add_result(rows, name, function)

    results = pd.DataFrame(rows)
    output = root / "GNRetain221_Support_Assets" / "validation" / "GNRetain221_smoke_test_report.csv"
    results.to_csv(output, index=False)
    print(results.to_string(index=False))
    return 0 if results["status"].eq("passed").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
