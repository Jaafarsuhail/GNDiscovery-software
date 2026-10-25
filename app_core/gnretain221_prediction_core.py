from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

from gnretain_source_feature_generator import (  # noqa: E402
    batch_generate_runtime_source_features,
    find_project_root,
    generate_runtime_source_features,
)

PUBLIC_ENDPOINT_KEY = "gnretention_v05"
PUBLIC_ENDPOINT_LABEL = "Gram-negative retention"
ENDPOINT_LABELS = {PUBLIC_ENDPOINT_KEY: PUBLIC_ENDPOINT_LABEL}
DEPLOY_MEMBERS = ["lgbm_seed42", "lgbm_seed52", "lgbm_seed62", "lgbm_seed72", "lgbm_seed82"]
DIAGNOSTIC_MEMBERS: list[str] = []
MODEL_NAME = "GNRetain cleaned-221 five-seed LightGBM ensemble"
FEATURE_SET_NAME = "GNRetain cleaned 221-feature panel"
EXPECTED_MODEL_FEATURE_COUNT = 221
EXPECTED_ENSEMBLE_MEMBERS = 5


def runtime_root(project_root=None) -> Path:
    return find_project_root(project_root) / "GNRetain221_Runtime_Model"


def model_path(project_root=None) -> Path:
    return runtime_root(project_root) / "model" / "GNRetain221_LightGBM_5seed_ensemble.joblib"


def metadata_path(project_root=None) -> Path:
    return runtime_root(project_root) / "model" / "GNRetain221_model_metadata.json"


def panel_path(project_root=None) -> Path:
    return runtime_root(project_root) / "schema" / "GNRetain221_feature_panel.csv"


@lru_cache(maxsize=4)
def load_feature_panel(root_str: str) -> list[str]:
    panel = pd.read_csv(panel_path(root_str)).sort_values("feature_order")
    required = {"feature_order", "feature", "feature_group"}
    if not required.issubset(panel.columns):
        raise ValueError(f"GNRetain feature schema is missing columns: {sorted(required - set(panel.columns))}")
    expected_order = list(range(1, EXPECTED_MODEL_FEATURE_COUNT + 1))
    actual_order = panel["feature_order"].astype(int).tolist()
    features = panel["feature"].astype(str).tolist()
    if actual_order != expected_order:
        raise ValueError("GNRetain feature_order must be a continuous 1-221 sequence")
    if len(features) != EXPECTED_MODEL_FEATURE_COUNT or len(set(features)) != EXPECTED_MODEL_FEATURE_COUNT:
        raise ValueError("GNRetain schema must contain 221 unique ordered features")
    return features


@lru_cache(maxsize=4)
def load_model_artifact(root_str: str):
    artifact = joblib.load(model_path(root_str))
    required = {"models", "features", "threshold", "seeds"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Unexpected GNRetain ensemble artifact structure")

    panel = load_feature_panel(root_str)
    if list(artifact["features"]) != panel:
        raise ValueError("Model feature order differs from the GNRetain 221-feature schema")

    models = list(artifact["models"])
    seeds = [int(seed) for seed in artifact["seeds"]]
    if len(models) != EXPECTED_ENSEMBLE_MEMBERS or len(seeds) != EXPECTED_ENSEMBLE_MEMBERS:
        raise ValueError(f"GNRetain requires {EXPECTED_ENSEMBLE_MEMBERS} LightGBM ensemble members")
    if len(set(seeds)) != EXPECTED_ENSEMBLE_MEMBERS:
        raise ValueError("GNRetain ensemble seeds must be unique")

    for index, model in enumerate(models, start=1):
        model_n = getattr(model, "n_features_in_", EXPECTED_MODEL_FEATURE_COUNT)
        if int(model_n) != EXPECTED_MODEL_FEATURE_COUNT:
            raise ValueError(f"GNRetain ensemble member {index} expects {model_n} features, not 221")
        feature_names = list(getattr(model, "feature_name_", panel))
        if feature_names != panel:
            raise ValueError(f"GNRetain ensemble member {index} feature order differs from the schema")

    threshold = float(artifact["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Invalid GNRetain decision threshold: {threshold}")
    return artifact


@lru_cache(maxsize=4)
def load_model_metadata(root_str: str) -> dict:
    data = json.loads(metadata_path(root_str).read_text(encoding="utf-8"))
    if int(data.get("n_features", -1)) != EXPECTED_MODEL_FEATURE_COUNT:
        raise ValueError("GNRetain metadata does not declare 221 model features")
    if int(data.get("ensemble_members", -1)) != EXPECTED_ENSEMBLE_MEMBERS:
        raise ValueError("GNRetain metadata does not declare five ensemble members")
    if list(data.get("features", [])) != load_feature_panel(root_str):
        raise ValueError("GNRetain metadata feature order differs from the schema")

    artifact = load_model_artifact(root_str)
    if not np.isclose(float(data.get("threshold", np.nan)), float(artifact["threshold"]), atol=1e-12):
        raise ValueError("GNRetain metadata threshold differs from the model artifact")
    if [int(x) for x in data.get("ensemble_seeds", [])] != [int(x) for x in artifact["seeds"]]:
        raise ValueError("GNRetain metadata seeds differ from the model artifact")
    return data


def positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probability = model.predict_proba(X)
    return np.asarray(probability[:, 1], dtype=float).reshape(-1)


def ensemble_positive_probability(models: list, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    member_probabilities = np.column_stack([positive_probability(model, X) for model in models])
    return member_probabilities.mean(axis=1), member_probabilities


def select_model_features(X: pd.DataFrame, project_root=None) -> pd.DataFrame:
    root = find_project_root(project_root)
    features = load_feature_panel(str(root))
    missing = [feature for feature in features if feature not in X.columns]
    if missing:
        raise ValueError(f"Generated source feature table is missing GNRetain inputs: {missing}")
    selected = X.loc[:, features].apply(pd.to_numeric, errors="coerce")
    if selected.isna().any().any() or np.isinf(selected.to_numpy(dtype=float)).any():
        raise ValueError("GNRetain inputs contain missing or infinite values")
    return selected


def predict_endpoint_from_features(
    X: pd.DataFrame,
    endpoint_key: str = PUBLIC_ENDPOINT_KEY,
    project_root=None,
    model_name: str | None = None,
) -> pd.DataFrame:
    if endpoint_key != PUBLIC_ENDPOINT_KEY:
        raise KeyError(f"Unsupported endpoint: {endpoint_key}")
    root = find_project_root(project_root)
    artifact = load_model_artifact(str(root))
    selected = select_model_features(X, root)

    probability, member_probabilities = ensemble_positive_probability(list(artifact["models"]), selected)
    threshold = float(artifact["threshold"])
    binary = (probability >= threshold).astype(int)

    out = pd.DataFrame(index=selected.index)
    out["proba_clean_ensemble"] = probability
    out["proba_retention_positive"] = probability
    out["threshold"] = threshold
    out["binary_class"] = binary
    out["class_label"] = np.where(binary == 1, "RP", "RN")
    out["endpoint_key"] = endpoint_key
    out["endpoint_label"] = ENDPOINT_LABELS[endpoint_key]
    out["model_name"] = model_name or "LightGBM_5seed_mean"
    out["feature_set"] = FEATURE_SET_NAME
    out["ensemble_members"] = EXPECTED_ENSEMBLE_MEMBERS
    out["member_probability_sd"] = member_probabilities.std(axis=1, ddof=0)
    for index, seed in enumerate(artifact["seeds"]):
        out[f"proba_lgbm_seed_{int(seed)}"] = member_probabilities[:, index]
    return out


def predict_from_features(X: pd.DataFrame, project_root=None) -> pd.DataFrame:
    return predict_endpoint_from_features(X, project_root=project_root).add_prefix("gnretention__")


def predict_smiles(smiles, project_root=None):
    Xsource, report = generate_runtime_source_features(smiles, project_root)
    wide = predict_from_features(Xsource, project_root)
    prefix = "gnretention__"
    result = pd.DataFrame(
        [
            {
                "input_smiles": report["input_smiles"],
                "runtime_smiles": report["runtime_smiles"],
                "endpoint_key": PUBLIC_ENDPOINT_KEY,
                "endpoint_label": PUBLIC_ENDPOINT_LABEL,
                "model_name": str(wide.loc[0, prefix + "model_name"]),
                "proba_clean_ensemble": float(wide.loc[0, prefix + "proba_clean_ensemble"]),
                "proba_retention_positive": float(wide.loc[0, prefix + "proba_retention_positive"]),
                "threshold": float(wide.loc[0, prefix + "threshold"]),
                "binary_class": int(wide.loc[0, prefix + "binary_class"]),
                "class_label": str(wide.loc[0, prefix + "class_label"]),
                "feature_set": FEATURE_SET_NAME,
                "ensemble_members": EXPECTED_ENSEMBLE_MEMBERS,
                "member_probability_sd": float(wide.loc[0, prefix + "member_probability_sd"]),
            }
        ]
    )
    report = dict(report)
    report.update(
        {
            "source_feature_count": int(Xsource.shape[1]),
            "model_feature_count": EXPECTED_MODEL_FEATURE_COUNT,
            "ensemble_members": EXPECTED_ENSEMBLE_MEMBERS,
            "feature_set": FEATURE_SET_NAME,
        }
    )
    return result, report, Xsource


def inspect_model_bundles(project_root=None) -> pd.DataFrame:
    root = find_project_root(project_root)
    path = model_path(root)
    schema = panel_path(root)
    metadata = metadata_path(root)
    row = {
        "endpoint_key": PUBLIC_ENDPOINT_KEY,
        "endpoint_label": PUBLIC_ENDPOINT_LABEL,
        "model_file": str(path),
        "schema_file": str(schema),
        "metadata_file": str(metadata),
        "file_exists": bool(path.exists() and schema.exists() and metadata.exists()),
        "status": "missing",
        "deploy_members_present": False,
        "primary_model": "LightGBM_5seed_mean",
        "algorithm": "LightGBM",
        "ensemble_members": EXPECTED_ENSEMBLE_MEMBERS,
        "n_features": EXPECTED_MODEL_FEATURE_COUNT,
        "threshold": np.nan,
        "training_n": 5452,
        "positives": 2536,
        "negatives": 2916,
        "error": "",
    }
    try:
        artifact = load_model_artifact(str(root))
        metadata_data = load_model_metadata(str(root))
        row.update(
            {
                "status": "ok",
                "deploy_members_present": True,
                "threshold": float(artifact["threshold"]),
                "ensemble_seeds": ", ".join(str(x) for x in artifact["seeds"]),
            }
        )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return pd.DataFrame([row])


def scientific_warning_text() -> str:
    return (
        "GNRetain predicts a Gram-negative cell-associated retention-positive versus retention-negative "
        "profile from a cleaned 221-feature molecular panel using a five-seed LightGBM ensemble. "
        "It does not predict MIC, antibacterial potency, toxicity, clinical susceptibility, or clinical outcome."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--smiles", default=None)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args(argv)
    root = find_project_root(args.root)

    if args.smiles:
        prediction, _, _ = predict_smiles(args.smiles, root)
        if args.out_csv:
            prediction.to_csv(args.out_csv, index=False)
        else:
            print(prediction.to_string(index=False))
        return 0

    if args.input_csv:
        source = pd.read_csv(args.input_csv)
        Xsource, reports = batch_generate_runtime_source_features(source[args.smiles_col].astype(str).tolist(), root)
        ok = reports.loc[reports["status"].eq("ok")].reset_index(drop=True)
        prediction = predict_from_features(Xsource, root).reset_index(drop=True) if len(Xsource) else pd.DataFrame()
        output = pd.concat([ok, prediction], axis=1)
        if args.out_csv:
            output.to_csv(args.out_csv, index=False)
        else:
            print(output.to_string(index=False))
        return 0

    print(inspect_model_bundles(root).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
