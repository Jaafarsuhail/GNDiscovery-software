from __future__ import annotations

import py_compile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "app_core") not in sys.path:
    sys.path.insert(0, str(ROOT / "app_core"))

from gnretain_source_feature_generator import batch_generate_runtime_source_features
from gnretain221_prediction_core import predict_from_features
from gnmic_ecoli_prediction_core import gnmic_runtime_health_check, predict_gnmic_from_source_features


def main() -> int:
    app_file = ROOT / "streamlit_app" / "gndiscovery_app.py"
    py_compile.compile(str(app_file), doraise=True)

    smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CCN(CC)CCCC(C)Nc1ncc2ccccc2n1",
        "O=C1NCC(=O)N1",
        "not_a_smiles",
    ]
    X, reports = batch_generate_runtime_source_features(smiles, project_root=ROOT)
    assert reports["status"].tolist() == ["ok", "ok", "ok", "failed"]
    assert X.shape == (3, 389)

    retention = predict_from_features(X, project_root=ROOT)
    assert len(retention) == 3
    assert retention["gnretention__proba_clean_ensemble"].between(0, 1).all()

    valid_smiles = reports.loc[reports["status"].eq("ok"), "runtime_smiles"].astype(str).tolist()
    mic, neighbours = predict_gnmic_from_source_features(valid_smiles, X, project_root=ROOT, top_n_neighbours=3)
    expected = np.asarray([0.13904158496184694, 0.0647502479378505, 0.01978929140420554])
    assert np.allclose(mic["probability_active"].to_numpy(), expected, rtol=0, atol=1e-5)
    assert mic["threshold"].eq(0.510).all()
    for column in [
        "prediction_support",
        "training_chemistry_label",
        "neighbourhood_evidence_label",
        "weighted_neighbourhood_agreement",
        "effective_neighbours",
        "estimated_oof_reliability",
    ]:
        assert column in mic.columns, column
    assert mic["weighted_neighbourhood_agreement"].between(0, 1).all()
    assert mic["estimated_oof_reliability"].between(0, 1).all()
    assert len(neighbours) == 3 and all(len(table) >= 3 for table in neighbours)
    for table in neighbours:
        assert {0, 1}.issubset(set(pd.to_numeric(table["binary_label"], errors="coerce").dropna().astype(int)))

    ready, status = gnmic_runtime_health_check(ROOT)
    assert ready, status

    source = app_file.read_text(encoding="utf-8")
    required = [
        "Choose a prediction workflow",
        "E. coli ΔtolC-associated retention",
        "E. coli ATCC 25922 MIC activity",
        "Integrated Assessment",
        "Prediction Support",
        "Training Analogues",
        "chemical_space_support_map",
        "descriptor_percentile_interval_plot",
        "ionization_solubility_space_map",
        "Beyond observed range",
        "Outside usual range",
        "Less typical features",
        "Nearest training similarity",
        "Neighbourhood evidence",
        "Accuracy for this support profile",
        "How prediction support was assessed",
        "Predicted-class confidence",
        "Distance from 0.510",
        "Directed class similarity",
        "Support categories in this rebuilt runtime",
        "The MIC prediction is supported by similar compounds in the GNMIC training set.",
    ]
    missing = [item for item in required if item not in source]
    assert not missing, missing
    print("GNDiscovery modular subapp smoke test passed")
    print({
        "source_features": X.shape,
        "retention_rows": len(retention),
        "gnmic_rows": len(mic),
        "gnmic_threshold": status["threshold"],
        "gnmic_features": status["n_features"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
