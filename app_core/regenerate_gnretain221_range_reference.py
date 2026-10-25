from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gnretain_source_feature_generator import find_project_root
from gnretain221_prediction_core import load_feature_panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate GNRetain cleaned-221 feature-range references.")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    root = find_project_root(args.root)

    matrix_path = root / "GNRetain221_Model_Audit" / "schemas" / "internal_V05_cleaned221_features.csv.gz"
    output_dir = root / "GNRetain221_Support_Assets" / "ad_reference"
    features = load_feature_panel(str(root))
    matrix = pd.read_csv(matrix_path).loc[:, features].apply(pd.to_numeric, errors="raise")

    quantiles = pd.DataFrame({
        "feature": features,
        "q0": [float(matrix[c].min()) for c in features],
        "q1": [float(matrix[c].quantile(0.01)) for c in features],
        "q5": [float(matrix[c].quantile(0.05)) for c in features],
        "q95": [float(matrix[c].quantile(0.95)) for c in features],
        "q99": [float(matrix[c].quantile(0.99)) for c in features],
        "q100": [float(matrix[c].max()) for c in features],
    })
    quantiles.to_csv(output_dir / "GNRetain221_feature_range_quantiles.csv", index=False)

    moderate = ((matrix.lt(matrix.quantile(0.01))) | (matrix.gt(matrix.quantile(0.99)))).sum(axis=1)
    inside_limit = int(moderate.quantile(0.95, interpolation="higher"))
    borderline_limit = int(moderate.quantile(0.995, interpolation="higher"))
    policy = {
        "endpoint": "E. coli ΔtolC-associated retention profile",
        "reference_dataset": "GNRetain V05 training set",
        "n_reference": int(len(matrix)),
        "feature_space": "GNRetain cleaned 221-feature model panel",
        "nearest_neighbor_fingerprint": "Morgan radius 2, 2048 bits",
        "inside_domain": {"nearest_neighbor_similarity_min": 0.45, "severe_max": 0, "moderate_max": inside_limit},
        "borderline_domain": {"nearest_neighbor_similarity_min": 0.30, "severe_max": 2, "moderate_max": borderline_limit},
        "cutoff_derivation": {
            "moderate_count_95th_percentile": inside_limit,
            "moderate_count_99_5th_percentile": borderline_limit,
            "reference_rows": int(len(matrix)),
        },
        "note": "PCA maps are descriptive. Formal support uses Morgan similarity and all 221 model-feature ranges.",
    }
    (output_dir / "GNRetain221_ad_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print({"features": len(features), "rows": len(matrix), "inside_moderate_max": inside_limit, "borderline_moderate_max": borderline_limit})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
