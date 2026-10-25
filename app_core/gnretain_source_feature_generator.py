from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from structural_motif_feature_generator import compute_motif_features, load_motif_dictionary

EXPECTED_SOURCE_FEATURE_COUNT = 389
FROZEN_COUNT = 46
RDKIT_UNIQUE_COUNT = 203
MOTIF_COUNT = 140
SOURCE_FEATURE_SET_NAME = "GNRetain source descriptor bank (389 features)"


def find_project_root(start=None) -> Path:
    """Locate the GNDiscovery application root with the GNRetain 221-feature runtime."""
    candidates: list[Path] = []
    if start is not None:
        start_s = str(start).strip().strip('"').strip("'")
        if start_s:
            p = Path(start_s).expanduser().resolve()
            if p.is_file():
                p = p.parent
            candidates.extend([p, *p.parents])
    here = Path(__file__).resolve()
    candidates.extend([Path.cwd().resolve(), here.parent, *here.parents])

    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        required = [
            p / "GNRetain_Source_Feature_Engine",
            p / "GNRetain221_Support_Assets",
            p / "GNRetain221_Runtime_Model",
        ]
        if all(path.exists() for path in required):
            return p
    base = str(start) if start is not None else str(Path.cwd())
    raise FileNotFoundError(f"Could not find the GNRetain app root from {base!r}")


def support_assets_root(project_root=None) -> Path:
    return find_project_root(project_root) / "GNRetain221_Support_Assets"


@lru_cache(maxsize=4)
def import_source_runtime_modules(project_root=None):
    root = find_project_root(project_root)
    engine_dir = (root / "GNRetain_Source_Feature_Engine").resolve()
    if str(engine_dir) not in sys.path:
        sys.path.insert(0, str(engine_dir))
    from app.modules.feature_generation import generate_1273_features_from_smiles
    from app.modules.smiles_standardization import standardize_smiles

    return root, generate_1273_features_from_smiles, standardize_smiles


@lru_cache(maxsize=4)
def load_1273_schema(project_root=None) -> list[str]:
    root = find_project_root(project_root)
    path = root / "GNRetain_Source_Feature_Engine" / "artifacts" / "schemas" / "frozen46_rdkit_fcfp4_1273_schema.csv"
    df = pd.read_csv(path)
    return df["feature"].astype(str).tolist() if "feature" in df.columns else df.iloc[:, 0].astype(str).tolist()


@lru_cache(maxsize=4)
def load_source_389_schema(project_root=None) -> list[str]:
    path = support_assets_root(project_root) / "schemas" / "gnretain_source_389_schema.csv"
    df = pd.read_csv(path)
    return df["feature"].astype(str).tolist() if "feature" in df.columns else df.iloc[:, 0].astype(str).tolist()


@lru_cache(maxsize=4)
def load_structural_motif_dictionary(project_root=None) -> pd.DataFrame:
    return load_motif_dictionary(support_assets_root(project_root) / "schemas" / "structural_motif_dictionary.csv")


def generate_runtime_source_features(smiles, project_root=None):
    root, generate_1273, standardize_smiles = import_source_runtime_modules(project_root)
    raw = str(smiles).strip()
    if not raw:
        raise ValueError("SMILES cannot be empty")
    runtime_smiles = standardize_smiles(raw)
    X1273 = generate_1273(runtime_smiles)
    X1273 = X1273.loc[:, load_1273_schema(root)].apply(pd.to_numeric, errors="coerce")
    Xbase = X1273.iloc[:, : FROZEN_COUNT + RDKIT_UNIQUE_COUNT].copy()
    Xmotif = compute_motif_features([runtime_smiles], load_structural_motif_dictionary(root))
    X = pd.concat([Xbase.reset_index(drop=True), Xmotif.reset_index(drop=True)], axis=1)
    schema = load_source_389_schema(root)
    X = X.loc[:, schema].apply(pd.to_numeric, errors="coerce")
    arr = X.to_numpy(dtype=float)
    report = {
        "input_smiles": raw,
        "runtime_smiles": runtime_smiles,
        "n_columns": int(X.shape[1]),
        "expected_n_columns": EXPECTED_SOURCE_FEATURE_COUNT,
        "nan_value_count": int(X.isna().sum().sum()),
        "inf_value_count": int(np.isinf(arr).sum()),
        "is_valid": bool(
            X.shape[1] == EXPECTED_SOURCE_FEATURE_COUNT
            and X.isna().sum().sum() == 0
            and np.isinf(arr).sum() == 0
        ),
        "feature_set": SOURCE_FEATURE_SET_NAME,
    }
    return X, report


def _generate_one_runtime_source_feature(item, project_root=None):
    i, smi = item
    try:
        X, report = generate_runtime_source_features(str(smi), project_root)
        report.update({"row_index": i, "status": "ok", "error": ""})
        return i, X, report
    except Exception as exc:
        report = {
            "row_index": i,
            "input_smiles": str(smi),
            "runtime_smiles": "",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "n_columns": 0,
            "is_valid": False,
        }
        return i, None, report


def batch_generate_runtime_source_features(
    smiles_values,
    project_root=None,
    max_workers: int = 1,
    parallel_backend: str = "threading",
):
    """Generate the frozen 389-feature source bank for a batch.

    ``parallel_backend='loky'`` uses independent worker processes for the
    Python-heavy descriptor stage and can utilise multiple physical CPU cores.
    ``parallel_backend='threading'`` is the lower-memory fallback. Exact input
    order and feature semantics are preserved. Any process-backend failure
    automatically falls back to threads rather than losing the batch.
    """
    values = list(smiles_values)
    workers = max(1, int(max_workers or 1))

    # Warm immutable runtime assets once in the parent process.
    import_source_runtime_modules(project_root)
    load_1273_schema(project_root)
    load_source_389_schema(project_root)
    load_structural_motif_dictionary(project_root)

    items = list(enumerate(values))
    if workers <= 1 or len(items) <= 1:
        generated = [_generate_one_runtime_source_feature(item, project_root) for item in items]
    else:
        workers = min(workers, len(items))
        backend = str(parallel_backend or "threading").lower()
        if backend == "loky":
            try:
                generated = Parallel(n_jobs=workers, backend="loky", batch_size="auto")(
                    delayed(_generate_one_runtime_source_feature)(item, project_root) for item in items
                )
            except Exception:
                # Streamlit/Windows installations can occasionally restrict
                # process creation. Keep a transparent safe fallback.
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gnretain-batch") as pool:
                    generated = list(pool.map(lambda item: _generate_one_runtime_source_feature(item, project_root), items))
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gnretain-batch") as pool:
                generated = list(pool.map(lambda item: _generate_one_runtime_source_feature(item, project_root), items))

    generated.sort(key=lambda x: x[0])
    reports = [report for _i, _X, report in generated]
    matrices = [X for _i, X, report in generated if report.get("status") == "ok" and X is not None]
    empty = pd.DataFrame(columns=load_source_389_schema(project_root))
    return (pd.concat(matrices, ignore_index=True) if matrices else empty), pd.DataFrame(reports)


def check_source_schema_compatibility(project_root=None) -> dict:
    schema = load_source_389_schema(project_root)
    motifs = load_structural_motif_dictionary(project_root)
    duplicate_count = len(schema) - len(set(schema))
    return {
        "source_feature_count": len(schema),
        "expected_source_feature_count": EXPECTED_SOURCE_FEATURE_COUNT,
        "source_schema_order_valid": len(schema) == EXPECTED_SOURCE_FEATURE_COUNT and duplicate_count == 0,
        "motif_dictionary_rows": len(motifs),
        "duplicate_source_feature_names": duplicate_count,
        "source_feature_set": SOURCE_FEATURE_SET_NAME,
    }
