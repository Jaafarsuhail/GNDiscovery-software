"""SMILES to frozen46 + RDKit_unique + FCFP4 feature generation.

This module recovers the Phase08I/Phase08K matrix-builder logic for the 1273
feature deployment schema. It uses only local RDKit and copied Phase08
calibrator artifacts.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

TARGET_FEATURE_COUNT = 1273
PHASE09_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PHASE09_ROOT.parent
ARTIFACT_DIR = PHASE09_ROOT / "artifacts"
FEATURE_ARTIFACT_DIR = ARTIFACT_DIR / "feature_generation"
SCHEMA_PATH = ARTIFACT_DIR / "schemas" / "frozen46_rdkit_fcfp4_1273_schema.csv"
PHASE08_SOURCE_PATH = FEATURE_ARTIFACT_DIR / "phase08_paper36_reverse_engineering_source.py"
PHASE08_SOURCE_FALLBACK = PROJECT_ROOT / "Phase08_Paper36_ReverseEngineering" / "scripts" / "run_paper36_reverse_engineering.py"
CALIBRATOR_PATH = FEATURE_ARTIFACT_DIR / "paper_mimic_36_deployment_calibrators.joblib"
CALIBRATOR_FALLBACK = PROJECT_ROOT / "Phase08_Paper36_ReverseEngineering" / "models" / "paper_mimic_36_deployment_calibrators.joblib"
CANDIDATE_SCHEMA_PATH = FEATURE_ARTIFACT_DIR / "candidate_descriptor_pool_schema.csv"
CANDIDATE_SCHEMA_FALLBACK = PROJECT_ROOT / "Phase08_Paper36_ReverseEngineering" / "tables" / "candidate_descriptor_pool_from_SMILES.csv"

GEN_FCFP4 = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=1024,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)


@lru_cache(maxsize=1)
def load_feature_schema() -> list[str]:
    """Load the approved 1273-feature schema in order."""
    df = pd.read_csv(SCHEMA_PATH)
    if "feature" in df.columns:
        schema = df["feature"].astype(str).tolist()
    elif "feature_name" in df.columns:
        schema = df["feature_name"].astype(str).tolist()
    else:
        schema = df.iloc[:, 0].astype(str).tolist()
    if len(schema) != TARGET_FEATURE_COUNT:
        raise ValueError(f"Expected {TARGET_FEATURE_COUNT} features, found {len(schema)}.")
    if len(set(schema)) != len(schema):
        dupes = pd.Series(schema)[pd.Series(schema).duplicated()].unique().tolist()
        raise ValueError(f"Feature schema contains duplicate names: {dupes[:20]}")
    return schema


@lru_cache(maxsize=1)
def _phase08_module():
    source = PHASE08_SOURCE_PATH if PHASE08_SOURCE_PATH.exists() else PHASE08_SOURCE_FALLBACK
    if not source.exists():
        raise FileNotFoundError(f"Phase08 descriptor source not found: {source}")
    spec = importlib.util.spec_from_file_location("phase09_phase08_descriptor_source", str(source))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase09_phase08_descriptor_source"] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def _calibrators():
    path = CALIBRATOR_PATH if CALIBRATOR_PATH.exists() else CALIBRATOR_FALLBACK
    if not path.exists():
        raise FileNotFoundError(f"Phase08 deployment calibrator not found: {path}")
    return load(path)


@lru_cache(maxsize=1)
def _candidate_schema() -> list[str]:
    path = CANDIDATE_SCHEMA_PATH if CANDIDATE_SCHEMA_PATH.exists() else CANDIDATE_SCHEMA_FALLBACK
    if not path.exists():
        raise FileNotFoundError(f"Candidate descriptor schema not found: {path}")
    return pd.read_csv(path, nrows=0).columns.astype(str).tolist()


def _as_smiles_list(smiles: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(smiles, (list, tuple, pd.Series, np.ndarray)):
        values = [str(s).strip() for s in smiles]
    else:
        values = [str(smiles).strip()]
    if any(not s for s in values):
        raise ValueError("SMILES cannot be empty.")
    return values


def _candidate_pool_from_smiles(smiles: str | list[str]) -> pd.DataFrame:
    ph08 = _phase08_module()
    rows = []
    for s in _as_smiles_list(smiles):
        d = ph08.descriptor_candidates_for_smiles(s)
        if not d:
            raise ValueError(f"Could not generate Phase08 descriptor candidates for SMILES: {s}")
        rows.append(d)
    cand = pd.DataFrame(rows)
    for c in _candidate_schema():
        if c not in cand.columns:
            cand[c] = np.nan
    cand = cand[_candidate_schema()].replace([np.inf, -np.inf], np.nan)
    return cand


def _rdkit_descriptor_frame(smiles: str | list[str]) -> pd.DataFrame:
    desc_names = [name for name, _func in Descriptors._descList]
    rows = []
    for s in _as_smiles_list(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            raise ValueError(f"Invalid SMILES for RDKit descriptors: {s}")
        try:
            raw = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan, silent=True)
        except Exception:
            raw = {}
        rows.append({f"rdkit_{name}": raw.get(name, np.nan) for name in desc_names})
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return X.mask(X.abs() > 1e30, np.nan)


def generate_rdkit_unique_features(smiles: str) -> dict:
    """Generate the 203 `ru_rdkit_*` features used by the 1273 schema."""
    schema = load_feature_schema()
    ru_cols = [c for c in schema if c.startswith("ru_rdkit_")]
    raw_cols = [c.removeprefix("ru_") for c in ru_cols]
    rd = _rdkit_descriptor_frame(smiles)
    missing = [c for c in raw_cols if c not in rd.columns]
    if missing:
        raise ValueError(f"RDKit descriptor columns missing from current RDKit build: {missing}")
    out = rd[raw_cols].copy()
    out.columns = ru_cols
    return out.iloc[0].to_dict()


def generate_fcfp4_features(smiles: str) -> dict:
    """Generate the 1024-bit FCFP4 fingerprint with Phase08 column names."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES for FCFP4 generation: {smiles}")
    bits = GEN_FCFP4.GetFingerprintAsNumPy(mol).astype(np.uint8)
    cols = [f"fcfp4_{i:04d}" for i in range(1024)]
    return dict(zip(cols, bits.tolist()))


def _mimic36_from_smiles(smiles: str | list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ph08 = _phase08_module()
    cand = _candidate_pool_from_smiles(smiles)
    raw = pd.DataFrame(index=range(len(cand)))
    for f in ph08.PAPER36_COLUMNS:
        raw[f] = cand[ph08.SELECTED_RAW_CANDIDATE[f]].values
    out = raw.values.astype(float).copy()
    cal = _calibrators()
    imp = cal["physchem_emulator_models"]["imputer"]
    Xap = imp.transform(cand)
    for f, reg in cal["physchem_emulator_models"]["regressors"].items():
        j = ph08.PAPER36_COLUMNS.index(f)
        out[:, j] = reg.predict(Xap)
    return pd.DataFrame(out, columns=ph08.PAPER36_COLUMNS), cand


def generate_frozen46_features(smiles: str) -> dict:
    """Generate the frozen46 block recovered from Phase08I/Phase08K logic."""
    base, cand = _mimic36_from_smiles(smiles)
    rd = _rdkit_descriptor_frame(smiles)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES for frozen46 generation: {smiles}")
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    add = pd.DataFrame(index=range(len(base)))
    add["fb_rdkit_BertzCT"] = rd["rdkit_BertzCT"].values
    add["fb_rdkit_BertzCT__dup1"] = add["fb_rdkit_BertzCT"].values
    add["eng2_prev_molecular_complexity_bertz"] = cand["molecular_complexity_bertz"].values
    add["fb_rdkit_HallKierAlpha__dup1"] = rd["rdkit_HallKierAlpha"].values
    add["fb_s04s_polar_buried_lipophilicity_proxy"] = float(logp * tpsa)
    add["eng2_prev_clogs_esol"] = cand["clogs_esol"].values
    add["fb_open_esol_logs"] = cand["clogs_esol"].values
    add["eng2_eng_clogs_minus_clogp"] = base["clogs"].values - base["clogp"].values
    add["fb_rdkit_HallKierAlpha"] = rd["rdkit_HallKierAlpha"].values
    add["fb_rdkit_Chi0"] = rd["rdkit_Chi0"].values
    frozen = pd.concat([base.reset_index(drop=True), add.reset_index(drop=True)], axis=1)
    frozen = _clean_numeric(frozen)
    expected = [c for c in load_feature_schema() if not c.startswith("ru_rdkit_") and not c.startswith("fcfp4_")]
    missing = [c for c in expected if c not in frozen.columns]
    if missing:
        raise ValueError(f"Frozen46 features missing: {missing}")
    return frozen[expected].iloc[0].to_dict()


def align_features_to_schema(feature_dict_or_df, schema: list[str] | None = None) -> pd.DataFrame:
    """Align a generated feature dictionary/DataFrame to the approved schema."""
    schema = schema or load_feature_schema()
    if isinstance(feature_dict_or_df, pd.DataFrame):
        df = feature_dict_or_df.copy()
    else:
        df = pd.DataFrame([feature_dict_or_df])
    missing = [c for c in schema if c not in df.columns]
    extra = [c for c in df.columns if c not in schema]
    if missing:
        raise ValueError(f"Generated feature set is missing {len(missing)} schema columns: {missing[:50]}")
    if extra:
        df = df.drop(columns=extra)
    X = _clean_numeric(df[schema])
    return X


def validate_feature_vector(X: pd.DataFrame) -> dict:
    """Validate shape, ordering, and finite numeric readiness of a feature vector."""
    schema = load_feature_schema()
    out = {
        "n_rows": int(len(X)),
        "n_columns": int(X.shape[1]),
        "schema_length": int(len(schema)),
        "column_order_matches_schema": list(X.columns) == schema,
        "missing_columns": [c for c in schema if c not in X.columns],
        "extra_columns": [c for c in X.columns if c not in schema],
        "nan_value_count": int(X.isna().sum().sum()),
        "inf_value_count": int(np.isinf(X.select_dtypes(include=[np.number]).to_numpy()).sum()),
    }
    out["is_valid"] = (
        out["n_columns"] == TARGET_FEATURE_COUNT
        and out["column_order_matches_schema"]
        and not out["missing_columns"]
        and not out["extra_columns"]
    )
    return out


def generate_1273_features_from_smiles(smiles: str) -> pd.DataFrame:
    """Return one row with exactly 1273 columns in approved schema order."""
    features = {}
    features.update(generate_frozen46_features(smiles))
    features.update(generate_rdkit_unique_features(smiles))
    features.update(generate_fcfp4_features(smiles))
    X = align_features_to_schema(features, load_feature_schema())
    valid = validate_feature_vector(X)
    if not valid["is_valid"]:
        raise ValueError(f"Generated feature vector failed validation: {valid}")
    return X
