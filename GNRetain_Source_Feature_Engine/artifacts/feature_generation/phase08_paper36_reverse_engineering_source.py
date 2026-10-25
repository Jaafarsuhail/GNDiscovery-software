#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Independent phase: exact/near-exact reverse engineering of the original paper36 descriptors.

Input source of truth:
    Phase01_Data_Curation_Label_QC_FULL_CHEMBL/_extracted_input/Original Data/J_Model_clean.csv

Final ML benchmark:
    HistGradientBoostingClassifier only.

No OC218, no FCFP4, no 1447 bank, no external validation.
The final HGB matrices contain only the 36 original paper descriptor names and order.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Avoid OpenMP deadlocks on some Windows/conda/scikit-learn/RDKit stacks.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

try:
    from joblib import dump
except Exception:  # pragma: no cover
    dump = None

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Fragments, GraphDescriptors, Lipinski, QED, rdMolDescriptors
    from rdkit.Chem.MolSurf import LabuteASA
    RDLogger.DisableLog("rdApp.*")
except Exception as e:  # pragma: no cover
    raise RuntimeError("RDKit is required for this phase. Install with: conda install -c conda-forge rdkit") from e

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PAPER36_COLUMNS = [
    "alkyl_amines", "amides", "amines", "aromatic_amines", "aromatic_atoms",
    "aromatic_nitrogens", "aromatic_rings", "basic_nitrogens", "c_atoms",
    "clogp", "clogs", "electro_negative_atoms", "h_acceptors", "h_donors",
    "metal_atoms", "molecular_complexity", "molecular_flexibility", "non_c_h_atoms",
    "non_h_atoms", "rings_closures", "rotatable_bonds", "small_rings", "sp3_atoms",
    "symmetric_atoms", "total_surface_area", "fraction_csp3", "aliphatic_rings",
    "aromatic_carbocycles", "aromatic_heterocycles", "hetero_atoms",
    "saturated_carbocycles", "saturated_heterocycles", "saturated_rings",
    "acidic_pka", "basic_pka", "clogd_ph_7_4",
]

FOCUS8_EMULATED = [
    "clogp", "clogs", "acidic_pka", "basic_pka", "clogd_ph_7_4",
    "molecular_complexity", "molecular_flexibility", "total_surface_area",
]

WEAK_FOCUS_FROM_REQUEST = [
    "small_rings", "aromatic_amines", "clogp", "clogs", "acidic_pka",
    "basic_pka", "clogd_ph_7_4", "molecular_complexity",
    "molecular_flexibility", "total_surface_area",
]

SMARTS = {
    "amide": Chem.MolFromSmarts("[NX3][CX3](=[OX1])"),
    "anilinelike_N": Chem.MolFromSmarts("[NX3;!$(NC=O);!$(N=O)]-[a]"),
    "amine": Chem.MolFromSmarts("[NX3;!$(NC=O);!$(N=O);!$(N~S(=O)=O)]"),
    "alkylamine": Chem.MolFromSmarts("[NX3;!$(NC=O);!$(N=O);!$(N~S(=O)=O);!$([N]-[a])]"),
    "basic_N_loose": Chem.MolFromSmarts("[N;!$(N=O);!$(NC=O);!$(N~S(=O)=O);!$([nH])]"),
    "acid_carbox": Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
    "acid_sulfonamide": Chem.MolFromSmarts("S(=O)(=O)[NX3H1,NX3H2]"),
    "phenol": Chem.MolFromSmarts("[c][OX2H]"),
    "tetrazole": Chem.MolFromSmarts("c1nn[nH]n1"),
}

SELECTED_RAW_CANDIDATE = {
    "alkyl_amines": "alkyl_amines_smarts",
    "amides": "amides_frag",
    "amines": "alkyl_amines_smarts",
    "aromatic_amines": "aromatic_amines_smarts",
    "aromatic_atoms": "aromatic_atoms_rdkit",
    "aromatic_nitrogens": "aromatic_nitrogens_rdkit",
    "aromatic_rings": "aromatic_rings_rdkit",
    "basic_nitrogens": "basic_nitrogens_smarts",
    "c_atoms": "c_atoms_rdkit",
    "clogp": "clogp_rdkit_crippen",
    "clogs": "clogs_esol",
    "electro_negative_atoms": "electro_negative_atoms_rdkit",
    "h_acceptors": "h_acceptors_rdkit",
    "h_donors": "h_donors_rdkit",
    "metal_atoms": "metal_atoms_rdkit",
    "molecular_complexity": "molecular_complexity_qed",
    "molecular_flexibility": "molecular_flexibility_rot_bonds",
    "non_c_h_atoms": "non_c_h_atoms_rdkit",
    "non_h_atoms": "non_h_atoms_rdkit",
    "rings_closures": "rings_closures_numrings",
    "rotatable_bonds": "rotatable_bonds_rdkit",
    "small_rings": "small_rings_size3_7",
    "sp3_atoms": "sp3_atoms_all",
    "symmetric_atoms": "symmetric_atoms_dup_rank_classes",
    "total_surface_area": "total_surface_area_labuteASA",
    "fraction_csp3": "fraction_csp3_rdkit",
    "aliphatic_rings": "aliphatic_rings_rdkit",
    "aromatic_carbocycles": "aromatic_carbocycles_rdkit",
    "aromatic_heterocycles": "aromatic_heterocycles_rdkit",
    "hetero_atoms": "hetero_atoms_rdkit",
    "saturated_carbocycles": "saturated_carbocycles_rdkit",
    "saturated_heterocycles": "saturated_heterocycles_rdkit",
    "saturated_rings": "saturated_rings_rdkit",
    "acidic_pka": "acidic_pka_heuristic_default7",
    "basic_pka": "basic_pka_heuristic_default7",
    "clogd_ph_7_4": "clogd_ph_7_4_HH_acidic_pka_heuristic_default7_basic_pka_heuristic_default0",
}


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "|", msg, flush=True)


def count_matches(mol: Chem.Mol, patt: Chem.Mol) -> int:
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt, uniquify=True))


def count_unique_atoms(mol: Chem.Mol, patt: Chem.Mol, atomic_num: int | None = None) -> int:
    if patt is None:
        return 0
    s = set()
    for match in mol.GetSubstructMatches(patt, uniquify=True):
        for idx in match:
            a = mol.GetAtomWithIdx(idx)
            if atomic_num is None or a.GetAtomicNum() == atomic_num:
                s.add(idx)
    return len(s)


def esol_logS(mol: Chem.Mol) -> float:
    """Delaney ESOL approximation using RDKit descriptors."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    aromatic_prop = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(1, mol.GetNumHeavyAtoms())
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot - 0.74 * aromatic_prop


def descriptor_candidates_for_smiles(smiles: str) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {}
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    atoms = list(mol.GetAtoms())
    heavy = mol.GetNumHeavyAtoms()
    c_count = sum(1 for a in atoms if a.GetAtomicNum() == 6)
    hetero = sum(1 for a in atoms if a.GetAtomicNum() not in (1, 6))
    elec = sum(1 for a in atoms if a.GetAtomicNum() in [7, 8, 9, 15, 16, 17, 35, 53])
    arom_atoms = sum(1 for a in atoms if a.GetIsAromatic())
    arom_n = sum(1 for a in atoms if a.GetIsAromatic() and a.GetAtomicNum() == 7)
    sp3_all = sum(1 for a in atoms if a.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
    sp3_c = sum(1 for a in atoms if a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    sym_atoms = sum(1 for r in ranks if ranks.count(r) > 1)
    sym_classes = sum(1 for r in set(ranks) if ranks.count(r) > 1)
    sym_total_minus_classes = heavy - len(set(ranks))
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_bonds = max(1, mol.GetNumBonds())
    clogp = Crippen.MolLogP(mol)
    bertz = GraphDescriptors.BertzCT(mol)
    balaban = GraphDescriptors.BalabanJ(mol) if heavy > 1 else 0.0

    alkyl_amines = count_unique_atoms(mol, SMARTS["alkylamine"], 7)
    amines = count_unique_atoms(mol, SMARTS["amine"], 7)
    aromatic_amines = count_unique_atoms(mol, SMARTS["anilinelike_N"], 7)
    ar_n_frag = Fragments.fr_Ar_N(mol)

    # Very simple open heuristic pKa proxies. These are not claimed to be exact; the diagnostics decide.
    acid_vals: List[float] = []
    acid_vals += [4.5] * count_matches(mol, SMARTS["acid_carbox"])
    acid_vals += [8.5] * count_matches(mol, SMARTS["acid_sulfonamide"])
    acid_vals += [10.0] * count_matches(mol, SMARTS["phenol"])
    acid_vals += [5.0] * count_matches(mol, SMARTS["tetrazole"])
    basic_vals: List[float] = []
    basic_vals += [9.8] * int(alkyl_amines)
    basic_vals += [4.7] * int(aromatic_amines)
    basic_vals += [5.2] * int(ar_n_frag)
    acid7 = min(acid_vals) if acid_vals else 7.0
    acid14 = min(acid_vals) if acid_vals else 14.0
    base7 = max(basic_vals) if basic_vals else 7.0
    base0 = max(basic_vals) if basic_vals else 0.0

    def hh_logd(logp: float, acid: float, base: float) -> float:
        corr = 0.0
        if acid < 13.5:
            corr += 10 ** (7.4 - acid)
        if base > 0.5:
            corr += 10 ** (base - 7.4)
        return logp - math.log10(1 + corr)

    d = {
        "c_atoms_rdkit": c_count,
        "hetero_atoms_rdkit": hetero,
        "electro_negative_atoms_rdkit": elec,
        "non_c_h_atoms_rdkit": hetero,
        "non_h_atoms_rdkit": heavy,
        "aromatic_atoms_rdkit": arom_atoms,
        "aromatic_nitrogens_rdkit": arom_n,
        "aromatic_rings_rdkit": rdMolDescriptors.CalcNumAromaticRings(mol),
        "aliphatic_rings_rdkit": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "aromatic_carbocycles_rdkit": rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        "aromatic_heterocycles_rdkit": rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        "saturated_carbocycles_rdkit": rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
        "saturated_heterocycles_rdkit": rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        "saturated_rings_rdkit": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "rings_closures_numrings": rdMolDescriptors.CalcNumRings(mol),
        "small_rings_size3_6": sum(1 for s in ring_sizes if s <= 6),
        "small_rings_size3_7": sum(1 for s in ring_sizes if s <= 7),
        "small_rings_size3_8": sum(1 for s in ring_sizes if s <= 8),
        "rotatable_bonds_rdkit": rot,
        "rotatable_bonds_lipinski": Lipinski.NumRotatableBonds(mol),
        "fraction_csp3_rdkit": rdMolDescriptors.CalcFractionCSP3(mol),
        "sp3_atoms_all": sp3_all,
        "sp3_atoms_c": sp3_c,
        "h_acceptors_rdkit": rdMolDescriptors.CalcNumHBA(mol),
        "h_donors_rdkit": rdMolDescriptors.CalcNumHBD(mol),
        "metal_atoms_rdkit": sum(1 for a in atoms if a.GetAtomicNum() in [3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,37,38,39,40,41,42,44,45,46,47,48,49,50,55,56,57,72,73,74,75,76,77,78,79,80,81,82,83]),
        "clogp_rdkit_crippen": clogp,
        "clogs_esol": esol_logS(mol),
        "total_surface_area_labuteASA": LabuteASA(mol),
        "total_surface_area_tpsa": rdMolDescriptors.CalcTPSA(mol),
        "molecular_complexity_bertz": bertz,
        "molecular_complexity_bertz_norm_heavy": bertz / max(1, heavy),
        "molecular_complexity_bertz_norm_heavy2": bertz / max(1, heavy * heavy),
        "molecular_complexity_balaban": balaban,
        "molecular_complexity_qed": QED.qed(mol),
        "molecular_flexibility_rot_heavy": rot / max(1, heavy),
        "molecular_flexibility_rot_bonds": rot / n_bonds,
        "symmetric_atoms_dup_rank_atoms": sym_atoms,
        "symmetric_atoms_dup_rank_classes": sym_classes,
        "symmetric_atoms_total_minus_classes": sym_total_minus_classes,
        "amides_frag": Fragments.fr_amide(mol),
        "amides_smarts": count_matches(mol, SMARTS["amide"]),
        "amines_fr_NH0_NH1_NH2": Fragments.fr_NH0(mol) + Fragments.fr_NH1(mol) + Fragments.fr_NH2(mol),
        "amines_smarts": amines,
        "aromatic_amines_aniline_frag": Fragments.fr_aniline(mol),
        "aromatic_amines_smarts": aromatic_amines,
        "alkyl_amines_smarts": alkyl_amines,
        "basic_nitrogens_smarts": count_unique_atoms(mol, SMARTS["basic_N_loose"], 7),
        "basic_nitrogens_frag_NH": Fragments.fr_NH0(mol) + Fragments.fr_NH1(mol) + Fragments.fr_NH2(mol) + ar_n_frag,
        "basic_nitrogens_arN_plus_amine": ar_n_frag + amines,
        "acidic_pka_heuristic_default7": acid7,
        "acidic_pka_heuristic_default14": acid14,
        "basic_pka_heuristic_default7": base7,
        "basic_pka_heuristic_default0": base0,
        "clogd_ph_7_4_HH_acidic_pka_heuristic_default7_basic_pka_heuristic_default7": hh_logd(clogp, acid7, base7),
        "clogd_ph_7_4_HH_acidic_pka_heuristic_default7_basic_pka_heuristic_default0": hh_logd(clogp, acid7, base0),
        "clogd_ph_7_4_HH_acidic_pka_heuristic_default14_basic_pka_heuristic_default7": hh_logd(clogp, acid14, base7),
        "clogd_ph_7_4_HH_acidic_pka_heuristic_default14_basic_pka_heuristic_default0": hh_logd(clogp, acid14, base0),
    }
    return d


def find_input_csv(root: Path, input_rel: str) -> Path:
    direct = root / input_rel
    if direct.exists():
        return direct
    zip_path = root / "Phase01_Data_Curation_Label_QC_FULL_CHEMBL.zip"
    if zip_path.exists():
        extract_dir = root / "Phase01_Data_Curation_Label_QC_FULL_CHEMBL"
        target = extract_dir / "_extracted_input" / "Original Data" / "J_Model_clean.csv"
        if not target.exists():
            log(f"Extracting J_Model_clean.csv from {zip_path.name}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                member = "Phase01_Data_Curation_Label_QC_FULL_CHEMBL/_extracted_input/Original Data/J_Model_clean.csv"
                zf.extract(member, root)
        if target.exists():
            return target
    raise FileNotFoundError(f"Could not find {input_rel} under root={root}")


def ensure_outdirs(outdir: Path) -> Dict[str, Path]:
    d = {
        "root": outdir,
        "tables": outdir / "tables",
        "matrices": outdir / "matrices",
        "models": outdir / "models",
        "logs": outdir / "logs",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def build_raw_mimic(candidates: pd.DataFrame) -> pd.DataFrame:
    missing = [v for v in SELECTED_RAW_CANDIDATE.values() if v not in candidates.columns]
    if missing:
        raise ValueError(f"Missing selected candidate columns: {missing}")
    return pd.DataFrame({col: candidates[SELECTED_RAW_CANDIDATE[col]] for col in PAPER36_COLUMNS})[PAPER36_COLUMNS]


def corr_metrics(y: pd.Series, x: pd.Series) -> Dict[str, float]:
    yy = pd.to_numeric(y, errors="coerce")
    xx = pd.to_numeric(x, errors="coerce")
    m = yy.notna() & xx.notna()
    if int(m.sum()) == 0:
        return {"n_paired": 0, "coverage": 0.0, "pearson_r": np.nan, "spearman_rho": np.nan, "mae": np.nan, "rmse": np.nan}
    a = yy[m].astype(float)
    b = xx[m].astype(float)
    return {
        "n_paired": int(m.sum()),
        "coverage": float(m.mean()),
        "pearson_r": float(a.corr(b, method="pearson")) if a.nunique() > 1 and b.nunique() > 1 else np.nan,
        "spearman_rho": float(a.corr(b, method="spearman")) if a.nunique() > 1 and b.nunique() > 1 else np.nan,
        "mae": float(np.mean(np.abs(a.values - b.values))),
        "rmse": float(np.sqrt(np.mean((a.values - b.values) ** 2))),
    }


def make_candidate_diagnostics(original: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in PAPER36_COLUMNS:
        for cand in candidates.columns:
            # keep candidate names related to target; this keeps the table useful and not huge.
            if cand.startswith(target) or target in cand or (
                target in ["amines", "alkyl_amines", "aromatic_amines", "basic_nitrogens"] and "amine" in cand
            ):
                met = corr_metrics(original[target], candidates[cand])
                met.update({"paper36_feature": target, "candidate_method": cand})
                rows.append(met)
    return pd.DataFrame(rows)[["paper36_feature", "candidate_method", "n_paired", "coverage", "pearson_r", "spearman_rho", "mae", "rmse"]]


def make_descriptor_diagnostics(original: pd.DataFrame, matrices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, mat in matrices.items():
        for col in PAPER36_COLUMNS:
            met = corr_metrics(original[col], mat[col])
            met.update({
                "feature_set": name,
                "paper36_feature": col,
                "selected_method": SELECTED_RAW_CANDIDATE.get(col, ""),
                "focus_descriptor": col in WEAK_FOCUS_FROM_REQUEST,
            })
            rows.append(met)
    return pd.DataFrame(rows)[["feature_set", "paper36_feature", "focus_descriptor", "selected_method", "n_paired", "coverage", "pearson_r", "spearman_rho", "mae", "rmse"]]


def hgb_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def quantile_map_fit_apply(x_train: np.ndarray, y_train: np.ndarray, x_apply: np.ndarray) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_apply = np.asarray(x_apply, dtype=float)
    med = np.nanmedian(x_train)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_apply = np.where(np.isfinite(x_apply), x_apply, med)
    order = np.argsort(x_train)
    sx = x_train[order]
    sy = np.sort(y_train)
    pct = np.searchsorted(sx, x_apply, side="right") / max(1, len(sx))
    grid = np.linspace(0, 1, len(sy), endpoint=True)
    return np.interp(pct, grid, sy)


def linear_calibrate_train_apply(raw_train: np.ndarray, ydesc_train: np.ndarray, raw_apply: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    out = np.zeros((raw_apply.shape[0], len(PAPER36_COLUMNS)), dtype=float)
    methods = []
    for j, col in enumerate(PAPER36_COLUMNS):
        pipe = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
        pipe.fit(raw_train[:, [j]], ydesc_train[:, j])
        out[:, j] = pipe.predict(raw_apply[:, [j]])
        methods.append({"paper36_feature": col, "calibration": "ridge_one_raw_candidate", "raw_candidate": SELECTED_RAW_CANDIDATE[col]})
    return out, methods


def quantile_calibrate_train_apply(raw_train: np.ndarray, ydesc_train: np.ndarray, raw_apply: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    out = np.zeros((raw_apply.shape[0], len(PAPER36_COLUMNS)), dtype=float)
    methods = []
    for j, col in enumerate(PAPER36_COLUMNS):
        out[:, j] = quantile_map_fit_apply(raw_train[:, j], ydesc_train[:, j], raw_apply[:, j])
        methods.append({"paper36_feature": col, "calibration": "quantile_map_one_raw_candidate", "raw_candidate": SELECTED_RAW_CANDIDATE[col]})
    return out, methods


def physchem_emulator_train_apply(
    raw_train: np.ndarray,
    ydesc_train: np.ndarray,
    candidate_train: pd.DataFrame,
    raw_apply: np.ndarray,
    candidate_apply: pd.DataFrame,
    seed: int,
    focus_features: List[str] = FOCUS8_EMULATED,
) -> Tuple[np.ndarray, List[dict], Dict[str, object]]:
    """Keep raw descriptors except focus physchem descriptors, which are HGB-regressed from the candidate pool.

    This is still a 36-column mimic matrix. The candidate pool is used only to emulate descriptor values,
    not as direct HGB classification input.
    """
    out = raw_apply.copy().astype(float)
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(candidate_train.replace([np.inf, -np.inf], np.nan))
    Xap = imp.transform(candidate_apply.replace([np.inf, -np.inf], np.nan))
    methods = []
    models = {"imputer": imp, "regressors": {}, "focus_features": focus_features}
    for f in focus_features:
        j = PAPER36_COLUMNS.index(f)
        reg = HistGradientBoostingRegressor(max_iter=40, early_stopping=True, random_state=seed + 101 * (j + 1))
        reg.fit(Xtr, ydesc_train[:, j])
        out[:, j] = reg.predict(Xap)
        methods.append({
            "paper36_feature": f,
            "calibration": "hgb_regressor_from_candidate_pool_train_only",
            "candidate_pool_n": int(candidate_train.shape[1]),
        })
        models["regressors"][f] = reg
    return out, methods, models


def evaluate_feature_matrix(name: str, X: pd.DataFrame | np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> List[dict]:
    Xarr = np.asarray(X, dtype=float)
    rows = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(Xarr, y), 1):
        log(f"HGB CV {name} fold {fold}/{n_splits}")
        clf = HistGradientBoostingClassifier(random_state=seed + fold, max_iter=100)
        clf.fit(Xarr[tr], y[tr])
        pred = clf.predict(Xarr[va])
        score = clf.predict_proba(Xarr[va])[:, 1]
        met = hgb_metrics(y[va], pred, score)
        met.update({"feature_set": name, "fold": fold, "n_train": int(len(tr)), "n_valid": int(len(va)), "model": "HGB"})
        rows.append(met)
    return rows


def evaluate_calibrated_feature_set(
    name: str,
    raw: pd.DataFrame,
    original: pd.DataFrame,
    candidates: pd.DataFrame,
    y: np.ndarray,
    mode: str,
    n_splits: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    raw_arr = raw.values.astype(float)
    orig_arr = original.values.astype(float)
    rows: List[dict] = []
    method_rows: List[dict] = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(cv.split(raw_arr, y), 1):
        log(f"HGB CV {name} mode={mode} fold {fold}/{n_splits}")
        if mode == "linear":
            Xtr, m1 = linear_calibrate_train_apply(raw_arr[tr], orig_arr[tr], raw_arr[tr])
            Xva, _ = linear_calibrate_train_apply(raw_arr[tr], orig_arr[tr], raw_arr[va])
            for r in m1:
                r.update({"feature_set": name, "fold": fold})
                method_rows.append(r)
        elif mode == "quantile":
            Xtr, m1 = quantile_calibrate_train_apply(raw_arr[tr], orig_arr[tr], raw_arr[tr])
            Xva, _ = quantile_calibrate_train_apply(raw_arr[tr], orig_arr[tr], raw_arr[va])
            for r in m1:
                r.update({"feature_set": name, "fold": fold})
                method_rows.append(r)
        elif mode == "physchem_hgb_emulator":
            # Optimized leakage-safe emulation: fit each descriptor emulator once on the training rows,
            # then predict both train and validation rows. The original version retrained twice per fold.
            Xtr = raw_arr[tr].copy().astype(float)
            Xva = raw_arr[va].copy().astype(float)
            imp = SimpleImputer(strategy="median")
            Ctr = imp.fit_transform(candidates.iloc[tr].replace([np.inf, -np.inf], np.nan))
            Cva = imp.transform(candidates.iloc[va].replace([np.inf, -np.inf], np.nan))
            m1 = []
            for f in FOCUS8_EMULATED:
                j = PAPER36_COLUMNS.index(f)
                reg = HistGradientBoostingRegressor(max_iter=40, early_stopping=True, random_state=seed + fold * 1000 + 101 * (j + 1))
                reg.fit(Ctr, orig_arr[tr, j])
                Xtr[:, j] = reg.predict(Ctr)
                Xva[:, j] = reg.predict(Cva)
                m1.append({
                    "paper36_feature": f,
                    "calibration": "hgb_regressor_from_candidate_pool_train_only",
                    "candidate_pool_n": int(candidates.shape[1]),
                })
            for r in m1:
                r.update({"feature_set": name, "fold": fold})
                method_rows.append(r)
        else:
            raise ValueError(mode)
        clf = HistGradientBoostingClassifier(random_state=seed + fold, max_iter=100)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xva)
        score = clf.predict_proba(Xva)[:, 1]
        met = hgb_metrics(y[va], pred, score)
        met.update({"feature_set": name, "fold": fold, "n_train": int(len(tr)), "n_valid": int(len(va)), "model": "HGB"})
        rows.append(met)
    return rows, method_rows


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fs, g in fold_df.groupby("feature_set", sort=False):
        row = {"feature_set": fs, "model": "HGB", "n_folds": int(g.shape[0])}
        for metric in ["mcc", "balanced_accuracy", "accuracy", "f1", "roc_auc", "pr_auc"]:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_sd"] = float(g[metric].std(ddof=1)) if g.shape[0] > 1 else np.nan
        for c in ["tn", "fp", "fn", "tp"]:
            row[f"{c}_sum"] = int(g[c].sum())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mcc_mean", ascending=False)
    original_mcc = out.loc[out["feature_set"].eq("paper36_original_from_J_Model_clean"), "mcc_mean"]
    if not original_mcc.empty:
        ref = float(original_mcc.iloc[0])
        out["delta_mcc_vs_original"] = out["mcc_mean"] - ref
    return out


def make_full_deployment_matrices(raw: pd.DataFrame, original: pd.DataFrame, candidates: pd.DataFrame, seed: int, model_dir: Path) -> Dict[str, pd.DataFrame]:
    raw_arr = raw.values.astype(float)
    orig_arr = original.values.astype(float)
    lin, _ = linear_calibrate_train_apply(raw_arr, orig_arr, raw_arr)
    qnt, _ = quantile_calibrate_train_apply(raw_arr, orig_arr, raw_arr)
    phys, methods, models = physchem_emulator_train_apply(raw_arr, orig_arr, candidates, raw_arr, candidates, seed=seed)
    if dump is not None:
        dump({
            "selected_raw_candidate": SELECTED_RAW_CANDIDATE,
            "paper36_columns": PAPER36_COLUMNS,
            "physchem_emulator_models": models,
            "note": "Deployment calibrators fitted on full J_Model_clean paper36 descriptors; do not use as CV evidence.",
        }, model_dir / "paper_mimic_36_deployment_calibrators.joblib")
    return {
        "paper_mimic_36_raw_rdkit_from_SMILES": raw.copy(),
        "paper_mimic_36_linear_calibrated_deployment_NOT_CV": pd.DataFrame(lin, columns=PAPER36_COLUMNS),
        "paper_mimic_36_quantile_calibrated_deployment_NOT_CV": pd.DataFrame(qnt, columns=PAPER36_COLUMNS),
        "paper_mimic_36_physchem_hgb_emulated_deployment_NOT_CV": pd.DataFrame(phys, columns=PAPER36_COLUMNS),
    }


def classify_descriptor_status(diag: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    sub = diag[diag["feature_set"].eq(feature_set)].copy()
    def status(r):
        pr = r["pearson_r"]
        mae = r["mae"]
        if pd.isna(pr):
            return "constant_or_uninformative"
        if pr >= 0.95:
            return "well_reproduced"
        if pr >= 0.85:
            return "usable_but_not_exact"
        if pr >= 0.70:
            return "partly_reproduced"
        return "problematic"
    sub["reproduction_status"] = sub.apply(status, axis=1)
    notes = []
    for _, r in sub.iterrows():
        f = r["paper36_feature"]
        if f in ["clogp", "clogs", "acidic_pka", "basic_pka", "clogd_ph_7_4"]:
            note = "likely requires original OpenChemLib/Chemaxon/DataWarrior/OPERA-style algorithm rather than RDKit-only heuristic"
        elif f in ["molecular_complexity", "molecular_flexibility", "total_surface_area"]:
            note = "OpenChemLib definition/scale likely differs from RDKit proxy; calibration helps scale but not exact local ranking"
        elif f in ["aromatic_amines", "amines", "basic_nitrogens", "alkyl_amines", "h_acceptors"]:
            note = "functional-group definition differs; needs exact SMARTS/atom-typing rules from original node"
        elif f == "small_rings":
            note = "RDKit SSSR small-ring count is close after repair, but exact ring perception may still differ for fused/bridged systems"
        else:
            note = "mostly reproduced from RDKit/open counting route"
        notes.append(note)
    sub["interpretation_note"] = notes
    return sub


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="Project root")
    parser.add_argument("--input-rel", type=str, default="Phase01_Data_Curation_Label_QC_FULL_CHEMBL/_extracted_input/Original Data/J_Model_clean.csv")
    parser.add_argument("--outdir", type=str, default="Phase08_Paper36_ReverseEngineering")
    parser.add_argument("--label-col", type=str, default="label_2350")
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--id-col", type=str, default="compound_id")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-candidates", action="store_true", help="Regenerate candidate descriptor pool even if cached CSV exists")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = root / outdir
    dirs = ensure_outdirs(outdir)
    input_csv = find_input_csv(root, args.input_rel)

    log(f"Loading source of truth: {input_csv}")
    df = pd.read_csv(input_csv)
    missing_cols = [c for c in [args.smiles_col, args.label_col] + PAPER36_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in J_Model_clean.csv: {missing_cols}")

    y = df[args.label_col].astype(int).values
    original = df[PAPER36_COLUMNS].apply(pd.to_numeric, errors="coerce")

    pd.DataFrame({"order": range(1, len(PAPER36_COLUMNS) + 1), "paper36_feature": PAPER36_COLUMNS}).to_csv(dirs["tables"] / "paper36_feature_list_frozen.csv", index=False)
    id_cols = [c for c in [args.id_col, args.smiles_col, args.label_col] if c in df.columns]
    original_export = pd.concat([df[id_cols].reset_index(drop=True), original.reset_index(drop=True)], axis=1)
    original_export.to_csv(dirs["matrices"] / "paper36_original_from_J_Model_clean.csv", index=False)

    candidate_cache = dirs["tables"] / "candidate_descriptor_pool_from_SMILES.csv"
    if candidate_cache.exists() and not args.overwrite_candidates:
        log(f"Loading cached candidate descriptor pool: {candidate_cache}")
        candidates = pd.read_csv(candidate_cache).replace([np.inf, -np.inf], np.nan)
    else:
        log("Generating RDKit/SMARTS/heuristic candidate descriptors from SMILES")
        cand_rows = [descriptor_candidates_for_smiles(s) for s in df[args.smiles_col]]
        candidates = pd.DataFrame(cand_rows).replace([np.inf, -np.inf], np.nan)
        candidates.to_csv(candidate_cache, index=False)

    log("Building selected 36-column raw mimic matrix")
    raw = build_raw_mimic(candidates)

    log("Running descriptor-level diagnostics")
    cand_diag = make_candidate_diagnostics(original, candidates).sort_values(["paper36_feature", "rmse", "mae"])
    cand_diag.to_csv(dirs["tables"] / "descriptor_candidate_diagnostics_long.csv", index=False)

    log("Running HGB CV: original and raw mimic")
    fold_rows = []
    fold_rows += evaluate_feature_matrix("paper36_original_from_J_Model_clean", original, y, args.n_splits, args.seed)
    fold_rows += evaluate_feature_matrix("paper_mimic_36_raw_rdkit_from_SMILES", raw, y, args.n_splits, args.seed)

    log("Running leakage-safe HGB CV: train-only linear calibration")
    rows, methods_linear = evaluate_calibrated_feature_set("paper_mimic_36_linear_calibrated_train_only", raw, original, candidates, y, "linear", args.n_splits, args.seed)
    fold_rows += rows

    log("Running leakage-safe HGB CV: train-only quantile calibration")
    rows, methods_quantile = evaluate_calibrated_feature_set("paper_mimic_36_quantile_calibrated_train_only", raw, original, candidates, y, "quantile", args.n_splits, args.seed)
    fold_rows += rows

    log("Running leakage-safe HGB CV: train-only physchem HGB emulator on focus descriptors")
    rows, methods_phys = evaluate_calibrated_feature_set("paper_mimic_36_physchem_hgb_emulated_train_only", raw, original, candidates, y, "physchem_hgb_emulator", args.n_splits, args.seed)
    fold_rows += rows

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(dirs["tables"] / "hgb_cv_fold_metrics.csv", index=False)
    summary = summarize_fold_metrics(fold_df)
    summary.to_csv(dirs["tables"] / "hgb_cv_summary.csv", index=False)

    pd.DataFrame(methods_linear + methods_quantile + methods_phys).to_csv(dirs["tables"] / "train_fold_calibration_emulation_methods.csv", index=False)

    log("Fitting full-dataset deployment/calculation mimic matrices")
    deployments = make_full_deployment_matrices(raw, original, candidates, args.seed, dirs["models"])
    matrices_for_diag = {"raw_rdkit": raw}
    for name, mat in deployments.items():
        export = pd.concat([df[id_cols].reset_index(drop=True), mat.reset_index(drop=True)], axis=1)
        export.to_csv(dirs["matrices"] / f"{name}.csv", index=False)
        matrices_for_diag[name] = mat
    desc_diag = make_descriptor_diagnostics(original, matrices_for_diag)
    desc_diag.to_csv(dirs["tables"] / "descriptor_diagnostics_final_matrices.csv", index=False)

    status = classify_descriptor_status(desc_diag, "raw_rdkit")
    status.to_csv(dirs["tables"] / "raw_mimic_descriptor_status_and_failure_modes.csv", index=False)

    original_mcc = float(summary.loc[summary["feature_set"].eq("paper36_original_from_J_Model_clean"), "mcc_mean"].iloc[0])
    best_mimic_row = summary[~summary["feature_set"].eq("paper36_original_from_J_Model_clean")].sort_values("mcc_mean", ascending=False).iloc[0]
    best_mimic_mcc = float(best_mimic_row["mcc_mean"])
    target_gap = best_mimic_mcc - original_mcc
    gap_ok = abs(target_gap) <= 0.03

    interpretation_rows = []
    for _, r in status.sort_values("pearson_r", ascending=True, na_position="last").iterrows():
        interpretation_rows.append({
            "paper36_feature": r["paper36_feature"],
            "raw_pearson_r": r["pearson_r"],
            "raw_spearman_rho": r["spearman_rho"],
            "raw_mae": r["mae"],
            "raw_rmse": r["rmse"],
            "reproduction_status": r["reproduction_status"],
            "interpretation_note": r["interpretation_note"],
        })
    pd.DataFrame(interpretation_rows).to_csv(dirs["tables"] / "remaining_gap_interpretation.csv", index=False)

    manifest = {
        "phase": "Phase08_Paper36_ReverseEngineering",
        "source_of_truth": str(input_csv),
        "n_rows": int(df.shape[0]),
        "label_col": args.label_col,
        "smiles_col": args.smiles_col,
        "paper36_feature_count": len(PAPER36_COLUMNS),
        "hgb_cv": {"n_splits": args.n_splits, "seed": args.seed, "model": "HistGradientBoostingClassifier", "max_iter": 100},
        "original_paper36_hgb_mcc_mean": original_mcc,
        "best_mimic_feature_set": str(best_mimic_row["feature_set"]),
        "best_mimic_hgb_mcc_mean": best_mimic_mcc,
        "best_mimic_delta_mcc_vs_original": target_gap,
        "within_0p03_mcc_target": bool(gap_ok),
        "focus_emulated_descriptors": FOCUS8_EMULATED,
        "validation_rule": "Descriptor calibration/emulation in CV is fitted only on outer-training rows and applied to validation rows. Full-dataset deployment matrices are not CV evidence.",
        "prohibited_blocks_not_used": ["OC218", "OC_mimic36 broad block", "FCFP4", "full1447 Phase04AG feature bank", "external validation"],
    }
    with open(dirs["root"] / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme = f"""# Phase08 Paper36 Reverse Engineering

Independent exact/near-exact reverse-engineering phase for the original 36 paper descriptors in `J_Model_clean.csv`.

## Source of truth
`{input_csv}`

## Frozen paper36 columns
See `tables/paper36_feature_list_frozen.csv`.

## HGB-only benchmark result
- Original paper36 HGB MCC mean: `{original_mcc:.6f}`
- Best mimic HGB MCC mean: `{best_mimic_mcc:.6f}`
- Best mimic feature set: `{best_mimic_row['feature_set']}`
- Delta mimic minus original: `{target_gap:.6f}`
- Within 0.02-0.03 MCC target: `{gap_ok}`

## Important interpretation
The final HGB input matrices contain only 36 descriptors with the same names and order as the original paper36 block.
No OC218, no FCFP4, no full1447 feature bank, and no external validation were used.

The full-dataset deployment/calculation matrices in `matrices/` are marked `NOT_CV` where calibration/emulation used all rows. They are calculation outputs for later deployment testing, not the evidence used for MCC.

## Main outputs
- `matrices/paper36_original_from_J_Model_clean.csv`
- `matrices/paper_mimic_36_raw_rdkit_from_SMILES.csv`
- `matrices/paper_mimic_36_linear_calibrated_deployment_NOT_CV.csv`
- `matrices/paper_mimic_36_quantile_calibrated_deployment_NOT_CV.csv`
- `matrices/paper_mimic_36_physchem_hgb_emulated_deployment_NOT_CV.csv`
- `tables/hgb_cv_summary.csv`
- `tables/hgb_cv_fold_metrics.csv`
- `tables/descriptor_diagnostics_final_matrices.csv`
- `tables/remaining_gap_interpretation.csv`

## Rerun command
```bat
cd "C:\\Users\\pharm\\OneDrive\\Desktop\\Project Workspace\\Fresh Start of the project"
python Phase08_Paper36_ReverseEngineering\\scripts\\run_paper36_reverse_engineering.py --root "." --outdir "Phase08_Paper36_ReverseEngineering"
```
"""
    (dirs["root"] / "README_Phase08_Paper36_ReverseEngineering.md").write_text(readme, encoding="utf-8")

    log("DONE")
    log(f"Original paper36 HGB MCC = {original_mcc:.6f}")
    log(f"Best mimic HGB MCC = {best_mimic_mcc:.6f} ({best_mimic_row['feature_set']})")
    log(f"Delta MCC = {target_gap:.6f}; within target? {gap_ok}")
    log(f"Outputs saved to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
