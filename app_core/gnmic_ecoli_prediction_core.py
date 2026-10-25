"""GNMIC-EC1792 inference for GNDiscovery.

The recovered frozen EC1792 ensemble, threshold and 1792 feature indices are
used for all prediction paths. Empirical Prediction Support is a separately
refitted reconstruction based on historical EC1792 out-of-fold probabilities.
"""
from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from GNMIC_prediction_support_v2 import support_basis

GNMIC_DIRNAME = "GNMIC_ECOLI_ATCC25922_Runtime"
EXPECTED_FEATURES = 1792
GENERATED_TOPOLOGY_FEATURES = 9970
EXPECTED_MODEL_SHA256 = "ecb000ece0356ef1fd53390ee888ed2acdd2befc6c972c833643e11384b38e0d"
EXPECTED_MANIFEST_SHA256 = "c942426f4a875ddd42c6ff60f8970829190d5d34435d97af77af7863e3e3efce"
EXPECTED_TRAINING_N = 10148


def _app_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        candidate = Path(project_root).expanduser().resolve()
        if (candidate / GNMIC_DIRNAME).exists():
            return candidate
    return Path(__file__).resolve().parents[1]


def find_gnmic_root(project_root: str | Path | None = None) -> Path:
    root = _app_root(project_root) / GNMIC_DIRNAME
    if not root.exists():
        raise FileNotFoundError(f"GNMIC runtime folder not found: {root}")
    return root


@lru_cache(maxsize=4)
def _load_resources_cached(root_text: str) -> dict[str, Any]:
    root = Path(root_text)
    model_bundle = joblib.load(root / "model/GNMIC-EC1792_LightGBM_3seed_fullcohort.joblib")
    metadata = json.loads((root / "model/model_metadata.json").read_text(encoding="utf-8"))
    feature_bundle = joblib.load(root / "feature_pipeline/reference/ATCC25922_feature_reference_bundle.joblib")
    gnretain_bundle = joblib.load(root / "feature_pipeline/runtime/model/GNRetain94_V05_fullfit.joblib")
    medians = np.load(root / "feature_pipeline/reference/GNRetain94_training_medians.npy")
    train_ecfp = sparse.load_npz(root / "feature_pipeline/reference/ATCC25922_ECFP4_2048_csr.npz").astype(np.float32)
    train_meta = pd.read_csv(root / "data/internal/ATCC25922_full_cohort_training.csv.gz")
    support_bundle = joblib.load(root / "feature_pipeline/reference/GNMIC_EC1792_prediction_support_refit.joblib")
    return {
        "root": root,
        "model_bundle": model_bundle,
        "metadata": metadata,
        "feature_bundle": feature_bundle,
        "gnretain_bundle": gnretain_bundle,
        "medians": medians,
        "train_ecfp": train_ecfp,
        "train_meta": train_meta,
        "support_bundle": support_bundle,
    }


def load_gnmic_resources(project_root: str | Path | None = None) -> dict[str, Any]:
    return _load_resources_cached(str(find_gnmic_root(project_root)))


def _bit_csr(rows: list[int], cols: list[int], n_rows: int, n_cols: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_rows, n_cols),
        dtype=np.float32,
    )


def _dict_csr(dict_rows: list[dict[int, int]], n_cols: int) -> sparse.csr_matrix:
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in dict_rows:
        for key, value in row.items():
            indices.append(int(key))
            data.append(float(value))
        indptr.append(len(data))
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(dict_rows), n_cols),
        dtype=np.float32,
    )


def generate_topology9970(
    standardized_smiles: list[str],
    source_features: pd.DataFrame,
    project_root: str | Path | None = None,
    num_threads: int = 1,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Build 9970 frozen topology inputs and select 1792 EC1792 inputs in model order."""
    resources = load_gnmic_resources(project_root)
    representation = resources["feature_bundle"]["representation"]
    selected = np.asarray(representation["baseline512"]["selected_indices_full4190"], dtype=np.int32)
    keep = np.asarray(representation["topology_feature_keep_mask"], dtype=bool)
    gn_features = list(resources["gnretain_bundle"]["features"])

    if len(standardized_smiles) != len(source_features):
        raise ValueError("SMILES and source-feature row counts do not match.")
    missing = [name for name in gn_features if name not in source_features.columns]
    if missing:
        raise KeyError(f"GNMIC is missing {len(missing)} GNRetain source features; first missing: {missing[:5]}")

    gn94 = source_features.loc[:, gn_features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    bad = ~np.isfinite(gn94)
    if bad.any():
        rr, cc = np.where(bad)
        gn94[rr, cc] = resources["medians"][cc]

    ecfp4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fcfp4 = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    morgan3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    atom_pair = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=4096)
    torsion = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=4096)

    er: list[int] = []
    ec: list[int] = []
    fr: list[int] = []
    fc: list[int] = []
    m3_rows: list[dict[int, int]] = []
    ap_rows: list[dict[int, int]] = []
    tt_rows: list[dict[int, int]] = []
    maccs_rows: list[dict[int, int]] = []

    mols = []
    for i, smiles in enumerate(standardized_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid standardized SMILES at row {i}: {smiles}")
        mols.append(mol)

    threads = max(1, int(num_threads or 1))
    # Native RDKit bulk fingerprint generation uses C++ worker threads and is
    # substantially faster than invoking each generator molecule-by-molecule.
    ecfp_fps = ecfp4.GetFingerprints(mols, numThreads=threads)
    fcfp_fps = fcfp4.GetFingerprints(mols, numThreads=threads)
    m3_fps = morgan3.GetCountFingerprints(mols, numThreads=threads)
    ap_fps = atom_pair.GetCountFingerprints(mols, numThreads=threads)
    tt_fps = torsion.GetCountFingerprints(mols, numThreads=threads)

    for i, (efp, ffp, m3fp, apfp, ttfp, mol) in enumerate(zip(ecfp_fps, fcfp_fps, m3_fps, ap_fps, tt_fps, mols)):
        for j in efp.GetOnBits():
            er.append(i); ec.append(int(j))
        for j in ffp.GetOnBits():
            fr.append(i); fc.append(int(j))
        m3_rows.append(dict(m3fp.GetNonzeroElements()))
        ap_rows.append(dict(apfp.GetNonzeroElements()))
        tt_rows.append(dict(ttfp.GetNonzeroElements()))
        maccs_rows.append({int(j): 1 for j in MACCSkeys.GenMACCSKeys(mol).GetOnBits()})

    n_rows = len(standardized_smiles)
    e = _bit_csr(er, ec, n_rows, 2048)
    f = _bit_csr(fr, fc, n_rows, 2048)
    baseline_full = sparse.hstack([e, f, sparse.csr_matrix(gn94)], format="csr")
    baseline512 = baseline_full[:, selected]
    unfiltered = sparse.hstack(
        [
            baseline512,
            _dict_csr(m3_rows, 4096),
            _dict_csr(ap_rows, 4096),
            _dict_csr(tt_rows, 4096),
            _dict_csr(maccs_rows, 167),
        ],
        format="csr",
    )
    topology = unfiltered[:, keep].astype(np.float32)
    if topology.shape[1] != GENERATED_TOPOLOGY_FEATURES:
        raise RuntimeError(f"Expected {GENERATED_TOPOLOGY_FEATURES:,} generator features, generated {topology.shape[1]:,}.")
    indices = np.asarray(resources["model_bundle"]["feature_indices_9970"], dtype=np.int32)
    if len(indices) != EXPECTED_FEATURES or len(set(indices)) != EXPECTED_FEATURES:
        raise RuntimeError("Invalid or duplicate EC1792 feature indices")
    if indices.min() < 0 or indices.max() >= GENERATED_TOPOLOGY_FEATURES:
        raise RuntimeError("EC1792 feature index outside generated topology bank")
    return topology[:, indices].tocsr(), e


def _query_scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def _deterministic_fold(scaffold_smiles: str, smiles: str, repeat: int) -> int:
    token = f"{repeat}|{scaffold_smiles or smiles}".encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:12], 16) % 3


def _oof_comparable_support(
    similarity: np.ndarray,
    query_smiles: str,
    predicted_label: int,
    probability: float,
    train_meta: pd.DataFrame,
    support_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Calculate leakage-controlled support and profile-based validation accuracy.

    A new query is assigned a deterministic pseudo-held-out scaffold fold for each
    repeat.  Support features are then calculated against the corresponding
    fold-training chemistry, matching the repeated scaffold-OOF development
    protocol used to train the empirical correctness model.
    """
    labels = train_meta["binary_label"].astype(int).to_numpy()
    scaffold_smiles = _query_scaffold_smiles(query_smiles)
    train_scaffolds = train_meta.get("scaffold_smiles", pd.Series([""] * len(train_meta))).fillna("").astype(str)
    k = int(support_bundle["selected_k"])
    gamma = float(support_bundle["similarity_power"])

    repeat_rows: list[dict[str, float]] = []
    for repeat in range(3):
        fold_col = f"scaffold_repeat{repeat}_fold"
        folds = pd.to_numeric(train_meta[fold_col], errors="coerce").fillna(0).astype(int).to_numpy()
        matched = np.where(train_scaffolds.to_numpy() == scaffold_smiles)[0] if scaffold_smiles else np.array([], dtype=int)
        if len(matched):
            assigned_fold = int(folds[matched[0]])
        else:
            assigned_fold = _deterministic_fold(scaffold_smiles, query_smiles, repeat)
        ref = np.where(folds != assigned_fold)[0]
        sim = similarity[ref]
        y_ref = labels[ref]
        if sim.size == 0:
            continue

        max_all = float(sim.max())
        max_active = float(sim[y_ref == 1].max()) if np.any(y_ref == 1) else np.nan
        max_inactive = float(sim[y_ref == 0].max()) if np.any(y_ref == 0) else np.nan
        raw_difference = max_active - max_inactive
        directed_difference = raw_difference if int(predicted_label) == 1 else -raw_difference

        n_keep = min(k, sim.size)
        top = np.argpartition(-sim, n_keep - 1)[:n_keep]
        top = top[np.argsort(-sim[top])]
        top_s = np.clip(sim[top].astype(float), 0.0, 1.0)
        top_y = y_ref[top].astype(float)
        weights = np.power(top_s, gamma)
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            weighted_active = float(np.dot(weights, top_y) / weight_sum)
            effective_n = float(weight_sum**2 / np.square(weights).sum())
        else:
            weighted_active = float(top_y.mean())
            effective_n = float(n_keep)
        agreement = weighted_active if int(predicted_label) == 1 else 1.0 - weighted_active
        repeat_rows.append(
            {
                "max_similarity": max_all,
                "max_active_similarity": max_active,
                "max_inactive_similarity": max_inactive,
                "directed_similarity_difference": directed_difference,
                "predicted_class_agreement": agreement,
                "effective_neighbours": effective_n,
            }
        )

    if not repeat_rows:
        return {
            "support_max_similarity": np.nan,
            "weighted_neighbourhood_agreement": np.nan,
            "effective_neighbours": 0.0,
            "estimated_oof_reliability": np.nan,
            "prediction_support": "Indeterminate",
            "training_chemistry_label": "Unavailable",
            "neighbourhood_evidence_label": "Unavailable",
            "support_interpretation": "Prediction support could not be estimated for this molecule.",
        }

    averaged = {
        key: float(np.nanmean([row[key] for row in repeat_rows]))
        for key in repeat_rows[0]
    }

    clipped = float(np.clip(probability, 1e-6, 1 - 1e-6))
    logit = np.log(clipped / (1.0 - clipped))
    calibrated_active = float(support_bundle["active_probability_calibrator"].predict_proba(np.asarray([[logit]], dtype=float))[0, 1])
    predicted_confidence = calibrated_active if int(predicted_label) == 1 else 1.0 - calibrated_active
    threshold = float(support_bundle["decision_threshold"])
    raw_margin = probability - threshold if int(predicted_label) == 1 else threshold - probability
    support_features = np.asarray(
        [[
            predicted_confidence,
            raw_margin,
            averaged["max_similarity"],
            averaged["directed_similarity_difference"],
            averaged["predicted_class_agreement"],
            averaged["effective_neighbours"],
            float(predicted_label),
        ]],
        dtype=float,
    )
    reliability = float(support_bundle["correctness_model"].predict_proba(support_basis(support_features))[0, 1])
    policy = support_bundle["category_policy"]

    if (
        averaged["max_similarity"] < float(policy["indeterminate_if_max_similarity_below"])
        or averaged["effective_neighbours"] < float(policy["indeterminate_if_effective_neighbours_below"])
    ):
        category = "Indeterminate"
    elif (
        reliability >= float(policy["high_reliability_threshold"])
        and averaged["predicted_class_agreement"] >= float(policy["high_agreement_threshold"])
    ):
        category = "High"
    elif (
        reliability >= float(policy["moderate_reliability_threshold"])
        and averaged["predicted_class_agreement"] >= float(policy["moderate_agreement_threshold"])
    ):
        category = "Moderate"
    else:
        category = "Low"

    # Descriptive chemistry/neighbourhood labels for the UI. These do not determine
    # the Prediction Support category except for the explicit sparse-evidence gate above.
    if averaged["max_similarity"] >= 0.60:
        chemistry_label = "Well represented"
    elif averaged["max_similarity"] >= 0.40:
        chemistry_label = "Partly represented"
    else:
        chemistry_label = "Sparsely represented"

    if averaged["effective_neighbours"] < float(policy["indeterminate_if_effective_neighbours_below"]):
        neighbourhood_label = "Sparse neighbourhood"
    elif averaged["predicted_class_agreement"] >= 0.60:
        neighbourhood_label = "Supports active prediction" if int(predicted_label) == 1 else "Supports inactive prediction"
    elif averaged["predicted_class_agreement"] >= 0.40:
        neighbourhood_label = "Mixed local evidence"
    else:
        neighbourhood_label = "Favours the opposite class"

    if category == "High":
        interpretation = (
            "The molecule is represented in the training chemistry, nearby compounds agree with the predicted class, "
            "and held-out predictions with similar support characteristics were usually correct."
        )
    elif category == "Moderate":
        interpretation = (
            "The molecule has usable training support, but the local evidence or validation performance for similar support profiles is less consistent."
        )
    elif category == "Low":
        interpretation = (
            "Nearby compounds provide conflicting evidence, or held-out predictions with similar support characteristics were often less reliable."
        )
    else:
        interpretation = (
            "The available training examples are too sparse to estimate prediction support reliably."
        )

    scaffold_count = int((train_scaffolds == scaffold_smiles).sum()) if scaffold_smiles else 0
    return {
        "support_max_similarity": averaged["max_similarity"],
        "support_max_active_similarity": averaged["max_active_similarity"],
        "support_max_inactive_similarity": averaged["max_inactive_similarity"],
        "support_directed_similarity_difference": averaged["directed_similarity_difference"],
        "weighted_neighbourhood_agreement": averaged["predicted_class_agreement"],
        "effective_neighbours": averaged["effective_neighbours"],
        "calibrated_active_probability_internal": calibrated_active,
        "predicted_class_confidence": predicted_confidence,
        "distance_from_threshold": abs(float(probability) - threshold),
        "directed_threshold_margin": raw_margin,
        "estimated_oof_reliability": reliability,
        "prediction_support": category,
        "training_chemistry_label": chemistry_label,
        "neighbourhood_evidence_label": neighbourhood_label,
        "support_interpretation": interpretation,
        "query_scaffold_smiles": scaffold_smiles,
        "scaffold_represented": bool(scaffold_count > 0),
        "same_scaffold_training_count": scaffold_count,
    }


def similarity_diagnostics(
    query_ecfp: sparse.csr_matrix,
    standardized_smiles: list[str],
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
    project_root: str | Path | None = None,
    top_n: int = 5,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    resources = load_gnmic_resources(project_root)
    train_ecfp = resources["train_ecfp"]
    train_meta = resources["train_meta"]
    support_bundle = resources["support_bundle"]
    labels = train_meta["binary_label"].astype(int).to_numpy()
    train_counts = np.asarray(train_ecfp.sum(axis=1)).ravel()
    query_counts = np.asarray(query_ecfp.sum(axis=1)).ravel()

    summaries: list[dict[str, Any]] = []
    neighbours: list[pd.DataFrame] = []
    # Calculate all train-query intersections in one sparse multiplication.
    # For the normal batch chunk sizes this keeps memory modest while avoiding
    # thousands of repeated sparse matrix multiplications.
    intersections_matrix = (train_ecfp @ query_ecfp.T).toarray()
    for i in range(query_ecfp.shape[0]):
        intersections = np.asarray(intersections_matrix[:, i]).ravel()
        denominator = train_counts + query_counts[i] - intersections
        similarity = np.divide(
            intersections,
            denominator,
            out=np.zeros_like(intersections, dtype=np.float32),
            where=denominator > 0,
        )
        max_similarity = float(similarity.max())
        max_positive = float(similarity[labels == 1].max())
        max_negative = float(similarity[labels == 0].max())
        empirical = _oof_comparable_support(
            similarity=similarity,
            query_smiles=str(standardized_smiles[i]),
            predicted_label=int(predicted_labels[i]),
            probability=float(probabilities[i]),
            train_meta=train_meta,
            support_bundle=support_bundle,
        )
        summaries.append(
            {
                "max_similarity_internal": max_similarity,
                "max_similarity_positive_internal": max_positive,
                "max_similarity_negative_internal": max_negative,
                "positive_minus_negative_similarity": max_positive - max_negative,
                "mic_training_support": empirical["prediction_support"],
                "mic_support_interpretation": empirical["support_interpretation"],
                **empirical,
            }
        )
        n_keep = max(1, int(top_n))
        overall_order = np.argsort(-similarity)[:n_keep]
        positive_idx = np.where(labels == 1)[0]
        negative_idx = np.where(labels == 0)[0]
        positive_order = positive_idx[np.argsort(-similarity[positive_idx])[:n_keep]]
        negative_order = negative_idx[np.argsort(-similarity[negative_idx])[:n_keep]]
        order = np.unique(np.concatenate([overall_order, positive_order, negative_order]))
        order = order[np.argsort(-similarity[order])]
        cols = [
            "standardized_smiles_no_stereo",
            "binary_label",
            "consensus_class",
            "consensus_mic_ug_ml",
            "consensus_confidence",
            "molecule_chembl_ids",
            "primary_document_id",
        ]
        available = [c for c in cols if c in train_meta.columns]
        nn = train_meta.iloc[order][available].copy()
        nn.insert(0, "rank", np.arange(1, len(nn) + 1))
        nn.insert(1, "tanimoto_similarity", similarity[order])
        nn["predicted_endpoint_label"] = nn.get("binary_label", pd.Series(index=nn.index)).map({1: "MIC-active", 0: "MIC-inactive"})
        nn["neighbour_scope"] = "class-aware retrieval"
        neighbours.append(nn.reset_index(drop=True))
    return pd.DataFrame(summaries), neighbours

def predict_gnmic_from_source_features(
    standardized_smiles: list[str],
    source_features: pd.DataFrame,
    project_root: str | Path | None = None,
    top_n_neighbours: int = 5,
    num_threads: int = 1,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    resources = load_gnmic_resources(project_root)
    X, query_ecfp = generate_topology9970(
        standardized_smiles, source_features, project_root, num_threads=num_threads
    )
    model_bundle = resources["model_bundle"]
    seed_probabilities = np.vstack([model.predict_proba(X)[:, 1] for model in model_bundle["models"]])
    probability = np.mean(seed_probabilities, axis=0)
    threshold = float(round(float(model_bundle["threshold"]), 3))
    predicted = (probability >= threshold).astype(int)
    support, neighbours = similarity_diagnostics(
        query_ecfp,
        standardized_smiles=standardized_smiles,
        probabilities=probability,
        predicted_labels=predicted,
        project_root=project_root,
        top_n=top_n_neighbours,
    )
    result = support.copy()
    result.insert(0, "probability_active", probability.astype(float))
    result.insert(1, "threshold", threshold)
    result.insert(2, "predicted_label", predicted.astype(int))
    result.insert(3, "predicted_class", np.where(predicted == 1, "MIC-active-like", "MIC-inactive-like"))
    result.insert(4, "organism", resources["metadata"]["organism"])
    result.insert(5, "endpoint", "Active: MIC <=16 µg/mL; inactive: MIC >32 µg/mL")
    result["seed_probability_sd_internal"] = np.std(seed_probabilities, axis=0, ddof=1)
    return result, neighbours


def gnmic_runtime_health_check(project_root: str | Path | None = None) -> tuple[bool, dict[str, Any]]:
    """Fail closed if EC1792 model, manifest, reference or support assets drift."""
    status: dict[str, Any] = {"ready": False, "error": ""}
    try:
        resources = load_gnmic_resources(project_root)
        root = resources["root"]
        metadata = resources["metadata"]
        model = resources["model_bundle"]
        support = resources["support_bundle"]
        representation = resources["feature_bundle"]["representation"]
        model_file = root / "model/GNMIC-EC1792_LightGBM_3seed_fullcohort.joblib"
        manifest_file = root / "model/GNMIC_EC1792_feature_manifest.csv"
        manifest = pd.read_csv(manifest_file)
        def sha256(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()

        # Git on Windows can convert the manifest's LF line endings to CRLF.
        # Normalize line endings only; changes to feature values or order still
        # fail the integrity check against the frozen manifest checksum.
        def manifest_sha256(p):
            return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        index = np.asarray(model.get("feature_indices_9970", []), dtype=np.int64)
        declared_index = manifest["feature_index_9970"].to_numpy(np.int64)
        checks = {
            "organism_ok": metadata.get("organism") == "Escherichia coli ATCC 25922",
            "model_identity_ok": model.get("model_name") == "GNMIC-EC1792" and metadata.get("model_name") == "GNMIC-EC1792",
            "model_sha256_ok": sha256(model_file) == EXPECTED_MODEL_SHA256 == metadata.get("model_sha256"),
            "manifest_sha256_ok": manifest_sha256(manifest_file) == EXPECTED_MANIFEST_SHA256 == metadata.get("feature_manifest_sha256"),
            "feature_count_ok": int(metadata.get("n_model_features", -1)) == EXPECTED_FEATURES and len(index) == EXPECTED_FEATURES,
            "feature_manifest_order_ok": np.array_equal(index, declared_index) and len(np.unique(index)) == EXPECTED_FEATURES,
            "topology_generator_ok": int(np.asarray(representation["topology_feature_keep_mask"], dtype=bool).sum()) == GENERATED_TOPOLOGY_FEATURES,
            "ensemble_ok": len(model.get("models", [])) == 3 and all(int(m.n_features_in_) == EXPECTED_FEATURES for m in model["models"]),
            "model_threshold_ok": np.isclose(float(model.get("threshold", np.nan)), 0.510),
            "support_threshold_ok": np.isclose(float(support.get("decision_threshold", np.nan)), float(model["threshold"])),
            "training_rows_ok": len(resources["train_meta"]) == EXPECTED_TRAINING_N == int(model.get("training_n", -1)),
            "training_fingerprint_rows_ok": resources["train_ecfp"].shape == (EXPECTED_TRAINING_N, 2048),
            "gnretain_panel_ok": len(resources["gnretain_bundle"].get("features", [])) == 94,
            "prediction_support_ok": support.get("primary_model") == "GNMIC-EC1792" and bool(support.get("crossfit_scaffold_grouped")),
        }
        status.update(checks)
        status.update({
            "threshold": float(model["threshold"]),
            "n_features": EXPECTED_FEATURES,
            "training_n": EXPECTED_TRAINING_N,
            "model_name": model.get("model_name"),
            "organism": metadata.get("organism"),
            "model_sha256": EXPECTED_MODEL_SHA256,
            "support_version": support.get("version"),
            "classifier_lineage": "Recovered frozen GNMIC-EC1792 primary estimator; separate reconstructed support layer",
            "manuscript_ec1792_estimator_present": True,
        })
        status["ready"] = all(checks.values())
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return bool(status["ready"]), status
