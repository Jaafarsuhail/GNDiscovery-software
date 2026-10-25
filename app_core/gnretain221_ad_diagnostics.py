from __future__ import annotations

from functools import lru_cache
from io import BytesIO

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

try:
    from rdkit.Chem import rdFingerprintGenerator
except Exception:  # pragma: no cover
    rdFingerprintGenerator = None

from gnretain_source_feature_generator import find_project_root, support_assets_root
from gnretain221_prediction_core import EXPECTED_MODEL_FEATURE_COUNT, load_feature_panel, select_model_features

PUBLIC_ENDPOINT_KEY = "gnretention_v05"
PUBLIC_ENDPOINT_LABEL = "Gram-negative retention model"

USER_ENDPOINT_LABELS = {
    PUBLIC_ENDPOINT_KEY: PUBLIC_ENDPOINT_LABEL,
    "gnretention": "Gram-negative retention",
}

CLASS_LABELS = {
    1: "RP",
    0: "RN",
    1.0: "RP",
    0.0: "RN",
    "1": "RP",
    "0": "RN",
    "RP": "RP",
    "RN": "RN",
}


def molecule_png_bytes(smiles: str, size=(420, 300)) -> bytes:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("Invalid SMILES for rendering")
    try:
        from rdkit.Chem import Draw

        image = Draw.MolToImage(mol, size=size)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "RDKit PNG molecule drawing is unavailable in this environment."
        ) from exc


def molecule_svg_markup(smiles: str, size=(420, 300)) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("Invalid SMILES for rendering")
    try:
        from rdkit.Chem.Draw import rdMolDraw2D

        width, height = int(size[0]), int(size[1])
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.replace('svg:', '')
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "RDKit SVG molecule drawing is unavailable in this environment."
        ) from exc


def _mol(smiles):
    return Chem.MolFromSmiles(str(smiles))


def molecular_diagnostics(smiles: str) -> pd.DataFrame:
    mol = _mol(smiles)
    if mol is None:
        return pd.DataFrame(
            [{"group": "structure", "property": "valid_structure", "value": False, "interpretation": "SMILES could not be parsed"}]
        )
    values = [
        ("basic physicochemistry", "Molecular weight", float(Descriptors.MolWt(mol)), "Size descriptor; extreme values may reduce prediction support"),
        ("basic physicochemistry", "RDKit cLogP", float(Crippen.MolLogP(mol)), "Lipophilicity estimate"),
        ("basic physicochemistry", "TPSA", float(rdMolDescriptors.CalcTPSA(mol)), "Polar surface area"),
        ("hydrogen bonding", "H-bond donors", int(Lipinski.NumHDonors(mol)), "Hydrogen-bond donor count"),
        ("hydrogen bonding", "H-bond acceptors", int(Lipinski.NumHAcceptors(mol)), "Hydrogen-bond acceptor count"),
        ("flexibility/shape", "Rotatable bonds", int(Lipinski.NumRotatableBonds(mol)), "Molecular flexibility"),
        ("flexibility/shape", "Fraction Csp3", float(rdMolDescriptors.CalcFractionCSP3(mol)), "Saturation / three-dimensionality proxy"),
        ("rings/aromaticity", "Aromatic rings", int(rdMolDescriptors.CalcNumAromaticRings(mol)), "Aromatic-ring count"),
        ("rings/aromaticity", "Saturated rings", int(rdMolDescriptors.CalcNumSaturatedRings(mol)), "Saturated-ring count"),
        ("size", "Heavy atoms", int(mol.GetNumHeavyAtoms()), "Non-hydrogen atom count"),
        ("charge", "Formal charge", int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())), "Formal molecular charge"),
    ]
    return pd.DataFrame(values, columns=["group", "property", "value", "interpretation"])


def _fingerprint(smiles: str):
    mol = _mol(smiles)
    if mol is None:
        return None
    if rdFingerprintGenerator is not None:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return generator.GetFingerprint(mol)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


@lru_cache(maxsize=4)
def _training_rows(root_str: str) -> pd.DataFrame:
    root = find_project_root(root_str)
    path = support_assets_root(root) / "metadata" / "GNRetain221_training_rows.csv"
    table = pd.read_csv(path)
    table = table.rename(columns={"label_2350": "label", "label_retention": "label"})
    table["endpoint"] = PUBLIC_ENDPOINT_KEY
    df = table[["compound_id", "smiles", "runtime_smiles", "label", "endpoint"]].copy()
    df["label"] = df["label"].map(CLASS_LABELS).fillna(df["label"])
    df["_fp"] = [_fingerprint(smi) for smi in df["runtime_smiles"].astype(str)]
    return df[df["_fp"].notna()].reset_index(drop=True)


def _nearest_neighbors(smiles: str, project_root=None, n: int = 20) -> pd.DataFrame:
    root = find_project_root(project_root)
    query_fp = _fingerprint(smiles)
    if query_fp is None:
        return pd.DataFrame()
    df = _training_rows(str(root))
    try:
        similarities = list(DataStructs.BulkTanimotoSimilarity(query_fp, list(df["_fp"])))
    except Exception:
        similarities = []
        for fingerprint in df["_fp"]:
            try:
                similarities.append(float(DataStructs.TanimotoSimilarity(query_fp, fingerprint)))
            except Exception:
                similarities.append(0.0)
    output = df.drop(columns=["_fp"]).copy()
    output["tanimoto_similarity"] = similarities
    output = output.sort_values("tanimoto_similarity", ascending=False).head(n).reset_index(drop=True)
    output["rank"] = np.arange(1, len(output) + 1)
    return output


@lru_cache(maxsize=4)
def _load_quantiles(root_str: str) -> pd.DataFrame:
    root = find_project_root(root_str)
    path = support_assets_root(root) / "ad_reference" / "GNRetain221_feature_range_quantiles.csv"
    table = pd.read_csv(path)
    required = {"feature", "q0", "q1", "q5", "q95", "q99", "q100"}
    if not required.issubset(table.columns):
        raise ValueError(f"GNRetain AD quantile table is missing columns: {sorted(required - set(table.columns))}")
    panel = load_feature_panel(str(root))
    features = table["feature"].astype(str).tolist()
    if features != panel or len(features) != EXPECTED_MODEL_FEATURE_COUNT:
        raise ValueError("GNRetain AD quantile features differ from the ordered 221-feature model panel")
    return table.set_index("feature")


def range_violation_table(X: pd.DataFrame, project_root=None) -> pd.DataFrame:
    root = find_project_root(project_root)
    Xmodel = select_model_features(X, root)
    quantiles = _load_quantiles(str(root))
    rows = []
    for feature in load_feature_panel(str(root)):
        value = float(Xmodel.iloc[0][feature])
        row = quantiles.loc[feature]
        severity = "inside"
        note = "inside training range"
        if value < float(row.q0) or value > float(row.q100):
            severity = "severe"
            note = "outside observed training min/max"
        elif value < float(row.q1) or value > float(row.q99):
            severity = "moderate"
            note = "outside p01-p99 training range"
        elif value < float(row.q5) or value > float(row.q95):
            severity = "borderline"
            note = "outside p05-p95 training range"
        rows.append(
            {
                "endpoint": PUBLIC_ENDPOINT_KEY,
                "feature": feature,
                "value": value,
                "q0": float(row.q0),
                "q1": float(row.q1),
                "q5": float(row.q5),
                "q95": float(row.q95),
                "q99": float(row.q99),
                "q100": float(row.q100),
                "severity": severity,
                "note": note,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_MODEL_FEATURE_COUNT:
        raise RuntimeError(f"AD evaluated {len(result)} features instead of 221")
    return result


def applicability_domain_summary(X: pd.DataFrame, smiles: str, endpoint_key_or_root=None, project_root=None):
    """Evaluate support using Morgan similarity and the 221 model-feature ranges.

    The optional endpoint argument is accepted for backward compatibility with
    earlier app calls but does not change the single GNRetain endpoint.
    """
    if project_root is None:
        project_root = endpoint_key_or_root
    root = find_project_root(project_root)
    neighbors = _nearest_neighbors(smiles, root, n=20)
    similarity = float(neighbors.iloc[0]["tanimoto_similarity"]) if len(neighbors) else 0.0
    flags = range_violation_table(X, root)
    severe = int((flags["severity"] == "severe").sum())
    moderate = int((flags["severity"] == "moderate").sum())
    borderline = int((flags["severity"] == "borderline").sum())

    # These cutoffs were rechecked against the 5,452-row V05 distribution after
    # restricting range diagnostics to the 221 deployed model inputs. In the
    # training set, 10 moderate flags correspond approximately to the 95th
    # percentile and 36 to the 99.5th percentile.
    if similarity >= 0.45 and severe == 0 and moderate <= 10:
        category = "inside domain"
    elif similarity >= 0.30 and severe <= 2 and moderate <= 36:
        category = "borderline domain"
    else:
        category = "outside domain"

    chemistry_status = (
        "known/near training chemistry"
        if similarity >= 0.45
        else ("related/less common" if similarity >= 0.30 else "novel or poorly supported")
    )
    warning = {
        "inside domain": "Strong training support from similar chemistry and the 221 model-feature ranges.",
        "borderline domain": "Moderate training support based on similarity and the 221 model-feature ranges.",
        "outside domain": "Low training support because the molecule is less represented in the GNRetain chemical domain.",
    }[category]
    summary = {
        "ad_category": category,
        "nearest_neighbor_similarity": round(similarity, 3),
        "training_chemistry_status": chemistry_status,
        "ad_feature_space": "GNRetain cleaned 221-feature model panel",
        "ad_features_evaluated": EXPECTED_MODEL_FEATURE_COUNT,
        "n_descriptor_range_severe": severe,
        "n_descriptor_range_moderate": moderate,
        "n_descriptor_range_borderline": borderline,
        "ad_warning": warning,
    }
    return summary, neighbors, flags



def batch_applicability_domain_summaries(
    X: pd.DataFrame,
    smiles_list: list[str],
    project_root=None,
    num_threads: int = 1,
) -> pd.DataFrame:
    """Vectorized GNRetain support summary for batch screening.

    This is scientifically equivalent to calling ``applicability_domain_summary``
    row-by-row for the fields used by the batch output, but avoids constructing
    221-row flag tables and 20-neighbour tables for every molecule.
    """
    root = find_project_root(project_root)
    if len(X) != len(smiles_list):
        raise ValueError("Feature and SMILES row counts do not match for batch applicability support.")
    if len(X) == 0:
        return pd.DataFrame(columns=[
            "ad_category", "nearest_neighbor_similarity", "training_chemistry_status",
            "ad_feature_space", "ad_features_evaluated", "n_descriptor_range_severe",
            "n_descriptor_range_moderate", "n_descriptor_range_borderline", "ad_warning",
        ])

    Xmodel = select_model_features(X, root)
    quantiles = _load_quantiles(str(root)).loc[load_feature_panel(str(root))]
    values = Xmodel.to_numpy(dtype=float)
    q0 = quantiles["q0"].to_numpy(dtype=float)
    q1 = quantiles["q1"].to_numpy(dtype=float)
    q5 = quantiles["q5"].to_numpy(dtype=float)
    q95 = quantiles["q95"].to_numpy(dtype=float)
    q99 = quantiles["q99"].to_numpy(dtype=float)
    q100 = quantiles["q100"].to_numpy(dtype=float)

    severe_mask = (values < q0) | (values > q100)
    moderate_mask = (~severe_mask) & ((values < q1) | (values > q99))
    borderline_mask = (~severe_mask) & (~moderate_mask) & ((values < q5) | (values > q95))
    severe = severe_mask.sum(axis=1).astype(int)
    moderate = moderate_mask.sum(axis=1).astype(int)
    borderline = borderline_mask.sum(axis=1).astype(int)

    training = _training_rows(str(root))
    train_fps = list(training["_fp"])
    query_mols = [Chem.MolFromSmiles(str(s)) for s in smiles_list]
    if any(m is None for m in query_mols):
        bad = next(i for i, m in enumerate(query_mols) if m is None)
        raise ValueError(f"Invalid standardized SMILES at batch support row {bad}: {smiles_list[bad]}")
    if rdFingerprintGenerator is not None:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        query_fps = generator.GetFingerprints(query_mols, numThreads=max(1, int(num_threads or 1)))
    else:  # pragma: no cover
        query_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in query_mols]

    similarities = np.zeros(len(query_fps), dtype=float)
    for i, qfp in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(qfp, train_fps)
        similarities[i] = max(sims) if sims else 0.0

    categories = []
    chemistry = []
    warnings = []
    for sim, sev, mod in zip(similarities, severe, moderate):
        if sim >= 0.45 and sev == 0 and mod <= 10:
            category = "inside domain"
        elif sim >= 0.30 and sev <= 2 and mod <= 36:
            category = "borderline domain"
        else:
            category = "outside domain"
        categories.append(category)
        chemistry.append(
            "known/near training chemistry" if sim >= 0.45
            else ("related/less common" if sim >= 0.30 else "novel or poorly supported")
        )
        warnings.append({
            "inside domain": "Strong training support from similar chemistry and the 221 model-feature ranges.",
            "borderline domain": "Moderate training support based on similarity and the 221 model-feature ranges.",
            "outside domain": "Low training support because the molecule is less represented in the GNRetain chemical domain.",
        }[category])

    return pd.DataFrame({
        "ad_category": categories,
        "nearest_neighbor_similarity": np.round(similarities, 3),
        "training_chemistry_status": chemistry,
        "ad_feature_space": "GNRetain cleaned 221-feature model panel",
        "ad_features_evaluated": EXPECTED_MODEL_FEATURE_COUNT,
        "n_descriptor_range_severe": severe,
        "n_descriptor_range_moderate": moderate,
        "n_descriptor_range_borderline": borderline,
        "ad_warning": warnings,
    })

def projection_from_features(X: pd.DataFrame, project_root=None) -> pd.DataFrame:
    return pd.DataFrame()


def _prob_category(probability: float, threshold: float, ad_category: str) -> str:
    if "outside" in str(ad_category).lower():
        return "low training support"
    if abs(probability - threshold) <= 0.10:
        return "borderline"
    if probability >= threshold + 0.15:
        return "high-confidence retained"
    if probability <= threshold - 0.15:
        return "high-confidence non-retained"
    return "moderate-confidence retained" if probability >= threshold else "moderate-confidence non-retained"


def triage_scores(flat_pred: pd.DataFrame, ad_summary: dict, diag: pd.DataFrame | dict) -> dict:
    if flat_pred is None or flat_pred.empty:
        raise ValueError("No prediction rows were supplied to triage_scores")
    row = flat_pred.iloc[0]
    probability = float(row["proba_clean_ensemble"] if "proba_clean_ensemble" in row else row["proba_retention_positive"])
    threshold = float(row.get("threshold", 0.5))
    predicted_class = str(row.get("class_label", "RP" if probability >= threshold else "RN"))
    ad = str(ad_summary.get("ad_category", "")).lower()
    ad_weight = 1.0 if "inside" in ad else (0.7 if "borderline" in ad else 0.35)
    confidence_weight = 0.65 if abs(probability - threshold) <= 0.10 else 1.0
    overall = float(probability * ad_weight * confidence_weight)
    probability_category = _prob_category(probability, threshold, ad_summary.get("ad_category", ""))

    if "outside" in ad:
        priority = "exploratory"
    elif probability_category == "high-confidence retained":
        priority = "high priority"
    elif predicted_class == "RP":
        priority = "medium priority"
    elif probability_category == "borderline":
        priority = "review"
    else:
        priority = "low priority"

    if "outside" in ad:
        interpretation = (
            "The molecule is less represented in the training chemical domain. "
            "The prediction is therefore presented with low training support."
        )
    elif predicted_class == "RP":
        interpretation = (
            "The molecule is predicted to be retention-positive for the Gram-negative retention endpoint. "
            "This supports prioritization for retention testing or antibacterial follow-up."
        )
    else:
        interpretation = (
            "The molecule is predicted to be retention-negative for the Gram-negative retention endpoint. "
            "This suggests weaker cell-associated retention under the trained domain."
        )

    return {
        "Gram_negative_retention_probability": round(probability, 6),
        "Gram_negative_retention_threshold": round(threshold, 6),
        "Gram_negative_retention_class": predicted_class,
        "retention_probability_category": probability_category,
        "heuristic_prioritization_score": round(overall, 3),
        "priority_category": priority,
        "endpoint_interpretation": interpretation,
        "prioritization_score_note": "Ranking aid only; not a calibrated probability or validated biological endpoint.",
    }


def simple_markdown_report(payload: dict) -> str:
    lines = ["# GNRetain Single-Molecule Prediction Report", ""]
    for key, value in payload.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) + "\n"
