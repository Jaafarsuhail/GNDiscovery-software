from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FOCUSED_FEATURE_MAP = {
    "Acidic pKa": "acidic_pka",
    "Basic pKa": "basic_pka",
    "cLogD7.4": "clogd_ph_7_4",
    "cLogS": "clogs",
    "cLogP": "clogp",
}


def focused_query_values(X: pd.DataFrame, diag: pd.DataFrame) -> dict[str, float]:
    """Collect the six query values required by the focused PCA.

    Formal charge is read directly from the molecular-diagnostics table because it
    is deliberately excluded from the eight-descriptor general diagnostic panel.
    """
    if X is None or not isinstance(X, pd.DataFrame) or X.empty:
        return {}

    vals: dict[str, float] = {}
    for display_name, feature_name in FOCUSED_FEATURE_MAP.items():
        if feature_name in X.columns:
            try:
                value = float(X.iloc[0][feature_name])
            except Exception:
                continue
            if np.isfinite(value):
                vals[display_name] = value

    if diag is not None and isinstance(diag, pd.DataFrame) and not diag.empty:
        required = {"property", "value"}
        if required.issubset(diag.columns):
            charge_rows = diag.loc[
                diag["property"].astype(str).eq("Formal charge"), "value"
            ]
            if len(charge_rows):
                try:
                    charge = float(charge_rows.iloc[0])
                except Exception:
                    charge = np.nan
                if np.isfinite(charge):
                    vals["Formal charge"] = charge

    return vals


def estimated_ionization_class(vals: dict[str, float], ph: float = 7.4) -> str:
    charge = float(vals.get("Formal charge", 0.0))
    acid = float(vals.get("Acidic pKa", np.nan))
    base = float(vals.get("Basic pKa", np.nan))

    if charge > 0:
        return "Formally cationic"
    if charge < 0:
        return "Formally anionic"

    acidic_tendency = np.isfinite(acid) and acid < ph
    basic_tendency = np.isfinite(base) and base > ph
    if acidic_tendency and basic_tendency:
        return "Ampholytic tendency"
    if basic_tendency:
        return "Cationic tendency"
    if acidic_tendency:
        return "Anionic tendency"
    return "Predominantly neutral tendency"


def transform_query_to_focused_pca(
    X: pd.DataFrame,
    diag: pd.DataFrame,
    meta: dict[str, Any],
) -> tuple[float, float] | None:
    vals = focused_query_values(X, diag)
    props = meta.get(
        "properties",
        ["Acidic pKa", "Basic pKa", "cLogD7.4", "cLogS", "cLogP", "Formal charge"],
    )
    if not all(p in vals and np.isfinite(vals[p]) for p in props):
        return None

    x = np.array([float(vals[p]) for p in props], dtype=float)
    mean = np.array(meta.get("mean", [0] * len(props)), dtype=float)
    scale = np.array(meta.get("scale", [1] * len(props)), dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    comps = np.array(meta.get("components", [[1, 0], [0, 1]]), dtype=float)

    if mean.shape[0] != x.shape[0] or scale.shape[0] != x.shape[0]:
        return None
    if comps.ndim != 2 or comps.shape[1] != x.shape[0] or comps.shape[0] < 2:
        return None

    z = (x - mean) / scale
    xy = z @ comps.T
    if not np.all(np.isfinite(xy[:2])):
        return None
    return float(xy[0]), float(xy[1])
