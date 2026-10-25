#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GNDiscovery: E. coli — modular visual Streamlit decision-support platform.

GNRetain and GNMIC are presented as separate subapps, with an optional integrated comparison page.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import os
import shutil
import sys
import time
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore", message=r".*If you are loading a serialized model.*", category=UserWarning)

APP_DIR = Path(__file__).resolve().parent
APP_ROOT = APP_DIR.parent
CORE_DIR = APP_ROOT / "app_core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from gnretain_source_feature_generator import (  # noqa: E402
    batch_generate_runtime_source_features,
    find_project_root,
)
from gnretain221_prediction_core import (  # noqa: E402
    PUBLIC_ENDPOINT_KEY,
    predict_from_features,
)
from gnretain221_ad_diagnostics import (  # noqa: E402
    applicability_domain_summary,
    batch_applicability_domain_summaries,
    molecule_png_bytes,
    molecule_svg_markup,
    molecular_diagnostics,
    triage_scores,
)
from gnretain221_prediction_core import inspect_model_bundles  # noqa: E402
from gnmic_ecoli_prediction_core import (  # noqa: E402
    gnmic_runtime_health_check,
    predict_gnmic_from_source_features,
)
from focused_visualization import (  # noqa: E402
    estimated_ionization_class,
    focused_query_values,
    transform_query_to_focused_pca,
)

st.set_page_config(
    page_title="GNDiscovery: E. coli",
    page_icon="GN",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "GNDiscovery: E. coli"
AMOXICILLIN_SMILES = "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"
NAVIGATION = [
    "Home",
    "GNRetain",
    "GNMIC",
    "Integrated Assessment",
    "Batch Screening",
    "Model Information",
    "Reports",
    "About",
]
APP_CITATION = (
    "Wadi, J. S. (2026). GNDiscovery: E. coli, a modular decision-support "
    "platform for E. coli retention-profile and MIC-activity prediction. "
    "AG Keusgen, Institute of Pharmaceutical Chemistry, "
    "Philipps-Universitaet Marburg, Germany."
)

DIAGNOSTIC_PROPERTIES = [
    "Molecular weight",
    "RDKit cLogP",
    "TPSA",
    "H-bond donors",
    "H-bond acceptors",
    "Rotatable bonds",
    "Fraction Csp3",
    "Aromatic rings",
]



def inject_css() -> None:
    st.markdown(
        """
<style>
:root {
  --navy:#0b2438; --ink:#143044; --muted:#66798a; --line:#dbe6ee;
  --soft:#f5f9fb; --teal:#128f8b; --teal-soft:#eaf8f6;
  --orange:#e86d10; --orange-soft:#fff3e9; --purple:#7551c7;
  --purple-soft:#f3effc; --blue:#2b78a6;
}
html, body, [class*="css"] { font-family: Inter, "Segoe UI", Arial, sans-serif; }
.stApp { background:linear-gradient(180deg,#f8fbfd 0%,#eef5f8 100%); color:var(--ink); }
header[data-testid="stHeader"] { background:rgba(248,251,253,.95); border-bottom:1px solid rgba(219,230,238,.75); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#061725 0%,#0a2a3c 100%); }
[data-testid="stSidebar"] * { color:#edf7fb; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  border:1px solid rgba(255,255,255,.09); border-radius:10px; padding:.62rem .75rem;
  margin-bottom:.28rem; background:rgba(255,255,255,.035);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background:linear-gradient(90deg,rgba(18,143,139,.96),rgba(43,120,166,.92));
  border-color:rgba(61,204,194,.8); box-shadow:0 8px 20px rgba(0,0,0,.18);
}
.block-container { padding-top:3.7rem !important; padding-bottom:3rem; max-width:1320px; }
h1,h2,h3 { color:#102a43; }
.gn-logo { border-bottom:1px solid rgba(255,255,255,.12); padding:.2rem 0 1rem; margin-bottom:.85rem; }
.gn-logo-title { margin:0; color:white; font-size:1.3rem; font-weight:800; }
.gn-logo-sub { margin:.25rem 0 0; color:#b5dce5; font-size:.78rem; line-height:1.45; }
.gn-header { display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; margin-bottom:1.1rem; }
.gn-title { font-size:2.05rem; line-height:1.05; font-weight:820; color:#102a43; }
.gn-subtitle { margin-top:.35rem; color:#526879; max-width:850px; line-height:1.45; }
.gn-scope { display:inline-flex; border-radius:999px; padding:.36rem .72rem; font-size:.77rem;
  font-weight:760; white-space:nowrap; background:#e8f7f4; color:#0b625e; border:1px solid #c4ebe5; }
.gn-workflow-card { background:white; border:1px solid var(--line); border-radius:16px; padding:1.15rem;
  min-height:205px; box-shadow:0 12px 28px rgba(20,48,68,.07); }
.gn-workflow-card.teal { border-top:5px solid var(--teal); }
.gn-workflow-card.orange { border-top:5px solid var(--orange); }
.gn-workflow-card.purple { border-top:5px solid var(--purple); }
.gn-workflow-kicker { font-size:.74rem; text-transform:uppercase; letter-spacing:.045em; font-weight:800; color:var(--muted); }
.gn-workflow-title { margin:.25rem 0 .35rem; color:#102a43; font-size:1.45rem; font-weight:820; }
.gn-workflow-text { color:#607487; font-size:.9rem; line-height:1.45; }
.gn-model-card { background:white; border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.15rem;
  box-shadow:0 10px 25px rgba(20,48,68,.065); min-height:235px; }
.gn-model-card.teal { border-top:5px solid var(--teal); }
.gn-model-card.orange { border-top:5px solid var(--orange); }
.gn-model-card.purple { border-top:5px solid var(--purple); }
.gn-model-name { font-size:.76rem; text-transform:uppercase; letter-spacing:.04em; color:#617586; font-weight:800; }
.gn-model-context { font-size:.78rem; color:#738596; margin-top:.12rem; }
.gn-model-value { font-size:1.75rem; line-height:1.08; color:#102a43; font-weight:830; margin:.55rem 0 .25rem; overflow-wrap:normal; word-break:normal; }
.gn-model-stat { color:#53697a; font-size:.86rem; line-height:1.48; }
.gn-support-badge { display:inline-block; border-radius:999px; padding:.23rem .55rem; font-size:.75rem; font-weight:780; margin-top:.55rem; }
.gn-support-badge.teal { background:var(--teal-soft); color:#0a6762; }
.gn-support-badge.orange { background:var(--orange-soft); color:#a94605; }
.gn-callout { border-radius:12px; padding:1rem 1.1rem; margin:.75rem 0 1rem; border:1px solid #cfe8e5; background:#effaf8; }
.gn-callout.purple { border-color:#ddd2f4; background:#f5f1fc; }
.gn-callout.orange { border-color:#f2d6bd; background:#fff6ee; }
.gn-callout-title { font-weight:820; color:#102a43; margin-bottom:.25rem; }
.gn-callout-text { color:#4d6577; line-height:1.48; font-size:.91rem; }
.gn-prob-box { background:white; border:1px solid var(--line); border-radius:12px; padding:.95rem 1rem; margin:.8rem 0; }
.gn-prob-heading { color:#102a43; font-size:1.02rem; font-weight:820; }
.gn-prob-value-row { display:flex; align-items:baseline; gap:.38rem; color:#53697a; font-size:.88rem; margin-top:.36rem; }
.gn-prob-value-row strong { color:#143044; font-size:1rem; font-variant-numeric:tabular-nums; }
.gn-bar { height:13px; background:#dce8ef; border-radius:999px; position:relative; overflow:visible; margin:.62rem 0 0; }
.gn-bar-fill { height:100%; border-radius:999px; }
.gn-bar-fill.teal { background:linear-gradient(90deg,#26b8aa,#2781aa); }
.gn-bar-fill.orange { background:linear-gradient(90deg,#f29b4b,#de5c07); }
.gn-marker { position:absolute; top:-4px; width:2px; height:21px; background:#18364b; transform:translateX(-1px); }
.gn-bar-scale { position:relative; height:2.65rem; margin-top:.3rem; color:#6a7c8c; font-size:.76rem; }
.gn-scale-start { position:absolute; left:0; top:0; }
.gn-scale-end { position:absolute; right:0; top:0; }
.gn-threshold-label { position:absolute; top:0; transform:translateX(-50%); text-align:center; color:#526879; line-height:1.25; white-space:nowrap; }
.gn-threshold-label strong { display:block; color:#143044; font-variant-numeric:tabular-nums; }
@media (max-width: 650px) {
  .gn-threshold-label { font-size:.7rem; }
}
.gn-reason-list { display:grid; gap:.55rem; margin:.45rem 0; }
.gn-reason-row { display:grid; grid-template-columns:minmax(150px,1fr) 100px minmax(240px,2fr); gap:.8rem; align-items:center; padding:.72rem .8rem; background:#fff; border:1px solid var(--line); border-radius:10px; }
.gn-reason-property { font-weight:800; color:#102a43; }
.gn-reason-value { color:#2b5973; font-variant-numeric:tabular-nums; }
.gn-reason-text { color:#586d7d; line-height:1.4; }
.gn-section-note { color:#607487; font-size:.87rem; margin-top:-.25rem; margin-bottom:.8rem; line-height:1.45; }
.gn-analogue-meta { color:#586d7d; font-size:.79rem; line-height:1.48; margin-top:.2rem; }
.gn-footer { color:#6c7f8e; font-size:.8rem; border-top:1px solid var(--line); padding-top:.85rem; margin-top:1.5rem; }
.stButton>button { border-radius:9px; border:1px solid #cad9e4; min-height:2.45rem; font-weight:720; }
.stButton>button[kind="primary"] { border:0; background:linear-gradient(90deg,#128f8b,#2b78a6); }
div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:11px; padding:.7rem .8rem; box-shadow:0 7px 18px rgba(20,48,68,.05); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }

.gn-structure-frame { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.55rem; box-shadow:0 7px 18px rgba(20,48,68,.04); }
.gn-pill { display:inline-block; border-radius:999px; padding:.18rem .5rem; border:1px solid #cddce8; font-size:.74rem; }
.gn-visual-title { color:#102a43; font-weight:820; font-size:1.02rem; margin:.25rem 0 .45rem; }
.gn-legend-row { display:flex; gap:1.3rem; align-items:center; flex-wrap:wrap; color:#53697a; font-size:.82rem; margin:.2rem 0 .55rem; }
.gn-dot { width:.72rem; height:.72rem; border-radius:50%; display:inline-block; margin-right:.3rem; }
.gn-neighbour-card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.6rem; min-height:255px; box-shadow:0 7px 18px rgba(20,48,68,.04); }
.gn-neighbour-card.active { border-top:4px solid #e86d10; }
.gn-neighbour-card.inactive { border-top:4px solid #2b78a6; }

@media (max-width:1000px) {
  .block-container { padding-left:1.1rem !important; padding-right:1.1rem !important; }
  div[data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:.75rem; }
  div[data-testid="column"] { min-width:280px; flex:1 1 300px !important; }
  .gn-header { flex-direction:column; }
}
@media (max-width:650px) {
  .gn-reason-row { grid-template-columns:1fr; gap:.2rem; }
  .gn-title { font-size:1.65rem; }
  .gn-model-value { font-size:1.42rem; }
  div[data-testid="column"] { min-width:100%; flex:1 1 100% !important; }
  .gn-small-row { flex-direction:column; gap:.15rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def safe_root_default() -> str:
    try:
        return str(find_project_root())
    except Exception:
        return str(APP_ROOT)


@st.cache_data(show_spinner=False)
def cached_shared_features(smiles: str, root_text: str) -> tuple[dict[str, Any], pd.DataFrame]:
    X, reports = batch_generate_runtime_source_features([str(smiles)], project_root=root_text)
    if reports.empty or str(reports.iloc[0].get("status", "failed")) != "ok" or X.empty:
        error = reports.iloc[0].get("error", "Feature generation failed") if len(reports) else "Feature generation failed"
        raise ValueError(str(error))
    return reports.iloc[0].to_dict(), X.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_prediction(smiles: str, root_text: str) -> dict[str, Any]:
    feature_report, X = cached_shared_features(smiles, root_text)
    pred_wide = predict_from_features(X, project_root=root_text)
    row = pred_wide.iloc[0]
    flat = pd.DataFrame([{
        "endpoint_key": PUBLIC_ENDPOINT_KEY,
        "proba_clean_ensemble": row["gnretention__proba_clean_ensemble"],
        "proba_retention_positive": row["gnretention__proba_retention_positive"],
        "threshold": row["gnretention__threshold"],
        "class_label": row["gnretention__class_label"],
    }])
    runtime_smiles = str(feature_report.get("runtime_smiles", smiles))
    ad_summary, retention_neighbors, range_flags = applicability_domain_summary(
        X, runtime_smiles, project_root=root_text
    )
    diagnostics = molecular_diagnostics(runtime_smiles)
    scores = triage_scores(flat, ad_summary, diagnostics)
    mic_table, mic_neighbors = predict_gnmic_from_source_features(
        [runtime_smiles], X, project_root=root_text, top_n_neighbours=10
    )
    return {
        "feature_report": feature_report,
        "X": X,
        "pred_wide": pred_wide,
        "flat": flat,
        "ad_summary": ad_summary,
        "retention_neighbors": retention_neighbors,
        "range_flags": range_flags,
        "diagnostics": diagnostics,
        "scores": scores,
        "mic": mic_table.iloc[0].to_dict(),
        "mic_neighbors": mic_neighbors[0] if mic_neighbors else pd.DataFrame(),
    }


@st.cache_data(show_spinner=False)
def cached_health(root_text: str) -> tuple[bool, dict[str, Any], bool, dict[str, Any]]:
    try:
        retention = inspect_model_bundles(root_text)
        if len(retention):
            retention_status = retention.iloc[0].to_dict()
            retention_ready = str(retention_status.get("status", "")).lower() == "ok"
        else:
            retention_ready = False
            retention_status = {"status": "missing", "error": "No GNRetain model bundle was found."}
        retention_status["ready"] = retention_ready
    except Exception as exc:
        retention_ready = False
        retention_status = {"ready": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "threshold": 0.575}
    mic_ready, mic_status = gnmic_runtime_health_check(root_text)
    return retention_ready, retention_status, mic_ready, mic_status


@st.cache_data(show_spinner=False)
def load_diagnostic_reference(root_text: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = find_project_root(root_text)
    ref_dir = root / "GNRetain221_Support_Assets" / "ad_reference"
    ref = pd.read_csv(ref_dir / "diagnostic_descriptor_reference.csv")
    quantiles = pd.read_csv(ref_dir / "diagnostic_descriptor_quantiles.csv")
    coords = pd.read_csv(ref_dir / "diagnostic_pca_reference.csv")
    meta_path = ref_dir / "diagnostic_pca_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return ref, quantiles, coords, meta


@st.cache_data(show_spinner=False)
def load_focused_reference(root_text: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = find_project_root(root_text)
    ref_dir = root / "GNRetain221_Support_Assets" / "ad_reference"
    ref = pd.read_csv(ref_dir / "focused_descriptor_reference.csv")
    quantiles = pd.read_csv(ref_dir / "focused_descriptor_quantiles.csv")
    coords = pd.read_csv(ref_dir / "focused_pca_reference.csv")
    meta_path = ref_dir / "focused_pca_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return ref, quantiles, coords, meta




def diagnostic_value_map(diag: pd.DataFrame) -> dict[str, float]:
    vals: dict[str, float] = {}
    if diag is None or not len(diag):
        return vals
    for _, row in diag.iterrows():
        prop = str(row.get("property", ""))
        if prop in DIAGNOSTIC_PROPERTIES:
            try:
                vals[prop] = float(row.get("value"))
            except Exception:
                pass
    return vals


def diagnostic_percentile_rows(diag: pd.DataFrame, root_text: str) -> pd.DataFrame:
    ref, quantiles, _, _ = load_diagnostic_reference(root_text)
    vals = diagnostic_value_map(diag)
    rows = []
    for prop in DIAGNOSTIC_PROPERTIES:
        if prop not in vals or prop not in ref.columns:
            continue
        q = quantiles[quantiles["property"].eq(prop)]
        if q.empty:
            continue
        qrow = q.iloc[0]
        training_vals = pd.to_numeric(ref[prop], errors="coerce").dropna().values
        query_value = float(vals[prop])
        pct = float((training_vals <= query_value).mean() * 100.0) if len(training_vals) else np.nan
        if query_value < float(qrow["q00"]):
            pct = 0.0
        elif query_value > float(qrow["q100"]):
            pct = 100.0
        if query_value < float(qrow["q01"]) or query_value > float(qrow["q99"]):
            status = "strong outlier"
        elif query_value < float(qrow["q05"]) or query_value > float(qrow["q95"]):
            status = "tail region"
        elif query_value < float(qrow["q25"]) or query_value > float(qrow["q75"]):
            status = "supported range"
        else:
            status = "typical range"
        direction = "inside"
        if query_value > float(qrow["q95"]):
            direction = "above q95"
        elif query_value < float(qrow["q05"]):
            direction = "below q05"
        rows.append(
            {
                "property": prop,
                "query_value": query_value,
                "query_percentile": round(pct, 2),
                "status": status,
                "direction": direction,
                "q05_value": float(qrow["q05"]),
                "q50_value": float(qrow["q50"]),
                "q95_value": float(qrow["q95"]),
                "n_training": int(qrow.get("n", len(training_vals))),
            }
        )
    return pd.DataFrame(rows)


def descriptor_percentile_interval_plot(diag: pd.DataFrame, root_text: str) -> pd.DataFrame:
    rows = diagnostic_percentile_rows(diag, root_text)
    if rows.empty:
        st.info("Descriptor percentile diagnostics are unavailable for this molecule.")
        return rows
    plot_rows = rows.copy()
    plot_rows["q05_pct"] = 5
    plot_rows["q95_pct"] = 95
    plot_rows["q25_pct"] = 25
    plot_rows["q75_pct"] = 75
    plot_rows["median_pct"] = 50
    plot_rows["property"] = pd.Categorical(plot_rows["property"], categories=DIAGNOSTIC_PROPERTIES[::-1], ordered=True)
    values = plot_rows.astype({"property": str}).to_dict("records")
    spec = {
        "height": 330,
        "layer": [
            {"data": {"values": values}, "mark": {"type": "bar", "size": 18, "color": "#dbe7ef"}, "encoding": {"y": {"field": "property", "type": "ordinal", "sort": DIAGNOSTIC_PROPERTIES[::-1], "axis": {"title": None}}, "x": {"field": "q05_pct", "type": "quantitative", "scale": {"domain": [0, 100]}, "axis": {"title": "Position relative to training compounds"}}, "x2": {"field": "q95_pct"}}},
            {"data": {"values": values}, "mark": {"type": "bar", "size": 18, "color": "#9fe2d9"}, "encoding": {"y": {"field": "property", "type": "ordinal", "sort": DIAGNOSTIC_PROPERTIES[::-1]}, "x": {"field": "q25_pct", "type": "quantitative"}, "x2": {"field": "q75_pct"}}},
            {"data": {"values": values}, "mark": {"type": "tick", "color": "#0b2638", "thickness": 2, "size": 24}, "encoding": {"y": {"field": "property", "type": "ordinal", "sort": DIAGNOSTIC_PROPERTIES[::-1]}, "x": {"field": "median_pct", "type": "quantitative"}}},
            {"data": {"values": values}, "mark": {"type": "point", "filled": True, "size": 120, "stroke": "#0b2638", "strokeWidth": 1.4}, "encoding": {"y": {"field": "property", "type": "ordinal", "sort": DIAGNOSTIC_PROPERTIES[::-1]}, "x": {"field": "query_percentile", "type": "quantitative"}, "color": {"field": "status", "type": "nominal", "scale": {"domain": ["typical range", "supported range", "tail region", "strong outlier"], "range": ["#16b8a6", "#2374ab", "#f0a202", "#cf2e2e"]}, "legend": {"title": "How typical is the molecule?"}}, "tooltip": [{"field": "property", "type": "nominal", "title": "Property"}, {"field": "query_value", "type": "quantitative", "format": ".3f", "title": "Molecule value"}, {"field": "query_percentile", "type": "quantitative", "format": ".1f", "title": "Position vs training dataset (%)"}, {"field": "status", "type": "nominal", "title": "Support level"}]}}],
        "config": {"view": {"stroke": "#dde7ef"}, "axis": {"gridColor": "#edf3f7", "labelColor": "#143044", "titleColor": "#143044"}},
    }
    st.vega_lite_chart(spec, use_container_width=True)
    st.caption("The chart shows where this molecule sits compared with the training compounds. Points near the center are better supported; points near the edges are less typical.")
    return rows


def diagnostic_explanation_card(percentile_rows: pd.DataFrame) -> None:
    if percentile_rows is None or percentile_rows.empty:
        return
    out = percentile_rows[percentile_rows["direction"].ne("inside")].copy()
    st.markdown('<div class="gn-visual-title">Properties contributing most to limited support</div>', unsafe_allow_html=True)
    if out.empty:
        st.success("No major diagnostic property lies outside the usual GNRetain training range.")
        return
    out["exceedance"] = np.where(out["query_percentile"] > 95, out["query_percentile"] - 95, 5 - out["query_percentile"])
    out = out.sort_values("exceedance", ascending=False).head(3)
    rows=[]
    for _, r in out.iterrows():
        pct=float(r['query_percentile']); qv=float(r['query_value'])
        direction = "Higher than usual" if pct > 95 else "Lower than usual"
        explanation = f"At approximately the {pct:.0f}th percentile of the training distribution."
        rows.append(f'<div class="gn-reason-row"><div class="gn-reason-property">{html.escape(str(r["property"]))}</div><div class="gn-reason-value">{qv:.3g}</div><div class="gn-reason-text"><strong>{direction}</strong><br>{explanation}</div></div>')
    st.markdown('<div class="gn-reason-list">'+''.join(rows)+'</div>', unsafe_allow_html=True)


def transform_query_to_diagnostic_pca(diag: pd.DataFrame, meta: dict) -> tuple[float, float] | None:
    vals = diagnostic_value_map(diag)
    props = meta.get("properties", DIAGNOSTIC_PROPERTIES)
    if not all(p in vals for p in props):
        return None
    x = np.array([float(vals[p]) for p in props], dtype=float)
    mean = np.array(meta.get("mean", [0] * len(props)), dtype=float)
    scale = np.array(meta.get("scale", [1] * len(props)), dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    comps = np.array(meta.get("components", [[1, 0], [0, 1]]), dtype=float)
    z = (x - mean) / scale
    xy = z @ comps.T
    return float(xy[0]), float(xy[1])


def chemical_space_support_map(diag: pd.DataFrame, nn: pd.DataFrame, root_text: str) -> None:
    try:
        _, _, coords, meta = load_diagnostic_reference(root_text)
        qxy = transform_query_to_diagnostic_pca(diag, meta)
    except Exception as e:
        st.info(f"Chemical-space support map is unavailable: {type(e).__name__}: {e}")
        return
    if coords.empty or qxy is None:
        st.info("Chemical-space support map is unavailable for this query.")
        return
    bg = coords.copy()
    if len(bg) > 3000:
        bg = bg.sample(3000, random_state=42)
    bg_values = bg[["pc1", "pc2", "compound_id", "runtime_smiles"]].fillna("").to_dict("records")
    nn_values = []
    if isinstance(nn, pd.DataFrame) and len(nn) and "runtime_smiles" in nn.columns:
        top = nn.head(8).copy()
        merged = top.merge(coords, on="runtime_smiles", how="left", suffixes=("", "_coord"))
        for _, r in merged.dropna(subset=["pc1", "pc2"]).iterrows():
            nn_values.append({"pc1": float(r["pc1"]), "pc2": float(r["pc2"]), "compound_id": str(r.get("compound_id", "")), "similarity": float(r.get("tanimoto_similarity", 0.0)), "label": str(r.get("label", ""))})
    query_value = [{"pc1": qxy[0], "pc2": qxy[1], "label": "Query molecule"}]
    spec = {
        "height": 380,
        "layer": [
            {"data": {"values": bg_values}, "mark": {"type": "point", "filled": True, "size": 20, "opacity": 0.18, "color": "#8aa0ad"}, "encoding": {"x": {"field": "pc1", "type": "quantitative", "axis": {"title": "Diagnostic PC1"}}, "y": {"field": "pc2", "type": "quantitative", "axis": {"title": "Diagnostic PC2"}}, "tooltip": [{"field": "runtime_smiles", "title": "SMILES"}]}},
            {"data": {"values": nn_values}, "mark": {"type": "point", "filled": True, "size": 95, "opacity": 0.92, "color": "#f0a202", "stroke": "#0b2638", "strokeWidth": 1}, "encoding": {"x": {"field": "pc1", "type": "quantitative"}, "y": {"field": "pc2", "type": "quantitative"}, "tooltip": [{"field": "similarity", "format": ".3f"}, "label"]}},
            {"data": {"values": query_value}, "mark": {"type": "point", "shape": "diamond", "filled": True, "size": 260, "color": "#0b2638", "stroke": "#16b8a6", "strokeWidth": 3}, "encoding": {"x": {"field": "pc1", "type": "quantitative"}, "y": {"field": "pc2", "type": "quantitative"}, "tooltip": ["label"]}},
        ],
        "config": {"view": {"stroke": "#dde7ef"}, "axis": {"gridColor": "#edf3f7", "labelColor": "#143044", "titleColor": "#143044"}},
    }
    st.vega_lite_chart(spec, use_container_width=True)
    ev = meta.get("explained_variance_ratio", [])
    if len(ev) >= 2:
        st.caption(f"Reader-facing PCA map from eight RDKit diagnostics; PC1+PC2 explain {100 * (ev[0] + ev[1]):.1f}% of diagnostic-descriptor variance.")
    else:
        st.caption("Reader-facing diagnostic projection. Exact applicability-domain status uses Morgan similarity and the 221 GNRetain model-feature ranges.")


def focused_descriptor_strip(X: pd.DataFrame, diag: pd.DataFrame) -> None:
    vals = focused_query_values(X, diag)
    if not vals:
        st.info("Focused ionization and solubility descriptors are unavailable for this query.")
        return

    metrics = [
        ("Estimated acidic pKa", "Acidic pKa", ".3f", "Estimated pKa of the most acidic ionizable site. Lower values generally indicate stronger acidity and greater deprotonation at pH 7.4."),
        ("Estimated basic pKa", "Basic pKa", ".3f", "Estimated pKa of the protonated conjugate acid of the strongest basic site. Higher values indicate stronger basicity and usually greater protonation at pH 7.4. This is not pKb."),
        ("Estimated cLogD7.4", "cLogD7.4", ".3f", "Estimated lipophilicity at pH 7.4, accounting approximately for ionization."),
        ("Estimated cLogS", "cLogS", ".3f", "Estimated base-10 logarithm of aqueous molar solubility. More negative values indicate lower predicted solubility."),
        ("Calculated cLogP", "cLogP", ".3f", "Calculated neutral-form octanol/water partition coefficient."),
        ("Formal charge", "Formal charge", ".0f", "Formal charge encoded in the standardized molecular structure; it is not a complete pH-dependent microstate calculation."),
    ]
    cols = st.columns(6)
    for col, (display_label, value_key, fmt, help_text) in zip(cols, metrics):
        value = vals.get(value_key, np.nan)
        with col:
            st.metric(display_label, format(value, fmt) if np.isfinite(value) else "NA", help=help_text)

    st.caption(
        f"Estimated ionization tendency at pH 7.4: **{estimated_ionization_class(vals)}**. "
        "Basic pKa is the pKa of the protonated conjugate acid (BH⁺), so a higher value means a stronger base; "
        "pKb runs in the opposite direction. Formal charge comes from the standardized input structure. "
        "All pKa, cLogD and cLogS values are computational estimates rather than experimental measurements or full microstate calculations."
    )


def ionization_solubility_space_map(X: pd.DataFrame, diag: pd.DataFrame, nn: pd.DataFrame, root_text: str) -> None:
    try:
        ref, _, coords, meta = load_focused_reference(root_text)
        qxy = transform_query_to_focused_pca(X, diag, meta)
    except Exception as e:
        st.info(f"Focused ionization and solubility map is unavailable: {type(e).__name__}: {e}")
        return
    if coords.empty or qxy is None:
        st.info("Focused ionization and solubility map is unavailable for this query.")
        return

    tooltip_cols = ["runtime_smiles", "Acidic pKa", "Basic pKa", "cLogD7.4", "cLogS", "cLogP", "Formal charge", "ionization_class"]
    bg = coords.merge(ref[["compound_id", *tooltip_cols]], on=["compound_id", "runtime_smiles", "ionization_class"], how="left")
    # Preserve rare ionization classes while limiting browser payload.
    sampled = []
    for _, group in bg.groupby("ionization_class", dropna=False):
        sampled.append(group if len(group) <= 1000 else group.sample(1000, random_state=42))
    bg = pd.concat(sampled, ignore_index=True) if sampled else bg
    bg = bg.rename(columns={
        "Acidic pKa": "acidic_pka_display",
        "Basic pKa": "basic_pka_display",
        "cLogD7.4": "clogd74_display",
        "cLogS": "clogs_display",
        "cLogP": "clogp_display",
        "Formal charge": "formal_charge_display",
    })
    bg_values = bg[[
        "pc1", "pc2", "compound_id", "runtime_smiles", "ionization_class",
        "acidic_pka_display", "basic_pka_display", "clogd74_display",
        "clogs_display", "clogp_display", "formal_charge_display",
    ]].fillna("").to_dict("records")

    nn_values = []
    if isinstance(nn, pd.DataFrame) and len(nn) and "runtime_smiles" in nn.columns:
        top = nn.head(8).copy()
        merged = top.merge(coords, on="runtime_smiles", how="left", suffixes=("", "_coord"))
        for _, row in merged.dropna(subset=["pc1", "pc2"]).iterrows():
            nn_values.append({
                "pc1": float(row["pc1"]),
                "pc2": float(row["pc2"]),
                "compound_id": str(row.get("compound_id", "")),
                "similarity": float(row.get("tanimoto_similarity", 0.0)),
                "label": str(row.get("label", "")),
                "ionization_class": str(row.get("ionization_class", "")),
            })

    qvals = focused_query_values(X, diag)
    query_value = [{
        "pc1": qxy[0],
        "pc2": qxy[1],
        "label": "Query molecule",
        "ionization_class": estimated_ionization_class(qvals),
        "acidic_pka_display": qvals.get("Acidic pKa"),
        "basic_pka_display": qvals.get("Basic pKa"),
        "clogd74_display": qvals.get("cLogD7.4"),
        "clogs_display": qvals.get("cLogS"),
        "clogp_display": qvals.get("cLogP"),
        "formal_charge_display": qvals.get("Formal charge"),
    }]
    class_domain = [
        "Predominantly neutral tendency",
        "Cationic tendency",
        "Anionic tendency",
        "Ampholytic tendency",
        "Formally cationic",
        "Formally anionic",
    ]
    class_range = ["#8aa0ad", "#2374ab", "#d97706", "#7c3aed", "#16b8a6", "#cf2e2e"]
    spec = {
        "height": 390,
        "layer": [
            {
                "data": {"values": bg_values},
                "mark": {"type": "point", "filled": True, "size": 25, "opacity": 0.24},
                "encoding": {
                    "x": {"field": "pc1", "type": "quantitative", "axis": {"title": "Ionization–solubility space, PC1"}},
                    "y": {"field": "pc2", "type": "quantitative", "axis": {"title": "Ionization–solubility space, PC2"}},
                    "color": {
                        "field": "ionization_class",
                        "type": "nominal",
                        "scale": {"domain": class_domain, "range": class_range},
                        "legend": {
                            "title": "Training compounds by estimated ionization tendency",
                            "orient": "bottom",
                            "direction": "horizontal",
                            "columns": 3,
                            "labelLimit": 260,
                            "titleLimit": 420,
                            "symbolSize": 90,
                            "padding": 8,
                        },
                    },
                    "tooltip": [
                        {"field": "compound_id", "title": "Compound"},
                        {"field": "ionization_class", "title": "Ionization tendency"},
                        {"field": "acidic_pka_display", "type": "quantitative", "format": ".3f", "title": "Acidic pKa"},
                        {"field": "basic_pka_display", "type": "quantitative", "format": ".3f", "title": "Basic pKa"},
                        {"field": "clogd74_display", "type": "quantitative", "format": ".3f", "title": "cLogD7.4"},
                        {"field": "clogs_display", "type": "quantitative", "format": ".3f", "title": "cLogS"},
                        {"field": "clogp_display", "type": "quantitative", "format": ".3f", "title": "cLogP"},
                        {"field": "formal_charge_display", "type": "quantitative", "format": ".0f", "title": "Formal charge"},
                    ],
                },
            },
            {
                "data": {"values": nn_values},
                "mark": {"type": "point", "filled": True, "size": 105, "opacity": 0.95, "color": "#f0a202", "stroke": "#0b2638", "strokeWidth": 1},
                "encoding": {
                    "x": {"field": "pc1", "type": "quantitative"},
                    "y": {"field": "pc2", "type": "quantitative"},
                    "tooltip": ["compound_id", {"field": "similarity", "format": ".3f"}, "label", "ionization_class"],
                },
            },
            {
                "data": {"values": query_value},
                "mark": {"type": "point", "shape": "diamond", "filled": True, "size": 280, "color": "#0b2638", "stroke": "#16b8a6", "strokeWidth": 3},
                "encoding": {
                    "x": {"field": "pc1", "type": "quantitative"},
                    "y": {"field": "pc2", "type": "quantitative"},
                    "tooltip": [
                        "label",
                        "ionization_class",
                        {"field": "acidic_pka_display", "type": "quantitative", "format": ".3f", "title": "Acidic pKa"},
                        {"field": "basic_pka_display", "type": "quantitative", "format": ".3f", "title": "Basic pKa"},
                        {"field": "clogd74_display", "type": "quantitative", "format": ".3f", "title": "cLogD7.4"},
                        {"field": "clogs_display", "type": "quantitative", "format": ".3f", "title": "cLogS"},
                        {"field": "clogp_display", "type": "quantitative", "format": ".3f", "title": "cLogP"},
                        {"field": "formal_charge_display", "type": "quantitative", "format": ".0f", "title": "Formal charge"},
                    ],
                },
            },
        ],
        "config": {"view": {"stroke": "#dde7ef"}, "axis": {"gridColor": "#edf3f7", "labelColor": "#143044", "titleColor": "#143044"}},
    }
    st.markdown(
        '<div style="display:flex;gap:1.4rem;align-items:center;flex-wrap:wrap;margin:0.1rem 0 0.55rem 0;">'
        '<span><span style="color:#0b2638;font-size:1.35rem;line-height:0;">◆</span> Query molecule</span>'
        '<span><span style="color:#f0a202;font-size:1.35rem;line-height:0;">●</span> Nearest training compounds</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.vega_lite_chart(spec, use_container_width=True)
    ev = meta.get("explained_variance_ratio", [])
    explained = f" PC1+PC2 explain {100 * (ev[0] + ev[1]):.1f}% of focused-descriptor variance." if len(ev) >= 2 else ""
    st.caption(
        "This focused projection uses acidic pKa, basic pKa, cLogD7.4, cLogS, cLogP, and formal charge."
        + explained
        + " It provides a descriptive view of the query relative to the training compounds."
    )


def motif_feature_strip(X: pd.DataFrame, root_text: str) -> None:
    if X is None or not isinstance(X, pd.DataFrame) or X.empty:
        return
    try:
        root = find_project_root(root_text)
        panel = pd.read_csv(root / "GNRetain221_Runtime_Model" / "schema" / "GNRetain221_feature_panel.csv")
        model_fragments = [
            str(feature) for feature in panel["feature"].astype(str)
            if str(feature).startswith("ru_rdkit_fr_") and str(feature) in X.columns
        ]
    except Exception:
        model_fragments = [str(c) for c in X.columns if str(c).startswith("ru_rdkit_fr_")]

    rows = []
    for feature in model_fragments:
        value = pd.to_numeric(pd.Series([X.iloc[0].get(feature, 0)]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) > 0:
            label = feature.removeprefix("ru_rdkit_fr_").replace("_", " ")
            rows.append({"fragment": label, "feature": feature, "count": float(value)})

    st.markdown('<div class="gn-visual-title">Model-detected structural fragments</div>', unsafe_allow_html=True)
    if not rows:
        st.markdown(
            '<div class="gn-card"><div class="gn-card-note">No positive RDKit fragment-count inputs were detected for this molecule. The model still uses the remaining physicochemical and structural descriptors.</div></div>',
            unsafe_allow_html=True,
        )
        return

    detected = pd.DataFrame(rows).sort_values(["count", "fragment"], ascending=[False, True])
    chips = [
        f'<span class="gn-pill" style="margin:0.18rem 0.25rem 0.18rem 0;background:#eef6fb;color:#143044;border-color:#cddce8;">{html.escape(str(row["fragment"]))} × {float(row["count"]):.0f}</span>'
        for _, row in detected.head(16).iterrows()
    ]
    st.markdown(
        '<div class="gn-card"><div class="gn-card-note">These are non-zero RDKit fragment-count descriptors that are direct inputs to the cleaned 221-feature GNRetain model. They are shown for structural interpretation and should not be read as causal determinants.</div>'
        '<div style="margin-top:0.55rem;line-height:2.05;">' + "".join(chips) + "</div></div>",
        unsafe_allow_html=True,
    )
    with st.expander("All detected model fragment descriptors", expanded=False):
        st.dataframe(detected[["fragment", "feature", "count"]], use_container_width=True, hide_index=True)


def navigate(page: str) -> None:
    st.session_state["app_page"] = page
    st.session_state["sidebar_page"] = page


def sidebar() -> str:
    st.sidebar.markdown(
        """
<div class="gn-logo">
  <p class="gn-logo-title">GNDiscovery</p>
  <p class="gn-logo-sub">E. coli decision-support platform<br>Separate GNRetain and GNMIC workflows</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.get("app_page") not in NAVIGATION:
        st.session_state["app_page"] = "Home"
    if st.session_state.get("sidebar_page") not in NAVIGATION:
        st.session_state["sidebar_page"] = st.session_state["app_page"]
    selected = st.sidebar.radio("Navigation", NAVIGATION, key="sidebar_page", label_visibility="collapsed")
    st.session_state["app_page"] = selected
    return selected


def header(page: str) -> None:
    subtitle_map = {
        "Home": "Choose a prediction workflow for E. coli compound prioritisation.",
        "GNRetain": "E. coli ΔtolC-associated retention-profile prediction.",
        "GNMIC": "E. coli ATCC 25922 strict MIC-activity classification.",
        "Integrated Assessment": "Compare both model outputs for the same standardized molecule.",
        "Batch Screening": "Screen compound libraries with one or both prediction modules.",
        "Model Information": "Endpoint definitions, training scope, performance and limitations.",
        "Reports": "Export model-specific or integrated prediction reports.",
        "About": "Scientific scope, interpretation policy and project information.",
    }
    st.markdown(
        f"""
<div class="gn-header">
  <div>
    <div class="gn-title">{APP_NAME}</div>
    <div class="gn-subtitle">{html.escape(subtitle_map.get(page, 'E. coli decision support'))}</div>
  </div>
  <div class="gn-scope">Current model scope: E. coli</div>
</div>
        """,
        unsafe_allow_html=True,
    )



def render_molecule(smiles: str, caption: str, width: int = 330, height: int = 245) -> None:
    '''Render a molecule robustly using PNG, inline SVG, then browser fallback.'''
    smiles_text = str(smiles).strip()
    if not smiles_text:
        st.info("No structure available.")
        return
    try:
        png = molecule_png_bytes(smiles_text, size=(int(width), int(height)))
        st.image(png, caption=caption, width=int(width))
        return
    except Exception:
        pass
    try:
        svg = molecule_svg_markup(smiles_text, size=(int(width), int(height)))
        safe_caption = html.escape(str(caption))
        components.html(
            f'''<div style="width:{int(width)}px;max-width:100%;border:1px solid #dbe6ee;border-radius:10px;padding:8px;background:white;">
            <div style="font-family:Segoe UI,Arial,sans-serif;font-size:13px;font-weight:700;margin-bottom:5px;color:#143044;">{safe_caption}</div>
            <div style="display:flex;justify-content:center;align-items:center;overflow:hidden;">{svg}</div></div>''',
            height=int(height)+62,
            scrolling=False,
        )
        return
    except Exception:
        pass
    canvas_id = "mol_" + hashlib.md5(smiles_text.encode("utf-8")).hexdigest()[:12]
    safe_smiles = json.dumps(smiles_text)
    safe_caption = html.escape(str(caption))
    components.html(
        f'''<div style="width:{int(width)}px;max-width:100%;border:1px solid #dbe6ee;border-radius:10px;padding:8px;background:white;">
        <div style="font-family:Segoe UI,Arial,sans-serif;font-size:13px;font-weight:700;margin-bottom:5px;color:#143044;">{safe_caption}</div>
        <canvas id="{canvas_id}" width="{int(width)}" height="{int(height)}" style="max-width:100%;"></canvas>
        <div id="{canvas_id}_msg" style="font-family:monospace;font-size:11px;color:#657586;word-break:break-all;"></div></div>
        <script src="https://unpkg.com/smiles-drawer@2.0.1/dist/smiles-drawer.min.js"></script>
        <script>(function(){{const s={safe_smiles};const id="{canvas_id}";const msg=document.getElementById(id+"_msg");
        function fail(t){{if(msg)msg.innerHTML="Structure renderer unavailable. SMILES: "+s+"<br>"+t;}}
        function draw(){{if(typeof SmilesDrawer==="undefined"){{fail("Browser renderer could not be loaded.");return;}}
        try{{let d=new SmilesDrawer.Drawer({{width:{int(width)},height:{int(height)},padding:10}});SmilesDrawer.parse(s,function(tree){{d.draw(tree,id,"light",false);}},function(e){{fail(e);}});}}catch(e){{fail(e);}}}}
        if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",draw);else setTimeout(draw,100);}})();</script>''',
        height=int(height)+68,
        scrolling=False,
    )



def probability_bar(label: str, probability: float, threshold: float, theme: str) -> None:
    p = float(np.clip(probability, 0, 1))
    t = float(np.clip(threshold, 0, 1))
    st.markdown(
        f"""
<div class="gn-prob-box">
  <div class="gn-prob-heading">{html.escape(label)}</div>
  <div class="gn-prob-value-row"><span>Model probability:</span><strong>{p:.3f}</strong></div>
  <div class="gn-bar">
    <div class="gn-bar-fill {theme}" style="width:{p*100:.2f}%"></div>
    <div class="gn-marker" style="left:{t*100:.2f}%" title="Classification threshold {t:.3f}"></div>
  </div>
  <div class="gn-bar-scale">
    <span class="gn-scale-start">0.00</span>
    <span class="gn-threshold-label" style="left:{t*100:.2f}%">Classification threshold<strong>{t:.3f}</strong></span>
    <span class="gn-scale-end">1.00</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def model_card(
    theme: str,
    model_name: str,
    context: str,
    value: str,
    probability_label: str,
    probability: float,
    threshold: float,
    support: str,
    similarity: float,
) -> None:
    st.markdown(
        f"""
<div class="gn-model-card {theme}">
  <div class="gn-model-name">{html.escape(model_name)}</div>
  <div class="gn-model-context">{html.escape(context)}</div>
  <div class="gn-model-value">{html.escape(value)}</div>
  <div class="gn-model-stat">{html.escape(probability_label)}: <strong>{probability:.3f}</strong><br>
  Decision threshold: <strong>{threshold:.3f}</strong></div>
  <div class="gn-support-badge {theme}">Support: {html.escape(str(support))}</div>
  <div class="gn-model-stat">Nearest training similarity: <strong>{similarity:.3f}</strong></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def support_dimension_card(title: str, value: str, detail: str, theme: str = "orange") -> None:
    st.markdown(
        f"""
<div class="gn-model-card {html.escape(theme)}" style="min-height:155px;">
  <div class="gn-model-name" style="font-size:.82rem;">{html.escape(title)}</div>
  <div class="gn-model-value" style="font-size:1.35rem;margin-top:.35rem;">{html.escape(value)}</div>
  <div class="gn-model-stat" style="margin-top:.55rem;">{html.escape(detail)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _retention_label(value: Any) -> str:
    return "Retention-positive" if str(value) in {"RP", "1", "1.0"} else "Retention-negative"


def _retention_support(ad_summary: dict[str, Any]) -> tuple[str, str]:
    category = str(ad_summary.get("ad_category", "outside domain")).lower()
    if "inside" in category:
        return "Supported", "The molecule lies within the supported GNRetain training domain."
    if "borderline" in category:
        return "Borderline", "The molecule has partial representation in GNRetain training chemistry."
    return "Low", "The molecule is poorly represented in GNRetain training chemistry."


def _integrated_interpretation(result: dict[str, Any]) -> tuple[str, str]:
    scores = result["scores"]
    ad = result["ad_summary"]
    mic = result["mic"]
    rp = str(scores.get("Gram_negative_retention_class", "")) == "RP"
    mic_active = str(mic.get("predicted_class", "")) == "MIC-active-like"
    retention_support, _ = _retention_support(ad)
    mic_support = str(mic.get("mic_training_support", "Low"))
    retention_limited = retention_support in {"Borderline", "Low"}
    mic_limited = mic_support == "Low"

    if rp and mic_active and retention_limited and not mic_limited:
        text = (
            "The MIC prediction is supported by similar compounds in the GNMIC training set. "
            "The retention prediction is positive, but the molecule is less well represented "
            "in the GNRetain training set."
        )
    elif rp and mic_active and not retention_limited and mic_limited:
        text = (
            "The retention prediction is supported by similar compounds in the GNRetain training set. "
            "The MIC prediction is active-like, but the molecule is less well represented in the "
            "GNMIC training set."
        )
    elif rp and mic_active and retention_limited and mic_limited:
        text = (
            "Both models give favourable predictions, but the molecule is not well represented in "
            "either training set."
        )
    elif rp and mic_active:
        text = "Both models give favourable predictions and both are supported by similar training compounds."
    elif rp and not mic_active:
        text = (
            "The retention prediction is positive, while the GNMIC result is MIC-inactive-like. "
            "The two models therefore give different outcomes for this molecule."
        )
    elif (not rp) and mic_active:
        text = (
            "The MIC prediction is active-like, while the GNRetain result is retention-negative. "
            "The two models therefore give different outcomes for this molecule."
        )
    else:
        text = "GNRetain predicts a retention-negative profile and GNMIC predicts an MIC-inactive-like profile."

    return "Interpretation", text


def render_retention_analogues(neighbors: pd.DataFrame, top_n: int = 6) -> None:
    if neighbors is None or neighbors.empty:
        st.info("No GNRetain training analogues were available.")
        return
    view = neighbors.head(top_n).reset_index(drop=True)
    for start in range(0, len(view), 3):
        cols = st.columns(3)
        for col, (_, row) in zip(cols, view.iloc[start:start+3].iterrows()):
            with col:
                label = "Retention-positive" if str(row.get("label", "")) == "RP" else "Retention-negative"
                smiles = str(row.get("runtime_smiles") or row.get("smiles") or "")
                st.markdown(f"**#{int(row.get('rank', start+1))} · {label}**")
                render_molecule(smiles, "Training analogue", width=235, height=180)
                st.markdown(
                    f'<div class="gn-analogue-meta">Similarity: <strong>{float(row.get("tanimoto_similarity", 0)):.3f}</strong><br>'
                    '</div>',
                    unsafe_allow_html=True,
                )


def render_mic_analogues(neighbors: pd.DataFrame, top_n: int = 6) -> None:
    if neighbors is None or neighbors.empty:
        st.info("No GNMIC training analogues were available.")
        return
    view = neighbors.head(top_n).reset_index(drop=True)
    for start in range(0, len(view), 3):
        cols = st.columns(3)
        for col, (_, row) in zip(cols, view.iloc[start:start+3].iterrows()):
            with col:
                active = float(row.get("binary_label", 0)) == 1.0
                label = "MIC-active" if active else "MIC-inactive"
                smiles = str(row.get("standardized_smiles_no_stereo", ""))
                st.markdown(f"**#{int(row.get('rank', start+1))} · {label}**")
                render_molecule(smiles, "Training analogue", width=235, height=180)
                mic_value = pd.to_numeric(pd.Series([row.get("consensus_mic_ug_ml")]), errors="coerce").iloc[0]
                mic_text = f"{mic_value:.3g} µg/mL" if pd.notna(mic_value) else "Not available"
                st.markdown(
                    f'<div class="gn-analogue-meta">Similarity: <strong>{float(row.get("tanimoto_similarity", 0)):.3f}</strong><br>'
                    f'Consensus MIC: <strong>{mic_text}</strong><br>'
                    f'Evidence: {html.escape(str(row.get("consensus_confidence", "")))}</div>',
                    unsafe_allow_html=True,
                )



def render_mic_analogue_groups(neighbors: pd.DataFrame, top_n_each: int = 3) -> None:
    if neighbors is None or neighbors.empty:
        st.info("No GNMIC training analogues were available.")
        return
    labels = pd.to_numeric(neighbors.get("binary_label"), errors="coerce")
    active = neighbors[labels.eq(1)].head(top_n_each)
    inactive = neighbors[labels.eq(0)].head(top_n_each)
    tab_a, tab_i, tab_all = st.tabs(["Closest active analogues", "Closest inactive analogues", "Closest overall"])
    with tab_a:
        if active.empty: st.info("No active analogue appeared in the retrieved neighbour set.")
        else: render_mic_analogues(active, top_n=len(active))
    with tab_i:
        if inactive.empty: st.info("No inactive analogue appeared in the retrieved neighbour set.")
        else: render_mic_analogues(inactive, top_n=len(inactive))
    with tab_all:
        render_mic_analogues(neighbors, top_n=min(9, len(neighbors)))


def prediction_input(root_text: str, runtime_ready: bool, workflow: str) -> tuple[str, dict[str, Any] | None]:
    if "single_smiles_input" not in st.session_state:
        st.session_state["single_smiles_input"] = st.session_state.get("last_smiles", AMOXICILLIN_SMILES)
    left, right = st.columns([1.15, .85])
    with left:
        smiles = st.text_area("SMILES input", height=96, key="single_smiles_input", placeholder="Paste one SMILES string")
        b1, b2, b3 = st.columns([1.15, .85, .75])
        run = b1.button(f"Run {workflow}", type="primary", use_container_width=True, disabled=not runtime_ready)
        if b2.button("Load example", use_container_width=True):
            st.session_state["single_smiles_input"] = AMOXICILLIN_SMILES
            st.rerun()
        if b3.button("Clear", use_container_width=True):
            st.session_state["single_smiles_input"] = ""
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_smiles", None)
            st.rerun()
        st.caption("The molecular parent is standardized once and used consistently by the selected model workflow.")
    with right:
        render_molecule(smiles.strip() or AMOXICILLIN_SMILES, "Input structure preview")
    if run:
        if not smiles.strip():
            st.warning("Enter a SMILES string before running the prediction.")
        else:
            try:
                with st.spinner("Standardizing the structure, generating descriptors and assessing model support..."):
                    result = cached_prediction(smiles, root_text)
                st.session_state["last_result"] = result
                st.session_state["last_smiles"] = smiles
                st.success("Prediction completed.")
            except Exception as exc:
                st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
    return smiles, st.session_state.get("last_result")


def home_page() -> None:
    st.subheader("Choose a prediction workflow")
    st.markdown('<div class="gn-section-note">Each module has its own endpoint, threshold, support system and interpretation.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """<div class="gn-workflow-card teal"><div class="gn-workflow-kicker">Retention layer</div>
<div class="gn-workflow-title">GNRetain</div><div class="gn-workflow-text">Predict an E. coli ΔtolC-associated retention-positive or retention-negative molecular profile, with domain diagnostics and retention-labelled analogues.</div></div>""",
            unsafe_allow_html=True,
        )
        st.button("Open GNRetain", use_container_width=True, on_click=navigate, args=("GNRetain",))
    with c2:
        st.markdown(
            """<div class="gn-workflow-card orange"><div class="gn-workflow-kicker">Activity layer</div>
<div class="gn-workflow-title">GNMIC</div><div class="gn-workflow-text">Predict strict MIC-active-class membership for E. coli ATCC 25922, with compact chemical-support and training-analogue interpretation.</div></div>""",
            unsafe_allow_html=True,
        )
        st.button("Open GNMIC", use_container_width=True, on_click=navigate, args=("GNMIC",))
    with c3:
        st.markdown(
            """<div class="gn-workflow-card purple"><div class="gn-workflow-kicker">Comparison layer</div>
<div class="gn-workflow-title">Integrated Assessment</div><div class="gn-workflow-text">Run both models for the same molecule and compare their predictions and support side by side.</div></div>""",
            unsafe_allow_html=True,
        )
        st.button("Compare both models", use_container_width=True, on_click=navigate, args=("Integrated Assessment",))
    st.markdown(
        """<div class="gn-callout"><div class="gn-callout-title">Scientific separation</div>
<div class="gn-callout-text">GNRetain and GNMIC are presented as separate workflows because they address different experimental questions.</div></div>""",
        unsafe_allow_html=True,
    )



def gnretain_page(root_text: str, ready: bool) -> None:
    st.subheader("GNRetain")
    st.caption("E. coli ΔtolC-associated retention-profile prediction")
    _, result = prediction_input(root_text, ready, "GNRetain")
    if not result:
        return
    scores = result["scores"]
    ad = result["ad_summary"]
    support, support_text = _retention_support(ad)
    p = float(scores.get("Gram_negative_retention_probability", np.nan))
    threshold = float(scores.get("Gram_negative_retention_threshold", np.nan))
    label = _retention_label(scores.get("Gram_negative_retention_class"))

    model_card("teal", "GNRetain", "E. coli ΔtolC-associated retention", label, "P(retention-positive)", p, threshold, support, float(ad.get("nearest_neighbor_similarity", np.nan)))
    probability_bar("Retention-positive probability", p, threshold, "teal")
    st.markdown(
        f'<div class="gn-callout"><div class="gn-callout-title">{html.escape(support)} prediction support</div>'
        f'<div class="gn-callout-text">{html.escape(support_text)}</div></div>', unsafe_allow_html=True,
    )

    tabs = st.tabs(["Summary", "Support / Applicability", "Retention Analogues", "Molecular Diagnostics", "Ionization Map"])
    with tabs[0]:
        c1, c2 = st.columns([.62, .38])
        with c1:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric(
                "Distance from threshold",
                f"{abs(p-threshold):.3f}",
                help="Absolute difference between the ensemble probability and the classification threshold. Larger values indicate a decision farther from the boundary, but not necessarily better chemical support.",
            )
            a2.metric(
                "Nearest training similarity",
                f"{float(ad.get('nearest_neighbor_similarity', np.nan)):.3f}",
                help="Highest Morgan-fingerprint Tanimoto similarity to a retention-labelled training compound. Values closer to 1 indicate more closely represented chemistry.",
            )
            a3.metric(
                "Prioritisation score",
                f"{float(scores.get('heuristic_prioritization_score', np.nan)):.3f}",
                help="Heuristic ranking score = P(retention-positive) × training-support weight × threshold-distance weight. Support weights are 1.0, 0.7 and 0.35 for supported, borderline and low support; predictions within 0.10 of the threshold receive a 0.65 distance weight. It is not a calibrated biological probability.",
            )
            seed_sd = float(result["pred_wide"].iloc[0].get("gnretention__member_probability_sd", np.nan))
            a4.metric(
                "Across-seed variation",
                f"{seed_sd:.3f}" if np.isfinite(seed_sd) else "NA",
                help="Standard deviation of the five seed-specific LightGBM probabilities. A lower value means the five fitted ensembles agree more closely. This captures stochastic model-fit variation, not total predictive uncertainty.",
            )
            st.markdown(f"**Training chemistry status:** {ad.get('training_chemistry_status', 'Unavailable')}")
            st.markdown(f"**Feature space:** {ad.get('ad_feature_space', 'GNRetain cleaned 221-feature model panel')}")
        with c2:
            render_molecule(result["feature_report"].get("runtime_smiles", ""), "Standardized molecular parent", width=315, height=225)

    with tabs[1]:
        st.markdown('<div class="gn-section-note">Prediction support combines nearest-neighbour structural similarity with all 221 cleaned model-feature ranges. The PCA map is descriptive only; it does not assign the support category.</div>', unsafe_allow_html=True)
        with st.expander("How GNRetain prediction support is assigned", expanded=False):
            st.markdown(
                """
**Supported:** nearest-neighbour similarity ≥0.45, no feature beyond the observed training minimum–maximum range, and no more than 10 features outside the 1st–99th percentile range.

**Borderline:** nearest-neighbour similarity ≥0.30, no more than 2 features beyond the observed range, and no more than 36 features outside the 1st–99th percentile range.

**Low support:** the molecule does not meet either condition. The 10- and 36-feature limits approximate the 95th and 99.5th percentiles of moderate-range counts in the 5,452-compound training matrix.
                """
            )
        a1, a2, a3 = st.columns(3)
        a1.metric("Beyond observed range", int(ad.get("n_descriptor_range_severe", 0)), help="Model features outside the minimum-to-maximum range observed in training.")
        a2.metric("Outside usual range", int(ad.get("n_descriptor_range_moderate", 0)), help="Model features outside the 1st–99th percentile range but still within the observed range.")
        a3.metric("Less typical features", int(ad.get("n_descriptor_range_borderline", 0)), help="Model features outside the central 5th–95th percentile range.")
        warning = str(ad.get("ad_warning", support_text))
        if support == "Low":
            st.warning(warning)
        else:
            st.info(warning)
        chemical_space_support_map(result["diagnostics"], result["retention_neighbors"], root_text)

    with tabs[2]:
        st.markdown('<div class="gn-section-note">Closest retention-labelled structures from the GNRetain training dataset.</div>', unsafe_allow_html=True)
        render_retention_analogues(result["retention_neighbors"], top_n=12)

    with tabs[3]:
        st.markdown('<div class="gn-section-note">Visual molecular diagnostics show whether key calculated properties are typical or unusual relative to the GNRetain training chemistry.</div>', unsafe_allow_html=True)
        percentile_rows = descriptor_percentile_interval_plot(result["diagnostics"], root_text)
        diagnostic_explanation_card(percentile_rows)
        st.markdown('<div class="gn-visual-title">Detected structural motifs</div>', unsafe_allow_html=True)
        motif_feature_strip(result["X"], root_text)
        with st.expander("Raw molecular diagnostics and standardized SMILES", expanded=False):
            diag = result["diagnostics"].copy()
            if len(diag):
                diag["value"] = pd.to_numeric(diag["value"], errors="coerce").round(3)
                st.dataframe(diag[["property", "value", "interpretation"]], use_container_width=True, hide_index=True)
            st.code(result["feature_report"].get("runtime_smiles", ""), language="text")

    with tabs[4]:
        st.markdown('<div class="gn-section-note">The query is positioned within the training-set ionization, lipophilicity, solubility and charge space. Structures for the nearest compounds are available in the Retention Analogues tab.</div>', unsafe_allow_html=True)
        focused_descriptor_strip(result["X"], result["diagnostics"])
        ionization_solubility_space_map(result["X"], result["diagnostics"], result["retention_neighbors"], root_text)




def gnmic_page(root_text: str, ready: bool) -> None:
    st.subheader("GNMIC")
    st.caption("E. coli ATCC 25922 strict MIC-activity classification with empirical Prediction Support")
    _, result = prediction_input(root_text, ready, "GNMIC")
    if not result:
        return
    mic = result["mic"]
    p = float(mic.get("probability_active", np.nan))
    threshold = float(mic.get("threshold", np.nan))
    support = str(mic.get("mic_training_support", "Unavailable"))
    model_card("orange", "GNMIC", "E. coli ATCC 25922 MIC activity", str(mic.get("predicted_class", "Unavailable")), "P(MIC-active)", p, threshold, support, float(mic.get("max_similarity_internal", np.nan)))
    probability_bar("MIC-active probability", p, threshold, "orange")

    tabs = st.tabs(["Summary", "Prediction Support", "Training Analogues"])
    with tabs[0]:
        left, right = st.columns([.58, .42])
        with left:
            st.markdown('<div class="gn-visual-title">Endpoint definition</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="gn-section-note"><strong>MIC-active-like</strong> corresponds to the curated MIC ≤16 µg/mL class. '
                '<strong>MIC-inactive-like</strong> corresponds to MIC >32 µg/mL.</div>',
                unsafe_allow_html=True,
            )
            st.caption("The output is a class probability, not a numerical MIC value.")
        with right:
            render_molecule(result["feature_report"].get("runtime_smiles", ""), "Standardized molecular parent", width=315, height=225)

    with tabs[1]:
        st.markdown(
            '<div class="gn-section-note">Support combines training-chemistry representation, agreement among nearby compounds, and validation accuracy for predictions with a similar support profile.</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            support_dimension_card(
                "Training chemistry",
                str(mic.get("training_chemistry_label", "Unavailable")),
                f"Nearest training similarity: {float(mic.get('max_similarity_internal', np.nan)):.3f}",
            )
        with c2:
            support_dimension_card(
                "Neighbourhood evidence",
                str(mic.get("neighbourhood_evidence_label", "Unavailable")),
                f"Weighted agreement: {100 * float(mic.get('weighted_neighbourhood_agreement', np.nan)):.0f}%",
            )
        with c3:
            reliability = float(mic.get("estimated_oof_reliability", np.nan))
            reliability_text = f"{100 * reliability:.0f}%" if pd.notna(reliability) else "Unavailable"
            support_dimension_card(
                "Accuracy for this support profile",
                reliability_text,
                "Held-out cases with similar support characteristics",
            )

        st.markdown(
            f'<div class="gn-callout orange"><div class="gn-callout-title">{html.escape(str(mic.get("mic_training_support", "Unavailable")))} prediction support</div>'
            f'<div class="gn-callout-text">{html.escape(str(mic.get("mic_support_interpretation", "")))}</div></div>',
            unsafe_allow_html=True,
        )

        with st.expander("How prediction support was assessed", expanded=False):
            st.markdown(
                "Prediction Support is separate from **P(MIC-active)**. It estimates how often predictions with a similar "
                "support profile were correct in scaffold-held-out validation. The primary GNMIC probability is not recalibrated "
                "or replaced by this score."
            )
            e1, e2, e3, e4 = st.columns(4)
            e1.metric(
                "Predicted-class confidence",
                f"{100 * float(mic.get('predicted_class_confidence', np.nan)):.0f}%",
                help="Calibrated confidence assigned to the class predicted at the fixed 0.510 operating threshold. This is one input to the correctness model, not the final Prediction Support score.",
            )
            e2.metric(
                "Distance from 0.510",
                f"{float(mic.get('distance_from_threshold', np.nan)):.3f}",
                help="Absolute distance between P(MIC-active) and the fixed decision threshold. The correctness model uses the directed margin, so values are positive in the direction of the predicted class.",
            )
            e3.metric(
                "Directed class similarity",
                f"{float(mic.get('support_directed_similarity_difference', np.nan)):+.3f}",
                help="Nearest active-versus-inactive similarity difference oriented toward the predicted class. Positive values mean class-specific analogue evidence favours the prediction.",
            )
            e4.metric(
                "Weighted agreement",
                f"{100 * float(mic.get('weighted_neighbourhood_agreement', np.nan)):.0f}%",
                help="Agreement of the ten nearest training compounds with the predicted class. Each neighbour is weighted by the fourth power of its ECFP4 Tanimoto similarity.",
            )

            d1, d2, d3 = st.columns(3)
            d1.metric("Nearest active similarity", f"{float(mic.get('max_similarity_positive_internal', np.nan)):.3f}")
            d2.metric("Nearest inactive similarity", f"{float(mic.get('max_similarity_negative_internal', np.nan)):.3f}")
            d3.metric(
                "Effective neighbours",
                f"{float(mic.get('effective_neighbours', np.nan)):.1f}",
                help="Effective number of influential neighbours after s^4 similarity weighting. It falls when the neighbourhood is dominated by only one or a few compounds.",
            )

            st.markdown("**Neighbour equations**")
            st.latex(r"w_j=s_j^4")
            st.latex(r"A=\frac{\sum_{j=1}^{10} w_j\,I(y_j=\hat y)}{\sum_{j=1}^{10} w_j}")
            st.latex(r"N_{\mathrm{eff}}=\frac{(\sum_j w_j)^2}{\sum_j w_j^2}")
            st.latex(r"D=(2\hat y-1)(s_A-s_I)")
            st.caption(
                "Here s is ECFP4 Tanimoto similarity, A is similarity-weighted agreement with the predicted class, "
                "N_eff is the effective number of contributing neighbours, and D is the directed active-versus-inactive similarity difference."
            )

            st.markdown("**Support categories in this rebuilt runtime**")
            st.markdown(
                "- **Indeterminate:** maximum similarity <0.2471 or effective neighbours <2.0.  \n"
                "- **High:** estimated correctness ≥0.8856 and weighted agreement ≥0.60.  \n"
                "- **Moderate:** estimated correctness ≥0.6866 and weighted agreement ≥0.50.  \n"
                "- **Low:** all remaining predictions with sufficient neighbour evidence."
            )
            st.caption(
                "Indeterminate is therefore an evidence-sufficiency category, not simply the lowest correctness probability. "
                "This new correctness model was refitted using original EC1792 out of fold probabilities with scaffold grouped cross-fitting; the categories are retrospective estimates, not guarantees."
            )

            scaffold_count = int(mic.get("same_scaffold_training_count", 0) or 0)
            st.markdown(f"**Exact Bemis–Murcko scaffold matches:** {scaffold_count}")
            st.caption(
                "An exact match requires the complete ring-and-linker framework to be identical. Closely related analogues can therefore have zero exact scaffold matches."
            )
            st.info(
                "Estimated accuracy for this support profile is an empirical correctness estimate from held-out predictions with similar model-confidence and neighbourhood characteristics. "
                "It is not the probability of antibacterial activity and is not a guarantee for the submitted molecule."
            )

    with tabs[2]:
        st.markdown('<div class="gn-section-note">Nearest ATCC 25922 training compounds are shown as molecular structures and separated into active and inactive neighbourhoods.</div>', unsafe_allow_html=True)
        render_mic_analogue_groups(result["mic_neighbors"], top_n_each=3)
        with st.expander("Technical analogue table", expanded=False):
            display = result["mic_neighbors"].copy()
            cols = [c for c in ["rank", "tanimoto_similarity", "predicted_endpoint_label", "consensus_mic_ug_ml", "consensus_confidence", "molecule_chembl_ids", "primary_document_id"] if c in display.columns]
            st.dataframe(display[cols], use_container_width=True, hide_index=True)




def integrated_page(root_text: str, ready: bool) -> None:
    st.subheader("Integrated Assessment")
    st.caption("Compare GNRetain and GNMIC for the same molecule. This is a secondary comparison workflow.")
    _, result = prediction_input(root_text, ready, "both models")
    if not result:
        return
    scores = result["scores"]
    ad = result["ad_summary"]
    mic = result["mic"]
    retention_support, _ = _retention_support(ad)
    title, text = _integrated_interpretation(result)

    c1, c2 = st.columns(2)
    with c1:
        model_card(
            "teal", "GNRetain", "E. coli ΔtolC-associated retention",
            _retention_label(scores.get("Gram_negative_retention_class")),
            "P(retention-positive)", float(scores.get("Gram_negative_retention_probability", np.nan)),
            float(scores.get("Gram_negative_retention_threshold", np.nan)), retention_support,
            float(ad.get("nearest_neighbor_similarity", np.nan)),
        )
    with c2:
        model_card(
            "orange", "GNMIC", "E. coli ATCC 25922 MIC activity",
            str(mic.get("predicted_class", "Unavailable")), "P(MIC-active)",
            float(mic.get("probability_active", np.nan)), float(mic.get("threshold", np.nan)),
            str(mic.get("mic_training_support", "Unavailable")), float(mic.get("max_similarity_internal", np.nan)),
        )
    st.markdown(
        f'<div class="gn-callout purple"><div class="gn-callout-title">{html.escape(title)}</div>'
        f'<div class="gn-callout-text">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )
    b1, b2 = st.columns(2)
    b1.button("Open detailed GNRetain analysis", use_container_width=True, on_click=navigate, args=("GNRetain",))
    b2.button("Open detailed GNMIC analysis", use_container_width=True, on_click=navigate, args=("GNMIC",))


def run_batch(
    df: pd.DataFrame,
    smiles_col: str,
    root_text: str,
    source_workers: int = 1,
    rdkit_threads: int = 1,
    parallel_backend: str = "threading",
) -> pd.DataFrame:
    """Run both frozen GNDiscovery endpoints for one dataframe slice.

    Batch metadata columns whose names begin with ``__batch_`` are used only to
    preserve file/row provenance across chunked multi-file screening and are not
    copied back as ordinary user columns.

    CPU settings accelerate chemistry feature/fingerprint generation only;
    they do not alter the fitted models, feature order, thresholds, or support
    definitions.
    """
    source_workers = max(1, int(source_workers or 1))
    rdkit_threads = max(1, int(rdkit_threads or 1))
    X, reports = batch_generate_runtime_source_features(
        df[smiles_col].astype(str).tolist(),
        project_root=root_text,
        max_workers=source_workers,
        parallel_backend=parallel_backend,
    )
    valid = reports[reports["status"].eq("ok")].copy() if len(reports) else pd.DataFrame()
    retention_pred = predict_from_features(X, project_root=root_text) if len(X) else pd.DataFrame()

    valid_smiles = valid["runtime_smiles"].astype(str).tolist() if len(valid) else []
    retention_ad = (
        batch_applicability_domain_summaries(
            X.reset_index(drop=True),
            valid_smiles,
            project_root=root_text,
            num_threads=rdkit_threads,
        )
        if len(X) and len(valid)
        else pd.DataFrame()
    )

    mic_pred = pd.DataFrame()
    if len(X) and len(valid):
        mic_pred, _ = predict_gnmic_from_source_features(
            valid_smiles,
            X.reset_index(drop=True),
            project_root=root_text,
            top_n_neighbours=1,
            num_threads=rdkit_threads,
        )

    rows: list[dict[str, Any]] = []
    pred_i = 0
    for _, report in reports.iterrows():
        local_row_index = int(report["row_index"])
        input_row = df.iloc[local_row_index]
        global_row_index = int(input_row.get("__batch_global_row", local_row_index))
        source_row = int(input_row.get("__batch_source_row", local_row_index))
        source_file = str(input_row.get("__batch_source_file", "")).strip()
        base: dict[str, Any] = {
            "row_index": global_row_index,
            "submitted_smiles": report.get("input_smiles", ""),
            "standardized_smiles": report.get("runtime_smiles", ""),
            "status": report.get("status", "failed"),
            "error": report.get("error", ""),
        }
        if source_file:
            base["source_file"] = source_file
            base["source_row"] = source_row
        for c in df.columns:
            if c != smiles_col and not str(c).startswith("__batch_"):
                base[c] = input_row[c]
        if report.get("status") != "ok":
            rows.append(base)
            continue

        rp = retention_pred.iloc[pred_i]
        ad = retention_ad.iloc[pred_i].to_dict() if len(retention_ad) > pred_i else {}
        mic = mic_pred.iloc[pred_i].to_dict() if len(mic_pred) > pred_i else {}
        rp_probability = rp.get("gnretention__proba_clean_ensemble", rp.get("gnretention__proba_retention_positive"))
        rp_class = rp.get("gnretention__class_label", "RN")
        base.update({
            "GNRetain_probability": round(float(rp_probability), 6) if rp_probability is not None else np.nan,
            "GNRetain_class": _retention_label(rp_class),
            "GNRetain_threshold": round(float(rp.get("gnretention__threshold")), 6),
            "GNRetain_support": _retention_support(ad)[0],
            "GNRetain_similarity": ad.get("nearest_neighbor_similarity"),
            "GNMIC_probability": mic.get("probability_active"),
            "GNMIC_class": mic.get("predicted_class"),
            "GNMIC_threshold": mic.get("threshold"),
            "GNMIC_support": mic.get("mic_training_support"),
            "GNMIC_similarity": mic.get("max_similarity_internal"),
            "GNMIC_active_similarity": mic.get("max_similarity_positive_internal"),
            "GNMIC_inactive_similarity": mic.get("max_similarity_negative_internal"),
            "GNMIC_training_chemistry": mic.get("training_chemistry_label"),
            "GNMIC_neighbourhood_evidence": mic.get("neighbourhood_evidence_label"),
            "GNMIC_weighted_agreement": mic.get("weighted_neighbourhood_agreement"),
            "GNMIC_estimated_OOF_reliability": mic.get("estimated_oof_reliability"),
            "GNMIC_effective_neighbours": mic.get("effective_neighbours"),
            "GNMIC_exact_murcko_scaffold_matches": mic.get("same_scaffold_training_count"),
        })
        rows.append(base)
        pred_i += 1
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)


def _detect_batch_smiles_column(df: pd.DataFrame) -> str | None:
    if df.empty and len(df.columns) == 0:
        return None
    candidates = [c for c in df.columns if "smile" in str(c).lower() or "structure" in str(c).lower()]
    return str(candidates[0]) if candidates else (str(df.columns[0]) if len(df.columns) else None)


def _batch_chunk_key(file_index: int, start: int, stop: int) -> str:
    return f"{file_index}:{start}:{stop}"


def _batch_chunk_sort_key(key: str) -> tuple[int, int, int]:
    try:
        file_index, start, stop = (int(x) for x in key.split(":", 2))
        return file_index, start, stop
    except Exception:
        return (10**9, 10**9, 10**9)


def _batch_concat_results(chunks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [chunks[k] for k in sorted(chunks, key=_batch_chunk_sort_key) if isinstance(chunks[k], pd.DataFrame)]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "row_index" in out.columns:
        out = out.sort_values("row_index", kind="stable").reset_index(drop=True)
    return out


def _batch_processed_by_file(chunks: dict[str, pd.DataFrame], file_count: int) -> list[int]:
    counts = [0] * file_count
    for key, frame in chunks.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        try:
            file_index = int(key.split(":", 1)[0])
        except Exception:
            continue
        if 0 <= file_index < file_count:
            counts[file_index] += len(frame)
    return counts


def _batch_counts(chunks: dict[str, pd.DataFrame]) -> tuple[int, int, int]:
    processed = successful = failed = 0
    for frame in chunks.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        processed += len(frame)
        if "status" in frame.columns:
            ok = int(frame["status"].eq("ok").sum())
            successful += ok
            failed += len(frame) - ok
    return processed, successful, failed


BATCH_CHECKPOINT_SCHEMA_VERSION = 3  # Invalidates pre-EC1792 checkpoint predictions


def _batch_job_signature(file_jobs: list[dict[str, Any]]) -> str:
    """Identity of the uploaded batch independent of chunk/compute settings."""
    payload = {
        "checkpoint_schema": BATCH_CHECKPOINT_SCHEMA_VERSION,
        "files": [
            {
                "index": int(job["index"]),
                "name": str(job["name"]),
                "sha256": str(job["sha256"]),
                "rows": int(len(job["df"])),
                "smiles_col": str(job["smiles_col"]),
            }
            for job in file_jobs
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _batch_checkpoint_dir(root_text: str, signature: str) -> Path:
    # Cloud instances may serve multiple authenticated Streamlit sessions.
    # Prevent sessions from reading each other's uploaded data and results.
    # Checkpoints live on the host's temporary disk and do not survive a restart.
    key = st.session_state.get("private_cloud_checkpoint_directory")
    if not key:
        key = tempfile.mkdtemp(prefix="gndiscovery_private_batch_")
        st.session_state["private_cloud_checkpoint_directory"] = key
    return Path(key) / signature


def _batch_checkpoint_chunk_path(directory: Path, file_index: int, start: int, stop: int) -> Path:
    return directory / f"chunk_{int(file_index):04d}_{int(start):09d}_{int(stop):09d}.csv.gz"


def _batch_save_chunk_to_disk(
    directory: Path,
    file_index: int,
    start: int,
    stop: int,
    frame: pd.DataFrame,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = _batch_checkpoint_chunk_path(directory, file_index, start, stop)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)
    return path


def _batch_load_chunks_from_disk(directory: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    chunks: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    if not directory.exists():
        return chunks, errors
    for path in sorted(directory.glob("chunk_*.csv.gz")):
        try:
            stem = path.name[:-7]  # remove .csv.gz
            parts = stem.split("_")
            if len(parts) != 4 or parts[0] != "chunk":
                continue
            file_index, start, stop = (int(parts[1]), int(parts[2]), int(parts[3]))
            frame = pd.read_csv(path)
            expected = max(stop - start, 0)
            if len(frame) != expected:
                errors.append(
                    f"Ignored {path.name}: expected {expected:,} rows but found {len(frame):,}."
                )
                continue
            chunks[_batch_chunk_key(file_index, start, stop)] = frame
        except Exception as exc:
            errors.append(f"Could not restore {path.name}: {type(exc).__name__}: {exc}")
    return chunks, errors


def _batch_write_checkpoint_metadata(
    directory: Path,
    signature: str,
    file_jobs: list[dict[str, Any]],
    chunk_size: int,
    source_workers: int,
    rdkit_threads: int,
    chunks: dict[str, pd.DataFrame],
    complete: bool,
    last_error: str | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    processed, successful, failed = _batch_counts(chunks)
    payload = {
        "checkpoint_schema": BATCH_CHECKPOINT_SCHEMA_VERSION,
        "signature": signature,
        "updated_epoch": time.time(),
        "complete": bool(complete),
        "processed_rows": int(processed),
        "successful_rows": int(successful),
        "failed_rows": int(failed),
        "chunk_size_last_used": int(chunk_size),
        "source_workers_last_used": int(source_workers),
        "rdkit_threads_last_used": int(rdkit_threads),
        "last_error": last_error or "",
        "files": [
            {
                "index": int(job["index"]),
                "name": str(job["name"]),
                "sha256": str(job["sha256"]),
                "rows": int(len(job["df"])),
                "smiles_col": str(job["smiles_col"]),
            }
            for job in file_jobs
        ],
    }
    path = directory / "checkpoint_metadata.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _batch_delete_checkpoint(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)


def _batch_latest_chunk_mtime(directory: Path) -> float:
    mtimes = [p.stat().st_mtime for p in directory.glob("chunk_*.csv.gz")] if directory.exists() else []
    return max(mtimes) if mtimes else 0.0


def _batch_ensure_combined_checkpoint_file(directory: Path, out: pd.DataFrame) -> Path:
    """Create/update a plain CSV only when saved chunks are newer than it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "completed_results.csv"
    newest_chunk = _batch_latest_chunk_mtime(directory)
    if not path.exists() or path.stat().st_mtime + 1e-6 < newest_chunk:
        tmp = path.with_name(path.name + ".tmp")
        out.to_csv(tmp, index=False)
        os.replace(tmp, path)
    return path


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "—"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def batch_page(root_text: str, ready: bool) -> None:
    st.subheader("Batch Screening")
    workflow = st.radio("Output workflow", ["GNRetain", "GNMIC", "Combined"], horizontal=True)
    template = pd.DataFrame({"compound_id": ["example_1", "example_2"], "smiles": [AMOXICILLIN_SMILES, "CCO"]})
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode(),
        "gndiscovery_batch_template.csv",
        "text/csv",
    )

    uploaded_files = st.file_uploader(
        "CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV libraries. Each file can use a different SMILES column.",
    )
    if not uploaded_files:
        return

    raw_jobs: list[dict[str, Any]] = []
    read_errors: list[str] = []
    for file_index, uploaded in enumerate(uploaded_files):
        raw = uploaded.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            read_errors.append(f"{uploaded.name}: {type(exc).__name__}: {exc}")
            continue
        raw_jobs.append({
            "index": file_index,
            "name": uploaded.name,
            "sha256": digest,
            "df": df,
            "default_smiles_col": _detect_batch_smiles_column(df),
        })

    if read_errors:
        for message in read_errors:
            st.error(f"Could not read CSV — {message}")
    if not raw_jobs:
        return

    file_jobs: list[dict[str, Any]] = []
    with st.expander("Review SMILES columns", expanded=len(raw_jobs) <= 3):
        for job in raw_jobs:
            df = job["df"]
            if len(df.columns) == 0:
                st.error(f"{job['name']} has no columns.")
                continue
            default_col = job["default_smiles_col"] or str(df.columns[0])
            column_names = [str(c) for c in df.columns]
            default_index = column_names.index(default_col) if default_col in column_names else 0
            selected = st.selectbox(
                f"{job['name']} — SMILES column",
                column_names,
                index=default_index,
                key=f"batch_smiles_col_{job['index']}_{job['sha256'][:12]}",
            )
            resolved = dict(job)
            resolved["smiles_col"] = selected
            file_jobs.append(resolved)

    if not file_jobs:
        return

    summary_rows: list[dict[str, Any]] = []
    total_rows = 0
    nonempty_smiles = 0
    global_offset = 0
    for job in file_jobs:
        df = job["df"]
        smiles_col = job["smiles_col"]
        not_blank = df[smiles_col].notna() & df[smiles_col].astype(str).str.strip().ne("")
        n_nonempty = int(not_blank.sum())
        total_rows += len(df)
        nonempty_smiles += n_nonempty
        job["global_offset"] = global_offset
        global_offset += len(df)
        summary_rows.append({
            "File": job["name"],
            "Rows": len(df),
            "Non-empty SMILES": n_nonempty,
            "SMILES column": smiles_col,
            "Status": "Ready",
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    with st.expander("Preview first uploaded file", expanded=False):
        first = file_jobs[0]
        st.caption(f"{first['name']} — first 20 rows")
        st.dataframe(first["df"].head(20), use_container_width=True, hide_index=True)

    settings_left, settings_right = st.columns(2)
    with settings_left:
        chunk_size = st.select_slider(
            "Processing chunk size",
            options=[50, 100, 250, 500, 1000],
            value=250,
            help=(
                "Smaller chunks save progress and refresh counters more often. "
                "You can change chunk size when resuming; completed rows will not be repeated."
            ),
        )
    logical_cpus = max(1, int(os.cpu_count() or 1))
    with settings_right:
        compute_mode = st.selectbox(
            "Compute mode",
            ["Automatic (recommended)", "Maximum CPU", "Balanced", "Compatibility (1 CPU)"],
            index=0,
            help="Controls CPU parallelism for RDKit feature/fingerprint generation. Model definitions are unchanged.",
        )
    estimated_physical = max(1, logical_cpus // 2) if logical_cpus >= 4 else logical_cpus
    if compute_mode == "Maximum CPU":
        source_workers = estimated_physical
        rdkit_threads = logical_cpus
        parallel_backend = "loky"
    elif compute_mode == "Balanced":
        source_workers = max(1, estimated_physical // 2)
        rdkit_threads = max(1, logical_cpus // 2)
        parallel_backend = "loky" if source_workers > 1 else "threading"
    elif compute_mode == "Compatibility (1 CPU)":
        source_workers = 1
        rdkit_threads = 1
        parallel_backend = "threading"
    else:
        source_workers = max(1, min(8, max(1, estimated_physical - 1)))
        rdkit_threads = max(1, min(logical_cpus, 12))
        parallel_backend = "loky" if source_workers > 1 else "threading"
    # Community Cloud has a small shared CPU and memory budget.
    # Avoid spawning many subprocesses regardless of the user's selected mode.
    source_workers = min(source_workers, 2)
    rdkit_threads = min(rdkit_threads, 2)
    parallel_backend = "threading"
    st.caption(
        f"Compute: {source_workers} descriptor worker(s) + {rdkit_threads} RDKit thread(s) "
        f"on {logical_cpus} logical CPU thread(s). Stages run sequentially to avoid oversubscription. "
        "GPU is not required for the frozen inference workflow."
    )

    # Chunk size and compute mode deliberately do not define batch identity.
    # This allows a stopped job to resume with different performance settings.
    signature = _batch_job_signature(file_jobs)
    checkpoint_dir = _batch_checkpoint_dir(root_text, signature)
    restored_errors: list[str] = []
    restored_from_disk = False
    if st.session_state.get("batch_job_signature") != signature:
        disk_chunks, restored_errors = _batch_load_chunks_from_disk(checkpoint_dir)
        st.session_state["batch_job_signature"] = signature
        st.session_state["batch_result_chunks"] = disk_chunks
        st.session_state["batch_job_complete"] = False
        st.session_state["batch_job_error"] = None
        st.session_state["batch_job_running"] = False
        st.session_state["batch_job_interrupted"] = False
        st.session_state.pop("batch_results", None)
        restored_from_disk = bool(disk_chunks)

    chunks = st.session_state.get("batch_result_chunks", {})
    if not isinstance(chunks, dict):
        chunks = {}
        st.session_state["batch_result_chunks"] = chunks

    # If Streamlit's Stop button interrupted the previous execution, this flag
    # remains True. On the next rerun, convert it into an explicit paused state.
    if bool(st.session_state.get("batch_job_running", False)):
        st.session_state["batch_job_running"] = False
        st.session_state["batch_job_interrupted"] = True

    processed, successful, failed = _batch_counts(chunks)
    remaining = max(total_rows - processed, 0)

    metric_row1 = st.columns(3)
    metric_row2 = st.columns(3)
    metric_slots = [c.empty() for c in [*metric_row1, *metric_row2]]
    progress_slot = st.empty()
    status_slot = st.empty()
    performance_slot = st.empty()
    per_file_slot = st.empty()

    def refresh_progress(current_text: str | None = None, rate: float | None = None) -> tuple[int, int, int]:
        current_processed, current_successful, current_failed = _batch_counts(chunks)
        current_remaining = max(total_rows - current_processed, 0)
        values = [
            ("Files uploaded", len(uploaded_files)),
            ("Total molecules", total_rows),
            ("Processed", current_processed),
            ("Remaining", current_remaining),
            ("Successful", current_successful),
            ("Failed / invalid", current_failed),
        ]
        for slot, (label, value) in zip(metric_slots, values):
            slot.metric(label, f"{int(value):,}")
        fraction = 1.0 if total_rows == 0 else min(max(current_processed / total_rows, 0.0), 1.0)
        progress_text = current_text or f"{current_processed:,} / {total_rows:,} molecules processed ({fraction:.1%})"
        progress_slot.progress(fraction, text=progress_text)
        if rate is not None and rate > 0:
            eta = current_remaining / rate if current_remaining else 0.0
            performance_slot.caption(f"Current run speed: {rate:,.1f} molecules/s · estimated time remaining: {_format_eta(eta)}")
        else:
            performance_slot.caption(f"Checkpoint directory: {checkpoint_dir}")

        completed_by_file = _batch_processed_by_file(chunks, len(uploaded_files))
        progress_rows: list[dict[str, Any]] = []
        for job in file_jobs:
            done = completed_by_file[job["index"]] if job["index"] < len(completed_by_file) else 0
            rows = len(job["df"])
            file_remaining = max(rows - done, 0)
            if rows == 0 or done >= rows:
                state = "Complete"
            elif done > 0:
                state = "Partial"
            else:
                state = "Queued"
            progress_rows.append({
                "File": job["name"],
                "Rows": rows,
                "Processed": done,
                "Remaining": file_remaining,
                "Status": state,
            })
        per_file_slot.dataframe(pd.DataFrame(progress_rows), use_container_width=True, hide_index=True)
        return current_processed, current_successful, current_failed

    refresh_progress()
    st.caption(f"Rows with non-empty SMILES before structure validation: {nonempty_smiles:,} of {total_rows:,}.")

    for message in restored_errors:
        st.warning(message)
    if restored_from_disk:
        status_slot.success(f"Restored {processed:,} completed molecules from the disk checkpoint.")
    elif bool(st.session_state.get("batch_job_interrupted", False)) and processed > 0:
        status_slot.warning(
            f"Previous processing was stopped. {processed:,} completed molecules are safe. "
            "Download them below or press Resume batch screening."
        )
    elif st.session_state.get("batch_job_error"):
        status_slot.warning(str(st.session_state.get("batch_job_error")))
    elif processed > 0 and remaining > 0:
        status_slot.info(f"Checkpoint ready: {processed:,} processed and {remaining:,} remaining.")
    elif processed >= total_rows and total_rows > 0:
        status_slot.success(f"Batch checkpoint is complete: {processed:,} molecules.")

    controls = st.columns([1, 1, 2])
    action_label = "Resume batch screening" if processed > 0 and remaining > 0 else ("Run batch again" if remaining == 0 and total_rows > 0 else "Run batch screening")
    run_clicked = controls[0].button(
        action_label,
        type="primary",
        disabled=(not ready or total_rows == 0),
        use_container_width=True,
    )
    refresh_clicked = controls[1].button("Refresh checkpoint view", use_container_width=True)
    controls[2].caption(
        "Safe pause: use Streamlit Stop. Every completed chunk is written to disk. "
        "After stopping, click Refresh checkpoint view to expose the latest download/resume state."
    )
    if refresh_clicked:
        st.rerun()

    with st.expander("Checkpoint management", expanded=False):
        st.caption(
            "Private cloud checkpoints are temporary and limited to this browser session. Download completed results before disconnecting. A cloud restart will erase checkpoints."
        )
        confirm_reset = st.checkbox(
            "I understand that resetting deletes all saved progress for this batch.",
            key=f"batch_reset_confirm_{signature[:12]}",
        )
        reset_clicked = st.button(
            "Delete checkpoint and reset progress",
            disabled=(processed == 0 or not confirm_reset),
            type="secondary",
        )
        if reset_clicked:
            _batch_delete_checkpoint(checkpoint_dir)
            st.session_state["batch_result_chunks"] = {}
            st.session_state["batch_job_complete"] = False
            st.session_state["batch_job_error"] = None
            st.session_state["batch_job_running"] = False
            st.session_state["batch_job_interrupted"] = False
            st.session_state.pop("batch_results", None)
            st.rerun()

    if run_clicked:
        # A deliberate rerun of an already complete job starts a new clean checkpoint.
        if total_rows > 0 and processed >= total_rows:
            _batch_delete_checkpoint(checkpoint_dir)
            chunks.clear()
            st.session_state["batch_result_chunks"] = chunks
            st.session_state["batch_job_complete"] = False
            st.session_state["batch_job_error"] = None
            st.session_state.pop("batch_results", None)
            processed = successful = failed = 0
            remaining = total_rows
            refresh_progress("Starting a new batch run...")

        st.session_state["batch_job_running"] = True
        st.session_state["batch_job_interrupted"] = False
        st.session_state["batch_job_error"] = None
        run_start = time.perf_counter()
        run_start_processed = _batch_counts(chunks)[0]
        batch_error: str | None = None
        stop_processing = False
        completed_chunk_count = len(chunks)

        completed_by_file = _batch_processed_by_file(chunks, len(uploaded_files))
        remaining_chunk_count = 0
        for job in file_jobs:
            done = completed_by_file[job["index"]] if job["index"] < len(completed_by_file) else 0
            left = max(len(job["df"]) - done, 0)
            if left:
                remaining_chunk_count += (left + int(chunk_size) - 1) // int(chunk_size)
        total_chunk_target = completed_chunk_count + remaining_chunk_count

        for file_position, job in enumerate(file_jobs, start=1):
            df = job["df"]
            done = completed_by_file[job["index"]] if job["index"] < len(completed_by_file) else 0
            start_from = min(max(int(done), 0), len(df))
            for start in range(start_from, len(df), int(chunk_size)):
                stop = min(start + int(chunk_size), len(df))
                key = _batch_chunk_key(job["index"], start, stop)
                status_text = (
                    f"Processing file {file_position}/{len(file_jobs)}: {job['name']} — "
                    f"rows {start + 1:,}–{stop:,} (overall chunk {completed_chunk_count + 1:,}/{total_chunk_target:,})"
                )
                status_slot.info(status_text)

                current_processed = _batch_counts(chunks)[0]
                elapsed = max(time.perf_counter() - run_start, 1e-9)
                rate = (current_processed - run_start_processed) / elapsed if current_processed > run_start_processed else None
                refresh_progress(status_text, rate=rate)

                chunk_df = df.iloc[start:stop].copy().reset_index(drop=True)
                chunk_df["__batch_source_file"] = job["name"]
                chunk_df["__batch_source_row"] = np.arange(start + 1, stop + 1, dtype=int)
                chunk_df["__batch_global_row"] = np.arange(
                    int(job["global_offset"]) + start,
                    int(job["global_offset"]) + stop,
                    dtype=int,
                )
                try:
                    chunk_out = run_batch(
                        chunk_df,
                        job["smiles_col"],
                        root_text,
                        source_workers=source_workers,
                        rdkit_threads=rdkit_threads,
                        parallel_backend=parallel_backend,
                    )
                    if len(chunk_out) != (stop - start):
                        raise RuntimeError(
                            f"Chunk row-count mismatch: input {stop - start:,}, output {len(chunk_out):,}."
                        )
                    # Persist atomically before advancing the in-memory checkpoint.
                    # If the disk write fails, this chunk is deliberately not marked
                    # complete and will be retried on resume.
                    _batch_save_chunk_to_disk(
                        checkpoint_dir,
                        job["index"],
                        start,
                        stop,
                        chunk_out,
                    )
                    chunks[key] = chunk_out
                    st.session_state["batch_result_chunks"] = chunks
                    completed_chunk_count += 1
                    processed_now = _batch_counts(chunks)[0]
                    _batch_write_checkpoint_metadata(
                        checkpoint_dir,
                        signature,
                        file_jobs,
                        int(chunk_size),
                        source_workers,
                        rdkit_threads,
                        chunks,
                        complete=processed_now >= total_rows,
                    )
                except Exception as exc:
                    batch_error = (
                        f"Stopped at {job['name']} rows {start + 1:,}–{stop:,}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    st.session_state["batch_job_error"] = batch_error
                    stop_processing = True
                    try:
                        _batch_write_checkpoint_metadata(
                            checkpoint_dir,
                            signature,
                            file_jobs,
                            int(chunk_size),
                            source_workers,
                            rdkit_threads,
                            chunks,
                            complete=False,
                            last_error=batch_error,
                        )
                    except Exception:
                        pass
                    break

                elapsed = max(time.perf_counter() - run_start, 1e-9)
                processed_now = _batch_counts(chunks)[0]
                rate = (processed_now - run_start_processed) / elapsed if processed_now > run_start_processed else None
                refresh_progress(status_text, rate=rate)

            if stop_processing:
                break

        st.session_state["batch_job_running"] = False
        processed, successful, failed = refresh_progress()
        out_now = _batch_concat_results(chunks)
        if not out_now.empty:
            st.session_state["batch_results"] = out_now

        if batch_error:
            st.session_state["batch_job_complete"] = False
            status_slot.error(
                batch_error + " Completed chunks were saved to disk; correct the cause and press Resume batch screening."
            )
        elif processed >= total_rows:
            st.session_state["batch_job_complete"] = True
            st.session_state["batch_job_error"] = None
            status_slot.success(f"Batch screening completed: {processed:,} molecules across {len(file_jobs):,} file(s).")
            try:
                _batch_write_checkpoint_metadata(
                    checkpoint_dir,
                    signature,
                    file_jobs,
                    int(chunk_size),
                    source_workers,
                    rdkit_threads,
                    chunks,
                    complete=True,
                )
            except Exception:
                pass
        else:
            st.session_state["batch_job_complete"] = False
            status_slot.warning(f"Batch paused with {max(total_rows - processed, 0):,} molecules remaining.")

    out = _batch_concat_results(chunks)
    if out.empty:
        stored_out = st.session_state.get("batch_results")
        if isinstance(stored_out, pd.DataFrame):
            out = stored_out
    if not isinstance(out, pd.DataFrame) or out.empty:
        return

    is_complete = len(out) >= total_rows and total_rows > 0
    st.markdown("### Completed results")
    if not is_complete:
        st.info(
            f"Checkpoint contains {len(out):,} processed molecules; {max(total_rows - len(out), 0):,} remain. "
            "Download these results now. Resume is available while this browser session remains active; a cloud restart will erase checkpoints."
        )

    valid = out["status"].eq("ok") if "status" in out.columns else pd.Series(False, index=out.index)
    a1, a2, a3 = st.columns(3)
    a1.metric("Valid compounds", f"{int(valid.sum()):,}")
    a2.metric("Invalid structures", f"{int((~valid).sum()):,}")
    if workflow == "GNRetain":
        positive_count = int((out.loc[valid, "GNRetain_class"] == "Retention-positive").sum()) if "GNRetain_class" in out.columns else 0
        a3.metric("Retention-positive", f"{positive_count:,}")
        cols = [c for c in out.columns if not c.startswith("GNMIC_")]
    elif workflow == "GNMIC":
        active_count = int((out.loc[valid, "GNMIC_class"] == "MIC-active-like").sum()) if "GNMIC_class" in out.columns else 0
        a3.metric("MIC-active-like", f"{active_count:,}")
        cols = [c for c in out.columns if not c.startswith("GNRetain_")]
    else:
        if "GNRetain_class" in out.columns and "GNMIC_class" in out.columns:
            concordant = (
                out.loc[valid, "GNRetain_class"].eq("Retention-positive")
                & out.loc[valid, "GNMIC_class"].eq("MIC-active-like")
            ).sum()
        else:
            concordant = 0
        a3.metric("Concordant favourable", f"{int(concordant):,}")
        cols = list(out.columns)

    preview_cols = st.columns([1, 1, 2])
    preview_mode = preview_cols[0].selectbox("Preview", ["Latest", "First"], index=0)
    preview_n = preview_cols[1].selectbox("Rows to show", [100, 250, 500, 1000], index=1)
    preview_cols[2].caption(
        f"The table preview is limited for responsiveness. The download contains all {len(out):,} completed rows."
    )
    preview = out[cols].tail(int(preview_n)) if preview_mode == "Latest" else out[cols].head(int(preview_n))
    st.dataframe(preview, use_container_width=True, hide_index=True)

    try:
        combined_path = _batch_ensure_combined_checkpoint_file(checkpoint_dir, out)
        download_name = "gndiscovery_batch_results.csv" if is_complete else "gndiscovery_batch_checkpoint_results.csv"
        download_label = "Download full batch results" if is_complete else "Download completed checkpoint results"
        st.download_button(
            download_label,
            data=combined_path.read_bytes(),
            file_name=download_name,
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Could not prepare checkpoint download: {type(exc).__name__}: {exc}")

    with st.expander("Checkpoint details", expanded=False):
        st.code(str(checkpoint_dir), language=None)
        st.caption(
            "The checkpoint is stored as atomic compressed chunks plus metadata. "
            "It is safe to restart Streamlit and restore by re-uploading the same files."
        )


def model_information_page(retention_status: dict[str, Any], mic_status: dict[str, Any]) -> None:
    st.subheader("Model Information")
    tab1, tab2 = st.tabs(["GNRetain", "GNMIC"])
    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Training compounds", f"{int(retention_status.get('training_n', 5452)):,}")
        c2.metric("Features", f"{int(retention_status.get('n_features', 221)):,}")
        c3.metric("Threshold", f"{float(retention_status.get('threshold', .575)):.3f}")
        c4.metric("Algorithm", "LightGBM ensemble", help="Five LightGBM classifiers use identical training data and hyperparameters but different random seeds. Their positive-class probabilities are averaged.")
        c5.metric("Ensemble members", f"{int(retention_status.get('ensemble_members', 5))} seeds")
        st.markdown(
            """
**Endpoint:** retention-positive versus retention-negative profile derived from an *E. coli* TolC-deficient experimental context.

**GNRetain deployment:** The five-seed LightGBM ensemble uses 221 ordered molecular features and was fitted on 5,452 compounds. The fixed retention-positive classification threshold is **0.575**. The prediction is the mean positive-class probability from seeds 42, 52, 62, 72 and 82.

**Prediction context:** Training analogues, descriptor distributions and applicability-domain indicators are calculated using the same 5,452-compound reference. They describe the model's chemical support, not experimentally established retention for a new molecule.
            """
        )
    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training compounds", f"{int(mic_status.get('training_n', 10148)):,}")
        c2.metric("MIC features", "1,792")
        c3.metric("Operating threshold", f"{float(mic_status.get('threshold', .510)):.3f}")
        c4.metric("Classifier", "3-seed LightGBM")
        st.markdown(
            """
**Endpoint:** active MIC ≤16 µg/mL; inactive MIC >32 µg/mL. The >16–32 µg/mL zone is excluded from the binary endpoint.

**Prediction Support:** top-10 ECFP4/Tanimoto neighbours, fourth-power similarity weighting, weighted predicted-class agreement, effective-neighbour information, directed class-similarity evidence, model confidence, threshold distance and predicted class are combined in a scaffold-cross-fitted correctness model.

**Prediction Support performance:** The scaffold-grouped evaluation reported correctness ROC-AUC 0.812 and Brier score 0.113. These values concern the Prediction Support estimator, not MIC prediction performance.
            """
        )
        st.info(
            "GNMIC-EC1792 uses 1,792 ordered molecular features and a fixed decision threshold of 0.510. "
            "Prediction Support is estimated separately from the model probability; its reported evaluation is scaffold-grouped."
        )


def report_page() -> None:
    st.subheader("Reports")
    result = st.session_state.get("last_result")
    if not result:
        st.info("Run a single-molecule prediction first.")
        return
    report_type = st.radio("Report type", ["GNRetain", "GNMIC", "Integrated Assessment"], horizontal=True)
    smiles = result["feature_report"].get("input_smiles", "")
    runtime = result["feature_report"].get("runtime_smiles", "")
    scores = result["scores"]
    ad = result["ad_summary"]
    mic = result["mic"]
    if report_type == "GNRetain":
        md = f"""# GNRetain prediction report

- Submitted SMILES: `{smiles}`
- Standardized SMILES: `{runtime}`
- Predicted class: {_retention_label(scores.get('Gram_negative_retention_class'))}
- P(retention-positive): {float(scores.get('Gram_negative_retention_probability', np.nan)):.4f}
- Decision threshold: {float(scores.get('Gram_negative_retention_threshold', np.nan)):.4f}
- Prediction support: {_retention_support(ad)[0]}
- Nearest training similarity: {float(ad.get('nearest_neighbor_similarity', np.nan)):.4f}
- Model: five-seed LightGBM probability-averaged ensemble
- Model features: 221 cleaned descriptors
- Across-seed probability SD: {float(result["pred_wide"].iloc[0].get("gnretention__member_probability_sd", np.nan)):.4f}

"""
    elif report_type == "GNMIC":
        md = f"""# GNMIC prediction report

- Submitted SMILES: `{smiles}`
- Standardized SMILES: `{runtime}`
- Predicted class: {mic.get('predicted_class')}
- P(MIC-active): {float(mic.get('probability_active', np.nan)):.4f}
- Decision threshold: {float(mic.get('threshold', np.nan)):.4f}
- Prediction support: {mic.get('mic_training_support')}
- Predicted-class confidence: {float(mic.get('predicted_class_confidence', np.nan)):.4f}
- Distance from threshold: {float(mic.get('distance_from_threshold', np.nan)):.4f}
- Directed class-similarity difference: {float(mic.get('support_directed_similarity_difference', np.nan)):+.4f}
- Nearest training similarity: {float(mic.get('max_similarity_internal', np.nan)):.4f}
- Training chemistry: {mic.get('training_chemistry_label')}
- Neighbourhood evidence: {mic.get('neighbourhood_evidence_label')}
- Similarity-weighted agreement: {float(mic.get('weighted_neighbourhood_agreement', np.nan)):.4f}
- Estimated accuracy for this support profile: {float(mic.get('estimated_oof_reliability', np.nan)):.4f}
- Effective neighbours: {float(mic.get('effective_neighbours', np.nan)):.2f}
- Exact Bemis–Murcko scaffold matches: {int(mic.get('same_scaffold_training_count', 0) or 0)}
- Endpoint: active MIC ≤16 µg/mL; inactive MIC >32 µg/mL

The output is a class probability, not a numerical MIC estimate.
Estimated profile accuracy is derived from scaffold-held-out validation cases with similar support characteristics and is not a guarantee for the submitted molecule.
"""
    else:
        title, text = _integrated_interpretation(result)
        md = f"""# GNDiscovery integrated assessment

- Submitted SMILES: `{smiles}`
- Standardized SMILES: `{runtime}`

## GNRetain
- Class: {_retention_label(scores.get('Gram_negative_retention_class'))}
- Probability: {float(scores.get('Gram_negative_retention_probability', np.nan)):.4f}
- Support: {_retention_support(ad)[0]}

## GNMIC
- Class: {mic.get('predicted_class')}
- Probability: {float(mic.get('probability_active', np.nan)):.4f}
- Support: {mic.get('mic_training_support')}

## Interpretation
{text}

"""
    metadata = {
        "report_type": report_type,
        "submitted_smiles": smiles,
        "standardized_smiles": runtime,
        "GNRetain": {**scores, **ad},
        "GNMIC": mic,
        "citation": APP_CITATION,
    }
    c1, c2 = st.columns(2)
    c1.download_button("Download Markdown", md.encode(), f"{report_type.lower().replace(' ', '_')}_report.md", "text/markdown", use_container_width=True)
    c2.download_button("Download JSON", json.dumps(metadata, indent=2, default=str).encode(), f"{report_type.lower().replace(' ', '_')}_metadata.json", "application/json", use_container_width=True)
    st.code(md, language="markdown")


def about_page() -> None:
    st.subheader("About GNDiscovery")
    st.markdown(
        """
GNDiscovery is an E. coli decision-support platform organized as separate scientific subapps:

- **GNRetain:** E. coli ΔtolC-associated retention-profile prediction using a cleaned 221-feature, five-seed LightGBM ensemble.
- **GNMIC:** E. coli ATCC 25922 strict MIC-activity classification with a three-seed LightGBM classifier and empirical Prediction Support.
- **Integrated Assessment:** optional side-by-side comparison of the two model outputs and their support.

GNDiscovery v5.7 is intended for early compound prioritisation, library triage, analogue review and pre-synthesis assessment. Predictions require experimental confirmation.
        """
    )
    st.markdown("#### Developer")
    st.markdown(
        """
**Jaafar Wadi**  
PhD Candidate, Medicinal Chemistry  
AG Keusgen, Institute of Pharmaceutical Chemistry  
Philipps-Universitaet Marburg, Germany  
Supported by the German Academic Exchange Service (DAAD).
        """
    )
    st.markdown("#### Citation")
    st.write(APP_CITATION)


def main() -> None:
    inject_css()
    root_text = safe_root_default()
    retention_ready, retention_status, mic_ready, mic_status = cached_health(root_text)
    page = sidebar()
    header(page)

    if page == "Home":
        home_page()
    elif page == "GNRetain":
        if not retention_ready:
            st.error(f"GNRetain resources are unavailable: {retention_status.get('error', 'runtime validation failed')}")
        gnretain_page(root_text, retention_ready)
    elif page == "GNMIC":
        if not mic_ready:
            st.error(f"GNMIC resources are unavailable: {mic_status.get('error', 'runtime validation failed')}")
        gnmic_page(root_text, mic_ready)
    elif page == "Integrated Assessment":
        if not (retention_ready and mic_ready):
            st.error("Both GNRetain and GNMIC resources are required for integrated assessment.")
        integrated_page(root_text, retention_ready and mic_ready)
    elif page == "Batch Screening":
        batch_page(root_text, retention_ready and mic_ready)
    elif page == "Model Information":
        model_information_page(retention_status, mic_status)
    elif page == "Reports":
        report_page()
    elif page == "About":
        about_page()

    st.markdown(
        '<div class="gn-footer">GNDiscovery keeps retention and MIC predictions scientifically separate and uses model-specific support diagnostics.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
