from __future__ import annotations

import py_compile
from pathlib import Path

from gnretain_source_feature_generator import find_project_root


def main() -> int:
    root = find_project_root()
    app_file = root / "streamlit_app" / "gndiscovery_app.py"
    py_compile.compile(str(app_file), doraise=True)
    source = app_file.read_text(encoding="utf-8")
    required = [
        "GNDiscovery: E. coli",
        "Choose a prediction workflow",
        "E. coli ΔtolC-associated retention",
        "E. coli ATCC 25922 MIC activity",
        "Integrated Assessment",
        "Batch Screening",
        "Model Information",
        "Reports",
        "Prediction Support",
        "Training Analogues",
        "render_retention_analogues",
        "render_mic_analogues",
        "molecule_svg_markup",
        "chemical_space_support_map",
        "descriptor_percentile_interval_plot",
        "ionization_solubility_space_map",
        "ionization_solubility_space_map",
        "Nearest training similarity",
        "Neighbourhood evidence",
        "Accuracy for this support profile",
        "How prediction support was assessed",
        "The MIC prediction is supported by similar compounds in the GNMIC training set.",
    ]
    missing = [text for text in required if text not in source]
    if missing:
        raise AssertionError(f"UI text missing from app file: {missing}")
    forbidden = [
        "Phase09_Motif389_Runtime_Model",
        "Endpoint table and model-member diagnostic probabilities",
        "ChEMBL active chemical space",
        "Simple mode",
        "Advanced mode",
        "Overall similarity",
        "Suggested next step",
        "Consider ATCC 25922 MIC testing first",
    ]
    present = [text for text in forbidden if text in source]
    if present:
        raise AssertionError(f"Legacy UI terminology remains: {present}")
    removed = [
        "GNRetain predicts a retention-associated class. It does not predict MIC or guaranteed wild-type antibacterial activity.",
        'st.warning(ad.get("ad_warning", support_text)) if support == "Low" else st.info',
    ]
    remaining = [text for text in removed if text in source]
    if remaining:
        raise AssertionError(f"Removed/broken UI code remains: {remaining}")
    print("UI compile/text smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
