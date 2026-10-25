# GNDiscovery

**GNDiscovery** is a cheminformatics application for evaluating two distinct molecular endpoints in *Escherichia coli*: compound retention and antibacterial activity. It integrates two independently trained machine-learning models within a Streamlit interface and provides complementary information about chemical similarity and prediction support.

## Models

| Component | Prediction | Training set | Model | Operating threshold |
| --- | --- | ---: | --- | ---: |
| **GNRetain** | Retention-positive versus retention-negative classification | 5,452 compounds | Five-model LightGBM ensemble; 221 molecular features | 0.575 |
| **GNMIC-EC1792** | Binary MIC activity against *E. coli* ATCC 25922 | 10,148 compounds | Three-model LightGBM ensemble; 1,792 molecular features | 0.510 |

GNMIC defines active compounds as having MIC ≤ 16 µg/mL and inactive compounds as having MIC > 32 µg/mL; intermediate MIC values were excluded during model construction. The predicted probability of activity is not a quantitative MIC estimate.

The two endpoints are reported separately. A predicted retention-positive compound is not necessarily antibacterial, and a predicted MIC-active compound is not necessarily retained. Prediction Support is an additional estimate of reliability, not a replacement for the model probability.

## Retention reference setup

The GNRetain row-level reference is recreated locally from Le Goff et al. Supplementary Table S3 rather than distributed with this repository. After installing dependencies, download the original supplementary workbook from [the published study](https://doi.org/10.1038/s41598-025-10208-6), then run:

```bash
python scripts/rehydrate_gnretain_from_s3.py --source-xlsx /path/to/Supplementary_Table_S3.xlsx --root .
```

Launch Streamlit after the reconstruction finishes. See [data sources](DATA_SOURCES.md) for endpoint provenance.

## Installation

GNDiscovery requires **Python 3.12**. Clone the repository and install its dependencies in a new virtual environment.

**Windows (PowerShell):**

```powershell
git clone https://github.com/Jaafarsuhail/GNDiscovery-software.git
cd GNDiscovery-software
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\rehydrate_gnretain_from_s3.py --source-xlsx PATH_TO_S3.xlsx --root .
.\.venv\Scripts\python.exe -m streamlit run streamlit_app\gndiscovery_app.py
```

**Linux / macOS:**

```bash
git clone https://github.com/Jaafarsuhail/GNDiscovery-software.git
cd GNDiscovery-software
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/rehydrate_gnretain_from_s3.py --source-xlsx PATH_TO_S3.xlsx --root .
./.venv/bin/python -m streamlit run streamlit_app/gndiscovery_app.py
```

The application is served locally, typically at `http://localhost:8501`. Linux environments must provide the OpenMP runtime required by LightGBM (see `packages.txt`). Installation and regression tests have been verified on Windows with Python 3.12; the Linux/macOS instructions have not been independently tested.

## Application features

The interface supports single-molecule prediction from SMILES, GNRetain and GNMIC results, nearest-neighbour chemistry, applicability-domain and prediction-support information, dual-endpoint assessment, and batch screening with downloadable results. Example input files are available in `examples/`.

## Reproducibility

The package contains frozen inference estimators, ordered feature manifests, GNMIC reference resources, and the code required to regenerate GNRetain reference tables locally from the published supplementary workbook. The main directories are:

- `streamlit_app/` — user interface
- `app_core/` — descriptor preparation, model inference and prediction support
- `GNRetain221_Runtime_Model/` and `GNRetain221_Support_Assets/` — GNRetain model and reference resources
- `GNRetain_Source_Feature_Engine/` — GNRetain feature generation
- `GNMIC_ECOLI_ATCC25922_Runtime/` — GNMIC model, feature definitions and reference resources
- `qa/` — regression tests

Run the tests from the repository root using the active Python environment:

```bash
python qa/run_regression.py
python app_core/gndiscovery_smoke_test.py
python app_core/gnretain221_smoke_test.py
python qa/verify_deployed_gnretain.py
```

The regression and smoke-test scripts are included above. The underlying full deployment has passed automated Windows checks; when using this data-excluded distribution, run the tests after reconstructing the retention reference tables.

## Scientific context

The GNRetain and GNMIC models address experimentally different endpoints; their outputs should not be interpreted as direct evidence of intracellular localization, mechanism of action, or experimentally measured potency. Predictions should be evaluated in light of their chemical context and supported by experimental validation.

The retention component builds on the measurements reported by **Le Goff et al.**, *Scientific Reports* **15**, 25431 (2025), https://doi.org/10.1038/s41598-025-10208-6. The activity component uses **ChEMBL** bioactivity records; see **Zdrazil et al.**, *Nucleic Acids Research* **52**, D1180–D1192 (2024), https://doi.org/10.1093/nar/gkad1004.

See [third-party notices](THIRD_PARTY_NOTICES.md), [data sources](DATA_SOURCES.md), and [licensing](DATA_LICENSE.md) for source attribution and applicable licensing terms. Original application code is released under the MIT license to the extent the author holds the rights; ChEMBL-derived database content retains CC BY-SA 3.0 terms. The original software code, third-party data, and derived model artifacts are distinct components and may be subject to different rights.
