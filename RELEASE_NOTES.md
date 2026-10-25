# GNDiscovery v1.0.0

GNDiscovery is a local Streamlit application providing independent predictions for two *Escherichia coli* endpoints: cell-associated compound retention and antibacterial activity. It also provides chemical-neighbour context and model-support information for interpreting predictions and screening compound libraries.

## Models and inference

The deployed **GNRetain** classifier is the fixed five-member LightGBM ensemble trained on 5,452 compounds with 221 ordered features and a retention-positive probability threshold of **0.575**. The **GNMIC-EC1792** classifier uses the historical three-member LightGBM ensemble for *E. coli* ATCC 25922, with 1,792 features and a binary activity threshold of **0.510**. The two model probabilities are reported separately.

## Installation and reference data

The repository contains Python 3.12 source code, dependency requirements, serialized inference models, feature manifests, ChEMBL-derived GNMIC reference data, example SMILES, and regression checks. To enable the full GNRetain interface, install the dependencies and use the included `scripts/rehydrate_gnretain_from_s3.py` to reconstruct seven local retention reference tables from Le Goff et al. Supplementary Table S3. These seven compiled tables and the publisher's workbook are not included in the distribution. See `README.md` and `DATA_SOURCES.md` for setup details.

## Provenance and citation

The independently authored application code is licensed under MIT. ChEMBL-derived reference data retain their CC BY-SA 3.0 terms, and the published retention source remains subject to its applicable rights. See `DATA_LICENSE.md` and `THIRD_PARTY_NOTICES.md` for attribution and scope. The software citation is provided in `CITATION.cff`.
