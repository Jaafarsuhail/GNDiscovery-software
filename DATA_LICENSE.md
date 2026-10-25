# Software and data licensing

GNDiscovery distinguishes the application source code, trained models and scientific data. A license for one category must not be read as permission to reuse every file in the repository.

**GNDiscovery source code.** The `LICENSE` file gives the MIT license for the independently authored GNDiscovery code and documentation to the extent the named copyright holder has the authority to grant it. It does not relicense embedded third-party software, published datasets or external reference data. All third-party dependencies retain their own licenses.

**ChEMBL.** The processed GNMIC-EC1792 training cohort and the reference content derived from it are sourced from ChEMBL and are made available subject to Creative Commons Attribution–ShareAlike 3.0 Unported (**CC BY-SA 3.0**), rather than MIT. The upstream data are credited to ChEMBL, EMBL-EBI and Zdrazil et al. (2024). The cohort comprises 10,148 processed records; filtering, standardization, binary MIC labels and model-specific fingerprint representations are changes made during GNDiscovery data preparation. Do not interpret this notice as a license covering unrelated third-party material.

Original data license: https://creativecommons.org/licenses/by-sa/3.0/  
Source: https://www.ebi.ac.uk/chembl/  
ChEMBL publication: https://doi.org/10.1093/nar/gkad1004

**Le Goff retention data.** The complete study's supplementary reference tables are **not included**. Users download Supplementary Table S3 separately from the publisher and locally generate seven GNRetain reference CSVs with `scripts/rehydrate_gnretain_from_s3.py`. GNDiscovery does not grant the user redistribution rights to that third-party workbook or the reconstructed reference table. The publisher describes the article as CC BY-NC-ND 4.0. See `DATA_SOURCES.md`.

**Trained models and fitted resources.** The serialized GNRetain model and other trained resources are technically distinct from original source code, and neither the MIT license nor this notice constitutes a determination of whether their redistribution is permitted by all applicable upstream rights. Public distribution of those assets must have an independently supportable basis. This archive is a release candidate until such a basis is confirmed.
