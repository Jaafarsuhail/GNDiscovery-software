# GNDiscovery data sources and local reconstruction

GNDiscovery predicts two distinct endpoints: cell-associated retention and binary MIC activity in *E. coli*. Each model uses experimentally derived reference information. Training measurements and model software have different provenance.

## Bacterial retention

Retention data are from Le Goff F, Hazemann J, Christen L, *et al.* **Measurement and prediction of small molecule retention by Gram-negative bacteria based on a large-scale LC/MS screen.** *Scientific Reports* **15**, 25431 (2025). DOI: https://doi.org/10.1038/s41598-025-10208-6.

The published **Supplementary Table S3** contains 6,416 classified entries and original SMILES strings for 6,407 entries. GNRetain's deployed retention reference consists of 5,452 selected compounds. The selection is specified by the 1-based data-row identifiers in `data_manifests/gnretain_s3_row_ids.txt` (the first data record appears at Excel row 7). Identifiers and selection order are supplied, but molecular structures and retention labels from the source study are not included in this package.

To construct the local reference tables, download Supplementary Table S3 from the publication's *Supplementary Information* section. Then run, using Python 3.12 with GNDiscovery dependencies installed:

```bash
python scripts/rehydrate_gnretain_from_s3.py --source-xlsx /path/to/Supplementary_Table_S3.xlsx --root .
```

The script reads the original workbook, selects the 5,452 recorded source rows, standardizes each SMILES with GNDiscovery's frozen RDKit pipeline, and reconstructs the seven runtime reference tables for neighbour retrieval, physicochemical diagnostics, and two PCA projections. It does not upload, transmit or commit the publisher's workbook or regenerated rows. The generated tables are excluded by `.gitignore`.

The frozen model reference contains a documented label difference from the source workbook: record **4208** is classified *Positive* in Supplementary Table S3 but labelled **RN** in GNRetain's deployed training reference. Two display-only differences also occur in the historical visualization reference: a stereochemical depiction for record **2178** and a formal-charge annotation for record **3638**. These differences are handled explicitly in the reconstruction script to preserve the originally deployed reference tables. They do not change the trained estimators.

The reconstruction was validated against the private deployed GNRetain reference: **5,452/5,452** training records and all seven reconstructed tables matched at absolute tolerance 10⁻⁶ for numeric entries and exact equality for strings and identifiers. The source workbook used for this parity verification has SHA-256 `6f42d8cdee34fbc559a25129a18f3c70bbddd0ac1b4bc2afcadb45de7202e27a`. A publisher-hosted copy may have different workbook metadata; matching workbook structure and contents are validated by the script.

The published research article is offered under **CC BY-NC-ND 4.0**. Readers remain responsible for observing the publisher's source terms. This code package does not grant new rights over the supplementary tables or user-reconstructed data.

## Antibacterial activity

GNMIC-EC1792 uses a curated *E. coli* ATCC 25922 subset of ChEMBL experimental MIC records, with 10,148 training compounds and a fixed three-model LightGBM ensemble. ChEMBL is published under **CC BY-SA 3.0** (https://creativecommons.org/licenses/by-sa/3.0/). The ChEMBL source and derived covered database content must retain the appropriate credit and ShareAlike conditions; these terms do **not** automatically apply to independently authored GNDiscovery source code.

ChEMBL citation: Zdrazil B, *et al.* **The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods.** *Nucleic Acids Research* **52**, D1180–D1192 (2024). https://doi.org/10.1093/nar/gkad1004. License guidance: https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions.

The GNMIC model and accompanying ChEMBL-derived records remain unchanged in this data-excluded candidate. The underlying ChEMBL extraction and MIC consensus curation cannot yet be rebuilt from a plain current ChEMBL download with validated row-level parity from the materials distributed here; the historical 10,148-row curated training table is retained as a versioned derived resource. ChEMBL's commercial-property caveats must also be observed; the distributed cohort contains no intentionally imported commercial-property predictions.

## Package scope

This distribution supports **reconstruction of GNRetain inference references from the published supplementary workbook**, rather than redistributing the compiled assay-reference tables. The rebuilt tables are intended for local use. The present package preserves the historical GNMIC inference model and its ChEMBL-derived reference assets; it does not include a complete independent retraining workflow for GNMIC.
