# Benchmark Dataset Access & Reconstruction Protocol

To comply strictly with third-party **Data Use Agreements (DUAs)** and institutional privacy policies, participant-level clinical CSV files (ADNI, OASIS-3) and large pre-extracted histopathology/omics binary `.npz` archives are **not** redistributed directly inside this public GitHub repository.

The benchmark scripts ingest harmonized credentialed CSV exports and pre-extracted feature caches; they do not bundle raw LONI neuroimaging preprocessing or whole-slide tiling pipelines. Prepare those inputs separately and place them under `data_benchmark/` as structured below to reproduce the reported benchmark stages:

1. **ADNI (Alzheimer's Disease Neuroimaging Initiative — Tracks 1, 2, 3A)**
   - Apply for credentialed research access at https://adni.loni.usc.edu/.
   - Export harmonized cohort tables to `data_benchmark/adni_processed/`:
     - `cohort1_adni_ptau181_313roi.csv` (ADNI Cohort 1, $N=1,137$)
     - `cohort2_adni_ptau217_gfap.csv` (ADNI Cohort 2 plasma pTau217/GFAP, $N=289$)

2. **OASIS-3 External Validation Cohort (Track 3B)**
   - Apply for OASIS-3 access at https://www.oasis-brains.org/.
   - Place `cohort4_oasis3_external.csv` and `OASIS3_diagnosis.csv` under `data_benchmark/adni_processed/` ($N=1,026$ external participants).

3. **TCGA Pan-Cancer Multi-Omics & UNI2-h WSIs (Track 4)**
   - Clinical & genomic instability biomarkers plus RNA-Seq 1,000-MAD candidate universe can be fetched directly via GDC / cBioPortal using:
     ```bash
     PYTHONPATH=. python benchmarks/fetch_tcga_1000genes_cbioportal.py
     ```
   - Slide-level UNI2-h embeddings (`TCGA-*_uni2h_slides.npz`, $1,536$-D mean + $1,536$-D std) are stored under `data_benchmark/wsi_data/tcga_uni2h/`.

4. **Multi-Foundation-Model Computational Pathology Caches (Track 5)**
   - Public patch/tile datasets originate from **Camelyon17** (`N=2,500`) and **NCT-CRC-HE-100K / CRC-VAL-HE-7K** (`N=2,349`), with pre-extracted features from 6 foundation models (`uni2_h`, `phikon_v2`, `kaiko_vitl14`, `phikon`, `ctranspath`, `plip`) placed under `data_benchmark/wsi_data/multi_fm_features/`.
   - The Track 5 NCT-CRC evaluation uses the binary tumor-or-stroma endpoint (`is_tumor_or_stroma`) derived from the original 9-class NCT-CRC-HE-100K / CRC-VAL-HE-7K datasets.

5. **Synthetic End-to-End Quickstart (No External Downloads Required)**
   - Run unit tests & theorems:
     ```bash
     pytest -q
     ```
   - Run the synthetic 4-modality pipeline demo:
     ```bash
     python examples/quickstart_synthetic_demo.py
     ```
