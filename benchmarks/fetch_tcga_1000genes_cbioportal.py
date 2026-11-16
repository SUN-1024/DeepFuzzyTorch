#!/usr/bin/env python3
"""Fetch label-free transductive pan-cancer protein-coding RNA-Seq from cBioPortal DataHub / REST API.

Scope & Protocol Disclosure (Unsupervised Transductive Pre-Screen):
- Zero manual `PRIORITY_CANCER_GENES` lists, zero survival/outcome labels used.
- Zero alphabetical or lowest-Entrez-ID truncation.
- Downloads full assayed PanCanAtlas continuous RSEM expression (`data_mrna_seq_v2_rsem.txt`,
  ~20,531 genes; 18,495 common protein-coding genes) across all 7 TCGA cohorts (`brca`,
  `coadread`, `kirc`, `kirp`, `luad`, `lusc`, `ucec`), aligns row-for-row to N=2,946 Track 4
  patients, applies an unsupervised, label-free transductive cohort-wide variance pre-screen
  (protein-coding genes assayed in >=95% of observed patients with median log2(RSEM+1) >= 1.0,
  yielding 15,636 QC-passing genes), and caches:
  1. `tcga_pancancer_raw_proteincoding_rnaseq.npz`: Top 5,000 highest-MAD expressed
     protein-coding genes (`log2(RSEM + 1)` continuous values + `observed_rna_mask`, unscaled
     so downstream supervised feature selection and standardization run inside CV folds).
  2. `tcga_pancancer_1000genes_rnaseq.npz`: Unsupervised top 1,000 highest-MAD expressed
     protein-coding genes (`log2(RSEM + 1)` continuous values + `observed_rna_mask`; zero label
     usage, zero manual pathway curation). Also downloads official human `ReactomePathways.gmt`
     for custom-background hypergeometric enrichment testing.
"""
from __future__ import annotations

import io
import json
import os
import time
import urllib.request
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WSI_DIR = os.path.join(ROOT, "data_benchmark/wsi_data")
BIO_CSV = os.path.join(WSI_DIR, "clinical/tcga_precision_oncology_sample_biomarkers.csv")
OUT_RAW5000_NPZ = os.path.join(
    WSI_DIR, "clinical/tcga_pancancer_raw_proteincoding_rnaseq.npz"
)
OUT_1000_NPZ = os.path.join(
    WSI_DIR, "clinical/tcga_pancancer_1000genes_rnaseq.npz"
)
OUT_REACTOME_GMT = os.path.join(WSI_DIR, "clinical/ReactomePathways.gmt")

CBIO_API = "https://www.cbioportal.org/api"
DATAHUB_BASE = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
)
REACTOME_GMT_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"

COHORTS_TCGA = [
    "TCGA-BRCA",
    "TCGA-COAD",
    "TCGA-KIRC",
    "TCGA-KIRP",
    "TCGA-LUAD",
    "TCGA-LUSC",
    "TCGA-UCEC",
]

STUDIES = [
    "brca_tcga_pan_can_atlas_2018",
    "coadread_tcga_pan_can_atlas_2018",
    "kirc_tcga_pan_can_atlas_2018",
    "kirp_tcga_pan_can_atlas_2018",
    "luad_tcga_pan_can_atlas_2018",
    "lusc_tcga_pan_can_atlas_2018",
    "ucec_tcga_pan_can_atlas_2018",
]


def _http_get_bytes(
    url: str, headers: Dict[str, str] | None = None, retries: int = 4
) -> bytes:
    hdrs = {"User-Agent": "Mozilla/5.0"}
    if headers:
        hdrs.update(headers)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  Retry {attempt + 1}/{retries} for {url}: {exc}")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed GET {url}")


def load_track4_cohort() -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct exact row-for-row Track 4 patient cohort (N = 2,946)."""
    bio_df = pd.read_csv(BIO_CSV)
    pids_all: List[str] = []
    cohorts_all: List[str] = []
    y_list: List[np.ndarray] = []
    os_m_list: List[np.ndarray] = []
    os_ev_list: List[np.ndarray] = []

    for c_name in COHORTS_TCGA:
        meta_p = os.path.join(WSI_DIR, f"tcga_uni2h/{c_name}_uni2h_metadata.csv")
        meta = pd.read_csv(meta_p)
        merged = meta.merge(bio_df, on="patient_id", how="left")
        valid_vital = merged["vital_status"].isin(["Alive", "Dead"]).values
        merged = merged[valid_vital].reset_index(drop=True)
        os_dead_raw = merged["vital_status"].values == "Dead"
        os_m_raw = merged["cbio_OS_MONTHS"].values.astype(np.float32)
        fallback_days = np.where(
            os_dead_raw,
            merged["days_to_death"].values.astype(np.float32),
            merged["days_to_last_followup"].values.astype(np.float32),
        )
        os_m_raw = np.where(
            np.isnan(os_m_raw) | (os_m_raw <= 0), fallback_days / 30.4375, os_m_raw
        )
        has_pfs_raw = merged["cbio_PFS_STATUS"].notna().values
        strict_valid = (~np.isnan(os_m_raw)) & (os_m_raw > 0) & (has_pfs_raw | os_dead_raw)
        merged = merged[strict_valid].reset_index(drop=True)
        os_dead = os_dead_raw[strict_valid]
        os_m = os_m_raw[strict_valid]
        pfs_prog = merged["cbio_PFS_STATUS"].fillna("").astype(str).str.contains("1:").values
        y_c = (os_dead | pfs_prog).astype(np.float32)

        c_pids = merged["patient_id"].astype(str).tolist()
        pids_all.extend(c_pids)
        cohorts_all.extend([c_name] * len(c_pids))
        y_list.append(y_c)
        os_m_list.append(os_m.astype(np.float32))
        os_ev_list.append(os_dead.astype(np.float32))

    return (
        pids_all,
        np.array(cohorts_all, dtype="<U9"),
        np.concatenate(y_list),
        np.concatenate(os_m_list),
        np.concatenate(os_ev_list),
    )


def fetch_protein_coding_genes() -> Tuple[Dict[int, str], Dict[str, int], set]:
    """Return Entrez->HUGO, HUGO->Entrez, and set of protein-coding HUGO symbols."""
    raw = _http_get_bytes(
        f"{CBIO_API}/genes?pageSize=50000&projection=DETAILED",
        headers={"Accept": "application/json"},
    )
    all_genes = json.loads(raw.decode("utf-8"))
    entrez_to_sym: Dict[int, str] = {}
    sym_to_entrez: Dict[str, int] = {}
    pc_symbols: set = set()
    for g in all_genes:
        if g.get("type") != "protein-coding":
            continue
        eid = int(g.get("entrezGeneId", -1))
        sym = str(g.get("hugoGeneSymbol", "")).strip()
        if eid > 0 and sym:
            entrez_to_sym[eid] = sym
            sym_to_entrez[sym] = eid
            pc_symbols.add(sym)
    return entrez_to_sym, sym_to_entrez, pc_symbols


def parse_study_rsem_tsv(
    tsv_bytes: bytes,
    target_patients: set,
    entrez_to_sym: Dict[int, str],
    sym_to_entrez: Dict[str, int],
    pc_symbols: set,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, int]]:
    """Parse cBioPortal PanCanAtlas `data_mrna_seq_v2_rsem.txt` aligned to target patients."""
    text = tsv_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        return [], {}, {}

    header = lines[0].split("\t")
    sample_cols = header[2:]

    pid_to_col: Dict[str, Tuple[int, str]] = {}
    for col_idx, sid in enumerate(sample_cols):
        pid = sid[:12]
        if pid not in target_patients:
            continue
        if pid not in pid_to_col:
            pid_to_col[pid] = (col_idx, sid)
        else:
            prev_sid = pid_to_col[pid][1]
            if sid.endswith("-01") and not prev_sid.endswith("-01"):
                pid_to_col[pid] = (col_idx, sid)

    matched_pids = sorted(pid_to_col.keys())
    if not matched_pids:
        return [], {}, {}
    col_indices = np.array([pid_to_col[p][0] + 2 for p in matched_pids], dtype=np.int64)

    gene_to_vals: Dict[str, np.ndarray] = {}
    gene_to_eid: Dict[str, int] = {}
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        hugo = parts[0].strip()
        eid_str = parts[1].strip()
        eid = int(eid_str) if eid_str.isdigit() else -1
        if not hugo and eid in entrez_to_sym:
            hugo = entrez_to_sym[eid]
        if not hugo:
            continue
        if hugo not in pc_symbols and eid not in entrez_to_sym:
            continue
        if eid <= 0:
            eid = sym_to_entrez.get(hugo, 0)

        raw_vals = np.empty(len(col_indices), dtype=np.float32)
        for out_i, c_i in enumerate(col_indices):
            if c_i < len(parts):
                s_val = parts[c_i].strip()
                if s_val == "" or s_val in ("NA", "NaN", "null"):
                    raw_vals[out_i] = np.nan
                else:
                    try:
                        v = float(s_val)
                        raw_vals[out_i] = (
                            np.log2(max(v, 0.0) + 1.0) if np.isfinite(v) else np.nan
                        )
                    except ValueError:
                        raw_vals[out_i] = np.nan
            else:
                raw_vals[out_i] = np.nan

        if hugo in gene_to_vals:
            prev = gene_to_vals[hugo]
            if np.isfinite(raw_vals).sum() > np.isfinite(prev).sum():
                gene_to_vals[hugo] = raw_vals
                gene_to_eid[hugo] = eid
        else:
            gene_to_vals[hugo] = raw_vals
            gene_to_eid[hugo] = eid

    return matched_pids, gene_to_vals, gene_to_eid


def download_reactome_gmt() -> int:
    """Download official human ReactomePathways.gmt for exact custom-background enrichment."""
    raw_zip = _http_get_bytes(REACTOME_GMT_URL)
    zf = zipfile.ZipFile(io.BytesIO(raw_zip))
    gmt_name = [n for n in zf.namelist() if n.endswith(".gmt")][0]
    gmt_bytes = zf.read(gmt_name)
    os.makedirs(os.path.dirname(OUT_REACTOME_GMT), exist_ok=True)
    with open(OUT_REACTOME_GMT, "wb") as f:
        f.write(gmt_bytes)
    return len(gmt_bytes.decode("utf-8", errors="replace").splitlines())


def main() -> None:
    t0 = time.time()
    patient_ids, cohorts_arr, y_event, os_months, os_event = load_track4_cohort()
    pid_to_indices: Dict[str, List[int]] = {}
    for i, pid in enumerate(patient_ids):
        pid_to_indices.setdefault(pid, []).append(i)
    target_patients = set(patient_ids)
    N = len(patient_ids)
    print(f"Loaded exact Track 4 TCGA patient cohort: N = {N}")

    entrez_to_sym, sym_to_entrez, pc_symbols = fetch_protein_coding_genes()
    print(
        f"Fetched {len(pc_symbols)} official human protein-coding HUGO symbols from cBioPortal."
    )

    per_study_pids: List[List[str]] = []
    per_study_dicts: List[Dict[str, np.ndarray]] = []
    gene_study_counts: Dict[str, int] = {}
    gene_entrez_map: Dict[str, int] = {}

    for study in STUDIES:
        t_s = time.time()
        url = f"{DATAHUB_BASE}/{study}/data_mrna_seq_v2_rsem.txt"
        tsv_bytes = _http_get_bytes(url)
        matched_pids, gene_to_vals, gene_to_eid = parse_study_rsem_tsv(
            tsv_bytes, target_patients, entrez_to_sym, sym_to_entrez, pc_symbols
        )
        per_study_pids.append(matched_pids)
        per_study_dicts.append(gene_to_vals)
        for g, eid in gene_to_eid.items():
            gene_study_counts[g] = gene_study_counts.get(g, 0) + 1
            if eid > 0:
                gene_entrez_map[g] = eid
        print(
            f"  [{study}] {len(tsv_bytes)/1e6:.1f} MB parsed in {time.time()-t_s:.2f}s | "
            f"matched patients={len(matched_pids)} | protein-coding genes={len(gene_to_vals)}"
        )

    common_genes = sorted(
        [g for g, cnt in gene_study_counts.items() if cnt == len(STUDIES)]
    )
    G_all = len(common_genes)
    print(f"Common protein-coding genes assayed across all 7 TCGA cohorts: {G_all}")

    X_all = np.full((N, G_all), np.nan, dtype=np.float32)
    obs_patient = np.zeros(N, dtype=bool)
    for matched_pids, gene_to_vals in zip(per_study_pids, per_study_dicts):
        row_idx_list: List[int] = []
        val_pos_list: List[int] = []
        for pos_i, p in enumerate(matched_pids):
            for r_i in pid_to_indices[p]:
                row_idx_list.append(r_i)
                val_pos_list.append(pos_i)
        row_idx = np.array(row_idx_list, dtype=np.int64)
        val_pos = np.array(val_pos_list, dtype=np.int64)
        obs_patient[row_idx] = True
        for g_idx, g_sym in enumerate(common_genes):
            X_all[row_idx, g_idx] = gene_to_vals[g_sym][val_pos]

    n_obs = int(obs_patient.sum())
    X_obs = X_all[obs_patient]
    print(
        f"Patients with matched PanCanAtlas RNA-Seq primary tumor profiles: {n_obs}/{N} ({100.0*n_obs/N:.1f}%)"
    )

    # Unsupervised technical quality filter on observed samples:
    # 1. Non-missing in >= 95% of RNA-Seq-observed patients
    # 2. Median log2(RSEM + 1) >= 1.0 (expressed above background noise)
    valid_frac = np.isfinite(X_obs).mean(axis=0)
    with np.errstate(all="ignore"):
        med_expr = np.nanmedian(X_obs, axis=0)
        mad_expr = np.nanmedian(np.abs(X_obs - med_expr[None, :]), axis=0)

    keep_qc = (
        (valid_frac >= 0.95)
        & np.isfinite(med_expr)
        & (med_expr >= 1.0)
        & np.isfinite(mad_expr)
        & (mad_expr > 0.0)
    )
    qc_indices = np.where(keep_qc)[0]
    print(
        f"Expressed protein-coding genes passing QC (valid>=95%, median log2(RSEM+1)>=1.0): {len(qc_indices)}"
    )

    # Rank strictly by unsupervised Median Absolute Deviation (MAD) descending
    qc_mads = mad_expr[qc_indices]
    order_desc = np.argsort(-qc_mads, kind="mergesort")
    ranked_indices = qc_indices[order_desc]

    # 1. Save top 5,000 raw expressed protein-coding genes (unscaled log2(RSEM+1))
    k_raw = min(5000, len(ranked_indices))
    idx_5000 = ranked_indices[:k_raw]
    genes_5000 = np.array([common_genes[i] for i in idx_5000], dtype="<U20")
    eids_5000 = np.array(
        [gene_entrez_map.get(common_genes[i], 0) for i in idx_5000], dtype=np.int32
    )
    X_raw_5000 = X_all[:, idx_5000].astype(np.float32)

    np.savez_compressed(
        OUT_RAW5000_NPZ,
        X_rna_log2rsem=X_raw_5000,
        gene_symbols=genes_5000,
        entrez_ids=eids_5000,
        gene_mad=mad_expr[idx_5000].astype(np.float32),
        gene_median=med_expr[idx_5000].astype(np.float32),
        observed_rna_mask=obs_patient.astype(np.float32),
        patient_ids=np.array(patient_ids, dtype="<U12"),
        cohorts=cohorts_arr,
        y_event=y_event,
        os_months=os_months,
        os_event=os_event,
        total_common_proteincoding_genes=np.int64(G_all),
        total_qc_expressed_genes=np.int64(len(qc_indices)),
    )
    print(f"Saved raw {X_raw_5000.shape} protein-coding cache to {OUT_RAW5000_NPZ}")

    # 2. Save label-free transductive top 1,000 highest-MAD protein-coding genes
    idx_1000 = ranked_indices[:1000]
    genes_1000 = np.array([common_genes[i] for i in idx_1000], dtype="<U20")
    eids_1000 = np.array(
        [gene_entrez_map.get(common_genes[i], 0) for i in idx_1000], dtype=np.int32
    )
    X_1000_with_nan = X_all[:, idx_1000].astype(np.float32)
    nan_mask_1000 = ~np.isfinite(X_1000_with_nan)
    X_1000_clean = np.where(nan_mask_1000, 0.0, X_1000_with_nan).astype(np.float32)

    np.savez_compressed(
        OUT_1000_NPZ,
        X_rna_1000=X_1000_clean,
        X_rna_1000_with_nan=X_1000_with_nan,
        nan_mask_1000=nan_mask_1000,
        observed_rna_mask=obs_patient.astype(np.float32),
        obs_rna=obs_patient,
        patient_ids=np.array(patient_ids, dtype="<U12"),
        cohorts=cohorts_arr,
        gene_symbols=genes_1000,
        gene_names=genes_1000,
        entrez_ids=eids_1000,
        gene_mad=mad_expr[idx_1000].astype(np.float32),
        gene_median=med_expr[idx_1000].astype(np.float32),
        y_event=y_event,
        os_months=os_months,
        os_event=os_event,
        selection_protocol=np.array(
            "Label-free transductive top-1000 pan-cancer MAD across all 7-cohort expressed protein-coding genes "
            "(cBioPortal PanCanAtlas log2(RSEM+1), valid>=95%, median>=1.0; zero manual gene list)",
            dtype=object,
        ),
        total_common_proteincoding_genes=np.int64(G_all),
        total_qc_expressed_genes=np.int64(len(qc_indices)),
    )
    print(f"Saved label-free transductive {X_1000_clean.shape} RNA-Seq matrix to {OUT_1000_NPZ}")
    print(f"Top 25 label-free transductive highest-MAD genes: {genes_1000[:25].tolist()}")

    n_gmt = download_reactome_gmt()
    print(
        f"Downloaded official ReactomePathways.gmt ({n_gmt} human pathways) to {OUT_REACTOME_GMT}"
    )
    print(f"Completed label-free transductive TCGA RNA-Seq rebuild in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
