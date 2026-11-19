#!/usr/bin/env python3
"""Strictly inductive TCGA RNA screening, lineage and within-cohort audits.

The full run reads the Track 4 inputs and archived splits, and writes only an
aggregate JSON and, if needed, the separate all-gene RNA cache.  Smoke mode
uses synthetic data, does not access the network, and does not write files.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import tempfile
import time
import warnings
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# deepfuzzy's import otherwise probes temporary-directory writability by creating a file.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.environ.get("TMPDIR") or "/tmp")

from deepfuzzy.rough_reducer import FuzzyRoughFeatureReducer
from deepfuzzy.operators import UnifiedDeepFuzzyModel
from deepfuzzy.metrics import bootstrap_confidence_interval, paired_bootstrap_pvalue
from benchmarks import fetch_tcga_1000genes_cbioportal as fetch
from benchmarks.run_full_benchmark import (
    _compute_c_index,
    compute_bh_fdr_map,
    fit_strict_preprocessor,
    pairwise_jaccard_summary,
    set_all_seeds,
    train_deepfuzzy_or_baseline,
)
from benchmarks.run_upgrades_and_pathway_audit import (
    _select_mrmr_top_k,
    run_custom_background_reactome_enrichment,
)

COHORTS = list(fetch.COHORTS_TCGA)
LUNG = {"TCGA-LUAD", "TCGA-LUSC"}
LUNG_GENES = {"SFTPA2", "SFTPD", "NKX2-1"}
SURFACTANT_GENES = {"SFTPA1", "SFTPA2", "SFTPB", "SFTPC", "SFTPD"}
LUNG_LINEAGE_GENES = SURFACTANT_GENES | {"NKX2-1", "NAPSA", "ABCA3", "SLC34A2"}
EXPECTED_N = 2946
EXPECTED_GENES = 18495
EXPECTED_FULL_QC = 15636
CLINICAL = ROOT / "data_benchmark/wsi_data/clinical"
ARCHIVED_RNA = CLINICAL / "tcga_pancancer_1000genes_rnaseq.npz"
ALL_RNA = CLINICAL / "tcga_pancancer_all_18495_proteincoding_rnaseq.npz"
SPLITS = ROOT / "results/oof_predictions_and_splits.npz"
RESULT = ROOT / "results/v024_tcga_inductive_nonlung_owa_audit.json"
GMT = CLINICAL / "ReactomePathways.gmt"
GENOMIC_COLUMNS = [
    "ANEUPLOIDY_SCORE", "FRACTION_GENOME_ALTERED", "MSI_SCORE_MANTIS",
    "MSI_SENSOR_SCORE", "MUTATION_COUNT", "TBL_SCORE", "TMB_NONSYNONYMOUS",
    "cbio_BUFFA_HYPOXIA_SCORE", "cbio_RAGNUM_HYPOXIA_SCORE", "cbio_WINTER_HYPOXIA_SCORE",
]


class OWAFuzzyRoughReducer(FuzzyRoughFeatureReducer):
    """Identical views/pool/greedy search to FuzzyRough, with inverse OWA lower regions.

    For binary outcomes the same-class implications equal one.  Sorting each
    similarity row once in descending order sorts all opposite-class implications
    ascending, with the same-class ones last.  Cumulative opposite-class counts
    therefore give their exact OWA ranks for each observed/permuted labeling.
    This avoids sorting a separate implication matrix for each permutation.
    """

    @torch.no_grad()
    def _eval_view_from_R(self, R_mat: torch.Tensor, ctx: Dict[str, object],
                          exact_f64_obs: bool = False) -> Tuple[float, float]:
        n = int(ctx["N"])
        if n <= 1:
            raise ValueError("OWA dependency requires at least two rows")
        if not bool(ctx["is_binary"]):
            raise ValueError("OWA selector requires binary labels")
        if not torch.isfinite(R_mat).all():
            raise ValueError("Nonfinite similarity encountered")
        L = n - 1
        use_f64 = bool(exact_f64_obs)
        calc_dev = torch.device("cpu") if (use_f64 and R_mat.device.type == "mps") else R_mat.device
        calc_dtype = torch.float64 if use_f64 else torch.float32
        R = R_mat.to(device=calc_dev, dtype=calc_dtype).clone()
        # Diagonal is placed last and discarded. Values are otherwise in [0, 1].
        R.fill_diagonal_(-1.0)
        vals, order = torch.sort(R, dim=1, descending=True)
        vals = vals[:, :L].contiguous()
        order = order[:, :L]
        y_src = ctx["Y1"]
        Y_t = (y_src.to(device=calc_dev) > 0.5).T.contiguous()  # [P, n]
        P = Y_t.shape[0]
        y_ord = Y_t[:, order.reshape(-1)].reshape(P, n, L)
        opp = (Y_t.unsqueeze(-1) != y_ord).to(dtype=calc_dtype)
        ranks = torch.cumsum(opp, dim=-1)
        weights_unnorm = (L + 1.0 - ranks) * opp
        if use_f64:
            c_norm = 2.0 / (L * (L + 1.0))
            lower_all = 1.0 - c_norm * (weights_unnorm * vals.unsqueeze(0)).sum(dim=-1)
            opp_vals = vals.unsqueeze(0) * opp
            row_min = 1.0 - opp_vals.max(dim=-1).values
            row_mean = 1.0 - opp_vals.sum(dim=-1) / float(L)
            tol = 1e-12
            bounds_ok = torch.all(
                (lower_all >= row_min - tol) & (lower_all <= row_mean + tol)
                & (lower_all >= -tol) & (lower_all <= 1.0 + tol)
            )
            if not bool(bounds_ok.item()):
                raise RuntimeError("OWA lower approximation violated its min/mean/[0,1] bounds")
            gammas = lower_all.mean(dim=-1)
        else:
            sums = torch.mv(weights_unnorm.reshape(P, n * L), vals.reshape(n * L))
            gammas = 1.0 - (2.0 / (L * (L + 1.0) * float(n))) * sums
        observed = float(gammas[0].item())
        null = float(gammas[1:].mean().item()) if P > 1 else 0.0
        return observed, null


def qc_statistics(X: np.ndarray, rows: np.ndarray, block: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact finite fraction, median, and unscaled MAD with bounded memory."""
    if len(rows) == 0:
        raise ValueError("No observed training RNA rows")
    d = X.shape[1]
    valid = np.empty(d, dtype=np.float64)
    med = np.empty(d, dtype=np.float32)
    mad = np.empty(d, dtype=np.float32)
    for start in range(0, d, block):
        stop = min(start + block, d)
        part = np.asarray(X[np.ix_(rows, np.arange(start, stop))], dtype=np.float32)
        finite = np.isfinite(part)
        valid[start:stop] = finite.mean(axis=0)
        part[~finite] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(part, axis=0)
            dev = np.nanmedian(np.abs(part - m), axis=0)
        med[start:stop] = m
        mad[start:stop] = dev
    return valid, med, mad


def qc_mask(valid: np.ndarray, med: np.ndarray, mad: np.ndarray) -> np.ndarray:
    return (valid >= 0.95) & np.isfinite(med) & (med >= 1.0) & np.isfinite(mad) & (mad > 0)


def require_equal(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise RuntimeError("Track 4 alignment mismatch: " + name)
    if actual.dtype.kind in "fc" and expected.dtype.kind in "fc":
        equal = np.array_equal(actual, expected, equal_nan=True)
    else:
        equal = np.array_equal(actual, expected)
    if not equal:
        raise RuntimeError("Track 4 alignment mismatch: " + name)


def check_cache(cache: Dict[str, np.ndarray], track: Tuple, archived: Dict[str, np.ndarray],
                check_qc: bool = True) -> None:
    pids, cohorts, y, times, events = track
    n = len(pids)
    if n != EXPECTED_N:
        raise RuntimeError("Expected %d Track 4 rows; got %d" % (EXPECTED_N, n))
    for k, expected in (("patient_ids", np.asarray(pids)), ("cohorts", cohorts),
                        ("y_event", y), ("os_months", times), ("os_event", events)):
        require_equal(k, np.asarray(cache[k]), np.asarray(expected))
        require_equal("archived " + k, np.asarray(archived[k]), np.asarray(expected))
    mask = np.asarray(cache["observed_rna_mask"])
    require_equal("observed_rna_mask", mask, np.asarray(archived["observed_rna_mask"]))
    X = cache["X_rna_log2rsem"]
    names = cache["gene_symbols"]
    if X.shape != (n, EXPECTED_GENES) or X.dtype != np.float32:
        raise RuntimeError("All-gene matrix must be float32 with shape (%d, %d)" % (n, EXPECTED_GENES))
    if len(names) != X.shape[1] or len(set(names.astype(str))) != len(names):
        raise RuntimeError("Invalid or duplicate all-gene symbols")
    if len(cache["entrez_ids"]) != X.shape[1]:
        raise RuntimeError("Entrez IDs do not align to all-gene matrix")
    if not np.all((mask == 0) | (mask == 1)):
        raise RuntimeError("RNA observation mask is not binary")
    if np.isfinite(X[mask < 0.5]).any() or np.any(~np.isfinite(X[mask > 0.5]).any(axis=1)):
        raise RuntimeError("All-gene rows disagree with observed_rna_mask")
    if check_qc:
        valid, med, mad = qc_statistics(X, np.flatnonzero(mask > 0.5))
        n_qc = int(qc_mask(valid, med, mad).sum())
        if n_qc != EXPECTED_FULL_QC:
            raise RuntimeError("Expected %d full-set QC genes, got %d" % (EXPECTED_FULL_QC, n_qc))


def build_all_gene_cache(track: Tuple, archived: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Rebuild without invoking fetch.main or touching either archived RNA file."""
    pids, cohorts, y, times, events = track
    if len(pids) != EXPECTED_N:
        raise RuntimeError("Unexpected patient count before all-gene fetch")
    entrez_to_sym, sym_to_entrez, pc_symbols = fetch.fetch_protein_coding_genes()
    pid_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, pid in enumerate(pids):
        pid_to_rows[pid].append(i)
    study_data = []
    counts: Counter = Counter()
    entrez: Dict[str, int] = {}
    for study in fetch.STUDIES:
        print("Fetching " + study, flush=True)
        raw = fetch._http_get_bytes(fetch.DATAHUB_BASE + "/" + study + "/data_mrna_seq_v2_rsem.txt")
        matched, genes, ids = fetch.parse_study_rsem_tsv(
            raw, set(pids), entrez_to_sym, sym_to_entrez, pc_symbols
        )
        del raw
        counts.update(genes.keys())
        entrez.update({g: int(v) for g, v in ids.items() if v > 0})
        study_data.append((matched, genes))
    names = sorted(g for g, count in counts.items() if count == len(fetch.STUDIES))
    if len(names) != EXPECTED_GENES:
        raise RuntimeError("Expected %d common genes; source returned %d" % (EXPECTED_GENES, len(names)))
    X = np.full((len(pids), len(names)), np.nan, dtype=np.float32)
    observed = np.zeros(len(pids), dtype=np.float32)
    for matched, genes in study_data:
        row_positions = [(row, pos) for pos, pid in enumerate(matched) for row in pid_to_rows[pid]]
        rows = np.asarray([r for r, _ in row_positions], dtype=np.int64)
        positions = np.asarray([p for _, p in row_positions], dtype=np.int64)
        if np.any(observed[rows] > 0):
            raise RuntimeError("A Track 4 patient matched more than one TCGA study")
        observed[rows] = 1.0
        for j, gene in enumerate(names):
            X[rows, j] = genes[gene][positions]
    result = {
        "X_rna_log2rsem": X, "gene_symbols": np.asarray(names, dtype=str),
        "entrez_ids": np.asarray([entrez.get(g, 0) for g in names], dtype=np.int32),
        "observed_rna_mask": observed, "patient_ids": np.asarray(pids, dtype=str),
        "cohorts": cohorts, "y_event": y, "os_months": times, "os_event": events,
    }
    check_cache(result, track, archived)
    CLINICAL.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".all_gene_", suffix=".npz", dir=str(CLINICAL))
    os.close(fd)
    try:
        np.savez_compressed(name, **result)
        os.replace(name, ALL_RNA)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return result


def read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


@contextmanager
def reactome_gmt():
    """Use the local official GMT or fetch it into a temporary directory."""
    if GMT.is_file():
        yield GMT
        return
    print("Fetching official Reactome pathways into temporary storage", flush=True)
    raw = fetch._http_get_bytes(fetch.REACTOME_GMT_URL)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".gmt")]
        if not names:
            raise RuntimeError("Reactome download contained no GMT file")
        contents = archive.read(names[0])
    with tempfile.TemporaryDirectory(prefix="tcga_reactome_gmt_") as temp:
        path = Path(temp) / "ReactomePathways.gmt"
        path.write_bytes(contents)
        yield path


def load_track4_modalities(track: Tuple) -> Dict[str, np.ndarray]:
    """Reconstruct WSI, genomic and clinical inputs in the benchmark's row order."""
    bio = pd.read_csv(fetch.BIO_CSV)
    collected: Dict[str, List[np.ndarray]] = defaultdict(list)
    for ci, cohort in enumerate(COHORTS):
        meta = pd.read_csv(Path(fetch.WSI_DIR) / ("tcga_uni2h/" + cohort + "_uni2h_metadata.csv"))
        with np.load(Path(fetch.WSI_DIR) / ("tcga_uni2h/" + cohort + "_uni2h_slides.npz"), allow_pickle=True) as slide:
            wf = slide["weighted_feats"].astype(np.float32)
            sf = slide["std_feats"].astype(np.float32)
        merged = meta.merge(bio, on="patient_id", how="left")
        if len(merged) != len(wf):
            raise RuntimeError("Metadata join changed slide row count for " + cohort)
        vital = merged["vital_status"].isin(["Alive", "Dead"]).values
        merged = merged[vital].reset_index(drop=True)
        wf, sf = wf[vital], sf[vital]
        dead = merged["vital_status"].values == "Dead"
        months = merged["cbio_OS_MONTHS"].values.astype(np.float32)
        days = np.where(dead, merged["days_to_death"].values.astype(np.float32),
                        merged["days_to_last_followup"].values.astype(np.float32))
        months = np.where(np.isnan(months) | (months <= 0), days / 30.4375, months)
        valid = (~np.isnan(months)) & (months > 0) & (merged["cbio_PFS_STATUS"].notna().values | dead)
        merged = merged[valid].reset_index(drop=True)
        months, dead = months[valid], dead[valid]
        stage = merged["ajcc_stage"].fillna("").astype(str).str.strip()
        st = np.where(stage.str.contains("IV"), 4.0, np.where(stage.str.contains("III"), 3.0,
                      np.where(stage.str.contains("II"), 2.0, np.where(stage.str.contains("I"), 1.0, np.nan))))
        t = merged["ajcc_t"].fillna("").astype(str).str.strip()
        tn = np.where(t.str.contains("T4"), 4.0, np.where(t.str.contains("T3"), 3.0,
                      np.where(t.str.contains("T2"), 2.0, np.where(t.str.contains("T1|T0|Tis"), 1.0, np.nan))))
        nn = merged["ajcc_n"].fillna("").astype(str).str.strip()
        nd = np.where(nn.str.contains("N2|N3"), 2.0,
                      np.where(nn.str.contains("N1"), 1.0, np.where(nn.str.contains("N0"), 0.0, np.nan)))
        gender = merged["gender"].fillna("").astype(str).str.strip().str.lower()
        sex = np.where(gender == "male", 1.0, np.where(gender == "female", 0.0, np.nan))
        onehot = np.zeros((len(merged), len(COHORTS)), dtype=np.float32)
        onehot[:, ci] = 1
        clin = np.column_stack([merged["age_numeric"].values.astype(np.float32), sex, st, tn, nd, onehot])
        prog = merged["cbio_PFS_STATUS"].fillna("").astype(str).str.contains("1:").values
        collected["wsi"].append(np.concatenate([wf[valid], sf[valid]], axis=1))
        collected["gen"].append(merged[GENOMIC_COLUMNS].values.astype(np.float32))
        collected["clin"].append(clin.astype(np.float32))
        collected["patient_ids"].append(merged["patient_id"].astype(str).values)
        collected["cohorts"].append(np.repeat(cohort, len(merged)))
        collected["y"].append((dead | prog).astype(np.float32))
        collected["times"].append(months.astype(np.float32))
        collected["events"].append(dead.astype(np.float32))
    out = {k: np.concatenate(v, axis=0) for k, v in collected.items()}
    for key, expected in zip(("patient_ids", "cohorts", "y", "times", "events"), track):
        require_equal("modalities " + key, out[key], np.asarray(expected))
    if out["wsi"].shape[1] != 3072 or out["gen"].shape[1] != 10 or out["clin"].shape[1] != 12:
        raise RuntimeError("Unexpected Track 4 modality dimensions")
    return out


def cohort_z(Xtr: np.ndarray, Xva: np.ndarray, ctr: np.ndarray, cva: np.ndarray,
             observed_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit means and SD on observed training rows, separately by cohort."""
    A = np.asarray(Xtr, dtype=np.float32).copy()
    B = np.asarray(Xva, dtype=np.float32).copy()
    A[~np.isfinite(A)] = np.nan
    B[~np.isfinite(B)] = np.nan
    for cohort in np.unique(ctr):
        fit = (ctr == cohort) & observed_tr
        if not np.any(fit):
            raise RuntimeError("No observed training RNA for cohort " + str(cohort))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(A[fit], axis=0)
            sd = np.nanstd(A[fit], axis=0) + 1e-5
        if not np.isfinite(mu).all() or not np.isfinite(sd).all():
            raise RuntimeError("Undefined training cohort RNA mean or SD")
        ix_tr, ix_va = ctr == cohort, cva == cohort
        A[ix_tr] = (A[ix_tr] - mu) / sd
        B[ix_va] = (B[ix_va] - mu) / sd
    if not set(np.unique(cva)).issubset(set(np.unique(ctr))):
        raise RuntimeError("Validation cohort absent from training")
    return A, B


def choose_gene_pool(X: np.ndarray, tr: np.ndarray, va: np.ndarray, obs: np.ndarray,
                     cohorts: np.ndarray, residualized: bool, pool_size: int) -> Tuple:
    tr_obs = tr[obs[tr] > 0.5]
    valid, med, mad = qc_statistics(X, tr_obs)
    passed = np.flatnonzero(qc_mask(valid, med, mad))
    if len(passed) < pool_size:
        raise RuntimeError("Only %d genes pass training-fold QC; need %d" % (len(passed), pool_size))
    if residualized:
        dispersion = np.empty(len(passed), dtype=np.float32)
        for start in range(0, len(passed), 256):
            ids = passed[start:start + 256]
            Z, _ = cohort_z(X[np.ix_(tr, ids)], X[np.ix_(va, ids)], cohorts[tr], cohorts[va], obs[tr] > 0.5)
            observed = Z[obs[tr] > 0.5]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmedian(observed, axis=0)
                dispersion[start:start + len(ids)] = np.nanmedian(np.abs(observed - m), axis=0)
        if not np.isfinite(dispersion).all():
            raise RuntimeError("Nonfinite residualized dispersion")
        ranks = np.argsort(-dispersion, kind="mergesort")
    else:
        ranks = np.argsort(-mad[passed], kind="mergesort")
    ids = passed[ranks[:pool_size]]
    tr_rna = np.asarray(X[np.ix_(tr, ids)], dtype=np.float32)
    va_rna = np.asarray(X[np.ix_(va, ids)], dtype=np.float32)
    tr_rna[~np.isfinite(tr_rna)] = np.nan
    va_rna[~np.isfinite(va_rna)] = np.nan
    tr_rna[obs[tr] <= 0.5] = np.nan
    va_rna[obs[va] <= 0.5] = np.nan
    if residualized:
        tr_rna, va_rna = cohort_z(tr_rna, va_rna, cohorts[tr], cohorts[va], obs[tr] > 0.5)
    return ids, tr_rna, va_rna, len(passed)


def prepare_rna(tr: np.ndarray, va: np.ndarray, obs_tr: np.ndarray, obs_va: np.ndarray) -> Tuple:
    if not np.any(obs_tr):
        raise RuntimeError("No observed training RNA")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(tr[obs_tr], axis=0)
    if not np.isfinite(median).all():
        raise RuntimeError("RNA training median is nonfinite")
    tr_imp = np.where(np.isfinite(tr), tr, median)
    va_imp = np.where(np.isfinite(va), va, median)
    mu = tr_imp[obs_tr].mean(axis=0)
    sd = tr_imp[obs_tr].std(axis=0) + 1e-5
    tr_std, va_std = (tr_imp - mu) / sd, (va_imp - mu) / sd
    if not np.isfinite(tr_std).all() or not np.isfinite(va_std).all():
        raise RuntimeError("Nonfinite standardized RNA")
    return tr_imp, tr_std.astype(np.float32), va_std.astype(np.float32)


def fit_selectors(names: List[str], Ximp: np.ndarray, Xstd: np.ndarray, y: np.ndarray,
                  symbols: List[str], fold: int, smoke: bool, device: torch.device) -> Tuple:
    selected: Dict[str, Any] = {}
    times: Dict[str, float] = {}
    scores: Dict[str, Tuple[float, float]] = {}
    for name in names:
        if name == "All-1000":
            selected[name] = np.arange(Xstd.shape[1])
            continue
        start = time.perf_counter()
        if name in ("FuzzyRough-20", "OWA-FuzzyRough-20"):
            cls = FuzzyRoughFeatureReducer if name == "FuzzyRough-20" else OWAFuzzyRoughReducer
            reducer = cls(max_features=20, sigma=0.85, max_samples=4096,
                          n_null_permutations=1 if smoke else 8,
                          n_stability_bootstraps=1 if smoke else 8, seed=42 + fold)
            reducer.fit(Ximp, y, feature_names=symbols, device=str(device))
            selected[name] = np.asarray(reducer.selected_indices_, dtype=np.int64)
            if name == "FuzzyRough-20":
                scores = {g: (float(reducer.univariate_excess_scores_[i]), float(reducer.univariate_scores_[i]))
                          for i, g in enumerate(symbols)}
        elif name == "mRMR-20":
            selected[name] = np.asarray(_select_mrmr_top_k(Xstd, y, k=20))
        elif name == "Lasso-20":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # The old benchmark uses penalty='l1'; l1_ratio is its modern equivalent.
                try:
                    clf = LogisticRegression(penalty="l1", solver="saga", C=0.15, max_iter=400, random_state=42 + fold)
                    clf.fit(Xstd, y)
                except (TypeError, ValueError) as exc:
                    if "penalty" not in str(exc):
                        raise
                    clf = LogisticRegression(l1_ratio=1.0, solver="saga", C=0.15, max_iter=400, random_state=42 + fold)
                    clf.fit(Xstd, y)
            selected[name] = np.argsort(-np.abs(clf.coef_.ravel()), kind="mergesort")[:20]
        elif name == "MI-20":
            mi = mutual_info_classif(Xstd, y, random_state=42 + fold)
            selected[name] = np.argsort(-mi, kind="mergesort")[:20]
        elif name == "PCA-20":
            pca = PCA(n_components=20, random_state=42 + fold)
            pca.fit(Xstd)
            selected[name] = pca
        else:
            raise ValueError("Unknown selector " + name)
        times[name] = time.perf_counter() - start
        if name != "PCA-20" and len(selected[name]) != 20:
            raise RuntimeError("Selector did not return 20 features: " + name)
    return selected, times, scores


def breslow_cox_loss(log_risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Full risk sets; equal-time deaths share the denominator (Breslow ties)."""
    risk = log_risk.reshape(-1).clamp(-15.0, 15.0)
    t = times.reshape(-1)
    e = events.reshape(-1) > 0.5
    if risk.numel() != t.numel() or t.numel() != e.numel():
        raise ValueError("Cox inputs have unequal lengths")
    if not torch.isfinite(t).all() or not torch.isfinite(risk).all():
        raise ValueError("Cox inputs contain nonfinite values")
    if not bool(e.any().item()):
        return risk.sum() * 0.0
    order = torch.argsort(t, descending=True, stable=True)
    ts, rs, es = t[order], risk[order], e[order]
    log_cum = torch.logcumsumexp(rs, dim=0)
    # For each tied time use the last descending-order cumulative sum: t_j >= t_i.
    _, counts = torch.unique_consecutive(ts, return_counts=True)
    ends = torch.cumsum(counts, dim=0) - 1
    denom = torch.repeat_interleave(log_cum[ends], counts)
    return -(rs[es] - denom[es]).sum() / es.sum()


def prob_logit(p: torch.Tensor) -> torch.Tensor:
    return torch.logit(p.clamp(1e-7, 1.0 - 1e-7)).clamp(-15.0, 15.0)


def train_cox(X: List[np.ndarray], masks: np.ndarray, times: np.ndarray, events: np.ndarray,
              seed: int, device: torch.device, epochs: int, embed_dim: int) -> UnifiedDeepFuzzyModel:
    """Same backbone as Track 4, initialized from OS events and trained on full risk sets."""
    set_all_seeds(seed)
    dims = [x.shape[1] for x in X]
    model = UnifiedDeepFuzzyModel(
        modality_dims=dims, embed_dim=embed_dim, num_classes=1, prior_vacuity=0.45,
        conflict_penalty=0.55, use_rough_bottleneck=True, use_feature_gates=True,
        use_tsk_head=True, num_tsk_concepts=min(8, len(dims) + dims[-1]),
        num_tsk_rules=6, dropout=0.18,
    ).to(device)
    with torch.no_grad():
        rels = []
        for m, x in enumerate(X):
            observed = masks[:, m] > 0.5
            yy = events[observed] - events[observed].mean() if observed.any() else events - events.mean()
            xx = x[observed] if observed.any() else x
            corr = np.abs((xx * yy[:, None]).mean(axis=0))
            rels.append(float(np.mean(np.sort(corr)[-min(5, len(corr)):])) )
            z = (corr - corr.mean()) / (corr.std() + 1e-5)
            model.rough_gates[m].copy_(torch.tensor(z * 0.45, device=device, dtype=torch.float32))
        rels_arr = np.asarray(rels, dtype=np.float32)
        rel_z = (rels_arr - rels_arr.mean()) / (rels_arr.std() + 1e-5)
        model.choquet_fusion.fuzzy_density_logits.copy_(torch.tensor(rel_z[:, None] * 0.85, device=device))
    xt = [torch.tensor(x, dtype=torch.float32, device=device) for x in X]
    mt = torch.tensor(masks, dtype=torch.float32, device=device)
    tt = torch.tensor(times, dtype=torch.float32, device=device)
    et = torch.tensor(events, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(epochs):
        mm = mt.clone()
        if ep > 5:
            draw = torch.tensor((rng.random(mm.shape) > 0.20).astype(np.float32), device=device)
            aug = mm * draw
            no_mod = aug.sum(dim=1) < 0.5
            aug[no_mod] = mm[no_mod]
            mm = aug
        inputs = [x * mm[:, m:m + 1] for m, x in enumerate(xt)]
        optimizer.zero_grad()
        out = model(inputs, modality_masks=mm)
        loss = breslow_cox_loss(prob_logit(out["prob"]), tt, et)
        aux = loss.new_zeros(())
        for m, ifn in enumerate(out["modality_ifns"]):
            observed = mm[:, m] > 0.5
            if bool(observed.any().item()):
                aux = aux + breslow_cox_loss(prob_logit(ifn.expected_probability()[observed]), tt[observed], et[observed])
        loss = loss + (0.35 / len(dims)) * aux + 0.04 * out["rough_boundary"].mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Nonfinite Cox loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
    return model.eval().cpu()


def consensus_from_folds(fold_genes: List[List[str]], fold_scores: List[Dict[str, Tuple[float, float]]],
                         n: int = 20) -> List[Dict[str, Any]]:
    counts = Counter(g for genes in fold_genes for g in genes)
    score_lists: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for scores in fold_scores:
        for gene, value in scores.items():
            score_lists[gene].append(value)
    def key(g: str) -> Tuple:
        vals = score_lists.get(g, [])
        excess = np.mean([v[0] for v in vals]) if vals else 0.0
        dependency = np.mean([v[1] for v in vals]) if vals else 0.0
        return (-counts[g], -excess, -dependency, g)
    ranked = sorted(counts, key=key)[:n]
    return [{"rank": i + 1, "gene": g, "fold_frequency": counts[g],
             "mean_univariate_excess_when_in_pool": float(np.mean([v[0] for v in score_lists[g]])) if g in score_lists else None,
             "mean_univariate_dependency_when_in_pool": float(np.mean([v[1] for v in score_lists[g]])) if g in score_lists else None}
            for i, g in enumerate(ranked)]


def safe_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def jaccard_details(fold_sets: List[List[str]]) -> Optional[Dict[str, Any]]:
    if len(fold_sets) < 2:
        return None
    result = pairwise_jaccard_summary(fold_sets)
    sets = [set(genes) for genes in fold_sets]
    result["fold_pairs"] = [
        {"fold_a": i, "fold_b": j, "intersection": len(sets[i] & sets[j]),
         "union": len(sets[i] | sets[j]),
         "jaccard": len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))}
        for i in range(len(sets)) for j in range(i + 1, len(sets))
    ]
    return result


def summarize_predictions(records: Dict[str, Dict[str, Any]], y: np.ndarray, times: np.ndarray,
                          events: np.ndarray, cohorts: np.ndarray, observed_rna: np.ndarray,
                          evaluation: np.ndarray, n_bootstraps: int) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    cohort_names = sorted(set(cohorts[evaluation].tolist()))
    idx = np.flatnonzero(evaluation)
    if len(idx) == 0:
        raise RuntimeError("No validation predictions")
    for name, record in records.items():
        p = record["prob"][idx]
        risk = record["risk"][idx]
        pnw = record["missing"][idx]
        prna = record["rna"][idx]
        if not np.isfinite(p).all() or not np.isfinite(risk).all() or not np.isfinite(pnw).all() or not np.isfinite(prna).all():
            raise RuntimeError("Missing or nonfinite validation predictions for " + name)
        auc, low, high = bootstrap_confidence_interval(p, y[idx], "auroc", n_bootstraps=n_bootstraps)
        per_auc = {c: safe_auc(y[idx][cohorts[idx] == c], p[cohorts[idx] == c]) for c in cohort_names}
        per_cindex = {c: float(_compute_c_index(risk[cohorts[idx] == c], times[idx][cohorts[idx] == c],
                                                events[idx][cohorts[idx] == c])) for c in cohort_names}
        auc_values = [v for v in per_auc.values() if v is not None]
        o = observed_rna[idx] > 0.5
        sets = record["sets"]
        if name == "PCA-20":
            stability = {"mean": None, "std": None, "min": None, "max": None,
                         "note": "PCA components are projections, not gene subsets"}
        elif len(sets) < 2:
            stability = {"mean": None, "std": None, "min": None, "max": None,
                         "note": "Fewer than two folds evaluated"}
        else:
            stability = jaccard_details(sets)
        results[name] = {
            "auroc_4mod": auc, "auroc_4mod_ci_95": [low, high],
            "cohort_auroc": per_auc, "macro_cohort_auroc": float(np.mean(auc_values)) if auc_values else None,
            "macro_auroc_cohorts_used": len(auc_values),
            "os_c_index": float(_compute_c_index(risk, times[idx], events[idx])),
            "within_cohort_os_c_index": per_cindex,
            "within_cohort_os_c_index_mean": float(np.mean(list(per_cindex.values()))) if per_cindex else None,
            "rna_only_observed_auroc": safe_auc(y[idx][o], prna[o]),
            "missing_wsi_and_clin_auroc": safe_auc(y[idx], pnw),
            "gene_stability_jaccard_5fold": stability,
            "selector_fit_time_mean_seconds": float(np.mean(record["fit_times"])) if record["fit_times"] else None,
            "consensus_top20_genes": consensus_from_folds(sets, record["scores"]) if name not in ("All-1000", "PCA-20", "CoxPH_Inductive20") else None,
        }
        if name == "All-1000":
            results[name]["selector_fit_time_note"] = "No supervised selector; shared QC and MAD pool construction are excluded"
        consensus = results[name]["consensus_top20_genes"]
        if consensus is not None:
            overlap = sorted({g["gene"] for g in consensus} & LUNG_GENES)
            results[name]["lung_marker_overlap"] = overlap
            results[name]["lung_marker_overlap_count"] = len(overlap)
    # Every pair uses the same class-stratified resamples as the existing benchmark.
    pairwise: Dict[str, Dict[str, float]] = {name: {} for name in records}
    names = list(records)
    for i, a in enumerate(names):
        pairwise[a][a] = 1.0
        for b in names[i + 1:]:
            value = paired_bootstrap_pvalue(records[a]["prob"][idx], records[b]["prob"][idx],
                                           y[idx], "auroc", n_bootstraps=n_bootstraps)
            pairwise[a][b] = value
            pairwise[b][a] = value
    for ref in ("FuzzyRough-20", "All-1000"):
        if ref not in records:
            continue
        pmap = {name: pairwise[name][ref] for name in names}
        qmap = compute_bh_fdr_map(pmap, exclude_keys=(ref,))
        for name in names:
            results[name]["p_vs_" + ref] = pmap[name]
            results[name]["q_bh_vs_" + ref] = qmap[name]
    return {"models": results, "paired_bootstrap_p_values_all_pairs": pairwise}


def _predict(model: torch.nn.Module, X: List[np.ndarray], mask: np.ndarray,
             cox: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        tensors = [torch.tensor(x, dtype=torch.float32) for x in X]
        masks = torch.tensor(mask, dtype=torch.float32)
        prob = model(tensors, modality_masks=masks)["prob"].reshape(-1)
        missing_mask = masks.clone()
        missing_mask[:, 0] = 0
        missing_mask[:, 3] = 0
        missing_X = [torch.zeros_like(tensors[0]), tensors[1], tensors[2], torch.zeros_like(tensors[3])]
        prob_missing = model(missing_X, modality_masks=missing_mask)["prob"].reshape(-1)
        risk = prob_logit(prob) if cox else prob
        p = torch.sigmoid(risk) if cox else prob
        pm = torch.sigmoid(prob_logit(prob_missing)) if cox else prob_missing
    return p.numpy(), pm.numpy(), risk.numpy()


def run_analysis(data: Dict[str, np.ndarray], gene_names: np.ndarray, archived_genes: set,
                 fold_ids: np.ndarray, variant: str, smoke: bool, device: torch.device,
                 gmt_path: Optional[Path] = None,
                 selector_device: Optional[torch.device] = None) -> Dict[str, Any]:
    sel_device = selector_device if selector_device is not None else device
    if variant not in ("all", "nonlung", "residualized"):
        raise ValueError("Unknown audit variant")
    n = len(data["y"])
    all_selectors = ["FuzzyRough-20", "OWA-FuzzyRough-20", "mRMR-20", "Lasso-20", "MI-20", "PCA-20", "All-1000"]
    nonlung_selectors = [name for name in all_selectors if name != "PCA-20"]
    residual_selectors = ["FuzzyRough-20", "OWA-FuzzyRough-20", "mRMR-20", "Lasso-20", "All-1000"]
    selectors = all_selectors if variant == "all" else (nonlung_selectors if variant == "nonlung" else residual_selectors)
    model_names = selectors + ["CoxPH_Inductive20"]
    records = {name: {"prob": np.full(n, np.nan), "risk": np.full(n, np.nan), "missing": np.full(n, np.nan),
                      "rna": np.full(n, np.nan), "sets": [], "fit_times": [], "scores": []}
               for name in model_names}
    pools: List[List[str]] = []
    fold_summaries: List[Dict[str, Any]] = []
    evaluation = np.zeros(n, dtype=bool)
    obs = data["observed_rna_mask"]
    pool_size = min(1000, data["rna"].shape[1]) if smoke else 1000
    for fold in range(5):
        tr, va = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        if len(va) == 0 or len(tr) == 0:
            raise RuntimeError("Invalid or empty validation fold")
        obs_tr, obs_va = obs[tr] > 0.5, obs[va] > 0.5
        ids, rna_tr, rna_va, n_qc = choose_gene_pool(data["rna"], tr, va, obs, data["cohorts"],
                                                      variant == "residualized", pool_size)
        symbols = gene_names[ids].astype(str).tolist()
        pools.append(symbols)
        tr_imp, tr_std, va_std = prepare_rna(rna_tr, rna_va, obs_tr, obs_va)
        selected, fit_times, fr_scores = fit_selectors(selectors, tr_imp[obs_tr], tr_std[obs_tr],
                                                        data["y"][tr][obs_tr], symbols, fold, smoke, sel_device)
        Wtr, Wva = data["wsi"][tr], data["wsi"][va]
        wsi_mean = Wtr.mean(axis=0, keepdims=True)
        wsi_std = Wtr.std(axis=0, keepdims=True) + 1e-5
        pca = PCA(n_components=48, random_state=42 + fold)
        Wtr_pca = pca.fit_transform((Wtr - wsi_mean) / wsi_std).astype(np.float32)
        Wva_pca = pca.transform((Wva - wsi_mean) / wsi_std).astype(np.float32)
        other_tr, other_va, mask_tr_other, mask_va_other = fit_strict_preprocessor(
            [Wtr_pca, data["gen"][tr], data["clin"][tr]],
            [Wva_pca, data["gen"][va], data["clin"][va]],
        )
        masks_tr = np.column_stack([mask_tr_other[:, 0], obs_tr, mask_tr_other[:, 1], mask_tr_other[:, 2]]).astype(np.float32)
        masks_va = np.column_stack([mask_va_other[:, 0], obs_va, mask_va_other[:, 1], mask_va_other[:, 2]]).astype(np.float32)
        weights = np.where(data["y"][tr] == 1, 1.0, np.minimum(1.0, data["times"][tr] / 36.0)).astype(np.float32)
        fr_inputs = None
        fr_rna_predictions = None
        for name in selectors:
            sel = selected[name]
            if name == "PCA-20":
                rtr = sel.transform(tr_std).astype(np.float32)
                rva = sel.transform(va_std).astype(np.float32)
                rtr[~obs_tr] = np.nan
                rva[~obs_va] = np.nan
                pp_tr, pp_va, _, _ = fit_strict_preprocessor([rtr], [rva])
                rtr_s, rva_s = pp_tr[0], pp_va[0]
            else:
                rtr_s = np.clip(tr_std[:, sel], -5.0, 5.0) * obs_tr[:, None]
                rva_s = np.clip(va_std[:, sel], -5.0, 5.0) * obs_va[:, None]
                genes = [symbols[int(i)] for i in sel]
                records[name]["sets"].append(genes)
            records[name]["fit_times"].extend([fit_times[name]] if name in fit_times else [])
            if name == "All-1000":
                records[name]["fit_times"].append(0.0)
            if name == "FuzzyRough-20":
                records[name]["scores"].append(fr_scores)
            Xtr = [other_tr[0], rtr_s.astype(np.float32), other_tr[1], other_tr[2]]
            Xva = [other_va[0], rva_s.astype(np.float32), other_va[1], other_va[2]]
            clf = LogisticRegression(C=0.25, max_iter=300)
            clf.fit(Xtr[1][obs_tr], data["y"][tr][obs_tr])
            rna_p = clf.predict_proba(Xva[1])[:, 1]
            model = train_deepfuzzy_or_baseline(
                "DeepFuzzyTorch", Xtr, masks_tr, data["y"][tr], epochs=1 if smoke else 28,
                lr=1.8e-3, embed_dim=12 if smoke else 40,
                ablation_flags={"use_rough_bottleneck": True, "use_feature_gates": True},
                sample_weight=weights, seed=200 + fold, device=device,
            )
            p, pm, risk = _predict(model, Xva, masks_va)
            records[name]["prob"][va], records[name]["missing"][va], records[name]["rna"][va] = p, pm, rna_p
            records[name]["risk"][va] = risk
            if name == "FuzzyRough-20":
                fr_inputs, fr_rna_predictions = (Xtr, Xva), rna_p
        if fr_inputs is None or fr_rna_predictions is None:
            raise RuntimeError("FuzzyRough tensors missing for Cox control")
        cox = train_cox(fr_inputs[0], masks_tr, data["times"][tr], data["events"][tr],
                        200 + fold, device, 1 if smoke else 28, 12 if smoke else 40)
        p, pm, risk = _predict(cox, fr_inputs[1], masks_va, cox=True)
        cr = records["CoxPH_Inductive20"]
        cr["prob"][va], cr["missing"][va], cr["rna"][va] = p, pm, fr_rna_predictions
        cr["risk"][va] = risk
        cr["sets"].append(records["FuzzyRough-20"]["sets"][-1])
        evaluation[va] = True
        overlap = len(set(symbols) & archived_genes)
        fold_summaries.append({"fold": fold, "train_rows": len(tr), "validation_rows": len(va),
                               "observed_training_rna": int(obs_tr.sum()), "qc_gene_count": n_qc,
                               "candidate_pool_size": len(symbols), "archived_1000_overlap_count": overlap,
                               "archived_1000_jaccard": overlap / max(1, len(set(symbols) | archived_genes))})
        print("%s fold %d completed (QC=%d, pool=%d)" % (variant, fold + 1, n_qc, len(symbols)), flush=True)
    if not smoke and not evaluation.all():
        raise RuntimeError("Not every patient received exactly one validation prediction")
    output = summarize_predictions(records, data["y"], data["times"], data["events"], data["cohorts"],
                                   obs, evaluation, 20 if smoke else 1000)
    out_fr = output["models"]["FuzzyRough-20"]
    cox_result = output["models"]["CoxPH_Inductive20"]
    cox_result["gene_stability_jaccard_5fold"] = out_fr["gene_stability_jaccard_5fold"]
    cox_result["consensus_top20_genes"] = out_fr["consensus_top20_genes"]
    cox_result["lung_marker_overlap"] = out_fr["lung_marker_overlap"]
    cox_result["lung_marker_overlap_count"] = out_fr["lung_marker_overlap_count"]
    cox_result["selector_fit_time_mean_seconds"] = out_fr["selector_fit_time_mean_seconds"]
    background = sorted(set().union(*(set(pool) for pool in pools)))
    fr_consensus = out_fr["consensus_top20_genes"]
    consensus_names = {g["gene"] for g in fr_consensus}
    output.update({
        "variant": variant,
        "n_patients": n,
        "n_events": int(data["y"].sum()),
        "n_os_deaths": int(data["events"].sum()),
        "cohort_counts": {str(c): int(np.sum(data["cohorts"] == c)) for c in sorted(set(data["cohorts"]))},
        "clinical_dimensions": int(data["clin"].shape[1]),
        "folds": fold_summaries,
        "candidate_pool_jaccard_5fold": jaccard_details(pools),
        "consensus_ranking": "selection frequency, then mean univariate excess and dependency across folds in which the gene entered the candidate pool, then gene symbol",
        "lung_marker_overlap": sorted(set(g["gene"] for g in fr_consensus) & LUNG_GENES),
        "lung_marker_overlap_count": len(set(g["gene"] for g in fr_consensus) & LUNG_GENES),
        "lung_surfactant_inspection": {
            "predefined_surfactant_genes": sorted(SURFACTANT_GENES),
            "surfactant_genes_in_consensus": sorted(consensus_names & SURFACTANT_GENES),
            "surfactant_gene_count": len(consensus_names & SURFACTANT_GENES),
            "surfactant_consensus_fraction": len(consensus_names & SURFACTANT_GENES) / max(1, len(consensus_names)),
            "predefined_lung_lineage_genes": sorted(LUNG_LINEAGE_GENES),
            "lung_lineage_genes_in_consensus": sorted(consensus_names & LUNG_LINEAGE_GENES),
            "lung_lineage_gene_count": len(consensus_names & LUNG_LINEAGE_GENES),
        },
        "reactome_background": {"definition": "union of the five in-fold top-MAD 1000-gene candidate pools", "N_bg": len(background)},
    })
    if not smoke:
        if gmt_path is None or not gmt_path.is_file():
            raise FileNotFoundError("A ReactomePathways.gmt file is required for enrichment")
        with tempfile.TemporaryDirectory(prefix="tcga_reactome_") as temp:
            enrichment = run_custom_background_reactome_enrichment(
                [g["gene"] for g in fr_consensus], background, str(gmt_path),
                os.path.join(temp, "enrichment.json"), os.path.join(temp, "pathways.csv"),
            )
        enrichment["method"] = ("Exact one-sided hypergeometric using the union of in-fold 1000-gene pools; "
                                "BH over all human Reactome pathways with K_bg >= 2; displayed pathways have k >= 2")
        output["reactome_enrichment"] = enrichment
    return output


def construct_folds(y: np.ndarray, cohorts: np.ndarray, nonlung: bool) -> np.ndarray:
    labels = np.char.add(np.char.add(cohorts.astype(str), "_"), y.astype(int).astype(str)) if nonlung else y
    fold_ids = np.full(len(y), -1, dtype=np.int32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, va) in enumerate(skf.split(np.zeros(len(y)), labels)):
        if np.any(fold_ids[va] >= 0):
            raise RuntimeError("Duplicate validation membership")
        fold_ids[va] = fold
    if np.any(fold_ids < 0):
        raise RuntimeError("Incomplete folds")
    return fold_ids


def select_nonlung(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    keep = ~np.isin(data["cohorts"], list(LUNG))
    out = {k: v[keep] for k, v in data.items()}
    retained = [i for i, c in enumerate(COHORTS) if c not in LUNG]
    out["clin"] = np.column_stack([out["clin"][:, :5], out["clin"][:, 5 + np.asarray(retained)]])
    if out["clin"].shape[1] != 10 or set(out["cohorts"]) != set(COHORTS) - LUNG:
        raise RuntimeError("Non-lung cohort or 10-D clinical encoding mismatch")
    return out


def synthetic_data() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(2401)
    n, g = 120, 120
    cohorts = np.asarray([COHORTS[i % 7] for i in range(n)])
    y = np.asarray([(i // 7) % 2 for i in range(n)], dtype=np.float32)
    y = y[rng.permutation(n)]
    # Guarantee both classes within each cohort for cohort-target stratification.
    for c in COHORTS:
        ix = np.flatnonzero(cohorts == c)
        y[ix] = np.arange(len(ix)) % 2
        rng.shuffle(y[ix])
    events = (rng.random(n) < (0.25 + 0.45 * y)).astype(np.float32)
    times = rng.uniform(2, 100, n).astype(np.float32)
    rna = (3.5 + rng.normal(size=(n, g)) + y[:, None] * np.r_[np.full(8, 0.45), np.zeros(g - 8)]).astype(np.float32)
    for i, c in enumerate(COHORTS):
        rna[cohorts == c, 8:16] += i * 0.2
    obs = np.ones(n, dtype=np.float32)
    obs[[3, 21, 54, 87]] = 0
    rna[obs == 0] = np.nan
    rna[1, 0], rna[4, 1] = np.nan, np.nan
    wsi = rng.normal(size=(n, 64)).astype(np.float32)
    gen = rng.normal(size=(n, 10)).astype(np.float32)
    clin = np.column_stack([rng.normal(size=(n, 5)), np.eye(7)[[COHORTS.index(c) for c in cohorts]]]).astype(np.float32)
    symbols = np.asarray(["GENE_%03d" % i for i in range(g)])
    return {"rna": rna, "observed_rna_mask": obs, "wsi": wsi, "gen": gen, "clin": clin,
            "cohorts": cohorts, "y": y, "times": times, "events": events}, symbols


def run_unit_checks() -> None:
    """Check the vectorized OWA against the literal definition and Cox ties by hand."""
    rng = np.random.default_rng(501)
    x = rng.normal(size=(9, 4)).astype(np.float32)
    y = np.asarray([0, 0, 1, 1, 0, 1, 0, 1, 1], dtype=np.float32)
    reducer = OWAFuzzyRoughReducer(max_features=2, seed=1)
    z, _, _ = reducer._robust_standardize_fit(x)
    col = torch.tensor(z[:, 0:1], dtype=torch.float64)
    R = torch.exp(-torch.abs(col - col.T) / reducer.sigma)
    context = reducer._build_view_context(torch.tensor(z), y, 0.08, 2, rng)
    got, got_null = reducer._eval_view_from_R(R, context, exact_f64_obs=True)
    expected = []
    for labels in context["Y1"].numpy().T:
        rows = []
        for i in range(len(y)):
            imp = [min(1.0, 1.0 - float(R[i, j]) + float(labels[i] == labels[j]))
                   for j in range(len(y)) if i != j]
            vals = np.sort(imp)
            L = len(vals)
            weights = 2.0 * np.arange(L, 0, -1) / (L * (L + 1.0))
            lower = float(np.dot(weights, vals))
            if lower < vals[0] - 1e-12 or lower > vals.mean() + 1e-12 or not 0 <= lower <= 1:
                raise RuntimeError("Literal OWA bound check failed")
            rows.append(lower)
        expected.append(float(np.mean(rows)))
    if not np.allclose([got, got_null], [expected[0], np.mean(expected[1:])], rtol=0, atol=1e-12):
        raise RuntimeError("Vectorized OWA differs from literal off-diagonal weighted sorting")
    t = torch.tensor([3.0, 2.0, 2.0, 1.0])
    e = torch.tensor([0.0, 1.0, 1.0, 1.0])
    log_risk = torch.tensor([0.0, np.log(2), np.log(3), np.log(4)], requires_grad=True)
    got_loss = breslow_cox_loss(log_risk, t, e)
    expected_loss = -(np.log(2 / 6) + np.log(3 / 6) + np.log(4 / 10)) / 3
    if not np.isclose(float(got_loss.detach()), expected_loss, rtol=1e-6):
        raise RuntimeError("Breslow Cox tied-time risk sets are incorrect")
    got_loss.backward()
    if log_risk.grad is None or not torch.isfinite(log_risk.grad).all():
        raise RuntimeError("Cox loss does not have finite gradients")


def smoke_test() -> None:
    torch.set_num_threads(2)
    run_unit_checks()
    data, symbols = synthetic_data()
    if data["rna"].shape != (120, 120) or len(set(data["cohorts"])) != 7:
        raise RuntimeError("Unexpected synthetic fixture dimensions")
    ids = construct_folds(data["y"], data["cohorts"], False)
    nonlung = select_nonlung(data)
    nonlung_ids = construct_folds(nonlung["y"], nonlung["cohorts"], True)
    device = torch.device("cpu")
    for variant, d, folds in (("all", data, ids), ("nonlung", nonlung, nonlung_ids),
                               ("residualized", data, ids)):
        output = run_analysis(d, symbols, set(symbols), folds, variant, True, device)
        json.dumps(json_safe(output), allow_nan=False)
        if len(output["folds"]) != 5 or len(output["candidate_pool_jaccard_5fold"]["fold_pairs"]) != 10:
            raise RuntimeError("Smoke test did not aggregate all five folds")
        if variant == "nonlung" and set(output["cohort_counts"]) & LUNG:
            raise RuntimeError("Non-lung smoke test retained a lung cohort")
        if output["models"]["CoxPH_Inductive20"]["auroc_4mod"] is None:
            raise RuntimeError("Synthetic Cox AUROC is undefined")
        if variant == "all" and set(output["models"]) != {
            "FuzzyRough-20", "OWA-FuzzyRough-20", "mRMR-20", "Lasso-20", "MI-20",
            "PCA-20", "All-1000", "CoxPH_Inductive20"
        }:
            raise RuntimeError("Smoke test did not execute all required selectors")
    print("Smoke test passed: all selectors, CoxPH, non-lung and within-cohort residualization.", flush=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        raise TypeError("Array in output would risk publishing participant-level vectors")
    return value


def full_run() -> None:
    torch.set_num_threads(4)
    run_unit_checks()
    track = fetch.load_track4_cohort()
    if not ARCHIVED_RNA.is_file():
        raise FileNotFoundError("The archived pooled 1000-gene RNA file is required: " + str(ARCHIVED_RNA))
    archived = read_npz(ARCHIVED_RNA)
    if ALL_RNA.is_file():
        cache = read_npz(ALL_RNA)
        check_cache(cache, track, archived)
    else:
        cache = build_all_gene_cache(track, archived)
    data = load_track4_modalities(track)
    data["rna"] = cache["X_rna_log2rsem"]
    data["observed_rna_mask"] = cache["observed_rna_mask"]
    gene_names = cache["gene_symbols"]
    if not SPLITS.is_file():
        raise FileNotFoundError("Track 4 archived fold IDs are required: " + str(SPLITS))
    split = read_npz(SPLITS)
    for key, actual in (("tcga_y", data["y"]), ("tcga_os_time_months", data["times"]), ("tcga_os_event", data["events"])):
        require_equal("archived split " + key, split[key], actual)
    exact_ids = construct_folds(data["y"], data["cohorts"], False)
    require_equal("tcga_fold_ids", split["tcga_fold_ids"], exact_ids)
    nonlung = select_nonlung(data)
    nonlung_ids = construct_folds(nonlung["y"], nonlung["cohorts"], True)
    archived_genes = set(archived["gene_symbols"].astype(str))
    if len(archived_genes) != 1000:
        raise RuntimeError("Archived pooled RNA set does not contain 1,000 unique genes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selector_device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    protocol = {
            "purpose": "Strictly inductive 18,495-gene pan-cancer, non-lung, and within-cohort audit",
            "raw_cache_shape": list(cache["X_rna_log2rsem"].shape),
            "full_set_qc_expressed_genes_verified": EXPECTED_FULL_QC,
            "qc": "training observed RNA only; finite fraction >= 0.95, median >= 1, MAD > 0",
            "pool": "top 1000 training-fold MAD genes; residualized analysis ranks training cohort-Z MAD after raw QC",
            "residualization": "per-cohort training-observed mean and SD + 1e-5 applied to train and validation before screening and fusion",
            "selector_settings": {"k": 20, "sigma": 0.85, "max_samples": 4096,
                                  "univariate_cap_for_more_than_128_features": 768, "candidate_pool": 60,
                                  "null_permutations": 8, "stability_views": 8, "seed": "42+fold"},
            "owa": "off-diagonal ascending implicators, weights 2*(L+1-r)/(L*(L+1)); inherited null-calibrated forward search",
            "cohort_splits": "all/residualized: exact archived tcga_fold_ids; nonlung: five-fold stratified cohort and event, shuffle seed 42",
            "classification_training": "matched 4-modality UnifiedDeepFuzzyModel, 28 epochs, lr 0.0018, embed 40, seed 200+fold, maturity weights",
            "rna_only": "observed-validation AUROC of in-fold logistic regression C=0.25 on the same RNA representation",
            "cox": ("same backbone and selected tensors; logit of fused probability as continuous log-risk clamped [-15,15]; "
                    "unweighted full-training-risk-set Breslow loss, plus 0.35 times mean observed-modality Cox loss and "
                    "0.04 times mean rough boundary; OS-event gate initialization, no composite-outcome weights"),
            "cox_auroc": "secondary composite-event classification AUROC of sigmoid(log-risk), monotone in Cox risk",
            "c_index": "benchmark _compute_c_index; greater risk for earlier OS death; tied times excluded and tied risks receive half credit",
            "bootstrap": "1000 class-stratified paired resamples; BH separately within each comparator-reference family",
            "reactome": "union of five fold candidate pools; all human pathways with K_bg >= 2 form BH family, k >= 2 shown",
            "lung_markers": "SFTPA2, SFTPD and NKX2-1 are reported as observed; marker exclusion is not forced",
            "no_participant_level_output": True,
    }
    with reactome_gmt() as gmt_path:
        results = {
            "protocol": protocol,
            "B1a_all_cohorts": run_analysis(data, gene_names, archived_genes, exact_ids, "all", False, device, gmt_path, selector_device),
            "B1b_nonlung": run_analysis(nonlung, gene_names, archived_genes, nonlung_ids, "nonlung", False, device, gmt_path, selector_device),
            "B1c_within_cohort_z": run_analysis(data, gene_names, archived_genes, exact_ids, "residualized", False, device, gmt_path, selector_device),
        }
    safe = json_safe(results)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".v024_", suffix=".json", dir=str(RESULT.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(safe, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, RESULT)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    print("Wrote aggregate audit to " + str(RESULT), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Run file-free synthetic pipeline checks")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        full_run()


if __name__ == "__main__":
    main()
