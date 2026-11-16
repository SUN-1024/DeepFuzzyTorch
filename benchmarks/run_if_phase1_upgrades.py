"""
Phase-1 Execution Script for Information Fusion Revision & Benchmark Upgrades.

Computes and persists into `results/benchmark_results.json` and
`results/oof_predictions_and_splits.npz`:
  1. Complete-case 4-modality audit on ADNI Cohort 1 (N_comp=854 vs full N=1,137
     with natural missingness and same-visit amyloid reference n=924), plus ADAS13
     imputation disclosure (6/1,137).
  2. Five additional baselines evaluated under matched 5-fold CV on ADNI Cohort 1
     and Cohort 2 (plus 15 combinatorial missingness subsets and BH-FDR):
       - DDEF (adapted from Shao et al., Information Fusion 2024)
       - FUML (adapted from Duan et al., ICML 2025)
       - XGBoost (native NaN handling + modality indicator bits)
       - CatBoost (nan_mode='Min' + modality indicator bits)
       - TabPFN v2 (Nature 2025, ModelVersion.V2 + modality indicator bits)
  3. Separated cross-modal conflict detection audit at gamma=0.8 (and full sweep
     gamma in [0, 1]): composite 0.5*(C_O + pi_fused) (=0.6542), pure C_O, and
     pure pi_fused across all Cohort 1 models and ablations.
  4. Matched-coverage selective referral curves across identical retained coverages
     and fixed operating sample counts (n=198 deduplicated real-world discordant ADNI
     and N=1,137 full Cohort 1) with exact Clopper-Pearson 95% CIs, AURC, and
     paired significance comparisons.
"""

from __future__ import annotations

import atexit
import json
import os
import pickle
import struct
import subprocess
import sys
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as scipy_beta
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
torch.set_num_threads(4)

from deepfuzzy import (
    CombinatorialMissingModalityBenchmark,
    CrossModalityConflictInjector,
    FuzzyRoughFeatureReducer,
    UnifiedDeepFuzzyModel,
    bootstrap_confidence_interval,
    compute_binary_metrics,
    compute_decision_curve_analysis,
    paired_bootstrap_pvalue,
)
from benchmarks.run_full_benchmark import (
    attach_bh_fdr_to_table,
    fit_strict_preprocessor,
    load_adni_cohort1,
    set_all_seeds,
    train_deepfuzzy_or_baseline,
)
from benchmarks.run_upgrades_and_pathway_audit import (
    clopper_pearson_ci,
    evaluate_dedup_cross_cohort_triage_nested,
    evaluate_nested_discordant_triage,
)


def _binary_entropy_np(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    h = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return np.clip(h, 0.0, 1.0).astype(np.float32)


def _build_tabular_matrix(mod_arrays: List[np.ndarray], mask_arr: np.ndarray) -> np.ndarray:
    """
    Concatenate standardized modality arrays with native NaNs for unobserved modalities
    alongside explicit binary modality-observation indicator columns.
    """
    parts: List[np.ndarray] = []
    for m_i, x_m in enumerate(mod_arrays):
        obs = mask_arr[:, m_i : m_i + 1] > 0.5
        parts.append(np.where(obs, np.asarray(x_m, dtype=np.float32), np.nan).astype(np.float32))
    parts.append(np.asarray(mask_arr, dtype=np.float32))
    return np.concatenate(parts, axis=1)


_GBDT_WORKER_SRC = r"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import pickle
import struct
import sys

_proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)

import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

_models = {}
_next_id = 1
_stdin = sys.stdin.buffer
while True:
    hdr = _read_exact(_stdin, 4)
    if hdr is None:
        break
    (sz,) = struct.unpack("!I", hdr)
    raw = _read_exact(_stdin, sz)
    if raw is None:
        break
    cmd = pickle.loads(raw)
    op = cmd[0]
    if op == "close":
        resp = pickle.dumps(("ok", True), protocol=pickle.HIGHEST_PROTOCOL)
        _proto_out.write(struct.pack("!I", len(resp)) + resp)
        break
    elif op == "fit":
        _, model_name, X_tr_tab, y_int, seed = cmd
        if model_name == "XGBoost":
            clf = XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.5,
                min_child_weight=3,
                eval_metric="logloss",
                random_state=int(seed),
                n_jobs=1,
            )
        elif model_name == "CatBoost":
            clf = CatBoostClassifier(
                iterations=400,
                depth=5,
                learning_rate=0.04,
                l2_leaf_reg=4.0,
                nan_mode="Min",
                verbose=0,
                random_seed=int(seed),
                thread_count=1,
            )
        else:
            raise ValueError(f"Unknown GBDT model: {model_name}")
        clf.fit(X_tr_tab, y_int)
        mid = _next_id
        _next_id += 1
        _models[mid] = clf
        resp = pickle.dumps(("ok", mid), protocol=pickle.HIGHEST_PROTOCOL)
    elif op == "predict_proba":
        _, mid, X_tab = cmd
        probs = _models[mid].predict_proba(X_tab).astype(np.float32)
        resp = pickle.dumps(("ok", probs), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        resp = pickle.dumps(("err", f"Unknown op: {op}"), protocol=pickle.HIGHEST_PROTOCOL)
    _proto_out.write(struct.pack("!I", len(resp)) + resp)
"""


class _GBDTWorkerClient:
    """Isolated non-PyTorch subprocess for XGBoost & CatBoost to avoid macOS ARM64 libomp collision."""

    def __init__(self) -> None:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _GBDT_WORKER_SRC],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )
        atexit.register(self.close)

    def _call(self, payload: Any) -> Any:
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(struct.pack("!I", len(data)) + data)
        self.proc.stdin.flush()
        hdr = bytearray()
        while len(hdr) < 4:
            chunk = self.proc.stdout.read(4 - len(hdr))
            if not chunk:
                raise RuntimeError("GBDT worker process exited unexpectedly")
            hdr.extend(chunk)
        (sz,) = struct.unpack("!I", bytes(hdr))
        body = bytearray()
        while len(body) < sz:
            chunk = self.proc.stdout.read(sz - len(body))
            if not chunk:
                raise RuntimeError("GBDT worker pipe truncated")
            body.extend(chunk)
        status, val = pickle.loads(bytes(body))
        if status != "ok":
            raise RuntimeError(f"GBDT worker error: {val}")
        return val

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self._call(("close",))
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


_GBDT_WORKER: Optional[_GBDTWorkerClient] = None


def _get_gbdt_worker() -> _GBDTWorkerClient:
    global _GBDT_WORKER
    if _GBDT_WORKER is None or _GBDT_WORKER.proc.poll() is not None:
        _GBDT_WORKER = _GBDTWorkerClient()
    return _GBDT_WORKER


class _RemoteGBDTClassifier:
    def __init__(self, worker: _GBDTWorkerClient, model_id: int) -> None:
        self._worker = worker
        self._model_id = int(model_id)

    def predict_proba(self, X_tab: np.ndarray) -> np.ndarray:
        return self._worker._call(("predict_proba", self._model_id, np.asarray(X_tab, dtype=np.float32)))


def _fit_tabular_baseline(
    model_name: str,
    X_tr_list: List[np.ndarray],
    mask_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
    tabpfn_device: str,
):
    X_tr_tab = _build_tabular_matrix(X_tr_list, mask_tr)
    y_int = np.asarray(y_tr, dtype=np.int64).ravel()
    if model_name in ("XGBoost", "CatBoost"):
        worker = _get_gbdt_worker()
        mid = worker._call(("fit", model_name, X_tr_tab, y_int, int(seed)))
        return _RemoteGBDTClassifier(worker, mid)
    if model_name == "TabPFN":
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            device=tabpfn_device,
            n_estimators=8,
            fit_mode="fit_with_cache",
            random_state=seed,
        )
        clf.fit(X_tr_tab, y_int)
        return clf
    raise ValueError(f"Unknown tabular model: {model_name}")


def _predict_tabular(clf, X_list: List[np.ndarray], mask_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_miss = np.sum(mask_arr > 0.5, axis=1) < 0.5
    X_tab = _build_tabular_matrix(X_list, mask_arr)
    probs = clf.predict_proba(X_tab)[:, 1].astype(np.float32)
    probs = np.where(all_miss, np.float32(0.5), probs)
    hes = _binary_entropy_np(probs)
    hes = np.where(all_miss, np.float32(1.0), hes)
    conf = np.zeros_like(probs, dtype=np.float32)
    return probs, hes, conf


def _nested_ecdf_percentile(s_tr: np.ndarray, s_va: np.ndarray) -> np.ndarray:
    """Mid-rank empirical CDF of validation scores against training calibration pool (in [0, 1])."""
    tr = np.asarray(s_tr, dtype=np.float64).ravel()
    va = np.asarray(s_va, dtype=np.float64).ravel()
    lt = np.mean(tr[:, None] < va[None, :], axis=0)
    le = np.mean(tr[:, None] <= va[None, :], axis=0)
    return (0.5 * (lt + le)).astype(np.float64)


def _eval_subset_metrics_table(
    oof_probs: Dict[str, np.ndarray],
    oof_hes: Dict[str, np.ndarray],
    y: np.ndarray,
    subset_mask: Optional[np.ndarray] = None,
    model_order: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    idx_mask = np.ones(len(y), dtype=bool) if subset_mask is None else np.asarray(subset_mask, dtype=bool)
    y_sub = y[idx_mask]
    p_ours = oof_probs["DeepFuzzyTorch"][idx_mask]
    keys = model_order if model_order is not None else list(oof_probs.keys())
    tbl: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        if k not in oof_probs:
            continue
        p_k = oof_probs[k][idx_mask]
        h_k = oof_hes[k][idx_mask]
        m_res = compute_binary_metrics(p_k, y_sub, uncertainty=h_k)
        _, au_lo, au_hi = bootstrap_confidence_interval(p_k, y_sub, "auroc", n_bootstraps=1000, seed=42)
        _, br_lo, br_hi = bootstrap_confidence_interval(p_k, y_sub, "brier", n_bootstraps=1000, seed=42)
        m_res["auroc_ci"] = [float(au_lo), float(au_hi)]
        m_res["brier_ci"] = [float(br_lo), float(br_hi)]
        m_res["p_value_vs_ours"] = (
            1.0
            if k == "DeepFuzzyTorch"
            else float(paired_bootstrap_pvalue(p_ours, p_k, y_sub, "auroc", n_bootstraps=1000, seed=42))
        )
        tbl[k] = m_res
    attach_bh_fdr_to_table(tbl)
    return tbl


def _compute_calib_entry(probs: np.ndarray, y: np.ndarray, ece: float, brier: float) -> Dict[str, Any]:
    bins = np.linspace(0.0, 1.0, 9)
    mean_pred, frac_pos = [], []
    p_arr = np.asarray(probs, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    for b_i in range(8):
        msk = (p_arr >= bins[b_i]) & (p_arr <= bins[b_i + 1] if b_i == 7 else p_arr < bins[b_i + 1])
        if int(np.sum(msk)) >= 5:
            mean_pred.append(float(np.mean(p_arr[msk])))
            frac_pos.append(float(np.mean(y_arr[msk])))
    return {
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
        "ece": float(ece),
        "brier": float(brier),
    }


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _compute_matched_coverage_grid(
    model_names: List[str],
    probs_map: Dict[str, np.ndarray],
    rank_scores_map: Dict[str, np.ndarray],
    raw_tie_map: Dict[str, np.ndarray],
    y_sub: np.ndarray,
    keep_counts: List[Tuple[str, int]],
) -> Dict[str, Any]:
    n_tot = int(len(y_sub))
    out_models: Dict[str, Any] = {}
    for m in model_names:
        p = np.asarray(probs_map[m], dtype=np.float64).ravel()
        r_primary = np.asarray(rank_scores_map[m], dtype=np.float64).ravel()
        r_tie = np.asarray(raw_tie_map[m], dtype=np.float64).ravel()
        order = np.lexsort((r_tie, r_primary))
        bm_full = compute_binary_metrics(p, y_sub, uncertainty=r_primary)
        pts: Dict[str, Any] = {}
        for label_k, k_ret in keep_counts:
            k_eff = min(n_tot, max(1, int(k_ret)))
            idx = order[:k_eff]
            sub_m = compute_binary_metrics(p[idx], y_sub[idx])
            k_corr = int(np.sum((p[idx] >= 0.5).astype(np.int64) == y_sub[idx].astype(np.int64)))
            ci_lo, ci_hi = clopper_pearson_ci(k_corr, k_eff)
            pts[label_k] = {
                "n_retained": int(k_eff),
                "coverage": float(k_eff / max(1, n_tot)),
                "k_correct": k_corr,
                "acc": float(sub_m["acc"]),
                "acc_ci_cp": [float(ci_lo), float(ci_hi)],
                "auroc": float(sub_m["auroc"]),
                "f1": float(sub_m["f1"]),
                "brier": float(sub_m["brier"]),
            }
        out_models[m] = {
            "aurc": float(bm_full["aurc"]),
            "points": pts,
        }

    from scipy.stats import binomtest as _binomtest

    if "DeepFuzzyTorch" in out_models:
        p_ours_full = np.asarray(probs_map["DeepFuzzyTorch"], dtype=np.float64).ravel()
        corr_ours = (p_ours_full >= 0.5).astype(np.int64) == y_sub.astype(np.int64)
        for m in model_names:
            if m == "DeepFuzzyTorch":
                out_models[m]["mcnemar_full_p_vs_ours"] = 1.0
            else:
                p_m_full = np.asarray(probs_map[m], dtype=np.float64).ravel()
                corr_m = (p_m_full >= 0.5).astype(np.int64) == y_sub.astype(np.int64)
                n01 = int(np.sum(corr_ours & (~corr_m)))
                n10 = int(np.sum((~corr_ours) & corr_m))
                out_models[m]["mcnemar_full_p_vs_ours"] = (
                    float(_binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if (n01 + n10) > 0 else 1.0
                )
    return {
        "n_total": n_tot,
        "grid": [{"key": k, "n_retained": int(c), "coverage": float(c / max(1, n_tot))} for k, c in keep_counts],
        "models": out_models,
    }


def _paired_aurc_bootstrap_p(p_a: np.ndarray, u_a: np.ndarray, p_b: np.ndarray, u_b: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aa = compute_binary_metrics(p_a[idx], y[idx], uncertainty=u_a[idx])["aurc"]
        bb = compute_binary_metrics(p_b[idx], y[idx], uncertainty=u_b[idx])["aurc"]
        diffs[b] = bb - aa  # >0 means model A has lower AURC (better) than B
    p_ge = float(np.mean(diffs <= 0.0))
    p_le = float(np.mean(diffs >= 0.0))
    return float(min(1.0, 2.0 * min(p_ge, p_le)))


def main() -> None:
    t_start = time.time()
    set_all_seeds(42)
    cpu_dev = torch.device("cpu")
    tabpfn_dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[Phase-1 Upgrades] Start | TorchReplayDev=cpu | TabPFNDev={tabpfn_dev}")

    json_path = os.path.join(WORKSPACE_DIR, "results/benchmark_results.json")
    npz_path = os.path.join(WORKSPACE_DIR, "results/oof_predictions_and_splits.npz")
    adni_dir = os.path.join(WORKSPACE_DIR, "data_benchmark/adni_processed")

    with open(json_path, "r", encoding="utf-8") as f:
        master = json.load(f)
    npz_loaded = np.load(npz_path, allow_pickle=True)
    npz_data: Dict[str, Any] = {k: npz_loaded[k] for k in npz_loaded.files}

    # =========================================================================
    # 1. ADNI COHORT 1 REPLAY + NEW BASELINES + N_comp=854 + SEPARATED CONFLICT
    # =========================================================================
    adni1 = load_adni_cohort1(adni_dir)
    c1_df = adni1["df_amyloid"].copy().reset_index(drop=True)
    X1_raw = adni1["X_modalities"]
    y1 = adni1["y_amyloid"]
    mri1_cols = adni1["mri_cols"]
    mod_names = adni1["modality_names"]
    same_visit_bl_924_mask = adni1["same_visit_bl_ref_mask"]
    N1 = len(y1)

    n_adas13_missing = int(c1_df["ADAS13"].isna().sum())
    print(f"[Audit C1] N_total={N1}, n_same_visit_924={int(same_visit_bl_924_mask.sum())}, ADAS13_missing={n_adas13_missing}")

    concept_names_8 = [
        "MRI_NetSupport", "Plasma_NetSupport", "FDG_NetSupport", "Clin_NetSupport",
        "APOE4", "MMSE", "ADAS13", "CDRSB",
    ]
    existing_model_names = [
        "DeepFuzzyTorch", "RCML", "QMF", "DrFuse", "EvidentialDL",
        "GatedAttention", "DempsterShafer", "DeepTSK", "LateAvg", "EarlyConcat",
    ]
    ablation_configs = {
        "Ablation_wo_VacuityMask": {"disable_vacuity_mask": True},
        "Ablation_wo_Choquet": {"uniform_choquet": True},
        "Ablation_wo_ConflictDiscount": {"conflict_penalty": 0.0},
        "Ablation_wo_FuzzyRough": {"use_rough_bottleneck": False, "use_feature_gates": True},
        "Ablation_wo_FuzzyRough_Unreduced320": {"use_rough_bottleneck": False, "use_feature_gates": False, "unreduced_mri": True},
        "Ablation_wo_TSKHead": {"use_tsk_head": False},
    }
    new_neural_names = ["DDEF", "FUML"]
    tabular_names = ["XGBoost", "CatBoost", "TabPFN"]
    all_neural_eval_keys = existing_model_names + new_neural_names + list(ablation_configs.keys())
    all_c1_report_keys = existing_model_names + new_neural_names + tabular_names + list(ablation_configs.keys())
    baseline_and_ours_c1 = existing_model_names + new_neural_names + tabular_names

    has_pl1 = (c1_df["plasma_ptau181_bl"].notna() & (c1_df["blood_is_bl_visit"].fillna(0) == 1)).values
    clin1_pos = ((c1_df["DX_bl"] != "CN") | (c1_df["CDRSB"] >= 0.5)).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fids_c1 = np.zeros(N1, dtype=np.int32)
    fold_val_indices: List[np.ndarray] = []
    fold_val_tensors: List[List[torch.Tensor]] = []
    fold_val_tensors_unred: List[List[torch.Tensor]] = []
    fold_val_masks: List[torch.Tensor] = []

    fold_models_c1: Dict[str, List[Any]] = {k: [] for k in all_c1_report_keys}
    oof_c1_probs: Dict[str, np.ndarray] = {k: np.zeros(N1, dtype=np.float32) for k in all_c1_report_keys}
    oof_c1_hes: Dict[str, np.ndarray] = {k: np.zeros(N1, dtype=np.float32) for k in all_c1_report_keys}
    oof_c1_conf: Dict[str, np.ndarray] = {k: np.zeros(N1, dtype=np.float32) for k in all_c1_report_keys}

    # Nested caches for Cohort 1 discordant calibration (both percentile and ecdf rank)
    c1_nested_cache: Dict[str, Dict[str, Any]] = {
        k: {
            "keep20_comp": np.zeros(N1, dtype=bool),
            "keep15_comp": np.zeros(N1, dtype=bool),
            "keep20_hes": np.zeros(N1, dtype=bool),
            "keep15_hes": np.zeros(N1, dtype=bool),
            "thr20_comp": [],
            "thr15_comp": [],
            "thr20_hes": [],
            "thr15_hes": [],
            "ecdf_comp": np.zeros(N1, dtype=np.float64),
            "ecdf_conf": np.zeros(N1, dtype=np.float64),
            "ecdf_hes": np.zeros(N1, dtype=np.float64),
        }
        for k in baseline_and_ours_c1
    }

    archived_sel_indices = master["track1_adni_cohort1"]["rough_summary"]["per_fold_selected_indices"]
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X1_raw[0], y1)):
        fids_c1[va_idx] = fold_idx
        fold_val_indices.append(va_idx)

        X_mri_tr_s, _, _, _ = fit_strict_preprocessor([X1_raw[0][tr_idx]], [X1_raw[0][va_idx]])
        reducer_f = FuzzyRoughFeatureReducer(
            max_features=30,
            sigma=0.85,
            max_samples=4096,
            n_null_permutations=8,
            n_stability_bootstraps=5,
            seed=42 + fold_idx,
        )
        reducer_f.fit(X_mri_tr_s[0], y1[tr_idx], feature_names=mri1_cols, device="cpu")
        sel_mri_idx = reducer_f.selected_indices_[:24]
        assert list(map(int, sel_mri_idx)) == archived_sel_indices[fold_idx], f"Fold {fold_idx} MRI idx drift"

        X_tr_raw = [X1_raw[0][tr_idx][:, sel_mri_idx], X1_raw[1][tr_idx], X1_raw[2][tr_idx], X1_raw[3][tr_idx]]
        X_va_raw = [X1_raw[0][va_idx][:, sel_mri_idx], X1_raw[1][va_idx], X1_raw[2][va_idx], X1_raw[3][va_idx]]
        X_tr_s, X_va_s, m_tr, m_va = fit_strict_preprocessor(X_tr_raw, X_va_raw)

        X_tr_unred = [X1_raw[0][tr_idx], X1_raw[1][tr_idx], X1_raw[2][tr_idx], X1_raw[3][tr_idx]]
        X_va_unred = [X1_raw[0][va_idx], X1_raw[1][va_idx], X1_raw[2][va_idx], X1_raw[3][va_idx]]
        X_tr_unred_s, X_va_unred_s, _, _ = fit_strict_preprocessor(X_tr_unred, X_va_unred)

        tr_tensors = [torch.tensor(x, dtype=torch.float32) for x in X_tr_s]
        va_tensors = [torch.tensor(x, dtype=torch.float32) for x in X_va_s]
        tr_tensors_unred = [torch.tensor(x, dtype=torch.float32) for x in X_tr_unred_s]
        va_tensors_unred = [torch.tensor(x, dtype=torch.float32) for x in X_va_unred_s]
        tr_mask_t = torch.tensor(m_tr, dtype=torch.float32)
        va_mask_t = torch.tensor(m_va, dtype=torch.float32)

        fold_val_tensors.append(va_tensors)
        fold_val_tensors_unred.append(va_tensors_unred)
        fold_val_masks.append(va_mask_t)

        # In-fold C1 discordant pool for nested quantile/ECDF calibration
        ptau1_med_f = float(np.nanmedian(c1_df.loc[tr_idx, "plasma_ptau181_bl"].values[has_pl1[tr_idx]]))
        hipp1_med_f = float(np.nanmedian(c1_df.loc[tr_idx, "Hippocampus_ICV_ratio"].values))
        pl1_pos_tr = c1_df.loc[tr_idx, "plasma_ptau181_bl"].values >= ptau1_med_f
        mri1_pos_tr = c1_df.loc[tr_idx, "Hippocampus_ICV_ratio"].values <= hipp1_med_f
        cl1_tr = clin1_pos[tr_idx]
        disc1_tr_in_fold = has_pl1[tr_idx] & (
            (pl1_pos_tr & (~mri1_pos_tr) & (~cl1_tr)) | ((~pl1_pos_tr) & mri1_pos_tr & cl1_tr)
        )
        cal_pool = disc1_tr_in_fold if int(np.sum(disc1_tr_in_fold)) >= 5 else np.ones(len(tr_idx), dtype=bool)

        for m_key in all_neural_eval_keys:
            ab_flags = ablation_configs.get(m_key, {})
            X_tr_use = X_tr_unred_s if ab_flags.get("unreduced_mri", False) else X_tr_s
            tr_t_use = tr_tensors_unred if ab_flags.get("unreduced_mri", False) else tr_tensors
            va_t_use = va_tensors_unred if ab_flags.get("unreduced_mri", False) else va_tensors

            mod = train_deepfuzzy_or_baseline(
                model_name=m_key,
                X_tr_list=X_tr_use,
                mask_tr=m_tr,
                y_tr=y1[tr_idx],
                epochs=28,
                lr=1.8e-3,
                embed_dim=40,
                concept_names=concept_names_8,
                ablation_flags=ab_flags,
                seed=42 + fold_idx,
                device=cpu_dev,
            )
            fold_models_c1[m_key].append(mod)
            with torch.no_grad():
                if isinstance(mod, UnifiedDeepFuzzyModel):
                    pm_tr = torch.ones_like(tr_mask_t) if ab_flags.get("disable_vacuity_mask", False) else tr_mask_t
                    pm_va = torch.ones_like(va_mask_t) if ab_flags.get("disable_vacuity_mask", False) else va_mask_t
                    c_tr = tr_t_use[-1] if mod.use_tsk_head else None
                    c_va = va_t_use[-1] if mod.use_tsk_head else None
                    out_tr = mod(tr_t_use, modality_masks=pm_tr, clinical_concepts=c_tr)
                    out_va = mod(va_t_use, modality_masks=pm_va, clinical_concepts=c_va)
                else:
                    out_tr = mod(tr_t_use, modality_masks=tr_mask_t)
                    out_va = mod(va_t_use, modality_masks=va_mask_t)

            oof_c1_probs[m_key][va_idx] = out_va["prob"].view(-1).numpy()
            hes_va = out_va["hesitancy"].view(-1).numpy()
            conf_va = out_va["conflict"].view(-1).numpy()
            oof_c1_hes[m_key][va_idx] = hes_va
            oof_c1_conf[m_key][va_idx] = conf_va

            if m_key in c1_nested_cache:
                hes_tr = out_tr["hesitancy"].view(-1).numpy()
                conf_tr = out_tr["conflict"].view(-1).numpy()
                comp_tr = 0.5 * hes_tr + 0.5 * conf_tr
                comp_va = 0.5 * hes_va + 0.5 * conf_va
                t20_c = float(np.percentile(comp_tr[cal_pool], 80.0))
                t15_c = float(np.percentile(comp_tr[cal_pool], 85.0))
                t20_h = float(np.percentile(hes_tr[cal_pool], 80.0))
                t15_h = float(np.percentile(hes_tr[cal_pool], 85.0))
                nc = c1_nested_cache[m_key]
                nc["thr20_comp"].append(round(t20_c, 5))
                nc["thr15_comp"].append(round(t15_c, 5))
                nc["thr20_hes"].append(round(t20_h, 5))
                nc["thr15_hes"].append(round(t15_h, 5))
                nc["keep20_comp"][va_idx] = comp_va <= t20_c
                nc["keep15_comp"][va_idx] = comp_va <= t15_c
                nc["keep20_hes"][va_idx] = hes_va <= t20_h
                nc["keep15_hes"][va_idx] = hes_va <= t15_h
                nc["ecdf_comp"][va_idx] = _nested_ecdf_percentile(comp_tr[cal_pool], comp_va)
                nc["ecdf_conf"][va_idx] = _nested_ecdf_percentile(conf_tr[cal_pool], conf_va)
                nc["ecdf_hes"][va_idx] = _nested_ecdf_percentile(hes_tr[cal_pool], hes_va)

        for t_key in tabular_names:
            clf = _fit_tabular_baseline(t_key, X_tr_s, m_tr, y1[tr_idx], seed=42 + fold_idx, tabpfn_device=tabpfn_dev)
            fold_models_c1[t_key].append(clf)
            p_tr, h_tr, c_tr = _predict_tabular(clf, X_tr_s, m_tr)
            p_va, h_va, c_va = _predict_tabular(clf, X_va_s, m_va)
            oof_c1_probs[t_key][va_idx] = p_va
            oof_c1_hes[t_key][va_idx] = h_va
            oof_c1_conf[t_key][va_idx] = c_va

            comp_tr = 0.5 * h_tr + 0.5 * c_tr
            comp_va = 0.5 * h_va + 0.5 * c_va
            t20_c = float(np.percentile(comp_tr[cal_pool], 80.0))
            t15_c = float(np.percentile(comp_tr[cal_pool], 85.0))
            t20_h = float(np.percentile(h_tr[cal_pool], 80.0))
            t15_h = float(np.percentile(h_tr[cal_pool], 85.0))
            nc = c1_nested_cache[t_key]
            nc["thr20_comp"].append(round(t20_c, 5))
            nc["thr15_comp"].append(round(t15_c, 5))
            nc["thr20_hes"].append(round(t20_h, 5))
            nc["thr15_hes"].append(round(t15_h, 5))
            nc["keep20_comp"][va_idx] = comp_va <= t20_c
            nc["keep15_comp"][va_idx] = comp_va <= t15_c
            nc["keep20_hes"][va_idx] = h_va <= t20_h
            nc["keep15_hes"][va_idx] = h_va <= t15_h
            nc["ecdf_comp"][va_idx] = _nested_ecdf_percentile(comp_tr[cal_pool], comp_va)
            nc["ecdf_conf"][va_idx] = _nested_ecdf_percentile(c_tr[cal_pool], c_va)
            nc["ecdf_hes"][va_idx] = _nested_ecdf_percentile(h_tr[cal_pool], h_va)

        print(
            f"  [C1 Fold {fold_idx+1}/5] DDEF={compute_binary_metrics(oof_c1_probs['DDEF'][va_idx], y1[va_idx])['auroc']:.4f} | "
            f"FUML={compute_binary_metrics(oof_c1_probs['FUML'][va_idx], y1[va_idx])['auroc']:.4f} | "
            f"XGB={compute_binary_metrics(oof_c1_probs['XGBoost'][va_idx], y1[va_idx])['auroc']:.4f} | "
            f"Cat={compute_binary_metrics(oof_c1_probs['CatBoost'][va_idx], y1[va_idx])['auroc']:.4f} | "
            f"TabPFN={compute_binary_metrics(oof_c1_probs['TabPFN'][va_idx], y1[va_idx])['auroc']:.4f}",
            flush=True,
        )

    # Assert deterministic replay (within tiny float32 BLAS/OpenMP reduction ULP) & restore archived NPZ arrays
    assert np.array_equal(fids_c1, npz_data["adni1_fold_ids"].astype(np.int32)), "C1 fold_id mismatch"
    for m_key in existing_model_names + list(ablation_configs.keys()):
        if f"adni1_prob_{m_key}" in npz_data:
            p_arch = npz_data[f"adni1_prob_{m_key}"].astype(np.float32)
            diff_p = float(np.max(np.abs(oof_c1_probs[m_key] - p_arch)))
            au_new = compute_binary_metrics(oof_c1_probs[m_key], y1)["auroc"]
            au_old = compute_binary_metrics(p_arch, y1)["auroc"]
            assert diff_p < 5e-3 and abs(au_new - au_old) < 5e-4, (
                f"C1 replay mismatch on {m_key}: diff_p={diff_p}, au_diff={abs(au_new - au_old)}"
            )
            oof_c1_probs[m_key] = p_arch.copy()
            if f"adni1_hes_{m_key}" in npz_data:
                oof_c1_hes[m_key] = npz_data[f"adni1_hes_{m_key}"].astype(np.float32).copy()
            if f"adni1_conf_{m_key}" in npz_data:
                oof_c1_conf[m_key] = npz_data[f"adni1_conf_{m_key}"].astype(np.float32).copy()

    # Persist new C1 OOF arrays into NPZ
    for m_key in new_neural_names + tabular_names:
        npz_data[f"adni1_prob_{m_key}"] = oof_c1_probs[m_key].astype(np.float32)
        npz_data[f"adni1_hes_{m_key}"] = oof_c1_hes[m_key].astype(np.float32)
        npz_data[f"adni1_conf_{m_key}"] = oof_c1_conf[m_key].astype(np.float32)

    # Assemble full OOF tensors & natural masks
    full_val_tensors = [torch.zeros(N1, fold_val_tensors[0][m].shape[1]) for m in range(4)]
    full_val_tensors_unred = [torch.zeros(N1, fold_val_tensors_unred[0][m].shape[1]) for m in range(4)]
    full_val_masks = torch.zeros(N1, 4)
    for f_i, va_idx in enumerate(fold_val_indices):
        for m in range(4):
            full_val_tensors[m][va_idx] = fold_val_tensors[f_i][m]
            full_val_tensors_unred[m][va_idx] = fold_val_tensors_unred[f_i][m]
        full_val_masks[va_idx] = fold_val_masks[f_i]

    full_val_masks_np = full_val_masks.numpy()
    complete_4mod_mask = np.all(full_val_masks_np > 0.5, axis=1)
    n_complete_4mod = int(np.sum(complete_4mod_mask))
    assert n_complete_4mod == 854, f"Expected N_comp=854 complete 4-modality subjects, got {n_complete_4mod}"

    def make_c1_oof_predict_fn(m_key: str) -> Callable[[List[torch.Tensor], torch.Tensor], Dict[str, torch.Tensor]]:
        ab_flags = ablation_configs.get(m_key, {})

        def _pred(mod_tensors_full: List[torch.Tensor], mask_full: torch.Tensor) -> Dict[str, torch.Tensor]:
            probs_full = torch.zeros(N1, 1)
            hes_full = torch.zeros(N1, 1)
            conf_full = torch.zeros(N1, 1)
            m_full_np = mask_full.detach().cpu().numpy()
            for f_i, va_idx in enumerate(fold_val_indices):
                mod_f = fold_models_c1[m_key][f_i]
                if m_key in tabular_names:
                    sub_xs_np = [t[va_idx].detach().cpu().numpy() for t in mod_tensors_full]
                    p_va, h_va, c_va = _predict_tabular(mod_f, sub_xs_np, m_full_np[va_idx])
                    probs_full[va_idx] = torch.from_numpy(p_va).unsqueeze(-1)
                    hes_full[va_idx] = torch.from_numpy(h_va).unsqueeze(-1)
                    conf_full[va_idx] = torch.from_numpy(c_va).unsqueeze(-1)
                else:
                    sub_xs = [t[va_idx] for t in mod_tensors_full]
                    sub_m = mask_full[va_idx]
                    if isinstance(mod_f, UnifiedDeepFuzzyModel):
                        pass_m = torch.ones_like(sub_m) if ab_flags.get("disable_vacuity_mask", False) else sub_m
                        c_in = sub_xs[-1] if mod_f.use_tsk_head else None
                        out_f = mod_f(sub_xs, modality_masks=pass_m, clinical_concepts=c_in)
                    else:
                        out_f = mod_f(sub_xs, modality_masks=sub_m)
                    probs_full[va_idx] = out_f["prob"].detach().cpu()
                    hes_full[va_idx] = out_f["hesitancy"].detach().cpu()
                    conf_full[va_idx] = out_f["conflict"].detach().cpu()
            return {"prob": probs_full, "hesitancy": hes_full, "conflict": conf_full}

        return _pred

    # Build/update Cohort 1 tables (preserving exact values for existing models while recomputing BH FDR & new tables)
    t1 = master["track1_adni_cohort1"]
    t1["n_complete_4mod"] = n_complete_4mod
    t1["preprocessing_audit"] = {
        "n_total": int(N1),
        "n_same_visit_bl_ref": int(np.sum(same_visit_bl_924_mask)),
        "n_complete_4mod": n_complete_4mod,
        "n_adas13_missing_median_imputed_in_clin": n_adas13_missing,
        "adas13_imputation_note": (
            f"{n_adas13_missing}/{N1} ADAS13 entries in ADNI Cohort 1 are missing and median-imputed "
            "strictly within training folds by fit_strict_preprocessor_with_params; Clinical modality-level "
            "mask remains 1.0 when >=50% of 17 clinical indicators are observed, while optional "
            "clinical_concept_masks enables sub-indicator masking when supplied explicitly."
        ),
    }

    main_table = t1["main_table"]
    same_visit_table = t1["same_visit_bl_924_table"]
    comb_bench = CombinatorialMissingModalityBenchmark(mod_names)
    combinatorial_results = t1["combinatorial_15subsets"]
    missing_drop_curves = t1["missing_drop_curves"]
    dca_data = t1["dca_data"]
    calib_data = t1["calib_data"]

    for m_key in new_neural_names + tabular_names:
        m_res = compute_binary_metrics(oof_c1_probs[m_key], y1, uncertainty=oof_c1_hes[m_key])
        _, au_lo, au_hi = bootstrap_confidence_interval(oof_c1_probs[m_key], y1, "auroc", n_bootstraps=1000, seed=42)
        _, br_lo, br_hi = bootstrap_confidence_interval(oof_c1_probs[m_key], y1, "brier", n_bootstraps=1000, seed=42)
        m_res["auroc_ci"] = [float(au_lo), float(au_hi)]
        m_res["brier_ci"] = [float(br_lo), float(br_hi)]
        m_res["p_value_vs_ours"] = float(
            paired_bootstrap_pvalue(oof_c1_probs["DeepFuzzyTorch"], oof_c1_probs[m_key], y1, "auroc", n_bootstraps=1000, seed=42)
        )
        m_res["is_tabular_single_table"] = m_key in tabular_names

        pred_fn = make_c1_oof_predict_fn(m_key)
        comb_res = comb_bench.evaluate_all_subsets(
            predict_fn=pred_fn,
            modality_tensors=full_val_tensors,
            natural_masks=full_val_masks,
            targets=y1,
        )
        combinatorial_results[m_key] = comb_res
        m_res["macro_15subset_auroc"] = float(comb_res["macro_avg_auroc"])
        m_res["macro_15subset_auroc_eligible"] = float(comb_res["macro_avg_auroc_eligible"])
        m_res["mrr_14incomplete"] = float(comb_res["missing_robustness_retention"])
        m_res["mrr_14incomplete_eligible"] = float(comb_res["missing_robustness_retention_eligible"])
        main_table[m_key] = m_res

        missing_drop_curves[m_key] = comb_bench.evaluate_missing_rate_curve(
            predict_fn=pred_fn,
            modality_tensors=full_val_tensors,
            natural_masks=full_val_masks,
            targets=y1,
        )
        dca_data[m_key] = compute_decision_curve_analysis(oof_c1_probs[m_key], y1)
        calib_data[m_key] = _compute_calib_entry(oof_c1_probs[m_key], y1, m_res["ece"], m_res["brier"])

        # Same-visit n=924 subset
        y_924 = y1[same_visit_bl_924_mask]
        p_924 = oof_c1_probs[m_key][same_visit_bl_924_mask]
        h_924 = oof_c1_hes[m_key][same_visit_bl_924_mask]
        res_924 = compute_binary_metrics(p_924, y_924, uncertainty=h_924)
        _, lo924, hi924 = bootstrap_confidence_interval(p_924, y_924, "auroc", n_bootstraps=1000, seed=42)
        res_924["auroc_ci"] = [float(lo924), float(hi924)]
        res_924["p_value_vs_ours"] = float(
            paired_bootstrap_pvalue(oof_c1_probs["DeepFuzzyTorch"][same_visit_bl_924_mask], p_924, y_924, "auroc", n_bootstraps=1000, seed=42)
        )
        same_visit_table[m_key] = res_924

    attach_bh_fdr_to_table(main_table)
    attach_bh_fdr_to_table(same_visit_table)

    # Complete 4-modality subset table (N_comp=854) across all C1 models + ablations
    complete_4mod_854_table = _eval_subset_metrics_table(
        oof_c1_probs, oof_c1_hes, y1, subset_mask=complete_4mod_mask, model_order=all_c1_report_keys
    )
    t1["complete_4mod_854_table"] = complete_4mod_854_table
    for k in all_c1_report_keys:
        if k in main_table and k in complete_4mod_854_table:
            main_table[k]["complete_4mod_854_auroc"] = complete_4mod_854_table[k]["auroc"]

    assert abs(main_table["DeepFuzzyTorch"]["auroc"] - 0.826973) < 5e-4
    assert abs(complete_4mod_854_table["DeepFuzzyTorch"]["auroc"] - 0.837143) < 5e-4
    print(
        f"[Audit C1] DeepFuzzyTorch Full(N=1137)={main_table['DeepFuzzyTorch']['auroc']:.4f} | "
        f"SameVisit(n=924)={same_visit_table['DeepFuzzyTorch']['auroc']:.4f} | "
        f"Complete4Mod(N_comp=854)={complete_4mod_854_table['DeepFuzzyTorch']['auroc']:.4f}"
    )

    # Re-run separated conflict sweep across ALL Cohort 1 models & ablations
    print("[Audit C1] Running separated conflict sweep (composite / pure C_O / pure pi_fused)...")
    injector = CrossModalityConflictInjector(4, mod_names)
    conflict_results: Dict[str, Any] = {}
    for m_key in all_c1_report_keys:
        ab_flags = ablation_configs.get(m_key, {})
        t_input = full_val_tensors_unred if ab_flags.get("unreduced_mri", False) else full_val_tensors
        c_res = injector.evaluate_conflict_sweep(
            predict_fn=make_c1_oof_predict_fn(m_key),
            modality_tensors=t_input,
            natural_masks=full_val_masks,
            targets=y1,
            corrupted_modality_idx=1,
            fold_ids=fids_c1,
        )
        c_res["is_tabular_single_table"] = m_key in tabular_names
        conflict_results[m_key] = c_res
        rec_08 = [r for r in c_res["sweep_records"] if abs(r["severity_gamma"] - 0.8) < 1e-4][0]
        main_table[m_key]["conflict_det_auroc_g08"] = float(rec_08["conflict_detection_auroc"])
        main_table[m_key]["conflict_det_auroc_conflict_only_g08"] = (
            None if m_key in tabular_names else float(rec_08["conflict_detection_auroc_conflict_only"])
        )
        main_table[m_key]["conflict_det_auroc_hes_only_g08"] = float(rec_08["conflict_detection_auroc_hes_only"])
        main_table[m_key]["referral_20pct_auroc_g08"] = float(rec_08["referral_curve"]["refer_20pct"]["auroc"])
        main_table[m_key]["referral_20pct_auroc_conflict_only_g08"] = (
            None if m_key in tabular_names else float(rec_08["referral_curve_conflict_only"]["refer_20pct"]["auroc"])
        )
        main_table[m_key]["referral_20pct_auroc_hes_only_g08"] = float(
            rec_08["referral_curve_hes_only"]["refer_20pct"]["auroc"]
        )

    assert abs(main_table["DeepFuzzyTorch"]["conflict_det_auroc_g08"] - 0.654173) < 2e-4
    print(
        f"[Conflict g=0.8 DeepFuzzyTorch] Composite={main_table['DeepFuzzyTorch']['conflict_det_auroc_g08']:.4f} | "
        f"Pure C_O={main_table['DeepFuzzyTorch']['conflict_det_auroc_conflict_only_g08']:.4f} | "
        f"Pure pi_fused={main_table['DeepFuzzyTorch']['conflict_det_auroc_hes_only_g08']:.4f}"
    )
    t1["conflict_results"] = conflict_results

    # Matched-coverage selective referral on Full ADNI Cohort 1 (N=1,137)
    c1_coverages = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60]
    c1_keep_counts = [(f"cov_{int(round(c * 100))}pct", int(round(N1 * c))) for c in c1_coverages]
    comp_oof_c1 = {m: 0.5 * oof_c1_hes[m] + 0.5 * oof_c1_conf[m] for m in baseline_and_ours_c1}
    tie_oof_c1 = {m: 1.0 - 2.0 * np.abs(oof_c1_probs[m] - 0.5) for m in baseline_and_ours_c1}
    t1["matched_coverage_referral_c1"] = {
        "composite_oof": _compute_matched_coverage_grid(
            baseline_and_ours_c1, oof_c1_probs, comp_oof_c1, oof_c1_hes, y1, c1_keep_counts
        ),
        "hes_only_oof": _compute_matched_coverage_grid(
            baseline_and_ours_c1, oof_c1_probs, oof_c1_hes, tie_oof_c1, y1, c1_keep_counts
        ),
    }

    # =========================================================================
    # 2. ADNI COHORT 2 REPLAY + NEW BASELINES + DEDUP MATCHED-COVERAGE (n=198)
    # =========================================================================
    c2_full = pd.read_csv(os.path.join(adni_dir, "cohort2_adni_ptau217_gfap.csv")).copy()
    c2_full = c2_full.assign(PTGENDER_num=(c2_full["PTGENDER"] == "Male").astype(float)).copy()
    sub2 = c2_full[c2_full["amyloid_gold_pos"].notna()].copy().reset_index(drop=True)
    y2 = sub2["amyloid_gold_pos"].values.astype(np.float32)
    N2 = len(y2)
    fids_c2 = npz_data["adni2_fold_ids"].astype(np.int32)

    strict_90d_mask = sub2["c2_blood_within_90d_bl"].fillna(0).values == 1
    bl_amyloid_mask2 = sub2["is_same_visit_baseline_amyloid_reference"].fillna(0).values == 1
    strict_176_mask = strict_90d_mask & bl_amyloid_mask2

    macro_cols2 = [
        "Ventricles_ICV_ratio", "Hippocampus_ICV_ratio", "WholeBrain_ICV_ratio",
        "Entorhinal_ICV_ratio", "Fusiform_ICV_ratio", "MidTemp_ICV_ratio",
    ]
    mri2_cols_all = macro_cols2 + sorted([c for c in sub2.columns if c.startswith("FSX_ST")])
    plasma2_cols = ["pT217_F", "log_pT217_F", "GFAP_Q", "log_GFAP_Q", "ptau217_gfap_ratio"]
    fdg2_cols = ["FDG_suvr"]
    clin2_cols = [
        "APOE4", "MMSE", "ADAS13", "CDRSB", "LDELTOTAL",
        "RAVLT_immediate", "FAQ", "AGE", "PTGENDER_num",
    ]
    X2_mri_full = sub2[mri2_cols_all].values.astype(np.float32)
    X2_plasma = sub2[plasma2_cols].values.astype(np.float32)
    X2_fdg = sub2[fdg2_cols].values.astype(np.float32)
    X2_fdg[(sub2["FDG_is_bl_visit"].fillna(0).values != 1), :] = np.nan
    X2_clin = sub2[clin2_cols].values.astype(np.float32)
    clin2_pos = ((sub2["DX_bl"] != "CN") | (sub2["CDRSB"] >= 0.5)).values

    c2_existing_models = ["DeepFuzzyTorch", "RCML", "QMF", "DrFuse", "EvidentialDL", "GatedAttention", "EarlyConcat"]
    c2_all_models = c2_existing_models + new_neural_names + tabular_names

    oof_c2_probs: Dict[str, np.ndarray] = {k: np.zeros(N2, dtype=np.float32) for k in c2_all_models}
    oof_c2_hes: Dict[str, np.ndarray] = {k: np.zeros(N2, dtype=np.float32) for k in c2_all_models}
    oof_c2_conf: Dict[str, np.ndarray] = {k: np.zeros(N2, dtype=np.float32) for k in c2_all_models}
    c2_nested_cache: Dict[str, Dict[str, Any]] = {
        k: {
            "keep20_comp": np.zeros(N2, dtype=bool),
            "keep15_comp": np.zeros(N2, dtype=bool),
            "keep20_hes": np.zeros(N2, dtype=bool),
            "keep15_hes": np.zeros(N2, dtype=bool),
            "thr20_comp": [],
            "thr15_comp": [],
            "thr20_hes": [],
            "thr15_hes": [],
            "ecdf_comp": np.zeros(N2, dtype=np.float64),
            "ecdf_conf": np.zeros(N2, dtype=np.float64),
            "ecdf_hes": np.zeros(N2, dtype=np.float64),
        }
        for k in c2_all_models
    }

    for fold_idx, (tr_i, va_i) in enumerate(skf.split(X2_mri_full, y2)):
        assert np.array_equal(va_i, np.where(fids_c2 == fold_idx)[0]), f"C2 fold {fold_idx} mismatch"
        X_mri_tr_all_s, _, _, _ = fit_strict_preprocessor([X2_mri_full[tr_i]], [X2_mri_full[va_i]])
        red_c2 = FuzzyRoughFeatureReducer(
            max_features=24, sigma=0.85, max_samples=4096,
            n_null_permutations=8, n_stability_bootstraps=5, seed=100 + fold_idx,
        )
        red_c2.fit(X_mri_tr_all_s[0], y2[tr_i], feature_names=mri2_cols_all, device="cpu")
        sel_c2 = red_c2.selected_indices_[:24]

        X_tr_raw = [X2_mri_full[tr_i][:, sel_c2], X2_plasma[tr_i], X2_fdg[tr_i], X2_clin[tr_i]]
        X_va_raw = [X2_mri_full[va_i][:, sel_c2], X2_plasma[va_i], X2_fdg[va_i], X2_clin[va_i]]
        X_tr_s, X_va_s, m_tr, m_va = fit_strict_preprocessor(X_tr_raw, X_va_raw)
        tr_t = [torch.tensor(x, dtype=torch.float32) for x in X_tr_s]
        tr_m = torch.tensor(m_tr, dtype=torch.float32)
        va_t = [torch.tensor(x, dtype=torch.float32) for x in X_va_s]
        va_m = torch.tensor(m_va, dtype=torch.float32)

        ptau2_med_f = float(np.nanmedian(sub2.loc[tr_i, "pT217_F"].values))
        hipp2_med_f = float(np.nanmedian(sub2.loc[tr_i, "Hippocampus_ICV_ratio"].values))
        pl2_pos_tr = sub2.loc[tr_i, "pT217_F"].values >= ptau2_med_f
        mri2_pos_tr = sub2.loc[tr_i, "Hippocampus_ICV_ratio"].values <= hipp2_med_f
        cl2_tr = clin2_pos[tr_i]
        disc2_tr_in_fold = (pl2_pos_tr & (~mri2_pos_tr) & (~cl2_tr)) | ((~pl2_pos_tr) & mri2_pos_tr & cl2_tr)
        cal_pool2 = disc2_tr_in_fold if int(np.sum(disc2_tr_in_fold)) >= 5 else np.ones(len(tr_i), dtype=bool)

        for m_name in c2_existing_models + new_neural_names:
            mod = train_deepfuzzy_or_baseline(
                m_name, X_tr_s, m_tr, y2[tr_i],
                epochs=50, lr=2.5e-3, embed_dim=32, seed=100 + fold_idx, device=cpu_dev,
            )
            with torch.no_grad():
                out_tr = mod(tr_t, modality_masks=tr_m)
                out_va = mod(va_t, modality_masks=va_m)
            hes_tr = out_tr["hesitancy"].view(-1).numpy()
            conf_tr = out_tr["conflict"].view(-1).numpy()
            comp_tr = 0.5 * hes_tr + 0.5 * conf_tr

            prob_va = out_va["prob"].view(-1).numpy()
            hes_va = out_va["hesitancy"].view(-1).numpy()
            conf_va = out_va["conflict"].view(-1).numpy()
            comp_va = 0.5 * hes_va + 0.5 * conf_va

            oof_c2_probs[m_name][va_i] = prob_va
            oof_c2_hes[m_name][va_i] = hes_va
            oof_c2_conf[m_name][va_i] = conf_va

            t20_c = float(np.percentile(comp_tr[cal_pool2], 80.0))
            t15_c = float(np.percentile(comp_tr[cal_pool2], 85.0))
            t20_h = float(np.percentile(hes_tr[cal_pool2], 80.0))
            t15_h = float(np.percentile(hes_tr[cal_pool2], 85.0))
            nc2 = c2_nested_cache[m_name]
            nc2["thr20_comp"].append(round(t20_c, 5))
            nc2["thr15_comp"].append(round(t15_c, 5))
            nc2["thr20_hes"].append(round(t20_h, 5))
            nc2["thr15_hes"].append(round(t15_h, 5))
            nc2["keep20_comp"][va_i] = comp_va <= t20_c
            nc2["keep15_comp"][va_i] = comp_va <= t15_c
            nc2["keep20_hes"][va_i] = hes_va <= t20_h
            nc2["keep15_hes"][va_i] = hes_va <= t15_h
            nc2["ecdf_comp"][va_i] = _nested_ecdf_percentile(comp_tr[cal_pool2], comp_va)
            nc2["ecdf_conf"][va_i] = _nested_ecdf_percentile(conf_tr[cal_pool2], conf_va)
            nc2["ecdf_hes"][va_i] = _nested_ecdf_percentile(hes_tr[cal_pool2], hes_va)

        for t_name in tabular_names:
            clf = _fit_tabular_baseline(t_name, X_tr_s, m_tr, y2[tr_i], seed=100 + fold_idx, tabpfn_device=tabpfn_dev)
            p_tr, h_tr, c_tr = _predict_tabular(clf, X_tr_s, m_tr)
            p_va, h_va, c_va = _predict_tabular(clf, X_va_s, m_va)
            oof_c2_probs[t_name][va_i] = p_va
            oof_c2_hes[t_name][va_i] = h_va
            oof_c2_conf[t_name][va_i] = c_va

            comp_tr = 0.5 * h_tr + 0.5 * c_tr
            comp_va = 0.5 * h_va + 0.5 * c_va
            t20_c = float(np.percentile(comp_tr[cal_pool2], 80.0))
            t15_c = float(np.percentile(comp_tr[cal_pool2], 85.0))
            t20_h = float(np.percentile(h_tr[cal_pool2], 80.0))
            t15_h = float(np.percentile(h_tr[cal_pool2], 85.0))
            nc2 = c2_nested_cache[t_name]
            nc2["thr20_comp"].append(round(t20_c, 5))
            nc2["thr15_comp"].append(round(t15_c, 5))
            nc2["thr20_hes"].append(round(t20_h, 5))
            nc2["thr15_hes"].append(round(t15_h, 5))
            nc2["keep20_comp"][va_i] = comp_va <= t20_c
            nc2["keep15_comp"][va_i] = comp_va <= t15_c
            nc2["keep20_hes"][va_i] = h_va <= t20_h
            nc2["keep15_hes"][va_i] = h_va <= t15_h
            nc2["ecdf_comp"][va_i] = _nested_ecdf_percentile(comp_tr[cal_pool2], comp_va)
            nc2["ecdf_conf"][va_i] = _nested_ecdf_percentile(c_tr[cal_pool2], c_va)
            nc2["ecdf_hes"][va_i] = _nested_ecdf_percentile(h_tr[cal_pool2], h_va)

    for m_name in c2_existing_models:
        p2_arch = npz_data[f"adni2_prob_{m_name}"].astype(np.float32)
        diff2 = float(np.max(np.abs(oof_c2_probs[m_name] - p2_arch)))
        au2_new = compute_binary_metrics(oof_c2_probs[m_name], y2)["auroc"]
        au2_old = compute_binary_metrics(p2_arch, y2)["auroc"]
        assert diff2 < 5e-3 and abs(au2_new - au2_old) < 5e-4, (
            f"C2 replay mismatch on {m_name}: diff2={diff2}, au_diff={abs(au2_new - au2_old)}"
        )
        oof_c2_probs[m_name] = p2_arch.copy()
        if f"adni2_hes_{m_name}" in npz_data:
            oof_c2_hes[m_name] = npz_data[f"adni2_hes_{m_name}"].astype(np.float32).copy()
        if f"adni2_conf_{m_name}" in npz_data:
            oof_c2_conf[m_name] = npz_data[f"adni2_conf_{m_name}"].astype(np.float32).copy()
    for m_name in new_neural_names + tabular_names:
        npz_data[f"adni2_prob_{m_name}"] = oof_c2_probs[m_name].astype(np.float32)
        npz_data[f"adni2_hes_{m_name}"] = oof_c2_hes[m_name].astype(np.float32)
        npz_data[f"adni2_conf_{m_name}"] = oof_c2_conf[m_name].astype(np.float32)

    t2 = master["track2_adni_cohort2_ptau217"]
    for m_name in new_neural_names + tabular_names:
        res_c2 = compute_binary_metrics(oof_c2_probs[m_name], y2, uncertainty=oof_c2_hes[m_name])
        _, lo_m, hi_m = bootstrap_confidence_interval(oof_c2_probs[m_name], y2, "auroc", n_bootstraps=1000, seed=42)
        res_c2["auroc_ci"] = [float(lo_m), float(hi_m)]
        res_c2["p_value_vs_ours"] = float(
            paired_bootstrap_pvalue(oof_c2_probs["DeepFuzzyTorch"], oof_c2_probs[m_name], y2, "auroc", n_bootstraps=1000, seed=42)
        )
        t2["cohort2_metrics"][m_name] = res_c2

        res_90 = compute_binary_metrics(
            oof_c2_probs[m_name][strict_90d_mask], y2[strict_90d_mask], uncertainty=oof_c2_hes[m_name][strict_90d_mask]
        )
        _, lo90, hi90 = bootstrap_confidence_interval(oof_c2_probs[m_name][strict_90d_mask], y2[strict_90d_mask], "auroc", n_bootstraps=1000, seed=42)
        res_90["auroc_ci"] = [float(lo90), float(hi90)]
        res_90["p_value_vs_ours"] = float(
            paired_bootstrap_pvalue(oof_c2_probs["DeepFuzzyTorch"][strict_90d_mask], oof_c2_probs[m_name][strict_90d_mask], y2[strict_90d_mask], "auroc", n_bootstraps=1000, seed=42)
        )
        t2["cohort2_strict_90d_metrics"][m_name] = res_90

        res_176 = compute_binary_metrics(
            oof_c2_probs[m_name][strict_176_mask], y2[strict_176_mask], uncertainty=oof_c2_hes[m_name][strict_176_mask]
        )
        _, lo176, hi176 = bootstrap_confidence_interval(oof_c2_probs[m_name][strict_176_mask], y2[strict_176_mask], "auroc", n_bootstraps=1000, seed=42)
        res_176["auroc_ci"] = [float(lo176), float(hi176)]
        res_176["p_value_vs_ours"] = float(
            paired_bootstrap_pvalue(oof_c2_probs["DeepFuzzyTorch"][strict_176_mask], oof_c2_probs[m_name][strict_176_mask], y2[strict_176_mask], "auroc", n_bootstraps=1000, seed=42)
        )
        t2["cohort2_strict_176_concurrent_metrics"][m_name] = res_176

    attach_bh_fdr_to_table(t2["cohort2_metrics"])
    attach_bh_fdr_to_table(t2["cohort2_strict_90d_metrics"])
    attach_bh_fdr_to_table(t2["cohort2_strict_176_concurrent_metrics"])

    # Discordant masks (verify exact identity with NPZ masks)
    disc1_mask = npz_data["adni1_real_discordant_mask"].astype(bool)
    disc2_mask = npz_data["adni2_real_discordant_mask"].astype(bool)
    c2_unique_mask = npz_data["adni2_nonoverlap_c1plasma_mask"].astype(bool)
    uniq_on_disc2 = c2_unique_mask[disc2_mask]
    y_dedup_disc = np.concatenate([y1[disc1_mask], y2[disc2_mask][uniq_on_disc2]])
    assert len(y_dedup_disc) == 198, f"Expected n=198 deduplicated discordant patients, got {len(y_dedup_disc)}"

    rw = t2["real_world_discordance_audit"]
    for m_name in new_neural_names + tabular_names:
        rw["cohort1_discordant_triage"][m_name] = evaluate_nested_discordant_triage(
            oof_c1_probs[m_name], oof_c1_hes[m_name], oof_c1_conf[m_name], y1, disc1_mask, c1_nested_cache[m_name]
        )
        rw["cohort2_discordant_triage"][m_name] = evaluate_nested_discordant_triage(
            oof_c2_probs[m_name], oof_c2_hes[m_name], oof_c2_conf[m_name], y2, disc2_mask, c2_nested_cache[m_name]
        )
        rw["dedup_discordant_triage"][m_name] = evaluate_dedup_cross_cohort_triage_nested(
            oof_c1_probs[m_name], oof_c1_hes[m_name], oof_c1_conf[m_name], y1, disc1_mask, c1_nested_cache[m_name],
            oof_c2_probs[m_name], oof_c2_hes[m_name], oof_c2_conf[m_name], y2, disc2_mask, c2_unique_mask, c2_nested_cache[m_name],
        )

    # Matched-coverage selective referral curves on deduplicated n=198 discordant ADNI
    dd_keep_counts = [
        ("cov_100pct", 198),
        ("cov_95pct", 188),
        ("cov_90pct", 178),
        ("cov_85pct", 168),
        ("match_161_81p3pct", 161),
        ("cov_80pct", 158),
        ("match_152_76p8pct", 152),
        ("cov_75pct", 148),
        ("match_146_73p7pct", 146),
        ("match_141_71p2pct", 141),
        ("match_140_70p7pct", 140),
        ("cov_70pct", 139),
        ("cov_60pct", 119),
    ]
    dd_probs = {
        m: np.concatenate([oof_c1_probs[m][disc1_mask], oof_c2_probs[m][disc2_mask][uniq_on_disc2]])
        for m in c2_all_models
    }
    dd_ecdf_comp = {
        m: np.concatenate([c1_nested_cache[m]["ecdf_comp"][disc1_mask], c2_nested_cache[m]["ecdf_comp"][disc2_mask][uniq_on_disc2]])
        for m in c2_all_models
    }
    dd_ecdf_conf = {
        m: np.concatenate([c1_nested_cache[m]["ecdf_conf"][disc1_mask], c2_nested_cache[m]["ecdf_conf"][disc2_mask][uniq_on_disc2]])
        for m in c2_all_models
    }
    dd_ecdf_hes = {
        m: np.concatenate([c1_nested_cache[m]["ecdf_hes"][disc1_mask], c2_nested_cache[m]["ecdf_hes"][disc2_mask][uniq_on_disc2]])
        for m in c2_all_models
    }
    dd_raw_comp = {
        m: np.concatenate([
            0.5 * oof_c1_hes[m][disc1_mask] + 0.5 * oof_c1_conf[m][disc1_mask],
            0.5 * oof_c2_hes[m][disc2_mask][uniq_on_disc2] + 0.5 * oof_c2_conf[m][disc2_mask][uniq_on_disc2],
        ])
        for m in c2_all_models
    }
    dd_raw_conf = {
        m: np.concatenate([oof_c1_conf[m][disc1_mask], oof_c2_conf[m][disc2_mask][uniq_on_disc2]])
        for m in c2_all_models
    }
    dd_raw_hes = {
        m: np.concatenate([oof_c1_hes[m][disc1_mask], oof_c2_hes[m][disc2_mask][uniq_on_disc2]])
        for m in c2_all_models
    }

    dd_tie_ent = {m: 1.0 - 2.0 * np.abs(dd_probs[m] - 0.5) for m in c2_all_models}
    matched_cov_dd = {
        "description": (
            "Matched-coverage selective referral evaluation across identical retained sample counts "
            "and coverages on the deduplicated n=198 discordant ADNI cohort (N=1,112 unique participants), "
            "reported under both strictly nested outer-fold ECDF-calibrated uncertainty ranks (zero leakage "
            "and cross-cohort scale invariance) and raw OOF scores."
        ),
        "composite_nested_ecdf": _compute_matched_coverage_grid(
            c2_all_models, dd_probs, dd_ecdf_comp, dd_raw_comp, y_dedup_disc, dd_keep_counts
        ),
        "conflict_only_nested_ecdf": _compute_matched_coverage_grid(
            c2_all_models, dd_probs, dd_ecdf_conf, dd_raw_conf, y_dedup_disc, dd_keep_counts
        ),
        "hes_only_nested_ecdf": _compute_matched_coverage_grid(
            c2_all_models, dd_probs, dd_ecdf_hes, dd_raw_hes, y_dedup_disc, dd_keep_counts
        ),
        "composite_raw_oof": _compute_matched_coverage_grid(
            c2_all_models, dd_probs, dd_raw_comp, dd_tie_ent, y_dedup_disc, dd_keep_counts
        ),
        "hes_only_raw_oof": _compute_matched_coverage_grid(
            c2_all_models, dd_probs, dd_raw_hes, dd_tie_ent, y_dedup_disc, dd_keep_counts
        ),
    }
    # Add AURC paired bootstrap p-values vs DeepFuzzyTorch for composite_nested_ecdf
    for m in c2_all_models:
        p_aurc = 1.0 if m == "DeepFuzzyTorch" else _paired_aurc_bootstrap_p(
            dd_probs["DeepFuzzyTorch"], dd_ecdf_comp["DeepFuzzyTorch"],
            dd_probs[m], dd_ecdf_comp[m], y_dedup_disc, n_boot=1000, seed=42,
        )
        matched_cov_dd["composite_nested_ecdf"]["models"][m]["aurc_p_vs_ours"] = p_aurc

    rw["matched_coverage_referral"] = matched_cov_dd
    t2["ckd_subgroup"]["real_world_discordance_audit"] = rw

    # Print key matched-coverage summary for inspection
    mc_comp = matched_cov_dd["composite_nested_ecdf"]["models"]
    print("\n[Matched-Coverage Dedup Discordant Referral n=198 (Nested ECDF Composite)]")
    for m in c2_all_models:
        pts = mc_comp[m]["points"]
        print(
            f"  {m:16s} | AURC={mc_comp[m]['aurc']:.4f} | "
            f"100%(198)={pts['cov_100pct']['acc']:.4f} | "
            f"81.3%(161)={pts['match_161_81p3pct']['acc']:.4f} | "
            f"80%(158)={pts['cov_80pct']['acc']:.4f} | "
            f"70.7%(140)={pts['match_140_70p7pct']['acc']:.4f} | "
            f"60%(119)={pts['cov_60pct']['acc']:.4f}"
        )

    # Save updated NPZ & JSON
    for k_npz in list(npz_data.keys()):
        if getattr(npz_data[k_npz], "dtype", None) == object:
            npz_data[k_npz] = np.asarray(npz_data[k_npz].tolist(), dtype=str)
    np.savez_compressed(npz_path, **npz_data)
    with open(json_path, "w", encoding="utf-8") as f_out:
        json.dump(_to_json_safe(master), f_out, indent=2)

    elapsed = time.time() - t_start
    print(f"\n[Phase-1 Upgrades] Completed in {elapsed:.1f}s -> saved {json_path} & {npz_path}")


if __name__ == "__main__":
    main()
