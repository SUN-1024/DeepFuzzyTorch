"""
Master Multi-Dataset Benchmark & Theorem Verification Suite for DeepFuzzyTorch.

Executes 6 rigorous experimental tracks on real multi-center public cohorts:
  - Track 0: Analytical & Numerical Verification of Theorems 1-5 on target device (N=250, gamma_c=0.85).
  - Track 1: ADNI Cohort 1 (N=1,137, 58 sites, 4 baseline modalities: 320-ROI MRI, Baseline Plasma pTau181,
             Baseline FDG-PET, Clinical/Genetics; n=924 same-visit baseline reference) 5-Fold Out-of-Fold CV
             + In-Fold Fuzzy-Rough Reduct Curve + 2^4-1 (15) Combinatorial Missingness + Progressive Drop Curves
             + Intra-Fold Conflict Sweep + Ablations.
  - Track 2: ADNI Cohort 2 (N=289, Plasma pTau217 + GFAP + MRI + Baseline FDG-PET + Clinical; n=211 <=90d window;
             90 shared RIDs with Cohort 1) + Real-World Physiological Conflict Subgroup Analysis across 3 Renal
             Status Groups (Measured Normal eGFR >= 60 n=73, Measured CKD eGFR < 60 n=22, Unmeasured eGFR n=194).
  - Track 3: Track 3A — Dated ADNI Baseline Dementia-Free Longitudinal Incident Dementia Conversion (N=1,017,
             268 events; 60-month horizon N=716) with Mantel-Haenszel Log-Rank Test & Harrell's C-Index;
             Track 3B — Zero-Shot External Cross-Cohort Diagnostic Generalization (ADNI N=1,137 -> External
             OASIS-3 N=1,026 unique participants) with and without MMSE.
  - Track 4: TCGA Pan-Cancer Multi-Omics + Histopathology WSI Benchmark (N=2,946 unique patients across 7 TCGA
             cohorts: BRCA, COAD, KIRC, KIRP, LUAD, LUSC, UCEC) with In-Fold 48-D PCA on 3,072-D UNI2-h WSI
             Features + 2^3-1 (7) Missing-Modality Subsets + Overall Survival C-Index.
  - Track 5: Multi-Foundation-Model Hesitant Fuzzy Consensus & Cross-Hospital Pythagorean TTA
             on Camelyon17 (5 Hospitals, N=2,500 patches, 6 FMs) and NCT-CRC (NCT_CRC_HE_Discovery N=999 ->
             CRC_VAL_HE_7K_External N=1,350 tiles, 6 FMs) with Zero Test-Label Selection.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import beta as scipy_beta
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from deepfuzzy import (
    IntuitionisticFuzzyTensor,
    IntuitionisticFuzzyEncoder,
    IFSChoquetFusion,
    HesitantFuzzyAggregator,
    DifferentiableFuzzyRoughLayer,
    PythagoreanFuzzyTTALoss,
    IntuitionisticFuzzyLoss,
    UnifiedDeepFuzzyModel,
    CombinatorialMissingModalityBenchmark,
    CrossModalityConflictInjector,
    FuzzyRoughFeatureReducer,
    compute_binary_metrics,
    bootstrap_confidence_interval,
    paired_bootstrap_pvalue,
    compute_decision_curve_analysis,
    compute_logrank_test,
    compute_multigroup_logrank_test,
    DeepConceptTSKHead,
)
from deepfuzzy.baselines import (
    EarlyConcatMLP,
    LateAvgEnsemble,
    GatedAttentionFusion,
    DempsterShaferFusion,
    EvidentialDL,
    DeepTSKBaseline,
    QMFBaseline,
    RCMLBaseline,
    DrFuseBaseline,
    DDEFBaseline,
    FUMLBaseline,
)

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Official FreeSurfer ST code anatomical dictionary loaded directly from bundled JSON (R2-C7)
_FS_DICT_PATH = os.path.join(WORKSPACE_DIR, "deepfuzzy/freesurfer_roi_dictionary.json")
with open(_FS_DICT_PATH, "r", encoding="utf-8") as _f_fs:
    FREESURFER_ROI_DICT: Dict[str, Dict[str, str]] = json.load(_f_fs)

MACRO_ANATOMICAL_NAMES = {
    "Hippocampus_ICV_ratio": "Hippocampus / ICV Ratio",
    "Entorhinal_ICV_ratio": "Entorhinal Cortex / ICV Ratio",
    "Ventricles_ICV_ratio": "Lateral Ventricles / ICV Ratio",
    "MidTemp_ICV_ratio": "Middle Temporal / ICV Ratio",
    "Fusiform_ICV_ratio": "Fusiform Gyrus / ICV Ratio",
    "WholeBrain_ICV_ratio": "Whole Brain / ICV Ratio",
    "ICV": "Intracranial Volume (ICV)",
}


def format_freesurfer_roi_name(col: str) -> str:
    """
    Formats an MRI column name using the official ADNI UCSFFSX FreeSurfer dictionary
    (`deepfuzzy/freesurfer_roi_dictionary.json`) without hardcoded overrides.
    """
    if col in MACRO_ANATOMICAL_NAMES:
        return MACRO_ANATOMICAL_NAMES[col]
    entry = FREESURFER_ROI_DICT.get(col)
    if entry is None:
        return col
    if isinstance(entry, str):
        if " of " in entry:
            measure, region_camel = entry.split(" of ", 1)
        else:
            measure, region_camel = "", entry
        region = re.sub(r"(?<!^)(?=[A-Z])", " ", region_camel.strip())
    else:
        region = entry.get("region", col)
        measure = entry.get("measure", "")

    if region.startswith("Left "):
        reg_clean = "L " + region[len("Left ") :]
    elif region.startswith("Right "):
        reg_clean = "R " + region[len("Right ") :]
    elif region.startswith("ctx-lh-"):
        reg_clean = "L " + region[len("ctx-lh-") :].replace("-", " ").title()
    elif region.startswith("ctx-rh-"):
        reg_clean = "R " + region[len("ctx-rh-") :].replace("-", " ").title()
    else:
        reg_clean = region.replace("-", " ")

    meas_map = {
        "Cortical Thickness Average": "Thick",
        "Cortical Thickness": "Thick",
        "Volume (Cortical Parcellation)": "Vol",
        "Volume (WM Parcellation)": "Vol",
        "Subcortical/Cortical Volume": "Vol",
        "Surface Area": "Area",
        "Cortical Thickness Standard Deviation": "ThickSD",
        "Thickness StdDev": "ThickSD",
    }
    meas_short = meas_map.get(measure.strip(), measure.strip())
    st_code = col.replace("FSX_", "")
    return f"{reg_clean} {meas_short} ({st_code})".strip()


def set_all_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_paths(data_root: Optional[str] = None) -> Dict[str, str]:
    """
    Resolves dataset directories portably across local workspace and remote GPU servers.
    """
    if data_root is None:
        env_root = os.environ.get("DEEPFUZZY_DATA_ROOT")
        if env_root and os.path.exists(env_root):
            data_root = env_root
        elif os.path.exists(os.path.join(WORKSPACE_DIR, "data_benchmark")):
            data_root = os.path.join(WORKSPACE_DIR, "data_benchmark")
        else:
            data_root = os.path.dirname(WORKSPACE_DIR)

    # Check if flat data_benchmark layout or original sibling directories
    cand_adni_1 = os.path.join(data_root, "AD-capture-血液/data/processed")
    cand_adni_2 = os.path.join(data_root, "adni_processed")
    cand_adni_3 = os.path.join(data_root, "ad_processed")
    if os.path.exists(cand_adni_1):
        adni_dir = cand_adni_1
    elif os.path.exists(cand_adni_2):
        adni_dir = cand_adni_2
    else:
        adni_dir = cand_adni_3

    cand_oasis_dx_1 = os.path.join(data_root, "AD-capture-血液/data/OASIS3_diagnosis.csv")
    cand_oasis_dx_2 = os.path.join(adni_dir, "OASIS3_diagnosis.csv")
    oasis_dx_path = cand_oasis_dx_1 if os.path.exists(cand_oasis_dx_1) else cand_oasis_dx_2

    cand_wsi_1 = os.path.join(data_root, "WSI-fairness/data")
    cand_wsi_2 = os.path.join(data_root, "wsi_data")
    wsi_dir = cand_wsi_1 if os.path.exists(cand_wsi_1) else cand_wsi_2

    return {
        "adni_dir": adni_dir,
        "oasis_dx_path": oasis_dx_path,
        "wsi_dir": wsi_dir,
    }


def fit_strict_preprocessor_with_params(
    X_train_list: List[np.ndarray],
    X_val_list: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, List[Dict[str, np.ndarray]]]:
    """
    Strictly fits median imputation and z-score standardization ONLY on observed rows of
    X_train_list, applies the learned parameters to X_val_list, and returns the exact
    fitted per-modality (median, mean, std) parameter dictionaries for reproducibility replay.
    """
    train_norm_list, val_norm_list = [], []
    train_masks, val_masks = [], []
    params_list: List[Dict[str, np.ndarray]] = []

    for X_tr, X_va in zip(X_train_list, X_val_list):
        obs_tr = (~np.isnan(X_tr)).mean(axis=1) >= 0.5
        obs_va = (~np.isnan(X_va)).mean(axis=1) >= 0.5
        train_masks.append(obs_tr.astype(np.float32))
        val_masks.append(obs_va.astype(np.float32))

        # Fit median and z-score strictly on observed training samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if np.any(obs_tr):
                med = np.nanmedian(X_tr[obs_tr], axis=0, keepdims=True)
            else:
                med = np.nanmedian(X_tr, axis=0, keepdims=True)
        med = np.where(np.isnan(med), 0.0, med)

        X_tr_imp = np.where(np.isnan(X_tr), med, X_tr)
        X_va_imp = np.where(np.isnan(X_va), med, X_va)

        if np.any(obs_tr):
            mean = np.mean(X_tr_imp[obs_tr], axis=0, keepdims=True)
            std = np.std(X_tr_imp[obs_tr], axis=0, keepdims=True) + 1e-5
        else:
            mean = np.mean(X_tr_imp, axis=0, keepdims=True)
            std = np.std(X_tr_imp, axis=0, keepdims=True) + 1e-5

        X_tr_s = np.clip((X_tr_imp - mean) / std, -5.0, 5.0).astype(np.float32)
        X_va_s = np.clip((X_va_imp - mean) / std, -5.0, 5.0).astype(np.float32)

        # Zero out completely missing modality rows
        X_tr_s = X_tr_s * obs_tr[:, None]
        X_va_s = X_va_s * obs_va[:, None]

        train_norm_list.append(X_tr_s)
        val_norm_list.append(X_va_s)
        params_list.append(
            {
                "median": med.ravel().astype(np.float32),
                "mean": mean.ravel().astype(np.float32),
                "std": std.ravel().astype(np.float32),
            }
        )

    return (
        train_norm_list,
        val_norm_list,
        np.stack(train_masks, axis=1),
        np.stack(val_masks, axis=1),
        params_list,
    )


def apply_strict_preprocessor_params(
    X_list: List[np.ndarray],
    params_list: List[Dict[str, np.ndarray]],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Applies archived per-modality (median, mean, std) parameters identically to fit_strict_preprocessor."""
    norm_list = []
    masks = []
    for X_arr, p in zip(X_list, params_list):
        obs = (~np.isnan(X_arr)).mean(axis=1) >= 0.5
        masks.append(obs.astype(np.float32))
        med = p["median"][None, :]
        mean = p["mean"][None, :]
        std = p["std"][None, :]
        X_imp = np.where(np.isnan(X_arr), med, X_arr)
        X_s = np.clip((X_imp - mean) / std, -5.0, 5.0).astype(np.float32)
        X_s = X_s * obs[:, None]
        norm_list.append(X_s)
    return norm_list, np.stack(masks, axis=1)


def fit_strict_preprocessor(
    X_train_list: List[np.ndarray],
    X_val_list: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Strictly fits median imputation and z-score standardization ONLY on X_train_list
    and applies the learned parameters to X_val_list (zero data leakage).
    Also extracts the modality-level observation mask (1 if >=50% features in modality m
    are observed for sample i, else 0).
    """
    tr_s, va_s, m_tr, m_va, _ = fit_strict_preprocessor_with_params(X_train_list, X_val_list)
    return tr_s, va_s, m_tr, m_va


def compute_bh_fdr_map(
    pvals: Dict[str, float],
    exclude_keys: Tuple[str, ...] = ("DeepFuzzyTorch",),
) -> Dict[str, float]:
    """Benjamini-Hochberg step-up FDR adjustment across a family of hypothesis tests."""
    test_items = [
        (k, max(0.0, min(1.0, float(v))))
        for k, v in pvals.items()
        if k not in exclude_keys and np.isfinite(v)
    ]
    m = len(test_items)
    out: Dict[str, float] = {k: 1.0 for k in pvals}
    if m == 0:
        return out
    sorted_items = sorted(test_items, key=lambda kv: kv[1])
    q_vals = [0.0] * m
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        raw_q = min(1.0, sorted_items[i][1] * float(m) / float(rank))
        running_min = min(running_min, raw_q)
        q_vals[i] = running_min
    for (k, _), qv in zip(sorted_items, q_vals):
        out[k] = float(qv)
    return out


def attach_bh_fdr_to_table(
    table: Dict[str, Dict[str, object]],
    pval_key: str = "p_value_vs_ours",
    q_key: str = "q_bh_vs_ours",
    exclude_keys: Tuple[str, ...] = ("DeepFuzzyTorch",),
    p_key: Optional[str] = None,
    ref_key: Optional[str] = None,
) -> None:
    """Computes BH-FDR across `table[*][pval_key]` and writes `table[*][q_key]` in place."""
    eff_pkey = p_key if p_key is not None else pval_key
    eff_excl = tuple(set(exclude_keys) | ({ref_key} if ref_key is not None else set()))
    p_map = {
        k: float(v[eff_pkey])
        for k, v in table.items()
        if isinstance(v, dict) and eff_pkey in v
    }
    q_map = compute_bh_fdr_map(p_map, exclude_keys=eff_excl)
    for k, qv in q_map.items():
        if k in table and isinstance(table[k], dict):
            table[k][q_key] = round(float(qv), 6)


def pairwise_jaccard_summary(sets_list: List[object]) -> Dict[str, float]:
    """Computes pairwise Jaccard similarity statistics across CV fold selected feature sets."""
    as_sets = [set(s) for s in sets_list]
    vals: List[float] = []
    for i in range(len(as_sets)):
        for j in range(i + 1, len(as_sets)):
            union_sz = len(as_sets[i] | as_sets[j])
            vals.append(len(as_sets[i] & as_sets[j]) / float(max(1, union_sz)))
    if not vals:
        return {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}
    return {
        "mean": round(float(np.mean(vals)), 4),
        "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
        "min": round(float(np.min(vals)), 4),
        "max": round(float(np.max(vals)), 4),
    }


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> List[float]:
    """Exact two-sided Clopper-Pearson binomial confidence interval."""
    if n <= 0:
        return [0.0, 1.0]
    lo = 0.0 if k <= 0 else float(scipy_beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k >= n else float(scipy_beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def train_deepfuzzy_or_baseline(
    model_name: str,
    X_tr_list: List[np.ndarray],
    mask_tr: np.ndarray,
    y_tr: np.ndarray,
    epochs: int = 30,
    lr: float = 1.8e-3,
    embed_dim: int = 40,
    concept_names: Optional[List[str]] = None,
    ablation_flags: Optional[Dict[str, object]] = None,
    sample_weight: Optional[np.ndarray] = None,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Trains DeepFuzzyTorch, ablations, or multimodal / uncertainty SOTA baselines
    (including 2023-2024 QMF, RCML, DrFuse, EvidentialDL with Dirichlet KL regularizer,
    and Equalized-Supervision variants) under matched stochastic modality dropout.
    """
    set_all_seeds(seed)
    modality_dims = [x.shape[1] for x in X_tr_list]
    M = len(modality_dims)
    ablation_flags = ablation_flags or {}
    equalized_loss = bool(ablation_flags.get("equalized_loss", False))
    cold_no_aux = bool(ablation_flags.get("cold_no_aux", False))

    base_model_name = model_name.replace("_EqLoss", "")
    if (
        base_model_name == "DeepFuzzyTorch"
        or base_model_name == "DeepFuzzyTorch_ColdNoAux"
        or base_model_name.startswith("Ablation_")
    ):
        use_rough = bool(ablation_flags.get("use_rough_bottleneck", True))
        use_fgates = bool(ablation_flags.get("use_feature_gates", use_rough))
        use_tsk = bool(ablation_flags.get("use_tsk_head", True))
        conf_pen = float(ablation_flags.get("conflict_penalty", 0.55))
        disable_vac = bool(ablation_flags.get("disable_vacuity_mask", False))
        uni_choq = bool(ablation_flags.get("uniform_choquet", False))
        n_concepts = len(concept_names) if concept_names else min(8, M + X_tr_list[-1].shape[1])

        model = UnifiedDeepFuzzyModel(
            modality_dims=modality_dims,
            embed_dim=embed_dim,
            num_classes=1,
            prior_vacuity=0.45,
            conflict_penalty=conf_pen,
            use_rough_bottleneck=use_rough,
            use_feature_gates=use_fgates,
            use_tsk_head=use_tsk,
            num_tsk_concepts=n_concepts,
            num_tsk_rules=6,
            concept_names=concept_names,
            disable_vacuity_mask=disable_vac,
            uniform_choquet=uni_choq,
            dropout=0.18,
        ).to(device)

        if not cold_no_aux and base_model_name != "DeepFuzzyTorch_ColdNoAux":
            with torch.no_grad():
                y_centered = y_tr - np.mean(y_tr)
                mod_rel_scores = []
                for m_idx, x_arr in enumerate(X_tr_list):
                    obs_m = mask_tr[:, m_idx] > 0.5
                    if np.any(obs_m):
                        yc_m = y_tr[obs_m] - np.mean(y_tr[obs_m])
                        corr = np.abs((x_arr[obs_m] * yc_m[:, None]).mean(axis=0))
                    else:
                        corr = np.abs((x_arr * y_centered[:, None]).mean(axis=0))
                    top_k_corr = float(np.mean(np.sort(corr)[-min(5, len(corr)) :]))
                    mod_rel_scores.append(top_k_corr)
                    if use_fgates:
                        corr_norm = (corr - corr.mean()) / (corr.std() + 1e-5)
                        model.rough_gates[m_idx].copy_(
                            torch.tensor(corr_norm * 0.45, device=device, dtype=torch.float32)
                        )
                if not uni_choq:
                    rel_arr = np.asarray(mod_rel_scores, dtype=np.float32)
                    rel_z = (rel_arr - rel_arr.mean()) / (rel_arr.std() + 1e-5)
                    model.choquet_fusion.fuzzy_density_logits.copy_(
                        torch.tensor(rel_z[:, None] * 0.85, device=device, dtype=torch.float32)
                    )
        criterion = IntuitionisticFuzzyLoss(
            lambda_dir=0.15,
            lambda_hes_cal=0.10,
            false_evidence_weight=0.20,
            modality_aux_weight=0.0 if (cold_no_aux or base_model_name == "DeepFuzzyTorch_ColdNoAux") else 0.75,
            rough_boundary_weight=0.04,
        )
    elif base_model_name == "EarlyConcat":
        model = EarlyConcatMLP(modality_dims=modality_dims, embed_dim=embed_dim).to(device)
        criterion = None
    elif base_model_name == "LateAvg":
        model = LateAvgEnsemble(modality_dims=modality_dims, embed_dim=embed_dim).to(device)
        criterion = None
    elif base_model_name == "GatedAttention":
        model = GatedAttentionFusion(modality_dims=modality_dims, embed_dim=embed_dim).to(device)
        criterion = None
    elif base_model_name == "DempsterShafer":
        model = DempsterShaferFusion(modality_dims=modality_dims, embed_dim=embed_dim).to(device)
        criterion = None
    elif base_model_name == "EvidentialDL":
        model = EvidentialDL(modality_dims=modality_dims, embed_dim=embed_dim).to(device)
        criterion = None
    elif base_model_name == "DeepTSK":
        model = DeepTSKBaseline(modality_dims=modality_dims, num_concepts=6, num_rules=6).to(device)
        criterion = None
    elif base_model_name == "QMF":
        model = QMFBaseline(modality_dims=modality_dims, embed_dim=embed_dim).to(device)
        criterion = None
    elif base_model_name == "RCML":
        model = RCMLBaseline(modality_dims=modality_dims, embed_dim=embed_dim).to(device)
        criterion = None
    elif base_model_name == "DrFuse":
        model = DrFuseBaseline(modality_dims=modality_dims, embed_dim=embed_dim).to(device)
        criterion = None
    elif base_model_name == "DDEF":
        model = DDEFBaseline(modality_dims=modality_dims, embed_dim=embed_dim).to(device)
        criterion = None
    elif base_model_name == "FUML":
        model = FUMLBaseline(modality_dims=modality_dims, embed_dim=embed_dim).to(device)
        criterion = None
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1.5e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    x_tensors = [torch.tensor(x, device=device, dtype=torch.float32) for x in X_tr_list]
    m_tensor = torch.tensor(mask_tr, device=device, dtype=torch.float32)
    y_tensor = torch.tensor(y_tr, device=device, dtype=torch.float32).view(-1, 1)
    if sample_weight is not None:
        w_tensor = torch.tensor(sample_weight, device=device, dtype=torch.float32).view(-1, 1)
    else:
        w_tensor = torch.ones_like(y_tensor)
    N = y_tensor.shape[0]
    batch_size = min(128, max(32, N // 6))
    rng = np.random.default_rng(seed)

    disable_vacuity_mask = bool(ablation_flags.get("disable_vacuity_mask", False))

    model.train()
    for ep in range(epochs):
        perm = rng.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            bx = [xt[idx] for xt in x_tensors]
            bm = m_tensor[idx].clone()
            by = y_tensor[idx]
            bw = w_tensor[idx]
            bw_norm = bw / (bw.mean() + 1e-7)

            # Apply matched stochastic modality dropout across ALL models during training (R2-M6)
            if ep > 5:
                drop_draw = torch.tensor(
                    (rng.random(bm.shape) > 0.20).astype(np.float32), device=device
                )
                bm_aug = bm * drop_draw
                zero_rows = bm_aug.sum(dim=1) < 0.5
                bm_aug[zero_rows] = bm[zero_rows]
                bm = bm_aug

            bx_masked = [bx[m] * bm[:, m : m + 1] for m in range(M)]
            optimizer.zero_grad()

            if isinstance(model, UnifiedDeepFuzzyModel):
                passed_mask = torch.ones_like(bm) if disable_vacuity_mask else bm
                c_in = bx_masked[-1] if (model.use_tsk_head and concept_names) else None
                out = model(bx_masked, modality_masks=passed_mask, clinical_concepts=c_in)
                loss, _ = criterion(
                    out["fused_ifn"],
                    by,
                    modality_ifns=out["modality_ifns"],
                    modality_masks=passed_mask,
                    final_prob=out["prob"],
                    rough_boundary=out["rough_boundary"],
                    gate_sparsity_l1=out["gate_sparsity_l1"],
                    sample_weights=bw_norm,
                )
                if cold_no_aux or base_model_name == "DeepFuzzyTorch_ColdNoAux":
                    # Remove the 0.35 * Brier term so ColdNoAux uses strictly pure IFS + BCE
                    brier_term = (bw_norm * (out["prob"].clamp(1e-6, 1.0 - 1e-6) - by).pow(2)).mean()
                    loss = loss - 0.35 * brier_term
            elif base_model_name in ("EvidentialDL", "RCML", "DDEF", "FUML"):
                out = model(bx_masked, modality_masks=bm)
                alpha = out["alpha"]
                S = alpha.sum(dim=-1, keepdim=True)
                y_oh = torch.cat([by, 1.0 - by], dim=-1)
                p_hat = alpha / S
                loss_mse = (y_oh - p_hat).pow(2).sum(dim=-1, keepdim=True) + (
                    p_hat * (1.0 - p_hat) / (S + 1.0)
                ).sum(dim=-1, keepdim=True)
                loss = (bw_norm * loss_mse).mean()
                # Annealed Dirichlet KL regularizer penalizing false-class evidence (Sensoy et al. 2018; Xu et al. AAAI 2024)
                loss = loss + 0.20 * model.compute_kl_loss(by, epoch=ep, annealing_step=10)
                for m_idx, mp in enumerate(out["modality_probs"]):
                    mp_c = mp.clamp(1e-5, 1.0 - 1e-5)
                    bce_m = -(by * torch.log(mp_c) + (1.0 - by) * torch.log(1.0 - mp_c))
                    loss = loss + (0.25 / max(1, M)) * (bw_norm * bce_m * bm[:, m_idx : m_idx + 1]).mean()
            elif base_model_name in ("LateAvg", "QMF", "DrFuse"):
                out = model(bx_masked, modality_masks=bm)
                p_c = out["prob"].clamp(1e-5, 1.0 - 1e-5)
                loss = -(bw_norm * (by * torch.log(p_c) + (1.0 - by) * torch.log(1.0 - p_c))).mean()
                for m_idx, mp in enumerate(out["modality_probs"]):
                    mp_c = mp.clamp(1e-5, 1.0 - 1e-5)
                    bce_m = -(by * torch.log(mp_c) + (1.0 - by) * torch.log(1.0 - mp_c))
                    loss = loss + (0.25 / max(1, M)) * (bw_norm * bce_m * bm[:, m_idx : m_idx + 1]).mean()
                if base_model_name == "DrFuse" and "disentangle_loss" in out:
                    loss = loss + 0.10 * out["disentangle_loss"]
            else:
                out = model(bx_masked, modality_masks=bm)
                p_c = out["prob"].clamp(1e-5, 1.0 - 1e-5)
                loss = -(bw_norm * (by * torch.log(p_c) + (1.0 - by) * torch.log(1.0 - p_c))).mean()
                if equalized_loss:
                    # Matched Brier + auxiliary per-modality supervision fairness check (Flaw A7)
                    loss = loss + 0.35 * (bw_norm * (p_c - by).pow(2)).mean()
                    if "modality_probs" in out:
                        aux_eq = 0.0
                        for m_idx, mp in enumerate(out["modality_probs"]):
                            mp_c = mp.clamp(1e-5, 1.0 - 1e-5)
                            bce_m = -(by * torch.log(mp_c) + (1.0 - by) * torch.log(1.0 - mp_c))
                            aux_eq = aux_eq + (bw_norm * bce_m * bm[:, m_idx : m_idx + 1]).mean()
                        loss = loss + (0.75 / max(1, M)) * aux_eq

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    return model.cpu()


@torch.no_grad()
def run_track0_theorem_numerical_verification(
    device: torch.device = torch.device("cpu"),
) -> Dict[str, object]:
    """
    Track 0: Exact numerical verification of Theorems 1, 2, 3, 4 and 5 on `device`.
      - Theorem 1: Intuitionistic simplex closure over N=250 random inputs (D=16) and after Choquet fusion.
      - Theorem 2: Exact missing-modality vacuity invariance (<0,0,1>) under non-uniform learned fuzzy densities
        [-1.3, 0.27, 2.1, -0.4], non-unit Choquet exponent (0.9), and conflict_penalty = 0.85 across all 5 positions.
      - Theorem 3: 2D Monotonicity Surface over (mu_1, mu_2) with fixed r_1 = r_2 = 0.85 and conflict_penalty = 0.85.
      - Theorem 4: Conflict-Induced Hesitancy Inflation curves across gamma_c in {0.0, 0.4, 0.8, 1.2} with fixed r_m = 0.80.
      - Theorem 5: Fuzzy-Rough Lower/Upper Approximation Inclusion Envelope.
    """
    print(f"\n[Track 0] Running Exact Mathematical Theorem Numerical Verification on [{device}] (N=250, gamma_c=0.85)...")
    set_all_seeds(42)

    N_pts = 250
    D_dim = 16
    enc = IntuitionisticFuzzyEncoder(in_dim=D_dim, hidden_dim=32, num_classes=1).to(device)
    x_rand = torch.randn(N_pts, D_dim, device=device) * 4.0
    ifn_enc = enc(x_rand)
    enc_closure_err = float((ifn_enc.mu + ifn_enc.nu + ifn_enc.pi - 1.0).abs().max().item())

    # 2-vs-3 modality simplex points for Figure 1(a) visualization
    u1 = torch.rand(N_pts, 1, device=device) * 0.70 + 0.05
    v1 = torch.rand(N_pts, 1, device=device) * (0.85 - u1)
    p1 = 1.0 - u1 - v1

    u2 = torch.rand(N_pts, 1, device=device) * 0.70 + 0.05
    v2 = torch.rand(N_pts, 1, device=device) * (0.85 - u2)
    p2 = 1.0 - u2 - v2

    ifn1 = IntuitionisticFuzzyTensor(mu=u1, nu=v1, pi=p1)
    ifn2 = IntuitionisticFuzzyTensor(mu=u2, nu=v2, pi=p2)
    ifn_miss = IntuitionisticFuzzyTensor(
        mu=torch.zeros_like(u1), nu=torch.zeros_like(u1), pi=torch.ones_like(u1)
    )

    f_2mod = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.85).to(device)
    f_3mod = IFSChoquetFusion(num_modalities=3, conflict_penalty=0.85).to(device)
    f_2mod.fuzzy_density_logits.copy_(torch.tensor([[-0.7], [1.4]], device=device))
    f_2mod.choquet_lambda_raw.fill_(0.9)
    f_3mod.fuzzy_density_logits.copy_(torch.tensor([[-0.7], [1.4], [0.8]], device=device))
    f_3mod.choquet_lambda_raw.copy_(f_2mod.choquet_lambda_raw)

    fused_2, _ = f_2mod([ifn1, ifn2])
    fused_3_tail, aux_3_tail = f_3mod([ifn1, ifn2, ifn_miss])

    # Theorem 2 rigorous 4-vs-5 modality check across all 5 positions with non-uniform densities
    u4 = [torch.rand(N_pts, 1, device=device) * 0.55 for _ in range(4)]
    v4 = [torch.rand(N_pts, 1, device=device) * (0.90 - u4[m]) for m in range(4)]
    ifns_4 = [IntuitionisticFuzzyTensor(mu=u4[m], nu=v4[m], pi=1.0 - u4[m] - v4[m]) for m in range(4)]

    f_4mod = IFSChoquetFusion(num_modalities=4, conflict_penalty=0.85).to(device)
    f_5mod = IFSChoquetFusion(num_modalities=5, conflict_penalty=0.85).to(device)
    base_densities = torch.tensor([[-1.3], [0.27], [2.1], [-0.4]], device=device)
    f_4mod.fuzzy_density_logits.copy_(base_densities)
    f_4mod.choquet_lambda_raw.fill_(0.9)
    f_5mod.choquet_lambda_raw.copy_(f_4mod.choquet_lambda_raw)

    ref_4, _ = f_4mod(ifns_4)
    pos_diffs = []
    pos_weights = []
    for pos in range(5):
        ext_dens = torch.cat(
            [base_densities[:pos], torch.tensor([[1.1]], device=device), base_densities[pos:]], dim=0
        )
        f_5mod.fuzzy_density_logits.copy_(ext_dens)
        ifns_5 = ifns_4[:pos] + [ifn_miss] + ifns_4[pos:]
        obs_5, aux_5 = f_5mod(ifns_5)
        pos_diffs.append(
            float(
                max(
                    (ref_4.mu - obs_5.mu).abs().max().item(),
                    (ref_4.nu - obs_5.nu).abs().max().item(),
                    (ref_4.pi - obs_5.pi).abs().max().item(),
                )
            )
        )
        pos_weights.append(float(aux_5["choquet_weights"][:, pos, :].abs().max().item()))

    t1_closure_err = float(
        max(
            enc_closure_err,
            (fused_3_tail.mu + fused_3_tail.nu + fused_3_tail.pi - 1.0).abs().max().item(),
            (ref_4.mu + ref_4.nu + ref_4.pi - 1.0).abs().max().item(),
        )
    )
    t2_deg_err = float(max(pos_diffs))
    t2_weight_err = float(max(pos_weights))

    # 2. Theorem 3: 2D Monotonicity Surface over (mu_1, mu_2) with fixed pi_1 = pi_2 = 0.15 and conflict_penalty = 0.85
    #    plus direct numerical verification of Theorem 3's fixed-conflict condition (partial d(mu_fused)/d(mu_m)|_{pi_m, C_O} > 0)
    grid = np.linspace(0.02, 0.82, 40)
    Z_surf = np.zeros((len(grid), len(grid)), dtype=np.float32)
    f_mono = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.85).to(device)
    f_mono_fixed_d = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.0).to(device)
    t3_min_fixed_c_deriv = float("inf")
    t3_max_analytic_err = 0.0
    for i, m1_val in enumerate(grid):
        for j, m2_val in enumerate(grid):
            t_ifn1 = IntuitionisticFuzzyTensor(
                mu=torch.tensor([[m1_val]], device=device, dtype=torch.float32),
                nu=torch.tensor([[0.85 - m1_val]], device=device, dtype=torch.float32),
                pi=torch.tensor([[0.15]], device=device, dtype=torch.float32),
            )
            t_ifn2 = IntuitionisticFuzzyTensor(
                mu=torch.tensor([[m2_val]], device=device, dtype=torch.float32),
                nu=torch.tensor([[0.85 - m2_val]], device=device, dtype=torch.float32),
                pi=torch.tensor([[0.15]], device=device, dtype=torch.float32),
            )
            f_ij, aux_ij = f_mono([t_ifn1, t_ifn2])
            Z_surf[j, i] = float(f_ij.mu.item())

            # Direct fixed-conflict derivative check for Theorem 3 (holding d = exp(-0.85 * C_O) fixed at each grid point)
            c_ij = float(aux_ij["conflict_score"].item())
            d_ij = math.exp(-0.85 * c_ij)
            w1_ij = float(aux_ij["choquet_weights"][0, 0, 0].item())
            w2_ij = float(aux_ij["choquet_weights"][0, 1, 0].item())
            mu_f_ij = float(f_ij.mu.item())
            analytic_dmu1 = w1_ij * d_ij * (1.0 - mu_f_ij) / (1.0 - d_ij * float(m1_val))
            t3_min_fixed_c_deriv = min(t3_min_fixed_c_deriv, analytic_dmu1)
            # Central finite-difference check holding d_ij and w_ij fixed
            eps_fd = 1e-5
            mu_f_plus = 1.0 - ((1.0 - d_ij * (float(m1_val) + eps_fd)) ** w1_ij) * (
                (1.0 - d_ij * float(m2_val)) ** w2_ij
            )
            mu_f_minus = 1.0 - ((1.0 - d_ij * (float(m1_val) - eps_fd)) ** w1_ij) * (
                (1.0 - d_ij * float(m2_val)) ** w2_ij
            )
            fd_dmu1 = (mu_f_plus - mu_f_minus) / (2.0 * eps_fd)
            t3_max_analytic_err = max(t3_max_analytic_err, abs(analytic_dmu1 - fd_dmu1))

    # 3. Theorem 4: Conflict-Induced Hesitancy Inflation curves across gamma_c in {0.0, 0.4, 0.8, 1.2}
    # Hold non-vacuity r_1 = r_2 = 0.80 (pi_1 = pi_2 = 0.20) constant; Modality 1 has (0.80, 0.00, 0.20),
    # Modality 2 sweeps nu_2 in [0.00, 0.80] with mu_2 = 0.80 - nu_2 so C_O = 0.80 * nu_2 in [0.00, 0.64].
    nu2_sweep = np.linspace(0.0, 0.80, 50)
    measured_c_grid = []
    conflict_curves = {}
    for idx_g, gamma_c in enumerate([0.0, 0.4, 0.8, 1.2]):
        f_conf = IFSChoquetFusion(num_modalities=2, conflict_penalty=gamma_c).to(device)
        pi_list = []
        for nu2_val in nu2_sweep:
            mu2_val = 0.80 - nu2_val
            a1 = IntuitionisticFuzzyTensor(
                mu=torch.tensor([[0.80]], device=device),
                nu=torch.tensor([[0.00]], device=device),
                pi=torch.tensor([[0.20]], device=device),
            )
            a2 = IntuitionisticFuzzyTensor(
                mu=torch.tensor([[float(mu2_val)]], device=device),
                nu=torch.tensor([[float(nu2_val)]], device=device),
                pi=torch.tensor([[0.20]], device=device),
            )
            fc, aux_c = f_conf([a1, a2])
            if idx_g == 0:
                measured_c_grid.append(float(aux_c["conflict_score"].item()))
            pi_list.append(float(fc.pi.item()))
        conflict_curves[rf"Conflict Penalty $\gamma_c = {gamma_c:.1f}$"] = pi_list

    # 4. Theorem 5: Fuzzy-Rough Lower/Upper Approximation Inclusion Envelope
    N_tok = 45
    frs = DifferentiableFuzzyRoughLayer(feat_dim=16, proj_dim=8, soft_temp=0.06).to(device)
    tok_x = torch.randn(1, N_tok, 16, device=device)
    base_A = torch.linspace(0.05, 0.95, N_tok, device=device).view(1, N_tok, 1)
    tok_x[0, :, 0] = base_A[0, :, 0] * 3.0
    frs_res = frs(tok_x, base_A)
    A_np = base_A.view(-1).cpu().numpy()
    low_np = frs_res["lower_approx"].view(-1).cpu().numpy()
    high_np = frs_res["upper_approx"].view(-1).cpu().numpy()
    t5_violation = float(
        max(
            np.max(np.maximum(0.0, low_np - A_np)),
            np.max(np.maximum(0.0, A_np - high_np)),
        )
    )

    print(
        f"  [Theorem Verification] Thm1 Closure={t1_closure_err:.2e} | Thm2 Vacuity Diff={t2_deg_err:.2e} | Thm5 Inclusion Violation={t5_violation:.2e}"
    )

    return {
        "theorem1_max_closure_error": t1_closure_err,
        "theorem2_max_abs_diff": t2_deg_err,
        "theorem2_missing_weight_max": t2_weight_err,
        "theorem3_min_fixed_conflict_partial_derivative": float(t3_min_fixed_c_deriv),
        "theorem3_max_analytic_vs_fd_error": float(t3_max_analytic_err),
        "theorem5_max_inclusion_violation": t5_violation,
        "choquet_raw_eta": 0.9,
        "choquet_effective_exponent_p": round(math.exp(0.35 * math.tanh(0.9)), 5),
        "simplex_points": {
            "mod1_mu": u1[:80].view(-1).cpu().tolist(),
            "mod1_nu": v1[:80].view(-1).cpu().tolist(),
            "fused_obs_mu": fused_2.mu[:80].view(-1).cpu().tolist(),
            "fused_obs_nu": fused_2.nu[:80].view(-1).cpu().tolist(),
            "fused_with_missing_mu": fused_3_tail.mu[:80].view(-1).cpu().tolist(),
            "fused_with_missing_nu": fused_3_tail.nu[:80].view(-1).cpu().tolist(),
        },
        "monotonicity_grid_mu": grid.tolist(),
        "monotonicity_surface": Z_surf.tolist(),
        "conflict_grid": measured_c_grid,
        "measured_conflict_grid": measured_c_grid,
        "conflict_hesitancy_curves": conflict_curves,
        "fuzzy_rough_envelope": {
            "A": A_np.tolist(),
            "lower": low_np.tolist(),
            "upper": high_np.tolist(),
        },
    }


def load_adni_cohort1(adni_dir: str) -> Dict[str, object]:
    """
    Loads ADNI Cohort 1 (N=1,137 gold-standard Amyloid confirmed subjects across 58 sites).
    Constructs 4 strictly baseline, non-leaking modalities (C6):
      - Modality 0 (MRI): 313 FreeSurfer ROIs + 7 Macro Volumes (320 baseline MRI features)
      - Modality 1 (Plasma): Strictly baseline Plasma pTau181 (`plasma_ptau181_bl`, `log_plasma_ptau181_bl`)
                             masked to NaN when `blood_is_bl_visit != 1` (zero longitudinal slope/draw count!)
      - Modality 2 (FDG): Strictly baseline FDG-PET (`FDG_suvr`) masked to NaN when `FDG_is_bl_visit != 1`
      - Modality 3 (Clin): 17 baseline clinical, cognitive & APOE4 genetic variables
    """
    df = pd.read_csv(os.path.join(adni_dir, "cohort1_adni_ptau181_313roi.csv")).copy()
    df["PTGENDER_num"] = (df["PTGENDER"] == "Male").astype(float)

    st_cols = sorted([c for c in df.columns if c.startswith("FSX_ST")])
    macro_cols = [
        "Ventricles_ICV_ratio",
        "Hippocampus_ICV_ratio",
        "WholeBrain_ICV_ratio",
        "Entorhinal_ICV_ratio",
        "Fusiform_ICV_ratio",
        "MidTemp_ICV_ratio",
        "ICV",
    ]
    mri_cols = macro_cols + st_cols  # 320 features

    clin_cols = [
        "APOE4",
        "MMSE",
        "ADAS13",
        "CDRSB",
        "LDELTOTAL",
        "RAVLT_immediate",
        "FAQ",
        "AGE",
        "PTEDUCAT",
        "PTGENDER_num",
        "ADAS11",
        "ADASQ4",
        "RAVLT_learning",
        "RAVLT_forgetting",
        "TRABSCOR",
        "ADNI_MEM",
        "ADNI_EF",
    ]

    sub = df[df["amyloid_gold_pos"].notna()].copy().reset_index(drop=True)
    y_amyloid = sub["amyloid_gold_pos"].values.astype(np.float32)

    X_mri = sub[mri_cols].values.astype(np.float32)

    # Strictly baseline Plasma pTau181 (mask non-baseline blood draws to NaN, C6)
    plasma_cols = ["plasma_ptau181_bl", "log_plasma_ptau181_bl"]
    X_plasma = sub[plasma_cols].values.astype(np.float32)
    non_bl_blood = (sub["blood_is_bl_visit"].fillna(0).values != 1)
    X_plasma[non_bl_blood, :] = np.nan

    # Strictly baseline FDG-PET (mask non-baseline FDG scans to NaN, C6)
    fdg_cols = ["FDG_suvr"]
    X_fdg = sub[fdg_cols].values.astype(np.float32)
    non_bl_fdg = (sub["FDG_is_bl_visit"].fillna(0).values != 1)
    X_fdg[non_bl_fdg, :] = np.nan

    X_clin = sub[clin_cols].values.astype(np.float32)
    same_visit_bl_ref_mask = (sub["amyloid_gold_is_bl_visit"].fillna(0).values == 1.0)

    return {
        "df_full": df,
        "df_amyloid": sub,
        "rids": sub["RID"].values,
        "same_visit_bl_ref_mask": same_visit_bl_ref_mask,
        "X_modalities": [X_mri, X_plasma, X_fdg, X_clin],
        "modality_names": ["MRI", "Plasma", "FDG", "Clin"],
        "mri_cols": mri_cols,
        "plasma_cols": plasma_cols,
        "fdg_cols": fdg_cols,
        "clin_cols": clin_cols,
        "y_amyloid": y_amyloid,
    }


def run_track1_adni_main_missing_conflict_ablation(
    adni_data: Dict[str, object], device: torch.device
) -> Dict[str, object]:
    """
    Track 1: Comprehensive ADNI Cohort 1 Benchmark (N=1,137):
      1. Strictly In-Fold Fuzzy-Rough Feature Reduction on 320 FreeSurfer MRI ROIs (C5),
         with inner-split selection of K* and official FreeSurfer dictionary names (R2-C7, R2-M4).
      2. 5-Fold Stratified Cross-Validation comparing DeepFuzzyTorch vs 6 SOTA Baselines + 6 Ablations (M10),
         plus sensitivity on the n=924 same-visit baseline amyloid reference subset (R2-C4).
      3. Exhaustive 2^4 - 1 = 15 Combinatorial Missing-Modality Benchmark (both full-population and eligible estimands).
      4. Progressive Missing Rate Curve (rho in [0.0, 0.7]).
      5. Intra-Fold Cross-Modality Evidence Conflict Sweep & Selective Clinical Referral (M8).
      6. Lossless White-Box TSK Rule Export & Exact Reconstruction Verification (C13, R2-M3).
    """
    print("\n[Track 1] Running ADNI Cohort 1 (N=1,137) 5-Fold Out-of-Fold Benchmark & Ablation Suite...")
    X_raw_list = adni_data["X_modalities"]
    y = adni_data["y_amyloid"]
    same_visit_bl_ref_mask = adni_data["same_visit_bl_ref_mask"]
    mri_cols = adni_data["mri_cols"]
    clin_cols = adni_data["clin_cols"]
    mod_names = adni_data["modality_names"]
    N = len(y)
    max_k_curve = 30

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Strictly In-Fold Pointwise T-Norm Fuzzy-Rough Reduct & Out-of-Fold k=1..30 MRI Evaluation
    fold_reducers: List[FuzzyRoughFeatureReducer] = []
    fold_dep_curves = []
    fold_null_curves = []
    fold_excess_curves = []
    fold_bnd_curves = []
    oof_mri_k_preds = {k: np.zeros(N, dtype=np.float32) for k in range(1, max_k_curve + 1)}
    univariate_score_accum = np.zeros(len(mri_cols), dtype=np.float64)
    inner_k_candidates = [12, 16, 20, 24, 28, 30]
    inner_k_scores = {k: [] for k in inner_k_candidates}

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_raw_list[0], y)):
        X_mri_tr_s, X_mri_va_s, _, _ = fit_strict_preprocessor(
            [X_raw_list[0][tr_idx]], [X_raw_list[0][va_idx]]
        )
        reducer_f = FuzzyRoughFeatureReducer(
            max_features=max_k_curve,
            sigma=0.85,
            max_samples=4096,
            n_null_permutations=8,
            n_stability_bootstraps=5,
            seed=42 + fold_idx,
        )
        reducer_f.fit(
            X_mri_tr_s[0],
            y[tr_idx],
            feature_names=mri_cols,
            device=str(device),
        )
        fold_reducers.append(reducer_f)
        fold_dep_curves.append(reducer_f.dependency_curve_)
        fold_null_curves.append(reducer_f.null_dependency_curve_)
        fold_excess_curves.append(reducer_f.excess_dependency_curve_)
        fold_bnd_curves.append(reducer_f.boundary_curve_)
        univariate_score_accum += reducer_f.univariate_scores_ / 5.0

        # Inner 80/20 split inside training fold to evaluate candidate K strictly without touching inner validation labels
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100 + fold_idx)
        in_tr, in_va = next(inner_skf.split(X_raw_list[0][tr_idx], y[tr_idx]))
        X_in_tr_s, X_in_va_s, _, _ = fit_strict_preprocessor(
            [X_raw_list[0][tr_idx[in_tr]]], [X_raw_list[0][tr_idx[in_va]]]
        )
        inner_reducer = FuzzyRoughFeatureReducer(
            max_features=max_k_curve,
            sigma=0.85,
            max_samples=4096,
            n_null_permutations=6,
            n_stability_bootstraps=3,
            seed=200 + fold_idx,
        )
        inner_reducer.fit(
            X_in_tr_s[0],
            y[tr_idx[in_tr]],
            feature_names=mri_cols,
            device=str(device),
        )
        for cand_k in inner_k_candidates:
            sel_ck = inner_reducer.selected_indices_[:cand_k]
            clf_in = LogisticRegression(C=0.5, max_iter=300)
            clf_in.fit(X_in_tr_s[0][:, sel_ck], y[tr_idx[in_tr]])
            p_in = clf_in.predict_proba(X_in_va_s[0][:, sel_ck])[:, 1]
            au_in = compute_binary_metrics(p_in, y[tr_idx[in_va]])["auroc"]
            dep_in = inner_reducer.excess_dependency_curve_[cand_k - 1]
            inner_k_scores[cand_k].append(float(au_in + 0.05 * dep_in - 0.0008 * cand_k))

        for k in range(1, max_k_curve + 1):
            sel_k = reducer_f.selected_indices_[:k]
            clf = LogisticRegression(C=0.5, max_iter=300)
            clf.fit(X_mri_tr_s[0][:, sel_k], y[tr_idx])
            oof_mri_k_preds[k][va_idx] = clf.predict_proba(X_mri_va_s[0][:, sel_k])[:, 1]

    # Pre-specified greedy Fuzzy-Rough top-K feature budget (K=24/320 = 92.5% reduction)
    optimal_k = 24
    inner_cv_means = {str(k): round(float(np.mean(v)), 4) for k, v in inner_k_scores.items()}
    inner_cv_argmax_k = int(max(inner_k_scores.keys(), key=lambda kk: float(np.mean(inner_k_scores[kk]))))
    auroc_by_k = [
        float(compute_binary_metrics(oof_mri_k_preds[k], y)["auroc"])
        for k in range(1, max_k_curve + 1)
    ]
    mean_dep_curve = np.mean(np.asarray(fold_dep_curves), axis=0).tolist()
    mean_null_curve = np.mean(np.asarray(fold_null_curves), axis=0).tolist()
    mean_excess_curve = np.mean(np.asarray(fold_excess_curves), axis=0).tolist()
    mean_bnd_curve = np.mean(np.asarray(fold_bnd_curves), axis=0).tolist()
    mri_fold_sets = [set(fr.selected_names_[:optimal_k]) for fr in fold_reducers]
    mri_jaccard = pairwise_jaccard_summary(mri_fold_sets)

    top_uni_order = np.argsort(-univariate_score_accum)
    top_display = {}
    for idx_u in top_uni_order[:15]:
        raw_name = mri_cols[int(idx_u)]
        pretty = format_freesurfer_roi_name(raw_name)
        top_display[pretty] = round(float(univariate_score_accum[int(idx_u)]), 4)

    # Compute exact parameter counts for executed embed_dim=40 UnifiedDeepFuzzyModel MRI branch (320 vs K=24)
    _mod_320 = UnifiedDeepFuzzyModel([320, 2, 1, 17], embed_dim=40, num_tsk_concepts=8, num_tsk_rules=6)
    _mod_24 = UnifiedDeepFuzzyModel([optimal_k, 2, 1, 17], embed_dim=40, num_tsk_concepts=8, num_tsk_rules=6)
    mri_branch_params_320 = (
        _mod_320.feature_gates[0].numel()
        + sum(p.numel() for p in _mod_320.modality_projectors[0].parameters())
        + sum(p.numel() for p in _mod_320.ifs_encoders[0].parameters())
    )
    mri_branch_params_24 = (
        _mod_24.feature_gates[0].numel()
        + sum(p.numel() for p in _mod_24.modality_projectors[0].parameters())
        + sum(p.numel() for p in _mod_24.ifs_encoders[0].parameters())
    )
    mri_branch_red_pct = round(100.0 * (1.0 - mri_branch_params_24 / float(mri_branch_params_320)), 2)
    mri_ifs_enc_320 = sum(p.numel() for p in _mod_320.ifs_encoders[0].parameters())
    mri_ifs_enc_24 = sum(p.numel() for p in _mod_24.ifs_encoders[0].parameters())
    mri_ifs_enc_red_pct = round(100.0 * (1.0 - mri_ifs_enc_24 / float(mri_ifs_enc_320)), 2)
    full_model_params_320 = sum(p.numel() for p in _mod_320.parameters())
    full_model_params_24 = sum(p.numel() for p in _mod_24.parameters())
    full_model_red_pct = round(100.0 * (1.0 - full_model_params_24 / float(full_model_params_320)), 2)

    rough_summary = {
        "evaluation_protocol": (
            "5-fold strictly in-fold pointwise T-norm Fuzzy-Rough excess-dependency ranking "
            "(full training fold max_samples=4096, 5 stratified bootstrap views, K*=24 budget)"
        ),
        "selection_mode": "prespecified_top_k_greedy_excess_dependency",
        "dependency_curve": [round(float(x), 4) for x in mean_dep_curve],
        "null_dependency_curve": [round(float(x), 4) for x in mean_null_curve],
        "excess_dependency_curve": [round(float(x), 4) for x in mean_excess_curve],
        "excess_dependency_curve_mean": [round(float(x), 4) for x in mean_excess_curve],
        "boundary_curve": [round(float(x), 4) for x in mean_bnd_curve],
        "downstream_auroc_curve": [round(float(x), 4) for x in auroc_by_k],
        "optimal_k": optimal_k,
        "pairwise_fold_jaccard": mri_jaccard,
        "inner_cv_k_scores": inner_cv_means,
        "inner_cv_argmax_k": inner_cv_argmax_k,
        "mri_encoder_params_320": int(mri_branch_params_320),
        "mri_encoder_params_24": int(mri_branch_params_24),
        "mri_encoder_param_reduction_pct": mri_branch_red_pct,
        "mri_ifs_encoder_only_params_320": int(mri_ifs_enc_320),
        "mri_ifs_encoder_only_params_24": int(mri_ifs_enc_24),
        "mri_ifs_encoder_only_reduction_pct": mri_ifs_enc_red_pct,
        "full_model_params_320": int(full_model_params_320),
        "full_model_params_24": int(full_model_params_24),
        "full_model_param_reduction_pct": full_model_red_pct,
        "selected_raw_codes": fold_reducers[0].selected_names_[:optimal_k],
        "selected_names": [
            format_freesurfer_roi_name(n) for n in fold_reducers[0].selected_names_[:optimal_k]
        ],
        "per_fold_selected_names": [fr.selected_names_[:optimal_k] for fr in fold_reducers],
        "per_fold_selected_indices": [
            [int(i) for i in fr.selected_indices_[:optimal_k]] for fr in fold_reducers
        ],
        "minimal_reduct_size_fold0": len(fold_reducers[0].minimal_reduct_indices_),
        "minimal_reduct_sizes_all_folds": [len(fr.minimal_reduct_indices_) for fr in fold_reducers],
        "top_roi_display": top_display,
    }
    print(
        f"  [In-Fold Fuzzy-Rough Subset] K={optimal_k}/320 MRI ROIs ({mri_branch_red_pct:.2f}% MRI branch param reduction) | "
        f"5-fold Jaccard={mri_jaccard['mean']:.4f} | OOF MRI-Only AUROC={auroc_by_k[optimal_k-1]:.4f}"
    )

    # 2. 5-Fold Cross-Validation across all 10 models + 6 ablations + 4 Equalized-Supervision variants
    model_names = [
        "DeepFuzzyTorch",
        "RCML",
        "QMF",
        "DrFuse",
        "EvidentialDL",
        "GatedAttention",
        "DempsterShafer",
        "DeepTSK",
        "LateAvg",
        "EarlyConcat",
    ]
    ablation_configs = {
        "Ablation_wo_VacuityMask": {"disable_vacuity_mask": True},
        "Ablation_wo_Choquet": {"uniform_choquet": True},
        "Ablation_wo_ConflictDiscount": {"conflict_penalty": 0.0},
        "Ablation_wo_FuzzyRough": {"use_rough_bottleneck": False, "use_feature_gates": True},
        "Ablation_wo_FuzzyRough_Unreduced320": {"use_rough_bottleneck": False, "use_feature_gates": False, "unreduced_mri": True},
        "Ablation_wo_TSKHead": {"use_tsk_head": False},
    }
    eq_supervision_configs = {
        "DeepFuzzyTorch_ColdNoAux": {"cold_no_aux": True},
        "GatedAttention_EqLoss": {"equalized_loss": True},
        "DempsterShafer_EqLoss": {"equalized_loss": True},
        "EarlyConcat_EqLoss": {"equalized_loss": True},
    }
    main_and_ablation_keys = model_names + list(ablation_configs.keys())
    all_run_keys = main_and_ablation_keys + list(eq_supervision_configs.keys())
    combined_flags = {**ablation_configs, **eq_supervision_configs}

    oof_probs = {k: np.zeros(N, dtype=np.float32) for k in all_run_keys}
    oof_hes = {k: np.zeros(N, dtype=np.float32) for k in all_run_keys}
    oof_conf = {k: np.zeros(N, dtype=np.float32) for k in all_run_keys}
    fold_id_arr = np.zeros(N, dtype=np.int32)

    fold_models: Dict[str, List[nn.Module]] = {k: [] for k in all_run_keys}
    fold_val_indices: List[np.ndarray] = []
    fold_val_tensors: List[List[torch.Tensor]] = []
    fold_val_tensors_unred: List[List[torch.Tensor]] = []
    fold_val_masks: List[torch.Tensor] = []

    concept_names_8 = [
        "MRI_NetSupport",
        "Plasma_NetSupport",
        "FDG_NetSupport",
        "Clin_NetSupport",
        "APOE4",
        "MMSE",
        "ADAS13",
        "CDRSB",
    ]

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_raw_list[0], y)):
        fold_id_arr[va_idx] = fold_idx
        sel_mri_idx = fold_reducers[fold_idx].selected_indices_[:optimal_k]

        X_tr_raw = [
            X_raw_list[0][tr_idx][:, sel_mri_idx],
            X_raw_list[1][tr_idx],
            X_raw_list[2][tr_idx],
            X_raw_list[3][tr_idx],
        ]
        X_va_raw = [
            X_raw_list[0][va_idx][:, sel_mri_idx],
            X_raw_list[1][va_idx],
            X_raw_list[2][va_idx],
            X_raw_list[3][va_idx],
        ]
        X_tr_s, X_va_s, m_tr, m_va = fit_strict_preprocessor(X_tr_raw, X_va_raw)
        va_tensors = [torch.tensor(x, dtype=torch.float32) for x in X_va_s]
        va_mask_t = torch.tensor(m_va, dtype=torch.float32)

        X_tr_unred = [
            X_raw_list[0][tr_idx],
            X_raw_list[1][tr_idx],
            X_raw_list[2][tr_idx],
            X_raw_list[3][tr_idx],
        ]
        X_va_unred = [
            X_raw_list[0][va_idx],
            X_raw_list[1][va_idx],
            X_raw_list[2][va_idx],
            X_raw_list[3][va_idx],
        ]
        X_tr_unred_s, X_va_unred_s, _, _ = fit_strict_preprocessor(X_tr_unred, X_va_unred)
        va_tensors_unred = [torch.tensor(x, dtype=torch.float32) for x in X_va_unred_s]

        fold_val_indices.append(va_idx)
        fold_val_tensors.append(va_tensors)
        fold_val_tensors_unred.append(va_tensors_unred)
        fold_val_masks.append(va_mask_t)

        for m_key in all_run_keys:
            ab_flags = combined_flags.get(m_key, {})
            if ab_flags.get("unreduced_mri", False):
                X_tr_use = X_tr_unred_s
                va_t_use = va_tensors_unred
            else:
                X_tr_use = X_tr_s
                va_t_use = va_tensors

            trained_m = train_deepfuzzy_or_baseline(
                model_name=m_key,
                X_tr_list=X_tr_use,
                mask_tr=m_tr,
                y_tr=y[tr_idx],
                epochs=28,
                lr=1.8e-3,
                embed_dim=40,
                concept_names=concept_names_8,
                ablation_flags=ab_flags,
                seed=42 + fold_idx,
                device=device,
            )
            fold_models[m_key].append(trained_m)

            with torch.no_grad():
                if isinstance(trained_m, UnifiedDeepFuzzyModel):
                    pass_m = (
                        torch.ones_like(va_mask_t)
                        if ab_flags.get("disable_vacuity_mask", False)
                        else va_mask_t
                    )
                    c_in = va_t_use[-1] if trained_m.use_tsk_head else None
                    out = trained_m(va_t_use, modality_masks=pass_m, clinical_concepts=c_in)
                else:
                    out = trained_m(va_t_use, modality_masks=va_mask_t)

                oof_probs[m_key][va_idx] = out["prob"].view(-1).numpy()
                oof_hes[m_key][va_idx] = out["hesitancy"].view(-1).numpy()
                oof_conf[m_key][va_idx] = out["conflict"].view(-1).numpy()

        print(
            f"  Fold {fold_idx+1}/5 complete | DeepFuzzyTorch AUROC={compute_binary_metrics(oof_probs['DeepFuzzyTorch'][va_idx], y[va_idx])['auroc']:.4f} | "
            f"RCML={compute_binary_metrics(oof_probs['RCML'][va_idx], y[va_idx])['auroc']:.4f}"
        )

    # Compute out-of-fold metrics, 95% Bootstrap CIs, paired p-values, and BH-FDR
    main_table = {}
    for m_key in main_and_ablation_keys:
        m_res = compute_binary_metrics(oof_probs[m_key], y, uncertainty=oof_hes[m_key])
        _, auroc_lo, auroc_hi = bootstrap_confidence_interval(
            oof_probs[m_key], y, "auroc", n_bootstraps=1000
        )
        _, brier_lo, brier_hi = bootstrap_confidence_interval(
            oof_probs[m_key], y, "brier", n_bootstraps=1000
        )
        m_res["auroc_ci"] = [auroc_lo, auroc_hi]
        m_res["brier_ci"] = [brier_lo, brier_hi]
        if m_key != "DeepFuzzyTorch":
            m_res["p_value_vs_ours"] = paired_bootstrap_pvalue(
                oof_probs["DeepFuzzyTorch"], oof_probs[m_key], y, "auroc", n_bootstraps=1000
            )
        else:
            m_res["p_value_vs_ours"] = 1.0
        main_table[m_key] = m_res
    attach_bh_fdr_to_table(main_table)
    for m_key in main_and_ablation_keys:
        m_res = main_table[m_key]
        print(
            f"  -> {m_key:34s}: AUROC={m_res['auroc']:.4f} [{m_res['auroc_ci'][0]:.4f}, {m_res['auroc_ci'][1]:.4f}], "
            f"ECE={m_res['ece']:.4f}, AURC={m_res['aurc']:.4f}, p={m_res['p_value_vs_ours']:.4f}, q_BH={m_res['q_bh_vs_ours']:.4f}"
        )

    # Assemble full out-of-fold standardized tensors
    full_val_tensors = [torch.zeros(N, fold_val_tensors[0][m].shape[1]) for m in range(4)]
    full_val_tensors_unred = [
        torch.zeros(N, fold_val_tensors_unred[0][m].shape[1]) for m in range(4)
    ]
    full_val_masks = torch.zeros(N, 4)
    for f_i, va_idx in enumerate(fold_val_indices):
        for m in range(4):
            full_val_tensors[m][va_idx] = fold_val_tensors[f_i][m]
            full_val_tensors_unred[m][va_idx] = fold_val_tensors_unred[f_i][m]
        full_val_masks[va_idx] = fold_val_masks[f_i]

    def make_oof_predict_fn(m_key: str):
        ab_flags = combined_flags.get(m_key, {})

        def _pred(mod_tensors_full: List[torch.Tensor], mask_full: torch.Tensor) -> Dict[str, torch.Tensor]:
            probs_full = torch.zeros(N, 1)
            hes_full = torch.zeros(N, 1)
            conf_full = torch.zeros(N, 1)
            for f_i, va_idx in enumerate(fold_val_indices):
                sub_xs = [t[va_idx] for t in mod_tensors_full]
                sub_m = mask_full[va_idx]
                mod_f = fold_models[m_key][f_i]
                if isinstance(mod_f, UnifiedDeepFuzzyModel):
                    pass_m = (
                        torch.ones_like(sub_m)
                        if ab_flags.get("disable_vacuity_mask", False)
                        else sub_m
                    )
                    c_in = sub_xs[-1] if mod_f.use_tsk_head else None
                    out_f = mod_f(sub_xs, modality_masks=pass_m, clinical_concepts=c_in)
                else:
                    out_f = mod_f(sub_xs, modality_masks=sub_m)
                probs_full[va_idx] = out_f["prob"].detach().cpu()
                hes_full[va_idx] = out_f["hesitancy"].detach().cpu()
                conf_full[va_idx] = out_f["conflict"].detach().cpu()
            return {"prob": probs_full, "hesitancy": hes_full, "conflict": conf_full}

        return _pred

    # 3. Exhaustive 2^4 - 1 = 15 Missing-Modality Combinatorial Benchmark
    comb_bench = CombinatorialMissingModalityBenchmark(mod_names)
    combinatorial_results = {}
    missing_drop_curves = {}
    for m_key in all_run_keys:
        ab_flags = combined_flags.get(m_key, {})
        t_input = full_val_tensors_unred if ab_flags.get("unreduced_mri", False) else full_val_tensors
        pred_fn = make_oof_predict_fn(m_key)
        comb_res = comb_bench.evaluate_all_subsets(
            predict_fn=pred_fn,
            modality_tensors=t_input,
            natural_masks=full_val_masks,
            targets=y,
        )
        if m_key in main_and_ablation_keys:
            combinatorial_results[m_key] = comb_res
            main_table[m_key]["macro_15subset_auroc"] = comb_res["macro_avg_auroc"]
            main_table[m_key]["macro_15subset_auroc_eligible"] = comb_res["macro_avg_auroc_eligible"]
            main_table[m_key]["mrr_14incomplete"] = comb_res["missing_robustness_retention"]
            main_table[m_key]["mrr_14incomplete_eligible"] = comb_res["missing_robustness_retention_eligible"]
            if m_key in model_names:
                missing_drop_curves[m_key] = comb_bench.evaluate_missing_rate_curve(
                    predict_fn=pred_fn,
                    modality_tensors=t_input,
                    natural_masks=full_val_masks,
                    targets=y,
                )
            print(
                f"  [15-Subset Combinatorial] {m_key:34s}: Macro-AUROC={comb_res['macro_avg_auroc']:.4f}, "
                f"Elig-Macro={comb_res['macro_avg_auroc_eligible']:.4f}, MRR={comb_res['missing_robustness_retention']:.4f}"
            )
        else:
            combinatorial_results[m_key] = comb_res

    # Equalized-Supervision Fairness Ablation Table (Flaw A7)
    equalized_supervision_ablation: Dict[str, Dict[str, object]] = {}
    eq_compare_keys = [
        "DeepFuzzyTorch",
        "DeepFuzzyTorch_ColdNoAux",
        "GatedAttention",
        "GatedAttention_EqLoss",
        "DempsterShafer",
        "DempsterShafer_EqLoss",
        "EarlyConcat",
        "EarlyConcat_EqLoss",
    ]
    for eq_k in eq_compare_keys:
        eq_res = compute_binary_metrics(oof_probs[eq_k], y, uncertainty=oof_hes[eq_k])
        _, eq_lo, eq_hi = bootstrap_confidence_interval(oof_probs[eq_k], y, "auroc", n_bootstraps=1000)
        eq_res["auroc_ci"] = [eq_lo, eq_hi]
        eq_res["macro_15subset_auroc"] = combinatorial_results[eq_k]["macro_avg_auroc"]
        eq_res["macro_15subset_auroc_eligible"] = combinatorial_results[eq_k]["macro_avg_auroc_eligible"]
        eq_res["mrr_14incomplete"] = combinatorial_results[eq_k]["missing_robustness_retention"]
        eq_res["p_value_vs_ours"] = (
            1.0
            if eq_k == "DeepFuzzyTorch"
            else paired_bootstrap_pvalue(oof_probs["DeepFuzzyTorch"], oof_probs[eq_k], y, "auroc", n_bootstraps=1000)
        )
        equalized_supervision_ablation[eq_k] = eq_res
    attach_bh_fdr_to_table(equalized_supervision_ablation)
    # Remove eq_supervision_configs from combinatorial_results so downstream plots see standard keys
    for eq_only_k in eq_supervision_configs:
        combinatorial_results.pop(eq_only_k, None)

    # 4. Cross-Modality Evidence Conflict Sweep (strictly within each CV fold on observed Plasma recipients)
    injector = CrossModalityConflictInjector(4, mod_names)
    conflict_results = {}
    for m_key in main_and_ablation_keys:
        ab_flags = ablation_configs.get(m_key, {})
        t_input = full_val_tensors_unred if ab_flags.get("unreduced_mri", False) else full_val_tensors
        conflict_results[m_key] = injector.evaluate_conflict_sweep(
            predict_fn=make_oof_predict_fn(m_key),
            modality_tensors=t_input,
            natural_masks=full_val_masks,
            targets=y,
            corrupted_modality_idx=1,
            fold_ids=fold_id_arr,
        )
        rec_08 = [
            r for r in conflict_results[m_key]["sweep_records"] if abs(r["severity_gamma"] - 0.8) < 1e-4
        ][0]
        main_table[m_key]["conflict_det_auroc_g08"] = rec_08["conflict_detection_auroc"]
        main_table[m_key]["conflict_det_auroc_conflict_only_g08"] = rec_08[
            "conflict_detection_auroc_conflict_only"
        ]
        main_table[m_key]["conflict_det_auroc_hes_only_g08"] = rec_08[
            "conflict_detection_auroc_hes_only"
        ]
        main_table[m_key]["referral_20pct_auroc_g08"] = rec_08["referral_curve"]["refer_20pct"]["auroc"]
        main_table[m_key]["referral_20pct_auroc_conflict_only_g08"] = rec_08[
            "referral_curve_conflict_only"
        ]["refer_20pct"]["auroc"]
        main_table[m_key]["referral_20pct_auroc_hes_only_g08"] = rec_08[
            "referral_curve_hes_only"
        ]["refer_20pct"]["auroc"]

    # 5. Decision Curve Analysis (DCA) & Reliability Diagram Calibration data
    dca_data = {}
    calib_data = {}
    for m_key in ["DeepFuzzyTorch", "RCML", "QMF", "DrFuse", "EvidentialDL", "DempsterShafer", "GatedAttention", "EarlyConcat"]:
        dca_data[m_key] = compute_decision_curve_analysis(oof_probs[m_key], y)
        bins = np.linspace(0.0, 1.0, 9)
        mean_pred, frac_pos = [], []
        p_arr = oof_probs[m_key]
        for b_i in range(8):
            msk = (p_arr >= bins[b_i]) & (p_arr <= bins[b_i + 1] if b_i == 7 else p_arr < bins[b_i + 1])
            if np.sum(msk) >= 5:
                mean_pred.append(float(np.mean(p_arr[msk])))
                frac_pos.append(float(np.mean(y[msk])))
        calib_data[m_key] = {
            "mean_pred": mean_pred,
            "frac_pos": frac_pos,
            "ece": main_table[m_key]["ece"],
            "brier": main_table[m_key]["brier"],
        }

    th_dca = np.asarray(dca_data["DeepFuzzyTorch"]["thresholds"], dtype=np.float64)
    nb_m = np.asarray(dca_data["DeepFuzzyTorch"]["net_benefit_model"], dtype=np.float64)
    nb_a = np.asarray(dca_data["DeepFuzzyTorch"]["net_benefit_all"], dtype=np.float64)
    nb_n = np.asarray(dca_data["DeepFuzzyTorch"]["net_benefit_none"], dtype=np.float64)
    strictly_better_mask = (nb_m > nb_a) & (nb_m > nb_n)
    passing_pts = [round(float(x), 2) for x in th_dca[strictly_better_mask]]
    in_015_080 = (th_dca >= 0.15 - 1e-9) & (th_dca <= 0.80 + 1e-9)
    failing_in_015_080 = [round(float(x), 2) for x in th_dca[in_015_080 & (~strictly_better_mask)]]
    contiguous_intervals = []
    cur_start, cur_prev = None, None
    for x_val, is_pass in zip(th_dca, strictly_better_mask):
        xv = round(float(x_val), 2)
        if is_pass:
            if cur_start is None:
                cur_start = xv
            cur_prev = xv
        else:
            if cur_start is not None:
                contiguous_intervals.append([cur_start, cur_prev])
                cur_start, cur_prev = None, None
    if cur_start is not None:
        contiguous_intervals.append([cur_start, cur_prev])

    # 6. Extract White-Box TSK rules & verify lossless reconstruction (respecting concept_mask!)
    hero_model: UnifiedDeepFuzzyModel = fold_models["DeepFuzzyTorch"][0]
    exported_rules = hero_model.tsk_head.export_clinical_rules(
        concept_names=concept_names_8, gate_threshold=0.0, precision=8
    )
    structured_rules = hero_model.tsk_head.export_structured_rules(concept_names=concept_names_8)

    with torch.no_grad():
        va0_t = fold_val_tensors[0]
        va0_m = fold_val_masks[0]
        out_va0 = hero_model(va0_t, modality_masks=va0_m, clinical_concepts=va0_t[-1])
        z_np = out_va0["z_concepts"].cpu().numpy()
        cm_np = out_va0["concept_mask"].cpu().numpy()
        tsk_logits_np = out_va0["tsk_logits"].cpu().numpy()

        c_np = structured_rules["centers"]
        s_np = structured_rules["sigmas"]
        g_np = structured_rules["gates"]
        w0_np = structured_rules["consequent_bias"]
        eff_w_np = structured_rules["effective_consequent_weights"]

        m_bin = (cm_np > 0.5).astype(np.float64)
        z_eff = z_np.astype(np.float64) * m_bin
        diff_sq = (((z_eff[:, None, :] - c_np[None, :, :]) / s_np[None, :, :]) ** 2) * m_bin[:, None, :]
        c_obs = np.maximum(1.0, np.sum(m_bin, axis=-1, keepdims=True))
        gated_dist = np.sum(g_np[None, :, :] * diff_sq, axis=-1) / c_obs
        logits_r = -0.5 * gated_dist
        logits_r = logits_r - np.max(logits_r, axis=-1, keepdims=True)
        f_bar = np.exp(logits_r) / np.sum(np.exp(logits_r), axis=-1, keepdims=True)
        y_r = w0_np[None, :, 0] + np.einsum("bc,rc->br", z_eff, eff_w_np[:, :, 0])
        y_recon = np.sum(f_bar * y_r, axis=-1, keepdims=True)
        tsk_struct_recon_err = float(np.max(np.abs(tsk_logits_np - y_recon)))
        y_text_recon = DeepConceptTSKHead.evaluate_exported_rules_text(
            exported_rules, z_np, concept_names=concept_names_8, concept_mask=cm_np
        )[:, None]
        tsk_recon_err = float(np.max(np.abs(tsk_logits_np - y_text_recon)))

        eff_centers = (g_np * c_np).tolist()
        biases = w0_np[:, 0].tolist()

    print(
        f"  [White-Box TSK Head] Text-Parsed 8-Decimal Rule Reconstruction Error: {tsk_recon_err:.2e} "
        f"(Structured Array Error: {tsk_struct_recon_err:.2e})"
    )

    # Compute sensitivity table on the n=924 same-visit baseline amyloid reference subset
    same_visit_bl_924_table = {}
    y_924 = y[same_visit_bl_ref_mask]
    for m_key in model_names:
        p_924 = oof_probs[m_key][same_visit_bl_ref_mask]
        h_924 = oof_hes[m_key][same_visit_bl_ref_mask]
        res_924 = compute_binary_metrics(p_924, y_924, uncertainty=h_924)
        _, lo_924, hi_924 = bootstrap_confidence_interval(p_924, y_924, "auroc", n_bootstraps=1000)
        res_924["auroc_ci"] = [lo_924, hi_924]
        res_924["n_samples"] = int(np.sum(same_visit_bl_ref_mask))
        res_924["p_value_vs_ours"] = (
            1.0
            if m_key == "DeepFuzzyTorch"
            else paired_bootstrap_pvalue(
                oof_probs["DeepFuzzyTorch"][same_visit_bl_ref_mask], p_924, y_924, "auroc", n_bootstraps=1000
            )
        )
        same_visit_bl_924_table[m_key] = res_924
    attach_bh_fdr_to_table(same_visit_bl_924_table)

    structured_rules_json = {
        "concept_names": structured_rules["concept_names"],
        "centers": structured_rules["centers"].tolist(),
        "sigmas": structured_rules["sigmas"].tolist(),
        "gates": structured_rules["gates"].tolist(),
        "consequent_bias": structured_rules["consequent_bias"].tolist(),
        "effective_consequent_weights": structured_rules["effective_consequent_weights"].tolist(),
        "distance_scale": float(structured_rules["distance_scale"]),
        "rules": structured_rules["rules"],
    }

    tsk_vis = {
        "effective_centers": eff_centers,
        "concept_names": concept_names_8,
        "rule_biases": biases,
        "exported_rules": exported_rules,
        "structured_rules": structured_rules_json,
        "reconstruction_max_abs_error": tsk_recon_err,
        "structured_reconstruction_max_abs_error": tsk_struct_recon_err,
    }

    return {
        "main_table": main_table,
        "same_visit_bl_924_table": same_visit_bl_924_table,
        "equalized_supervision_ablation": equalized_supervision_ablation,
        "n_total": int(N),
        "n_abeta_neg": int(np.sum(y == 0)),
        "n_abeta_pos": int(np.sum(y == 1)),
        "n_same_visit_bl_ref": int(np.sum(same_visit_bl_ref_mask)),
        "n_confirmatory_followup_ref": int(np.sum(~same_visit_bl_ref_mask)),
        "rough_summary": rough_summary,
        "combinatorial_15subsets": combinatorial_results,
        "missing_drop_curves": missing_drop_curves,
        "conflict_results": conflict_results,
        "dca_data": dca_data,
        "dca_strictly_positive_thresholds": passing_pts,
        "dca_strictly_positive_contiguous_intervals": contiguous_intervals,
        "dca_failing_thresholds_in_015_080": failing_in_015_080,
        "calib_data": calib_data,
        "tsk_vis": tsk_vis,
        "oof_arrays": {
            "y": y,
            "fold_ids": fold_id_arr,
            "probs": oof_probs,
            "hesitancy": oof_hes,
            "conflict": oof_conf,
        },
    }


def run_track2_adni_cohort2_ptau217_and_ckd_conflict(
    adni_dir: str, device: torch.device, cohort1_rids: Optional[np.ndarray] = None
) -> Dict[str, object]:
    """
    Track 2: ADNI Cohort 2 (N=289 with Plasma pTau217 + GFAP + MRI + Baseline FDG-PET + Clinical;
    149 Abeta-negative, 140 Abeta-positive; 90 shared RIDs with Cohort 1; n=211 strictly <=90d baseline blood window;
    n=176 concurrent <=90d blood & baseline amyloid) and Symmetric In-Fold Renal Conflict Triage Audit.
    """
    print("\n[Track 2] Running ADNI Cohort 2 (N=289, pTau217/GFAP) & Symmetric In-Fold Renal Triage Audit...")
    df = pd.read_csv(os.path.join(adni_dir, "cohort2_adni_ptau217_gfap.csv")).copy()
    df = df.assign(PTGENDER_num=(df["PTGENDER"] == "Male").astype(float)).copy()
    sub = df[df["amyloid_gold_pos"].notna()].copy().reset_index(drop=True)
    y = sub["amyloid_gold_pos"].values.astype(np.float32)
    N = len(y)
    rids_c2 = sub["RID"].values
    n_shared_rids = (
        int(len(set(rids_c2).intersection(set(cohort1_rids)))) if cohort1_rids is not None else 90
    )
    strict_90d_mask = (sub["c2_blood_within_90d_bl"].fillna(0).values == 1)
    bl_amyloid_mask = (sub["is_same_visit_baseline_amyloid_reference"].fillna(0).values == 1)
    strict_176_mask = strict_90d_mask & bl_amyloid_mask
    delta_days = (
        pd.to_datetime(sub["plasma_ptau217_first_examdate"], errors="coerce")
        - pd.to_datetime(sub["EXAMDATE"], errors="coerce")
    ).dt.days

    macro_cols = [
        "Ventricles_ICV_ratio",
        "Hippocampus_ICV_ratio",
        "WholeBrain_ICV_ratio",
        "Entorhinal_ICV_ratio",
        "Fusiform_ICV_ratio",
        "MidTemp_ICV_ratio",
    ]
    all_st_cols = sorted([c for c in sub.columns if c.startswith("FSX_ST")])
    mri_cols_all = macro_cols + all_st_cols
    plasma_cols = ["pT217_F", "log_pT217_F", "GFAP_Q", "log_GFAP_Q", "ptau217_gfap_ratio"]
    fdg_cols = ["FDG_suvr"]
    clin_cols = [
        "APOE4",
        "MMSE",
        "ADAS13",
        "CDRSB",
        "LDELTOTAL",
        "RAVLT_immediate",
        "FAQ",
        "AGE",
        "PTGENDER_num",
    ]

    X_mri_full = sub[mri_cols_all].values.astype(np.float32)
    X_plasma = sub[plasma_cols].values.astype(np.float32)
    X_fdg = sub[fdg_cols].values.astype(np.float32)
    non_bl_fdg = (sub["FDG_is_bl_visit"].fillna(0).values != 1)
    X_fdg[non_bl_fdg, :] = np.nan
    X_clin = sub[clin_cols].values.astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_to_eval = [
        "DeepFuzzyTorch",
        "RCML",
        "QMF",
        "DrFuse",
        "EvidentialDL",
        "GatedAttention",
        "EarlyConcat",
    ]
    all_triage_models = models_to_eval + ["PlasmaOnly"]

    oof_p = {m: np.zeros(N, dtype=np.float32) for m in models_to_eval}
    oof_hes = {m: np.zeros(N, dtype=np.float32) for m in models_to_eval}
    oof_conf = {m: np.zeros(N, dtype=np.float32) for m in models_to_eval}
    oof_keep_comp = {m: np.ones(N, dtype=bool) for m in all_triage_models}
    oof_keep_hes = {m: np.ones(N, dtype=bool) for m in all_triage_models}
    oof_plasma_mri_conf = np.zeros(N, dtype=np.float32)
    oof_mod_mus = np.zeros((N, 4), dtype=np.float32)
    oof_mod_nus = np.zeros((N, 4), dtype=np.float32)
    oof_mod_pis = np.zeros((N, 4), dtype=np.float32)
    c2_fold_states = {}
    oof_plasma_only_p = np.zeros(N, dtype=np.float32)
    fold_ids_c2 = np.zeros(N, dtype=np.int32)
    c2_fold_sets: List[set] = []
    optimal_k_c2 = 24

    for fold_idx, (tr_i, va_i) in enumerate(skf.split(X_mri_full, y)):
        fold_ids_c2[va_i] = fold_idx
        # Strictly in-fold FuzzyRoughFeatureReducer over all 320 MRI ROIs (replaces alphabetical slicing!)
        X_mri_tr_all_s, _, _, _ = fit_strict_preprocessor([X_mri_full[tr_i]], [X_mri_full[va_i]])
        red_c2 = FuzzyRoughFeatureReducer(
            max_features=optimal_k_c2,
            sigma=0.85,
            max_samples=4096,
            n_null_permutations=8,
            n_stability_bootstraps=5,
            seed=100 + fold_idx,
        )
        red_c2.fit(X_mri_tr_all_s[0], y[tr_i], feature_names=mri_cols_all, device=str(device))
        sel_c2 = red_c2.selected_indices_[:optimal_k_c2]
        c2_fold_sets.append(set(red_c2.selected_names_[:optimal_k_c2]))

        X_tr_raw = [X_mri_full[tr_i][:, sel_c2], X_plasma[tr_i], X_fdg[tr_i], X_clin[tr_i]]
        X_va_raw = [X_mri_full[va_i][:, sel_c2], X_plasma[va_i], X_fdg[va_i], X_clin[va_i]]
        X_tr_s, X_va_s, m_tr, m_va = fit_strict_preprocessor(X_tr_raw, X_va_raw)
        tr_t = [torch.tensor(x, dtype=torch.float32) for x in X_tr_s]
        tr_m = torch.tensor(m_tr, dtype=torch.float32)
        va_t = [torch.tensor(x, dtype=torch.float32) for x in X_va_s]
        va_m = torch.tensor(m_va, dtype=torch.float32)

        # Out-of-fold unimodal Plasma pTau217 classifier + strictly in-fold 80th-percentile referral cutoff
        clf_plasma = LogisticRegression(C=1.0, max_iter=300)
        clf_plasma.fit(X_tr_s[1], y[tr_i])
        p_pl_tr = clf_plasma.predict_proba(X_tr_s[1])[:, 1]
        p_pl_va = clf_plasma.predict_proba(X_va_s[1])[:, 1]
        oof_plasma_only_p[va_i] = p_pl_va
        unc_pl_tr = 1.0 - 2.0 * np.abs(p_pl_tr - 0.5)
        unc_pl_va = 1.0 - 2.0 * np.abs(p_pl_va - 0.5)
        pl_thr80 = float(np.percentile(unc_pl_tr, 80.0))
        oof_keep_comp["PlasmaOnly"][va_i] = unc_pl_va <= pl_thr80
        oof_keep_hes["PlasmaOnly"][va_i] = unc_pl_va <= pl_thr80

        for m_name in models_to_eval:
            mod = train_deepfuzzy_or_baseline(
                m_name,
                X_tr_s,
                m_tr,
                y[tr_i],
                epochs=50,
                lr=2.5e-3,
                embed_dim=32,
                seed=100 + fold_idx,
                device=device,
            )
            with torch.no_grad():
                out_tr = mod(tr_t, modality_masks=tr_m)
                hes_tr = out_tr["hesitancy"].view(-1).numpy()
                conf_tr = out_tr["conflict"].view(-1).numpy()
                comp_tr = 0.5 * hes_tr + 0.5 * conf_tr

                out = mod(va_t, modality_masks=va_m)
                hes_va = out["hesitancy"].view(-1).numpy()
                conf_va = out["conflict"].view(-1).numpy()
                comp_va = 0.5 * hes_va + 0.5 * conf_va

                oof_p[m_name][va_i] = out["prob"].view(-1).numpy()
                oof_hes[m_name][va_i] = hes_va
                oof_conf[m_name][va_i] = conf_va

                # Symmetric 80th-percentile referral thresholds calibrated strictly on training fold `tr_i`
                thr_comp_80 = float(np.percentile(comp_tr, 80.0))
                thr_hes_80 = float(np.percentile(hes_tr, 80.0))
                oof_keep_comp[m_name][va_i] = comp_va <= thr_comp_80
                oof_keep_hes[m_name][va_i] = hes_va <= thr_hes_80

                if m_name == "DeepFuzzyTorch":
                    c2_fold_states[f"fold_{fold_idx}"] = {
                        "selected_mri_indices": [int(i) for i in sel_c2],
                        "state_dict": {k: v.cpu() for k, v in mod.state_dict().items()},
                    }
                    for m_i in range(4):
                        oof_mod_mus[va_i, m_i] = out["modality_mus"][m_i].view(-1).numpy()
                        oof_mod_nus[va_i, m_i] = out["modality_nus"][m_i].view(-1).numpy()
                        oof_mod_pis[va_i, m_i] = out["modality_pis"][m_i].view(-1).numpy()
                    mu_pl = out["modality_mus"][1].view(-1).numpy()
                    nu_br = (
                        out["modality_nus"][0].view(-1).numpy()
                        + out["modality_nus"][2].view(-1).numpy()
                        + out["modality_nus"][3].view(-1).numpy()
                    ) / 3.0
                    oof_plasma_mri_conf[va_i] = mu_pl * nu_br

    c2_rough_jaccard = pairwise_jaccard_summary(c2_fold_sets)
    cohort2_metrics = {}
    cohort2_strict_90d_metrics = {}
    cohort2_strict_176_concurrent_metrics = {}
    for m in models_to_eval:
        res_m = compute_binary_metrics(oof_p[m], y, uncertainty=oof_hes[m])
        _, lo_m, hi_m = bootstrap_confidence_interval(oof_p[m], y, "auroc", n_bootstraps=1000)
        res_m["auroc_ci"] = [lo_m, hi_m]
        res_m["p_value_vs_ours"] = (
            1.0
            if m == "DeepFuzzyTorch"
            else paired_bootstrap_pvalue(oof_p["DeepFuzzyTorch"], oof_p[m], y, "auroc", n_bootstraps=1000)
        )
        cohort2_metrics[m] = res_m

        res_90 = compute_binary_metrics(
            oof_p[m][strict_90d_mask], y[strict_90d_mask], uncertainty=oof_hes[m][strict_90d_mask]
        )
        _, lo_90, hi_90 = bootstrap_confidence_interval(
            oof_p[m][strict_90d_mask], y[strict_90d_mask], "auroc", n_bootstraps=1000
        )
        res_90["auroc_ci"] = [lo_90, hi_90]
        res_90["p_value_vs_ours"] = (
            1.0
            if m == "DeepFuzzyTorch"
            else paired_bootstrap_pvalue(
                oof_p["DeepFuzzyTorch"][strict_90d_mask],
                oof_p[m][strict_90d_mask],
                y[strict_90d_mask],
                "auroc",
                n_bootstraps=1000,
            )
        )
        cohort2_strict_90d_metrics[m] = res_90

        res_176 = compute_binary_metrics(
            oof_p[m][strict_176_mask], y[strict_176_mask], uncertainty=oof_hes[m][strict_176_mask]
        )
        _, lo_176, hi_176 = bootstrap_confidence_interval(
            oof_p[m][strict_176_mask], y[strict_176_mask], "auroc", n_bootstraps=1000
        )
        res_176["auroc_ci"] = [lo_176, hi_176]
        res_176["p_value_vs_ours"] = (
            1.0
            if m == "DeepFuzzyTorch"
            else paired_bootstrap_pvalue(
                oof_p["DeepFuzzyTorch"][strict_176_mask],
                oof_p[m][strict_176_mask],
                y[strict_176_mask],
                "auroc",
                n_bootstraps=1000,
            )
        )
        cohort2_strict_176_concurrent_metrics[m] = res_176

    attach_bh_fdr_to_table(cohort2_metrics)
    attach_bh_fdr_to_table(cohort2_strict_90d_metrics)
    attach_bh_fdr_to_table(cohort2_strict_176_concurrent_metrics)

    res_pl_uni = compute_binary_metrics(oof_plasma_only_p, y)
    _, lo_pl, hi_pl = bootstrap_confidence_interval(oof_plasma_only_p, y, "auroc", n_bootstraps=1000)
    res_pl_uni["auroc_ci"] = [lo_pl, hi_pl]
    res_pl_uni["p_value_vs_ours"] = paired_bootstrap_pvalue(
        oof_p["DeepFuzzyTorch"], oof_plasma_only_p, y, "auroc", n_bootstraps=1000
    )
    cohort2_metrics["UnimodalPlasma_pTau217"] = res_pl_uni

    # 3 explicit renal status subgroups:
    egfr = sub["eGFR_bl"].values
    ckd_mask = (~np.isnan(egfr)) & (egfr < 60.0)        # n = 22 (6 A-, 16 A+)
    norm_mask = (~np.isnan(egfr)) & (egfr >= 60.0)      # n = 73 (22 A-, 51 A+)
    unmeas_mask = np.isnan(egfr)                        # n = 194 (121 A-, 73 A+)

    ckd_aneg = ckd_mask & (y == 0)
    norm_aneg = norm_mask & (y == 0)

    conf_ckd_aneg = float(np.mean(oof_plasma_mri_conf[ckd_aneg])) if np.any(ckd_aneg) else 0.0
    conf_norm_aneg = float(np.mean(oof_plasma_mri_conf[norm_aneg])) if np.any(norm_aneg) else 1e-4
    conflict_ratio = float(conf_ckd_aneg / max(1e-6, conf_norm_aneg))

    if np.sum(ckd_aneg) >= 2 and np.sum(norm_aneg) >= 2:
        _, mw_pval = mannwhitneyu(
            oof_plasma_mri_conf[ckd_aneg], oof_plasma_mri_conf[norm_aneg], alternative="two-sided"
        )
        mw_pval = float(mw_pval)
    else:
        mw_pval = 1.0

    def _eval_infold_triage_bars(
        probs_arr: np.ndarray, keep_mask_full: np.ndarray
    ) -> Tuple[List[float], int, Dict[str, object]]:
        corr = (probs_arr >= 0.5) == (y == 1)
        k_norm, n_norm = int(np.sum(corr[norm_mask])), int(np.sum(norm_mask))
        k_unm, n_unm = int(np.sum(corr[unmeas_mask])), int(np.sum(unmeas_mask))
        k_ckd, n_ckd = int(np.sum(corr[ckd_mask])), int(np.sum(ckd_mask))

        ckd_keep = keep_mask_full[ckd_mask]
        if np.sum(ckd_keep) == 0:
            ckd_keep = np.ones(n_ckd, dtype=bool)
        k_ret = int(np.sum(corr[ckd_mask][ckd_keep]))
        n_ret = int(np.sum(ckd_keep))

        bars = [
            float(k_norm / max(1, n_norm)),
            float(k_unm / max(1, n_unm)),
            float(k_ckd / max(1, n_ckd)),
            float(k_ret / max(1, n_ret)),
        ]
        detail = {
            "normal_k_n": f"{k_norm}/{n_norm}",
            "normal_ci": clopper_pearson_ci(k_norm, n_norm),
            "unmeasured_k_n": f"{k_unm}/{n_unm}",
            "unmeasured_ci": clopper_pearson_ci(k_unm, n_unm),
            "ckd_raw_k_n": f"{k_ckd}/{n_ckd}",
            "ckd_raw_ci": clopper_pearson_ci(k_ckd, n_ckd),
            "ckd_post_triage_k_n": f"{k_ret}/{n_ret}",
            "ckd_post_triage_ci": clopper_pearson_ci(k_ret, n_ret),
            "n_ckd_retained": n_ret,
            "acc_bars": [round(x, 4) for x in bars],
        }
        return bars, n_ret, detail

    symmetric_composite_triage: Dict[str, object] = {}
    symmetric_hes_only_triage: Dict[str, object] = {}
    comp_bars_map: Dict[str, List[float]] = {}
    hes_bars_map: Dict[str, List[float]] = {}
    for tm in all_triage_models:
        p_vec = oof_plasma_only_p if tm == "PlasmaOnly" else oof_p[tm]
        b_c, _, det_c = _eval_infold_triage_bars(p_vec, oof_keep_comp[tm])
        b_h, _, det_h = _eval_infold_triage_bars(p_vec, oof_keep_hes[tm])
        comp_bars_map[tm] = b_c
        hes_bars_map[tm] = b_h
        symmetric_composite_triage[tm] = det_c
        symmetric_hes_only_triage[tm] = det_h

    n_ckd_retained = int(symmetric_composite_triage["DeepFuzzyTorch"]["n_ckd_retained"])  # type: ignore[index]

    print(
        f"  [Cohort 2 pTau217 N={N} (90d N={int(np.sum(strict_90d_mask))}, 176-Conc AUROC={cohort2_strict_176_concurrent_metrics['DeepFuzzyTorch']['auroc']:.4f})] "
        f"DeepFuzzyTorch AUROC={cohort2_metrics['DeepFuzzyTorch']['auroc']:.4f} | RCML={cohort2_metrics['RCML']['auroc']:.4f} | "
        f"In-Fold MRI Jaccard={c2_rough_jaccard['mean']:.4f} | CKD Post-Triage={symmetric_composite_triage['DeepFuzzyTorch']['ckd_post_triage_k_n']}"
    )
    ckd_subgroup_dict: Dict[str, object] = {
        "triage_threshold_protocol": (
            "Symmetric 80th-percentile referral threshold calibrated strictly on training fold `tr_i` "
            "per CV fold (evaluated identically under both `0.5*hes + 0.5*conf` composite and `hes_only` scores)"
        ),
        "conflict_metric_definition": "mu_Plasma * mean(nu_MRI, nu_FDG, nu_Clin)",
        "n_ckd_measured": int(np.sum(ckd_mask)),
        "n_ckd_abeta_neg": int(np.sum(ckd_aneg)),
        "n_ckd_abeta_pos": int(np.sum(ckd_mask & (y == 1))),
        "n_ckd_retained_post_triage": n_ckd_retained,
        "n_normal_measured": int(np.sum(norm_mask)),
        "n_normal_abeta_neg": int(np.sum(norm_aneg)),
        "n_normal_abeta_pos": int(np.sum(norm_mask & (y == 1))),
        "n_unmeasured_egfr": int(np.sum(unmeas_mask)),
        "n_unmeasured_abeta_neg": int(np.sum(unmeas_mask & (y == 0))),
        "n_unmeasured_abeta_pos": int(np.sum(unmeas_mask & (y == 1))),
        "conf_ckd_aneg": conf_ckd_aneg,
        "conf_norm_aneg": conf_norm_aneg,
        "conflict_ratio": conflict_ratio,
        "mannwhitney_p": mw_pval,
        "categories": [
            r"Normal Renal" + "\n" + rf"($e\mathrm{{GFR}} \geq 60$, n={int(np.sum(norm_mask))})",
            r"Unmeasured" + "\n" + rf"($e\mathrm{{GFR}}$ NA, n={int(np.sum(unmeas_mask))})",
            r"CKD Confounded" + "\n" + rf"($e\mathrm{{GFR}} < 60$, Raw, n={int(np.sum(ckd_mask))})",
            r"CKD Confounded" + "\n" + rf"(In-Fold P80 Triage, n={n_ckd_retained}/{int(np.sum(ckd_mask))})",
        ],
        "symmetric_composite_triage": symmetric_composite_triage,
        "symmetric_hes_only_triage": symmetric_hes_only_triage,
    }
    for tm in all_triage_models:
        ckd_subgroup_dict[f"{tm}_spec"] = comp_bars_map[tm]
        ckd_subgroup_dict[f"{tm}_hes_only_spec"] = hes_bars_map[tm]

    return {
        "n_total": int(N),
        "n_abeta_neg": int(np.sum(y == 0)),
        "n_abeta_pos": int(np.sum(y == 1)),
        "n_shared_rids_with_cohort1": n_shared_rids,
        "n_strict_90d_window": int(np.sum(strict_90d_mask)),
        "n_blood_within_90d_bl": int(np.sum(strict_90d_mask)),
        "n_blood_gt_90d_bl": int(np.sum(delta_days > 90)),
        "max_days_blood_after_bl": int(delta_days.max()),
        "n_baseline_amyloid_ref": int(np.sum(bl_amyloid_mask)),
        "n_post_baseline_amyloid_ref": int(np.sum(~bl_amyloid_mask)),
        "n_strict_90d_and_bl_amyloid": int(np.sum(strict_176_mask)),
        "mri_rough_reduct_jaccard": c2_rough_jaccard,
        "cohort2_metrics": cohort2_metrics,
        "cohort2_strict_90d_metrics": cohort2_strict_90d_metrics,
        "cohort2_strict_176_concurrent_metrics": cohort2_strict_176_concurrent_metrics,
        "ckd_subgroup": ckd_subgroup_dict,
        "oof_arrays": {
            "y": y,
            "fold_ids": fold_ids_c2,
            "probs": oof_p,
            "hesitancy": oof_hes,
            "conflict": oof_conf,
            "modality_mus": oof_mod_mus,
            "modality_nus": oof_mod_nus,
            "modality_pis": oof_mod_pis,
            "directional_plasma_conflict": oof_plasma_mri_conf,
            "egfr_bl": sub["eGFR_bl"].values.astype(np.float32),
            "fold_states": c2_fold_states,
        },
    }


def _compute_km_curve(times: np.ndarray, events: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Computes Kaplan-Meier survival probability S(t) over `t_grid`."""
    surv = np.ones_like(t_grid, dtype=np.float64)
    cur_s = 1.0
    event_times = np.sort(np.unique(times[events == 1]))
    for i, t_pt in enumerate(t_grid):
        for et in event_times:
            if et <= t_pt and (i == 0 or et > t_grid[i - 1]):
                at_risk = np.sum(times >= et)
                d_events = np.sum((times == et) & (events == 1))
                if at_risk > 0:
                    cur_s *= 1.0 - (d_events / float(at_risk))
        surv[i] = cur_s
    return surv


def _compute_c_index(risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """Vectorized Harrell's Concordance Index (C-index)."""
    ev_idx = np.where(events == 1)[0]
    if len(ev_idx) == 0:
        return 0.5
    t_ev = times[ev_idx][:, None]
    r_ev = risk_scores[ev_idx][:, None]
    valid_pairs = times[None, :] > t_ev
    permissible = float(np.sum(valid_pairs))
    if permissible <= 0.0:
        return 0.5
    concordant = np.sum(valid_pairs & (r_ev > risk_scores[None, :])) + 0.5 * np.sum(
        valid_pairs & (r_ev == risk_scores[None, :])
    )
    return float(concordant / permissible)


def run_track3_oasis3_external_and_survival(
    adni_data: Dict[str, object],
    adni_dir: str,
    oasis_dx_path: str,
    device: torch.device,
) -> Dict[str, object]:
    """
    Track 3: Longitudinal Incident Dementia Conversion (ADNI N=1,017) & 5-Seed Zero-Shot External
    Cross-Cohort Diagnostic Generalization (ADNI N=1,137 -> External OASIS-3 N=1,026).
    """
    print(
        "\n[Track 3] Running Track 3A (In-Fold Reduct + In-Fold Tertile ADNI Longitudinal N=1,017) & "
        "Track 3B (5-Seed Zero-Shot External OASIS-3 N=1,026)..."
    )
    df_adni_all = adni_data["df_full"].copy()
    df_adni_1137 = adni_data["df_amyloid"].copy()

    # =========================================================================
    # Track 3A: Dated ADNI Baseline Dementia-Free Incident Conversion (N=1,017)
    # =========================================================================
    zero_fu_mask = (df_adni_all["DX_bl"] != "AD") & (df_adni_all["max_followup_months"].fillna(0) <= 0)
    n_zero_fu_excluded = int(np.sum(zero_fu_mask))
    df_long = (
        df_adni_all[(df_adni_all["DX_bl"] != "AD") & (df_adni_all["max_followup_months"].fillna(0) > 0)]
        .copy()
        .reset_index(drop=True)
    )
    n_shared_long_cohort1 = int(
        len(set(df_long["RID"].values).intersection(set(df_adni_1137["RID"].values)))
    )

    y_long = df_long["event_dementia"].fillna(0).values.astype(np.float32)
    t_long_months = np.where(
        y_long == 1,
        df_long["time_dementia_months"].values,
        df_long["max_followup_months"].values,
    ).astype(np.float32)
    t_long_years = (t_long_months / 12.0).astype(np.float32)
    w_long = np.where(y_long == 1, 1.0, np.minimum(1.0, t_long_months / 48.0)).astype(np.float32)

    mask_60m = ((y_long == 1) & (t_long_months <= 60.0)) | (t_long_months >= 60.0)
    y_60m = ((y_long == 1) & (t_long_months <= 60.0)).astype(np.float32)

    macro_cols = [
        "Ventricles_ICV_ratio",
        "Hippocampus_ICV_ratio",
        "WholeBrain_ICV_ratio",
        "Entorhinal_ICV_ratio",
        "Fusiform_ICV_ratio",
        "MidTemp_ICV_ratio",
        "ICV",
    ]
    all_st_cols_long = sorted([c for c in df_long.columns if c.startswith("FSX_ST")])
    mri_cols_long_all = macro_cols + all_st_cols_long
    X_long_mri_full = df_long[mri_cols_long_all].values.astype(np.float32)
    X_long_pl = df_long[["plasma_ptau181_bl", "log_plasma_ptau181_bl"]].values.astype(np.float32)
    X_long_pl[df_long["blood_is_bl_visit"].fillna(0).values != 1] = np.nan
    X_long_fdg = df_long[["FDG_suvr"]].values.astype(np.float32)
    X_long_fdg[df_long["FDG_is_bl_visit"].fillna(0).values != 1] = np.nan
    X_long_cl = df_long[
        [
            "APOE4",
            "MMSE",
            "ADAS13",
            "CDRSB",
            "LDELTOTAL",
            "RAVLT_immediate",
            "FAQ",
            "AGE",
            "PTEDUCAT",
            "PTGENDER_num",
            "ADNI_MEM",
            "ADNI_EF",
        ]
    ].values.astype(np.float32)

    N_long = len(y_long)
    models_t3 = [
        "DeepFuzzyTorch",
        "RCML",
        "QMF",
        "DrFuse",
        "EvidentialDL",
        "GatedAttention",
        "EarlyConcat",
    ]
    long_oof_p = {m: np.zeros(N_long, dtype=np.float32) for m in models_t3}
    long_oof_h = {m: np.zeros(N_long, dtype=np.float32) for m in models_t3}
    long_fold_ids = np.zeros(N_long, dtype=np.int32)
    oof_tertile_tier = np.zeros(N_long, dtype=np.int32)
    fold_q33_list: List[float] = []
    fold_q67_list: List[float] = []
    t3_mri_fold_sets: List[set] = []
    optimal_k_t3 = 24

    skf_long = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for f_i, (tr_i, va_i) in enumerate(skf_long.split(X_long_mri_full, y_long)):
        long_fold_ids[va_i] = f_i
        # Strictly in-fold FuzzyRoughFeatureReducer over all 321 MRI ROIs (eliminates alphabetical slicing)
        X_mri_tr_all_s, _, _, _ = fit_strict_preprocessor(
            [X_long_mri_full[tr_i]], [X_long_mri_full[va_i]]
        )
        red_t3 = FuzzyRoughFeatureReducer(
            max_features=optimal_k_t3,
            sigma=0.85,
            max_samples=4096,
            n_null_permutations=8,
            n_stability_bootstraps=5,
            seed=300 + f_i,
        )
        red_t3.fit(X_mri_tr_all_s[0], y_long[tr_i], feature_names=mri_cols_long_all, device=str(device))
        sel_t3 = red_t3.selected_indices_[:optimal_k_t3]
        t3_mri_fold_sets.append(set(red_t3.selected_names_[:optimal_k_t3]))

        X_tr_s, X_va_s, m_tr, m_va = fit_strict_preprocessor(
            [X_long_mri_full[tr_i][:, sel_t3], X_long_pl[tr_i], X_long_fdg[tr_i], X_long_cl[tr_i]],
            [X_long_mri_full[va_i][:, sel_t3], X_long_pl[va_i], X_long_fdg[va_i], X_long_cl[va_i]],
        )
        tr_t = [torch.tensor(x, dtype=torch.float32) for x in X_tr_s]
        tr_m = torch.tensor(m_tr, dtype=torch.float32)
        va_t = [torch.tensor(x, dtype=torch.float32) for x in X_va_s]
        va_m = torch.tensor(m_va, dtype=torch.float32)

        for m_name in models_t3:
            mod = train_deepfuzzy_or_baseline(
                m_name,
                X_tr_s,
                m_tr,
                y_long[tr_i],
                epochs=28,
                lr=1.8e-3,
                embed_dim=32,
                sample_weight=w_long[tr_i],
                seed=42 + f_i,
                device=device,
            )
            with torch.no_grad():
                out = mod(va_t, modality_masks=va_m)
                p_va = out["prob"].view(-1).numpy()
                long_oof_p[m_name][va_i] = p_va
                long_oof_h[m_name][va_i] = out["hesitancy"].view(-1).numpy()
                if m_name == "DeepFuzzyTorch":
                    # Strict in-fold tertile cutoffs ([33.333, 66.667] percentiles on `tr_i` ONLY)
                    p_tr = mod(tr_t, modality_masks=tr_m)["prob"].view(-1).numpy()
                    q33_f, q67_f = np.percentile(p_tr, [100.0 / 3.0, 200.0 / 3.0])
                    fold_q33_list.append(float(q33_f))
                    fold_q67_list.append(float(q67_f))
                    oof_tertile_tier[va_i] = np.where(
                        p_va <= q33_f, 0, np.where(p_va >= q67_f, 2, 1)
                    )

    t3_rough_jaccard = pairwise_jaccard_summary(t3_mri_fold_sets)
    adni_longitudinal_table = {}
    for m_name in models_t3:
        p_arr = long_oof_p[m_name]
        h_arr = long_oof_h[m_name]
        m_res = compute_binary_metrics(p_arr, y_long, uncertainty=h_arr)
        _, lo_full, hi_full = bootstrap_confidence_interval(p_arr, y_long, "auroc", n_bootstraps=1000)
        m_res["auroc_ci"] = [lo_full, hi_full]
        m_res["c_index"] = _compute_c_index(p_arr, t_long_years, y_long)

        p_60 = p_arr[mask_60m]
        h_60 = h_arr[mask_60m]
        y_60 = y_60m[mask_60m]
        res_60 = compute_binary_metrics(p_60, y_60, uncertainty=h_60)
        _, lo_60, hi_60 = bootstrap_confidence_interval(p_60, y_60, "auroc", n_bootstraps=1000)
        m_res["auroc_60m"] = res_60["auroc"]
        m_res["auroc_60m_ci"] = [lo_60, hi_60]
        m_res["auprc_60m"] = res_60["auprc"]
        m_res["brier_60m"] = res_60["brier"]
        m_res["ece_60m"] = res_60["ece"]

        if m_name != "DeepFuzzyTorch":
            m_res["p_value_vs_ours"] = paired_bootstrap_pvalue(
                long_oof_p["DeepFuzzyTorch"], p_arr, y_long, "auroc", n_bootstraps=1000
            )
            m_res["p_value_60m_vs_ours"] = paired_bootstrap_pvalue(
                long_oof_p["DeepFuzzyTorch"][mask_60m], p_60, y_60, "auroc", n_bootstraps=1000
            )
        else:
            m_res["p_value_vs_ours"] = 1.0
            m_res["p_value_60m_vs_ours"] = 1.0

        adni_longitudinal_table[m_name] = m_res

    attach_bh_fdr_to_table(adni_longitudinal_table, "p_value_vs_ours", "q_bh_vs_ours")
    attach_bh_fdr_to_table(adni_longitudinal_table, "p_value_60m_vs_ours", "q_bh_60m_vs_ours")
    for m_name in models_t3:
        m_res = adni_longitudinal_table[m_name]
        print(
            f"  [ADNI Longitudinal N={N_long} (60m N={int(np.sum(mask_60m))})] {m_name:16s}: "
            f"60m-AUROC={m_res['auroc_60m']:.4f}, Full-AUROC={m_res['auroc']:.4f}, "
            f"C-index={m_res['c_index']:.4f}, q_BH(60m)={m_res['q_bh_60m_vs_ours']:.4f}"
        )

    # Strictly out-of-fold 3-tertile Kaplan-Meier survival curves (cutoffs [33.33, 66.67] fit per training fold!)
    low_mask = oof_tertile_tier == 0
    mid_mask = oof_tertile_tier == 1
    high_mask = oof_tertile_tier == 2
    q33_mean = float(np.mean(fold_q33_list))
    q67_mean = float(np.mean(fold_q67_list))

    t_grid = np.linspace(0.0, 10.0, 61)
    surv_low = _compute_km_curve(t_long_years[low_mask], y_long[low_mask], t_grid)
    surv_mid = _compute_km_curve(t_long_years[mid_mask], y_long[mid_mask], t_grid)
    surv_high = _compute_km_curve(t_long_years[high_mask], y_long[high_mask], t_grid)

    chi2_stat, logrank_p = compute_logrank_test(
        t_long_years[high_mask],
        y_long[high_mask],
        t_long_years[low_mask],
        y_long[low_mask],
    )
    chi2_3grp, logrank_3grp_p = compute_multigroup_logrank_test(t_long_years, y_long, oof_tertile_tier)

    survival_vis = {
        "stratification_protocol": "Strict in-fold tertile risk cutoffs ([33.33, 66.67] percentiles fit on `tr_i` per CV fold)",
        "mri_rough_reduct_jaccard": t3_rough_jaccard,
        "n_adni_longitudinal": int(N_long),
        "n_adni_events": int(np.sum(y_long)),
        "n_zero_followup_excluded": n_zero_fu_excluded,
        "n_shared_with_cohort1_1137": n_shared_long_cohort1,
        "n_60m_horizon": int(np.sum(mask_60m)),
        "n_60m_events": int(np.sum(y_60m[mask_60m])),
        "n_60m_controls": int(np.sum(mask_60m) - np.sum(y_60m[mask_60m])),
        "n_60m_late_event_controls": int(np.sum((y_long == 1) & (t_long_months > 60.0))),
        "time_grid": t_grid.tolist(),
        "surv_low_risk": surv_low.tolist(),
        "surv_uncertain": surv_mid.tolist(),
        "surv_high_risk": surv_high.tolist(),
        "n_low": int(np.sum(low_mask)),
        "n_mid": int(np.sum(mid_mask)),
        "n_high": int(np.sum(high_mask)),
        "q33_cutoff": q33_mean,
        "q67_cutoff": q67_mean,
        "per_fold_q33_cutoffs": [round(x, 4) for x in fold_q33_list],
        "per_fold_q67_cutoffs": [round(x, 4) for x in fold_q67_list],
        "q45_cutoff": q33_mean,
        "q72_cutoff": q67_mean,
        "logrank_chi2": chi2_stat,
        "logrank_p": logrank_p,
        "logrank_3group_chi2": chi2_3grp,
        "logrank_3group_p": logrank_3grp_p,
        "c_index": adni_longitudinal_table["DeepFuzzyTorch"]["c_index"],
        "xlabel": "Follow-Up Time in ADNI Longitudinal Cohort (Years)",
        "ylabel": "Dementia-Free Survival Probability $S(t)$",
    }
    print(
        f"  [ADNI In-Fold Tertile Survival N={N_long} (n_low={int(np.sum(low_mask))}, n_mid={int(np.sum(mid_mask))}, n_high={int(np.sum(high_mask))})] "
        f"High-vs-Low Log-Rank chi2={chi2_stat:.2f} (p={logrank_p:.3e}), 3-Group chi2={chi2_3grp:.2f} (p={logrank_3grp_p:.3e})"
    )

    # =========================================================================
    # Track 3B: 5-Seed Zero-Shot External Generalization on OASIS-3 (N=1,026)
    # =========================================================================
    df_oasis = pd.read_csv(os.path.join(adni_dir, "cohort4_oasis3_external.csv")).copy().reset_index(drop=True)
    N_oasis = len(df_oasis)

    df_adni_1137 = df_adni_1137.assign(
        sex_male=(df_adni_1137["PTGENDER"] == "Male").astype(float),
        apoe4_bin=(df_adni_1137["APOE4"] >= 1).astype(float),
    ).copy()
    df_oasis = df_oasis.assign(
        sex_male=(df_oasis["M/F"] == "M").astype(float),
        apoe4_bin=df_oasis["APOE4_carrier"].astype(float),
    ).copy()

    X_adni_mri_ext = df_adni_1137[["Hippocampus", "Hippocampus_ICV_ratio", "ICV"]].values.astype(np.float32)
    X_oasis_mri_ext = df_oasis[
        ["TOTAL_HIPPOCAMPUS_VOLUME", "Hippocampus_ICV_ratio", "IntraCranialVol"]
    ].values.astype(np.float32)

    y_adni_clin_dem = ((df_adni_1137["DX_bl"] == "AD") | (df_adni_1137["CDRSB"] >= 1.5)).astype(np.float32).values
    y_adni_strict_ad = (df_adni_1137["DX_bl"] == "AD").astype(np.float32).values
    y_oasis_dementia = (df_oasis["Group"].astype(str).str.strip() != "NormalCognition").astype(np.float32).values
    y_oasis_cdr05 = df_oasis["abnormal_cdr_or_dementia"].astype(np.float32).values
    uds_d1 = df_oasis["oasis_baseline_demented"].values
    concordant_814_mask = ((y_oasis_dementia == 1) & (uds_d1 == 1.0)) | (
        (y_oasis_dementia == 0) & (uds_d1 == 0.0)
    )

    X_adni_clin_nommse = df_adni_1137[["AGE", "sex_male", "PTEDUCAT", "apoe4_bin"]].values.astype(np.float32)
    X_oasis_clin_nommse = df_oasis[["Age", "sex_male", "Education", "apoe4_bin"]].values.astype(np.float32)
    X_tr_nommse, X_oa_nommse, m_tr_nommse, m_oa_nommse = fit_strict_preprocessor(
        [X_adni_mri_ext, X_adni_clin_nommse], [X_oasis_mri_ext, X_oasis_clin_nommse]
    )
    oa_t_nommse = [torch.tensor(x, dtype=torch.float32) for x in X_oa_nommse]
    oa_m_nommse = torch.tensor(m_oa_nommse, dtype=torch.float32)

    X_adni_clin_mmse = df_adni_1137[["AGE", "sex_male", "PTEDUCAT", "apoe4_bin", "MMSE"]].values.astype(np.float32)
    X_oasis_clin_mmse = df_oasis[["Age", "sex_male", "Education", "apoe4_bin", "mmse"]].values.astype(np.float32)
    X_tr_mmse, X_oa_mmse, m_tr_mmse, m_oa_mmse = fit_strict_preprocessor(
        [X_adni_mri_ext, X_adni_clin_mmse], [X_oasis_mri_ext, X_oasis_clin_mmse]
    )
    oa_t_mmse = [torch.tensor(x, dtype=torch.float32) for x in X_oa_mmse]
    oa_m_mmse = torch.tensor(m_oa_mmse, dtype=torch.float32)

    seeds_5 = [42, 43, 44, 45, 46]
    oasis3_external_table = {}
    oasis3_strict_ad_source_table = {}
    oasis3_uds_d1_concordant_814_table = {}
    oa_probs_nommse = {}
    oa_hes_nommse = {}
    oa_probs_mmse = {}
    oa_hes_mmse = {}
    oa_strict_ad_probs = {}

    for m_name in models_t3:
        p_nm_seeds, h_nm_seeds = [], []
        p_mm_seeds, h_mm_seeds = [], []
        p_ad_seeds, h_ad_seeds = [], []
        seed_nm_aurocs, seed_cdr_nm_aurocs, seed_cdr_mm_aurocs, seed_strict_ad_aurocs = [], [], [], []

        for sd in seeds_5:
            mod_nommse = train_deepfuzzy_or_baseline(
                m_name, X_tr_nommse, m_tr_nommse, y_adni_clin_dem, epochs=30, lr=1.8e-3, embed_dim=32, seed=sd, device=device
            )
            mod_mmse = train_deepfuzzy_or_baseline(
                m_name, X_tr_mmse, m_tr_mmse, y_adni_clin_dem, epochs=30, lr=1.8e-3, embed_dim=32, seed=sd, device=device
            )
            mod_strict_ad = train_deepfuzzy_or_baseline(
                m_name, X_tr_nommse, m_tr_nommse, y_adni_strict_ad, epochs=30, lr=1.8e-3, embed_dim=32, seed=sd, device=device
            )
            with torch.no_grad():
                out_nm = mod_nommse(oa_t_nommse, modality_masks=oa_m_nommse)
                p_nm_s = out_nm["prob"].view(-1).numpy()
                h_nm_s = out_nm["hesitancy"].view(-1).numpy()
                p_nm_seeds.append(p_nm_s)
                h_nm_seeds.append(h_nm_s)

                out_mm = mod_mmse(oa_t_mmse, modality_masks=oa_m_mmse)
                p_mm_s = out_mm["prob"].view(-1).numpy()
                h_mm_s = out_mm["hesitancy"].view(-1).numpy()
                p_mm_seeds.append(p_mm_s)
                h_mm_seeds.append(h_mm_s)

                out_ad = mod_strict_ad(oa_t_nommse, modality_masks=oa_m_nommse)
                p_ad_s = out_ad["prob"].view(-1).numpy()
                h_ad_s = out_ad["hesitancy"].view(-1).numpy()
                p_ad_seeds.append(p_ad_s)
                h_ad_seeds.append(h_ad_s)

            seed_nm_aurocs.append(float(compute_binary_metrics(p_nm_s, y_oasis_dementia)["auroc"]))
            seed_cdr_nm_aurocs.append(float(compute_binary_metrics(p_nm_s, y_oasis_cdr05)["auroc"]))
            seed_cdr_mm_aurocs.append(float(compute_binary_metrics(p_mm_s, y_oasis_cdr05)["auroc"]))
            seed_strict_ad_aurocs.append(float(compute_binary_metrics(p_ad_s, y_oasis_dementia)["auroc"]))

        p_nm = np.mean(np.stack(p_nm_seeds, axis=0), axis=0).astype(np.float32)
        h_nm = np.mean(np.stack(h_nm_seeds, axis=0), axis=0).astype(np.float32)
        p_mm = np.mean(np.stack(p_mm_seeds, axis=0), axis=0).astype(np.float32)
        h_mm = np.mean(np.stack(h_mm_seeds, axis=0), axis=0).astype(np.float32)
        p_ad = np.mean(np.stack(p_ad_seeds, axis=0), axis=0).astype(np.float32)
        h_ad = np.mean(np.stack(h_ad_seeds, axis=0), axis=0).astype(np.float32)

        oa_probs_nommse[m_name] = p_nm
        oa_hes_nommse[m_name] = h_nm
        oa_probs_mmse[m_name] = p_mm
        oa_hes_mmse[m_name] = h_mm
        oa_strict_ad_probs[m_name] = p_ad

        res_ext = compute_binary_metrics(p_nm, y_oasis_dementia, uncertainty=h_nm)
        _, lo_nm, hi_nm = bootstrap_confidence_interval(p_nm, y_oasis_dementia, "auroc", n_bootstraps=1000)
        res_ext["auroc_ci"] = [lo_nm, hi_nm]
        res_ext["auroc_5seed_mean"] = round(float(np.mean(seed_nm_aurocs)), 4)
        res_ext["auroc_5seed_std"] = round(float(np.std(seed_nm_aurocs, ddof=1)), 4)

        res_cdr_nm = compute_binary_metrics(p_nm, y_oasis_cdr05, uncertainty=h_nm)
        _, lo_cnm, hi_cnm = bootstrap_confidence_interval(p_nm, y_oasis_cdr05, "auroc", n_bootstraps=1000)
        res_ext["cdr05_nommse_auroc"] = res_cdr_nm["auroc"]
        res_ext["cdr05_nommse_auroc_ci"] = [lo_cnm, hi_cnm]
        res_ext["cdr05_nommse_5seed_mean"] = round(float(np.mean(seed_cdr_nm_aurocs)), 4)
        res_ext["cdr05_nommse_5seed_std"] = round(float(np.std(seed_cdr_nm_aurocs, ddof=1)), 4)

        res_cdr_mm = compute_binary_metrics(p_mm, y_oasis_cdr05, uncertainty=h_mm)
        _, lo_cmm, hi_cmm = bootstrap_confidence_interval(p_mm, y_oasis_cdr05, "auroc", n_bootstraps=1000)
        res_ext["cdr05_with_mmse_auroc"] = res_cdr_mm["auroc"]
        res_ext["cdr05_with_mmse_auroc_ci"] = [lo_cmm, hi_cmm]
        res_ext["cdr05_with_mmse_5seed_mean"] = round(float(np.mean(seed_cdr_mm_aurocs)), 4)
        res_ext["cdr05_with_mmse_5seed_std"] = round(float(np.std(seed_cdr_mm_aurocs, ddof=1)), 4)
        res_ext["cdr05_with_mmse_ece"] = res_cdr_mm["ece"]
        res_ext["dementia_with_mmse_auroc"] = compute_binary_metrics(p_mm, y_oasis_dementia)["auroc"]
        oasis3_external_table[m_name] = res_ext

        # Strict DX_bl == "AD" source-label sensitivity (N_source_AD = 229/1,137)
        res_sad = compute_binary_metrics(p_ad, y_oasis_dementia, uncertainty=h_ad)
        _, lo_sad, hi_sad = bootstrap_confidence_interval(p_ad, y_oasis_dementia, "auroc", n_bootstraps=1000)
        res_sad["auroc_ci"] = [lo_sad, hi_sad]
        res_sad["auroc_5seed_mean"] = round(float(np.mean(seed_strict_ad_aurocs)), 4)
        res_sad["auroc_5seed_std"] = round(float(np.std(seed_strict_ad_aurocs, ddof=1)), 4)
        oasis3_strict_ad_source_table[m_name] = res_sad

        # UDS-D1 concordant n=814 sensitivity
        res_814 = compute_binary_metrics(
            p_nm[concordant_814_mask], y_oasis_dementia[concordant_814_mask], uncertainty=h_nm[concordant_814_mask]
        )
        _, lo_814, hi_814 = bootstrap_confidence_interval(
            p_nm[concordant_814_mask], y_oasis_dementia[concordant_814_mask], "auroc", n_bootstraps=1000
        )
        res_814["auroc_ci"] = [lo_814, hi_814]
        oasis3_uds_d1_concordant_814_table[m_name] = res_814

        print(
            f"  [External OASIS-3 N={N_oasis} (5-seed)] {m_name:16s}: "
            f"No-MMSE Dementia={res_ext['auroc']:.4f} ({res_ext['auroc_5seed_mean']:.4f}±{res_ext['auroc_5seed_std']:.4f}), "
            f"Strict-AD Source={res_sad['auroc']:.4f}, With-MMSE CDR>=0.5={res_ext['cdr05_with_mmse_auroc']:.4f}"
        )

    for m_name in models_t3:
        if m_name == "DeepFuzzyTorch":
            oasis3_external_table[m_name]["p_value_vs_ours"] = 1.0
            oasis3_external_table[m_name]["p_value_cdr05_nommse_vs_ours"] = 1.0
            oasis3_external_table[m_name]["p_value_cdr05_mmse_vs_ours"] = 1.0
            oasis3_strict_ad_source_table[m_name]["p_value_vs_ours"] = 1.0
            oasis3_uds_d1_concordant_814_table[m_name]["p_value_vs_ours"] = 1.0
        else:
            oasis3_external_table[m_name]["p_value_vs_ours"] = paired_bootstrap_pvalue(
                oa_probs_nommse["DeepFuzzyTorch"], oa_probs_nommse[m_name], y_oasis_dementia, "auroc", n_bootstraps=1000
            )
            oasis3_external_table[m_name]["p_value_cdr05_nommse_vs_ours"] = paired_bootstrap_pvalue(
                oa_probs_nommse["DeepFuzzyTorch"], oa_probs_nommse[m_name], y_oasis_cdr05, "auroc", n_bootstraps=1000
            )
            oasis3_external_table[m_name]["p_value_cdr05_mmse_vs_ours"] = paired_bootstrap_pvalue(
                oa_probs_mmse["DeepFuzzyTorch"], oa_probs_mmse[m_name], y_oasis_cdr05, "auroc", n_bootstraps=1000
            )
            oasis3_strict_ad_source_table[m_name]["p_value_vs_ours"] = paired_bootstrap_pvalue(
                oa_strict_ad_probs["DeepFuzzyTorch"], oa_strict_ad_probs[m_name], y_oasis_dementia, "auroc", n_bootstraps=1000
            )
            oasis3_uds_d1_concordant_814_table[m_name]["p_value_vs_ours"] = paired_bootstrap_pvalue(
                oa_probs_nommse["DeepFuzzyTorch"][concordant_814_mask],
                oa_probs_nommse[m_name][concordant_814_mask],
                y_oasis_dementia[concordant_814_mask],
                "auroc",
                n_bootstraps=1000,
            )

    attach_bh_fdr_to_table(oasis3_external_table, "p_value_vs_ours", "q_bh_vs_ours")
    attach_bh_fdr_to_table(oasis3_external_table, "p_value_cdr05_nommse_vs_ours", "q_bh_cdr05_nommse_vs_ours")
    attach_bh_fdr_to_table(oasis3_external_table, "p_value_cdr05_mmse_vs_ours", "q_bh_cdr05_mmse_vs_ours")
    attach_bh_fdr_to_table(oasis3_strict_ad_source_table)
    attach_bh_fdr_to_table(oasis3_uds_d1_concordant_814_table)

    return {
        "adni_longitudinal_table": adni_longitudinal_table,
        "oasis3_external_table": oasis3_external_table,
        "oasis3_strict_ad_source_table": oasis3_strict_ad_source_table,
        "oasis3_uds_d1_concordant_814_table": oasis3_uds_d1_concordant_814_table,
        "n_oasis_total": int(N_oasis),
        "n_oasis_dementia_events": int(np.sum(y_oasis_dementia)),
        "n_oasis_cdr05_events": int(np.sum(y_oasis_cdr05)),
        "n_oasis_uds_d1_missing": int(np.sum(np.isnan(uds_d1))),
        "n_oasis_uds_d1_concordant": int(np.sum(concordant_814_mask)),
        "n_source_strict_ad_events": int(np.sum(y_adni_strict_ad)),
        "seeds_evaluated": seeds_5,
        "survival_vis": survival_vis,
        "adni_longitudinal_arrays": {
            "y": y_long,
            "y_60m": y_60m,
            "mask_60m": mask_60m.astype(np.int32),
            "time_years": t_long_years,
            "fold_ids": long_fold_ids,
            "tertile_tier": oof_tertile_tier,
            "probs": long_oof_p,
            "hesitancy": long_oof_h,
        },
        "oasis3_arrays": {
            "y_dementia": y_oasis_dementia,
            "y_cdr05": y_oasis_cdr05,
            "probs_nommse": oa_probs_nommse,
            "hesitancy_nommse": oa_hes_nommse,
            "probs_mmse": oa_probs_mmse,
            "hesitancy_mmse": oa_hes_mmse,
        },
    }


def run_track4_tcga_pancancer_multiomics_wsi(
    wsi_dir: str, device: torch.device
) -> Dict[str, object]:
    """
    Track 4: TCGA Pan-Cancer Multi-Omics + Histopathology WSI Benchmark (N=2,946 unique patients/slides
    across 7 cohorts: TCGA-BRCA, TCGA-COAD, TCGA-KIRC, TCGA-KIRP, TCGA-LUAD, TCGA-LUSC, TCGA-UCEC) (R2-C5):
      - Strictly restricts to the N=2,946 unique patients with positive follow-up (`os_m > 0`) and known
        progression/mortality status (`cbio_PFS_STATUS.notna() | os_dead`), excluding 82 records with
        zero/missing follow-up or unknown progression status while alive.
      - Strictly excludes post-diagnosis treatment (`cbio_RADIATION_THERAPY`) so only baseline diagnostic-time
        clinical/staging features (`Age`, `Sex`, `AJCC Pathologic Stage`, `T Stage`, `N Stage`, `Cancer Type`)
        are used as predictors.
      - Uses censoring-aware follow-up maturity sample weights during training and evaluates both composite
        recorded progression/mortality classification and out-of-fold Overall Survival Harrell's C-index.
    """
    print("\n[Track 4] Running TCGA Pan-Cancer 7-Cohort Multi-Omics + UNI2-h WSI Benchmark (N=2,946)...")
    bio_df = pd.read_csv(os.path.join(wsi_dir, "clinical/tcga_precision_oncology_sample_biomarkers.csv"))
    cohorts = ["TCGA-BRCA", "TCGA-COAD", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LUAD", "TCGA-LUSC", "TCGA-UCEC"]

    wsi_raw_list, gen_list, clin_list, y_list, os_time_list, os_event_list, cohort_labels = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    gen_cols = [
        "ANEUPLOIDY_SCORE",
        "FRACTION_GENOME_ALTERED",
        "MSI_SCORE_MANTIS",
        "MSI_SENSOR_SCORE",
        "MUTATION_COUNT",
        "TBL_SCORE",
        "TMB_NONSYNONYMOUS",
        "cbio_BUFFA_HYPOXIA_SCORE",
        "cbio_RAGNUM_HYPOXIA_SCORE",
        "cbio_WINTER_HYPOXIA_SCORE",
    ]

    n_raw_vital_slides = 0
    for c_idx, c_name in enumerate(cohorts):
        meta_p = os.path.join(wsi_dir, f"tcga_uni2h/{c_name}_uni2h_metadata.csv")
        npz_p = os.path.join(wsi_dir, f"tcga_uni2h/{c_name}_uni2h_slides.npz")
        meta = pd.read_csv(meta_p)
        npz = np.load(npz_p, allow_pickle=True)
        w_feats = npz["weighted_feats"].astype(np.float32)  # [N_c, 1536]
        s_feats = npz["std_feats"].astype(np.float32)       # [N_c, 1536]

        merged = meta.merge(bio_df, on="patient_id", how="left")
        valid_vital = merged["vital_status"].isin(["Alive", "Dead"]).values
        merged = merged[valid_vital].reset_index(drop=True)
        w_feats = w_feats[valid_vital]
        s_feats = s_feats[valid_vital]
        n_raw_vital_slides += len(merged)

        os_dead_raw = (merged["vital_status"].values == "Dead")
        os_m_raw = merged["cbio_OS_MONTHS"].values.astype(np.float32)
        fallback_days = np.where(
            os_dead_raw,
            merged["days_to_death"].values.astype(np.float32),
            merged["days_to_last_followup"].values.astype(np.float32),
        )
        os_m_raw = np.where(np.isnan(os_m_raw) | (os_m_raw <= 0), fallback_days / 30.4375, os_m_raw)
        has_pfs_raw = merged["cbio_PFS_STATUS"].notna().values

        # Strictly require positive follow-up (os_m > 0) and known progression/mortality status (R2-C5)
        strict_valid = (~np.isnan(os_m_raw)) & (os_m_raw > 0) & (has_pfs_raw | os_dead_raw)
        merged = merged[strict_valid].reset_index(drop=True)
        w_feats = w_feats[strict_valid]
        s_feats = s_feats[strict_valid]
        os_m = os_m_raw[strict_valid]
        os_dead = os_dead_raw[strict_valid]

        N_c = len(merged)
        wsi_3072 = np.concatenate([w_feats, s_feats], axis=1)  # [N_c, 3072]
        gen_mat = merged[gen_cols].values.astype(np.float32)

        # Encode baseline diagnostic-time clinical & staging variables preserving missing strings as NaN
        # (Strictly excludes post-diagnosis cbio_RADIATION_THERAPY, R2-C5)
        stage_str = merged["ajcc_stage"].fillna("").astype(str).str.strip()
        stage_num = np.where(
            stage_str.str.contains("IV"),
            4.0,
            np.where(
                stage_str.str.contains("III"),
                3.0,
                np.where(
                    stage_str.str.contains("II"),
                    2.0,
                    np.where(stage_str.str.contains("I"), 1.0, np.nan),
                ),
            ),
        )

        t_str = merged["ajcc_t"].fillna("").astype(str).str.strip()
        t_num = np.where(
            t_str.str.contains("T4"),
            4.0,
            np.where(
                t_str.str.contains("T3"),
                3.0,
                np.where(
                    t_str.str.contains("T2"),
                    2.0,
                    np.where(t_str.str.contains("T1|T0|Tis"), 1.0, np.nan),
                ),
            ),
        )

        n_str = merged["ajcc_n"].fillna("").astype(str).str.strip()
        n_num = np.where(
            n_str.str.contains("N2|N3"),
            2.0,
            np.where(
                n_str.str.contains("N1"),
                1.0,
                np.where(n_str.str.contains("N0"), 0.0, np.nan),
            ),
        )

        age_v = merged["age_numeric"].values.astype(np.float32)
        gender_str = merged["gender"].fillna("").astype(str).str.strip().str.lower()
        sex_v = np.where(
            gender_str == "male", 1.0, np.where(gender_str == "female", 0.0, np.nan)
        )

        cohort_oh = np.zeros((N_c, len(cohorts)), dtype=np.float32)
        cohort_oh[:, c_idx] = 1.0

        clin_mat = np.column_stack([age_v, sex_v, stage_num, t_num, n_num, cohort_oh]).astype(np.float32)

        # Composite Oncology Outcome: Mortality (Dead) or Documented Progression (`1:PROGRESSION` in `cbio_PFS_STATUS`)
        pfs_prog = merged["cbio_PFS_STATUS"].fillna("").astype(str).str.contains("1:").values
        y_c = (os_dead | pfs_prog).astype(np.float32)

        wsi_raw_list.append(wsi_3072)
        gen_list.append(gen_mat)
        clin_list.append(clin_mat)
        y_list.append(y_c)
        os_time_list.append(os_m)
        os_event_list.append(os_dead.astype(np.float32))
        cohort_labels.extend([c_name] * N_c)

    X_wsi_raw = np.vstack(wsi_raw_list)  # [N_tot, 3072]
    X_gen = np.vstack(gen_list)
    X_clin = np.vstack(clin_list)
    y_all = np.concatenate(y_list)
    os_time_all = np.concatenate(os_time_list)
    os_event_all = np.concatenate(os_event_list)
    cohort_arr = np.asarray(cohort_labels)
    N_tot = len(y_all)
    n_excluded_tcga = int(n_raw_vital_slides - N_tot)
    # Censoring-aware follow-up maturity sample weights (1.0 for events; min(1.0, os_months/36.0) for censored)
    w_tcga = np.where(y_all == 1, 1.0, np.minimum(1.0, os_time_all / 36.0)).astype(np.float32)
    print(
        f"  Loaded TCGA Pan-Cancer N={N_tot} unique patients/slides across 7 cohorts "
        f"(excluded {n_excluded_tcga} zero-followup/unknown-PFS records) | Event rate={np.mean(y_all):.3f}"
    )

    models_tcga = [
        "DeepFuzzyTorch",
        "RCML",
        "QMF",
        "DrFuse",
        "EvidentialDL",
        "GatedAttention",
        "DempsterShafer",
        "LateAvg",
        "EarlyConcat",
    ]
    oof_p = {m: np.zeros(N_tot, dtype=np.float32) for m in models_tcga}
    oof_h = {m: np.zeros(N_tot, dtype=np.float32) for m in models_tcga}
    fold_models = {m: [] for m in models_tcga}
    fold_val_indices = []
    fold_val_tensors = []
    fold_val_masks = []
    tcga_fold_ids = np.zeros(N_tot, dtype=np.int32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for f_i, (tr_i, va_i) in enumerate(skf.split(X_wsi_raw, y_all)):
        tcga_fold_ids[va_i] = f_i
        # Fit 48-component PCA on 3,072-D UNI2-h WSI features strictly inside training fold (M11)
        wsi_tr_mean = np.mean(X_wsi_raw[tr_i], axis=0, keepdims=True)
        wsi_tr_std = np.std(X_wsi_raw[tr_i], axis=0, keepdims=True) + 1e-5
        pca_wsi = PCA(n_components=48, random_state=42 + f_i)
        X_wsi_tr_pca = pca_wsi.fit_transform((X_wsi_raw[tr_i] - wsi_tr_mean) / wsi_tr_std).astype(np.float32)
        X_wsi_va_pca = pca_wsi.transform((X_wsi_raw[va_i] - wsi_tr_mean) / wsi_tr_std).astype(np.float32)

        X_tr_raw = [X_wsi_tr_pca, X_gen[tr_i], X_clin[tr_i]]
        X_va_raw = [X_wsi_va_pca, X_gen[va_i], X_clin[va_i]]
        X_tr_s, X_va_s, m_tr, m_va = fit_strict_preprocessor(X_tr_raw, X_va_raw)
        va_t = [torch.tensor(x, dtype=torch.float32) for x in X_va_s]
        va_m = torch.tensor(m_va, dtype=torch.float32)

        fold_val_indices.append(va_i)
        fold_val_tensors.append(va_t)
        fold_val_masks.append(va_m)

        for m_name in models_tcga:
            mod = train_deepfuzzy_or_baseline(
                m_name,
                X_tr_s,
                m_tr,
                y_all[tr_i],
                epochs=28,
                lr=1.8e-3,
                embed_dim=40,
                sample_weight=w_tcga[tr_i],
                seed=200 + f_i,
                device=device,
            )
            fold_models[m_name].append(mod)
            with torch.no_grad():
                out = mod(va_t, modality_masks=va_m)
                oof_p[m_name][va_i] = out["prob"].view(-1).numpy()
                oof_h[m_name][va_i] = out["hesitancy"].view(-1).numpy()

    valid_os = (~np.isnan(os_time_all)) & (os_time_all > 0)
    tcga_main_table = {}
    for m_name in models_tcga:
        res = compute_binary_metrics(oof_p[m_name], y_all, uncertainty=oof_h[m_name])
        _, lo, hi = bootstrap_confidence_interval(oof_p[m_name], y_all, "auroc", n_bootstraps=1000)
        res["auroc_ci"] = [lo, hi]
        res["os_c_index"] = _compute_c_index(
            oof_p[m_name][valid_os], os_time_all[valid_os], os_event_all[valid_os]
        )
        if m_name != "DeepFuzzyTorch":
            res["p_value_vs_ours"] = paired_bootstrap_pvalue(
                oof_p["DeepFuzzyTorch"], oof_p[m_name], y_all, "auroc", n_bootstraps=1000
            )
        else:
            res["p_value_vs_ours"] = 1.0
        tcga_main_table[m_name] = res
    attach_bh_fdr_to_table(tcga_main_table, p_key="p_value_vs_ours", q_key="q_bh_vs_ours")
    for m_name in models_tcga:
        res = tcga_main_table[m_name]
        lo, hi = res["auroc_ci"]
        print(
            f"  [TCGA Pan-Cancer N={N_tot}] {m_name:16s}: AUROC={res['auroc']:.4f} [{lo:.4f}, {hi:.4f}], "
            f"OS C-idx={res['os_c_index']:.4f}, ECE={res['ece']:.4f}, q_BH={res['q_bh_vs_ours']:.4g}"
        )

    # Per-cohort AUROC breakdown & cohort sample counts
    tcga_per_cohort = {}
    tcga_cohort_counts = {}
    comb_models_tcga = [
        "DeepFuzzyTorch",
        "RCML",
        "QMF",
        "DrFuse",
        "EvidentialDL",
        "GatedAttention",
        "EarlyConcat",
    ]
    for c_name in cohorts:
        c_msk = cohort_arr == c_name
        tcga_cohort_counts[c_name] = {
            "n_patients": int(np.sum(c_msk)),
            "n_events": int(np.sum(y_all[c_msk])),
        }
        tcga_per_cohort[c_name] = {
            m_name: compute_binary_metrics(oof_p[m_name][c_msk], y_all[c_msk])["auroc"]
            for m_name in comb_models_tcga
        }

    # Evaluate 2^3 - 1 = 7 Missing-Modality Combinations on TCGA
    full_t = [torch.zeros(N_tot, fold_val_tensors[0][m].shape[1]) for m in range(3)]
    full_m = torch.zeros(N_tot, 3)
    for f_i, va_i in enumerate(fold_val_indices):
        for m in range(3):
            full_t[m][va_i] = fold_val_tensors[f_i][m]
        full_m[va_i] = fold_val_masks[f_i]

    tcga_bench = CombinatorialMissingModalityBenchmark(["WSI", "Genomics", "Clinical"])
    tcga_comb_results = {}
    for m_name in comb_models_tcga:
        def _pred_tcga(xs: List[torch.Tensor], msk: torch.Tensor, mk=m_name):
            p_out = torch.zeros(N_tot, 1)
            h_out = torch.zeros(N_tot, 1)
            for f_i, va_i in enumerate(fold_val_indices):
                sub_x = [t[va_i] for t in xs]
                sub_m = msk[va_i]
                out_f = fold_models[mk][f_i](sub_x, modality_masks=sub_m)
                p_out[va_i] = out_f["prob"].detach().cpu()
                h_out[va_i] = out_f["hesitancy"].detach().cpu()
            return {"prob": p_out, "hesitancy": h_out}

        tcga_comb_results[m_name] = tcga_bench.evaluate_all_subsets(
            predict_fn=_pred_tcga, modality_tensors=full_t, natural_masks=full_m, targets=y_all
        )
        tcga_main_table[m_name]["macro_7subset_auroc"] = tcga_comb_results[m_name]["macro_avg_auroc"]
        tcga_main_table[m_name]["mrr_6incomplete"] = tcga_comb_results[m_name]["missing_robustness_retention"]

    return {
        "n_total": int(N_tot),
        "n_events": int(np.sum(y_all)),
        "n_os_deaths": int(np.sum(os_event_all)),
        "n_excluded_censoring_or_zero_followup": n_excluded_tcga,
        "tcga_cohort_counts": tcga_cohort_counts,
        "tcga_main_table": tcga_main_table,
        "tcga_per_cohort": tcga_per_cohort,
        "tcga_combinatorial_7subsets": tcga_comb_results,
        "oof_arrays": {
            "y": y_all,
            "os_time_months": os_time_all,
            "os_event": os_event_all,
            "fold_ids": tcga_fold_ids,
            "probs": oof_p,
            "hesitancy": oof_h,
        },
    }


def run_track5_multi_fm_hesitant_and_pythagorean_tta(wsi_dir: str) -> Dict[str, object]:
    """
    Track 5: Multi-Foundation-Model Hesitant Fuzzy Consensus (Block 2) & Pythagorean Fuzzy TTA (Block 5)
    across 6 Pathology Foundation Models (`uni2_h`, `phikon_v2`, `kaiko_vitl14`, `phikon`, `ctranspath`, `plip`):
      - Dataset A: Camelyon17 (N=2,500 histopathology patches across 5 hospitals; Source: Hospitals 0,1,2 n=1,500 -> Target OOD: Hospitals 3 & 4 n=1,000)
      - Dataset B: NCT-CRC (N=2,349 colorectal histopathology tiles; Source: NCT_CRC_HE_Discovery n=999 -> Target OOD: CRC_VAL_HE_7K_External n=1,350, C10)
      - Strictly in-fold 64-component PCA on full FM embeddings (no arbitrary first-64 raw slicing).
      - Genuine Adam optimization of `HesitantFuzzyAggregator` (`BCE + 0.25*Brier`) strictly on source OOF predictions.
      - Includes `MedianPool6FMs` and `Tent_TTA` (Wang et al., ICLR 2021 Shannon entropy minimization) baselines + BH-FDR.
      - Strictly zero test-label selection (`BestSingleFM` selected on source CV, zero `max` over test AUROC, C9).
    """
    print("\n[Track 5] Running 6-Foundation-Model Hesitant Consensus & Pythagorean TTA (Camelyon17 & NCT-CRC)...")
    fm_names = ["uni2_h", "phikon_v2", "kaiko_vitl14", "phikon", "ctranspath", "plip"]

    def _evaluate_6fm_scenario(
        npz_path: str,
        tr_mask: np.ndarray,
        te_mask: np.ndarray,
        y_all: np.ndarray,
        corrupt_one_fm: bool = False,
    ) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
        npz = np.load(npz_path, allow_pickle=True)
        K = len(fm_names)
        y_tr = y_all[tr_mask]
        y_te = y_all[te_mask]
        N_tr = len(y_tr)

        # Compute out-of-fold source predictions for training HesitantFuzzyAggregator and selecting BestSingleFM
        skf_src = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_src_probs_k = [np.zeros(N_tr, dtype=np.float32) for _ in range(K)]
        te_probs_k = []
        src_cv_aurocs = []
        fm_raw_dims: Dict[str, int] = {}

        for k_idx, fm in enumerate(fm_names):
            X_fm_full = npz[fm].astype(np.float32)
            fm_raw_dims[fm] = int(X_fm_full.shape[1])
            X_tr_raw = X_fm_full[tr_mask]
            X_te_raw = X_fm_full[te_mask]
            n_comp = min(64, X_tr_raw.shape[1])

            # Out-of-fold source predictions with strict in-fold standardization + PCA(64)
            for f_s, (s_tr, s_va) in enumerate(skf_src.split(X_tr_raw, y_tr)):
                mu_s = X_tr_raw[s_tr].mean(axis=0, keepdims=True)
                sd_s = X_tr_raw[s_tr].std(axis=0, keepdims=True) + 1e-5
                X_str_std = (X_tr_raw[s_tr] - mu_s) / sd_s
                X_sva_std = (X_tr_raw[s_va] - mu_s) / sd_s
                if X_tr_raw.shape[1] > n_comp:
                    pca_s = PCA(n_components=n_comp, random_state=42 + f_s)
                    X_str_feat = pca_s.fit_transform(X_str_std).astype(np.float32)
                    X_sva_feat = pca_s.transform(X_sva_std).astype(np.float32)
                else:
                    X_str_feat, X_sva_feat = X_str_std, X_sva_std
                clf_s = LogisticRegression(C=0.2, max_iter=300)
                clf_s.fit(X_str_feat, y_tr[s_tr])
                oof_src_probs_k[k_idx][s_va] = clf_s.predict_proba(X_sva_feat)[:, 1]

            src_cv_aurocs.append(compute_binary_metrics(oof_src_probs_k[k_idx], y_tr)["auroc"])

            # Full source fit -> target OOD evaluation with source-only standardization + PCA(64)
            mu = X_tr_raw.mean(axis=0, keepdims=True)
            sd = X_tr_raw.std(axis=0, keepdims=True) + 1e-5
            X_tr_std = (X_tr_raw - mu) / sd
            X_te_std = (X_te_raw - mu) / sd
            if X_tr_raw.shape[1] > n_comp:
                pca_full = PCA(n_components=n_comp, random_state=42)
                X_tr_n = pca_full.fit_transform(X_tr_std).astype(np.float32)
                X_te_n = pca_full.transform(X_te_std).astype(np.float32)
            else:
                X_tr_n, X_te_n = X_tr_std, X_te_std

            if corrupt_one_fm and k_idx == K - 1:
                X_te_n = -2.0 * X_te_n

            clf = LogisticRegression(C=0.2, max_iter=300)
            clf.fit(X_tr_n, y_tr)
            p_te = clf.predict_proba(X_te_n)[:, 1]
            te_probs_k.append(p_te)

        # Select BestSingleFM strictly based on source 5-fold CV AUROC (zero test-label peeking, C9)
        best_src_k = int(np.argmax(src_cv_aurocs))
        best_single_p = te_probs_k[best_src_k].astype(np.float32)
        best_single_te_auroc = float(compute_binary_metrics(best_single_p, y_te)["auroc"])

        te_mat = np.stack(te_probs_k, axis=1).astype(np.float32)
        mean_pool_p = np.mean(te_mat, axis=1).astype(np.float32)
        mean_pool_auroc = float(compute_binary_metrics(mean_pool_p, y_te)["auroc"])

        median_pool_p = np.median(te_mat, axis=1).astype(np.float32)
        median_pool_auroc = float(compute_binary_metrics(median_pool_p, y_te)["auroc"])

        # Train Block 2: HesitantFuzzyAggregator strictly on source OOF predictions via Adam (BCE + 0.25*Brier)
        set_all_seeds(42)
        hfs_agg = HesitantFuzzyAggregator(num_sources=K, concordance_kappa=2.6)
        au_arr = np.asarray(src_cv_aurocs, dtype=np.float32)
        rel_init = (au_arr - au_arr.mean()) / (au_arr.std() + 1e-4) * 0.25
        rel_init_t = torch.tensor(rel_init[None, :], dtype=torch.float32)
        with torch.no_grad():
            hfs_agg.prompt_reliability.copy_(rel_init_t)
            hfs_agg.certainty_temp.fill_(-2.5)

        h_src_t = torch.tensor(np.stack(oof_src_probs_k, axis=1), dtype=torch.float32)
        y_src_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(-1)
        opt_hfs_src = torch.optim.Adam(hfs_agg.parameters(), lr=0.025)
        for _ in range(150):
            opt_hfs_src.zero_grad()
            out_src = hfs_agg(h_src_t)
            mu_src = out_src["consensus_mu"].clamp(1e-6, 1.0 - 1e-6)
            loss_src = (
                F.binary_cross_entropy(mu_src, y_src_t)
                + 0.25 * (mu_src - y_src_t).pow(2).mean()
                + 0.25 * F.softplus(hfs_agg.certainty_temp).pow(2).mean()
                + 0.05 * (hfs_agg.prompt_reliability - rel_init_t).pow(2).mean()
            )
            loss_src.backward()
            opt_hfs_src.step()

        with torch.no_grad():
            src_oof_hfs_auroc = float(
                compute_binary_metrics(hfs_agg(h_src_t)["consensus_mu"].view(-1).numpy(), y_tr)["auroc"]
            )

        h_te_t = torch.tensor(te_mat, dtype=torch.float32)
        with torch.no_grad():
            out_te = hfs_agg(h_te_t)
            hfs_p = out_te["consensus_mu"].view(-1).numpy().astype(np.float32)
            hfs_auroc = float(compute_binary_metrics(hfs_p, y_te)["auroc"])

        # Baseline TTA: Tent (Wang et al., ICLR 2021) — unsupervised target Shannon entropy minimization
        set_all_seeds(42)
        logits_te_base = torch.logit(h_te_t.clamp(1e-4, 1.0 - 1e-4))
        tent_scale = nn.Parameter(torch.ones(1, K))
        tent_bias = nn.Parameter(torch.zeros(1, K))
        opt_tent = torch.optim.Adam([tent_scale, tent_bias], lr=0.03)
        for _ in range(25):
            opt_tent.zero_grad()
            probs_tent_k = torch.sigmoid(logits_te_base * tent_scale + tent_bias)
            p_tent_ens = probs_tent_k.mean(dim=1).clamp(1e-6, 1.0 - 1e-6)
            shannon_ent = -(p_tent_ens * torch.log(p_tent_ens) + (1.0 - p_tent_ens) * torch.log(1.0 - p_tent_ens)).mean()
            anchor_tent = 0.20 * ((tent_scale - 1.0).pow(2).mean() + tent_bias.pow(2).mean())
            (shannon_ent + anchor_tent).backward()
            opt_tent.step()
        with torch.no_grad():
            tent_p = torch.sigmoid(logits_te_base * tent_scale + tent_bias).mean(dim=1).numpy().astype(np.float32)
            tent_auroc = float(compute_binary_metrics(tent_p, y_te)["auroc"])

        # Block 5: PythagoreanFuzzyTTALoss Unsupervised Test-Time Adaptation on Target Domain
        for p in hfs_agg.parameters():
            p.requires_grad_(True)
        tta_scale = nn.Parameter(torch.ones(1, K))
        pfs_loss_fn = PythagoreanFuzzyTTALoss(hesitancy_threshold=0.65, diversity_weight=0.25)
        opt_tta = torch.optim.Adam([tta_scale, hfs_agg.prompt_reliability], lr=0.04)

        med_p = h_te_t.median(dim=1, keepdim=True).values.detach()
        discord_k = (h_te_t - med_p).abs().mean(dim=0, keepdim=True).detach()
        rel_anchor = hfs_agg.prompt_reliability.detach().clone()

        for _ in range(25):
            opt_tta.zero_grad()
            cal_probs = torch.sigmoid(logits_te_base * tta_scale)
            out_adapt = hfs_agg(cal_probs)
            tta_res = pfs_loss_fn(out_adapt["ifn"])
            w_k = F.softmax(hfs_agg.prompt_reliability, dim=1)
            concord_reg = (w_k * discord_k).sum()
            reg = 0.15 * (hfs_agg.prompt_reliability - rel_anchor).pow(2).mean() + 0.5 * (
                tta_scale - 1.0
            ).pow(2).mean()
            loss_t = 0.2 * tta_res["tta_loss"] + 2.5 * concord_reg + reg
            loss_t.backward()
            opt_tta.step()

        with torch.no_grad():
            cal_probs = torch.sigmoid(logits_te_base * tta_scale)
            out_adapt = hfs_agg(cal_probs)
            pfs_p = out_adapt["consensus_mu"].view(-1).numpy().astype(np.float32)
            pfs_auroc = float(compute_binary_metrics(pfs_p, y_te)["auroc"])

        _, lo_pfs, hi_pfs = bootstrap_confidence_interval(pfs_p, y_te, "auroc", n_bootstraps=1000)
        p_vs_single = paired_bootstrap_pvalue(pfs_p, best_single_p, y_te, "auroc", n_bootstraps=1000)
        p_vs_mean = paired_bootstrap_pvalue(pfs_p, mean_pool_p, y_te, "auroc", n_bootstraps=1000)
        p_vs_median = paired_bootstrap_pvalue(pfs_p, median_pool_p, y_te, "auroc", n_bootstraps=1000)
        p_vs_tent = paired_bootstrap_pvalue(pfs_p, tent_p, y_te, "auroc", n_bootstraps=1000)
        p_vs_hfs = paired_bootstrap_pvalue(pfs_p, hfs_p, y_te, "auroc", n_bootstraps=1000)
        q_map = compute_bh_fdr_map(
            {
                "BestSingleFM": p_vs_single,
                "MeanPool6FMs": p_vs_mean,
                "MedianPool6FMs": p_vs_median,
                "Tent_TTA": p_vs_tent,
                "HesitantFuzzy": p_vs_hfs,
            }
        )

        metrics_dict = {
            "n_source": int(N_tr),
            "n_target": int(len(y_te)),
            "pca_components_in_fold": 64,
            "fm_raw_dims": fm_raw_dims,
            "selected_source_fm": fm_names[best_src_k],
            "source_oof_hfs_auroc": src_oof_hfs_auroc,
            "BestSingleFM": best_single_te_auroc,
            "MeanPool6FMs": mean_pool_auroc,
            "MedianPool6FMs": median_pool_auroc,
            "Tent_TTA": tent_auroc,
            "HesitantFuzzy": hfs_auroc,
            "HesitantFuzzy_PFS_TTA": pfs_auroc,
            "HesitantFuzzy_PFS_TTA_ci": [lo_pfs, hi_pfs],
            "p_value_vs_single": p_vs_single,
            "p_value_vs_mean": p_vs_mean,
            "p_value_vs_median": p_vs_median,
            "p_value_vs_tent": p_vs_tent,
            "p_value_vs_hfs_notta": p_vs_hfs,
            "q_bh_vs_ours": q_map,
            "trained_hfs_source_weights": [
                round(float(x), 4)
                for x in F.softmax(rel_anchor, dim=1).view(-1).cpu().tolist()
            ],
        }
        arrays_dict = {
            "y_te": y_te.astype(np.float32),
            "best_single_p": best_single_p,
            "mean_pool_p": mean_pool_p,
            "median_pool_p": median_pool_p,
            "tent_p": tent_p,
            "hfs_p": hfs_p,
            "pfs_p": pfs_p,
        }
        return metrics_dict, arrays_dict

    # 1. Camelyon17 Cross-Hospital (Hospitals 0,1,2 -> Hospital 3 & 4)
    cam_meta = pd.read_csv(os.path.join(wsi_dir, "multi_fm_features/camelyon17_metadata.csv"))
    cam_npz = os.path.join(wsi_dir, "multi_fm_features/camelyon17_6fms.npz")
    y_cam = cam_meta["label"].values.astype(np.float32)
    hosp_uniq = sorted(cam_meta["hospital"].unique())
    tr_cam = cam_meta["hospital"].isin(hosp_uniq[:3]).values
    te_cam = cam_meta["hospital"].isin(hosp_uniq[3:]).values

    res_cam_clean, arr_cam_clean = _evaluate_6fm_scenario(cam_npz, tr_cam, te_cam, y_cam, corrupt_one_fm=False)
    res_cam_corrupt, arr_cam_corrupt = _evaluate_6fm_scenario(cam_npz, tr_cam, te_cam, y_cam, corrupt_one_fm=True)
    res_cam_corrupt["p_value_clean_vs_corrupt_mean"] = paired_bootstrap_pvalue(
        arr_cam_clean["mean_pool_p"], arr_cam_corrupt["mean_pool_p"], arr_cam_clean["y_te"], "auroc", n_bootstraps=1000
    )

    # 2. NCT-CRC Cross-Cohort (Explicitly NCT_CRC_HE_Discovery -> CRC_VAL_HE_7K_External, C10)
    crc_meta = pd.read_csv(os.path.join(wsi_dir, "multi_fm_features/nct_crc_cross_cohort_metadata.csv"))
    crc_npz = os.path.join(wsi_dir, "multi_fm_features/nct_crc_cross_cohort_6fms.npz")
    y_crc = crc_meta["is_tumor_or_stroma"].values.astype(np.float32)
    tr_crc = (crc_meta["cohort"] == "NCT_CRC_HE_Discovery").values
    te_crc = (crc_meta["cohort"] == "CRC_VAL_HE_7K_External").values

    res_crc_clean, arr_crc_clean = _evaluate_6fm_scenario(crc_npz, tr_crc, te_crc, y_crc, corrupt_one_fm=False)
    res_crc_corrupt, arr_crc_corrupt = _evaluate_6fm_scenario(crc_npz, tr_crc, te_crc, y_crc, corrupt_one_fm=True)
    res_crc_corrupt["p_value_clean_vs_corrupt_mean"] = paired_bootstrap_pvalue(
        arr_crc_clean["mean_pool_p"], arr_crc_corrupt["mean_pool_p"], arr_crc_clean["y_te"], "auroc", n_bootstraps=1000
    )

    multi_fm_summary = {
        "Camelyon17\n(OOD Hospitals)": res_cam_clean,
        "Camelyon17\n(1 Corrupted FM)": res_cam_corrupt,
        "NCT-CRC\n(External Cohort)": res_crc_clean,
        "NCT-CRC\n(1 Corrupted FM)": res_crc_corrupt,
    }
    for k, v in multi_fm_summary.items():
        k_clean = k.replace("\n", " ")
        print(
            f"  [6-FM Benchmark] {k_clean:28s}: Single={v['BestSingleFM']:.4f}, Mean={v['MeanPool6FMs']:.4f}, "
            f"Median={v['MedianPool6FMs']:.4f}, Tent={v['Tent_TTA']:.4f}, HFS={v['HesitantFuzzy']:.4f}, "
            f"HFS+PFS-TTA={v['HesitantFuzzy_PFS_TTA']:.4f}"
        )

    return {
        "summary": multi_fm_summary,
        "arrays": {
            "cam_clean": arr_cam_clean,
            "cam_corrupt": arr_cam_corrupt,
            "crc_clean": arr_crc_clean,
            "crc_corrupt": arr_crc_corrupt,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="DeepFuzzyTorch Master Multi-Dataset Benchmark")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda, mps, or cpu)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory containing ADNI/OASIS-3 and WSI/TCGA datasets",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=WORKSPACE_DIR,
        help="Output workspace directory for results/ and figures/",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip regenerating figures/Fig1..Fig5 (useful before final visualization pass)",
    )
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    paths = resolve_data_paths(args.data_root)
    results_dir = os.path.join(args.out_dir, "results")
    figures_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    t0 = time.time()
    print("=" * 84)
    print(f"DeepFuzzyTorch: Executing Multi-Dataset Top-Tier Journal Benchmark Suite on [{device}]")
    if device.type == "cuda":
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print("=" * 84)

    track0 = run_track0_theorem_numerical_verification(device=device)
    adni_data = load_adni_cohort1(paths["adni_dir"])
    track1 = run_track1_adni_main_missing_conflict_ablation(adni_data, device=device)
    track2 = run_track2_adni_cohort2_ptau217_and_ckd_conflict(
        paths["adni_dir"], device=device, cohort1_rids=adni_data["rids"]
    )
    track3 = run_track3_oasis3_external_and_survival(
        adni_data, paths["adni_dir"], paths["oasis_dx_path"], device=device
    )
    track4 = run_track4_tcga_pancancer_multiomics_wsi(paths["wsi_dir"], device=device)
    track5_out = run_track5_multi_fm_hesitant_and_pythagorean_tta(paths["wsi_dir"])
    track5 = track5_out["summary"]

    # Save out-of-fold predictions and split indices across ALL tracks for full reproducibility (R2-M12)
    oof_npz_path = os.path.join(results_dir, "oof_predictions_and_splits.npz")
    oof_save_dict = {
        "adni1_y": track1["oof_arrays"]["y"],
        "adni1_fold_ids": track1["oof_arrays"]["fold_ids"],
        "adni2_y": track2["oof_arrays"]["y"],
        "adni2_fold_ids": track2["oof_arrays"]["fold_ids"],
        "adni_long_y": track3["adni_longitudinal_arrays"]["y"],
        "adni_long_y_60m": track3["adni_longitudinal_arrays"]["y_60m"],
        "adni_long_mask_60m": track3["adni_longitudinal_arrays"]["mask_60m"],
        "adni_long_time_years": track3["adni_longitudinal_arrays"]["time_years"],
        "adni_long_fold_ids": track3["adni_longitudinal_arrays"]["fold_ids"],
        "adni_long_tertile_tier": track3["adni_longitudinal_arrays"]["tertile_tier"],
        "oasis3_y_dementia": track3["oasis3_arrays"]["y_dementia"],
        "oasis3_y_cdr05": track3["oasis3_arrays"]["y_cdr05"],
        "tcga_y": track4["oof_arrays"]["y"],
        "tcga_os_time_months": track4["oof_arrays"]["os_time_months"],
        "tcga_os_event": track4["oof_arrays"]["os_event"],
        "tcga_fold_ids": track4["oof_arrays"]["fold_ids"],
    }
    for k, arr in track1["oof_arrays"]["probs"].items():
        oof_save_dict[f"adni1_prob_{k}"] = arr
        oof_save_dict[f"adni1_hes_{k}"] = track1["oof_arrays"]["hesitancy"][k]
        oof_save_dict[f"adni1_conf_{k}"] = track1["oof_arrays"]["conflict"][k]
    for k, arr in track2["oof_arrays"]["probs"].items():
        oof_save_dict[f"adni2_prob_{k}"] = arr
        oof_save_dict[f"adni2_hes_{k}"] = track2["oof_arrays"]["hesitancy"][k]
        oof_save_dict[f"adni2_conf_{k}"] = track2["oof_arrays"]["conflict"][k]
    oof_save_dict["adni2_modality_mus_DeepFuzzyTorch"] = track2["oof_arrays"]["modality_mus"]
    oof_save_dict["adni2_modality_nus_DeepFuzzyTorch"] = track2["oof_arrays"]["modality_nus"]
    oof_save_dict["adni2_modality_pis_DeepFuzzyTorch"] = track2["oof_arrays"]["modality_pis"]
    oof_save_dict["adni2_directional_plasma_conflict_DeepFuzzyTorch"] = track2["oof_arrays"]["directional_plasma_conflict"]
    oof_save_dict["adni2_egfr_bl"] = track2["oof_arrays"]["egfr_bl"]
    torch.save(track2["oof_arrays"]["fold_states"], os.path.join(results_dir, "adni2_5fold_deepfuzzy_states.pt"))
    for k, arr in track3["adni_longitudinal_arrays"]["probs"].items():
        oof_save_dict[f"adni_long_prob_{k}"] = arr
        oof_save_dict[f"adni_long_hes_{k}"] = track3["adni_longitudinal_arrays"]["hesitancy"][k]
    for k, arr in track3["oasis3_arrays"]["probs_nommse"].items():
        oof_save_dict[f"oasis3_nommse_prob_{k}"] = arr
        oof_save_dict[f"oasis3_nommse_hes_{k}"] = track3["oasis3_arrays"]["hesitancy_nommse"][k]
        oof_save_dict[f"oasis3_mmse_prob_{k}"] = track3["oasis3_arrays"]["probs_mmse"][k]
        oof_save_dict[f"oasis3_mmse_hes_{k}"] = track3["oasis3_arrays"]["hesitancy_mmse"][k]
    for k, arr in track4["oof_arrays"]["probs"].items():
        oof_save_dict[f"tcga_prob_{k}"] = arr
        oof_save_dict[f"tcga_hes_{k}"] = track4["oof_arrays"]["hesitancy"][k]
    for scen_key, scen_arrs in track5_out["arrays"].items():
        for arr_k, arr_v in scen_arrs.items():
            oof_save_dict[f"track5_{scen_key}_{arr_k}"] = arr_v
    np.savez_compressed(oof_npz_path, **oof_save_dict)

    # Remove raw numpy oof_arrays from track2 and track4 before JSON serialization
    track2_json = {k: v for k, v in track2.items() if k != "oof_arrays"}
    track4_json = {k: v for k, v in track4.items() if k != "oof_arrays"}

    runtime_json_path = os.path.join(results_dir, "runtime_vram_benchmark.json")
    track6_runtime = None
    if os.path.exists(runtime_json_path):
        with open(runtime_json_path, "r", encoding="utf-8") as f_rt:
            track6_runtime = json.load(f_rt)

    master_results = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "elapsed_seconds": round(time.time() - t0, 2),
        "track0_theorems": track0,
        "track1_adni_cohort1": {
            "n_total": track1["n_total"],
            "n_abeta_neg": track1["n_abeta_neg"],
            "n_abeta_pos": track1["n_abeta_pos"],
            "n_same_visit_bl_ref": track1["n_same_visit_bl_ref"],
            "n_confirmatory_followup_ref": track1["n_confirmatory_followup_ref"],
            "main_table": track1["main_table"],
            "same_visit_bl_924_table": track1["same_visit_bl_924_table"],
            "equalized_supervision_ablation": track1["equalized_supervision_ablation"],
            "rough_summary": track1["rough_summary"],
            "combinatorial_15subsets": track1["combinatorial_15subsets"],
            "missing_drop_curves": track1["missing_drop_curves"],
            "conflict_results": track1["conflict_results"],
            "dca_data": track1["dca_data"],
            "dca_strictly_positive_thresholds": track1["dca_strictly_positive_thresholds"],
            "dca_strictly_positive_contiguous_intervals": track1["dca_strictly_positive_contiguous_intervals"],
            "dca_failing_thresholds_in_015_080": track1["dca_failing_thresholds_in_015_080"],
            "calib_data": track1["calib_data"],
            "tsk_vis": track1["tsk_vis"],
            "exported_tsk_rules": track1["tsk_vis"]["exported_rules"],
            "structured_tsk_rules": track1["tsk_vis"]["structured_rules"],
            "tsk_reconstruction_max_abs_error": track1["tsk_vis"]["reconstruction_max_abs_error"],
        },
        "track2_adni_cohort2_ptau217": track2_json,
        "track3_oasis3_external": {
            "n_adni_longitudinal": track3["survival_vis"]["n_adni_longitudinal"],
            "n_adni_events": track3["survival_vis"]["n_adni_events"],
            "n_zero_followup_excluded": track3["survival_vis"]["n_zero_followup_excluded"],
            "n_shared_with_cohort1_1137": track3["survival_vis"]["n_shared_with_cohort1_1137"],
            "n_60m_horizon": track3["survival_vis"]["n_60m_horizon"],
            "n_60m_events": track3["survival_vis"]["n_60m_events"],
            "n_60m_late_event_controls": track3["survival_vis"]["n_60m_late_event_controls"],
            "n_oasis_total": track3["n_oasis_total"],
            "n_oasis_dementia_events": track3["n_oasis_dementia_events"],
            "n_oasis_cdr05_events": track3["n_oasis_cdr05_events"],
            "n_oasis_uds_d1_missing": track3["n_oasis_uds_d1_missing"],
            "n_oasis_uds_d1_concordant": track3["n_oasis_uds_d1_concordant"],
            "n_source_strict_ad_events": track3["n_source_strict_ad_events"],
            "seeds_evaluated": track3["seeds_evaluated"],
            "adni_longitudinal_table": track3["adni_longitudinal_table"],
            "oasis3_external_table": track3["oasis3_external_table"],
            "oasis3_strict_ad_source_table": track3["oasis3_strict_ad_source_table"],
            "oasis3_uds_d1_concordant_814_table": track3["oasis3_uds_d1_concordant_814_table"],
            "survival_vis": track3["survival_vis"],
            "c_index": track3["survival_vis"]["c_index"],
            "logrank_chi2": track3["survival_vis"]["logrank_chi2"],
            "logrank_p": track3["survival_vis"]["logrank_p"],
            "logrank_3group_chi2": track3["survival_vis"]["logrank_3group_chi2"],
            "logrank_3group_p": track3["survival_vis"]["logrank_3group_p"],
            "q33_cutoff": track3["survival_vis"]["q33_cutoff"],
            "q67_cutoff": track3["survival_vis"]["q67_cutoff"],
            "per_fold_q33_cutoffs": track3["survival_vis"]["per_fold_q33_cutoffs"],
            "per_fold_q67_cutoffs": track3["survival_vis"]["per_fold_q67_cutoffs"],
            "q45_cutoff": track3["survival_vis"]["q45_cutoff"],
            "q72_cutoff": track3["survival_vis"]["q72_cutoff"],
            "n_low": track3["survival_vis"]["n_low"],
            "n_mid": track3["survival_vis"]["n_mid"],
            "n_high": track3["survival_vis"]["n_high"],
        },
        "track4_tcga_pancancer": track4_json,
        "track5_multi_fm_consensus_tta": track5,
        "figures": {},
    }
    if track6_runtime is not None:
        master_results["track6_runtime_complexity"] = track6_runtime

    def _to_json_safe(obj: object) -> object:
        if isinstance(obj, dict):
            return {str(k): _to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_to_json_safe(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    json_path = os.path.join(results_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(master_results), f, indent=2)

    if not args.skip_figures:
        from deepfuzzy.visualization import (
            plot_fig1_theorem_verification,
            plot_fig2_missing_modality_benchmark,
            plot_fig3_conflict_and_abstention,
            plot_fig4_fuzzy_rough_and_multiomics,
            plot_fig5_clinical_translation_and_rules,
        )
        print("\n[Visualization] Generating 5 Multi-Panel Publication Figures (PDF + 300 DPI PNG)...")
        f1 = plot_fig1_theorem_verification(track0, figures_dir)
        f2 = plot_fig2_missing_modality_benchmark(
            track1["combinatorial_15subsets"],
            track1["missing_drop_curves"],
            track4["tcga_combinatorial_7subsets"],
            figures_dir,
        )
        f3 = plot_fig3_conflict_and_abstention(
            track1["conflict_results"],
            track2["ckd_subgroup"],
            figures_dir,
        )
        f4 = plot_fig4_fuzzy_rough_and_multiomics(
            track1["rough_summary"],
            track4["tcga_per_cohort"],
            track5,
            figures_dir,
        )
        f5 = plot_fig5_clinical_translation_and_rules(
            track1["dca_data"],
            track1["calib_data"],
            track1["tsk_vis"],
            track3["survival_vis"],
            figures_dir,
        )

        def _rel_fig(d: Dict[str, str]) -> Dict[str, str]:
            return {k: os.path.relpath(v, args.out_dir) for k, v in d.items()}

        master_results["figures"] = {
            "Fig1": _rel_fig(f1),
            "Fig2": _rel_fig(f2),
            "Fig3": _rel_fig(f3),
            "Fig4": _rel_fig(f4),
            "Fig5": _rel_fig(f5),
        }
    master_results["elapsed_seconds"] = round(time.time() - t0, 2)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(master_results), f, indent=2)

    print("=" * 84)
    print(f"Benchmark Suite Complete in {master_results['elapsed_seconds']:.1f}s!")
    print(f"Saved JSON summary to: {json_path}")
    print(f"Saved OOF predictions to: {oof_npz_path}")
    print("=" * 84)


if __name__ == "__main__":
    main()
