"""
Execute In-Fold Empirical Upgrades 1 & 2 for DeepFuzzyTorch:
  - Upgrade 1: Multi-Cohort Real-World Clinical Biomarker Discordance Audit across ADNI Cohort 1,
    ADNI Cohort 2, and Deduplicated Unique Cross-Cohort ADNI Participants, using strictly in-fold
    biomarker median cutoffs, raw in-sample outer-training 80th/85th percentile referral cutoffs, and
    symmetric evaluation of both `composite` (0.5*hesitancy + 0.5*conflict) and `hes_only` triage
    across all models (including RCML, QMF, DrFuse, EvidentialDL, GatedAttention, EarlyConcat).
  - Upgrade 2: Supervised In-Fold Fuzzy-Rough Attribute Reduction on the Unsupervised Transductive
    1,000-Gene TCGA PanCancer Atlas MAD Pool (zero manual gene curation, zero alphabetical truncation),
    Stabilized Pointwise T-Norm Reduct (K*=20) vs. 4 Standard Feature-Selection Baselines
    (PCA-20, Lasso-20, mRMR-20, MI-20), Fair All-1,000-Gene Ablation (matched rough bottleneck &
    feature gates), SOTA Multimodal Baselines, and Exact Custom-Background Hypergeometric Reactome
    Pathway Enrichment (N_bg=1,000, m=676 K_bg>=2 pathways).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch

torch.set_num_threads(4)

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from deepfuzzy import (
    UnifiedDeepFuzzyModel,
    FuzzyRoughFeatureReducer,
    compute_binary_metrics,
    bootstrap_confidence_interval,
    paired_bootstrap_pvalue,
)
from deepfuzzy.visualization import (
    plot_fig3_conflict_and_abstention,
    plot_fig6_tcga_highdim_reduct_and_reactome,
)
from benchmarks.run_full_benchmark import (
    resolve_data_paths,
    load_adni_cohort1,
    fit_strict_preprocessor,
    fit_strict_preprocessor_with_params,
    apply_strict_preprocessor_params,
    train_deepfuzzy_or_baseline,
    _compute_c_index,
    compute_bh_fdr_map,
    attach_bh_fdr_to_table,
    pairwise_jaccard_summary,
    clopper_pearson_ci,
)


def _eval_precomputed_triage(
    prob_full: np.ndarray,
    score_full: np.ndarray,
    y_full: np.ndarray,
    disc_mask: np.ndarray,
    keep20_full: np.ndarray,
    keep15_full: np.ndarray,
    fold_thr20: List[float],
    fold_thr15: List[float],
) -> Dict[str, Any]:
    """
    Summarize 15% (85th pct) and 20% (80th pct) selective referral on discordant patients
    from strictly nested outer-fold calibration masks and thresholds.
    """
    p_d = prob_full[disc_mask]
    s_d = score_full[disc_mask]
    y_d = y_full[disc_mask]
    k20 = keep20_full[disc_mask]
    k15 = keep15_full[disc_mask]

    m0 = compute_binary_metrics(p_d, y_d, uncertainty=s_d)
    m20 = compute_binary_metrics(p_d[k20], y_d[k20], uncertainty=s_d[k20]) if np.sum(k20) >= 4 else m0
    m15 = compute_binary_metrics(p_d[k15], y_d[k15], uncertainty=s_d[k15]) if np.sum(k15) >= 4 else m0

    n0 = int(len(y_d))
    c0 = int(np.sum((p_d >= 0.5).astype(np.float32) == y_d))
    n20 = int(np.sum(k20))
    c20 = int(np.sum((p_d[k20] >= 0.5).astype(np.float32) == y_d[k20]))
    n15 = int(np.sum(k15))
    c15 = int(np.sum((p_d[k15] >= 0.5).astype(np.float32) == y_d[k15]))

    return {
        "n_unreferred": n0,
        "k_correct_unreferred": c0,
        "acc_unreferred": float(m0["acc"]),
        "acc_unreferred_ci_cp": clopper_pearson_ci(c0, n0),
        "auroc_unreferred": float(m0["auroc"]),
        "f1_unreferred": float(m0["f1"]),
        "n_refer15_retained": n15,
        "k_correct_refer15": c15,
        "acc_refer15": float(m15["acc"]),
        "acc_refer15_ci_cp": clopper_pearson_ci(c15, n15),
        "auroc_refer15": float(m15["auroc"]),
        "n_refer20_retained": n20,
        "k_correct_refer20": c20,
        "acc_refer20": float(m20["acc"]),
        "acc_refer20_ci_cp": clopper_pearson_ci(c20, n20),
        "auroc_refer20": float(m20["auroc"]),
        "f1_refer20": float(m20["f1"]),
        "in_fold_thr80_cutoffs": fold_thr20,
        "in_fold_thr85_cutoffs": fold_thr15,
    }


def evaluate_nested_discordant_triage(
    prob_full: np.ndarray,
    hes_full: np.ndarray,
    conf_full: np.ndarray,
    y_full: np.ndarray,
    disc_mask: np.ndarray,
    nested_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Symmetrically summarizes strictly nested `composite` (0.5*hes + 0.5*conf) and `hes_only`
    discordant triage from precomputed outer-fold training calibration decisions.
    """
    comp_clean = _eval_precomputed_triage(
        prob_full,
        0.5 * hes_full + 0.5 * conf_full,
        y_full,
        disc_mask,
        nested_cache["keep20_comp"],
        nested_cache["keep15_comp"],
        nested_cache["thr20_comp"],
        nested_cache["thr15_comp"],
    )
    hes_clean = _eval_precomputed_triage(
        prob_full,
        hes_full,
        y_full,
        disc_mask,
        nested_cache["keep20_hes"],
        nested_cache["keep15_hes"],
        nested_cache["thr20_hes"],
        nested_cache["thr15_hes"],
    )
    out = dict(comp_clean)
    out["evaluation_protocol"] = "strictly_nested_outer_training_fold_calibration"
    out["scoring_rule_top_level"] = "symmetric_composite_0.5hes_0.5conf_strict_nested"
    out["composite"] = comp_clean
    out["hes_only"] = hes_clean
    return out


def evaluate_dedup_cross_cohort_triage_nested(
    p1: np.ndarray,
    h1: np.ndarray,
    c1: np.ndarray,
    y1: np.ndarray,
    disc1: np.ndarray,
    nested1: Dict[str, Any],
    p2: np.ndarray,
    h2: np.ndarray,
    c2: np.ndarray,
    y2: np.ndarray,
    disc2: np.ndarray,
    c2_uniq: np.ndarray,
    nested2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates deduplicated cross-cohort discordant triage where C1 and C2 thresholds are each
    calibrated strictly on their own outer training folds (including C1/C2 training-fold models
    and training-fold medians) before combining non-overlapping validation decisions.
    """
    def _combine(
        score1: np.ndarray,
        score2: np.ndarray,
        k20_1: np.ndarray,
        k15_1: np.ndarray,
        k20_2: np.ndarray,
        k15_2: np.ndarray,
    ) -> Dict[str, Any]:
        uniq_on_disc2 = c2_uniq[disc2]
        p_dd = np.concatenate([p1[disc1], p2[disc2][uniq_on_disc2]])
        s_dd = np.concatenate([score1[disc1], score2[disc2][uniq_on_disc2]])
        y_dd = np.concatenate([y1[disc1], y2[disc2][uniq_on_disc2]])
        k20_dd = np.concatenate([k20_1[disc1], k20_2[disc2][uniq_on_disc2]])
        k15_dd = np.concatenate([k15_1[disc1], k15_2[disc2][uniq_on_disc2]])

        m0 = compute_binary_metrics(p_dd, y_dd, uncertainty=s_dd)
        m20 = compute_binary_metrics(p_dd[k20_dd], y_dd[k20_dd], uncertainty=s_dd[k20_dd])
        m15 = compute_binary_metrics(p_dd[k15_dd], y_dd[k15_dd], uncertainty=s_dd[k15_dd])
        n0 = int(len(y_dd))
        c0 = int(np.sum((p_dd >= 0.5).astype(np.float32) == y_dd))
        n20 = int(np.sum(k20_dd))
        c20 = int(np.sum((p_dd[k20_dd] >= 0.5).astype(np.float32) == y_dd[k20_dd]))
        n15 = int(np.sum(k15_dd))
        c15 = int(np.sum((p_dd[k15_dd] >= 0.5).astype(np.float32) == y_dd[k15_dd]))
        return {
            "n_unreferred": n0,
            "k_correct_unreferred": c0,
            "acc_unreferred": float(m0["acc"]),
            "acc_unreferred_ci_cp": clopper_pearson_ci(c0, n0),
            "auroc_unreferred": float(m0["auroc"]),
            "f1_unreferred": float(m0["f1"]),
            "n_refer15_retained": n15,
            "k_correct_refer15": c15,
            "acc_refer15": float(m15["acc"]),
            "acc_refer15_ci_cp": clopper_pearson_ci(c15, n15),
            "auroc_refer15": float(m15["auroc"]),
            "n_refer20_retained": n20,
            "k_correct_refer20": c20,
            "acc_refer20": float(m20["acc"]),
            "acc_refer20_ci_cp": clopper_pearson_ci(c20, n20),
            "auroc_refer20": float(m20["auroc"]),
            "f1_refer20": float(m20["f1"]),
        }

    comp_clean = _combine(
        0.5 * h1 + 0.5 * c1,
        0.5 * h2 + 0.5 * c2,
        nested1["keep20_comp"],
        nested1["keep15_comp"],
        nested2["keep20_comp"],
        nested2["keep15_comp"],
    )
    hes_clean = _combine(
        h1,
        h2,
        nested1["keep20_hes"],
        nested1["keep15_hes"],
        nested2["keep20_hes"],
        nested2["keep15_hes"],
    )
    out = dict(comp_clean)
    out["evaluation_protocol"] = "strictly_nested_outer_training_fold_calibration"
    out["scoring_rule_top_level"] = "symmetric_composite_0.5hes_0.5conf_strict_nested"
    out["composite"] = comp_clean
    out["hes_only"] = hes_clean
    return out


def compute_strictly_nested_adni_triage_caches(
    adni_dir: str,
    c1_df: pd.DataFrame,
    c2_df: pd.DataFrame,
    fids_c1: np.ndarray,
    fids_c2: np.ndarray,
    npz_data: Dict[str, Any],
    triage_models: List[str],
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Replay the exact 5 outer folds on ADNI Cohort 1 (N=1,137) and Cohort 2 (N=289) so that:
      - Discordant patient masks on training fold `tr_i` use ONLY fold-`i` training medians;
      - Hesitancy and conflict scores on `tr_i` come ONLY from fold-`i`'s trained model `f_i`;
      - 80th and 85th percentile referral cutoffs are fit strictly on `tr_i` discordant patients
        and applied out-of-fold to `va_i`, guaranteeing zero cross-fold leakage.
    """
    repro_report: Dict[str, Any] = {"c1_max_prob_abs_diff": {}, "c2_max_prob_abs_diff": {}}

    # ---- ADNI Cohort 1 replay (exact run_track1 settings) ----
    adni1_data = load_adni_cohort1(adni_dir)
    assert np.array_equal(adni1_data["rids"], c1_df["RID"].values), "C1 row mismatch"
    X1_raw_list = adni1_data["X_modalities"]
    y1 = adni1_data["y_amyloid"]
    mri1_cols = adni1_data["mri_cols"]
    N1 = len(y1)
    has_pl1 = (c1_df["plasma_ptau181_bl"].notna() & (c1_df["blood_is_bl_visit"].fillna(0) == 1)).values
    clin1_pos = ((c1_df["DX_bl"] != "CN") | (c1_df["CDRSB"] >= 0.5)).values
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
    c1_models = [m for m in triage_models if f"adni1_prob_{m}" in npz_data]
    c1_caches: Dict[str, Dict[str, Any]] = {
        m: {
            "keep20_comp": np.zeros(N1, dtype=bool),
            "keep15_comp": np.zeros(N1, dtype=bool),
            "keep20_hes": np.zeros(N1, dtype=bool),
            "keep15_hes": np.zeros(N1, dtype=bool),
            "thr20_comp": [],
            "thr15_comp": [],
            "thr20_hes": [],
            "thr15_hes": [],
            "replayed_va_prob": np.zeros(N1, dtype=np.float32),
        }
        for m in c1_models
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X1_raw_list[0], y1)):
        expected_va = np.where(fids_c1 == fold_idx)[0]
        assert np.array_equal(va_idx, expected_va), f"C1 fold {fold_idx} index mismatch"

        X_mri_tr_s, _, _, _ = fit_strict_preprocessor([X1_raw_list[0][tr_idx]], [X1_raw_list[0][va_idx]])
        reducer_f = FuzzyRoughFeatureReducer(
            max_features=30,
            sigma=0.85,
            max_samples=4096,
            n_null_permutations=8,
            n_stability_bootstraps=5,
            seed=42 + fold_idx,
        )
        reducer_f.fit(X_mri_tr_s[0], y1[tr_idx], feature_names=mri1_cols, device=str(device))
        sel_mri_idx = reducer_f.selected_indices_[:24]

        X_tr_raw = [X1_raw_list[0][tr_idx][:, sel_mri_idx], X1_raw_list[1][tr_idx], X1_raw_list[2][tr_idx], X1_raw_list[3][tr_idx]]
        X_va_raw = [X1_raw_list[0][va_idx][:, sel_mri_idx], X1_raw_list[1][va_idx], X1_raw_list[2][va_idx], X1_raw_list[3][va_idx]]
        X_tr_s, X_va_s, m_tr, m_va = fit_strict_preprocessor(X_tr_raw, X_va_raw)
        tr_tensors = [torch.tensor(x, dtype=torch.float32) for x in X_tr_s]
        tr_mask_t = torch.tensor(m_tr, dtype=torch.float32)
        va_tensors = [torch.tensor(x, dtype=torch.float32) for x in X_va_s]
        va_mask_t = torch.tensor(m_va, dtype=torch.float32)

        ptau1_med_f = float(np.nanmedian(c1_df.loc[tr_idx, "plasma_ptau181_bl"].values[has_pl1[tr_idx]]))
        hipp1_med_f = float(np.nanmedian(c1_df.loc[tr_idx, "Hippocampus_ICV_ratio"].values))
        pl1_pos_tr = c1_df.loc[tr_idx, "plasma_ptau181_bl"].values >= ptau1_med_f
        mri1_pos_tr = c1_df.loc[tr_idx, "Hippocampus_ICV_ratio"].values <= hipp1_med_f
        cl1_tr = clin1_pos[tr_idx]
        disc1_tr_in_fold = has_pl1[tr_idx] & (
            (pl1_pos_tr & (~mri1_pos_tr) & (~cl1_tr)) | ((~pl1_pos_tr) & mri1_pos_tr & cl1_tr)
        )

        for m_name in c1_models:
            trained_m = train_deepfuzzy_or_baseline(
                model_name=m_name,
                X_tr_list=X_tr_s,
                mask_tr=m_tr,
                y_tr=y1[tr_idx],
                epochs=28,
                lr=1.8e-3,
                embed_dim=40,
                concept_names=concept_names_8,
                ablation_flags={},
                seed=42 + fold_idx,
                device=device,
            )
            with torch.no_grad():
                if isinstance(trained_m, UnifiedDeepFuzzyModel):
                    c_in_tr = tr_tensors[-1] if trained_m.use_tsk_head else None
                    out_tr = trained_m(tr_tensors, modality_masks=tr_mask_t, clinical_concepts=c_in_tr)
                    c_in_va = va_tensors[-1] if trained_m.use_tsk_head else None
                    out_va = trained_m(va_tensors, modality_masks=va_mask_t, clinical_concepts=c_in_va)
                else:
                    out_tr = trained_m(tr_tensors, modality_masks=tr_mask_t)
                    out_va = trained_m(va_tensors, modality_masks=va_mask_t)

            hes_tr = out_tr["hesitancy"].view(-1).numpy()
            conf_tr = out_tr["conflict"].view(-1).numpy()
            comp_tr = 0.5 * hes_tr + 0.5 * conf_tr

            prob_va = out_va["prob"].view(-1).numpy()
            hes_va = out_va["hesitancy"].view(-1).numpy()
            conf_va = out_va["conflict"].view(-1).numpy()
            comp_va = 0.5 * hes_va + 0.5 * conf_va

            c1_caches[m_name]["replayed_va_prob"][va_idx] = prob_va
            cal_pool = disc1_tr_in_fold if int(np.sum(disc1_tr_in_fold)) >= 5 else np.ones(len(tr_idx), dtype=bool)
            t20_c = float(np.percentile(comp_tr[cal_pool], 80.0))
            t15_c = float(np.percentile(comp_tr[cal_pool], 85.0))
            t20_h = float(np.percentile(hes_tr[cal_pool], 80.0))
            t15_h = float(np.percentile(hes_tr[cal_pool], 85.0))

            c1_caches[m_name]["thr20_comp"].append(round(t20_c, 5))
            c1_caches[m_name]["thr15_comp"].append(round(t15_c, 5))
            c1_caches[m_name]["thr20_hes"].append(round(t20_h, 5))
            c1_caches[m_name]["thr15_hes"].append(round(t15_h, 5))
            c1_caches[m_name]["keep20_comp"][va_idx] = comp_va <= t20_c
            c1_caches[m_name]["keep15_comp"][va_idx] = comp_va <= t15_c
            c1_caches[m_name]["keep20_hes"][va_idx] = hes_va <= t20_h
            c1_caches[m_name]["keep15_hes"][va_idx] = hes_va <= t15_h

    for m_name in c1_models:
        max_diff = float(np.max(np.abs(c1_caches[m_name]["replayed_va_prob"] - npz_data[f"adni1_prob_{m_name}"])))
        repro_report["c1_max_prob_abs_diff"][m_name] = max_diff
        assert max_diff < 1e-5, f"C1 replay mismatch for {m_name}: max_abs_diff={max_diff}"

    # ---- ADNI Cohort 2 replay (exact run_track2 settings) ----
    c2_full = pd.read_csv(os.path.join(adni_dir, "cohort2_adni_ptau217_gfap.csv")).copy()
    c2_full = c2_full.assign(PTGENDER_num=(c2_full["PTGENDER"] == "Male").astype(float)).copy()
    sub2 = c2_full[c2_full["amyloid_gold_pos"].notna()].copy().reset_index(drop=True)
    assert np.array_equal(sub2["RID"].values, c2_df["RID"].values), "C2 row mismatch"
    y2 = sub2["amyloid_gold_pos"].values.astype(np.float32)
    N2 = len(y2)
    macro_cols2 = [
        "Ventricles_ICV_ratio",
        "Hippocampus_ICV_ratio",
        "WholeBrain_ICV_ratio",
        "Entorhinal_ICV_ratio",
        "Fusiform_ICV_ratio",
        "MidTemp_ICV_ratio",
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
    clin2_pos = ((c2_df["DX_bl"] != "CN") | (c2_df["CDRSB"] >= 0.5)).values

    c2_models = [m for m in triage_models if f"adni2_prob_{m}" in npz_data]
    c2_caches: Dict[str, Dict[str, Any]] = {
        m: {
            "keep20_comp": np.zeros(N2, dtype=bool),
            "keep15_comp": np.zeros(N2, dtype=bool),
            "keep20_hes": np.zeros(N2, dtype=bool),
            "keep15_hes": np.zeros(N2, dtype=bool),
            "thr20_comp": [],
            "thr15_comp": [],
            "thr20_hes": [],
            "thr15_hes": [],
            "replayed_va_prob": np.zeros(N2, dtype=np.float32),
        }
        for m in c2_models
    }

    for fold_idx, (tr_i, va_i) in enumerate(skf.split(X2_mri_full, y2)):
        expected_va2 = np.where(fids_c2 == fold_idx)[0]
        assert np.array_equal(va_i, expected_va2), f"C2 fold {fold_idx} index mismatch"

        X_mri_tr_all_s, _, _, _ = fit_strict_preprocessor([X2_mri_full[tr_i]], [X2_mri_full[va_i]])
        red_c2 = FuzzyRoughFeatureReducer(
            max_features=24,
            sigma=0.85,
            max_samples=4096,
            n_null_permutations=8,
            n_stability_bootstraps=5,
            seed=100 + fold_idx,
        )
        red_c2.fit(X_mri_tr_all_s[0], y2[tr_i], feature_names=mri2_cols_all, device=str(device))
        sel_c2 = red_c2.selected_indices_[:24]

        X_tr_raw = [X2_mri_full[tr_i][:, sel_c2], X2_plasma[tr_i], X2_fdg[tr_i], X2_clin[tr_i]]
        X_va_raw = [X2_mri_full[va_i][:, sel_c2], X2_plasma[va_i], X2_fdg[va_i], X2_clin[va_i]]
        X_tr_s, X_va_s, m_tr, m_va = fit_strict_preprocessor(X_tr_raw, X_va_raw)
        tr_t = [torch.tensor(x, dtype=torch.float32) for x in X_tr_s]
        tr_m = torch.tensor(m_tr, dtype=torch.float32)
        va_t = [torch.tensor(x, dtype=torch.float32) for x in X_va_s]
        va_m = torch.tensor(m_va, dtype=torch.float32)

        ptau2_med_f = float(np.nanmedian(c2_df.loc[tr_i, "pT217_F"].values))
        hipp2_med_f = float(np.nanmedian(c2_df.loc[tr_i, "Hippocampus_ICV_ratio"].values))
        pl2_pos_tr = c2_df.loc[tr_i, "pT217_F"].values >= ptau2_med_f
        mri2_pos_tr = c2_df.loc[tr_i, "Hippocampus_ICV_ratio"].values <= hipp2_med_f
        cl2_tr = clin2_pos[tr_i]
        disc2_tr_in_fold = (pl2_pos_tr & (~mri2_pos_tr) & (~cl2_tr)) | ((~pl2_pos_tr) & mri2_pos_tr & cl2_tr)

        for m_name in c2_models:
            mod = train_deepfuzzy_or_baseline(
                m_name,
                X_tr_s,
                m_tr,
                y2[tr_i],
                epochs=50,
                lr=2.5e-3,
                embed_dim=32,
                seed=100 + fold_idx,
                device=device,
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

            c2_caches[m_name]["replayed_va_prob"][va_i] = prob_va
            cal_pool2 = disc2_tr_in_fold if int(np.sum(disc2_tr_in_fold)) >= 5 else np.ones(len(tr_i), dtype=bool)
            t20_c = float(np.percentile(comp_tr[cal_pool2], 80.0))
            t15_c = float(np.percentile(comp_tr[cal_pool2], 85.0))
            t20_h = float(np.percentile(hes_tr[cal_pool2], 80.0))
            t15_h = float(np.percentile(hes_tr[cal_pool2], 85.0))

            c2_caches[m_name]["thr20_comp"].append(round(t20_c, 5))
            c2_caches[m_name]["thr15_comp"].append(round(t15_c, 5))
            c2_caches[m_name]["thr20_hes"].append(round(t20_h, 5))
            c2_caches[m_name]["thr15_hes"].append(round(t15_h, 5))
            c2_caches[m_name]["keep20_comp"][va_i] = comp_va <= t20_c
            c2_caches[m_name]["keep15_comp"][va_i] = comp_va <= t15_c
            c2_caches[m_name]["keep20_hes"][va_i] = hes_va <= t20_h
            c2_caches[m_name]["keep15_hes"][va_i] = hes_va <= t15_h

    for m_name in c2_models:
        max_diff2 = float(np.max(np.abs(c2_caches[m_name]["replayed_va_prob"] - npz_data[f"adni2_prob_{m_name}"])))
        repro_report["c2_max_prob_abs_diff"][m_name] = max_diff2
        assert max_diff2 < 1e-5, f"C2 replay mismatch for {m_name}: max_abs_diff={max_diff2}"

    return c1_caches, c2_caches, repro_report


def _select_mrmr_top_k(X_std: np.ndarray, y: np.ndarray, k: int = 20) -> List[int]:
    """
    Greedy Minimum-Redundancy Maximum-Relevance (mRMR, Peng et al. IEEE TPAMI 2005)
    fit strictly on standardized training-fold expression matrix X_std [N_tr, D].
    Relevance: ANOVA F-statistic normalized to [0, 1]; Redundancy: mean |Pearson r| with selected set.
    """
    f_vals, _ = f_classif(X_std, y)
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    f_max = float(np.max(f_vals))
    rel = f_vals / f_max if f_max > 1e-12 else f_vals

    N, D = X_std.shape
    X_c = X_std - np.mean(X_std, axis=0, keepdims=True)
    norms = np.linalg.norm(X_c, axis=0, keepdims=True) + 1e-8
    X_unit = X_c / norms  # unit-norm columns so |X_unit.T @ X_unit[:, idx]| = |Pearson r|

    selected: List[int] = []
    remaining = np.ones(D, dtype=bool)
    red_sum = np.zeros(D, dtype=np.float64)

    first = int(np.argmax(rel))
    selected.append(first)
    remaining[first] = False

    for step in range(1, min(k, D)):
        last_idx = selected[-1]
        abs_corr = np.abs(X_unit.T @ X_unit[:, last_idx])
        red_sum += abs_corr
        mean_red = red_sum / float(len(selected))
        score = rel - mean_red
        score[~remaining] = -1e18
        nxt = int(np.argmax(score))
        selected.append(nxt)
        remaining[nxt] = False
    return selected


def run_custom_background_reactome_enrichment(
    consensus_genes: List[str],
    background_1000_genes: List[str],
    gmt_path: str,
    out_json_path: str,
    out_csv_path: str,
) -> Dict[str, Any]:
    """
    Exact one-sided hypergeometric overrepresentation analysis of the Top-20 consensus genes
    against the exact 1,000-gene assayed candidate background universe (N_bg = 1,000),
    using unique HGNC gene counts (never inflated physical entity/complex counts) and
    Benjamini-Hochberg FDR across all m = 676 predeclared pathways with K_bg >= 2
    background genes in the 1,000-gene pool (not restricted to k_genes >= 2 submitted hits).
    """
    bg_set = set(str(g).strip() for g in background_1000_genes if str(g).strip())
    query_set = set(str(g).strip() for g in consensus_genes if str(g).strip()) & bg_set
    N_bg = len(bg_set)
    n_query = len(query_set)

    all_human_pathways: List[Tuple[str, str, set[str]]] = []
    annotated_bg_union: set[str] = set()
    with open(gmt_path, "r", encoding="utf-8") as f_gmt:
        for line in f_gmt:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pname, stid = parts[0].strip(), parts[1].strip()
            if not stid.startswith("R-HSA-"):
                continue
            p_genes = set(g.strip() for g in parts[2:] if g.strip())
            bg_ov = p_genes & bg_set
            if bg_ov:
                annotated_bg_union |= bg_ov
            if len(bg_ov) >= 2:
                all_human_pathways.append((stid, pname, bg_ov))

    annotated_query_union = query_set & annotated_bg_union

    # Compute exact one-sided hypergeometric p-values on the exact N_bg=1,000 background
    bg_tested_records: List[Dict[str, Any]] = []
    for stid, pname, bg_ov in all_human_pathways:
        K_bg = len(bg_ov)
        hits = sorted(bg_ov & query_set)
        k_genes = len(hits)
        # P(X >= k_genes) under Hypergeometric(M=N_bg, n=K_bg, N=n_query)
        p_val = float(hypergeom.sf(k_genes - 1, N_bg, K_bg, n_query)) if k_genes >= 1 else 1.0
        expected = float(n_query * (K_bg / float(N_bg)))
        fold_enr = float(k_genes / max(expected, 1e-12)) if k_genes > 0 else 0.0
        bg_tested_records.append(
            {
                "stId": stid,
                "name": pname,
                "k_genes": int(k_genes),
                "K_bg": int(K_bg),
                "N_bg": int(N_bg),
                "n_query": int(n_query),
                "expected_genes": round(expected, 4),
                "fold_enrichment": round(fold_enr, 3),
                "p_value": p_val,
                "unique_submitted_genes": hits,
            }
        )

    # BH-FDR across ALL predeclared pathways with K_bg >= 2 in the 1,000-gene background (m=676 pathways)
    p_map_all_bg = {r["stId"]: r["p_value"] for r in bg_tested_records}
    q_map_all_bg = compute_bh_fdr_map(p_map_all_bg)

    # Display subset: pathways with k_genes >= 2 unique submitted genes (NOT used to shrink BH denominator m)
    k2_records = [r for r in bg_tested_records if r["k_genes"] >= 2]

    enriched_rows: List[Dict[str, Any]] = []
    for r in k2_records:
        stid = r["stId"]
        pname = r["name"]
        short_lbl = pname if len(pname) <= 38 else (pname[:35].rstrip() + "...")
        q_all = float(q_map_all_bg[stid])
        genes_str = ", ".join(r["unique_submitted_genes"])
        enriched_rows.append(
            {
                "stId": stid,
                "name": pname,
                "short_name": f"{short_lbl} ({stid})",
                "k_genes": r["k_genes"],
                "K_bg": r["K_bg"],
                "N_bg": r["N_bg"],
                "n_query": r["n_query"],
                "found": r["k_genes"],  # strictly unique submitted gene count k_genes!
                "total": r["K_bg"],     # strictly pathway size K_bg within the 1,000-gene candidate pool!
                "ratio": round(r["K_bg"] / float(N_bg), 5),
                "fold_enrichment": r["fold_enrichment"],
                "p_value": r["p_value"],
                "q_bh": q_all,
                "fdr": q_all,
                "bh_family_size_m": int(len(bg_tested_records)),
                "unique_submitted_genes": r["unique_submitted_genes"],
                "submitted_genes_found": genes_str,
            }
        )

    enriched_rows.sort(key=lambda x: (x["p_value"], -x["fold_enrichment"], x["stId"]))

    # Save CSV
    csv_rows = [
        {
            "Pathway identifier": r["stId"],
            "Pathway name": r["name"],
            "k_genes_unique": r["k_genes"],
            "K_bg_in_1000": r["K_bg"],
            "N_bg_universe": r["N_bg"],
            "n_query_genes": r["n_query"],
            "Fold enrichment": r["fold_enrichment"],
            "Hypergeometric pValue": r["p_value"],
            "BH FDR (m=676 bg K_bg>=2)": r["q_bh"],
            "Unique submitted genes": ";".join(r["unique_submitted_genes"]),
        }
        for r in enriched_rows
    ]
    pd.DataFrame(csv_rows).to_csv(out_csv_path, index=False)

    payload = {
        "method": (
            "Exact one-sided hypergeometric test (scipy.stats.hypergeom.sf) against custom 1,000-gene "
            "assayed background universe; BH-FDR across all predeclared background pathways with K_bg>=2"
        ),
        "gmt_source": "Official Reactome human pathway database (ReactomePathways.gmt, Homo sapiens R-HSA-*)",
        "N_bg_total_candidate_genes": int(N_bg),
        "N_bg_reactome_annotated_genes": int(len(annotated_bg_union)),
        "n_query_total_genes": int(n_query),
        "n_query_reactome_annotated_genes": int(len(annotated_query_union)),
        "query_consensus_genes": sorted(query_set),
        "bh_family_size_m": int(len(bg_tested_records)),
        "n_bg_pathways_with_K_ge_2": int(len(bg_tested_records)),
        "n_displayed_pathways_k_ge_2": int(len(enriched_rows)),
        "n_tested_pathways_with_k_ge_2": int(len(enriched_rows)),
        "n_fdr_lt_005": int(sum(1 for r in bg_tested_records if q_map_all_bg[r["stId"]] < 0.05)),
        "n_fdr_lt_010": int(sum(1 for r in bg_tested_records if q_map_all_bg[r["stId"]] < 0.10)),
        "n_fdr_lt_0005": int(sum(1 for r in bg_tested_records if q_map_all_bg[r["stId"]] < 0.005)),
        "n_raw_p_lt_005": int(sum(1 for r in bg_tested_records if r["p_value"] < 0.05)),
        "n_raw_p_lt_005_k_ge_2": int(sum(1 for r in enriched_rows if r["p_value"] < 0.05)),
        "pathways": enriched_rows,
    }
    with open(out_json_path, "w", encoding="utf-8") as f_js:
        json.dump(payload, f_js, indent=2)
    return payload


def _build_tcga_protocol_and_reactome_updates(reactome_enrichment: Dict[str, Any]) -> Dict[str, Any]:
    """Shared TCGA protocol disclosure and m=676 BH-FDR Reactome summary fields."""
    return {
        "candidate_universe_protocol": (
            "Unsupervised, label-free transductive top-1,000 MAD expressed protein-coding genes across "
            "7 cBioPortal PanCanAtlas cohorts (18,495 common protein-coding genes -> 15,636 passing "
            ">=95% non-missing & median log2(RSEM+1)>=1.0 QC; zero outcome labels used during QC/MAD "
            "pre-screen, zero manual cancer-gene priority lists, zero alphabetical truncation; sporadic "
            "NaNs imputed strictly per training fold)"
        ),
        "reactome_background_universe_size": reactome_enrichment["N_bg_total_candidate_genes"],
        "reactome_annotated_background_size": reactome_enrichment["N_bg_reactome_annotated_genes"],
        "reactome_bh_family_size_m": reactome_enrichment["bh_family_size_m"],
        "reactome_total_pathways_found": reactome_enrichment["n_tested_pathways_with_k_ge_2"],
        "reactome_bg_pathways_with_K_ge_2": reactome_enrichment["n_bg_pathways_with_K_ge_2"],
        "reactome_raw_p_lt_005_count": reactome_enrichment["n_raw_p_lt_005"],
        "reactome_raw_p_lt_005_k_ge_2_count": reactome_enrichment["n_raw_p_lt_005_k_ge_2"],
        "reactome_fdr_lt_005_count": reactome_enrichment["n_fdr_lt_005"],
        "reactome_fdr_lt_010_count": reactome_enrichment["n_fdr_lt_010"],
        "reactome_fdr_lt_0005_count": reactome_enrichment["n_fdr_lt_0005"],
        "reactome_top_pathways": reactome_enrichment["pathways"][:18],
    }


def main():
    parser = argparse.ArgumentParser(description="DeepFuzzyTorch Zero-Leakage Upgrades 1 & 2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-figures", action="store_true", help="Skip regenerating Fig3 and Fig6")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run strictly nested ADNI Upgrade 1 replay and Reactome ORA (m=676) update without re-running 4-mod TCGA training",
    )
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[Upgrades 1 & 2] Executing on device={device} ({gpu_name})", flush=True)

    paths = resolve_data_paths(os.path.join(WORKSPACE_DIR, "data_benchmark"))
    results_dir = os.path.join(WORKSPACE_DIR, "results")
    figures_dir = os.path.join(WORKSPACE_DIR, "figures")
    json_path = os.path.join(results_dir, "benchmark_results.json")
    npz_path = os.path.join(results_dir, "oof_predictions_and_splits.npz")

    with open(json_path, "r", encoding="utf-8") as f:
        master = json.load(f)
    npz_data = dict(np.load(npz_path, allow_pickle=True))

    def _to_json_safe(obj: Any) -> Any:
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

    def _save_outputs() -> None:
        if not args.skip_figures:
            f3 = plot_fig3_conflict_and_abstention(
                master["track1_adni_cohort1"]["conflict_results"],
                master["track2_adni_cohort2_ptau217"]["ckd_subgroup"],
                figures_dir,
            )
            f6 = plot_fig6_tcga_highdim_reduct_and_reactome(
                master["track4_tcga_pancancer"]["highdim_1000gene_reduct"],
                figures_dir,
            )
            master.setdefault("figures", {})["Fig3"] = {
                k: os.path.relpath(v, WORKSPACE_DIR) for k, v in f3.items()
            }
            master["figures"]["Fig6"] = {
                k: os.path.relpath(v, WORKSPACE_DIR) for k, v in f6.items()
            }

        for k_npz in list(npz_data.keys()):
            if npz_data[k_npz].dtype == object:
                npz_data[k_npz] = np.asarray(npz_data[k_npz].tolist(), dtype=str)
        np.savez_compressed(npz_path, **npz_data)
        with open(json_path, "w", encoding="utf-8") as f_out:
            json.dump(_to_json_safe(master), f_out, indent=2)

    # =========================================================================
    # UPGRADE 1: Zero-Leakage In-Fold Real-World Biomarker Discordance Audit
    # =========================================================================
    c1 = pd.read_csv(os.path.join(paths["adni_dir"], "cohort1_adni_ptau181_313roi.csv"))
    c1 = c1[c1["amyloid_gold_pos"].notna()].reset_index(drop=True)
    c2 = pd.read_csv(os.path.join(paths["adni_dir"], "cohort2_adni_ptau217_gfap.csv"))
    c2 = c2[c2["amyloid_gold_pos"].notna()].reset_index(drop=True)

    fids_c1 = npz_data["adni1_fold_ids"].astype(np.int32)
    fids_c2 = npz_data["adni2_fold_ids"].astype(np.int32)

    has_pl1 = (c1["plasma_ptau181_bl"].notna() & (c1["blood_is_bl_visit"].fillna(0) == 1)).values
    clin1_pos = ((c1["DX_bl"] != "CN") | (c1["CDRSB"] >= 0.5)).values
    disc1_mask = np.zeros(len(c1), dtype=bool)
    c1_fold_medians: List[Dict[str, float]] = []
    for f_i in range(5):
        tr_m = fids_c1 != f_i
        va_m = fids_c1 == f_i
        ptau1_med_f = float(np.nanmedian(c1.loc[tr_m & has_pl1, "plasma_ptau181_bl"]))
        hipp1_med_f = float(np.nanmedian(c1.loc[tr_m, "Hippocampus_ICV_ratio"]))
        c1_fold_medians.append({"fold": f_i, "ptau181_median": round(ptau1_med_f, 4), "hipp_icv_median": round(hipp1_med_f, 6)})
        pl1_pos_va = (c1.loc[va_m, "plasma_ptau181_bl"].values >= ptau1_med_f)
        mri1_pos_va = (c1.loc[va_m, "Hippocampus_ICV_ratio"].values <= hipp1_med_f)
        cl1_va = clin1_pos[va_m]
        disc1_mask[va_m] = has_pl1[va_m] & (
            (pl1_pos_va & (~mri1_pos_va) & (~cl1_va)) | ((~pl1_pos_va) & mri1_pos_va & cl1_va)
        )
    conc1_mask = has_pl1 & (~disc1_mask)

    clin2_pos = ((c2["DX_bl"] != "CN") | (c2["CDRSB"] >= 0.5)).values
    disc2_mask = np.zeros(len(c2), dtype=bool)
    c2_fold_medians: List[Dict[str, float]] = []
    for f_i in range(5):
        tr_m = fids_c2 != f_i
        va_m = fids_c2 == f_i
        ptau2_med_f = float(np.nanmedian(c2.loc[tr_m, "pT217_F"]))
        hipp2_med_f = float(np.nanmedian(c2.loc[tr_m, "Hippocampus_ICV_ratio"]))
        c2_fold_medians.append({"fold": f_i, "ptau217_median": round(ptau2_med_f, 4), "hipp_icv_median": round(hipp2_med_f, 6)})
        pl2_pos_va = (c2.loc[va_m, "pT217_F"].values >= ptau2_med_f)
        mri2_pos_va = (c2.loc[va_m, "Hippocampus_ICV_ratio"].values <= hipp2_med_f)
        cl2_va = clin2_pos[va_m]
        disc2_mask[va_m] = (pl2_pos_va & (~mri2_pos_va) & (~cl2_va)) | ((~pl2_pos_va) & mri2_pos_va & cl2_va)
    conc2_mask = ~disc2_mask

    c1_pl_rids = set(c1.loc[has_pl1, "RID"].values.tolist())
    c2_unique_mask = ~c2["RID"].isin(c1_pl_rids).values

    conf_c1 = npz_data["adni1_conf_DeepFuzzyTorch"]
    hes_c1 = npz_data["adni1_hes_DeepFuzzyTorch"]
    comb_c1 = 0.5 * hes_c1 + 0.5 * conf_c1

    conf_c2 = npz_data["adni2_conf_DeepFuzzyTorch"]
    hes_c2 = npz_data["adni2_hes_DeepFuzzyTorch"]
    comb_c2 = 0.5 * hes_c2 + 0.5 * conf_c2

    _, p_c1_conf = mannwhitneyu(conf_c1[disc1_mask], conf_c1[conc1_mask], alternative="two-sided")
    _, p_c1_comb = mannwhitneyu(comb_c1[disc1_mask], comb_c1[conc1_mask], alternative="two-sided")
    _, p_c2_conf = mannwhitneyu(conf_c2[disc2_mask], conf_c2[conc2_mask], alternative="two-sided")
    _, p_c2_comb = mannwhitneyu(comb_c2[disc2_mask], comb_c2[conc2_mask], alternative="two-sided")

    conf_dedup_disc = np.concatenate([conf_c1[disc1_mask], conf_c2[disc2_mask & c2_unique_mask]])
    conf_dedup_conc = np.concatenate([conf_c1[conc1_mask], conf_c2[conc2_mask & c2_unique_mask]])
    comb_dedup_disc = np.concatenate([comb_c1[disc1_mask], comb_c2[disc2_mask & c2_unique_mask]])
    comb_dedup_conc = np.concatenate([comb_c1[conc1_mask], comb_c2[conc2_mask & c2_unique_mask]])
    _, p_dedup_conf = mannwhitneyu(conf_dedup_disc, conf_dedup_conc, alternative="two-sided")
    _, p_dedup_comb = mannwhitneyu(comb_dedup_disc, comb_dedup_conc, alternative="two-sided")

    conf_pool_disc = np.concatenate([conf_c1[disc1_mask], conf_c2[disc2_mask]])
    conf_pool_conc = np.concatenate([conf_c1[conc1_mask], conf_c2[conc2_mask]])
    _, p_pool_conf = mannwhitneyu(conf_pool_disc, conf_pool_conc, alternative="two-sided")

    triage_models = [
        m
        for m in [
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
        if f"adni1_prob_{m}" in npz_data and f"adni2_prob_{m}" in npz_data
    ]
    c1_caches, c2_caches, repro_report = compute_strictly_nested_adni_triage_caches(
        adni_dir=paths["adni_dir"],
        c1_df=c1,
        c2_df=c2,
        fids_c1=fids_c1,
        fids_c2=fids_c2,
        npz_data=npz_data,
        triage_models=triage_models,
        device=device,
    )
    print(
        f"[Upgrade 1] Replayed outer-fold C1/C2 models cleanly | "
        f"max C1 prob diff={max(repro_report['c1_max_prob_abs_diff'].values()):.2e}, "
        f"max C2 prob diff={max(repro_report['c2_max_prob_abs_diff'].values()):.2e}",
        flush=True,
    )

    c1_triage_tbl: Dict[str, Any] = {}
    c2_triage_tbl: Dict[str, Any] = {}
    dedup_triage_tbl: Dict[str, Any] = {}
    y1_all = npz_data["adni1_y"]
    y2_all = npz_data["adni2_y"]
    for m_name in triage_models:
        p1_m = npz_data[f"adni1_prob_{m_name}"]
        h1_m = npz_data[f"adni1_hes_{m_name}"]
        c1_m = npz_data[f"adni1_conf_{m_name}"]
        p2_m = npz_data[f"adni2_prob_{m_name}"]
        h2_m = npz_data[f"adni2_hes_{m_name}"]
        c2_m = npz_data[f"adni2_conf_{m_name}"]

        c1_triage_tbl[m_name] = evaluate_nested_discordant_triage(
            p1_m, h1_m, c1_m, y1_all, disc1_mask, c1_caches[m_name]
        )
        c2_triage_tbl[m_name] = evaluate_nested_discordant_triage(
            p2_m, h2_m, c2_m, y2_all, disc2_mask, c2_caches[m_name]
        )
        dedup_triage_tbl[m_name] = evaluate_dedup_cross_cohort_triage_nested(
            p1_m, h1_m, c1_m, y1_all, disc1_mask, c1_caches[m_name],
            p2_m, h2_m, c2_m, y2_all, disc2_mask, c2_unique_mask, c2_caches[m_name],
        )

        # Save out-of-fold discordant keep20 masks into NPZ for independent audit
        npz_data[f"adni1_disc_keep20_comp_{m_name}"] = (c1_caches[m_name]["keep20_comp"] & disc1_mask).astype(np.int32)
        npz_data[f"adni1_disc_keep20_hes_{m_name}"] = (c1_caches[m_name]["keep20_hes"] & disc1_mask).astype(np.int32)
        npz_data[f"adni2_disc_keep20_comp_{m_name}"] = (c2_caches[m_name]["keep20_comp"] & disc2_mask).astype(np.int32)
        npz_data[f"adni2_disc_keep20_hes_{m_name}"] = (c2_caches[m_name]["keep20_hes"] & disc2_mask).astype(np.int32)

    npz_data["adni1_real_discordant_mask"] = disc1_mask.astype(np.int32)
    npz_data["adni1_real_concordant_mask"] = conc1_mask.astype(np.int32)
    npz_data["adni2_real_discordant_mask"] = disc2_mask.astype(np.int32)
    npz_data["adni2_real_concordant_mask"] = conc2_mask.astype(np.int32)
    npz_data["adni2_nonoverlap_c1plasma_mask"] = c2_unique_mask.astype(np.int32)

    rw_discordance_summary = {
        "definition": (
            "Strictly nested outer-fold clinical biomarker discordance: baseline plasma pTau (>= training-fold median) "
            "contradicting both preserved hippocampal MRI volume (> training-fold median) and normal baseline "
            "cognition (DX_bl=='CN' & CDRSB<0.5), or low plasma pTau (< training-fold median) contradicting both "
            "hippocampal atrophy (<= training-fold median) and clinical impairment; 80th/85th percentile referral "
            "thresholds are calibrated strictly on training-fold discordant patients scored by each outer fold's "
            "trained model (and training-fold medians) with zero cross-fold leakage, then applied out-of-fold and "
            "evaluated symmetrically under both `composite` (0.5*hes + 0.5*conf) and `hes_only` rules."
        ),
        "evaluation_protocol": "strictly_nested_outer_training_fold_calibration",
        "cutoff_protocol": "strictly_nested_outer_fold_training_models_medians_and_percentiles",
        "reproduction_verification": repro_report,
        "cohort1_in_fold_medians": c1_fold_medians,
        "cohort2_in_fold_medians": c2_fold_medians,
        "cohort1_n_observed_plasma": int(np.sum(has_pl1)),
        "cohort1_n_discordant": int(np.sum(disc1_mask)),
        "cohort1_n_concordant": int(np.sum(conc1_mask)),
        "cohort1_conflict_discordant_mean": float(np.mean(conf_c1[disc1_mask])),
        "cohort1_conflict_concordant_mean": float(np.mean(conf_c1[conc1_mask])),
        "cohort1_conflict_ratio": float(np.mean(conf_c1[disc1_mask]) / np.mean(conf_c1[conc1_mask])),
        "cohort1_mannwhitney_p": float(p_c1_conf),
        "cohort1_combined_discordant_mean": float(np.mean(comb_c1[disc1_mask])),
        "cohort1_combined_concordant_mean": float(np.mean(comb_c1[conc1_mask])),
        "cohort1_combined_mannwhitney_p": float(p_c1_comb),
        "cohort2_n_total": int(len(c2)),
        "cohort2_n_discordant": int(np.sum(disc2_mask)),
        "cohort2_n_concordant": int(np.sum(conc2_mask)),
        "cohort2_conflict_discordant_mean": float(np.mean(conf_c2[disc2_mask])),
        "cohort2_conflict_concordant_mean": float(np.mean(conf_c2[conc2_mask])),
        "cohort2_conflict_ratio": float(np.mean(conf_c2[disc2_mask]) / np.mean(conf_c2[conc2_mask])),
        "cohort2_mannwhitney_p": float(p_c2_conf),
        "cohort2_combined_mannwhitney_p": float(p_c2_comb),
        "dedup_n_unique_participants": int(np.sum(has_pl1) + np.sum(c2_unique_mask)),
        "dedup_n_c2_nonoverlapping": int(np.sum(c2_unique_mask)),
        "dedup_n_c2_overlapping_excluded": int(len(c2) - np.sum(c2_unique_mask)),
        "dedup_n_discordant": int(len(conf_dedup_disc)),
        "dedup_n_concordant": int(len(conf_dedup_conc)),
        "dedup_conflict_discordant_mean": float(np.mean(conf_dedup_disc)),
        "dedup_conflict_concordant_mean": float(np.mean(conf_dedup_conc)),
        "dedup_conflict_ratio": float(np.mean(conf_dedup_disc) / np.mean(conf_dedup_conc)),
        "dedup_mannwhitney_p": float(p_dedup_conf),
        "dedup_combined_discordant_mean": float(np.mean(comb_dedup_disc)),
        "dedup_combined_concordant_mean": float(np.mean(comb_dedup_conc)),
        "dedup_combined_mannwhitney_p": float(p_dedup_comb),
        "pooled_n_evaluations": int(len(conf_pool_disc) + len(conf_pool_conc)),
        "pooled_n_discordant": int(len(conf_pool_disc)),
        "pooled_n_concordant": int(len(conf_pool_conc)),
        "pooled_conflict_discordant_mean": float(np.mean(conf_pool_disc)),
        "pooled_conflict_concordant_mean": float(np.mean(conf_pool_conc)),
        "pooled_conflict_ratio": float(np.mean(conf_pool_disc) / np.mean(conf_pool_conc)),
        "pooled_mannwhitney_p": float(p_pool_conf),
        "cohort1_discordant_triage": c1_triage_tbl,
        "cohort2_discordant_triage": c2_triage_tbl,
        "dedup_discordant_triage": dedup_triage_tbl,
    }
    master["track2_adni_cohort2_ptau217"]["real_world_discordance_audit"] = rw_discordance_summary
    master["track2_adni_cohort2_ptau217"]["ckd_subgroup"]["real_world_discordance_audit"] = rw_discordance_summary
    print(
        f"[Upgrade 1] Zero-Leakage In-Fold Real-World Discordance Audit:\n"
        f"  - ADNI C1 (n={rw_discordance_summary['cohort1_n_observed_plasma']}): "
        f"n_disc={rw_discordance_summary['cohort1_n_discordant']} vs n_conc={rw_discordance_summary['cohort1_n_concordant']}, "
        f"ratio={rw_discordance_summary['cohort1_conflict_ratio']:.2f}x, p={p_c1_conf:.3e}\n"
        f"  - ADNI C2 (N={rw_discordance_summary['cohort2_n_total']}): "
        f"n_disc={rw_discordance_summary['cohort2_n_discordant']} vs n_conc={rw_discordance_summary['cohort2_n_concordant']}, "
        f"ratio={rw_discordance_summary['cohort2_conflict_ratio']:.2f}x, p={p_c2_conf:.3e}\n"
        f"  - Deduplicated Unique ADNI (N={rw_discordance_summary['dedup_n_unique_participants']}): "
        f"n_disc={rw_discordance_summary['dedup_n_discordant']} vs n_conc={rw_discordance_summary['dedup_n_concordant']}, "
        f"ratio={rw_discordance_summary['dedup_conflict_ratio']:.2f}x, p={p_dedup_conf:.3e}",
        flush=True,
    )

    # =========================================================================
    # UPGRADE 2: Label-Free Transductive 1,000-Gene TCGA Reduct + 4 Feature-Selection Baselines
    #            + Fair All-1000 Ablation + Custom-Background Reactome Audit
    # =========================================================================
    wsi_dir = paths["wsi_dir"]
    rna_npz_path = os.path.join(wsi_dir, "clinical", "tcga_pancancer_1000genes_rnaseq.npz")
    rna_raw5000_path = os.path.join(wsi_dir, "clinical", "tcga_pancancer_raw_proteincoding_rnaseq.npz")
    gmt_path = os.path.join(wsi_dir, "clinical", "ReactomePathways.gmt")
    reactome_json_path = os.path.join(results_dir, "tcga_top20_reactome_enrichment.json")
    reactome_csv_path = os.path.join(results_dir, "tcga_top20_reactome_pathways.csv")

    if args.audit_only:
        rna_d_audit = np.load(rna_npz_path, allow_pickle=True)
        gene_symbols_audit = rna_d_audit["gene_symbols"].astype(str).tolist()
        tcga_hd = master["track4_tcga_pancancer"]["highdim_1000gene_reduct"]
        consensus_gene_names_audit = [g["gene"] for g in tcga_hd["consensus_top20_genes"]]
        reactome_enrichment_audit = run_custom_background_reactome_enrichment(
            consensus_genes=consensus_gene_names_audit,
            background_1000_genes=gene_symbols_audit,
            gmt_path=gmt_path,
            out_json_path=reactome_json_path,
            out_csv_path=reactome_csv_path,
        )
        tcga_hd.update(_build_tcga_protocol_and_reactome_updates(reactome_enrichment_audit))
        print(
            f"  [Audit-Only Reactome Custom-BG (N_bg=1,000, m={reactome_enrichment_audit['bh_family_size_m']}, n_query=20)] "
            f"k>=2 displayed={reactome_enrichment_audit['n_displayed_pathways_k_ge_2']}, "
            f"raw p<0.05={reactome_enrichment_audit['n_raw_p_lt_005']}, "
            f"BH FDR<0.05={reactome_enrichment_audit['n_fdr_lt_005']}, "
            f"BH FDR<0.10={reactome_enrichment_audit['n_fdr_lt_010']}",
            flush=True,
        )
        _save_outputs()
        print(f"[Audit-Only] Completed cleanly in {time.time() - t0:.2f}s!", flush=True)
        return

    rna_d = np.load(rna_npz_path, allow_pickle=True)
    # Use X_rna_1000_with_nan so sporadic gene NaNs and the 17 missing-RNA patients are imputed strictly per training fold!
    X_rna_1000_nan = (
        rna_d["X_rna_1000_with_nan"].astype(np.float32)
        if "X_rna_1000_with_nan" in rna_d
        else rna_d["X_rna_1000"].astype(np.float32)
    )
    obs_rna_mask = rna_d["observed_rna_mask"].astype(np.float32)
    gene_symbols = rna_d["gene_symbols"].astype(str).tolist()
    entrez_ids = rna_d["entrez_ids"].astype(np.int32).tolist()

    # Load existing 3 modalities (WSI 3072-D, 10-feature signature, 12-D clinical) in exact Track 4 order
    bio_df = pd.read_csv(os.path.join(wsi_dir, "clinical/tcga_precision_oncology_sample_biomarkers.csv"))
    cohorts_tcga = ["TCGA-BRCA", "TCGA-COAD", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LUAD", "TCGA-LUSC", "TCGA-UCEC"]
    gen_cols_tcga = [
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
    wsi_raw_list, gen_list, clin_list, y_tcga_list, os_m_list, os_ev_list = [], [], [], [], [], []
    for c_idx, c_name in enumerate(cohorts_tcga):
        meta_p = os.path.join(wsi_dir, f"tcga_uni2h/{c_name}_uni2h_metadata.csv")
        npz_p = os.path.join(wsi_dir, f"tcga_uni2h/{c_name}_uni2h_slides.npz")
        meta = pd.read_csv(meta_p)
        npz_c = np.load(npz_p, allow_pickle=True)
        w_feats = npz_c["weighted_feats"].astype(np.float32)
        s_feats = npz_c["std_feats"].astype(np.float32)
        merged = meta.merge(bio_df, on="patient_id", how="left")
        valid_vital = merged["vital_status"].isin(["Alive", "Dead"]).values
        merged = merged[valid_vital].reset_index(drop=True)
        w_feats = w_feats[valid_vital]
        s_feats = s_feats[valid_vital]
        os_dead_raw = merged["vital_status"].values == "Dead"
        os_m_raw = merged["cbio_OS_MONTHS"].values.astype(np.float32)
        fallback_days = np.where(
            os_dead_raw,
            merged["days_to_death"].values.astype(np.float32),
            merged["days_to_last_followup"].values.astype(np.float32),
        )
        os_m_raw = np.where(np.isnan(os_m_raw) | (os_m_raw <= 0), fallback_days / 30.4375, os_m_raw)
        has_pfs_raw = merged["cbio_PFS_STATUS"].notna().values
        strict_valid = (~np.isnan(os_m_raw)) & (os_m_raw > 0) & (has_pfs_raw | os_dead_raw)
        merged = merged[strict_valid].reset_index(drop=True)
        w_feats = w_feats[strict_valid]
        s_feats = s_feats[strict_valid]
        os_dead = os_dead_raw[strict_valid]
        os_m = os_m_raw[strict_valid]
        N_c = len(merged)
        wsi_3072 = np.concatenate([w_feats, s_feats], axis=1)
        gen_mat = merged[gen_cols_tcga].values.astype(np.float32)
        stage_str = merged["ajcc_stage"].fillna("").astype(str).str.strip()
        stage_num = np.where(
            stage_str.str.contains("IV"),
            4.0,
            np.where(
                stage_str.str.contains("III"),
                3.0,
                np.where(stage_str.str.contains("II"), 2.0, np.where(stage_str.str.contains("I"), 1.0, np.nan)),
            ),
        )
        t_str = merged["ajcc_t"].fillna("").astype(str).str.strip()
        t_num = np.where(
            t_str.str.contains("T4"),
            4.0,
            np.where(
                t_str.str.contains("T3"),
                3.0,
                np.where(t_str.str.contains("T2"), 2.0, np.where(t_str.str.contains("T1|T0|Tis"), 1.0, np.nan)),
            ),
        )
        n_str = merged["ajcc_n"].fillna("").astype(str).str.strip()
        n_num = np.where(
            n_str.str.contains("N2|N3"),
            2.0,
            np.where(n_str.str.contains("N1"), 1.0, np.where(n_str.str.contains("N0"), 0.0, np.nan)),
        )
        age_v = merged["age_numeric"].values.astype(np.float32)
        gender_str = merged["gender"].fillna("").astype(str).str.strip().str.lower()
        sex_v = np.where(gender_str == "male", 1.0, np.where(gender_str == "female", 0.0, np.nan))
        cohort_oh = np.zeros((N_c, len(cohorts_tcga)), dtype=np.float32)
        cohort_oh[:, c_idx] = 1.0
        clin_mat = np.column_stack([age_v, sex_v, stage_num, t_num, n_num, cohort_oh]).astype(np.float32)
        pfs_prog = merged["cbio_PFS_STATUS"].fillna("").astype(str).str.contains("1:").values
        y_c = (os_dead | pfs_prog).astype(np.float32)
        wsi_raw_list.append(wsi_3072)
        gen_list.append(gen_mat)
        clin_list.append(clin_mat)
        y_tcga_list.append(y_c)
        os_m_list.append(os_m)
        os_ev_list.append(os_dead.astype(np.float32))

    X_wsi_raw = np.vstack(wsi_raw_list)
    X_gen_tcga = np.vstack(gen_list)
    X_clin_tcga = np.vstack(clin_list)
    y_tcga = np.concatenate(y_tcga_list)
    os_months_tcga = np.concatenate(os_m_list)
    os_event_tcga = np.concatenate(os_ev_list)
    N_tcga = len(y_tcga)
    w_tcga = np.where(y_tcga == 1, 1.0, np.minimum(1.0, os_months_tcga / 36.0)).astype(np.float32)

    # Check in-fold top-1,000 MAD vs pan-cancer top-1,000 MAD overlap if raw 5000-gene cache is present
    infold_mad1000_jaccards: List[float] = []
    X_raw_5000: Optional[np.ndarray] = None
    raw_5000_symbols: Optional[List[str]] = None
    if os.path.exists(rna_raw5000_path):
        d5000 = np.load(rna_raw5000_path, allow_pickle=True)
        X_raw_5000 = d5000["X_rna_log2rsem"].astype(np.float32)
        raw_5000_symbols = d5000["gene_symbols"].astype(str).tolist()

    skf_tcga = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    max_k_rna = 30
    optimal_k_rna = 20
    inner_k_cands = [8, 12, 16, 20, 24, 30]
    inner_k_rna_scores = {k: [] for k in inner_k_cands}

    fold_rna_reducers: List[FuzzyRoughFeatureReducer] = []
    gene_freq = {g: 0 for g in gene_symbols}
    mean_univ_dep = np.zeros(len(gene_symbols), dtype=np.float64)
    mean_univ_excess = np.zeros(len(gene_symbols), dtype=np.float64)
    oof_rna_curve_preds = {k: np.zeros(N_tcga, dtype=np.float32) for k in range(1, max_k_rna + 1)}

    # 5 Feature Selectors Compared on Exact Same 1,000-Gene Pool Inside Each Fold
    selector_names = ["FuzzyRough-20", "PCA-20", "Lasso-20", "mRMR-20", "MI-20"]
    sel_fold_sets: Dict[str, List[List[str]]] = {s: [] for s in selector_names}
    sel_fit_times: Dict[str, List[float]] = {s: [] for s in selector_names}
    sel_oof_rna_only = {s: np.zeros(N_tcga, dtype=np.float32) for s in selector_names}
    sel_oof_4mod_prob = {s: np.zeros(N_tcga, dtype=np.float32) for s in selector_names}
    sel_oof_4mod_hes = {s: np.zeros(N_tcga, dtype=np.float32) for s in selector_names}
    sel_oof_no_wsi_clin_prob = {s: np.zeros(N_tcga, dtype=np.float32) for s in selector_names}

    # Strictly nested 5,000 -> in-fold 1,000 MAD -> in-fold 20 FuzzyRough sensitivity check
    nested_5000_oof_prob = np.zeros(N_tcga, dtype=np.float32)
    nested_5000_fold_genes: List[List[str]] = []

    # 4-Modality Multimodal Models (with FAIR All-1000 ablation: identical rough bottleneck & feature gates!)
    models_4mod = {
        "DeepFuzzyTorch_4Mod_Top20": ("DeepFuzzyTorch", True, {}),
        "DeepFuzzyTorch_4Mod_All1000": ("DeepFuzzyTorch", False, {}),  # FAIR: rough_bottleneck=True, feature_gates=True!
        "RCML_4Mod_Top20": ("RCML", True, {}),
        "QMF_4Mod_Top20": ("QMF", True, {}),
        "DrFuse_4Mod_Top20": ("DrFuse", True, {}),
        "EvidentialDL_4Mod_Top20": ("EvidentialDL", True, {}),
        "GatedAttention_4Mod_Top20": ("GatedAttention", True, {}),
        "DempsterShafer_4Mod_Top20": ("DempsterShafer", True, {}),
        "LateAvg_4Mod_Top20": ("LateAvg", True, {}),
        "EarlyConcat_4Mod_Top20": ("EarlyConcat", True, {}),
        "EarlyConcat_4Mod_All1000": ("EarlyConcat", False, {}),
    }
    oof_4mod_probs = {k: np.zeros(N_tcga, dtype=np.float32) for k in models_4mod}
    oof_4mod_hes = {k: np.zeros(N_tcga, dtype=np.float32) for k in models_4mod}

    pancancer_1000_set = set(gene_symbols)

    for f_i, (tr_i, va_i) in enumerate(skf_tcga.split(X_wsi_raw, y_tcga)):
        tr_obs_mask = obs_rna_mask[tr_i] > 0.5
        tr_i_obs = tr_i[tr_obs_mask]

        # In-fold median imputation & standardization of the 1,000-gene candidate pool on observed training patients
        X_rna_tr_obs_raw = X_rna_1000_nan[tr_i_obs]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med_1000_f = np.nanmedian(X_rna_tr_obs_raw, axis=0, keepdims=True)
            med_1000_f = np.where(np.isnan(med_1000_f), 0.0, med_1000_f).astype(np.float32)
        X_rna_tr_obs_imp = np.where(np.isnan(X_rna_tr_obs_raw), med_1000_f, X_rna_tr_obs_raw).astype(np.float32)
        mu_1000_f = np.mean(X_rna_tr_obs_imp, axis=0, keepdims=True)
        sd_1000_f = np.std(X_rna_tr_obs_imp, axis=0, keepdims=True) + 1e-5
        X_rna_tr_obs_std = ((X_rna_tr_obs_imp - mu_1000_f) / sd_1000_f).astype(np.float32)
        y_tr_obs = y_tcga[tr_i_obs]

        # 1. Supervised in-fold fit of Stabilized Pointwise T-Norm FuzzyRoughFeatureReducer using observed training fold + labels y_tr_obs
        t_fr0 = time.perf_counter()
        fr_rna = FuzzyRoughFeatureReducer(
            max_features=max_k_rna,
            sigma=0.85,
            max_samples=4096,
            n_null_permutations=8,
            n_stability_bootstraps=8,
            seed=42 + f_i,
        )
        fr_rna.fit(X_rna_tr_obs_imp, y_tr_obs, feature_names=gene_symbols, device=str(device))
        fr_fit_sec = time.perf_counter() - t_fr0
        sel_fit_times["FuzzyRough-20"].append(fr_fit_sec)

        fold_rna_reducers.append(fr_rna)
        mean_univ_dep += fr_rna.univariate_scores_.astype(np.float64) / 5.0
        mean_univ_excess += fr_rna.univariate_excess_scores_.astype(np.float64) / 5.0
        for g_name in fr_rna.selected_names_[:optimal_k_rna]:
            gene_freq[g_name] += 1

        sel_rna_20 = fr_rna.selected_indices_[:optimal_k_rna]
        sel_fold_sets["FuzzyRough-20"].append([gene_symbols[idx] for idx in sel_rna_20])
        npz_data[f"tcga_fold{f_i}_rna1000_selected_idx"] = np.asarray(sel_rna_20, dtype=np.int32)
        npz_data[f"tcga_fold{f_i}_rna1000_minimal_reduct_idx"] = np.asarray(
            fr_rna.minimal_reduct_indices_, dtype=np.int32
        )

        # Evaluate downstream RNA-only AUROC curve across k = 1..30
        X_rna_tr_30, X_rna_va_30, _, _ = fit_strict_preprocessor(
            [X_rna_1000_nan[tr_i][:, fr_rna.selected_indices_[:max_k_rna]]],
            [X_rna_1000_nan[va_i][:, fr_rna.selected_indices_[:max_k_rna]]],
        )
        for k_step in range(1, max_k_rna + 1):
            clf_k = LogisticRegression(C=0.25, max_iter=300)
            clf_k.fit(X_rna_tr_30[0][tr_obs_mask, :k_step], y_tr_obs)
            oof_rna_curve_preds[k_step][va_i] = clf_k.predict_proba(X_rna_va_30[0][:, :k_step])[:, 1]
        sel_oof_rna_only["FuzzyRough-20"][va_i] = oof_rna_curve_preds[optimal_k_rna][va_i]

        # Inner 80/20 split on raw observed training fold (zero outer/inner-val imputation leakage)
        in_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=300 + f_i)
        in_tr, in_va = next(in_skf.split(X_rna_tr_obs_raw, y_tr_obs))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med_in_f = np.nanmedian(X_rna_tr_obs_raw[in_tr], axis=0, keepdims=True)
            med_in_f = np.where(np.isnan(med_in_f), 0.0, med_in_f).astype(np.float32)
        X_in_tr_imp = np.where(
            np.isnan(X_rna_tr_obs_raw[in_tr]), med_in_f, X_rna_tr_obs_raw[in_tr]
        ).astype(np.float32)
        fr_in = FuzzyRoughFeatureReducer(
            max_features=max_k_rna,
            sigma=0.85,
            max_samples=4096,
            n_null_permutations=6,
            n_stability_bootstraps=4,
            seed=400 + f_i,
        )
        fr_in.fit(X_in_tr_imp, y_tr_obs[in_tr], feature_names=gene_symbols, device=str(device))
        X_in_tr_s, X_in_va_s, _, _ = fit_strict_preprocessor(
            [X_rna_tr_obs_raw[in_tr]], [X_rna_tr_obs_raw[in_va]]
        )
        for ck in inner_k_cands:
            sel_ck = fr_in.selected_indices_[:ck]
            clf_in = LogisticRegression(C=0.25, max_iter=300)
            clf_in.fit(X_in_tr_s[0][:, sel_ck], y_tr_obs[in_tr])
            p_in = clf_in.predict_proba(X_in_va_s[0][:, sel_ck])[:, 1]
            au_in = compute_binary_metrics(p_in, y_tr_obs[in_va])["auroc"]
            ex_in = fr_in.normalized_excess_curve_[ck - 1]
            inner_k_rna_scores[ck].append(float(au_in + 0.08 * ex_in - 0.0006 * ck))

        # 2. Fit 4 Standard Feature-Selection Baselines Strictly on Exact Same Training Fold (PCA-20, Lasso-20, mRMR-20, MI-20)
        # (a) PCA-20
        t_s0 = time.perf_counter()
        pca_20 = PCA(n_components=optimal_k_rna, random_state=42 + f_i)
        pca_20.fit(X_rna_tr_obs_std)
        sel_fit_times["PCA-20"].append(time.perf_counter() - t_s0)

        # (b) Lasso-20 (L1-regularized logistic regression top-20 |beta|)
        t_s0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_clf = LogisticRegression(
                penalty="l1", solver="saga", C=0.15, max_iter=400, random_state=42 + f_i
            )
            lasso_clf.fit(X_rna_tr_obs_std, y_tr_obs)
        lasso_idx = np.argsort(-np.abs(lasso_clf.coef_.ravel()))[:optimal_k_rna].tolist()
        sel_fit_times["Lasso-20"].append(time.perf_counter() - t_s0)
        sel_fold_sets["Lasso-20"].append([gene_symbols[idx] for idx in lasso_idx])

        # (c) mRMR-20
        t_s0 = time.perf_counter()
        mrmr_idx = _select_mrmr_top_k(X_rna_tr_obs_std, y_tr_obs, k=optimal_k_rna)
        sel_fit_times["mRMR-20"].append(time.perf_counter() - t_s0)
        sel_fold_sets["mRMR-20"].append([gene_symbols[idx] for idx in mrmr_idx])

        # (d) MI-20 (Mutual Information)
        t_s0 = time.perf_counter()
        mi_scores = mutual_info_classif(X_rna_tr_obs_std, y_tr_obs, random_state=42 + f_i)
        mi_idx = np.argsort(-mi_scores)[:optimal_k_rna].tolist()
        sel_fit_times["MI-20"].append(time.perf_counter() - t_s0)
        sel_fold_sets["MI-20"].append([gene_symbols[idx] for idx in mi_idx])

        # Prepare WSI 48-D PCA strictly inside training fold
        wsi_tr_mean = np.mean(X_wsi_raw[tr_i], axis=0, keepdims=True)
        wsi_tr_std = np.std(X_wsi_raw[tr_i], axis=0, keepdims=True) + 1e-5
        pca_wsi = PCA(n_components=48, random_state=42 + f_i)
        X_wsi_tr_pca = pca_wsi.fit_transform((X_wsi_raw[tr_i] - wsi_tr_mean) / wsi_tr_std).astype(np.float32)
        X_wsi_va_pca = pca_wsi.transform((X_wsi_raw[va_i] - wsi_tr_mean) / wsi_tr_std).astype(np.float32)

        # Prepare FuzzyRough Top-20 4-Modality tensors
        X_tr_top20_raw = [X_wsi_tr_pca, X_rna_1000_nan[tr_i][:, sel_rna_20], X_gen_tcga[tr_i], X_clin_tcga[tr_i]]
        X_va_top20_raw = [X_wsi_va_pca, X_rna_1000_nan[va_i][:, sel_rna_20], X_gen_tcga[va_i], X_clin_tcga[va_i]]
        X_tr_top20_s, X_va_top20_s, m_tr_top20, m_va_top20, prep_4m = fit_strict_preprocessor_with_params(
            X_tr_top20_raw, X_va_top20_raw
        )
        va_rep_4m, m_rep_4m = apply_strict_preprocessor_params(X_va_top20_raw, prep_4m)
        assert float(np.max(np.abs(X_va_top20_s[1] - va_rep_4m[1]))) == 0.0
        assert float(np.max(np.abs(m_va_top20[:, 1] - m_rep_4m[:, 1]))) == 0.0
        npz_data[f"tcga4mod_fold{f_i}_rna20_median"] = prep_4m[1]["median"]
        npz_data[f"tcga4mod_fold{f_i}_rna20_mean"] = prep_4m[1]["mean"]
        npz_data[f"tcga4mod_fold{f_i}_rna20_std"] = prep_4m[1]["std"]

        # Prepare All-1,000 4-Modality tensors
        X_tr_1000_raw = [X_wsi_tr_pca, X_rna_1000_nan[tr_i], X_gen_tcga[tr_i], X_clin_tcga[tr_i]]
        X_va_1000_raw = [X_wsi_va_pca, X_rna_1000_nan[va_i], X_gen_tcga[va_i], X_clin_tcga[va_i]]
        X_tr_1000_s, X_va_1000_s, m_tr_1000, m_va_1000, _ = fit_strict_preprocessor_with_params(
            X_tr_1000_raw, X_va_1000_raw
        )

        # Train & evaluate all 4-Modality models
        for m_key, (arch_name, use_top20, ab_flags) in models_4mod.items():
            tr_inputs = X_tr_top20_s if use_top20 else X_tr_1000_s
            va_inputs = X_va_top20_s if use_top20 else X_va_1000_s
            m_tr_in = m_tr_top20 if use_top20 else m_tr_1000
            m_va_in = m_va_top20 if use_top20 else m_va_1000

            mod_4m = train_deepfuzzy_or_baseline(
                model_name=arch_name,
                X_tr_list=tr_inputs,
                mask_tr=m_tr_in,
                y_tr=y_tcga[tr_i],
                epochs=28,
                lr=1.8e-3,
                embed_dim=40,
                ablation_flags=ab_flags,
                sample_weight=w_tcga[tr_i],
                seed=200 + f_i,
                device=device,
            )
            with torch.no_grad():
                va_t = [torch.tensor(x, dtype=torch.float32) for x in va_inputs]
                va_m = torch.tensor(m_va_in, dtype=torch.float32)
                out_4m = mod_4m(va_t, modality_masks=va_m)
                oof_4mod_probs[m_key][va_i] = out_4m["prob"].view(-1).numpy()
                oof_4mod_hes[m_key][va_i] = out_4m["hesitancy"].view(-1).numpy()

                if m_key == "DeepFuzzyTorch_4Mod_Top20":
                    sel_oof_4mod_prob["FuzzyRough-20"][va_i] = oof_4mod_probs[m_key][va_i]
                    sel_oof_4mod_hes["FuzzyRough-20"][va_i] = oof_4mod_hes[m_key][va_i]
                    # Evaluate Missing WSI + Missing Clinical/Staging ([0, obs_RNA, obs_Gen, 0])
                    va_m_no_wsi_clin = va_m.clone()
                    va_m_no_wsi_clin[:, 0] = 0.0
                    va_m_no_wsi_clin[:, 3] = 0.0
                    va_t_masked = [
                        torch.zeros_like(va_t[0]),
                        va_t[1],
                        va_t[2],
                        torch.zeros_like(va_t[3]),
                    ]
                    out_nw = mod_4m(va_t_masked, modality_masks=va_m_no_wsi_clin)
                    sel_oof_no_wsi_clin_prob["FuzzyRough-20"][va_i] = out_nw["prob"].view(-1).numpy()

        # Evaluate the 4 baseline feature selectors (PCA-20, Lasso-20, mRMR-20, MI-20) under identical DeepFuzzyTorch protocol
        baseline_sel_specs: List[Tuple[str, Optional[List[int]]]] = [
            ("PCA-20", None),
            ("Lasso-20", lasso_idx),
            ("mRMR-20", mrmr_idx),
            ("MI-20", mi_idx),
        ]
        for s_name, s_idx in baseline_sel_specs:
            if s_name == "PCA-20":
                # Project standardized observed RNA via in-fold PCA-20, preserving NaN rows for the 17 missing-RNA patients
                X_tr_all_imp = np.where(np.isnan(X_rna_1000_nan[tr_i]), med_1000_f, X_rna_1000_nan[tr_i])
                X_va_all_imp = np.where(np.isnan(X_rna_1000_nan[va_i]), med_1000_f, X_rna_1000_nan[va_i])
                rna_tr_20 = pca_20.transform((X_tr_all_imp - mu_1000_f) / sd_1000_f).astype(np.float32)
                rna_va_20 = pca_20.transform((X_va_all_imp - mu_1000_f) / sd_1000_f).astype(np.float32)
                rna_tr_20[obs_rna_mask[tr_i] < 0.5] = np.nan
                rna_va_20[obs_rna_mask[va_i] < 0.5] = np.nan
            else:
                assert s_idx is not None
                rna_tr_20 = X_rna_1000_nan[tr_i][:, s_idx]
                rna_va_20 = X_rna_1000_nan[va_i][:, s_idx]

            X_tr_sel_s, X_va_sel_s, m_tr_sel, m_va_sel = fit_strict_preprocessor(
                [X_wsi_tr_pca, rna_tr_20, X_gen_tcga[tr_i], X_clin_tcga[tr_i]],
                [X_wsi_va_pca, rna_va_20, X_gen_tcga[va_i], X_clin_tcga[va_i]],
            )
            # RNA-only AUROC
            clf_s = LogisticRegression(C=0.25, max_iter=300)
            clf_s.fit(X_tr_sel_s[1][tr_obs_mask], y_tr_obs)
            sel_oof_rna_only[s_name][va_i] = clf_s.predict_proba(X_va_sel_s[1])[:, 1]

            # 4-Modality DeepFuzzyTorch
            mod_s = train_deepfuzzy_or_baseline(
                model_name="DeepFuzzyTorch",
                X_tr_list=X_tr_sel_s,
                mask_tr=m_tr_sel,
                y_tr=y_tcga[tr_i],
                epochs=28,
                lr=1.8e-3,
                embed_dim=40,
                ablation_flags={},
                sample_weight=w_tcga[tr_i],
                seed=200 + f_i,
                device=device,
            )
            with torch.no_grad():
                va_ts = [torch.tensor(x, dtype=torch.float32) for x in X_va_sel_s]
                va_ms = torch.tensor(m_va_sel, dtype=torch.float32)
                out_s = mod_s(va_ts, modality_masks=va_ms)
                sel_oof_4mod_prob[s_name][va_i] = out_s["prob"].view(-1).numpy()
                sel_oof_4mod_hes[s_name][va_i] = out_s["hesitancy"].view(-1).numpy()

                va_ms_nw = va_ms.clone()
                va_ms_nw[:, 0] = 0.0
                va_ms_nw[:, 3] = 0.0
                out_snw = mod_s(
                    [torch.zeros_like(va_ts[0]), va_ts[1], va_ts[2], torch.zeros_like(va_ts[3])],
                    modality_masks=va_ms_nw,
                )
                sel_oof_no_wsi_clin_prob[s_name][va_i] = out_snw["prob"].view(-1).numpy()

        # In-fold top-1,000 MAD re-screening inside pooled transductive 5,000-gene MAD cache -> in-fold Top-20 FuzzyRough sensitivity audit
        if X_raw_5000 is not None and raw_5000_symbols is not None:
            X5_tr_obs = X_raw_5000[tr_i_obs]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med5 = np.nanmedian(X5_tr_obs, axis=0)
                mad5 = np.nanmedian(np.abs(X5_tr_obs - med5[None, :]), axis=0)
            mad5 = np.nan_to_num(mad5, nan=-1.0)
            top1000_f_idx = np.argsort(-mad5, kind="mergesort")[:1000]
            fold1000_syms = [raw_5000_symbols[j] for j in top1000_f_idx]
            fold1000_set = set(fold1000_syms)
            infold_mad1000_jaccards.append(
                len(fold1000_set & pancancer_1000_set) / float(len(fold1000_set | pancancer_1000_set))
            )

            X1k_tr_obs = X5_tr_obs[:, top1000_f_idx]
            med1k = np.nanmedian(X1k_tr_obs, axis=0, keepdims=True)
            X1k_tr_obs_imp = np.where(np.isnan(X1k_tr_obs), med1k, X1k_tr_obs).astype(np.float32)
            fr_nest = FuzzyRoughFeatureReducer(
                max_features=optimal_k_rna,
                sigma=0.85,
                max_samples=4096,
                n_null_permutations=8,
                n_stability_bootstraps=8,
                seed=42 + f_i,
            )
            fr_nest.fit(X1k_tr_obs_imp, y_tr_obs, feature_names=fold1000_syms, device=str(device))
            nest_20_local = fr_nest.selected_indices_[:optimal_k_rna]
            nest_20_global = top1000_f_idx[nest_20_local]
            nested_5000_fold_genes.append([raw_5000_symbols[j] for j in nest_20_global])

            X_tr_n5_s, X_va_n5_s, m_tr_n5, m_va_n5 = fit_strict_preprocessor(
                [X_wsi_tr_pca, X_raw_5000[tr_i][:, nest_20_global], X_gen_tcga[tr_i], X_clin_tcga[tr_i]],
                [X_wsi_va_pca, X_raw_5000[va_i][:, nest_20_global], X_gen_tcga[va_i], X_clin_tcga[va_i]],
            )
            mod_n5 = train_deepfuzzy_or_baseline(
                "DeepFuzzyTorch",
                X_tr_n5_s,
                m_tr_n5,
                y_tcga[tr_i],
                epochs=28,
                lr=1.8e-3,
                embed_dim=40,
                sample_weight=w_tcga[tr_i],
                seed=200 + f_i,
                device=device,
            )
            with torch.no_grad():
                va_tn5 = [torch.tensor(x, dtype=torch.float32) for x in X_va_n5_s]
                va_mn5 = torch.tensor(m_va_n5, dtype=torch.float32)
                nested_5000_oof_prob[va_i] = mod_n5(va_tn5, modality_masks=va_mn5)["prob"].view(-1).numpy()

        print(f"  [Upgrade 2] Completed Label-Free Transductive TCGA 1,000-Gene Fold {f_i+1}/5 (FuzzyRough fit={fr_fit_sec:.2f}s)", flush=True)

    rna_only_auroc_curve = [
        round(float(compute_binary_metrics(oof_rna_curve_preds[k], y_tcga)["auroc"]), 4)
        for k in range(1, max_k_rna + 1)
    ]

    # Parameter reduction calculation for the RNA-Seq branch (1,000 genes vs Top-20 genes at embed_dim=40)
    _m_rna_1000 = UnifiedDeepFuzzyModel([48, 1000, 10, 12], embed_dim=40, num_tsk_concepts=8, num_tsk_rules=6)
    _m_rna_20 = UnifiedDeepFuzzyModel([48, optimal_k_rna, 10, 12], embed_dim=40, num_tsk_concepts=8, num_tsk_rules=6)
    rna_branch_1000 = (
        _m_rna_1000.feature_gates[1].numel()
        + sum(p.numel() for p in _m_rna_1000.modality_projectors[1].parameters())
        + sum(p.numel() for p in _m_rna_1000.ifs_encoders[1].parameters())
    )
    rna_branch_20 = (
        _m_rna_20.feature_gates[1].numel()
        + sum(p.numel() for p in _m_rna_20.modality_projectors[1].parameters())
        + sum(p.numel() for p in _m_rna_20.ifs_encoders[1].parameters())
    )
    rna_branch_red_pct = round(100.0 * (1.0 - rna_branch_20 / float(rna_branch_1000)), 2)
    rna_enc_1000 = sum(p.numel() for p in _m_rna_1000.ifs_encoders[1].parameters())
    rna_enc_20 = sum(p.numel() for p in _m_rna_20.ifs_encoders[1].parameters())
    rna_enc_red_pct = round(100.0 * (1.0 - rna_enc_20 / float(rna_enc_1000)), 2)
    full_4m_1000 = sum(p.numel() for p in _m_rna_1000.parameters())
    full_4m_20 = sum(p.numel() for p in _m_rna_20.parameters())
    full_4m_red_pct = round(100.0 * (1.0 - full_4m_20 / float(full_4m_1000)), 2)

    # Consensus Top-20 genes across 5 folds (ranked by fold frequency, then univariate excess dependency)
    ranked_gene_idx = sorted(
        range(len(gene_symbols)),
        key=lambda idx: (gene_freq[gene_symbols[idx]], mean_univ_excess[idx], mean_univ_dep[idx]),
        reverse=True,
    )
    consensus_top20_genes = [
        {
            "rank": r_i + 1,
            "gene": gene_symbols[idx],
            "entrez_id": int(entrez_ids[idx]),
            "fold_freq": int(gene_freq[gene_symbols[idx]]),
            "univariate_dependency": round(float(mean_univ_dep[idx]), 4),
            "univariate_excess_dependency": round(float(mean_univ_excess[idx]), 4),
        }
        for r_i, idx in enumerate(ranked_gene_idx[:20])
    ]
    consensus_gene_names = [g["gene"] for g in consensus_top20_genes]

    # Exact Custom-Background Hypergeometric Reactome Enrichment (N_bg=1,000, k_genes>=2)
    reactome_json_path = os.path.join(results_dir, "tcga_top20_reactome_enrichment.json")
    reactome_csv_path = os.path.join(results_dir, "tcga_top20_reactome_pathways.csv")
    reactome_enrichment = run_custom_background_reactome_enrichment(
        consensus_genes=consensus_gene_names,
        background_1000_genes=gene_symbols,
        gmt_path=gmt_path,
        out_json_path=reactome_json_path,
        out_csv_path=reactome_csv_path,
    )
    reactome_top_pathways = reactome_enrichment["pathways"][:18]
    print(
        f"  [Reactome Custom-BG Audit (N_bg=1,000, n_query=20)] "
        f"k>=2 pathways={reactome_enrichment['n_tested_pathways_with_k_ge_2']}, "
        f"raw p<0.05={reactome_enrichment['n_raw_p_lt_005']}, "
        f"BH FDR<0.05={reactome_enrichment['n_fdr_lt_005']}, BH FDR<0.10={reactome_enrichment['n_fdr_lt_010']}",
        flush=True,
    )

    # Evaluate Feature-Selection Baselines Table
    valid_os_tcga = (~np.isnan(os_months_tcga)) & (os_months_tcga > 0)
    obs_bool_tcga = obs_rna_mask > 0.5
    feature_selector_comparison: Dict[str, Any] = {}
    p_fr_4mod = sel_oof_4mod_prob["FuzzyRough-20"]
    for s_name in selector_names:
        p4 = sel_oof_4mod_prob[s_name]
        h4 = sel_oof_4mod_hes[s_name]
        prna = sel_oof_rna_only[s_name]
        pnw = sel_oof_no_wsi_clin_prob[s_name]
        m4 = compute_binary_metrics(p4, y_tcga, uncertainty=h4)
        _, lo_s, hi_s = bootstrap_confidence_interval(p4, y_tcga, "auroc", n_bootstraps=1000)
        jacc_info = (
            pairwise_jaccard_summary(sel_fold_sets[s_name])
            if s_name != "PCA-20"
            else {"mean": None, "std": None, "min": None, "max": None, "note": "linear_projection_components"}
        )
        p_vs_fr = (
            1.0
            if s_name == "FuzzyRough-20"
            else paired_bootstrap_pvalue(p_fr_4mod, p4, y_tcga, "auroc", n_bootstraps=1000)
        )
        feature_selector_comparison[s_name] = {
            "auroc_4mod": float(m4["auroc"]),
            "auroc_4mod_ci": [lo_s, hi_s],
            "auprc_4mod": float(m4["auprc"]),
            "brier_4mod": float(m4["brier"]),
            "ece_4mod": float(m4["ece"]),
            "os_c_index_4mod": float(
                _compute_c_index(p4[valid_os_tcga], os_months_tcga[valid_os_tcga], os_event_tcga[valid_os_tcga])
            ),
            "auroc_rna_only_all": float(compute_binary_metrics(prna, y_tcga)["auroc"]),
            "auroc_rna_only_observed": float(compute_binary_metrics(prna[obs_bool_tcga], y_tcga[obs_bool_tcga])["auroc"]),
            "auroc_missing_wsi_and_clin": float(compute_binary_metrics(pnw, y_tcga)["auroc"]),
            "jaccard_5fold": jacc_info,
            "fit_time_mean_sec": round(float(np.mean(sel_fit_times[s_name])), 4),
            "fit_time_std_sec": round(float(np.std(sel_fit_times[s_name], ddof=1)), 4),
            "p_value_4mod_vs_fuzzyrough": p_vs_fr,
        }
    attach_bh_fdr_to_table(
        feature_selector_comparison,
        p_key="p_value_4mod_vs_fuzzyrough",
        q_key="q_bh_4mod_vs_fuzzyrough",
        ref_key="FuzzyRough-20",
    )
    for s_name, s_row in feature_selector_comparison.items():
        j_str = f"{s_row['jaccard_5fold']['mean']:.3f}" if s_row["jaccard_5fold"]["mean"] is not None else "N/A(PCA)"
        print(
            f"  [Selector {s_name:14s}] 4Mod AUROC={s_row['auroc_4mod']:.4f}, "
            f"RNA-only={s_row['auroc_rna_only_observed']:.4f}, NoWSI/Clin={s_row['auroc_missing_wsi_and_clin']:.4f}, "
            f"Jaccard={j_str}, fit={s_row['fit_time_mean_sec']:.2f}s",
            flush=True,
        )

    # Evaluate 4-Modality Pan-Cancer Models
    multimodal_4mod_table: Dict[str, Any] = {}
    p_ours_4m = oof_4mod_probs["DeepFuzzyTorch_4Mod_Top20"]
    for m_key in models_4mod:
        p_arr = oof_4mod_probs[m_key]
        h_arr = oof_4mod_hes[m_key]
        npz_data[f"tcga4mod_prob_{m_key}"] = p_arr
        npz_data[f"tcga4mod_hes_{m_key}"] = h_arr
        m_res = compute_binary_metrics(p_arr, y_tcga, uncertainty=h_arr)
        _, lo_4m, hi_4m = bootstrap_confidence_interval(p_arr, y_tcga, "auroc", n_bootstraps=1000)
        m_res["auroc_ci"] = [lo_4m, hi_4m]
        m_res["os_c_index"] = float(
            _compute_c_index(p_arr[valid_os_tcga], os_months_tcga[valid_os_tcga], os_event_tcga[valid_os_tcga])
        )
        m_res["p_value_vs_ours_top20"] = (
            1.0
            if m_key == "DeepFuzzyTorch_4Mod_Top20"
            else paired_bootstrap_pvalue(p_ours_4m, p_arr, y_tcga, "auroc", n_bootstraps=1000)
        )
        multimodal_4mod_table[m_key] = m_res
    attach_bh_fdr_to_table(
        multimodal_4mod_table,
        p_key="p_value_vs_ours_top20",
        q_key="q_bh_vs_ours_top20",
        ref_key="DeepFuzzyTorch_4Mod_Top20",
    )
    for m_key, m_res in multimodal_4mod_table.items():
        lo_4m, hi_4m = m_res["auroc_ci"]
        print(
            f"  [TCGA 4-Modality {m_key:28s}] AUROC={m_res['auroc']:.4f} [{lo_4m:.4f}, {hi_4m:.4f}], "
            f"OS C-index={m_res['os_c_index']:.4f}, ECE={m_res['ece']:.4f}, "
            f"p={m_res['p_value_vs_ours_top20']:.4f}, q_BH={m_res['q_bh_vs_ours_top20']:.4f}",
            flush=True,
        )

    dep_rna_mat = np.asarray([fr.dependency_curve_ for fr in fold_rna_reducers], dtype=np.float64)
    null_dep_rna_mat = np.asarray([fr.null_dependency_curve_ for fr in fold_rna_reducers], dtype=np.float64)
    exc_dep_rna_mat = np.asarray([fr.excess_dependency_curve_ for fr in fold_rna_reducers], dtype=np.float64)
    norm_exc_rna_mat = np.asarray([fr.normalized_excess_curve_ for fr in fold_rna_reducers], dtype=np.float64)
    bnd_rna_mat = np.asarray([fr.boundary_curve_ for fr in fold_rna_reducers], dtype=np.float64)
    fr_jaccard_5fold = pairwise_jaccard_summary(sel_fold_sets["FuzzyRough-20"])

    nested_5000_summary: Dict[str, Any] = {}
    if len(infold_mad1000_jaccards) == 5:
        m_n5 = compute_binary_metrics(nested_5000_oof_prob, y_tcga)
        _, lo_n5, hi_n5 = bootstrap_confidence_interval(nested_5000_oof_prob, y_tcga, "auroc", n_bootstraps=1000)
        nested_5000_summary = {
            "in_fold_top1000_mad_vs_pancancer_jaccard_mean": round(float(np.mean(infold_mad1000_jaccards)), 4),
            "in_fold_top1000_mad_vs_pancancer_jaccard_per_fold": [round(x, 4) for x in infold_mad1000_jaccards],
            "nested_5000_to_1000_to_20_auroc_4mod": float(m_n5["auroc"]),
            "nested_5000_to_1000_to_20_auroc_ci": [lo_n5, hi_n5],
            "nested_5000_to_1000_to_20_jaccard_5fold": pairwise_jaccard_summary(nested_5000_fold_genes),
        }

    tcga_highdim_summary = {
        "n_total_patients": int(N_tcga),
        "n_observed_rnaseq": int(np.sum(obs_rna_mask)),
        "n_missing_rnaseq": int(N_tcga - np.sum(obs_rna_mask)),
        "total_input_genes": 1000,
        "optimal_k": optimal_k_rna,
        "gene_compression_pct": 98.0,
        "all1000_ablation_fairness_note": "DeepFuzzyTorch_4Mod_All1000 keeps use_rough_bottleneck=True and use_feature_gates=True identical to DeepFuzzyTorch_4Mod_Top20",
        "rna_branch_params_1000": int(rna_branch_1000),
        "rna_branch_params_20": int(rna_branch_20),
        "rna_branch_param_reduction_pct": rna_branch_red_pct,
        "rna_ifs_encoder_only_params_1000": int(rna_enc_1000),
        "rna_ifs_encoder_only_params_20": int(rna_enc_20),
        "rna_ifs_encoder_only_reduction_pct": rna_enc_red_pct,
        "full_4mod_params_1000": int(full_4m_1000),
        "full_4mod_params_20": int(full_4m_20),
        "full_4mod_param_reduction_pct": full_4m_red_pct,
        "inner_cv_k_scores": {str(k): round(float(np.mean(v)), 4) for k, v in inner_k_rna_scores.items()},
        "inner_cv_argmax_k": int(
            max(inner_k_rna_scores.keys(), key=lambda kk: float(np.mean(inner_k_rna_scores[kk])))
        ),
        "cross_fold_jaccard_stability": fr_jaccard_5fold,
        "in_fold_5000_mad_sensitivity": nested_5000_summary,
        "dependency_curve_mean": np.mean(dep_rna_mat, axis=0).tolist(),
        "dependency_curve_std": np.std(dep_rna_mat, axis=0).tolist(),
        "null_dependency_curve_mean": np.mean(null_dep_rna_mat, axis=0).tolist(),
        "excess_dependency_curve_mean": np.mean(exc_dep_rna_mat, axis=0).tolist(),
        "excess_dependency_curve_std": np.std(exc_dep_rna_mat, axis=0).tolist(),
        "normalized_excess_curve_mean": np.mean(norm_exc_rna_mat, axis=0).tolist(),
        "boundary_curve_mean": np.mean(bnd_rna_mat, axis=0).tolist(),
        "rna_only_auroc_curve": rna_only_auroc_curve,
        "per_fold_selected_gene_indices": [
            [int(i) for i in fr.selected_indices_[:optimal_k_rna]] for fr in fold_rna_reducers
        ],
        "per_fold_selected_gene_names": [fr.selected_names_[:optimal_k_rna] for fr in fold_rna_reducers],
        "minimal_reduct_sizes_all_folds": [len(fr.minimal_reduct_indices_) for fr in fold_rna_reducers],
        "consensus_top20_genes": consensus_top20_genes,
        "feature_selector_comparison": feature_selector_comparison,
        "multimodal_4mod_table": multimodal_4mod_table,
    }
    tcga_highdim_summary.update(_build_tcga_protocol_and_reactome_updates(reactome_enrichment))
    master["track4_tcga_pancancer"]["highdim_1000gene_reduct"] = tcga_highdim_summary

    _save_outputs()

    print(f"[Upgrades 1 & 2] Completed cleanly in {time.time() - t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
