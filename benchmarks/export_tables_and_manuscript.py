"""
Automatically formats all quantitative benchmark results from `results/benchmark_results.json`
into publication-ready Markdown reports (`results/BENCHMARK_TABLES_SUMMARY.md` and
`results/BENCHMARK_SUMMARY_REPORT.md`), with dynamic bolding of true numeric winners,
Benjamini-Hochberg FDR q-values across Tables 1-6, 2023-2024 SOTA baselines (`RCML (adapted)`,
`QMF (adapted)`, `DrFuse (adapted)`, `MedianPool6FMs`, `Logit-Tent (adapted)`), equalized-supervision fairness
checks, label-free unsupervised transductive TCGA 1,000-gene feature-selector comparisons &
custom-background Reactome enrichment (`m=676` predeclared `K_bg>=2` pathways), and Track 6 latency/memory scaling.
"""

from __future__ import annotations
import json
import math
import os
import sys
from typing import Any, Dict, List

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

RESULTS_JSON = os.path.join(WORKSPACE_DIR, "results/benchmark_results.json")
TABLES_MD = os.path.join(WORKSPACE_DIR, "results/BENCHMARK_TABLES_SUMMARY.md")
REPORT_MD = os.path.join(WORKSPACE_DIR, "results/BENCHMARK_SUMMARY_REPORT.md")


def format_pval(p: float | None, show_one: bool = False) -> str:
    if p is None:
        return "---"
    if p >= 0.9999 and not show_one:
        return "---"
    if p < 0.001:
        return f"{p:.2e}" if p < 1e-4 else "<0.001"
    return f"{p:.3f}"


def _benjamini_hochberg(pvals: Dict[str, float]) -> Dict[str, float]:
    """Adjust a family of valid comparator p-values by the BH step-up procedure."""
    ranked = sorted(pvals.items(), key=lambda item: item[1])
    adjusted: Dict[str, float] = {}
    running_min = 1.0
    for index in range(len(ranked) - 1, -1, -1):
        key, p = ranked[index]
        running_min = min(running_min, p * len(ranked) / (index + 1))
        adjusted[key] = running_min
    return adjusted


def _bold_winner(vals: List[float], fmt: str = ".4f", higher_is_better: bool = True) -> List[str]:
    if not vals:
        return []
    best = max(vals) if higher_is_better else min(vals)
    out = []
    for v in vals:
        s = f"{v:{fmt}}"
        if abs(v - best) < 1e-6:
            out.append(f"**{s}**")
        else:
            out.append(s)
    return out


def generate_markdown_summary(res: Dict[str, Any]) -> str:
    t0 = res["track0_theorems"]
    t1 = res["track1_adni_cohort1"]
    t2 = res["track2_adni_cohort2_ptau217"]
    t3 = res["track3_oasis3_external"]
    t4 = res["track4_tcga_pancancer"]
    t5 = res["track5_multi_fm_consensus_tta"]
    t6 = res.get("track6_runtime_complexity")

    md: List[str] = []
    md.append("# `DeepFuzzyTorch` Multi-Dataset Benchmark & Mathematical Verification Report\n")
    md.append(f"- **Execution Device**: `{res.get('device', 'cpu')}` (`{res.get('gpu_name', 'CPU')}`)")
    md.append(f"- **Total Primary Suite Execution Time**: `{res['elapsed_seconds']}s`")
    md.append(
        f"- **Theorem 1 Simplex Closure Max Residual ($N=250$)**: `{t0['theorem1_max_closure_error']:.2e}` ($\\mu + \\nu + \\pi = 1$)"
    )
    md.append(
        f"- **Theorem 2 Missing-Modality Degeneration Max Error ($\\gamma_c=0.85$)**: `{t0['theorem2_max_abs_diff']:.2e}` (Missing Choquet Weight = `{t0['theorem2_missing_weight_max']:.2e}`)"
    )
    md.append(
        f"- **Theorem 3 Fixed-Conflict Monotonicity Min Partial Derivative**: `{t0.get('theorem3_min_fixed_conflict_partial_derivative', 0.0):.4f}` (Analytic vs. FD Max Error = `{t0.get('theorem3_max_analytic_vs_fd_error', 0.0):.2e}`)"
    )
    md.append(
        f"- **Theorem 5 Fuzzy-Rough Inclusion Max Violation**: `{t0.get('theorem5_max_inclusion_violation', 0.0):.2e}`"
    )
    md.append(
        f"- **White-Box TSK Text Rule Reconstruction Max Error (Precision=8, With Concept Mask)**: `{t1.get('tsk_reconstruction_max_abs_error', 0.0):.3e}`"
    )
    md.append(
        "- **AURC Tie-Handling Convention**: `deepfuzzy.metrics.compute_aurc` averages 0-1 errors across equal-uncertainty ties for permutation invariance, whereas frozen `results/benchmark_results.json` stores initial index-ordered execution values (`31/39` Track 1–4 models including all 4 `DeepFuzzyTorch` tracks match to `0.00e+00`; `8/39` baseline/ablation values differ by $\\le 1.92 \\times 10^{-5}$; `39/39` are identical at 4 decimal places).\n"
    )

    # =========================================================================
    # Table 1: ADNI Cohort 1 Main 5-Fold Comparison
    # =========================================================================
    n_c1 = t1.get("n_total", 1137)
    n_c1_bl = t1.get("n_same_visit_bl_ref", 924)
    n_c1_comp = t1.get("n_complete_4mod", 854)
    adas_miss = t1.get("preprocessing_audit", {}).get("adas13_missing_count_total", 6)
    md.append(
        f"\n## Table 1: ADNI Cohort 1 ($N={n_c1}$ Full Cohort With Natural Missingness; Same-Visit Baseline Reference $n={n_c1_bl}$; Complete 4-Modality Subset $N_{{\\text{{comp}}}}={n_c1_comp}$) 5-Fold Cross-Validation\n"
    )
    md.append(
        f"> **Population & Imputation Accounting**: ADNI Cohort 1 ($N={n_c1}$) includes $1.3\\%$ missing MRI ($1,122/{n_c1}$), $23.3\\%$ missing Plasma pTau181 ($872/{n_c1}$), $10.5\\%$ missing FDG-PET ($1,018/{n_c1}$), and $0\\%$ missingness for Clin as a modality ($6/{n_c1}$ [$0.53\\%$] missing baseline `ADAS13` entries are imputed strictly in-fold using observed training-fold medians). Exactly $N_{{\\text{{comp}}}}={n_c1_comp}$ participants ($75.1\\%$) have all four baseline modalities observed simultaneously.\n"
    )
    md.append(
        f"> **Cohort-1 BH Family**: The predeclared $m=20$ family includes 14 main comparators and 6 Table-3 component ablations. The 4 equal-loss/cold-start controls report raw paired $p$-values only; complete-case $N_{{\\text{{comp}}}}={n_c1_comp}$ paired tests reside in Table 1C.\n"
    )
    md.append(
        f"| Method | Full AUROC ($N={n_c1}$, 95% CI) | Same-Visit AUROC ($n={n_c1_bl}$) | Complete 4-Mod AUROC ($N_{{\\text{{comp}}}}={n_c1_comp}$) | AUPRC | Acc | F1 | Brier $\\downarrow$ | ECE $\\downarrow$ | AURC $\\downarrow$ | 15-Subset Macro (Full / Elig) | MRR (14 Inc.) | Raw $p$ | BH $q_{{\\text{{BH}}}}$ |"
    )
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    order_t1 = [
        ("EarlyConcat", "EarlyConcat-MLP"),
        ("LateAvg", "LateAvg-Ensemble"),
        ("DeepTSK", "DeepTSK-RuleNet"),
        ("DempsterShafer", "Dempster-Shafer Fusion"),
        ("GatedAttention", "GatedAttention-Transformer"),
        ("EvidentialDL", "EvidentialDL (Dirichlet TMC)"),
        ("RCML", "RCML (adapted, Xu et al., AAAI'24)"),
        ("QMF", "QMF (adapted, Zhang et al., ICML'23)"),
        ("DrFuse", "DrFuse (adapted, Yao et al., AAAI'24)"),
        ("DDEF", "DDEF (adapted, Shao et al., Inf. Fusion'24)"),
        ("FUML", "FUML (adapted, Duan et al., ICML'25)"),
        ("XGBoost", "XGBoost (Native NaN + Indicators)"),
        ("CatBoost", "CatBoost (NaN='Min' + Indicators)"),
        ("TabPFN", "TabPFN v2 (Nature'25 + Indicators)"),
        ("DeepFuzzyTorch", "DeepFuzzyTorch (Ours)"),
    ]
    present_t1 = [(k, lbl) for k, lbl in order_t1 if k in t1["main_table"]]
    c4_tbl = t1.get("complete_4mod_854_table", {})
    au_s = _bold_winner([t1["main_table"][k]["auroc"] for k, _ in present_t1])
    au924_s = _bold_winner([t1.get("same_visit_bl_924_table", {}).get(k, t1["main_table"][k])["auroc"] for k, _ in present_t1])
    au854_s = _bold_winner([c4_tbl.get(k, {}).get("auroc", t1["main_table"][k].get("complete_4mod_854_auroc", 0.0)) for k, _ in present_t1])
    ap_s = _bold_winner([t1["main_table"][k]["auprc"] for k, _ in present_t1])
    ac_s = _bold_winner([t1["main_table"][k]["acc"] for k, _ in present_t1])
    f1_s = _bold_winner([t1["main_table"][k]["f1"] for k, _ in present_t1])
    br_s = _bold_winner([t1["main_table"][k]["brier"] for k, _ in present_t1], higher_is_better=False)
    ec_s = _bold_winner([t1["main_table"][k]["ece"] for k, _ in present_t1], higher_is_better=False)
    ar_s = _bold_winner([t1["main_table"][k]["aurc"] for k, _ in present_t1], higher_is_better=False)
    m15_s = _bold_winner([t1["main_table"][k].get("macro_15subset_auroc", 0.0) for k, _ in present_t1])
    m15e_s = _bold_winner([t1["main_table"][k].get("macro_15subset_auroc_eligible", 0.0) for k, _ in present_t1])
    mrr_s = _bold_winner([t1["main_table"][k].get("mrr_14incomplete", 0.0) for k, _ in present_t1])

    for i, (key, label) in enumerate(present_t1):
        m = t1["main_table"][key]
        ci = m["auroc_ci"]
        lbl_fmt = f"**{label}**" if key == "DeepFuzzyTorch" else label
        md.append(
            f"| {lbl_fmt} | {au_s[i]} ({ci[0]:.4f}–{ci[1]:.4f}) | {au924_s[i]} | {au854_s[i]} | {ap_s[i]} | {ac_s[i]} | {f1_s[i]} | "
            f"{br_s[i]} | {ec_s[i]} | {ar_s[i]} | {m15_s[i]} / {m15e_s[i]} | {mrr_s[i]} | "
            f"{format_pval(m.get('p_value_vs_ours'))} | {format_pval(m.get('q_bh_vs_ours'))} |"
        )

    if c4_tbl:
        md.append(f"\n### Table 1C: Complete 4-Modality Subset Audit ($N_{{\\text{{comp}}}}={n_c1_comp}$ Observed Across MRI, Plasma, FDG, and Clin)\n")
        md.append("| Method | Complete-4-Mod AUROC (95% CI) | AUPRC | Acc | F1 | Brier $\\downarrow$ | ECE $\\downarrow$ | AURC $\\downarrow$ | Raw $p$ | BH $q_{\\text{BH}}$ |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        p_c4 = [(k, lbl) for k, lbl in present_t1 if k in c4_tbl]
        c4_au = _bold_winner([c4_tbl[k]["auroc"] for k, _ in p_c4])
        c4_ap = _bold_winner([c4_tbl[k]["auprc"] for k, _ in p_c4])
        c4_ac = _bold_winner([c4_tbl[k]["acc"] for k, _ in p_c4])
        c4_f1 = _bold_winner([c4_tbl[k]["f1"] for k, _ in p_c4])
        c4_br = _bold_winner([c4_tbl[k]["brier"] for k, _ in p_c4], higher_is_better=False)
        c4_ec = _bold_winner([c4_tbl[k]["ece"] for k, _ in p_c4], higher_is_better=False)
        c4_ar = _bold_winner([c4_tbl[k]["aurc"] for k, _ in p_c4], higher_is_better=False)
        for i, (key, label) in enumerate(p_c4):
            cm = c4_tbl[key]
            cci = cm["auroc_ci"]
            lbl_fmt = f"**{label}**" if key == "DeepFuzzyTorch" else label
            md.append(
                f"| {lbl_fmt} | {c4_au[i]} ({cci[0]:.4f}–{cci[1]:.4f}) | {c4_ap[i]} | {c4_ac[i]} | {c4_f1[i]} | "
                f"{c4_br[i]} | {c4_ec[i]} | {c4_ar[i]} | {format_pval(cm.get('p_value_vs_ours'))} | {format_pval(cm.get('q_bh_vs_ours'))} |"
            )

    if "conflict_det_auroc_g08" in t1["main_table"].get("DeepFuzzyTorch", {}):
        md.append("\n### Table 1D: Separated Cross-Modal Conflict Anomaly Detection & 20%-Referral Under opposite_swap ($\\gamma=0.8$, ADNI Cohort 1)\n")
        md.append("| Method | Detect AUROC (Composite $0.5 C_{\\mathcal{O}}+0.5\\pi_f$) | Detect AUROC (Pure Conflict $C_{\\mathcal{O}}$) | Detect AUROC (Pure Hesitancy $\\pi_f$) | 20% Refer AUROC (Composite) | 20% Refer AUROC (Pure $C_{\\mathcal{O}}$) | 20% Refer AUROC (Pure $\\pi_f$) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for key, label in present_t1:
            mm = t1["main_table"][key]
            lbl_fmt = f"**{label}**" if key == "DeepFuzzyTorch" else label
            dc = mm.get("conflict_det_auroc_g08")
            dco = mm.get("conflict_det_auroc_conflict_only_g08")
            dh = mm.get("conflict_det_auroc_hes_only_g08")
            rc = mm.get("referral_20pct_auroc_g08")
            rco = mm.get("referral_20pct_auroc_conflict_only_g08")
            rh = mm.get("referral_20pct_auroc_hes_only_g08")
            md.append(
                f"| {lbl_fmt} | {f'{dc:.4f}' if dc is not None else '---'} | "
                f"{f'{dco:.4f}' if dco is not None else 'N/A (Single-Table)'} | "
                f"{f'{dh:.4f}' if dh is not None else '---'} | "
                f"{f'{rc:.4f}' if rc is not None else '---'} | "
                f"{f'{rco:.4f}' if rco is not None else 'N/A'} | "
                f"{f'{rh:.4f}' if rh is not None else '---'} |"
            )

    # Table 1B: Equalized-Supervision Fairness Check
    eq_ab = t1.get("equalized_supervision_ablation", {})
    if eq_ab:
        md.append("\n### Table 1B: Equalized-Supervision & Cold-Start Fairness Audit on ADNI Cohort 1 ($N=1,137$)\n")
        md.append(
            "| Supervision Regime / Variant | Full AUROC (95% CI) | AUPRC | Acc | F1 | Brier $\\downarrow$ | ECE $\\downarrow$ | AURC $\\downarrow$ | 15-Subset Macro AUROC | Incomplete MRR | Raw $p$ vs. Ours |"
        )
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        eq_order = [
            ("EarlyConcat_EqLoss", "EarlyConcat + Brier & Shared-Fused AuxWeight*"),
            ("DempsterShafer_EqLoss", "Dempster-Shafer + Matched Brier & Aux Supervision"),
            ("GatedAttention_EqLoss", "GatedAttention + Matched Brier & Aux Supervision"),
            ("DeepFuzzyTorch_ColdNoAux", "DeepFuzzyTorch (Joint Cold-Start + No Brier/Aux)"),
            ("DeepFuzzyTorch", "**DeepFuzzyTorch (Standard Full Objective)**"),
        ]
        eq_rows = [(ek, elbl, eq_ab.get(ek) or t1["main_table"].get(ek)) for ek, elbl in eq_order]
        eq_rows = [(ek, elbl, em) for ek, elbl, em in eq_rows if em]
        eq_au_b = _bold_winner([em["auroc"] for _, _, em in eq_rows])
        eq_ap_b = _bold_winner([em["auprc"] for _, _, em in eq_rows])
        eq_ac_b = _bold_winner([em["acc"] for _, _, em in eq_rows])
        eq_f1_b = _bold_winner([em["f1"] for _, _, em in eq_rows])
        eq_br_b = _bold_winner([em["brier"] for _, _, em in eq_rows], higher_is_better=False)
        eq_ec_b = _bold_winner([em["ece"] for _, _, em in eq_rows], higher_is_better=False)
        eq_ar_b = _bold_winner([em["aurc"] for _, _, em in eq_rows], higher_is_better=False)
        eq_m15_b = _bold_winner([em.get("macro_15subset_auroc", 0.0) for _, _, em in eq_rows])
        eq_mrr_b = _bold_winner([em.get("mrr_14incomplete", 0.0) for _, _, em in eq_rows])
        for idx_eq, (ek, elbl, em) in enumerate(eq_rows):
            eci = em.get("auroc_ci", [em["auroc"], em["auroc"]])
            md.append(
                f"| {elbl} | {eq_au_b[idx_eq]} ({eci[0]:.4f}–{eci[1]:.4f}) | {eq_ap_b[idx_eq]} | "
                f"{eq_ac_b[idx_eq]} | {eq_f1_b[idx_eq]} | {eq_br_b[idx_eq]} | {eq_ec_b[idx_eq]} | "
                f"{eq_ar_b[idx_eq]} | {eq_m15_b[idx_eq]} | {eq_mrr_b[idx_eq]} | "
                f"{format_pval(em.get('p_value_vs_ours'))} |"
            )
        md.append(
            "\n*\\*Methodological disclosure:* `EarlyConcatMLP` lacks separate unimodal heads (`modality_probs[m] == prob`), "
            "so its `_EqLoss` control optimizes $(1 + 0.75|\\mathcal{O}_i|/M)\\mathrm{BCE}(p_{\\mathrm{fused}}, y) + 0.35\\,\\mathrm{Brier}$ "
            "($1.75\\,\\mathrm{BCE} + 0.35\\,\\mathrm{Brier}$ when all $4$ modalities are observed), whereas `DempsterShafer_EqLoss` and "
            "`GatedAttention_EqLoss` supervise distinct unimodal predictions; `DeepFuzzyTorch_ColdNoAux` jointly removes correlation "
            "warm-starting, Brier, and unimodal auxiliary BCE rather than isolating a single factor."
        )

    # =========================================================================
    # Table 2: Ablation Study on ADNI Cohort 1
    # =========================================================================
    rs_c1 = t1.get("rough_summary", {})
    jacc_c1 = rs_c1.get("pairwise_fold_jaccard", rs_c1.get("cross_fold_jaccard_stability", {})).get("mean")
    jacc_c1_note = f", 5-Fold MRI Reduct Jaccard = `{jacc_c1:.4f}`" if jacc_c1 is not None else ""
    md.append(
        f"\n## Table 2: Component Ablation Study of `DeepFuzzyTorch` on ADNI Cohort 1 ($N=1,137${jacc_c1_note}; $\\Delta$ AUROC computed from unrounded OOF AUROCs)\n"
    )
    md.append(
        "| Ablation Variant | Full-Set AUROC (95% CI) | ECE $\\downarrow$ | AURC $\\downarrow$ | 15-Subset Macro AUROC | Incomplete MRR | Conflict Ref-20% AUROC ($\\gamma=0.8$) | $\\Delta$ AUROC | Raw $p$ | BH $q_{\\text{BH}}$ |"
    )
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    base_auroc = t1["main_table"]["DeepFuzzyTorch"]["auroc"]
    ab_order = [
        ("DeepFuzzyTorch", "**DeepFuzzyTorch (Full Unified Framework, $K^*=24$)**"),
        ("Ablation_wo_VacuityMask", "w/o Exact Vacuity Masking ($\\langle 0,0,1\\rangle \\to$ Zero Imputation)"),
        ("Ablation_wo_Choquet", "w/o Non-Additive Choquet Capacity (Uniform $1/\\lvert\\mathcal{O}\\rvert$)"),
        ("Ablation_wo_ConflictDiscount", "w/o Conflict Hesitancy Discount ($\\gamma_c = 0$)"),
        ("Ablation_wo_FuzzyRough", "w/o Fuzzy-Rough Gate (Matched $K^*=24$ MRI ROIs)"),
        ("Ablation_wo_FuzzyRough_Unreduced320", "w/o Fuzzy-Rough Reduct, FR Gate & Input Gates (Unreduced 320 MRI ROIs)"),
        ("Ablation_wo_TSKHead", "w/o White-Box Concept TSK Rule Head"),
    ]
    for key, label in ab_order:
        if key not in t1["main_table"]:
            continue
        m = t1["main_table"][key]
        ci = m["auroc_ci"]
        delta = m["auroc"] - base_auroc
        d_str = "---" if key == "DeepFuzzyTorch" else f"{delta:+.4f}"
        mac15 = m.get("macro_15subset_auroc", 0.0)
        mrr = m.get("mrr_14incomplete", 0.0)
        ref_g08 = m.get("referral_20pct_auroc_g08", 0.0)
        md.append(
            f"| {label} | {m['auroc']:.4f} ({ci[0]:.4f}–{ci[1]:.4f}) | {m['ece']:.4f} | {m['aurc']:.4f} | "
            f"{mac15:.4f} | {mrr:.4f} | {ref_g08:.4f} | {d_str} | {format_pval(m.get('p_value_vs_ours'))} | {format_pval(m.get('q_bh_vs_ours'))} |"
        )

    # =========================================================================
    # Table 3: Exhaustive 2^4 - 1 = 15 Missing-Modality Combinatorial Benchmark
    # =========================================================================
    md.append(
        "\n## Table 3: Exhaustive $2^4-1 = 15$ Missing-Modality Combinatorial Benchmark (ADNI $N=1,137$: Full-Population / Eligible-Subset AUROC)\n"
    )
    m_keys_t3 = [
        k
        for k in [
            "EarlyConcat", "GatedAttention", "EvidentialDL", "RCML", "QMF", "DrFuse",
            "DDEF", "FUML", "XGBoost", "CatBoost", "TabPFN", "DeepFuzzyTorch",
        ]
        if k in t1["combinatorial_15subsets"]
    ]
    disp_t3 = {
        "RCML": "RCML (adapt.)",
        "QMF": "QMF (adapt.)",
        "DrFuse": "DrFuse (adapt.)",
        "DDEF": "DDEF (adapt.)",
        "FUML": "FUML (adapt.)",
        "TabPFN": "TabPFN v2",
    }
    hdr_cols = " | ".join([f"{disp_t3.get(k, k)} (Full / Elig)" for k in m_keys_t3])
    sep_cols = " | ".join([":---:" for _ in m_keys_t3])
    md.append(f"| Modality Subset | $\\lvert\\mathcal{{O}}\\rvert$ | Eligible $N$ | Complete $N$ | {hdr_cols} | Ours $\\pi_{{\\text{{fused}}}}$ |")
    md.append(f"| :--- | :---: | :---: | :---: | {sep_cols} | :---: |")
    subsets_15 = list(t1["combinatorial_15subsets"]["DeepFuzzyTorch"]["subsets"].keys())
    for s in subsets_15:
        sub_info = t1["combinatorial_15subsets"]["DeepFuzzyTorch"]["subsets"][s]
        k_act = int(sub_info["num_modalities"])
        n_elig = int(sub_info["n_eligible"])
        n_comp = int(sub_info["n_complete_subset"])
        full_vals = [t1["combinatorial_15subsets"][mk]["subsets"][s]["auroc"] for mk in m_keys_t3]
        elig_vals = [t1["combinatorial_15subsets"][mk]["subsets"][s]["auroc_eligible"] for mk in m_keys_t3]
        full_strs = _bold_winner(full_vals)
        elig_strs = _bold_winner(elig_vals)
        cells = " | ".join([f"{full_strs[i]} / {elig_strs[i]}" for i in range(len(m_keys_t3))])
        df_pi = sub_info["mean_hesitancy"]
        md.append(f"| `{s}` | {k_act} | {n_elig} | {n_comp} | {cells} | {df_pi:.3f} |")

    mac_full = _bold_winner([t1["combinatorial_15subsets"][mk]["macro_avg_auroc"] for mk in m_keys_t3])
    mac_elig = _bold_winner([t1["combinatorial_15subsets"][mk]["macro_avg_auroc_eligible"] for mk in m_keys_t3])
    mac_cells = " | ".join([f"{mac_full[i]} / {mac_elig[i]}" for i in range(len(m_keys_t3))])
    md.append(f"| **Macro-Average Across All 15 Subsets** | **1–4** | --- | --- | {mac_cells} | --- |")

    mrr_full = _bold_winner([t1["combinatorial_15subsets"][mk]["missing_robustness_retention"] for mk in m_keys_t3])
    mrr_elig = _bold_winner([t1["combinatorial_15subsets"][mk]["missing_robustness_retention_eligible"] for mk in m_keys_t3])
    mrr_cells = " | ".join([f"{mrr_full[i]} / {mrr_elig[i]}" for i in range(len(m_keys_t3))])
    md.append(f"| **Incomplete-Subset Retention (MRR, 14 Subsets)** | **1–3** | --- | --- | {mrr_cells} | --- |")

    # =========================================================================
    # Table 4: Track 3A Longitudinal + Track 3B 5-Seed OASIS-3 + Track 2 Cohort 2
    # =========================================================================
    n_long = t3.get("n_adni_longitudinal", 1017)
    n_60m = t3.get("n_60m_horizon", 716)
    n_oa = t3.get("n_oasis_total", 1026)
    n_c2 = t2.get("n_total", 289)
    n_c2_90d = t2.get("n_strict_90d_window", 211)
    ckd_comp_map = t2["ckd_subgroup"].get("symmetric_composite_triage", {})
    ckd_hes_map = t2["ckd_subgroup"].get("symmetric_hes_only_triage", {})

    md.append(
        f"\n## Table 4: Longitudinal Conversion (ADNI $N={n_long}$, In-Fold Tertile Log-Rank $p={t3['logrank_p']:.2e}$), 5-Seed Zero-Shot OASIS-3 ($N={n_oa}$), & Plasma pTau217 Cohort 2 ($N={n_c2}$)\n"
    )
    md.append(
        "\n> **Baseline Low-eGFR Note**: Exploratory Cohort-2 `eGFR_bl < 60` ($n=22$) uses a "
        "single baseline measurement and does not establish the $\\ge 3$-month KDIGO chronicity "
        "criterion for clinical CKD diagnosis or exclude albuminuria.\n"
    )
    md.append(
        f"| Method | ADNI 60m AUROC ($N={n_60m}$, 95% CI, BH $q$) | ADNI Survival C-Index | OASIS-3 5-Seed Dementia AUROC ($N={n_oa}$, BH $q$) | OASIS-3 Strict AD-Source AUROC | Cohort 2 pTau217 AUROC ($N={n_c2}$ / 90d $n={n_c2_90d}$) | Symmetric Low-eGFR (<60) Triage Acc (Comp $k/n$ / HesOnly) |"
    )
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    t4_models = [
        ("EarlyConcat", "EarlyConcat-MLP"),
        ("GatedAttention", "GatedAttention-Transformer"),
        ("EvidentialDL", "EvidentialDL (Dirichlet TMC)"),
        ("RCML", "RCML (adapted, AAAI'24)"),
        ("QMF", "QMF (adapted, ICML'23)"),
        ("DrFuse", "DrFuse (adapted, AAAI'24)"),
        ("DeepFuzzyTorch", "DeepFuzzyTorch (Ours)"),
    ]
    present_t4 = [(k, lbl) for k, lbl in t4_models if k in t3["adni_longitudinal_table"]]
    au60_b = _bold_winner([t3["adni_longitudinal_table"][k]["auroc_60m"] for k, _ in present_t4])
    cidx_b = _bold_winner([t3["adni_longitudinal_table"][k]["c_index"] for k, _ in present_t4])
    oa_b = _bold_winner([
        t3["oasis3_external_table"][k].get("auroc_5seed_mean", t3["oasis3_external_table"][k]["auroc"])
        for k, _ in present_t4
    ])
    oa_ad_b = _bold_winner(
        [t3.get("oasis3_strict_ad_source_table", {}).get(k, t3["oasis3_external_table"][k])["auroc"] for k, _ in present_t4]
    )
    c2_b = _bold_winner([t2["cohort2_metrics"][k]["auroc"] for k, _ in present_t4])
    c2_90_b = _bold_winner([
        t2.get("cohort2_strict_90d_metrics", {}).get(k, t2["cohort2_metrics"][k])["auroc"]
        for k, _ in present_t4
    ])
    comp_acc_b = _bold_winner([
        ckd_comp_map[k]["acc_bars"][-1] if k in ckd_comp_map else -1.0 for k, _ in present_t4
    ])
    hes_acc_b = _bold_winner([
        ckd_hes_map[k]["acc_bars"][-1] if k in ckd_hes_map else -1.0 for k, _ in present_t4
    ])

    for i, (key, label) in enumerate(present_t4):
        lg = t3["adni_longitudinal_table"][key]
        oa = t3["oasis3_external_table"][key]
        c2m = t2["cohort2_metrics"][key]
        c2_90 = t2.get("cohort2_strict_90d_metrics", {}).get(key, c2m)
        lbl_fmt = f"**{label}**" if key == "DeepFuzzyTorch" else label
        if key in ckd_comp_map and key in ckd_hes_map:
            comp_kn = ckd_comp_map[key]["ckd_post_triage_k_n"]
            hes_kn = ckd_hes_map[key]["ckd_post_triage_k_n"]
            ckd_str = f"{comp_acc_b[i]} ({comp_kn}) / {hes_acc_b[i]} ({hes_kn})"
        else:
            spec_val = t2["ckd_subgroup"].get(f"{key}_spec", [0.0, 0.0, 0.0, c2m["acc"]])[-1]
            ckd_str = f"{spec_val:.4f}"
        ci60 = lg.get("auroc_60m_ci")
        ci60_s = (
            f" ({ci60[0]:.4f}–{ci60[1]:.4f})"
            if isinstance(ci60, (list, tuple)) and len(ci60) == 2
            else ""
        )
        md.append(
            f"| {lbl_fmt} | {au60_b[i]}{ci60_s} (q={format_pval(lg.get('q_bh_60m_vs_ours'))}) | {cidx_b[i]} | "
            f"{oa_b[i]}±{oa.get('auroc_5seed_std', 0.0):.4f} (q={format_pval(oa.get('q_bh_vs_ours'))}) | {oa_ad_b[i]} | "
            f"{c2_b[i]} / {c2_90_b[i]} | {ckd_str} |"
        )

    # Full Cohort 2 comparison including newly added baselines (DDEF, FUML, XGBoost, CatBoost, TabPFN)
    c2_all_order = t4_models[:-1] + [
        ("DDEF", "DDEF (adapted, Inf. Fusion'24)"),
        ("FUML", "FUML (adapted, ICML'25)"),
        ("XGBoost", "XGBoost"),
        ("CatBoost", "CatBoost"),
        ("TabPFN", "TabPFN v2"),
        ("UnimodalPlasma_pTau217", "Unimodal Plasma pTau217/GFAP"),
        ("DeepFuzzyTorch", "DeepFuzzyTorch (Ours)"),
    ]
    present_c2 = [(k, lbl) for k, lbl in c2_all_order if k in t2.get("cohort2_metrics", {})]
    if present_c2:
        md.append(f"\n### Table 4A-C2: ADNI Cohort 2 Plasma pTau217/GFAP Complete Comparison ($N={n_c2}$; $\\le 90$-Day Blood $n={n_c2_90d}$; Strict Same-Visit+90d $n=176$)\n")
        md.append("| Method | Full Cohort 2 AUROC (95% CI) | <=90-Day Blood (n=211) | Strict Concurrent (n=176) | Raw $p$ | BH $q_{\\text{BH}}$ |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for key, label in present_c2:
            cm = t2["cohort2_metrics"][key]
            c90 = t2.get("cohort2_strict_90d_metrics", {}).get(key, {})
            c176 = t2.get("cohort2_strict_176_concurrent_metrics", {}).get(key, {})
            ci = cm["auroc_ci"]
            lbl_fmt = f"**{label}**" if key == "DeepFuzzyTorch" else label
            s90 = f"{c90['auroc']:.4f}" if isinstance(c90, dict) and "auroc" in c90 else "---"
            s176 = f"{c176['auroc']:.4f}" if isinstance(c176, dict) and "auroc" in c176 else "---"
            md.append(
                f"| {lbl_fmt} | {cm['auroc']:.4f} ({ci[0]:.4f}–{ci[1]:.4f}) | {s90} | {s176} | "
                f"{format_pval(cm.get('p_value_vs_ours'))} | {format_pval(cm.get('q_bh_vs_ours'))} |"
            )

    # =========================================================================
    # Table 4B: Multi-Cohort Real-World Biomarker Discordance Audit
    # =========================================================================
    rw = t2.get("real_world_discordance_audit", {})
    if rw:
        md.append(
            f"\n## Table 4B: Strictly Nested Outer-Fold Real-World Clinical Biomarker Discordance Audit (Deduplicated Unique ADNI $N={rw['dedup_n_unique_participants']}$: $n_{{\\text{{disc}}}}={rw['dedup_n_discordant']}$ vs. $n_{{\\text{{conc}}}}={rw['dedup_n_concordant']}$, Observed-Pair Conflict $C_{{\\mathcal{{O}}}}$ Mann–Whitney $p = {rw['dedup_mannwhitney_p']:.2e}$)\n"
        )
        md.append(
            "| Cohort / Discordance Stratum (In-Fold Medians) | Discordant $n_{\\text{disc}}$ vs. Concordant $n_{\\text{conc}}$ | Mean Conflict $C_{\\mathcal{O}}$ (Disc vs. Conc) | Conflict Ratio | Mann–Whitney Two-Sided $p$ | Combined $0.5\\pi_{\\text{fused}}+0.5 C_{\\mathcal{O}}$ $p$ |"
        )
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        md.append(
            f"| ADNI Cohort 1 Baseline Plasma pTau181 ($n={rw['cohort1_n_observed_plasma']}$) | {rw['cohort1_n_discordant']} vs. {rw['cohort1_n_concordant']} | {rw['cohort1_conflict_discordant_mean']:.4f} vs. {rw['cohort1_conflict_concordant_mean']:.4f} | **{rw['cohort1_conflict_ratio']:.2f}$\\times$** | **{rw['cohort1_mannwhitney_p']:.2e}** | {rw['cohort1_combined_mannwhitney_p']:.2e} |"
        )
        md.append(
            f"| ADNI Cohort 2 Plasma pTau217/GFAP ($N={rw['cohort2_n_total']}$) | {rw['cohort2_n_discordant']} vs. {rw['cohort2_n_concordant']} | {rw['cohort2_conflict_discordant_mean']:.4f} vs. {rw['cohort2_conflict_concordant_mean']:.4f} | **{rw['cohort2_conflict_ratio']:.2f}$\\times$** | **{rw['cohort2_mannwhitney_p']:.2e}** | {rw['cohort2_combined_mannwhitney_p']:.2e} |"
        )
        md.append(
            f"| **Deduplicated Unique Cross-Cohort ADNI ($N={rw['dedup_n_unique_participants']}$)** | **{rw['dedup_n_discordant']} vs. {rw['dedup_n_concordant']}** | **{rw['dedup_conflict_discordant_mean']:.4f} vs. {rw['dedup_conflict_concordant_mean']:.4f}** | **{rw['dedup_conflict_ratio']:.2f}$\\times$** | **{rw['dedup_mannwhitney_p']:.2e}** | **{rw['dedup_combined_mannwhitney_p']:.2e}** |\n"
        )
        md.append(
            "| Method (Strictly Nested Outer-Fold 80th-Pct Discordant Triage) | Dedup Unreferred Acc / AUROC | Dedup 20% Triage (`Composite`) Acc (Retained $k/n$, AUROC) | Dedup 20% Triage (`HesOnly`) Acc (Retained $k/n$, AUROC) | C1 20% Triage (`Composite`) Acc (Retained $k/n$, AUROC) |"
        )
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        rw_order = t4_models[:-1] + [
            ("DDEF", "DDEF (adapted, Inf. Fusion'24)"),
            ("FUML", "FUML (adapted, ICML'25)"),
            ("XGBoost", "XGBoost"),
            ("CatBoost", "CatBoost"),
            ("TabPFN", "TabPFN v2"),
            ("DeepFuzzyTorch", "DeepFuzzyTorch (Ours)"),
        ]
        rw_rows = [(mk, mlbl) for mk, mlbl in rw_order if mk in rw.get("dedup_discordant_triage", {})]
        dd_comp_acc_b = _bold_winner([rw["dedup_discordant_triage"][mk]["composite"]["acc_refer20"] for mk, _ in rw_rows])
        dd_comp_au_b = _bold_winner([rw["dedup_discordant_triage"][mk]["composite"]["auroc_refer20"] for mk, _ in rw_rows])
        dd_hes_acc_b = _bold_winner([rw["dedup_discordant_triage"][mk]["hes_only"]["acc_refer20"] for mk, _ in rw_rows])
        dd_hes_au_b = _bold_winner([rw["dedup_discordant_triage"][mk]["hes_only"]["auroc_refer20"] for mk, _ in rw_rows])
        c1_comp_acc_b = _bold_winner([rw["cohort1_discordant_triage"][mk]["composite"]["acc_refer20"] for mk, _ in rw_rows])
        c1_comp_au_b = _bold_winner([rw["cohort1_discordant_triage"][mk]["composite"]["auroc_refer20"] for mk, _ in rw_rows])
        for idx_rw, (mk, mlbl) in enumerate(rw_rows):
            dd_t = rw["dedup_discordant_triage"][mk]
            c1_t = rw["cohort1_discordant_triage"][mk]
            dd_comp = dd_t.get("composite", dd_t)
            dd_hes = dd_t.get("hes_only", dd_t)
            c1_comp = c1_t.get("composite", c1_t)
            lbl_fmt = f"**{mlbl}**" if mk == "DeepFuzzyTorch" else mlbl
            md.append(
                f"| {lbl_fmt} | {dd_t['acc_unreferred']:.4f} / {dd_t['auroc_unreferred']:.4f} | "
                f"{dd_comp_acc_b[idx_rw]} ({dd_comp['n_refer20_retained']}/{dd_t['n_unreferred']}, {dd_comp_au_b[idx_rw]}) | "
                f"{dd_hes_acc_b[idx_rw]} ({dd_hes['n_refer20_retained']}/{dd_t['n_unreferred']}, {dd_hes_au_b[idx_rw]}) | "
                f"{c1_comp_acc_b[idx_rw]} ({c1_comp['n_refer20_retained']}/{c1_t['n_unreferred']}, {c1_comp_au_b[idx_rw]}) |"
            )

        mc_dd = rw.get("matched_coverage_referral", {})
        if mc_dd and "composite_nested_ecdf" in mc_dd:
            mc_grid = mc_dd["composite_nested_ecdf"]
            mc_models = mc_grid["models"]
            md.append(
                "\n## Table 4C: Matched-Coverage Selective Referral Curve (Deduplicated Real-World Discordant ADNI, $n=198$, Nested Outer-Fold ECDF Ranks)\n"
            )
            md.append(
                "| Method | AURC $\\downarrow$ (raw $p_{\\text{AURC}}$; $q_{\\text{BH}}$ vs Ours) | 100% (198) | 95% (188) | 90% (178) | 85% (168) | 81.3% (161) | 80% (158) | 76.8% (152) | 75% (148) | 70.7% (140) | 60% (119) |"
            )
            md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            mc_keys = [
                ("EarlyConcat", "EarlyConcat-MLP"),
                ("GatedAttention", "GatedAttention-Transformer"),
                ("EvidentialDL", "EvidentialDL (Dirichlet TMC)"),
                ("RCML", "RCML (adapted)"),
                ("QMF", "QMF (adapted)"),
                ("DrFuse", "DrFuse (adapted)"),
                ("DDEF", "DDEF (adapted)"),
                ("FUML", "FUML (adapted)"),
                ("XGBoost", "XGBoost"),
                ("CatBoost", "CatBoost"),
                ("TabPFN", "TabPFN v2"),
                ("DeepFuzzyTorch", "DeepFuzzyTorch (Ours, Comp.)"),
            ]
            cols_mc = [
                "cov_100pct", "cov_95pct", "cov_90pct", "cov_85pct",
                "match_161_81p3pct", "cov_80pct", "match_152_76p8pct",
                "cov_75pct", "match_140_70p7pct", "cov_60pct",
            ]
            mc_pvals = {
                mk: p for mk, _ in mc_keys if mk != "DeepFuzzyTorch" and mk in mc_models
                if isinstance((p := mc_models[mk].get("aurc_p_vs_ours")), (int, float))
                and math.isfinite(p) and 0.0 <= p <= 1.0
            }
            mc_qvals = _benjamini_hochberg(mc_pvals)
            for mk, mlbl in mc_keys:
                if mk not in mc_models:
                    continue
                mv = mc_models[mk]
                pts = mv["points"]
                lbl_fmt = f"**{mlbl}**" if mk == "DeepFuzzyTorch" else mlbl
                p_au = format_pval(mc_pvals.get(mk), show_one=True)
                q_au = format_pval(mc_qvals.get(mk), show_one=True)
                row_vals = " | ".join(f"{pts[ck]['acc']:.4f}" for ck in cols_mc)
                md.append(f"| {lbl_fmt} | {mv['aurc']:.4f} ({p_au}; {q_au}) | {row_vals} |")
            if "conflict_only_nested_ecdf" in mc_dd and "DeepFuzzyTorch" in mc_dd["conflict_only_nested_ecdf"]["models"]:
                mv_co = mc_dd["conflict_only_nested_ecdf"]["models"]["DeepFuzzyTorch"]
                row_co = " | ".join(f"{mv_co['points'][ck]['acc']:.4f}" for ck in cols_mc)
                md.append(f"| **DeepFuzzyTorch (Pure $C_{{\\mathcal{{O}}}}$ ECDF)** | {mv_co['aurc']:.4f} (---; ---) | {row_co} |")
            md.append(
                f"\nRaw paired-bootstrap $p_{{\\text{{AURC}}}}$ values are adjusted by Benjamini--Hochberg across the {len(mc_pvals)} available comparator tests; Ours rows are excluded from the family.\n"
            )

    # =========================================================================
    # Table 5: TCGA Pan-Cancer 3-Modality Benchmark
    # =========================================================================
    n_tcga = t4.get("n_total", 2946)
    md.append(
        f"\n## Table 5: TCGA Pan-Cancer 3-Modality Multi-Omics + WSI Benchmark ($N={n_tcga}$ Unique Patients Across 7 Cohorts)\n"
    )
    md.append(
        "| Method | Pan-Cancer AUROC (95% CI) | OS C-Index | 7-Subset Macro AUROC | BRCA | COAD | KIRC | KIRP | LUAD | LUSC | UCEC | Raw $p$ | BH $q_{\\text{BH}}$ |"
    )
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    t5_models = [
        ("EarlyConcat", "EarlyConcat-MLP"),
        ("LateAvg", "LateAvg-Ensemble"),
        ("DempsterShafer", "Dempster-Shafer Fusion"),
        ("GatedAttention", "GatedAttention-Transformer"),
        ("EvidentialDL", "EvidentialDL (Dirichlet TMC)"),
        ("RCML", "RCML (adapted, AAAI'24)"),
        ("QMF", "QMF (adapted, ICML'23)"),
        ("DrFuse", "DrFuse (adapted, AAAI'24)"),
        ("DeepFuzzyTorch", "DeepFuzzyTorch (Ours)"),
    ]
    present_tcga = [(k, lbl) for k, lbl in t5_models if k in t4["tcga_main_table"]]
    tc_au_b = _bold_winner([t4["tcga_main_table"][k]["auroc"] for k, _ in present_tcga])
    tc_os_b = _bold_winner([t4["tcga_main_table"][k].get("os_c_index", 0.0) for k, _ in present_tcga])
    pc = t4["tcga_per_cohort"]
    for i, (key, label) in enumerate(present_tcga):
        tm = t4["tcga_main_table"][key]
        tc_macro = (
            f"{t4['tcga_combinatorial_7subsets'][key]['macro_avg_auroc']:.4f}"
            if key in t4.get("tcga_combinatorial_7subsets", {})
            else "---"
        )
        lbl_fmt = f"**{label}**" if key == "DeepFuzzyTorch" else label
        cohort_cells = [
            f"{pc[c][key]:.4f}" if key in pc.get(c, {}) else "---"
            for c in ["TCGA-BRCA", "TCGA-COAD", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LUAD", "TCGA-LUSC", "TCGA-UCEC"]
        ]
        md.append(
            f"| {lbl_fmt} | {tc_au_b[i]} ({tm['auroc_ci'][0]:.4f}–{tm['auroc_ci'][1]:.4f}) | {tc_os_b[i]} | {tc_macro} | "
            + " | ".join(cohort_cells)
            + f" | {format_pval(tm.get('p_value_vs_ours'))} | {format_pval(tm.get('q_bh_vs_ours'))} |"
        )

    # =========================================================================
    # Table 5B: Supervised In-Fold TCGA Reduct on Unsupervised Transductive Pool
    # =========================================================================
    hd = t4.get("highdim_1000gene_reduct", {})
    if hd:
        jacc_fr = hd.get("cross_fold_jaccard_stability", {}).get("mean", 0.0)
        md.append(
            f"\n## Table 5B: Supervised In-Fold TCGA PanCancer Atlas RNA-Seq Reduct on the Unsupervised Label-Free Transductive 1,000-MAD Gene Pool ($1,000 \\to K^*={hd['optimal_k']}$ Genes, 5-Fold Jaccard = `{jacc_fr:.4f}`, {hd['rna_branch_param_reduction_pct']:.2f}% RNA Branch Parameter Reduction)\n"
        )
        fsc = hd.get("feature_selector_comparison", {})
        if fsc:
            md.append(
                "### Table 5B.1: Supervised In-Fold Feature-Selection Baseline Comparison on the Unsupervised Label-Free Transductive 1,000-MAD Gene Pool ($K^*=20$)\n"
            )
            md.append(
                "| In-Fold Feature Selector ($1,000 \\to 20$) | 4-Modality AUROC (95% CI) | OS C-Index | Observed RNA-Only AUROC | Missing WSI & Staging (`RNA+GenSig`) AUROC | 5-Fold Pairwise Jaccard | Fit Time / Fold (s) | Raw $p$ | BH $q_{\\text{BH}}$ |"
            )
            md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            sel_order = [
                ("PCA-20", "PCA-20 (Unsupervised Orthogonal Projection)"),
                ("MI-20", "Mutual Information Top-20 (`MI-20`)"),
                ("Lasso-20", "$L_1$-Regularized Logistic Regression (`Lasso-20`)"),
                ("mRMR-20", "Min-Redundancy Max-Relevance (`mRMR-20`)"),
                ("FuzzyRough-20", "**Pointwise T-Norm Fuzzy-Rough Reduct (`FuzzyRough-20`, Ours)**"),
            ]
            s_present = [(sk, slbl) for sk, slbl in sel_order if sk in fsc]
            s_au_b = _bold_winner([fsc[sk]["auroc_4mod"] for sk, _ in s_present])
            s_os_b = _bold_winner([fsc[sk]["os_c_index_4mod"] for sk, _ in s_present])
            s_rna_b = _bold_winner([fsc[sk]["auroc_rna_only_observed"] for sk, _ in s_present])
            s_nw_b = _bold_winner([fsc[sk]["auroc_missing_wsi_and_clin"] for sk, _ in s_present])
            for idx_s, (sk, slbl) in enumerate(s_present):
                sr = fsc[sk]
                sci = sr["auroc_4mod_ci"]
                jm = sr["jaccard_5fold"]["mean"]
                j_cell = f"{jm:.4f} ({sr['jaccard_5fold']['min']:.3f}–{sr['jaccard_5fold']['max']:.3f})" if jm is not None else "N/A (Subspace)"
                md.append(
                    f"| {slbl} | {s_au_b[idx_s]} ({sci[0]:.4f}–{sci[1]:.4f}) | {s_os_b[idx_s]} | {s_rna_b[idx_s]} | "
                    f"{s_nw_b[idx_s]} | {j_cell} | {sr['fit_time_mean_sec']:.2f}±{sr['fit_time_std_sec']:.2f} | "
                    f"{format_pval(sr.get('p_value_4mod_vs_fuzzyrough'))} | {format_pval(sr.get('q_bh_4mod_vs_fuzzyrough'))} |"
                )

        md.append("\n### Table 5B.2: 4-Modality Pan-Cancer Multimodal Fusion Comparison ($N=2,946$)\n")
        md.append(
            "| 4-Modality Configuration (WSI + RNA-Seq + 10-Feature Signatures + Clinical) | Pan-Cancer AUROC (95% CI) | OS C-Index | Acc | F1 | Brier $\\downarrow$ | ECE $\\downarrow$ | Raw $p$ | BH $q_{\\text{BH}}$ |"
        )
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        m4_order = [
            ("EarlyConcat_4Mod_All1000", "EarlyConcat-MLP (Unreduced 1,000 Genes)"),
            ("EarlyConcat_4Mod_Top20", "EarlyConcat-MLP (Top-20 Gene Reduct)"),
            ("LateAvg_4Mod_Top20", "LateAvg-Ensemble (Top-20 Gene Reduct)"),
            ("DempsterShafer_4Mod_Top20", "Dempster-Shafer Fusion (Top-20 Gene Reduct)"),
            ("GatedAttention_4Mod_Top20", "GatedAttention-Transformer (Top-20 Gene Reduct)"),
            ("EvidentialDL_4Mod_Top20", "EvidentialDL (Top-20 Gene Reduct)"),
            ("RCML_4Mod_Top20", "RCML (adapted, AAAI'24, Top-20 Gene Reduct)"),
            ("QMF_4Mod_Top20", "QMF (adapted, ICML'23, Top-20 Gene Reduct)"),
            ("DrFuse_4Mod_Top20", "DrFuse (adapted, AAAI'24, Top-20 Gene Reduct)"),
            ("DeepFuzzyTorch_4Mod_All1000", "DeepFuzzyTorch (Unreduced 1,000 Genes, Fair Matched Gates/Rough)"),
            ("DeepFuzzyTorch_4Mod_Top20", "DeepFuzzyTorch (Ours, Top-20 Fuzzy-Rough Gene Reduct)"),
        ]
        m4_present = [(mk, mlbl) for mk, mlbl in m4_order if mk in hd["multimodal_4mod_table"]]
        m4_au_b = _bold_winner([hd["multimodal_4mod_table"][mk]["auroc"] for mk, _ in m4_present])
        m4_os_b = _bold_winner([hd["multimodal_4mod_table"][mk]["os_c_index"] for mk, _ in m4_present])
        for idx_m, (mk4, mlbl4) in enumerate(m4_present):
            m4 = hd["multimodal_4mod_table"][mk4]
            ci4 = m4["auroc_ci"]
            lbl_fmt = f"**{mlbl4}**" if mk4 == "DeepFuzzyTorch_4Mod_Top20" else mlbl4
            md.append(
                f"| {lbl_fmt} | {m4_au_b[idx_m]} ({ci4[0]:.4f}–{ci4[1]:.4f}) | {m4_os_b[idx_m]} | {m4['acc']:.4f} | "
                f"{m4['f1']:.4f} | {m4['brier']:.4f} | {m4['ece']:.4f} | "
                f"{format_pval(m4.get('p_value_vs_ours_top20'))} | {format_pval(m4.get('q_bh_vs_ours_top20'))} |"
            )

        top_g_str = ", ".join(
            [
                f"`{g['gene']}` ({g['fold_freq']}/5, $\\Delta\\gamma={g.get('univariate_excess_dependency', g['univariate_dependency']):.4f}$)"
                for g in hd["consensus_top20_genes"]
            ]
        )
        md.append(f"\n- **Supervised In-Fold Consensus Top-20 Fuzzy-Rough Transcriptomic Genes (from Label-Free Transductive 1,000-MAD Pool)**: {top_g_str}")
        m_bh = hd.get("reactome_bh_family_size_m", hd.get("reactome_bg_pathways_with_K_ge_2", 676))
        md.append(
            f"- **Custom-Background Exact Hypergeometric Reactome Enrichment ($N_{{\\text{{bg}}}}={hd.get('reactome_background_universe_size', 1000)}$ candidate genes from unsupervised label-free transductive MAD pre-screen; BH family $m={m_bh}$ predeclared $K_{{\\text{{bg}}}} \\ge 2$ pathways; $k_{{\\text{{genes}}}} \\ge 2$ displayed={hd['reactome_total_pathways_found']}: {hd['reactome_fdr_lt_005_count']} with predeclared BH $\\text{{FDR}}<0.05$)**:"
        )
        for p_row in hd["reactome_top_pathways"][:10]:
            fe = p_row.get("fold_enrichment", 0.0)
            md.append(
                f"  - `{p_row['stId']}` *{p_row['name']}*: $k_{{\\text{{genes}}}}={p_row['found']}/{p_row['total']}$ ({fe:.2f}$\\times$ enrichment), raw $p={p_row['p_value']:.2e}$, BH $q={p_row['fdr']:.2e}$ (Unique Genes: `{p_row['submitted_genes_found']}`)"
            )

    # =========================================================================
    # Table 6: 6-Foundation-Model Hesitant Fuzzy Consensus & Pythagorean TTA
    # =========================================================================
    md.append(
        "\n## Table 6: 6-Foundation-Model Hesitant Consensus (Block 2, Source OOF Adam-Trained) & Pythagorean TTA (Block 5) on Camelyon17 ($N=2,500$) & NCT-CRC ($N=2,349$; binary `is_tumor_or_stroma` endpoint; corrupted-FM scenario uses -2x PLIP PCA-64 corruption)\n"
    )
    md.append(
        "| Benchmark Scenario (In-Fold PCA-64) | Source / Target $N$ | Best Single FM | MeanPool 6-FMs | MedianPool 6-FMs | Logit-Tent TTA (adapted) | Hesitant Fuzzy (Block 2) | Hesitant Fuzzy + PFS-TTA (Ours) | BH $q$ vs. Mean / Tent |"
    )
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for k, v in t5.items():
        k_clean = k.replace("\n", " ")
        n_src = v.get("n_source", "---")
        n_tgt = v.get("n_target", "---")
        row_vals = [
            v["BestSingleFM"],
            v["MeanPool6FMs"],
            v.get("MedianPool6FMs", v["MeanPool6FMs"]),
            v.get("Tent_TTA", v["MeanPool6FMs"]),
            v["HesitantFuzzy"],
            v["HesitantFuzzy_PFS_TTA"],
        ]
        row_strs = _bold_winner(row_vals)
        q_map = v.get("q_bh_vs_ours", {})
        q_pair = f"{format_pval(q_map.get('MeanPool6FMs'))} / {format_pval(q_map.get('Tent_TTA'))}"
        md.append(
            f"| {k_clean} | {n_src} / {n_tgt} | {row_strs[0]} | {row_strs[1]} | {row_strs[2]} | {row_strs[3]} | {row_strs[4]} | {row_strs[5]} | {q_pair} |"
        )

    # =========================================================================
    # Table 7: Track 6 Wall-Clock Latency & Exact Peak Live Tensor Memory
    # =========================================================================
    if t6:
        hw = t6.get("hardware_environment", {})
        md.append(
            f"\n## Table 7: Computational Complexity, Wall-Clock Latency & Peak Live Tensor Memory (`{hw.get('device', 'cpu')}`, `{hw.get('processor', 'arm')}`)\n"
        )
        md.append("### Table 7A: Batch Scaling ($N \\in \\{64, 256, 1024, 4096\\}$, $D=100$, $M=4$) & Modality Scaling ($M \\in \\{2, 4, 8, 16\\}$, $N=256$, $D=100$)\n")
        md.append(
            "| Model | $N=64$ Fwd+Bwd (ms / MiB) | $N=256$ Fwd+Bwd (ms / MiB) | $N=1024$ Fwd+Bwd (ms / MiB) | $N=4096$ Fwd+Bwd (ms / MiB) | $M=4 \\to 8 \\to 16$ Latency (ms) |"
        )
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        by_n = {(r["model"], r["N"]): r for r in t6.get("sweep_N_D100_M4", [])}
        by_m = {(r["model"], r["M_modalities"]): r for r in t6.get("sweep_M_N256_D100", [])}
        for m_rt in ["EarlyConcat", "GatedAttention", "EvidentialDL", "RCML", "QMF", "DrFuse", "UnifiedDeepFuzzyModel"]:
            if (m_rt, 256) not in by_n:
                continue
            cells_n = [
                f"{by_n[(m_rt, n)]['fwd_bwd_mean_ms']:.2f} / {by_n[(m_rt, n)]['peak_memory_mib']:.1f}"
                for n in [64, 256, 1024, 4096]
            ]
            m_str = (
                f"{by_m[(m_rt, 4)]['fwd_bwd_mean_ms']:.2f} → {by_m[(m_rt, 8)]['fwd_bwd_mean_ms']:.2f} → {by_m[(m_rt, 16)]['fwd_bwd_mean_ms']:.2f}"
                if (m_rt, 16) in by_m
                else "---"
            )
            rt_map = {"RCML": "RCML (adapted)", "QMF": "QMF (adapt.)", "DrFuse": "DrFuse (adapted)"}
            lbl_rt = "**UnifiedDeepFuzzyModel (Ours)**" if m_rt == "UnifiedDeepFuzzyModel" else rt_map.get(m_rt, m_rt)
            md.append(f"| {lbl_rt} | " + " | ".join(cells_n) + f" | {m_str} |")

        r_n = t6.get("reducer_sweep_N_D100", [])
        r_tcga = t6.get("reducer_N1024_D1000", {})
        if r_n:
            rn_str = ", ".join([f"$N={r['N']}$: `{r['fit_mean_sec']:.3f}s` (`{r['peak_memory_mib']:.1f}MiB`)" for r in r_n])
            md.append(
                f"\n- **`FuzzyRoughFeatureReducer` Offline Fit Scaling ($K^*=20$, 8 Null Permutations)**: {rn_str}; TCGA-scale ($N=1,024, D=1,000$): `{r_tcga.get('fit_mean_sec', 0.0):.3f}s` (`{r_tcga.get('peak_memory_mib', 0.0):.1f}MiB`)."
            )

    md.append("\n## Exported White-Box Clinical Decision Rules (`DeepConceptTSKHead`, Precision=8)\n")
    for r_str in t1["exported_tsk_rules"]:
        md.append(f"- `{r_str}`")

    return "\n".join(md) + "\n"


if __name__ == "__main__":
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        md_text = generate_markdown_summary(data)
        with open(TABLES_MD, "w", encoding="utf-8") as f:
            f.write(md_text)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"Wrote Markdown tables to {TABLES_MD} and {REPORT_MD}")
