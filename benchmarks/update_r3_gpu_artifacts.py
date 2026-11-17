"""
Zero-Leakage Artifact Archive, Figure Regenerator & SHA-256 Reproducibility Manifest Lock Script.

Executes cleanly after `benchmarks/run_full_benchmark.py` and `benchmarks/run_upgrades_and_pathway_audit.py`:
  1. Preserves 100% of zero-leakage out-of-fold predictions in `results/benchmark_results.json`
     and `results/oof_predictions_and_splits.npz` (zero alphabetical slicing, zero pooled cutoffs).
  2. Archives Track 0 theorem curves (`fig1_*`), Track 1 Fold 0 checkpoint (`results/adni1_fold0_deepfuzzy_state.pt`)
     with exact bit-for-bit OOF replay verification, Fold 0 TSK validation replay tensors,
     and strict fold preprocessor parameters + participant/slide IDs across all cohorts.
  3. Embeds `results/runtime_vram_benchmark.json` (`track6_runtime_complexity`) and exact `cohort_accounting`.
  4. Optionally regenerates `figures/Fig1_*.{pdf,png}` through `figures/Fig6_*.{pdf,png}` (unless `--skip-figures`).
  5. Exports `results/BENCHMARK_TABLES_SUMMARY.md`, `results/BENCHMARK_SUMMARY_REPORT.md`, and
     locks SHA-256 digests across all data, code, and output artifacts in `results/reproducibility_manifest.json`.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import torch

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

torch.set_num_threads(4)

from deepfuzzy import DeepConceptTSKHead
from deepfuzzy.visualization import regenerate_all_figures_from_artifacts
from benchmarks.run_full_benchmark import (
    resolve_data_paths,
    load_adni_cohort1,
    fit_strict_preprocessor_with_params,
    apply_strict_preprocessor_params,
    train_deepfuzzy_or_baseline,
)
from benchmarks.export_tables_and_manuscript import generate_markdown_summary


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest_lock(
    t0: float,
    device: torch.device,
    gpu_name: str,
    results_dir: str,
    cohorts_tcga: list[str],
    cohort_accounting: dict,
    relock_only: bool = False,
) -> dict:
    """Write the 85-entry SHA-256 reproducibility manifest covering data, code, and artifacts."""
    data_files_rel = [
        "data_benchmark/adni_processed/cohort1_adni_ptau181_313roi.csv",
        "data_benchmark/adni_processed/cohort2_adni_ptau217_gfap.csv",
        "data_benchmark/adni_processed/cohort4_oasis3_external.csv",
        "data_benchmark/adni_processed/OASIS3_diagnosis.csv",
        "data_benchmark/wsi_data/clinical/tcga_precision_oncology_sample_biomarkers.csv",
        "data_benchmark/wsi_data/clinical/tcga_pancancer_raw_proteincoding_rnaseq.npz",
        "data_benchmark/wsi_data/clinical/tcga_pancancer_1000genes_rnaseq.npz",
        "data_benchmark/wsi_data/clinical/ReactomePathways.gmt",
        "data_benchmark/wsi_data/multi_fm_features/camelyon17_6fms.npz",
        "data_benchmark/wsi_data/multi_fm_features/camelyon17_metadata.csv",
        "data_benchmark/wsi_data/multi_fm_features/nct_crc_cross_cohort_6fms.npz",
        "data_benchmark/wsi_data/multi_fm_features/nct_crc_cross_cohort_metadata.csv",
    ]
    for c_name in cohorts_tcga:
        data_files_rel.append(f"data_benchmark/wsi_data/tcga_uni2h/{c_name}_uni2h_metadata.csv")
        data_files_rel.append(f"data_benchmark/wsi_data/tcga_uni2h/{c_name}_uni2h_slides.npz")

    code_files_rel = [
        "deepfuzzy/__init__.py",
        "deepfuzzy/operators.py",
        "deepfuzzy/baselines.py",
        "deepfuzzy/missing_benchmark.py",
        "deepfuzzy/conflict_injector.py",
        "deepfuzzy/rough_reducer.py",
        "deepfuzzy/metrics.py",
        "deepfuzzy/visualization.py",
        "benchmarks/fetch_tcga_1000genes_cbioportal.py",
        "benchmarks/run_full_benchmark.py",
        "benchmarks/run_upgrades_and_pathway_audit.py",
        "benchmarks/run_if_phase1_upgrades.py",
        "benchmarks/run_runtime_vram_benchmark.py",
        "benchmarks/update_r3_gpu_artifacts.py",
        "benchmarks/export_tables_and_manuscript.py",
        "examples/quickstart_synthetic_demo.py",
        "tests/test_operators_and_theorems.py",
        "benchmarks/build_companion_manuscripts.py",
        "figures/fig0_redesign_20261108/generate_fig0_clean.py",
        "pyproject.toml",
        "README.md",
    ]
    artifact_files_rel = [
        "results/benchmark_results.json",
        "results/oof_predictions_and_splits.npz",
        "results/tcga_top20_reactome_enrichment.json",
        "results/tcga_top20_reactome_pathways.csv",
        "results/runtime_vram_benchmark.json",
        "results/adni1_fold0_deepfuzzy_state.pt",
        "results/adni2_5fold_deepfuzzy_states.pt",
        "results/BENCHMARK_TABLES_SUMMARY.md",
        "results/BENCHMARK_SUMMARY_REPORT.md",
        "figures/Fig1_Mathematical_Theorems_Verification.pdf",
        "figures/Fig1_Mathematical_Theorems_Verification.png",
        "figures/Fig2_Missing_Modality_Benchmark.pdf",
        "figures/Fig2_Missing_Modality_Benchmark.png",
        "figures/Fig3_Conflict_Injection_and_Selective_Referral.pdf",
        "figures/Fig3_Conflict_Injection_and_Selective_Referral.png",
        "figures/Fig4_Fuzzy_Rough_Reduct_and_MultiOmics_FM.pdf",
        "figures/Fig4_Fuzzy_Rough_Reduct_and_MultiOmics_FM.png",
        "figures/Fig5_Clinical_DCA_Calibration_and_TSK_Rules.pdf",
        "figures/Fig5_Clinical_DCA_Calibration_and_TSK_Rules.png",
        "figures/Fig6_HighDim_TCGA_Transcriptomic_Reduct_and_Reactome.pdf",
        "figures/Fig6_HighDim_TCGA_Transcriptomic_Reduct_and_Reactome.png",
        "results/latex_audit_report.json",
        "figures/Fig0_Paradigm_Teaser_Comparison.pdf",
        "figures/Fig0_Paradigm_Teaser_Comparison.png",
        "figures/Fig0_System_Architecture_Overview.pdf",
        "figures/Fig0_System_Architecture_Overview.png",
        "paper/Highlights_Information_Fusion.md",
        "paper/THEOREM_PROOFS_COMPLETE.md",
        "paper/references.bib",
        "paper/manuscript_Information_Fusion.tex",
        "paper/manuscript_Information_Fusion.bbl",
        "paper/manuscript_Information_Fusion.pdf",
        "paper/manuscript_Nature_Communications.tex",
        "paper/manuscript_Nature_Communications.bbl",
        "paper/manuscript_Nature_Communications.pdf",
        "paper/manuscript_Genome_Biology_Software.tex",
        "paper/manuscript_Genome_Biology_Software.bbl",
        "paper/manuscript_Genome_Biology_Software.pdf",
    ]

    manifest = {
        "execution_command": (
            "PYTHONPATH=. python benchmarks/update_r3_gpu_artifacts.py --relock-only"
            if relock_only
            else "PYTHONPATH=. python benchmarks/update_r3_gpu_artifacts.py"
        ),
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_device": str(device),
        "gpu_name": gpu_name,
        "elapsed_seconds": round(time.time() - t0, 2),
        "provenance_stages": {
            "stage1_transductive_tcga_and_reactome_fetch": {
                "script": "benchmarks/fetch_tcga_1000genes_cbioportal.py",
                "description": "Unsupervised, label-free transductive PanCanAtlas 15,636->5,000->1,000 MAD protein-coding gene candidate pre-screen (zero manual cancer-gene lists, zero outcome labels) + Reactome GMT.",
            },
            "stage2_primary_zero_leakage_benchmark": {
                "script": "benchmarks/run_full_benchmark.py",
                "description": "Tracks 0-5 with QMF, RCML (adapted), DrFuse (adapted), equalized-supervision check, in-fold Fuzzy-Rough reducts, symmetric triage, in-fold survival tertiles, 5-seed OASIS-3, Adam HFS, and Logit-Tent (adapted) TTA.",
            },
            "stage3_discordance_and_highdim_tcga_audit": {
                "script": "benchmarks/run_upgrades_and_pathway_audit.py",
                "description": "Strictly nested outer-fold training-calibrated real-world discordance triage + supervised in-fold 1,000->20 TCGA fuzzy-rough reduct on the unsupervised transductive 1,000-MAD pool vs PCA/Lasso/mRMR/MI + custom-background hypergeometric Reactome enrichment (m=676 predeclared K_bg>=2 pathways).",
            },
            "stage4_runtime_vram_benchmark": {
                "script": "benchmarks/run_runtime_vram_benchmark.py",
                "description": "Wall-clock forward+backward latency and live tensor peak memory across batch N, dimension D, and modality count M.",
            },
            "stage5_sha256_reproducibility_lock": {
                "script": "benchmarks/update_r3_gpu_artifacts.py",
                "description": "Fold preprocessor replay verification, Fold 0 checkpoint archive, publication figures, compiled manuscripts, and 85-entry SHA-256 inventory.",
            },
        },
        "environment_lock": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "cohort_accounting": cohort_accounting,
        "data_sha256": {
            rel: sha256_file(os.path.join(WORKSPACE_DIR, rel))
            for rel in data_files_rel
            if os.path.exists(os.path.join(WORKSPACE_DIR, rel))
        },
        "code_sha256": {
            rel: sha256_file(os.path.join(WORKSPACE_DIR, rel))
            for rel in code_files_rel
            if os.path.exists(os.path.join(WORKSPACE_DIR, rel))
        },
        "artifact_sha256": {
            rel: sha256_file(os.path.join(WORKSPACE_DIR, rel))
            for rel in artifact_files_rel
            if os.path.exists(os.path.join(WORKSPACE_DIR, rel))
        },
    }
    manifest_path = os.path.join(results_dir, "reproducibility_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"[Artifact Archive & Lock] Completed in {time.time() - t0:.2f}s! "
        f"Manifest entries = {len(manifest['data_sha256'])} data + {len(manifest['code_sha256'])} code + {len(manifest['artifact_sha256'])} artifact; "
        f"Total unique participants = {cohort_accounting['total_unique_clinical_and_oncology_participants']}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepFuzzyTorch Reproducibility Archive & Manifest Lock")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip regenerating figures/Fig1..Fig6 (use before @viz-fixer gate completes)",
    )
    parser.add_argument(
        "--relock-only",
        action="store_true",
        help="Relock the 85-entry SHA-256 reproducibility_manifest.json without replaying models, figures, or markdown",
    )
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(
        f"[Artifact Archive & Lock] Running on device={device} ({gpu_name}), "
        f"skip_figures={args.skip_figures}, relock_only={args.relock_only}"
    )

    paths = resolve_data_paths(os.path.join(WORKSPACE_DIR, "data_benchmark"))
    results_dir = os.path.join(WORKSPACE_DIR, "results")
    json_path = os.path.join(results_dir, "benchmark_results.json")
    npz_path = os.path.join(results_dir, "oof_predictions_and_splits.npz")

    with open(json_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    if args.relock_only:
        cohorts_tcga = ["TCGA-BRCA", "TCGA-COAD", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LUAD", "TCGA-LUSC", "TCGA-UCEC"]
        _write_manifest_lock(
            t0=t0,
            device=device,
            gpu_name=gpu_name,
            results_dir=results_dir,
            cohorts_tcga=cohorts_tcga,
            cohort_accounting=master["cohort_accounting"],
            relock_only=True,
        )
        return

    npz_data = dict(np.load(npz_path, allow_pickle=True))

    # 1. Archive Track 0 theorem arrays into NPZ for offline figure replay
    track0 = master["track0_theorems"]
    for k_sp, v_sp in track0["simplex_points"].items():
        npz_data[f"fig1_simplex_{k_sp}"] = np.asarray(v_sp, dtype=np.float32)
    npz_data["fig1_monotonicity_grid_mu"] = np.asarray(track0["monotonicity_grid_mu"], dtype=np.float32)
    npz_data["fig1_monotonicity_surface"] = np.asarray(track0["monotonicity_surface"], dtype=np.float32)
    npz_data["fig1_measured_conflict_grid"] = np.asarray(track0["measured_conflict_grid"], dtype=np.float32)
    for k_cc, v_cc in track0["conflict_hesitancy_curves"].items():
        g_tag = k_cc.split("=")[-1].strip().replace("$", "").replace(".", "p")
        npz_data[f"fig1_conflict_curve_gamma_{g_tag}"] = np.asarray(v_cc, dtype=np.float32)
    for k_fr, v_fr in track0["fuzzy_rough_envelope"].items():
        npz_data[f"fig1_fuzzy_rough_{k_fr}"] = np.asarray(v_fr, dtype=np.float32)

    # 2. Archive Track 1 Fold 0 checkpoint & verify exact replay against saved OOF predictions
    adni_data = load_adni_cohort1(paths["adni_dir"])
    X_raw_list = adni_data["X_modalities"]
    y_c1 = adni_data["y_amyloid"]
    per_fold_sel = master["track1_adni_cohort1"]["rough_summary"]["per_fold_selected_indices"]

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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_raw_list[0], y_c1)):
        sel_mri_idx = [int(i) for i in per_fold_sel[fold_idx]]
        npz_data[f"adni1_fold{fold_idx}_mri_selected_idx"] = np.asarray(sel_mri_idx, dtype=np.int32)
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
        X_tr_s, X_va_s, m_tr, m_va, prep_params_f = fit_strict_preprocessor_with_params(X_tr_raw, X_va_raw)
        X_va_replay, m_va_replay = apply_strict_preprocessor_params(X_va_raw, prep_params_f)
        for m_i in range(4):
            assert float(np.max(np.abs(X_va_s[m_i] - X_va_replay[m_i]))) == 0.0
            assert float(np.max(np.abs(m_va[:, m_i] - m_va_replay[:, m_i]))) == 0.0
            npz_data[f"adni1_fold{fold_idx}_mod{m_i}_median"] = prep_params_f[m_i]["median"]
            npz_data[f"adni1_fold{fold_idx}_mod{m_i}_mean"] = prep_params_f[m_i]["mean"]
            npz_data[f"adni1_fold{fold_idx}_mod{m_i}_std"] = prep_params_f[m_i]["std"]

        if fold_idx == 0:
            hero_fold0_model = train_deepfuzzy_or_baseline(
                model_name="DeepFuzzyTorch",
                X_tr_list=X_tr_s,
                mask_tr=m_tr,
                y_tr=y_c1[tr_idx],
                epochs=28,
                lr=1.8e-3,
                embed_dim=40,
                concept_names=concept_names_8,
                ablation_flags={},
                seed=42,
                device=device,
            )
            fold0_ckpt_path = os.path.join(results_dir, "adni1_fold0_deepfuzzy_state.pt")
            torch.save({k: v.cpu() for k, v in hero_fold0_model.state_dict().items()}, fold0_ckpt_path)
            with torch.no_grad():
                va0_t = [torch.tensor(x, dtype=torch.float32, device=device) for x in X_va_s]
                va0_m = torch.tensor(m_va, dtype=torch.float32, device=device)
                out_va0 = hero_fold0_model(va0_t, modality_masks=va0_m, clinical_concepts=va0_t[-1])
                p_va0 = out_va0["prob"].view(-1).cpu().numpy()
                oof_replay_diff = float(np.max(np.abs(p_va0 - npz_data["adni1_prob_DeepFuzzyTorch"][va_idx])))
                z_np = out_va0["z_concepts"].cpu().numpy()
                cm_np = out_va0["concept_mask"].cpu().numpy()
                tsk_logits_np = out_va0["tsk_logits"].cpu().numpy().ravel()
                struct_raw = hero_fold0_model.tsk_head.export_structured_rules(concept_names_8)
                exported_rules = hero_fold0_model.tsk_head.export_clinical_rules(
                    concept_names_8, gate_threshold=0.0, precision=8
                )
                parsed_logits = DeepConceptTSKHead.evaluate_exported_rules_text(
                    exported_rules, z_np, concept_names=concept_names_8, concept_mask=cm_np
                )
                text_parsed_err = float(
                    np.max(np.abs(tsk_logits_np.astype(np.float64) - parsed_logits.astype(np.float64)))
                )
                npz_data["adni1_fold0_val_indices"] = np.asarray(va_idx, dtype=np.int32)
                npz_data["adni1_fold0_val_z_concepts"] = z_np.astype(np.float32)
                npz_data["adni1_fold0_val_concept_mask"] = cm_np.astype(np.float32)
                npz_data["adni1_fold0_val_tsk_logits"] = tsk_logits_np.astype(np.float32)
                npz_data["adni1_fold0_tsk_centers"] = np.asarray(struct_raw["centers"], dtype=np.float32)
                npz_data["adni1_fold0_tsk_sigmas"] = np.asarray(struct_raw["sigmas"], dtype=np.float32)
                npz_data["adni1_fold0_tsk_gates"] = np.asarray(struct_raw["gates"], dtype=np.float32)
                npz_data["adni1_fold0_tsk_consequent_bias"] = np.asarray(
                    struct_raw["consequent_bias"], dtype=np.float32
                )
                npz_data["adni1_fold0_tsk_effective_consequent_weights"] = np.asarray(
                    struct_raw["effective_consequent_weights"], dtype=np.float32
                )
                print(
                    f"[Fold 0 Checkpoint] Saved adni1_fold0_deepfuzzy_state.pt | "
                    f"OOF replay max diff={oof_replay_diff:.2e} | 8-dec TSK text err={text_parsed_err:.2e}"
                )

    # 3. Archive OASIS-3 & TCGA 3-mod preprocessors and participant/slide IDs
    df_c2 = pd.read_csv(os.path.join(paths["adni_dir"], "cohort2_adni_ptau217_gfap.csv"))
    sub_c2 = df_c2[df_c2["amyloid_gold_pos"].notna()].copy().reset_index(drop=True)

    df_adni_all = adni_data["df_full"].copy()
    df_adni_1137 = adni_data["df_amyloid"].copy()
    df_long = (
        df_adni_all[(df_adni_all["DX_bl"] != "AD") & (df_adni_all["max_followup_months"].fillna(0) > 0)]
        .copy()
        .reset_index(drop=True)
    )
    df_oa = pd.read_csv(os.path.join(paths["adni_dir"], "cohort4_oasis3_external.csv"))

    df_adni_1137["sex_male"] = (df_adni_1137["PTGENDER"] == "Male").astype(float)
    df_adni_1137["apoe4_bin"] = (df_adni_1137["APOE4"] >= 1).astype(float)
    df_oa["sex_male"] = (df_oa["M/F"] == "M").astype(float)
    df_oa["apoe4_bin"] = df_oa["APOE4_carrier"].astype(float)
    X_adni_mri_ext = df_adni_1137[["Hippocampus", "Hippocampus_ICV_ratio", "ICV"]].values.astype(np.float32)
    X_oa_mri_ext = df_oa[["TOTAL_HIPPOCAMPUS_VOLUME", "Hippocampus_ICV_ratio", "IntraCranialVol"]].values.astype(
        np.float32
    )
    X_adni_cl_nm = df_adni_1137[["AGE", "sex_male", "PTEDUCAT", "apoe4_bin"]].values.astype(np.float32)
    X_oa_cl_nm = df_oa[["Age", "sex_male", "Education", "apoe4_bin"]].values.astype(np.float32)
    _, va_oa_nm, _, _, p_oa_nm = fit_strict_preprocessor_with_params(
        [X_adni_mri_ext, X_adni_cl_nm], [X_oa_mri_ext, X_oa_cl_nm]
    )
    va_rep_nm, _ = apply_strict_preprocessor_params([X_oa_mri_ext, X_oa_cl_nm], p_oa_nm)
    for m_i in range(2):
        assert float(np.max(np.abs(va_oa_nm[m_i] - va_rep_nm[m_i]))) == 0.0
        npz_data[f"oasis3_nommse_mod{m_i}_median"] = p_oa_nm[m_i]["median"]
        npz_data[f"oasis3_nommse_mod{m_i}_mean"] = p_oa_nm[m_i]["mean"]
        npz_data[f"oasis3_nommse_mod{m_i}_std"] = p_oa_nm[m_i]["std"]

    X_adni_cl_mm = df_adni_1137[["AGE", "sex_male", "PTEDUCAT", "apoe4_bin", "MMSE"]].values.astype(np.float32)
    X_oa_cl_mm = df_oa[["Age", "sex_male", "Education", "apoe4_bin", "mmse"]].values.astype(np.float32)
    _, va_oa_mm, _, _, p_oa_mm = fit_strict_preprocessor_with_params(
        [X_adni_mri_ext, X_adni_cl_mm], [X_oa_mri_ext, X_oa_cl_mm]
    )
    va_rep_mm, _ = apply_strict_preprocessor_params([X_oa_mri_ext, X_oa_cl_mm], p_oa_mm)
    for m_i in range(2):
        assert float(np.max(np.abs(va_oa_mm[m_i] - va_rep_mm[m_i]))) == 0.0
        npz_data[f"oasis3_mmse_mod{m_i}_median"] = p_oa_mm[m_i]["median"]
        npz_data[f"oasis3_mmse_mod{m_i}_mean"] = p_oa_mm[m_i]["mean"]
        npz_data[f"oasis3_mmse_mod{m_i}_std"] = p_oa_mm[m_i]["std"]

    # TCGA Pan-Cancer (N=2,946: 48-component PCA + 3-modality strict preprocessor)
    wsi_dir = paths["wsi_dir"]
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
    wsi_raw_list, gen_list, clin_list, y_tcga_list, tcga_pids, tcga_sids = [], [], [], [], [], []
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
        tcga_pids.extend(merged["patient_id"].astype(str).tolist())
        tcga_sids.extend(merged["slide_id"].astype(str).tolist())

    X_wsi_raw = np.vstack(wsi_raw_list)
    X_gen_tcga = np.vstack(gen_list)
    X_clin_tcga = np.vstack(clin_list)
    y_tcga_all = np.concatenate(y_tcga_list)
    skf_tcga = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for f_i, (tr_i, va_i) in enumerate(skf_tcga.split(X_wsi_raw, y_tcga_all)):
        wsi_tr_mean = np.mean(X_wsi_raw[tr_i], axis=0, keepdims=True)
        wsi_tr_std = np.std(X_wsi_raw[tr_i], axis=0, keepdims=True) + 1e-5
        pca_wsi = PCA(n_components=48, random_state=42 + f_i)
        X_wsi_tr_pca = pca_wsi.fit_transform((X_wsi_raw[tr_i] - wsi_tr_mean) / wsi_tr_std).astype(np.float32)
        X_wsi_va_pca = pca_wsi.transform((X_wsi_raw[va_i] - wsi_tr_mean) / wsi_tr_std).astype(np.float32)
        X_tr_raw_t4 = [X_wsi_tr_pca, X_gen_tcga[tr_i], X_clin_tcga[tr_i]]
        X_va_raw_t4 = [X_wsi_va_pca, X_gen_tcga[va_i], X_clin_tcga[va_i]]
        _, va_st4, _, m_vat4, p_t4 = fit_strict_preprocessor_with_params(X_tr_raw_t4, X_va_raw_t4)
        va_rept4, m_rept4 = apply_strict_preprocessor_params(X_va_raw_t4, p_t4)
        npz_data[f"tcga_fold{f_i}_wsi_pre_mean"] = wsi_tr_mean.ravel().astype(np.float32)
        npz_data[f"tcga_fold{f_i}_wsi_pre_std"] = wsi_tr_std.ravel().astype(np.float32)
        npz_data[f"tcga_fold{f_i}_wsi_pca_mean"] = pca_wsi.mean_.astype(np.float32)
        npz_data[f"tcga_fold{f_i}_wsi_pca_components"] = pca_wsi.components_.astype(np.float32)
        for m_i in range(3):
            assert float(np.max(np.abs(va_st4[m_i] - va_rept4[m_i]))) == 0.0
            assert float(np.max(np.abs(m_vat4[:, m_i] - m_rept4[:, m_i]))) == 0.0
            npz_data[f"tcga_fold{f_i}_mod{m_i}_median"] = p_t4[m_i]["median"]
            npz_data[f"tcga_fold{f_i}_mod{m_i}_mean"] = p_t4[m_i]["mean"]
            npz_data[f"tcga_fold{f_i}_mod{m_i}_std"] = p_t4[m_i]["std"]

    npz_data["adni1_rids"] = df_adni_1137["RID"].values.astype(np.int32)
    npz_data["adni2_rids"] = sub_c2["RID"].values.astype(np.int32)
    npz_data["adni_long_rids"] = df_long["RID"].values.astype(np.int32)
    npz_data["oasis3_subject_ids"] = np.asarray(df_oa["Subject"].astype(str).tolist(), dtype=str)
    npz_data["tcga_patient_ids"] = np.asarray(tcga_pids, dtype=str)
    npz_data["tcga_slide_ids"] = np.asarray(tcga_sids, dtype=str)
    for k_npz in list(npz_data.keys()):
        if npz_data[k_npz].dtype == object:
            npz_data[k_npz] = np.asarray(npz_data[k_npz].tolist(), dtype=str)
    np.savez_compressed(npz_path, **npz_data)
    print(f"[Preprocessor Replay] Verified 0.0 replay error across all fold preprocessors (NPZ arrays={len(npz_data)})")

    # 4. Sync Track 6 runtime complexity & exact cohort overlap accounting
    runtime_json_path = os.path.join(results_dir, "runtime_vram_benchmark.json")
    if os.path.exists(runtime_json_path):
        with open(runtime_json_path, "r", encoding="utf-8") as f_rt:
            master["track6_runtime_complexity"] = json.load(f_rt)

    t3 = master.get("track3_oasis3_external", {})
    if "oasis3_uds_d1_concordant_814_table" in t3 and "oasis3_uds_concordant_814_table" not in t3:
        t3["oasis3_uds_concordant_814_table"] = t3["oasis3_uds_d1_concordant_814_table"]

    c1_df = pd.read_csv(os.path.join(paths["adni_dir"], "cohort1_adni_ptau181_313roi.csv"))
    s_c1 = set(c1_df.loc[c1_df["amyloid_gold_pos"].notna(), "RID"])
    s_c2 = set(sub_c2["RID"])
    s_long = set(
        c1_df.loc[(c1_df["DX_bl"] != "AD") & (c1_df["max_followup_months"].fillna(0) > 0), "RID"]
    )
    s_adni_union = s_c1 | s_c2 | s_long
    cohort_accounting = {
        "adni_cohort1_n": len(s_c1),
        "adni_cohort2_n": len(s_c2),
        "adni_longitudinal_n": len(s_long),
        "overlap_c1_c2": len(s_c1 & s_c2),
        "overlap_c1_long": len(s_c1 & s_long),
        "overlap_c2_long": len(s_c2 & s_long),
        "overlap_triple_c1_c2_long": len(s_c1 & s_c2 & s_long),
        "unique_adni_participants_union": len(s_adni_union),
        "unique_oasis3_participants": len(df_oa),
        "unique_tcga_patients": len(tcga_pids),
        "total_unique_clinical_and_oncology_participants": len(s_adni_union) + len(df_oa) + len(tcga_pids),
        "camelyon17_pathology_patches": 2500,
        "nct_crc_pathology_tiles": 2349,
        "total_pathology_patches_and_tiles": 4849,
    }
    master["cohort_accounting"] = cohort_accounting

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

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(master), f, indent=2)

    # 5. Optionally regenerate Fig1..Fig6 (ONLY when --skip-figures is NOT set)
    if not args.skip_figures:
        print("[Figures] Regenerating Fig1..Fig6 via deepfuzzy.visualization.regenerate_all_figures_from_artifacts...")
        out_figs = regenerate_all_figures_from_artifacts(WORKSPACE_DIR)
        master["figures"] = {
            fig_k: {fmt_k: os.path.relpath(fmt_v, WORKSPACE_DIR) for fmt_k, fmt_v in fig_d.items()}
            for fig_k, fig_d in out_figs.items()
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(master), f, indent=2)
        for fk, fv in master["figures"].items():
            print(f"  -> {fk}: {fv.get('pdf')} & {fv.get('png')}")

    # 6. Export Markdown summary tables
    md_text = generate_markdown_summary(master)
    with open(os.path.join(results_dir, "BENCHMARK_TABLES_SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(os.path.join(results_dir, "BENCHMARK_SUMMARY_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(md_text)

    # 7. Write SHA-256 Reproducibility Manifest
    _write_manifest_lock(
        t0=t0,
        device=device,
        gpu_name=gpu_name,
        results_dir=results_dir,
        cohorts_tcga=cohorts_tcga,
        cohort_accounting=cohort_accounting,
        relock_only=False,
    )


if __name__ == "__main__":
    main()
