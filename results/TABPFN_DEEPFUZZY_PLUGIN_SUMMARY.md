# TabPFN–DeepFuzzy plug-in audit

ADNI Cohort 1: N=1137; same-visit reference n=924; complete four-modality n=854. Five archived outer folds and archived fold-specific MRI indices are used.

| Model | AUROC (95% CI) | AUPRC | Acc | F1 | Brier | ECE | AURC | Same-visit AUROC | Complete AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepFuzzyTorch | 0.8270 (0.8005–0.8486) | 0.8680 | 0.7617 | 0.7686 | 0.1700 | 0.0651 | 0.1335 | 0.8387 | 0.8371 |
| TabPFN | 0.8443 (0.8196–0.8664) | 0.8844 | 0.7748 | 0.7782 | 0.1578 | 0.0269 | 0.1101 | 0.8569 | 0.8541 |
| UnimodalTabPFN_DeepFuzzyTorch | 0.8321 (0.8068–0.8530) | 0.8718 | 0.7634 | 0.7675 | 0.1685 | 0.0731 | 0.1242 | 0.8426 | 0.8403 |
| DualStream_ModDropTabPFN_DeepFuzzyTorch | 0.8349 (0.8107–0.8563) | 0.8744 | 0.7661 | 0.7730 | 0.1648 | 0.0364 | 0.1246 | 0.8445 | 0.8408 |

| Plug-in | p vs DeepFuzzyTorch | p vs TabPFN v2 | 15-subset macro AUROC | 14-subset MRR | MaxPaddingDiff | Conflict-only detection AUROC γ=0.8 | 20% composite referral AUROC γ=0.8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UnimodalTabPFN_DeepFuzzyTorch | 0.0100 | 0.0220 | 0.7803 | 0.9332 | 0.0 | 0.6946 | 0.8394 |
| DualStream_ModDropTabPFN_DeepFuzzyTorch | 0.0240 | 0.0370 | 0.7828 | 0.9332 | 0.0 | 0.6946 | 0.8408 |

| Modality subset | UnimodalTabPFN_DeepFuzzyTorch full / eligible AUROC | DualStream_ModDropTabPFN_DeepFuzzyTorch full / eligible AUROC |
|---|---:|---:|
| MRI | 0.7077 / 0.7093 | 0.7087 / 0.7103 |
| Plasma | 0.7173 / 0.7472 | 0.7198 / 0.7526 |
| FDG | 0.6755 / 0.6864 | 0.6743 / 0.6842 |
| Clin | 0.8163 / 0.8163 | 0.8179 / 0.8179 |
| MRI+Plasma | 0.7640 / 0.7640 | 0.7638 / 0.7638 |
| MRI+FDG | 0.7299 / 0.7300 | 0.7318 / 0.7319 |
| MRI+Clin | 0.8208 / 0.8208 | 0.8221 / 0.8221 |
| Plasma+FDG | 0.7446 / 0.7591 | 0.7488 / 0.7639 |
| Plasma+Clin | 0.8274 / 0.8274 | 0.8338 / 0.8338 |
| FDG+Clin | 0.8148 / 0.8148 | 0.8204 / 0.8204 |
| MRI+Plasma+FDG | 0.7725 / 0.7725 | 0.7719 / 0.7719 |
| MRI+Plasma+Clin | 0.8328 / 0.8328 | 0.8358 / 0.8358 |
| MRI+FDG+Clin | 0.8207 / 0.8207 | 0.8222 / 0.8222 |
| Plasma+FDG+Clin | 0.8277 / 0.8277 | 0.8363 / 0.8363 |
| MRI+Plasma+FDG+Clin | 0.8321 / 0.8321 | 0.8349 / 0.8349 |

Fold-0 TSK text reconstruction maximum absolute error: 2.38e-07.

The blender uses patient-disjoint inner predictions for both streams. Inner preprocessing and MRI selection are refitted on inner training patients. The dual stream retains the first variant's hesitancy and conflict; its final probability is the logistic blend.

The conflict experiment swaps observed plasma within outer validation folds; the referral score is 0.5 × (hesitancy + conflict). The JSON report includes all subset results, sweep records, fold audits, and six 8-decimal rules. Participant-level out-of-fold outputs are stored separately in DUA-restricted `results/tabpfn_plugin_oof_arrays.npz`.
