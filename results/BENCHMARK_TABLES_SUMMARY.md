# `DeepFuzzyTorch` Multi-Dataset Benchmark & Mathematical Verification Report

- **Execution Device**: `cpu` (`CPU`)
- **Total Primary Suite Execution Time**: `418.35s`
- **Theorem 1 Simplex Closure Max Residual ($N=250$)**: `5.96e-08` ($\mu + \nu + \pi = 1$)
- **Theorem 2 Missing-Modality Degeneration Max Error ($\gamma_c=0.85$)**: `0.00e+00` (Missing Choquet Weight = `0.00e+00`)
- **Theorem 3 Fixed-Conflict Monotonicity Min Partial Derivative**: `0.2072` (Analytic vs. FD Max Error = `5.20e-08`)
- **Theorem 5 Fuzzy-Rough Inclusion Max Violation**: `0.00e+00`
- **White-Box TSK Exact Rule Reconstruction Max Error (Precision=8, With Concept Mask)**: `2.384e-07`
- **AURC Tie-Handling Convention**: `deepfuzzy.metrics.compute_aurc` averages 0-1 errors across equal-uncertainty ties for permutation invariance, whereas frozen `results/benchmark_results.json` stores initial index-ordered execution values (`31/39` Track 1–4 models including all 4 `DeepFuzzyTorch` tracks match to `0.00e+00`; `8/39` baseline/ablation values differ by $\le 1.92 \times 10^{-5}$; `39/39` are identical at 4 decimal places).


## Table 1: ADNI Cohort 1 ($N=1137$ Full Cohort With Natural Missingness; Same-Visit Baseline Reference $n=924$; Complete 4-Modality Subset $N_{\text{comp}}=854$) 5-Fold Cross-Validation

> **Population & Imputation Accounting**: ADNI Cohort 1 ($N=1137$) includes $1.3\%$ missing MRI ($1,122/1137$), $23.3\%$ missing Plasma pTau181 ($872/1137$), $10.5\%$ missing FDG-PET ($1,018/1137$), and $0\%$ missingness for Clin as a modality ($6/1137$ [$0.53\%$] missing baseline `ADAS13` entries are imputed strictly in-fold using observed training-fold medians). Exactly $N_{\text{comp}}=854$ participants ($75.1\%$) have all four baseline modalities observed simultaneously.

| Method | Full AUROC ($N=1137$, 95% CI) | Same-Visit AUROC ($n=924$) | Complete 4-Mod AUROC ($N_{\text{comp}}=854$) | AUPRC | Acc | F1 | Brier $\downarrow$ | ECE $\downarrow$ | AURC $\downarrow$ | 15-Subset Macro (Full / Elig) | MRR (14 Inc.) | Raw $p$ | BH $q_{\text{BH}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EarlyConcat-MLP | 0.8212 (0.7961–0.8434) | 0.8304 | 0.8260 | 0.8693 | 0.7414 | 0.7443 | 0.1704 | 0.0431 | 0.1268 | 0.7690 / 0.7723 | 0.9318 | 0.350 | 0.769 |
| LateAvg-Ensemble | 0.8262 (0.8010–0.8481) | 0.8382 | 0.8346 | 0.8657 | 0.7502 | 0.7625 | 0.1724 | 0.0655 | 0.1543 | 0.7773 / 0.7811 | 0.9365 | 0.913 | 0.932 |
| DeepTSK-RuleNet | 0.8086 (0.7804–0.8334) | 0.8242 | 0.8181 | 0.8413 | 0.7573 | 0.7587 | 0.1781 | 0.0711 | 0.1462 | 0.7554 / 0.7591 | 0.9295 | 0.010 | 0.100 |
| Dempster-Shafer Fusion | 0.8115 (0.7855–0.8357) | 0.8242 | 0.8220 | 0.8523 | 0.7291 | 0.7403 | 0.1761 | 0.0371 | 0.1643 | 0.7668 / 0.7716 | **0.9409** | 0.032 | 0.213 |
| GatedAttention-Transformer | 0.8298 (0.8040–0.8527) | 0.8437 | 0.8403 | 0.8728 | 0.7546 | 0.7593 | 0.1676 | 0.0570 | 0.1214 | 0.7680 / 0.7717 | 0.9202 | 0.652 | 0.869 |
| EvidentialDL (Dirichlet TMC) | 0.8289 (0.8036–0.8507) | 0.8413 | 0.8378 | 0.8685 | 0.7537 | 0.7631 | 0.1676 | 0.0416 | 0.1456 | 0.7761 / 0.7800 | 0.9319 | 0.696 | 0.870 |
| RCML (adapted, Xu et al., AAAI'24) | 0.8229 (0.7968–0.8456) | 0.8366 | 0.8340 | 0.8632 | 0.7493 | 0.7587 | 0.1752 | 0.0760 | 0.1333 | 0.7742 / 0.7781 | 0.9366 | 0.458 | 0.769 |
| QMF (Zhang et al., ICML'23) | 0.8313 (0.8071–0.8540) | 0.8449 | 0.8394 | 0.8749 | 0.7608 | 0.7643 | 0.1639 | 0.0483 | 0.1244 | 0.7759 / 0.7795 | 0.9286 | 0.472 | 0.769 |
| DrFuse (adapted, Yao et al., AAAI'24) | 0.8316 (0.8062–0.8528) | 0.8411 | 0.8389 | 0.8757 | 0.7493 | 0.7587 | 0.1660 | 0.0481 | 0.1265 | 0.7690 / 0.7730 | 0.9193 | 0.463 | 0.769 |
| DDEF (adapted, Shao et al., Inf. Fusion'24) | 0.8323 (0.8069–0.8537) | 0.8435 | 0.8414 | 0.8748 | 0.7520 | 0.7594 | 0.1674 | 0.0613 | 0.1319 | 0.7740 / 0.7776 | 0.9250 | 0.286 | 0.769 |
| FUML (adapted, Duan et al., ICML'25) | 0.8363 (0.8121–0.8584) | 0.8477 | 0.8453 | 0.8759 | 0.7617 | 0.7690 | 0.1628 | 0.0279 | 0.1331 | **0.7790** / **0.7830** | 0.9266 | 0.052 | 0.260 |
| XGBoost (Native NaN + Indicators) | 0.8263 (0.8009–0.8487) | 0.8351 | 0.8333 | 0.8697 | 0.7493 | 0.7595 | 0.1733 | 0.0739 | 0.1254 | 0.7056 / 0.7125 | 0.8435 | 0.932 | 0.932 |
| CatBoost (NaN='Min' + Indicators) | 0.8262 (0.8009–0.8493) | 0.8375 | 0.8370 | 0.8693 | 0.7546 | 0.7617 | 0.1705 | 0.0621 | 0.1239 | 0.7405 / 0.7438 | 0.8889 | 0.924 | 0.932 |
| TabPFN v2 (Nature'25 + Indicators) | **0.8443** (0.8196–0.8664) | **0.8569** | **0.8541** | **0.8844** | **0.7748** | **0.7782** | **0.1578** | **0.0269** | **0.1101** | 0.7635 / 0.7668 | 0.8974 | 0.002 | 0.040 |
| **DeepFuzzyTorch (Ours)** | 0.8270 (0.8005–0.8486) | 0.8387 | 0.8371 | 0.8680 | 0.7617 | 0.7686 | 0.1700 | 0.0651 | 0.1335 | 0.7764 / 0.7800 | 0.9344 | --- | --- |

### Table 1C: Complete 4-Modality Subset Audit ($N_{\text{comp}}=854$ Observed Across MRI, Plasma, FDG, and Clin)

| Method | Complete-4-Mod AUROC (95% CI) | AUPRC | Acc | F1 | Brier $\downarrow$ | ECE $\downarrow$ | AURC $\downarrow$ | Raw $p$ | BH $q_{\text{BH}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EarlyConcat-MLP | 0.8260 (0.7983–0.8509) | 0.8733 | 0.7436 | 0.7456 | 0.1681 | 0.0579 | 0.1227 | 0.092 | 0.368 |
| LateAvg-Ensemble | 0.8346 (0.8078–0.8590) | 0.8748 | 0.7541 | 0.7646 | 0.1695 | 0.0772 | 0.1471 | 0.722 | 0.950 |
| DeepTSK-RuleNet | 0.8181 (0.7866–0.8450) | 0.8449 | 0.7670 | 0.7678 | 0.1735 | 0.0850 | 0.1391 | 0.018 | 0.180 |
| Dempster-Shafer Fusion | 0.8220 (0.7931–0.8470) | 0.8623 | 0.7412 | 0.7497 | 0.1710 | 0.0433 | 0.1412 | 0.032 | 0.213 |
| GatedAttention-Transformer | 0.8403 (0.8133–0.8658) | 0.8801 | 0.7646 | 0.7698 | 0.1612 | 0.0478 | 0.1149 | 0.668 | 0.950 |
| EvidentialDL (Dirichlet TMC) | 0.8378 (0.8108–0.8613) | 0.8770 | 0.7646 | 0.7719 | 0.1627 | 0.0502 | 0.1329 | 0.920 | 0.950 |
| RCML (adapted, Xu et al., AAAI'24) | 0.8340 (0.8062–0.8586) | 0.8720 | 0.7588 | 0.7659 | 0.1715 | 0.0853 | 0.1261 | 0.607 | 0.950 |
| QMF (Zhang et al., ICML'23) | 0.8394 (0.8099–0.8648) | 0.8809 | 0.7763 | 0.7771 | 0.1586 | 0.0540 | 0.1169 | 0.737 | 0.950 |
| DrFuse (adapted, Yao et al., AAAI'24) | 0.8389 (0.8111–0.8638) | 0.8821 | 0.7658 | 0.7743 | 0.1604 | 0.0350 | 0.1199 | 0.809 | 0.950 |
| DDEF (adapted, Shao et al., Inf. Fusion'24) | 0.8414 (0.8155–0.8653) | 0.8833 | 0.7623 | 0.7685 | 0.1620 | 0.0640 | 0.1233 | 0.358 | 0.895 |
| FUML (adapted, Duan et al., ICML'25) | 0.8453 (0.8170–0.8695) | 0.8835 | 0.7681 | 0.7745 | 0.1573 | 0.0368 | 0.1278 | 0.085 | 0.368 |
| XGBoost (Native NaN + Indicators) | 0.8333 (0.8059–0.8571) | 0.8752 | 0.7611 | 0.7708 | 0.1690 | 0.0760 | 0.1205 | 0.599 | 0.950 |
| CatBoost (NaN='Min' + Indicators) | 0.8370 (0.8090–0.8614) | 0.8773 | 0.7646 | 0.7749 | 0.1651 | 0.0673 | 0.1158 | 0.950 | 0.950 |
| TabPFN v2 (Nature'25 + Indicators) | **0.8541** (0.8276–0.8776) | **0.8928** | **0.7845** | **0.7875** | **0.1519** | **0.0251** | **0.1024** | 0.002 | 0.040 |
| **DeepFuzzyTorch (Ours)** | 0.8371 (0.8091–0.8612) | 0.8755 | 0.7635 | 0.7720 | 0.1664 | 0.0692 | 0.1284 | --- | --- |

### Table 1D: Separated Cross-Modal Conflict Anomaly Detection & 20%-Referral Under opposite_swap ($\gamma=0.8$, ADNI Cohort 1)

| Method | Detect AUROC (Composite $0.5 C_{\mathcal{O}}+0.5\pi_f$) | Detect AUROC (Pure Conflict $C_{\mathcal{O}}$) | Detect AUROC (Pure Hesitancy $\pi_f$) | 20% Refer AUROC (Composite) | 20% Refer AUROC (Pure $C_{\mathcal{O}}$) | 20% Refer AUROC (Pure $\pi_f$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| EarlyConcat-MLP | 0.5495 | 0.5495 | 0.5495 | 0.8176 | 0.8176 | 0.8176 |
| LateAvg-Ensemble | 0.6890 | 0.6865 | 0.6650 | 0.8076 | 0.8076 | 0.8112 |
| DeepTSK-RuleNet | 0.5560 | 0.4895 | 0.5827 | 0.7831 | 0.7518 | 0.7853 |
| Dempster-Shafer Fusion | 0.6287 | 0.7034 | 0.4544 | 0.7868 | 0.7896 | 0.7763 |
| GatedAttention-Transformer | 0.5803 | 0.5803 | 0.5803 | 0.8252 | 0.8252 | 0.8252 |
| EvidentialDL (Dirichlet TMC) | 0.5892 | 0.6965 | 0.4896 | 0.8178 | 0.8050 | 0.8149 |
| RCML (adapted, Xu et al., AAAI'24) | 0.6455 | 0.6857 | 0.5165 | 0.8127 | 0.8012 | 0.8175 |
| QMF (Zhang et al., ICML'23) | 0.6667 | 0.6812 | 0.6336 | 0.8212 | 0.8170 | 0.8236 |
| DrFuse (adapted, Yao et al., AAAI'24) | 0.6602 | 0.6777 | 0.6220 | 0.8211 | 0.7849 | 0.8279 |
| DDEF (adapted, Shao et al., Inf. Fusion'24) | 0.5888 | 0.6892 | 0.5622 | 0.8308 | 0.8185 | 0.8312 |
| FUML (adapted, Duan et al., ICML'25) | 0.5266 | 0.6644 | 0.4897 | 0.8307 | 0.8183 | 0.8265 |
| XGBoost (Native NaN + Indicators) | 0.5915 | N/A (Single-Table) | 0.5915 | 0.8256 | N/A | 0.8256 |
| CatBoost (NaN='Min' + Indicators) | 0.5557 | N/A (Single-Table) | 0.5557 | 0.8379 | N/A | 0.8379 |
| TabPFN v2 (Nature'25 + Indicators) | 0.5893 | N/A (Single-Table) | 0.5893 | 0.8325 | N/A | 0.8325 |
| **DeepFuzzyTorch (Ours)** | 0.6542 | 0.7000 | 0.5337 | 0.8302 | 0.8136 | 0.8307 |

### Table 1B: Equalized-Supervision & Cold-Start Fairness Audit on ADNI Cohort 1 ($N=1,137$)

| Supervision Regime / Variant | Full AUROC (95% CI) | AUPRC | Acc | F1 | Brier $\downarrow$ | ECE $\downarrow$ | AURC $\downarrow$ | 15-Subset Macro AUROC | Incomplete MRR | Raw $p$ vs. Ours |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EarlyConcat + Brier & Shared-Fused AuxWeight* | 0.8213 (0.7957–0.8435) | 0.8694 | 0.7441 | 0.7476 | 0.1705 | 0.0383 | 0.1265 | 0.7686 | 0.9312 | 0.364 |
| Dempster-Shafer + Matched Brier & Aux Supervision | 0.8101 (0.7836–0.8341) | 0.8502 | 0.7326 | 0.7437 | 0.1769 | **0.0316** | 0.1639 | 0.7674 | **0.9435** | 0.024 |
| GatedAttention + Matched Brier & Aux Supervision | **0.8313** (0.8055–0.8542) | **0.8735** | **0.7661** | **0.7695** | **0.1663** | 0.0569 | **0.1212** | 0.7714 | 0.9228 | 0.464 |
| DeepFuzzyTorch (Joint Cold-Start + No Brier/Aux) | 0.8300 (0.8046–0.8519) | 0.8700 | 0.7493 | 0.7639 | 0.1694 | 0.0639 | 0.1468 | 0.7740 | 0.9278 | 0.384 |
| **DeepFuzzyTorch (Standard Full Objective)** | 0.8270 (0.8005–0.8486) | 0.8680 | 0.7617 | 0.7686 | 0.1700 | 0.0651 | 0.1335 | **0.7764** | 0.9344 | --- |

*\*Methodological disclosure:* `EarlyConcatMLP` lacks separate unimodal heads (`modality_probs[m] == prob`), so its `_EqLoss` control optimizes $(1 + 0.75|\mathcal{O}_i|/M)\mathrm{BCE}(p_{\mathrm{fused}}, y) + 0.35\,\mathrm{Brier}$ ($1.75\,\mathrm{BCE} + 0.35\,\mathrm{Brier}$ when all $4$ modalities are observed), whereas `DempsterShafer_EqLoss` and `GatedAttention_EqLoss` supervise distinct unimodal predictions; `DeepFuzzyTorch_ColdNoAux` jointly removes correlation warm-starting, Brier, and unimodal auxiliary BCE rather than isolating a single factor.

## Table 2: Component Ablation Study of `DeepFuzzyTorch` on ADNI Cohort 1 ($N=1,137$, 5-Fold MRI Reduct Jaccard = `0.4330`; $\Delta$ AUROC computed from unrounded OOF AUROCs)

| Ablation Variant | Full-Set AUROC (95% CI) | ECE $\downarrow$ | AURC $\downarrow$ | 15-Subset Macro AUROC | Incomplete MRR | Conflict Ref-20% AUROC ($\gamma=0.8$) | $\Delta$ AUROC | Raw $p$ | BH $q_{\text{BH}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepFuzzyTorch (Full Unified Framework, $K^*=24$)** | 0.8270 (0.8005–0.8486) | 0.0651 | 0.1335 | 0.7764 | 0.9344 | 0.8302 | --- | --- | --- |
| w/o Exact Vacuity Masking ($\langle 0,0,1\rangle \to$ Zero Imputation) | 0.8286 (0.8030–0.8500) | 0.0546 | 0.1344 | 0.7690 | 0.9230 | 0.8269 | +0.0016 | 0.318 | 0.769 |
| w/o Non-Additive Choquet Capacity (Uniform $1/\lvert\mathcal{O}\rvert$) | 0.8263 (0.8005–0.8489) | 0.0953 | 0.1383 | 0.7750 | 0.9334 | 0.8218 | -0.0007 | 0.899 | 0.932 |
| w/o Conflict Hesitancy Discount ($\gamma_c = 0$) | 0.8266 (0.8007–0.8481) | 0.0629 | 0.1446 | 0.7759 | 0.9342 | 0.8355 | -0.0003 | 0.643 | 0.869 |
| w/o Fuzzy-Rough Gate (Matched $K^*=24$ MRI ROIs) | 0.8268 (0.8003–0.8483) | 0.0664 | 0.1341 | 0.7762 | 0.9344 | 0.8307 | -0.0002 | 0.500 | 0.769 |
| w/o Fuzzy-Rough Reduct, FR Gate & Input Gates (Unreduced 320 MRI ROIs) | 0.8221 (0.7967–0.8448) | 0.0447 | 0.1349 | 0.7677 | 0.9291 | 0.8293 | -0.0048 | 0.468 | 0.769 |
| w/o White-Box Concept TSK Rule Head | 0.8280 (0.8016–0.8494) | 0.0461 | 0.1322 | 0.7769 | 0.9338 | 0.8313 | +0.0011 | 0.190 | 0.760 |

## Table 3: Exhaustive $2^4-1 = 15$ Missing-Modality Combinatorial Benchmark (ADNI $N=1,137$: Full-Population / Eligible-Subset AUROC)

| Modality Subset | $\lvert\mathcal{O}\rvert$ | Eligible $N$ | Complete $N$ | EarlyConcat (Full / Elig) | GatedAttention (Full / Elig) | EvidentialDL (Full / Elig) | RCML (adapt.) (Full / Elig) | QMF (Full / Elig) | DrFuse (adapt.) (Full / Elig) | DDEF (adapt.) (Full / Elig) | FUML (adapt.) (Full / Elig) | XGBoost (Full / Elig) | CatBoost (Full / Elig) | TabPFN v2 (Full / Elig) | DeepFuzzyTorch (Full / Elig) | Ours $\pi_{\text{fused}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MRI` | 1 | 1122 | 1122 | 0.6801 / 0.6816 | 0.6682 / 0.6691 | 0.6895 / 0.6906 | 0.7030 / 0.7044 | 0.6842 / 0.6851 | 0.6670 / 0.6681 | 0.6820 / 0.6829 | 0.6903 / 0.6914 | 0.5515 / 0.5546 | 0.6584 / 0.6593 | 0.6631 / 0.6644 | **0.7054** / **0.7071** | 0.352 |
| `Plasma` | 1 | 872 | 872 | 0.7153 / 0.7400 | 0.7144 / 0.7452 | 0.7179 / 0.7490 | 0.7182 / 0.7508 | **0.7218** / 0.7515 | 0.7200 / 0.7514 | 0.7165 / 0.7438 | 0.7208 / **0.7542** | 0.6179 / 0.6999 | 0.6924 / 0.7229 | 0.7090 / 0.7401 | 0.7183 / 0.7475 | 0.471 |
| `FDG` | 1 | 1018 | 1018 | 0.6661 / 0.6755 | 0.6571 / 0.6649 | 0.6772 / 0.6895 | **0.6798** / 0.6905 | 0.6751 / 0.6841 | 0.6577 / 0.6700 | 0.6690 / 0.6781 | 0.6754 / 0.6862 | 0.4777 / 0.4681 | 0.5129 / 0.5202 | 0.5510 / 0.5552 | 0.6797 / **0.6907** | 0.661 |
| `Clin` | 1 | 1137 | 1137 | 0.8064 / 0.8064 | 0.8024 / 0.8024 | 0.8090 / 0.8090 | 0.8063 / 0.8063 | **0.8141** / **0.8141** | 0.7949 / 0.7949 | 0.8021 / 0.8021 | 0.8136 / 0.8136 | 0.7785 / 0.7785 | 0.7847 / 0.7847 | 0.8071 / 0.8071 | 0.8083 / 0.8083 | 0.227 |
| `MRI+Plasma` | 2 | 1136 | 858 | 0.7389 / 0.7390 | 0.7422 / 0.7423 | **0.7618** / **0.7619** | 0.7612 / 0.7613 | 0.7508 / 0.7509 | 0.7479 / 0.7480 | 0.7550 / 0.7551 | 0.7600 / 0.7601 | 0.6519 / 0.6513 | 0.7266 / 0.7268 | 0.7457 / 0.7458 | 0.7593 / 0.7593 | 0.368 |
| `MRI+FDG` | 2 | 1136 | 1004 | 0.7025 / 0.7026 | 0.6921 / 0.6922 | 0.7154 / 0.7155 | 0.7226 / 0.7227 | 0.7132 / 0.7133 | 0.7059 / 0.7060 | 0.7153 / 0.7154 | 0.7151 / 0.7151 | 0.5498 / 0.5489 | 0.6678 / 0.6686 | 0.6823 / 0.6824 | **0.7288** / **0.7288** | 0.439 |
| `MRI+Clin` | 2 | 1137 | 1122 | 0.8060 / 0.8060 | 0.8082 / 0.8082 | 0.8082 / 0.8082 | 0.8039 / 0.8039 | 0.8101 / 0.8101 | 0.8076 / 0.8076 | 0.8115 / 0.8115 | 0.8205 / 0.8205 | 0.8052 / 0.8052 | 0.8027 / 0.8027 | **0.8273** / **0.8273** | 0.8159 / 0.8159 | 0.300 |
| `Plasma+FDG` | 2 | 1022 | 868 | **0.7548** / **0.7682** | 0.7492 / 0.7638 | 0.7501 / 0.7639 | 0.7505 / 0.7642 | 0.7514 / 0.7646 | 0.7472 / 0.7628 | 0.7480 / 0.7638 | 0.7463 / 0.7613 | 0.6449 / 0.6757 | 0.6738 / 0.6820 | 0.7212 / 0.7347 | 0.7411 / 0.7543 | 0.495 |
| `Plasma+Clin` | 2 | 1137 | 872 | 0.8269 / 0.8269 | 0.8277 / 0.8277 | 0.8316 / 0.8316 | 0.8212 / 0.8212 | 0.8333 / 0.8333 | 0.8243 / 0.8243 | 0.8272 / 0.8272 | **0.8337** / **0.8337** | 0.8031 / 0.8031 | 0.8073 / 0.8073 | **0.8337** / **0.8337** | 0.8213 / 0.8213 | 0.285 |
| `FDG+Clin` | 2 | 1137 | 1018 | 0.8095 / 0.8095 | 0.8060 / 0.8060 | 0.8119 / 0.8119 | 0.8044 / 0.8044 | **0.8143** / **0.8143** | 0.8009 / 0.8009 | 0.8078 / 0.8078 | 0.8140 / 0.8140 | 0.7788 / 0.7788 | 0.7844 / 0.7844 | 0.8076 / 0.8076 | 0.8083 / 0.8083 | 0.283 |
| `MRI+Plasma+FDG` | 3 | 1136 | 854 | 0.7505 / 0.7507 | 0.7545 / 0.7547 | **0.7701** / **0.7701** | 0.7700 / 0.7700 | 0.7634 / 0.7635 | 0.7619 / 0.7620 | 0.7685 / 0.7685 | 0.7676 / 0.7677 | 0.6620 / 0.6614 | 0.7349 / 0.7353 | 0.7531 / 0.7533 | 0.7669 / 0.7669 | 0.399 |
| `MRI+Plasma+Clin` | 3 | 1137 | 858 | 0.8205 / 0.8205 | 0.8270 / 0.8270 | 0.8298 / 0.8298 | 0.8235 / 0.8235 | 0.8306 / 0.8306 | 0.8299 / 0.8299 | 0.8322 / 0.8322 | 0.8361 / 0.8361 | 0.8236 / 0.8236 | 0.8242 / 0.8242 | **0.8446** / **0.8446** | 0.8276 / 0.8276 | 0.308 |
| `MRI+FDG+Clin` | 3 | 1137 | 1004 | 0.8061 / 0.8061 | 0.8090 / 0.8090 | 0.8078 / 0.8078 | 0.8032 / 0.8032 | 0.8108 / 0.8108 | 0.8088 / 0.8088 | 0.8128 / 0.8128 | 0.8213 / 0.8213 | 0.8046 / 0.8046 | 0.8035 / 0.8035 | **0.8273** / **0.8273** | 0.8158 / 0.8158 | 0.303 |
| `Plasma+FDG+Clin` | 3 | 1137 | 868 | 0.8300 / 0.8300 | 0.8330 / 0.8330 | 0.8331 / 0.8331 | 0.8225 / 0.8225 | **0.8350** / **0.8350** | 0.8290 / 0.8290 | 0.8305 / 0.8305 | 0.8340 / 0.8340 | 0.8080 / 0.8080 | 0.8083 / 0.8083 | 0.8347 / 0.8347 | 0.8218 / 0.8218 | 0.294 |
| `MRI+Plasma+FDG+Clin` | 4 | 1137 | 854 | 0.8212 / 0.8212 | 0.8298 / 0.8298 | 0.8289 / 0.8289 | 0.8229 / 0.8229 | 0.8313 / 0.8313 | 0.8316 / 0.8316 | 0.8323 / 0.8323 | 0.8363 / 0.8363 | 0.8263 / 0.8263 | 0.8262 / 0.8262 | **0.8443** / **0.8443** | 0.8270 / 0.8270 | 0.310 |
| **Macro-Average Across All 15 Subsets** | **1–4** | --- | --- | 0.7690 / 0.7723 | 0.7680 / 0.7717 | 0.7761 / 0.7800 | 0.7742 / 0.7781 | 0.7759 / 0.7795 | 0.7690 / 0.7730 | 0.7740 / 0.7776 | **0.7790** / **0.7830** | 0.7056 / 0.7125 | 0.7405 / 0.7438 | 0.7635 / 0.7668 | 0.7764 / 0.7800 | --- |
| **Incomplete-Subset Retention (MRR, 14 Subsets)** | **1–3** | --- | --- | 0.9318 / 0.9361 | 0.9202 / 0.9249 | 0.9319 / 0.9369 | **0.9366** / **0.9417** | 0.9286 / 0.9332 | 0.9193 / 0.9245 | 0.9250 / 0.9296 | 0.9266 / 0.9318 | 0.8435 / 0.8525 | 0.8889 / 0.8931 | 0.8974 / 0.9017 | 0.9344 / 0.9392 | --- |

## Table 4: Longitudinal Conversion (ADNI $N=1017$, In-Fold Tertile Log-Rank $p=3.34e-72$), 5-Seed Zero-Shot OASIS-3 ($N=1026$), & Plasma pTau217 Cohort 2 ($N=289$)

| Method | ADNI 60m AUROC ($N=716$, 95% CI, BH $q$) | ADNI Survival C-Index | OASIS-3 5-Seed Dementia AUROC ($N=1026$, BH $q$) | OASIS-3 Strict AD-Source AUROC | Cohort 2 pTau217 AUROC ($N=289$ / 90d $n=211$) | Symmetric CKD Triage Acc (Comp $k/n$ / HesOnly) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| EarlyConcat-MLP | 0.9373 (0.9155–0.9546) (q=0.569) | 0.8732 | 0.8433±0.0023 (q=0.028) | 0.8481 | 0.8695 / 0.8783 | 0.8500 (17/20) / 0.8500 (17/20) |
| GatedAttention-Transformer | 0.9304 (0.9067–0.9497) (q=0.140) | 0.8700 | 0.8366±0.0079 (q=0.028) | 0.8547 | 0.8497 / 0.8554 | 0.8333 (15/18) / 0.8333 (15/18) |
| EvidentialDL (Dirichlet TMC) | 0.9338 (0.9128–0.9517) (q=0.140) | 0.8723 | 0.8572±0.0037 (q=0.443) | 0.8543 | 0.8762 / 0.8857 | 0.8421 (16/19) / 0.8182 (18/22) |
| RCML (adapted, AAAI'24) | 0.9391 (0.9197–0.9552) (q=0.569) | 0.8728 | 0.8585±0.0029 (q=0.479) | 0.8520 | **0.8782** / **0.8868** | 0.8333 (15/18) / 0.8333 (15/18) |
| QMF (ICML'23) | 0.9396 (0.9184–0.9566) (q=0.750) | 0.8747 | 0.8547±0.0038 (q=0.443) | 0.8441 | 0.8414 / 0.8557 | 0.8750 (14/16) / **0.8889** (16/18) |
| DrFuse (adapted, AAAI'24) | 0.9322 (0.9087–0.9513) (q=0.140) | 0.8696 | 0.8350±0.0096 (q=0.024) | 0.8553 | 0.8598 / 0.8628 | **0.9333** (14/15) / 0.8333 (15/18) |
| **DeepFuzzyTorch (Ours)** | **0.9419** (0.9200–0.9589) (q=---) | **0.8804** | **0.8639**±0.0010 (q=---) | **0.8603** | 0.8740 / 0.8844 | 0.8421 (16/19) / 0.8000 (16/20) |

### Table 4A-C2: ADNI Cohort 2 Plasma pTau217/GFAP Complete Comparison ($N=289$; $\le 90$-Day Blood $n=211$; Strict Same-Visit+90d $n=176$)

| Method | Full Cohort 2 AUROC (95% CI) | <=90-Day Blood (n=211) | Strict Concurrent (n=176) | Raw $p$ | BH $q_{\text{BH}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| EarlyConcat-MLP | 0.8695 (0.8279–0.9068) | 0.8783 | 0.8960 | 0.728 | 0.971 |
| GatedAttention-Transformer | 0.8497 (0.8069–0.8882) | 0.8554 | 0.8662 | 0.102 | 0.602 |
| EvidentialDL (Dirichlet TMC) | 0.8762 (0.8361–0.9119) | 0.8857 | 0.9045 | 0.711 | 0.971 |
| RCML (adapted, AAAI'24) | 0.8782 (0.8384–0.9147) | 0.8868 | 0.9010 | 0.430 | 0.737 |
| QMF (ICML'23) | 0.8414 (0.7906–0.8861) | 0.8557 | 0.8691 | 0.060 | 0.602 |
| DrFuse (adapted, AAAI'24) | 0.8598 (0.8164–0.8999) | 0.8628 | 0.8768 | 0.251 | 0.602 |
| DDEF (adapted, Inf. Fusion'24) | 0.8595 (0.8176–0.8989) | 0.8640 | 0.8958 | 0.204 | 0.602 |
| FUML (adapted, ICML'25) | 0.8608 (0.8183–0.8992) | 0.8697 | 0.8887 | 0.170 | 0.602 |
| XGBoost | 0.8749 (0.8346–0.9116) | 0.8829 | 0.8983 | 0.900 | 0.992 |
| CatBoost | 0.8741 (0.8335–0.9110) | 0.8830 | 0.8917 | 0.958 | 0.992 |
| TabPFN v2 | 0.8830 (0.8431–0.9182) | 0.8859 | 0.9037 | 0.335 | 0.670 |
| Unimodal Plasma pTau217/GFAP | 0.8739 (0.8322–0.9095) | --- | --- | 0.992 | 0.992 |
| **DeepFuzzyTorch (Ours)** | 0.8740 (0.8331–0.9106) | 0.8844 | 0.8969 | --- | --- |

## Table 4B: Strictly Nested Outer-Fold Real-World Clinical Biomarker Discordance Audit (Deduplicated Unique ADNI $N=1112$: $n_{\text{disc}}=198$ vs. $n_{\text{conc}}=914$, Observed-Pair Conflict $C_{\mathcal{O}}$ Mann–Whitney $p = 1.75e-17$)

| Cohort / Discordance Stratum (In-Fold Medians) | Discordant $n_{\text{disc}}$ vs. Concordant $n_{\text{conc}}$ | Mean Conflict $C_{\mathcal{O}}$ (Disc vs. Conc) | Conflict Ratio | Mann–Whitney Two-Sided $p$ | Combined $0.5\pi_{\text{fused}}+0.5 C_{\mathcal{O}}$ $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ADNI Cohort 1 Baseline Plasma pTau181 ($n=872$) | 159 vs. 713 | 0.1781 vs. 0.1198 | **1.49$\times$** | **4.94e-18** | 5.90e-14 |
| ADNI Cohort 2 Plasma pTau217/GFAP ($N=289$) | 47 vs. 242 | 0.2167 vs. 0.1372 | **1.58$\times$** | **1.39e-03** | 5.90e-04 |
| **Deduplicated Unique Cross-Cohort ADNI ($N=1112$)** | **198 vs. 914** | **0.1871 vs. 0.1242** | **1.51$\times$** | **1.75e-17** | **1.58e-15** |

| Method (Strictly Nested Outer-Fold 80th-Pct Discordant Triage) | Dedup Unreferred Acc / AUROC | Dedup 20% Triage (`Composite`) Acc (Retained $k/n$, AUROC) | Dedup 20% Triage (`HesOnly`) Acc (Retained $k/n$, AUROC) | C1 20% Triage (`Composite`) Acc (Retained $k/n$, AUROC) |
| :--- | :---: | :---: | :---: | :---: |
| EarlyConcat-MLP | 0.6869 / 0.7766 | 0.7655 (145/198, 0.8174) | 0.7655 (145/198, 0.8174) | 0.7583 (120/159, 0.8420) |
| GatedAttention-Transformer | 0.7475 / 0.7885 | 0.8000 (140/198, 0.8236) | 0.8000 (140/198, 0.8236) | 0.8051 (118/159, 0.8486) |
| EvidentialDL (Dirichlet TMC) | 0.7374 / 0.7821 | 0.7643 (157/198, 0.8131) | 0.7516 (157/198, 0.8049) | 0.7661 (124/159, 0.8324) |
| RCML (adapted, AAAI'24) | 0.7222 / 0.7718 | 0.7500 (152/198, 0.7985) | 0.7500 (156/198, 0.7976) | 0.7520 (125/159, 0.8083) |
| QMF (ICML'23) | 0.7475 / 0.7849 | 0.7945 (146/198, 0.8266) | 0.7902 (143/198, 0.8214) | 0.8033 (122/159, 0.8396) |
| DrFuse (adapted, AAAI'24) | 0.7374 / 0.7869 | 0.7872 (141/198, 0.8424) | 0.7972 (143/198, 0.8303) | 0.7797 (118/159, 0.8505) |
| DDEF (adapted, Inf. Fusion'24) | 0.7323 / 0.7847 | 0.7682 (151/198, 0.8113) | 0.7582 (153/198, 0.8058) | 0.7724 (123/159, 0.8318) |
| FUML (adapted, ICML'25) | 0.7121 / 0.7803 | 0.7342 (158/198, 0.8119) | 0.7405 (158/198, 0.8179) | 0.7422 (128/159, 0.8321) |
| XGBoost | 0.7121 / 0.7828 | 0.8384 (99/198, 0.8667) | 0.8384 (99/198, 0.8667) | 0.8630 (73/159, 0.9015) |
| CatBoost | 0.7222 / 0.7849 | **0.9062** (64/198, **0.9170**) | **0.9062** (64/198, **0.9170**) | **0.9216** (51/159, **0.9389**) |
| TabPFN v2 | 0.7525 / 0.8214 | 0.8264 (144/198, 0.8556) | 0.8264 (144/198, 0.8556) | 0.8276 (116/159, 0.8606) |
| **DeepFuzzyTorch (Ours)** | 0.7475 / 0.7892 | 0.7764 (161/198, 0.8126) | 0.7610 (159/198, 0.8066) | 0.7710 (131/159, 0.8115) |

## Table 4C: Matched-Coverage Selective Referral Curve (Deduplicated Real-World Discordant ADNI, $n=198$, Nested Outer-Fold ECDF Ranks)

| Method | AURC $\downarrow$ ($p_{\text{AURC}}$ vs Ours) | 100% (198) | 95% (188) | 90% (178) | 85% (168) | 81.3% (161) | 80% (158) | 76.8% (152) | 75% (148) | 70.7% (140) | 60% (119) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EarlyConcat-MLP | 0.1753 (0.722) | 0.6869 | 0.7074 | 0.7303 | 0.7500 | 0.7578 | 0.7658 | 0.7632 | 0.7703 | 0.7786 | 0.8151 |
| GatedAttention-Transformer | 0.1496 (0.434) | 0.7475 | 0.7660 | 0.7809 | 0.7917 | 0.7888 | 0.7975 | 0.7961 | 0.7905 | 0.8000 | 0.8319 |
| EvidentialDL (Dirichlet TMC) | 0.1716 (0.802) | 0.7374 | 0.7394 | 0.7472 | 0.7619 | 0.7640 | 0.7595 | 0.7632 | 0.7568 | 0.7643 | 0.7647 |
| RCML (adapted) | 0.1942 (0.150) | 0.7222 | 0.7234 | 0.7303 | 0.7381 | 0.7453 | 0.7468 | 0.7500 | 0.7500 | 0.7500 | 0.7731 |
| QMF (ICML'23) | 0.1722 (0.792) | 0.7475 | 0.7394 | 0.7528 | 0.7679 | 0.7826 | 0.7848 | 0.7895 | 0.7905 | 0.7929 | 0.8067 |
| DrFuse (adapted) | 0.1753 (0.724) | 0.7374 | 0.7500 | 0.7584 | 0.7738 | 0.7888 | 0.7911 | 0.7895 | 0.7905 | 0.7857 | 0.8151 |
| DDEF (adapted) | 0.1574 (0.620) | 0.7323 | 0.7394 | 0.7360 | 0.7440 | 0.7516 | 0.7595 | 0.7632 | 0.7635 | 0.7786 | 0.8151 |
| FUML (adapted) | 0.1975 (0.156) | 0.7121 | 0.7287 | 0.7360 | 0.7440 | 0.7391 | 0.7342 | 0.7434 | 0.7500 | 0.7500 | 0.7647 |
| XGBoost | 0.1636 (0.878) | 0.7121 | 0.7234 | 0.7247 | 0.7381 | 0.7391 | 0.7468 | 0.7500 | 0.7568 | 0.7500 | 0.8235 |
| CatBoost | 0.1519 (0.494) | 0.7222 | 0.7287 | 0.7360 | 0.7440 | 0.7516 | 0.7595 | 0.7632 | 0.7568 | 0.7714 | 0.7983 |
| TabPFN v2 | 0.1248 (0.026) | 0.7525 | 0.7606 | 0.7809 | 0.7976 | 0.8012 | 0.8101 | 0.8158 | 0.8243 | 0.8286 | 0.8571 |
| **DeepFuzzyTorch (Ours, Comp.)** | 0.1671 (---) | 0.7475 | 0.7553 | 0.7584 | 0.7679 | 0.7764 | 0.7785 | 0.7763 | 0.7770 | 0.7786 | 0.7983 |
| **DeepFuzzyTorch (Pure $C_{\mathcal{O}}$ ECDF)** | 0.1958 (---) | 0.7475 | 0.7553 | 0.7584 | 0.7679 | 0.7826 | 0.7911 | 0.8026 | 0.8041 | 0.7929 | 0.7983 |

## Table 5: TCGA Pan-Cancer 3-Modality Multi-Omics + WSI Benchmark ($N=2946$ Unique Patients Across 7 Cohorts)

| Method | Pan-Cancer AUROC (95% CI) | OS C-Index | 7-Subset Macro AUROC | BRCA | COAD | KIRC | KIRP | LUAD | LUSC | UCEC | Raw $p$ | BH $q_{\text{BH}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EarlyConcat-MLP | **0.7394** (0.7195–0.7558) | 0.7321 | 0.6898 | 0.7110 | 0.7036 | 0.8053 | 0.7094 | 0.6523 | 0.6528 | 0.6580 | 0.784 | 0.896 |
| LateAvg-Ensemble | 0.7298 (0.7103–0.7479) | 0.7348 | --- | --- | --- | --- | --- | --- | --- | --- | 0.026 | 0.181 |
| Dempster-Shafer Fusion | 0.7354 (0.7158–0.7536) | 0.7404 | --- | --- | --- | --- | --- | --- | --- | --- | 0.392 | 0.627 |
| GatedAttention-Transformer | 0.7268 (0.7059–0.7462) | 0.7164 | 0.6775 | 0.6974 | 0.6798 | 0.8097 | 0.6639 | 0.6277 | 0.6480 | 0.6507 | 0.068 | 0.181 |
| EvidentialDL (Dirichlet TMC) | 0.7363 (0.7177–0.7542) | 0.7375 | 0.6928 | 0.6721 | 0.7200 | 0.8071 | 0.6983 | 0.6440 | 0.6553 | 0.6673 | 0.622 | 0.829 |
| RCML (adapted, AAAI'24) | 0.7322 (0.7129–0.7504) | 0.7350 | 0.6931 | 0.6688 | 0.7266 | 0.7998 | 0.6955 | 0.6418 | 0.6525 | 0.6325 | 0.054 | 0.181 |
| QMF (ICML'23) | 0.7321 (0.7116–0.7500) | 0.7276 | 0.6898 | 0.6680 | 0.7147 | 0.8000 | 0.7013 | 0.6361 | 0.6534 | 0.6517 | 0.190 | 0.380 |
| DrFuse (adapted, AAAI'24) | 0.7376 (0.7177–0.7546) | 0.7227 | 0.6838 | 0.7043 | 0.7020 | 0.8204 | 0.6770 | 0.6410 | 0.6435 | 0.6729 | 0.982 | 0.982 |
| **DeepFuzzyTorch (Ours)** | 0.7377 (0.7183–0.7561) | **0.7492** | 0.6944 | 0.6624 | 0.7270 | 0.8227 | 0.7154 | 0.6594 | 0.6432 | 0.6170 | --- | --- |

## Table 5B: Supervised In-Fold TCGA PanCancer Atlas RNA-Seq Reduct on the Unsupervised Label-Free Transductive 1,000-MAD Gene Pool ($1,000 \to K^*=20$ Genes, 5-Fold Jaccard = `0.3816`, 94.05% RNA Branch Parameter Reduction)

### Table 5B.1: Supervised In-Fold Feature-Selection Baseline Comparison on the Unsupervised Label-Free Transductive 1,000-MAD Gene Pool ($K^*=20$)

| In-Fold Feature Selector ($1,000 \to 20$) | 4-Modality AUROC (95% CI) | OS C-Index | Observed RNA-Only AUROC | Missing WSI & Staging (`RNA+GenSig`) AUROC | 5-Fold Pairwise Jaccard | Fit Time / Fold (s) | Raw $p$ | BH $q_{\text{BH}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| PCA-20 (Unsupervised Orthogonal Projection) | **0.7450** (0.7254–0.7625) | **0.7569** | **0.6627** | **0.6639** | N/A (Subspace) | 0.01±0.00 | 0.018 | 0.072 |
| Mutual Information Top-20 (`MI-20`) | 0.7417 (0.7227–0.7588) | 0.7520 | 0.6324 | 0.6485 | 0.1058 (0.053–0.143) | 2.21±0.30 | 0.536 | 0.715 |
| $L_1$-Regularized Logistic Regression (`Lasso-20`) | 0.7398 (0.7206–0.7573) | 0.7508 | 0.6337 | 0.6308 | 0.1985 (0.143–0.250) | 2.09±0.29 | 0.842 | 0.842 |
| Min-Redundancy Max-Relevance (`mRMR-20`) | 0.7378 (0.7187–0.7561) | 0.7528 | 0.6489 | 0.6476 | 0.4696 (0.379–0.600) | 0.01±0.00 | 0.126 | 0.252 |
| **Pointwise T-Norm Fuzzy-Rough Reduct (`FuzzyRough-20`, Ours)** | 0.7403 (0.7208–0.7579) | 0.7514 | 0.6464 | 0.6540 | 0.3816 (0.290–0.481) | 106.86±6.15 | --- | --- |

### Table 5B.2: 4-Modality Pan-Cancer Multimodal Fusion Comparison ($N=2,946$)

| 4-Modality Configuration (WSI + RNA-Seq + 10-Feature Signatures + Clinical) | Pan-Cancer AUROC (95% CI) | OS C-Index | Acc | F1 | Brier $\downarrow$ | ECE $\downarrow$ | Raw $p$ | BH $q_{\text{BH}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EarlyConcat-MLP (Unreduced 1,000 Genes) | 0.6932 (0.6721–0.7123) | 0.6945 | 0.6670 | 0.5427 | 0.2588 | 0.2065 | <0.001 | 0.002 |
| EarlyConcat-MLP (Top-20 Gene Reduct) | 0.7320 (0.7116–0.7492) | 0.7365 | 0.6796 | 0.5732 | 0.2048 | 0.0814 | 0.098 | 0.163 |
| LateAvg-Ensemble (Top-20 Gene Reduct) | 0.7285 (0.7088–0.7462) | 0.7340 | 0.6823 | 0.5726 | 0.2030 | 0.0815 | <0.001 | 0.002 |
| Dempster-Shafer Fusion (Top-20 Gene Reduct) | 0.7373 (0.7181–0.7547) | 0.7441 | 0.6918 | 0.5910 | 0.2005 | 0.0790 | 0.308 | 0.342 |
| GatedAttention-Transformer (Top-20 Gene Reduct) | 0.7218 (0.7018–0.7388) | 0.7144 | 0.6741 | 0.5683 | 0.2222 | 0.1303 | <0.001 | 0.002 |
| EvidentialDL (Top-20 Gene Reduct) | 0.7377 (0.7184–0.7551) | 0.7413 | 0.6925 | 0.5798 | 0.1995 | 0.0651 | 0.396 | 0.396 |
| RCML (adapted, AAAI'24, Top-20 Gene Reduct) | 0.7350 (0.7152–0.7524) | 0.7384 | 0.6945 | 0.5771 | 0.1994 | 0.0745 | 0.062 | 0.124 |
| QMF (ICML'23, Top-20 Gene Reduct) | 0.7338 (0.7145–0.7511) | 0.7330 | 0.6884 | 0.5905 | 0.2028 | 0.0873 | 0.132 | 0.189 |
| DrFuse (adapted, AAAI'24, Top-20 Gene Reduct) | 0.7340 (0.7152–0.7522) | 0.7284 | 0.6925 | 0.5855 | 0.2075 | 0.0884 | 0.278 | 0.342 |
| DeepFuzzyTorch (Unreduced 1,000 Genes, Fair Matched Gates/Rough) | 0.6925 (0.6725–0.7114) | 0.6933 | 0.6453 | 0.5110 | 0.2412 | 0.1564 | <0.001 | 0.002 |
| **DeepFuzzyTorch (Ours, Top-20 Fuzzy-Rough Gene Reduct)** | **0.7403** (0.7208–0.7579) | **0.7514** | 0.6860 | 0.5923 | 0.2002 | 0.0889 | --- | --- |

- **Supervised In-Fold Consensus Top-20 Fuzzy-Rough Transcriptomic Genes (from Label-Free Transductive 1,000-MAD Pool)**: `IGF2BP3` (5/5, $\Delta\gamma=0.0053$), `SFTPA2` (5/5, $\Delta\gamma=0.0051$), `BARX1` (5/5, $\Delta\gamma=0.0034$), `HIF3A` (5/5, $\Delta\gamma=0.0028$), `NTS` (5/5, $\Delta\gamma=0.0027$), `FGA` (4/5, $\Delta\gamma=0.0044$), `IL6` (4/5, $\Delta\gamma=0.0043$), `SFTPD` (4/5, $\Delta\gamma=0.0041$), `GPR87` (4/5, $\Delta\gamma=0.0031$), `AADAC` (4/5, $\Delta\gamma=0.0025$), `PHACTR3` (4/5, $\Delta\gamma=0.0023$), `SCN9A` (3/5, $\Delta\gamma=0.0047$), `SOX2` (3/5, $\Delta\gamma=0.0038$), `GABRE` (3/5, $\Delta\gamma=0.0028$), `TOX3` (3/5, $\Delta\gamma=0.0024$), `STK31` (3/5, $\Delta\gamma=0.0023$), `KCNK3` (3/5, $\Delta\gamma=0.0023$), `EYA4` (2/5, $\Delta\gamma=0.0039$), `SEC14L4` (2/5, $\Delta\gamma=0.0032$), `GPR109A` (2/5, $\Delta\gamma=0.0030$)
- **Custom-Background Exact Hypergeometric Reactome Enrichment ($N_{\text{bg}}=1000$ candidate genes from unsupervised label-free transductive MAD pre-screen; BH family $m=676$ predeclared $K_{\text{bg}} \ge 2$ pathways; $k_{\text{genes}} \ge 2$ displayed=36: 8 with predeclared BH $\text{FDR}<0.05$)**:
  - `R-HSA-5686938` *Regulation of TLR by endogenous ligand*: $k_{\text{genes}}=3/6$ (25.00$\times$ enrichment), raw $p=1.32e-04$, BH $q=3.08e-02$ (Unique Genes: `FGA, SFTPA2, SFTPD`)
  - `R-HSA-166016` *Toll Like Receptor 4 (TLR4) Cascade*: $k_{\text{genes}}=3/7$ (21.43$\times$ enrichment), raw $p=2.28e-04$, BH $q=3.08e-02$ (Unique Genes: `FGA, SFTPA2, SFTPD`)
  - `R-HSA-168179` *Toll Like Receptor TLR1:TLR2 Cascade*: $k_{\text{genes}}=3/7$ (21.43$\times$ enrichment), raw $p=2.28e-04$, BH $q=3.08e-02$ (Unique Genes: `FGA, SFTPA2, SFTPD`)
  - `R-HSA-168898` *Toll-like Receptor Cascades*: $k_{\text{genes}}=3/7$ (21.43$\times$ enrichment), raw $p=2.28e-04$, BH $q=3.08e-02$ (Unique Genes: `FGA, SFTPA2, SFTPD`)
  - `R-HSA-181438` *Toll Like Receptor 2 (TLR2) Cascade*: $k_{\text{genes}}=3/7$ (21.43$\times$ enrichment), raw $p=2.28e-04$, BH $q=3.08e-02$ (Unique Genes: `FGA, SFTPA2, SFTPD`)
  - `R-HSA-391160` *Signal regulatory protein family interactions*: $k_{\text{genes}}=2/2$ (50.00$\times$ enrichment), raw $p=3.80e-04$, BH $q=3.21e-02$ (Unique Genes: `SFTPA2, SFTPD`)
  - `R-HSA-5688849` *Defective CSF2RB causes SMDP5*: $k_{\text{genes}}=2/2$ (50.00$\times$ enrichment), raw $p=3.80e-04$, BH $q=3.21e-02$ (Unique Genes: `SFTPA2, SFTPD`)
  - `R-HSA-5688890` *Defective CSF2RA causes SMDP4*: $k_{\text{genes}}=2/2$ (50.00$\times$ enrichment), raw $p=3.80e-04$, BH $q=3.21e-02$ (Unique Genes: `SFTPA2, SFTPD`)
  - `R-HSA-5687613` *Diseases associated with surfactant metabolism*: $k_{\text{genes}}=2/3$ (33.33$\times$ enrichment), raw $p=1.13e-03$, BH $q=8.47e-02$ (Unique Genes: `SFTPA2, SFTPD`)
  - `R-HSA-452723` *Transcriptional regulation of pluripotent stem cells*: $k_{\text{genes}}=2/5$ (20.00$\times$ enrichment), raw $p=3.67e-03$, BH $q=2.25e-01$ (Unique Genes: `HIF3A, SOX2`)

## Table 6: 6-Foundation-Model Hesitant Consensus (Block 2, Source OOF Adam-Trained) & Pythagorean TTA (Block 5) on Camelyon17 ($N=2,500$) & NCT-CRC ($N=2,349$)

| Benchmark Scenario (In-Fold PCA-64) | Source / Target $N$ | Best Single FM | MeanPool 6-FMs | MedianPool 6-FMs | Logit-Tent TTA (adapted) | Hesitant Fuzzy (Block 2) | Hesitant Fuzzy + PFS-TTA (Ours) | BH $q$ vs. Mean / Tent |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Camelyon17 (OOD Hospitals) | 1500 / 1000 | **0.9942** | 0.9915 | 0.9909 | 0.9914 | 0.9919 | 0.9919 | 0.536 / 0.536 |
| Camelyon17 (1 Corrupted FM) | 1500 / 1000 | **0.9942** | 0.9894 | 0.9912 | 0.9893 | 0.9922 | 0.9925 | 0.010 / 0.010 |
| NCT-CRC (External Cohort) | 999 / 1350 | 0.9760 | **0.9825** | 0.9818 | 0.9823 | 0.9814 | 0.9815 | 0.367 / 0.389 |
| NCT-CRC (1 Corrupted FM) | 999 / 1350 | 0.9760 | 0.9567 | **0.9812** | 0.9561 | 0.9781 | 0.9804 | 0.002 / 0.002 |

## Table 7: Computational Complexity, Wall-Clock Latency & Peak Live Tensor Memory (`cpu`, `arm`)

### Table 7A: Batch Scaling ($N \in \{64, 256, 1024, 4096\}$, $D=100$, $M=4$) & Modality Scaling ($M \in \{2, 4, 8, 16\}$, $N=256$, $D=100$)

| Model | $N=64$ Fwd+Bwd (ms / MiB) | $N=256$ Fwd+Bwd (ms / MiB) | $N=1024$ Fwd+Bwd (ms / MiB) | $N=4096$ Fwd+Bwd (ms / MiB) | $M=4 \to 8 \to 16$ Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| EarlyConcat | 0.32 / 0.9 | 0.62 / 2.6 | 1.69 / 10.0 | 5.77 / 39.6 | 0.63 → 0.93 → 1.52 |
| GatedAttention | 0.98 / 1.2 | 1.81 / 4.3 | 4.53 / 17.0 | 13.60 / 67.4 | 1.97 → 3.60 → 7.97 |
| EvidentialDL | 1.04 / 0.5 | 1.35 / 2.0 | 2.71 / 7.6 | 7.47 / 30.2 | 1.46 → 2.58 → 5.05 |
| RCML (adapted) | 0.84 / 0.5 | 1.18 / 2.0 | 2.55 / 7.6 | 7.33 / 30.2 | 1.29 → 2.32 → 4.93 |
| QMF | 0.65 / 0.6 | 0.98 / 2.1 | 2.30 / 8.3 | 6.73 / 32.8 | 1.06 → 1.84 → 3.57 |
| DrFuse (adapted) | 1.51 / 2.1 | 2.96 / 7.8 | 8.87 / 30.8 | 26.62 / 122.6 | 2.96 → 5.78 → 12.52 |
| **UnifiedDeepFuzzyModel (Ours)** | 3.70 / 1.8 | 5.38 / 6.3 | 10.28 / 24.5 | 26.60 / 97.4 | 5.41 → 8.82 → 15.94 |

- **`FuzzyRoughFeatureReducer` Offline Fit Scaling ($K^*=20$, 8 Null Permutations)**: $N=64$: `0.037s` (`1.9MiB`), $N=256$: `0.189s` (`29.6MiB`), $N=1024$: `1.388s` (`467.7MiB`), $N=4096$: `40.198s` (`3686.9MiB`); TCGA-scale ($N=1,024, D=1,000$): `2.190s` (`489.5MiB`).

## Exported White-Box Clinical Decision Rules (`DeepConceptTSKHead`, Precision=8)

- `Rule 1: IF MRI_NetSupport is LOW (c=-1.21046817, s=0.78434515, g=0.70293415) AND Plasma_NetSupport is HIGH (c=+0.53145021, s=0.77801806, g=0.68946695) AND FDG_NetSupport is HIGH (c=+0.71617073, s=0.78631312, g=0.68976200) AND Clin_NetSupport is MEDIUM (c=+0.02156039, s=0.77770233, g=0.70308208) AND APOE4 is HIGH (c=+0.81014568, s=0.79478186, g=0.70634025) AND MMSE is HIGH (c=+0.64622921, s=0.79508173, g=0.71516794) AND ADAS13 is LOW (c=-0.51061225, s=0.79350543, g=0.72452092) AND CDRSB is MEDIUM (c=-0.08696655, s=0.80625635, g=0.72631830) THEN Logit = +0.07537234 +0.11010553*MRI_NetSupport +0.01284528*Plasma_NetSupport +0.09810124*FDG_NetSupport +0.11041056*Clin_NetSupport +0.06192565*APOE4 -0.12409190*MMSE +0.11690342*ADAS13 +0.13564432*CDRSB`
- `Rule 2: IF MRI_NetSupport is LOW (c=-0.99922782, s=0.83870304, g=0.70333415) AND Plasma_NetSupport is HIGH (c=+0.33472717, s=0.89558482, g=0.73657620) AND FDG_NetSupport is LOW (c=-0.29583961, s=0.87559903, g=0.68909734) AND Clin_NetSupport is LOW (c=-0.34254384, s=0.81795973, g=0.72700584) AND APOE4 is HIGH (c=+0.61942506, s=0.82108992, g=0.69917977) AND MMSE is MEDIUM (c=+0.16614965, s=0.85098624, g=0.69719464) AND ADAS13 is LOW (c=-0.37936115, s=0.91458756, g=0.75452876) AND CDRSB is HIGH (c=+0.33185032, s=0.87133580, g=0.74479640) THEN Logit = +0.07615969 +0.14198942*MRI_NetSupport +0.25371945*Plasma_NetSupport +0.09199945*FDG_NetSupport +0.17207688*Clin_NetSupport -0.05645759*APOE4 -0.03525797*MMSE +0.23460656*ADAS13 +0.20979990*CDRSB`
- `Rule 3: IF MRI_NetSupport is MEDIUM (c=-0.13030300, s=0.80218136, g=0.69939554) AND Plasma_NetSupport is LOW (c=-1.03977740, s=0.79014105, g=0.69614702) AND FDG_NetSupport is LOW (c=-0.67050546, s=0.79463595, g=0.68867207) AND Clin_NetSupport is MEDIUM (c=-0.10371754, s=0.78528118, g=0.71900511) AND APOE4 is HIGH (c=+0.68907416, s=0.82439286, g=0.72120482) AND MMSE is HIGH (c=+0.51578158, s=0.79108268, g=0.72001940) AND ADAS13 is MEDIUM (c=+0.18882680, s=0.79573661, g=0.70457029) AND CDRSB is HIGH (c=+0.37714505, s=0.81032825, g=0.73537815) THEN Logit = +0.08330408 +0.12597466*MRI_NetSupport +0.08372324*Plasma_NetSupport +0.04891632*FDG_NetSupport +0.15511024*Clin_NetSupport +0.10192401*APOE4 -0.12894315*MMSE +0.08480065*ADAS13 +0.15485281*CDRSB`
- `Rule 4: IF MRI_NetSupport is MEDIUM (c=-0.02083903, s=0.90054542, g=0.70275587) AND Plasma_NetSupport is HIGH (c=+0.32280338, s=0.91612697, g=0.70903599) AND FDG_NetSupport is MEDIUM (c=+0.03872836, s=0.87620437, g=0.68979263) AND Clin_NetSupport is HIGH (c=+0.57711643, s=0.96379077, g=0.74933892) AND APOE4 is MEDIUM (c=+0.09827158, s=0.97671771, g=0.76543951) AND MMSE is MEDIUM (c=+0.19474193, s=0.84950465, g=0.69950247) AND ADAS13 is MEDIUM (c=+0.07445662, s=0.83391106, g=0.70374715) AND CDRSB is HIGH (c=+0.48981497, s=0.89194107, g=0.71307075) THEN Logit = +0.10057595 +0.12864080*MRI_NetSupport +0.13928451*Plasma_NetSupport +0.08474416*FDG_NetSupport +0.26039732*Clin_NetSupport +0.21942033*APOE4 -0.07298542*MMSE +0.06177503*ADAS13 +0.09806218*CDRSB`
- `Rule 5: IF MRI_NetSupport is LOW (c=-0.46784660, s=0.98205513, g=0.70703173) AND Plasma_NetSupport is MEDIUM (c=+0.15917119, s=0.99414581, g=0.70447499) AND FDG_NetSupport is LOW (c=-0.72747701, s=0.97750193, g=0.69130939) AND Clin_NetSupport is HIGH (c=+1.23972702, s=0.93650126, g=0.69694471) AND APOE4 is HIGH (c=+0.98145002, s=0.94742274, g=0.76647848) AND MMSE is LOW (c=-1.10914397, s=0.96530521, g=0.75127935) AND ADAS13 is HIGH (c=+0.51237315, s=0.97309947, g=0.72853357) AND CDRSB is MEDIUM (c=-0.02961428, s=0.95252407, g=0.74420977) THEN Logit = +0.14157306 +0.16437314*MRI_NetSupport +0.13232474*Plasma_NetSupport +0.11078682*FDG_NetSupport +0.05960159*Clin_NetSupport +0.19406943*APOE4 -0.22093694*MMSE +0.15357241*ADAS13 +0.21537533*CDRSB`
- `Rule 6: IF MRI_NetSupport is LOW (c=-0.61618173, s=0.88835335, g=0.73280948) AND Plasma_NetSupport is LOW (c=-0.30235690, s=0.94413459, g=0.73846990) AND FDG_NetSupport is MEDIUM (c=+0.17055708, s=0.91260546, g=0.68889290) AND Clin_NetSupport is HIGH (c=+0.44325787, s=0.94765270, g=0.73020726) AND APOE4 is LOW (c=-1.03530002, s=0.82156205, g=0.70123416) AND MMSE is LOW (c=-0.59967846, s=0.88751000, g=0.69203645) AND ADAS13 is LOW (c=-0.67190629, s=0.84240264, g=0.73098469) AND CDRSB is MEDIUM (c=-0.07136024, s=0.86504984, g=0.72713351) THEN Logit = -0.00173181 +0.25464988*MRI_NetSupport +0.27736852*Plasma_NetSupport +0.08682548*FDG_NetSupport +0.21347469*Clin_NetSupport +0.03655998*APOE4 -0.04391029*MMSE +0.12846011*ADAS13 +0.13984670*CDRSB`
