# DeepFuzzyTorch v0.2.4 Extended Sensitivity & Reviewer-Moat Audits Summary

**Status**: Executed cleanly across all local audit scripts (`v024_adni_fusion_exposure_ecdf_audit.json`, `v024_pathology_tta_ablation_audit.json`, `v024_tcga_inductive_nonlung_owa_audit.json`) with **zero modification** to the frozen main benchmark ledgers (`results/benchmark_results.json`, `results/oof_predictions_and_splits.npz`, `results/tabpfn_deepfuzzy_plugin_audit.json`). All reported values below are aggregate statistics (zero participant-level IDs or prediction vectors).

---

## 1. Track 1 / ADNI Fidelity, Exposure Symmetry & Strict Inner-Fold ECDF (`A1 + A2 + A3`)

- **Replay Integrity**: All 8 frozen C1 baselines (`DeepFuzzyTorch` `0.8270`, `FUML` `0.8363`, `QMF` `0.8313`, `DrFuse` `0.8316`, `DDEF` `0.8323`, `XGBoost` `0.8263`, `CatBoost` `0.8262`, `TabPFN` `0.8443`) and the `DualStream` TabPFN+DeepFuzzyTorch replay (`0.8349`) reproduced archived OOF AUROCs to `0.0` difference. Bitwise padding invariance across `900` probes (`{0, -999, 777, NaN}`) was `0.0` across all 7 added comparators (`FUML_Faithful`, `RFUML_Faithful`, `QMF_Faithful`, `MUSE_BatchGraph`, `XGBoost_ModDrop`, `CatBoost_ModDrop`, `TabPFN_ModDrop`).
- **Cohort Disentanglement**: In ADNI-1 (`N = 1,137`), `n = 924` share a same-visit baseline (`bl`), `n = 854` have all 4 modalities observed (`852` same-visit + `2` non-baseline complete), and `n = 872` are conflict-eligible (`Plasma` plus at least one other modality observed; `n = 858` have `MRI+Plasma`); external validation ADNI-2 (`n = 289`).

| Model | C1 AUROC (`N=1,137`) [95% CI] | C1 BL (`n=924`) | C1 Complete-4 (`n=854`) | C1 15-Sub Macro | BL-924 15-Sub Macro | ADNI-2 (`n=289`) | $\gamma=0.8$ (Corrupt -> r20% / Det) | $n=198$ Inner AURC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `DeepFuzzyTorch` | `0.8270` [`0.8005, 0.8486`] | `0.8387` | `0.8371` | `0.7764` | `0.7921` | `0.8740` | `0.8062 -> 0.8302` / `0.6542` | `0.1634` |
| `FUML` | `0.8363` [`0.8121, 0.8584`] | `0.8477` | `0.8453` | `0.7790` | `0.7944` | `0.8608` | `0.8029 -> 0.8307` / `0.5266` | `0.1973` |
| `QMF` | `0.8313` [`0.8071, 0.8540`] | `0.8449` | `0.8394` | `0.7759` | `0.7911` | `0.8414` | `0.7948 -> 0.8212` / `0.6667` | `0.1691` |
| `DrFuse` | `0.8316` [`0.8062, 0.8528`] | `0.8411` | `0.8389` | `0.7690` | `0.7819` | `0.8598` | `0.7965 -> 0.8211` / `0.6602` | `0.1725` |
| `DDEF` | `0.8323` [`0.8069, 0.8537`] | `0.8435` | `0.8414` | `0.7740` | `0.7884` | `0.8595` | `0.7986 -> 0.8308` / `0.5888` | `0.1589` |
| `XGBoost` | `0.8263` [`0.8009, 0.8487`] | `0.8351` | `0.8333` | `0.7056` | `0.7206` | `0.8749` | `0.7935 -> 0.8256` / `0.5915` | `0.1584` |
| `CatBoost` | `0.8262` [`0.8009, 0.8493`] | `0.8375` | `0.8370` | `0.7405` | `0.7557` | `0.8741` | `0.8041 -> 0.8379` / `0.5557` | `0.1497` |
| `TabPFN` | `0.8443` [`0.8196, 0.8664`] | `0.8569` | `0.8541` | `0.7635` | `0.7794` | `0.8830` | `0.8036 -> 0.8325` / `0.5893` | `0.1208` |
| `FUML_Faithful` (`A1`) | `0.8044` [`0.7782, 0.8286`] | `0.8200` | `0.8180` | `0.7116` | `0.7317` | `0.8384` | `0.7859 -> 0.8001` / `0.5749` | `0.2069` |
| `RFUML_Faithful` (`A1`) | `0.7973` [`0.7711, 0.8229`] | `0.8142` | `0.8117` | `0.7094` | `0.7306` | `0.8307` | `0.7790 -> 0.7877` / `0.5639` | `0.2327` |
| `QMF_Faithful` (`A1`) | `0.8294` [`0.8054, 0.8517`] | `0.8428` | `0.8374` | `0.7757` | `0.7905` | `0.8408` | `0.7977 -> 0.8249` / `0.6611` | `0.1780` |
| `MUSE_BatchGraph` (`A1`) | `0.8345` [`0.8095, 0.8567`] | `0.8481` | `0.8454` | `0.7752` | `0.7913` | `0.8623` | `0.8070 -> 0.7984` / `0.5297` | `0.3262` |
| `XGBoost_ModDrop` (`A2`) | `0.8260` [`0.8002, 0.8482`] | `0.8359` | `0.8301` | `0.7705` | `0.7842` | `0.8645` | `0.7956 -> 0.8251` / `0.5667` | `0.1775` |
| `CatBoost_ModDrop` (`A2`) | `0.8234` [`0.7967, 0.8469`] | `0.8301` | `0.8257` | `0.7612` | `0.7720` | `0.8703` | `0.8026 -> 0.8366` / `0.5508` | `0.1767` |
| `TabPFN_ModDrop` (`A2`) | `0.8192` [`0.7934, 0.8423`] | `0.8250` | `0.8174` | `0.7663` | `0.7779` | `0.8529` | `0.7892 -> 0.8207` / `0.5544` | `0.1950` |

- **Key `A1` Insights**: `QMF_Faithful` matches simplified `QMF` (`0.8294` vs `0.8313`). Pure Dubois–Prade possibilistic conjunction (`FUML_Faithful` `0.8044`, `RFUML_Faithful` `0.7973`) suffers under partial missingness (`15-sub 0.7116 / 0.7094`) compared to calibrated amortized fuzzy fusion (`0.7764`). `MUSE_BatchGraph` achieves strong batch-aligned AUROC (`0.8345`) but violates sample-level invariance with **`0.3770` max probability flip** under batch-composition perturbation (vs `1.19e-7` float32 roundoff for `DeepFuzzyTorch`), drops on $\gamma=0.8$ top-20% referral (`0.8070 -> 0.7984`), and collapses on discordant `n=198` referral (`0.3262` inner AURC, `0.3376` outer, $p < 0.001$).
- **Key `A2` Insights**: Giving GBDT baselines symmetric `p_drop=0.20` training exposure raises `XGBoost_ModDrop` 15-subset AUROC substantially (`0.7056 -> 0.7705` on `N=1,137`; `0.7206 -> 0.7842` on `bl924`), narrowing its deficit to `DeepFuzzyTorch` (`0.7764` / `0.7921`). By contrast, `TabPFN_ModDrop` drops on full-cohort AUROC (`0.8443 -> 0.8192`, `q_BH = 0.0023`), ADNI-2 (`0.8830 -> 0.8529`), ECE (`0.0269 -> 0.0477`), and discordant referral (`0.1208 -> 0.1950`, Spearman `0.2335`), showing that synthetic missingness masks degrade fixed-prior tabular in-context calibration.
- **Key `A3` Insights**: Evaluating strictly on same-visit baseline (`bl924`) confirms identical relative robustness across 15 subsets (`DeepFuzzyTorch 0.7921`, `FUML 0.7944`, `TabPFN 0.7794`, `XGBoost_ModDrop 0.7842`). On `n=198` deduplicated discordant cases (`159` ADNI-1 + `39` non-overlapping ADNI-2), strictly **inner-3-fold OOF ECDF** referral ranking (`rank_inner`, with per-inner-fold MRI feature selection on both C1 and C2) matches or slightly improves over in-sample outer ECDF (`rank_outer`) for `DeepFuzzyTorch` (`0.1634` vs `0.1671` AURC; Spearman rank correlation `0.9443`; 100%/81.3%/80%/60% retained accuracy `0.7475 -> 0.7764 -> 0.7785 -> 0.7899`) and outperforms pure `(C, O)` referral (`0.2015`, $p = 0.042$, $q_{\text{BH}} = 0.1575$). Meanwhile `TabPFN`'s entropy referral reaches `0.1208` ($p = 0.010$, $q_{\text{BH}} = 0.0750$).

---

## 2. Track 5 / Pathology 6-FM TTA Component Ablation & 6-Regime Sensitivity (`A4`)

Exact parity anchors verified: Camelyon17 `plip_invert_neg2x` `PFS_TTA_Full = 0.992508`, `Tent_TTA = 0.989296`; NCT-CRC `PFS_TTA_Full = 0.980352`, `Tent_TTA = 0.956143`.

| Regime | Dataset | `PFS_TTA_Full` (AUROC / ECE) | `Tent_TTA` (AUROC / ECE) | `SAR_TTA` (AUROC / ECE) | `EATA_TTA` (AUROC / ECE) | `Abl_LinearIFS_q1` | `Abl_wo_DiscordanceReg` | `Abl_wo_SourceAnchor` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `plip_invert_neg2x` | Camelyon17 | `0.9925` / `0.0282` | `0.9893` / `0.1528` ($q=0.012$) | `0.9894` / `0.1559` ($q=0.012$) | `0.9860` / `0.1533` ($q=0.012$) | `0.9925` / `0.0279` | `0.9923` / `0.0289` | `0.9919` / `0.0341` |
| `plip_invert_neg2x` | NCT-CRC | `0.9804` / `0.0604` | `0.9561` / `0.1016` ($q=0.002$) | `0.9566` / `0.1087` ($q=0.002$) | `0.9554` / `0.0987` ($q=0.002$) | `0.9804` / `0.0598` | `0.9780` / `0.0563` ($q=0.009$) | `0.9747` / `0.0729` ($q=0.002$) |
| `single_fm_invert_macro` | Camelyon17 | `0.9914` / `0.0231` | `0.9883` / `0.1584` ($q=0.0015$) | `0.9884` / `0.1602` ($q=0.0015$) | `0.9875` / `0.1559` ($q=0.0015$) | `0.9915` / `0.0220` | `0.9907` / `0.0250` ($q=0.0015$) | `0.9909` / `0.0280` |
| `single_fm_invert_macro` | NCT-CRC | `0.9797` / `0.0594` | `0.9396` / `0.1113` ($q=0.0015$) | `0.9417` / `0.1142` ($q=0.0015$) | `0.9363` / `0.1129` ($q=0.0015$) | `0.9799` / `0.0591` | `0.9748` / `0.0553` ($q=0.0015$) | `0.9725` / `0.0728` ($q=0.0015$) |
| `dual_fm_gauss_noise` | Camelyon17 | `0.9912` / `0.0271` | `0.9884` / `0.0542` ($q=0.018$) | `0.9884` / `0.0570` ($q=0.018$) | `0.9879` / `0.0490` ($q=0.012$) | `0.9912` / `0.0270` | `0.9905` / `0.0287` | `0.9906` / `0.0329` |
| `dual_fm_gauss_noise` | NCT-CRC | `0.9807` / `0.0591` | `0.9694` / `0.0460` ($q=0.002$) | `0.9705` / `0.0461` ($q=0.002$) | `0.9694` / `0.0456` ($q=0.002$) | `0.9807` / `0.0588` | `0.9800` / `0.0576` ($q=0.057$) | `0.9787` / `0.0686` ($q=0.002$) |
| `clean_ood` | Camelyon17 | `0.9919` / `0.0212` | `0.9914` / `0.0202` | `0.9914` / `0.0206` | `0.9911` / `0.0237` | `0.9919` / `0.0211` | `0.9918` / `0.0206` | `0.9918` / `0.0313` |
| `clean_ood` | NCT-CRC | `0.9815` / `0.0569` | `0.9823` / `0.0407` | `0.9825` / `0.0386` | `0.9824` / `0.0435` | `0.9815` / `0.0566` | `0.9814` / `0.0568` | `0.9806` / `0.0691` |
| `temp_scale_miscalib` | Camelyon17 | `0.9857` / `0.0291` | `0.9907` / `0.0401` | `0.9903` / `0.0498` | `0.9907` / `0.0349` | `0.9857` / `0.0292` | `0.9857` / `0.0321` | `0.9833` / `0.0395` |
| `temp_scale_miscalib` | NCT-CRC | `0.9811` / `0.0510` | `0.9779` / `0.0445` | `0.9782` / `0.0431` | `0.9776` / `0.0447` | `0.9811` / `0.0508` | `0.9809` / `0.0489` | `0.9787` / `0.0639` ($q=0.012$) |
| `single_fm_dead_zero_macro` | Camelyon17 | `0.9916` / `0.0204` | `0.9908` / `0.0912` | `0.9909` / `0.0926` | `0.9908` / `0.0927` | `—` | `—` | `—` |
| `single_fm_dead_zero_macro` | NCT-CRC | `0.9809` / `0.0629` | `0.9815` / `0.0635` | `0.9817` / `0.0622` | `0.9814` / `0.0648` | `—` | `—` | `—` |

- **Component & Boundary Findings**: Across antagonistic inversion (`plip_invert_neg2x`, `single_fm_invert_macro`) and `dual_fm_gauss_noise`, `PFS_TTA_Full` decisively improves over `Tent_TTA`, `SAR_TTA`, and `EATA_TTA` ($q_{\text{BH}} \le 0.018$). Component ablations prove **discordance regularization** (`Abl_wo_DiscordanceReg` drops NCT-CRC macro `0.9797 -> 0.9748`, $q=0.0015$) and **quadratic source anchoring** (`Abl_wo_SourceAnchor` drops `0.9797 -> 0.9725`, $q=0.0015$; Fisher diagonal weighting is used by `EATA_TTA`) carry the adaptation gain, while linear intuitionistic $q_P=1$ (`Abl_LinearIFS_q1`: `0.9915` / `0.9799`) performs on par with quadratic Pythagorean $q_P=2$ (recorded candidly as a non-hyperbolic design finding). Under scalar temperature miscalibration plus an expert-specific logit offset on Camelyon17, entropy-minimizing `Tent`/`EATA` slightly lead AUROC (`0.9907` vs `0.9857`) while `PFS_TTA_Full` retains lower ECE (`0.0291` vs `0.0349–0.0498`).

---

## 3. Track 4 / TCGA Strictly Inductive 18,495-Gene, Non-Lung & Within-Cohort-Z Audits (`B1a + B1b + B1c`)

Protocol: Strictly inductive in-fold QC (`18,495 -> ~15,576–15,653` expressed genes) -> in-fold top-`1,000` MAD (or within-cohort training Z-residualized MAD) -> `K* = 20` selection, paired with identical `UnifiedDeepFuzzyModel` 4-modality training and continuous-time Breslow `CoxPH_Inductive20`.

| Cohort Protocol | Selector / Model | 4-Mod AUROC [95% CI] | Macro-Cohort AUROC | OS C-Index (Pooled / Within-Mean) | RNA-Only AUROC | 5-Fold Jaccard | Lung Markers | Time (s/fold) | $q_{\text{BH}}$ vs `All-1000` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B1a` All 7 (`N=2,946`, 1,059 ev) | `FuzzyRough-20` | `0.7389` [`0.7196, 0.7564`] | `0.6945` | `0.7520` / `0.7247` | `0.6401` | `0.3830` | `SFTPA2, SFTPD` | `41.5` | `0.0012` |
| `B1a` All 7 (`N=2,946`) | `OWA-FuzzyRough-20` | `0.7412` [`0.7218, 0.7587`] | `0.6978` | `0.7549` / `0.7281` | `0.6487` | `0.5075` | `SFTPA2, SFTPD` | `258.7` | `0.0012` |
| `B1a` All 7 (`N=2,946`) | `mRMR-20` | `0.7392` [`0.7203, 0.7573`] | `0.6955` | `0.7532` / `0.7271` | `0.6487` | `0.4132` | `SFTPA2` | `0.01` | `0.0012` |
| `B1a` All 7 (`N=2,946`) | `Lasso-20` / `MI-20` | `0.7400` / `0.7409` | `0.6965` / `0.6962` | `0.7498` / `0.7510` | `0.6390` / `0.6315` | `0.1889` / `0.1093` | `None` / `None` | `2.1` / `2.1` | `0.0012` |
| `B1a` All 7 (`N=2,946`) | `PCA-20` (uninterpretable LCs) | `0.7450` [`0.7253, 0.7622`] | `0.7031` | `0.7565` / `0.7294` | `0.6629` | `n/a` | `None` | `0.01` | `0.0012` |
| `B1a` All 7 (`N=2,946`) | `All-1000` / `CoxPH_Inductive20` | `0.6869` / `0.6839` | `0.6440` / `0.6383` | `0.6900` / `0.7076` (w: `0.6664/0.6770`) | `0.6317` / `0.6401` | `0.9499` / `0.3830` | `—` / `2` | `0` / `41.5` | `—` |
| `B1b` Non-Lung 5 (`n=2,065`, 617 ev) | `FuzzyRough-20` | `0.7361` [`0.7128, 0.7593`] | `0.7154` | `0.7622` / `0.7542` | `0.6142` | `0.3283` | `None` | `21.5` | `0.0012` |
| `B1b` Non-Lung 5 (`n=2,065`) | `OWA-FuzzyRough-20` | `0.7365` [`0.7129, 0.7600`] | `0.7159` | `0.7628` / `0.7546` | `0.6171` | `0.3687` | `None` | `141.7` | `0.0012` |
| `B1b` Non-Lung 5 (`n=2,065`) | `mRMR-20` / `Lasso-20` / `MI-20` | `0.7387` / `0.7483` / `0.7366` | `0.7190` / `0.7288` / `0.7171` | `0.7653` / `0.7650` / `0.7608` | `0.6370` / `0.6416` / `0.6009` | `0.5958` / `0.2578` / `0.0508` | `None` | `0.004–1.6` | `0.0012` |
| `B1b` Non-Lung 5 (`n=2,065`) | `All-1000` / `CoxPH_Inductive20` | `0.6993` / `0.6892` | `0.6836` / `0.6739` | `0.7217` / `0.7304` (w: `0.7141/0.7192`) | `0.6335` / `0.6142` | `0.9516` / `0.3283` | `—` | `0` / `21.5` | `—` |
| `B1c` Within-Cohort Z (`N=2,946`) | `FuzzyRough-20` | `0.7413` [`0.7219, 0.7588`] | `0.6968` | `0.7533` / `0.7230` | `0.5949` | `0.1895` | `None` | `42.7` | `0.0012` |
| `B1c` Within-Cohort Z (`N=2,946`) | `OWA-FuzzyRough-20` | `0.7419` [`0.7236, 0.7598`] | `0.6962` | `0.7518` / `0.7222` | `0.5791` | `0.1586` | `None` | `264.8` | `0.0012` |
| `B1c` Within-Cohort Z (`N=2,946`) | `mRMR-20` / `Lasso-20` | `0.7431` / `0.7408` | `0.6985` / `0.6960` | `0.7562` / `0.7473` | `0.6012` / `0.5576` | `0.2709` / `0.1341` | `None` | `0.01` / `3.1` | `0.0012` |
| `B1c` Within-Cohort Z (`N=2,946`) | `All-1000` / `CoxPH_Inductive20` | `0.6761` / `0.6952` | `0.6371` / `0.6653` | `0.6812` / `0.7310` (w: `0.6599/0.7036`) | `0.5783` / `0.5949` | `0.6767` / `0.1895` | `—` | `0` / `42.7` | `—` |

- **Scientific Conclusions (`B1`)**: Strictly inductive `18,495 -> QC -> MAD -> K*=20` confirms both the benefit of fuzzy-rough sparsity over `All-1000` (`+0.0520` AUROC and `+0.0620` OS C-index in `B1a`, $q_{\text{BH}}=0.0012$) and over continuous-time `CoxPH_Inductive20` (`+0.0550` AUROC, $+0.0444$ C-index), **and** transparently identifies cohort confounding in un-residualized pan-cancer variance screening (`SFTPA2`/`SFTPD` appear in `B1a` driving `8` Reactome pathways with FDR $< 0.05$, notably `R-HSA-5686938`, $q_{\text{BH}} = 0.0282$). Crucially, removing `LUAD`/`LUSC` (`B1b`: `n=2,065`) or applying within-cohort training Z-residualization (`B1c`: `N=2,946`) eliminates all lung surfactant markers (`0` Reactome FDR $< 0.05$ hits) while preserving 4-modality fusion discrimination (`0.7361–0.7419` AUROC; within-cohort mean OS C-index `0.7222–0.7546`; $+0.1464$ over RNA-only `0.5949` in `B1c`), while `OWA-FuzzyRough-20` increases 5-fold marker Jaccard stability in `B1a` (`0.3830 -> 0.5075`) and `B1b` (`0.3283 -> 0.3687`) at ~6x selector runtime.
