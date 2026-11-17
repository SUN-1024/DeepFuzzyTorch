# `DeepFuzzyTorch`: A Unified Differentiable Intuitionistic, Hesitant, and Fuzzy-Rough Granular Framework for Incomplete and Conflicting Multimodal Medical Learning

[![Python >= 3.9, < 3.14](https://img.shields.io/badge/Python-%3E%3D3.9%2C%3C3.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`DeepFuzzyTorch` (`deepfuzzy`) is a modular, GPU-accelerated, end-to-end differentiable high-order fuzzy and granular computing library built on PyTorch for uncertainty-aware multimodal medical artificial intelligence, computational pathology, neuroimaging, and multi-omics integration.

---

## 🌟 Key Highlights, Differentiable Modules & Mathematical Guarantees

1. **Block 1 — Differentiable Atanassov Intuitionistic Fuzzy Encoding & Conflict-Aware Choquet Fusion (`IntuitionisticFuzzyEncoder`, `IFSChoquetFusion`)**:
   - Maps arbitrary modality embeddings into Intuitionistic Fuzzy Numbers $\tilde{\alpha}_m = \langle \mu_m, \nu_m, \pi_m \rangle \in \Delta_{\text{IFN}}$ satisfying $\mu_m + \nu_m + \pi_m = 1$ with piecewise $C^\infty$ smoothness on fixed active modality sets $\mathcal{O}$ outside the measure-zero certainty-tie locus (and global $C^0$ continuity of $\pi_{\text{fused}}$ on fixed $\mathcal{O}$, **Theorem 1**, measured float32 closure residual `5.96e-08`).
   - **End-to-End Missing-Modality & Concept-Masked Invariance (Theorem 2 & Corollary 2.1)**: Unobserved modalities ($o_m = 0$) collapse algebraically to pure epistemic vacuity $\langle 0, 0, 1 \rangle$, receive exact zero Choquet weight ($w_m \equiv 0$, `0.00e+00` deviation in `IFSChoquetFusion`), and—paired with binary observation masking $\bm{m}_z \in \{0, 1\}^C$ inside `DeepConceptTSKHead`—guarantee exact `0.00` fixed-width padding-value discrepancy across the end-to-end `UnifiedDeepFuzzyModel` (exact algebraic submodel equivalence in real arithmetic; $\le 2.98 \times 10^{-8}$ float32 and $\le 1.39 \times 10^{-17}$ float64 on verified default-scale submodels).
   - **Partial-Order Monotonicity & Idempotency (Theorem 3)**: Along constant-certainty ($r_m = 1 - \pi_m$) contours under frozen conflict discount $d > 0$, fused membership strictly respects the Atanassov partial order ($\partial \mu_{\text{fused}} / \partial \mu_m > 0$, verified min partial derivative `0.2072`, analytic-vs-FD max error `5.20e-08`).
   - **Conflict-Driven Epistemic Hesitancy Inflation (Theorem 4)**: Holding modality non-vacuity certainty masses $r_m = \mu_m + \nu_m \in (0, 1)$ fixed, pairwise cross-polarity contradiction $C_{\mathcal{O}}$ exponentially discounts conflicting directional beliefs and strictly inflates fused hesitancy ($\partial \ln \pi_{\text{fused}} / \partial C_{\mathcal{O}} > 0$), resolving Zadeh's high-conflict normalization collapse and enabling selective clinical triage.
2. **Block 2 — Differentiable Hesitant Fuzzy Multi-Expert / Multi-FM Aggregator (`HesitantFuzzyAggregator`)**:
   - Aggregates hesitant membership sets $h(x) = \{\mu^{(1)}(x), \dots, \mu^{(K)}(x)\}$ across multiple pathology foundation models, prompts, or clinical raters (Torra HFS) via hesitant fuzzy dispersion / deviation degree $\pi_{\text{HFS}}(h(x)) = \min\big(0.95,\, 2\sqrt{\operatorname{Var}(h(x))}\big)$ following **Xu & Xia (2012)** and **Liao & Xu (2013)**, optimized via 150 Adam steps on out-of-fold source predictions.
3. **Block 3 — GPU-Differentiable Fuzzy-Rough Approximations & Offline Pointwise T-Norm Minimal Reduct (`DifferentiableFuzzyRoughLayer`, `FuzzyRoughAttentionGate`, `FuzzyRoughFeatureReducer`)**:
   - Bridges Dubois–Prade Fuzzy-Rough Set Theory with deep neural networks via projected Soft-Łukasiewicz operators guaranteeing $\underline{R}_{\text{exact}}(A) \le \underline{R}_{\tau,\beta}(A) \le A \le \overline{R}_{\tau,\beta}(A) \le \overline{R}_{\text{exact}}(A)$ (**Theorem 5, Part I**, `0.00e+00` inclusion violation) and MAD-standardized pointwise T-norm granular monotonicity with permutation-null calibration (**Theorem 5, Part II**).
   - Selects **supervised in-fold** biomarker reducts from 320 FreeSurfer MRI ROIs ($320 \to K^*=24$, 5-fold Jaccard `0.4330`, `81.77%` MRI branch parameter reduction) and from an unsupervised, label-free transductive 1,000-gene TCGA PanCancer Atlas MAD candidate pool ($1,000 \to K^*=20$, 5-fold Jaccard `0.3816`, `94.05%` RNA branch parameter reduction; sensitivity check from the cached 5,000-gene pool yields `0.9685` candidate Jaccard and `0.7389` AUROC).
4. **Block 4 — Concept-Masked White-Box Takagi–Sugeno–Kang (TSK) Fuzzy Rule Head (`DeepConceptTSKHead`)**:
   - Operates on named modality net-support scores and clinical biomarkers with explicit concept observation masking (`concept_mask`), exporting human-readable 8-decimal `IF-THEN` rules via `.export_clinical_rules(precision=8)` that reconstruct neural TSK logits within **`2.384e-07`** max absolute error via `.evaluate_exported_rules_text()`.
5. **Block 5 — Pythagorean Fuzzy Divergence Test-Time Adaptation (`PythagoreanFuzzyTTALoss`)**:
   - Relaxes linear simplex constraints to Yager's Pythagorean Fuzzy disk ($\mu_P^2 + \nu_P^2 \le 1$) for unsupervised source-free test-time adaptation across out-of-distribution (OOD) hospitals and corrupted foundation models.
6. **14 Built-In & Phase-1 Tabular/Fusion Comparators (`deepfuzzy/baselines.py`, `benchmarks/run_if_phase1_upgrades.py`)**:
   - Classical & neural fusion: `EarlyConcatMLP`, `LateAvgEnsemble`, `DeepTSKBaseline`, `DempsterShaferFusion`, `GatedAttentionFusion`, and `EvidentialDL` (`EvidentialDLBaseline`, Dirichlet TMC).
   - **2023–2025 incomplete, evidential & fuzzy fusion baselines**:
     - `QMFBaseline` — QMF (adapted), *Quality-Aware Multimodal Fusion* (Zhang et al., *ICML* 2023)
     - `RCMLBaseline` (adapted) — *Reliable Conflictive Multi-View Learning* adaptation (Xu et al., *AAAI* 2024)
     - `DrFuseBaseline` (adapted) — *Disentangled Representation Learning for Multimodal Healthcare with Missing Modalities* adaptation (Yao et al., *AAAI* 2024)
     - `DDEFBaseline` (adapted) — *Dual-Level Deep Evidential Fusion* adaptation (Shao et al., *Information Fusion* 2024)
     - `FUMLBaseline` (adapted) — *Fuzzy Possibility/Necessity Multi-View Learning* adaptation (Duan et al., *ICML* 2025)
   - **Tabular GBDT & foundation model comparators** (with explicit modality-presence indicators & native `NaN` / masked subset routing): `XGBoost` (Chen & Guestrin, 2016), `CatBoost` (Prokhorenkova et al., 2018), and `TabPFN v2` (Hollmann et al., *Nature* 2025).

---

## 📦 Installation

DeepFuzzyTorch requires Python >= 3.9, < 3.14 and has been verified on Python 3.10–3.12. The < 3.14 upper bound reflects upstream `catboost` binary wheel availability for the `[benchmark]` extra.

```bash
git clone https://github.com/SUN-1024/DeepFuzzyTorch.git
cd DeepFuzzyTorch
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

To reproduce the tabular comparators (`XGBoost`, `CatBoost`, and `TabPFN v2`), install the optional benchmark dependencies as well:

```bash
.venv/bin/pip install -e ".[dev,benchmark]"
```

---

## 🚀 Quickstart Example

```python
import numpy as np
import torch
from deepfuzzy import (
    UnifiedDeepFuzzyModel,
    IntuitionisticFuzzyLoss,
    CombinatorialMissingModalityBenchmark,
    CrossModalityConflictInjector,
    QMFBaseline,
    RCMLBaseline,
    DrFuseBaseline,
)

torch.manual_seed(42)

# 1. Initialize Unified DeepFuzzyTorch Model for 4 clinical modalities
#    e.g., [24-ROI MRI Reduct, Plasma Biomarkers, FDG-PET, Clinical/Genetics]
concept_names = [
    "MRI_NetSupport",
    "Plasma_NetSupport",
    "FDG_NetSupport",
    "Clin_NetSupport",
    "APOE4",
    "MMSE",
    "ADAS13",
    "CDRSB",
]
model = UnifiedDeepFuzzyModel(
    modality_dims=[24, 4, 1, 17],
    embed_dim=48,
    num_classes=1,
    conflict_penalty=0.65,
    use_rough_bottleneck=True,
    use_tsk_head=True,
    num_tsk_concepts=8,
    num_tsk_rules=6,
    concept_names=concept_names,
)
model.eval()

# 2. Forward pass with partially missing modalities (0 = missing, 1 = observed)
#    Pass clinical_concepts=x_clin[:, :4] so slots [4:8] bind directly to [APOE4, MMSE, ADAS13, CDRSB]
B = 16
x_mri = torch.randn(B, 24)
x_plasma = torch.randn(B, 4)
x_fdg = torch.randn(B, 1)
x_clin = torch.randn(B, 17)
modality_masks = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]] * (B // 2), dtype=torch.float32)

out = model(
    [x_mri, x_plasma, x_fdg, x_clin],
    modality_masks=modality_masks,
    clinical_concepts=x_clin[:, :4],
)
fused_ifn = out["fused_ifn"]  # IntuitionisticFuzzyTensor <mu, nu, pi>
print("Valid Simplex Closure:", fused_ifn.validate())
print("Predicted Probabilities:", out["prob"].shape, "Fused Hesitancy:", out["hesitancy"].shape)

# 3. Export human-readable IF-THEN clinical decision rules & measure text reconstruction error
rules = model.tsk_head.export_clinical_rules(gate_threshold=0.0, precision=8)
recon_logits = model.tsk_head.evaluate_exported_rules_text(
    rules,
    out["z_concepts"],
    concept_names=concept_names,
    concept_mask=out["concept_mask"],
)
max_err = float(np.max(np.abs(out["tsk_logits"].detach().cpu().numpy().ravel() - recon_logits)))
print(f"Exported {len(rules)} rules | Max text-to-logit reconstruction error: {max_err:.2e}")
print(rules[0])
```

---

## 📊 Summary of Multi-Center Empirical Benchmarks (`results/BENCHMARK_TABLES_SUMMARY.md`)

Evaluated across **5,356 unique clinical & oncology participants** (`1,384` deduplicated ADNI + `1,026` external OASIS-3 + `2,946` 7-cohort TCGA) and **4,849 histopathology patches/tiles** (`Camelyon17` + `NCT-CRC`) under strictly in-fold supervised modeling and Benjamini–Hochberg FDR control ($q_{\text{BH}}$, with disclosed unsupervised transductive 1,000-gene MAD candidate screening on 4-modality TCGA and full-batch transductive unlabeled TTA on Track 5):

| Benchmark Track & Cohort | Primary Evaluation Task | Key `DeepFuzzyTorch` Result & SOTA Comparison |
| :--- | :--- | :--- |
| **Track 0: Mathematical Audit** ($N=250$ Tuples, $40\times 40$ Grid, 50-Pt Sweep, $N_{\text{tok}}=45$) | Simplex closure, vacuity invariance, monotonicity, conflict inflation & rough bounds | Closure `5.96e-08`; missing-modality vacuity error **`0.00e+00`** ($w_{\text{miss}}=0.00$); fixed-conflict $\min \partial\mu/\partial\mu_1 = 0.2072$; rough violation `0.00e+00` |
| **Track 1: ADNI Cohort 1** ($N=1,137$, 58 Sites, 4 Modalities; Complete 4M $N_{\text{comp}}=854$) | Baseline amyloid-$\beta$ PET/CSF classification ($6/1,137$ [$0.53\%$] missing `ADAS13` in-fold median imputed) & $2^4-1=15$ subsets | Full AUROC `0.8270` (`0.8387` same-visit $n=924$; `0.8371` [95% CI `0.8091–0.8612`] complete 4M $N_{\text{comp}}=854$); no statistically significant full-cohort AUROC difference after BH correction across 13 neural/evidential/GBDT comparators ($q_{\text{BH}} \ge 0.100$), including the 2023–2025 fusion models [`QMF (adapted)`, `RCML`, `DrFuse`, `DDEF`, `FUML`] ($q_{\text{BH}} \ge 0.260$); `TabPFN v2` leads significantly with full-cohort AUROC `0.8443` ($q_{\text{BH}}=0.040$); **`MRR=0.9344`** & 15-subset Macro-AUROC `0.7764` (vs. `FUML` `0.7790`, `TabPFN v2` `0.7635`, `CatBoost` `0.7405`, `XGBoost` `0.7056`); **pure $C_{\mathcal{O}}$ swap AUROC `0.7000`** ($0.6542$ composite, $0.5337$ pure $\pi_f$) & conflict 20%-triage `0.8302` ($\gamma=0.8$); TSK text rule reconstruction error **`2.384e-07`** |
| **Track 2: Real-World Discordance & Cohort 2** ($N=1,112$ Unique ADNI; C2 $N=289$) | Strictly nested in-fold plasma/MRI/cognitive discordance triage & pTau217/GFAP screening | Discordance vs. concordance elevates conflict $C_{\mathcal{O}}$ **`1.51x`** ($n=198$ vs. $n=914$, Mann–Whitney **$p = 1.75 \times 10^{-17}$**); nested matched-coverage ECDF triage (`81.3%`, $161/198$) raises accuracy from `0.7475` to **`0.7764`** (`0.7826` pure $C_{\mathcal{O}}$; `0.7911` at $80\%$, eliminating GBDT threshold shrinkage `32.3%`–`50.0%`); Cohort 2 AUROC `0.8740` (`0.8844` on $\le 90$d $n=211$, $q_{\text{BH}} \ge 0.602$ across 12 comparators); exploratory baseline low eGFR (<60, $n=22$) Acc `0.8421` (`16/19`) |
| **Track 3A & 3B: Longitudinal & OASIS-3** (ADNI $N=1,017$; OASIS-3 $N=1,026$) | 60-month MCI-to-dementia conversion & 5-seed zero-shot external transfer | **#1 60m conversion AUROC `0.9419`** & **#1 C-index `0.8804`** (in-fold tertile log-rank $p = 3.34 \times 10^{-72}$); **#1 5-seed external OASIS-3 Dementia AUROC `0.8639 ± 0.0010`** (`DrFuse (adapted)` $q_{\text{BH}}=0.024$, `GatedAttn`/`EarlyConcat` $q_{\text{BH}}=0.028$; Strict-AD source `0.8603`) |
| **Track 4: 7-Cohort TCGA Pan-Cancer** ($N=2,946$; Unsupervised Transductive 1,000-MAD Pool) | 3-modality WSI+Omics fusion & **supervised in-fold** $1,000 \to K^*=20$ gene reduct | 3-Mod AUROC `0.7377`, **#1 OS C-index `0.7492`**, **#1 7-subset Macro-AUROC `0.6944`**; **#1 fusion-model 4-Mod Top-20 AUROC `0.7403`** & OS C-index `0.7514` (`3.61x` higher Jaccard `0.3816` vs. `MI-20` `0.1058` at `0.7417`, `mRMR-20` C-index `0.7528`, `MI-20` `0.7520`; vs. fair 1,000-gene `DeepFuzzyTorch` `0.6925`, **$q_{\text{BH}} = 0.0025$**); custom-BG ($N_{\text{bg}}=1,000$) Reactome ORA recovers **`8` pathways at BH $\text{FDR}<0.05$ across all `676` predeclared $K_{\text{bg}}\ge 2$ BG pathways** ($q_{\text{BH}}=0.0308$, led by `R-HSA-5686938`) |
| **Track 5: 6-FM Pathology Consensus** (`Camelyon17` $N=2,500$; `NCT-CRC` $N=2,349$) | OOD hospital / cohort transfer & 1-corrupted-FM resilience (`PCA-64`, `uni2_h`/`phikon_v2`/`kaiko_vitl14`/`phikon`/`ctranspath`/`plip`); NCT-CRC binary tumor-or-stroma endpoint (`is_tumor_or_stroma`) derived from 9-class `NCT-CRC-HE-100K` / `CRC-VAL-HE-7K` | Under 1 corrupted FM: **`0.9925` AUROC** on Camelyon17 ($q_{\text{BH}}=0.010$ vs. `MeanPool` `0.9894` / `Logit-Tent (adapted)` `0.9893`) & **`0.9804` AUROC** on NCT-CRC ($q_{\text{BH}}=0.0025$ vs. `MeanPool` `0.9567` / `Logit-Tent (adapted)` `0.9561`) |
| **Track 6: Runtime & Memory Scaling** ($N \in [64, 4096]$, $M \in [2, 16]$) | Wall-clock forward+backward latency & peak live tensor memory | Linear memory scaling (`1.8 MiB` at $N=64 \to 97.4\text{ MiB}$ / `26.60 ms` fwd+bwd at $N=4,096$, matching latency & `20.6%` lighter than `DrFuse (adapted)` `26.62 ms` / `122.6 MiB`); at $N=256, M=4$, `4.90 ms` / `4.6 MiB` after $K^*=20$ reduction from raw $D_m=5,000$ excludes offline reducer fitting and raw input processing; pre-reduction peak memory is `123.1 MiB` vs. `DrFuse (adapted)` `92.3 MiB`. Offline reducer fitting at $N=1,024, D=1,000$ costs `2.190 s` / `489.5 MiB` |

A single baseline eGFR measurement does not establish the 3-month chronicity required by KDIGO.

> **AURC Tie-Group Convention**: `deepfuzzy.metrics.compute_aurc` averages empirical 0–1 errors within equal-uncertainty tie blocks for permutation invariance, whereas frozen `results/benchmark_results.json` retains execution-time index-ordered tie breaking (`31/39` Track 1–4 models—including all 4 `DeepFuzzyTorch` entries—agree to `0.00e+00`, `8/39` baseline/ablation AURCs differ by $\le 1.92 \times 10^{-5}$, and all `39/39` are identical at 4 decimal places).

---

## 🧪 Reproducing Unit Tests & Multi-Dataset Benchmarks

```bash
# Install editable package with dev/test dependencies (includes pytest)
.venv/bin/pip install -e ".[dev]"

# Include XGBoost, CatBoost, and TabPFN v2 for the tabular comparators
.venv/bin/pip install -e ".[dev,benchmark]"

# Run mathematical theorem & autograd verification suite (standalone & pytest)
PYTHONPATH=. .venv/bin/python tests/test_operators_and_theorems.py
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_operators_and_theorems.py

# Run DUA-free synthetic quickstart verification demo (no credentialed data required)
PYTHONPATH=. .venv/bin/python examples/quickstart_synthetic_demo.py

# Stage 1: Fetch unsupervised transductive 1,000-MAD-gene TCGA PanCanAtlas pool & Reactome GMT
PYTHONPATH=. .venv/bin/python benchmarks/fetch_tcga_1000genes_cbioportal.py

# Stage 2: Core zero-leakage multi-track benchmark (incl. QMF (adapted), adapted RCML & DrFuse, Logit-Tent TTA)
PYTHONPATH=. .venv/bin/python benchmarks/run_full_benchmark.py --device cuda --data-root /path/to/datasets

# Stage 3: Strictly nested in-fold ADNI discordance audit, TCGA selector baselines & custom-BG Reactome ORA
PYTHONPATH=. .venv/bin/python benchmarks/run_upgrades_and_pathway_audit.py --device cuda

# Stage 3B: Phase-1 upgrades (N_comp=854 audit, DDEF, FUML, XGBoost, CatBoost, TabPFN v2, matched-coverage ECDF)
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=. .venv/bin/python benchmarks/run_if_phase1_upgrades.py

# Stage 4: Wall-clock runtime & peak tensor memory scaling + SHA-256 artifact & figure lock
PYTHONPATH=. .venv/bin/python benchmarks/run_runtime_vram_benchmark.py
PYTHONPATH=. .venv/bin/python benchmarks/update_r3_gpu_artifacts.py --device cuda

# Stage 5: Export synchronized Markdown tables & single-venue manuscript templates
PYTHONPATH=. .venv/bin/python benchmarks/export_tables_and_manuscript.py
PYTHONPATH=. .venv/bin/python benchmarks/build_companion_manuscripts.py
```

**Data and reproducibility:** The benchmark scripts consume harmonized credentialed CSV exports and pre-extracted feature caches; see [`data_benchmark/README_DATA_ACCESS.md`](data_benchmark/README_DATA_ACCESS.md) for access and preparation. The offline `FuzzyRoughFeatureReducer` is a tensor-accelerated pointwise T-norm greedy selector, distinct from the differentiable neural modules. The 8-decimal TSK text export achieves an empirical maximum reconstruction error of `2.384e-07` on Fold 0; rounding means this is not exact functional identity in real arithmetic.

---

## 📄 Manuscript Formatting Templates (Single-Submission Policy)

The `paper/` directory provides **three mutually exclusive single-submission LaTeX formatting templates** compiled from the same verified codebase, theorems, and benchmark artifacts:
- **`paper/manuscript_Information_Fusion.tex` (Primary Target Venue — Elsevier *Information Fusion*)**: Full operator-library and mathematical fusion manuscript paired with plain-text `paper/Highlights_Information_Fusion.md`.
- **`paper/manuscript_Genome_Biology_Software.tex` (Alternative Template — BMC *Genome Biology* Software Format)**: Structured `Background` / `Results` / `Conclusions` abstract, hierarchical `Implementation` section, and mandatory BMC `Declarations` & software availability metadata.
- **`paper/manuscript_Nature_Communications.tex` (Alternative Template — *Nature Portfolio* Results-First Format)**: Concise $\le 150$-word unreferenced abstract, strict `Introduction` $\to$ `Results` $\to$ `Discussion` $\to$ `Methods` $\to$ `Data/Code Availability` hierarchy, and $\le 10$ main-text floats (with `Algorithm 1` and secondary tables placed in `Supplementary Information`).

> **Note**: These three `.tex` files are alternative formatting templates for selecting **one** submission venue at a time—they are **not** separate papers and must never be co-submitted simultaneously.

---

## 📂 Repository Structure

```text
Fuzzy/
├── deepfuzzy/
│   ├── __init__.py                     # Public API exports (v0.2.3)
│   ├── operators.py                    # 5 core differentiable fuzzy/granular operators + UnifiedDeepFuzzyModel
│   ├── baselines.py                    # Neural/evidential/fuzzy comparators (incl. QMF (adapted), RCML, DrFuse, DDEF, FUML)
│   ├── missing_benchmark.py            # 2^M - 1 combinatorial & stress-curve missing-modality evaluator
│   ├── conflict_injector.py            # Cross-modality conflict injection & selective referral evaluator
│   ├── rough_reducer.py                # Pointwise T-norm Fuzzy-Rough minimal attribute reduct selector
│   ├── freesurfer_roi_dictionary.json  # Anatomical ROI dictionary for 320-feature ADNI MRI reducts
│   ├── metrics.py                      # AUROC, AUPRC, ECE, Brier, AURC, Bootstrap 95% CI, BH-FDR, DCA
│   └── visualization.py                # One-click publication figure generator (PDF + 300 DPI PNG)
├── benchmarks/
│   ├── fetch_tcga_1000genes_cbioportal.py  # Label-free transductive TCGA 1,000-MAD-gene fetcher
│   ├── run_full_benchmark.py               # Core multi-track benchmark across ADNI, OASIS-3, TCGA, 6-FM WSI
│   ├── run_upgrades_and_pathway_audit.py   # Nested N=1,112 ADNI discordance, transductive 1,000-gene TCGA & Reactome audit
│   ├── run_if_phase1_upgrades.py       # N_comp=854, DDEF/FUML/XGBoost/CatBoost/TabPFN v2, matched-coverage ECDF
│   ├── run_tabpfn_deepfuzzy_plugin.py     # Nested TabPFN–DeepFuzzy plug-in audit
│   ├── run_runtime_vram_benchmark.py       # Batch, modality & dimensionality latency/memory profiler
│   ├── update_r3_gpu_artifacts.py          # Preprocessor replay check, figure regeneration & SHA-256 lock
│   ├── export_tables_and_manuscript.py     # Markdown benchmark summary & table exporter
│   └── build_companion_manuscripts.py      # Alternative single-submission venue template builder (GB & NC)
├── tests/
│   └── test_operators_and_theorems.py
├── figures/                            # Publication figures (Fig0 - Fig6 in PDF and 300 DPI PNG)
├── results/                            # Quantitative JSON/NPZ artifacts, Markdown tables, SHA-256 manifest
│   ├── TABPFN_DEEPFUZZY_PLUGIN_SUMMARY.md # Aggregate plug-in audit summary
│   ├── tabpfn_deepfuzzy_plugin_audit.json # Aggregate plug-in audit and archive digest
│   └── tabpfn_plugin_oof_arrays.npz       # Local, DUA-restricted participant-level outputs
├── paper/
│   ├── THEOREM_PROOFS_COMPLETE.md          # Complete derivations of Definitions 1-2, Theorems 1-5 & Corollary 2.1
│   ├── Highlights_Information_Fusion.md    # Elsevier plain-text Highlights (<= 85 chars/bullet, zero LaTeX)
│   ├── manuscript_Information_Fusion.tex   # Primary submission manuscript (Information Fusion)
│   ├── manuscript_Genome_Biology_Software.tex  # Alternative venue template (Genome Biology Software)
│   ├── manuscript_Nature_Communications.tex    # Alternative venue template (Nature Portfolio Results-first)
│   └── references.bib
├── pyproject.toml
├── requirements.txt
└── README.md
```

The plug-in OOF archive is local and excluded from Git because ADNI participant-level
predictions and split IDs are restricted by the ADNI Data Use Agreement. The aggregate
JSON records its SHA-256 digest for authorized local verification.
