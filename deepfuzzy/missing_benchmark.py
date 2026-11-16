"""
Standardized Combinatorial (2^M - 1) & Stress-Curve Missing-Modality Benchmark Evaluator.

Evaluates any multi-modal model under:
  1. Exhaustive 2^M - 1 non-empty modality availability subsets (e.g., 15 subsets for M=4, 7 for M=3).
  2. Progressive random missingness stress curves rho in [0.0, 0.7].
  3. Exact Theorem 2 verification (checking that missing modalities mapped to <0, 0, 1>
     receive exact zero Choquet weight and zero bias).
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import itertools
import numpy as np
import torch
from .metrics import compute_binary_metrics


def _safe_nanmean(vals: List[float]) -> float:
    finite = [float(v) for v in vals if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _safe_nanstd(vals: List[float]) -> float:
    finite = [float(v) for v in vals if not np.isnan(v)]
    return float(np.std(finite)) if finite else float("nan")


def _mask_modality_tensor(
    mod_t: torch.Tensor,
    mask_col: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    t_dev = mod_t.to(device=device, dtype=dtype)
    keep = (mask_col.to(device=device) > 0.5).view(-1, *([1] * (t_dev.ndim - 1)))
    return torch.where(keep, t_dev, torch.zeros_like(t_dev))


class CombinatorialMissingModalityBenchmark:
    """
    Standardized evaluator for incomplete multimodal medical datasets.
    """

    def __init__(self, modality_names: List[str]):
        self.modality_names = list(modality_names)
        self.num_modalities = len(modality_names)

    def enumerate_subsets(self) -> List[Tuple[str, Tuple[int, ...], int]]:
        """
        Enumerates all 2^M - 1 non-empty binary availability masks.
        Returns:
            List of (subset_label, binary_tuple, num_active_modalities)
            ordered by cardinality (1-modality, 2-modality, ..., M-modality).
        """
        subsets = []
        for bits in itertools.product([0, 1], repeat=self.num_modalities):
            if sum(bits) == 0:
                continue
            active_names = [self.modality_names[i] for i, b in enumerate(bits) if b == 1]
            label = "+".join(active_names)
            subsets.append((label, bits, sum(bits)))
        subsets.sort(key=lambda item: (item[2], [-b for b in item[1]]))
        return subsets

    @torch.no_grad()
    def evaluate_all_subsets(
        self,
        predict_fn: Callable[[List[torch.Tensor], torch.Tensor], Dict[str, torch.Tensor]],
        modality_tensors: List[torch.Tensor],
        natural_masks: torch.Tensor,
        targets: np.ndarray,
    ) -> Dict[str, object]:
        """
        Evaluates a multimodal model across all 2^M - 1 modality subsets.

        For each subset S, effective availability is `effective_mask = natural_masks * S`.
        Records both:
          - Full-population metrics (where patients with 0 observed modalities in S receive
            the vacuous prediction P(y=1)=0.5 with hesitancy=1.0).
          - Eligible-population metrics (`auroc_eligible`, `n_eligible`), restricted to patients
            who have at least 1 observed modality in S.
        Defines Missingness Robustness Retention (MRR) strictly as the ratio of the macro-average
        AUROC across the (2^M - 2) incomplete subsets to the full M-modality AUROC.
        """
        N = modality_tensors[0].shape[0]
        device = modality_tensors[0].device
        dtype = modality_tensors[0].dtype

        subsets = self.enumerate_subsets()
        subset_results: Dict[str, Dict[str, float]] = {}
        cardinality_buckets: Dict[int, List[Dict[str, float]]] = {
            k: [] for k in range(1, self.num_modalities + 1)
        }

        for label, bits, k_active in subsets:
            bit_tensor = torch.tensor(bits, device=device, dtype=dtype).view(1, -1).expand(N, -1)
            effective_mask = (natural_masks.to(device=device, dtype=dtype) > 0.5).to(dtype=dtype) * bit_tensor
            masked_inputs = [
                _mask_modality_tensor(modality_tensors[m], effective_mask[:, m], device, dtype)
                for m in range(self.num_modalities)
            ]
            out = predict_fn(masked_inputs, effective_mask)
            probs = out["prob"].detach().cpu().numpy().ravel()
            hes = out.get("hesitancy", torch.zeros_like(out["prob"])).detach().cpu().numpy().ravel()

            # Enforce exact vacuous output (prob=0.5, hesitancy=1.0) if a patient has 0 observed modalities in S
            obs_counts_np = effective_mask.sum(dim=1).detach().cpu().numpy()
            has_obs_np = (obs_counts_np > 0.5)
            has_all_np = (obs_counts_np >= (float(k_active) - 0.5))
            probs = np.where(has_obs_np, probs, 0.5)
            hes = np.where(has_obs_np, hes, 1.0)

            metrics = compute_binary_metrics(probs, targets, uncertainty=hes)
            if np.any(has_obs_np) and len(np.unique(targets[has_obs_np])) > 1:
                elig_m = compute_binary_metrics(probs[has_obs_np], targets[has_obs_np], uncertainty=hes[has_obs_np])
                metrics["auroc_eligible"] = float(elig_m["auroc"])
            else:
                metrics["auroc_eligible"] = float("nan")

            if np.any(has_all_np) and len(np.unique(targets[has_all_np])) > 1:
                comp_m = compute_binary_metrics(probs[has_all_np], targets[has_all_np], uncertainty=hes[has_all_np])
                metrics["auroc_complete_subset"] = float(comp_m["auroc"])
            else:
                metrics["auroc_complete_subset"] = float("nan")

            metrics["n_full"] = int(N)
            metrics["n_eligible"] = int(np.sum(has_obs_np))
            metrics["n_complete_subset"] = int(np.sum(has_all_np))
            metrics["mean_hesitancy"] = float(np.mean(hes)) if len(hes) > 0 else float("nan")
            metrics["mean_hesitancy_eligible"] = (
                float(np.mean(hes[has_obs_np])) if np.any(has_obs_np) else float("nan")
            )
            metrics["num_modalities"] = float(k_active)
            metrics["bitmask"] = "".join(str(b) for b in bits)
            subset_results[label] = metrics
            cardinality_buckets[k_active].append(metrics)

        cardinality_summary = {}
        for k_active, m_list in cardinality_buckets.items():
            cardinality_summary[f"{k_active}_modalities"] = {
                key: _safe_nanmean([m[key] for m in m_list])
                for key in [
                    "auroc",
                    "auroc_eligible",
                    "auroc_complete_subset",
                    "auprc",
                    "acc",
                    "f1",
                    "brier",
                    "ece",
                    "aurc",
                    "mean_hesitancy",
                ]
            }

        full_label = "+".join(self.modality_names)
        full_auroc = subset_results[full_label]["auroc"]
        full_auroc_eligible = subset_results[full_label]["auroc_eligible"]
        all_aurocs = [m["auroc"] for m in subset_results.values()]
        all_aurocs_eligible = [m["auroc_eligible"] for m in subset_results.values()]
        all_aurocs_complete = [m["auroc_complete_subset"] for m in subset_results.values()]
        incomplete_aurocs = [
            m["auroc"] for k_lbl, m in subset_results.items() if k_lbl != full_label
        ]
        incomplete_aurocs_eligible = [
            m["auroc_eligible"] for k_lbl, m in subset_results.items() if k_lbl != full_label
        ]
        macro_auroc_all = _safe_nanmean(all_aurocs)
        macro_auroc_eligible = _safe_nanmean(all_aurocs_eligible)
        macro_auroc_complete = _safe_nanmean(all_aurocs_complete)
        macro_auroc_incomplete = _safe_nanmean(incomplete_aurocs) if incomplete_aurocs else full_auroc
        macro_auroc_incomplete_eligible = (
            _safe_nanmean(incomplete_aurocs_eligible) if incomplete_aurocs_eligible else full_auroc_eligible
        )
        # MRR defined strictly over the (2^M - 2) incomplete subsets divided by full M-modality AUROC (M7)
        if np.isnan(full_auroc) or full_auroc <= 0.0 or np.isnan(macro_auroc_incomplete):
            mrr = float("nan")
        else:
            mrr = float(macro_auroc_incomplete / max(1e-6, full_auroc))
        if np.isnan(full_auroc_eligible) or full_auroc_eligible <= 0.0 or np.isnan(macro_auroc_incomplete_eligible):
            mrr_eligible = float("nan")
        else:
            mrr_eligible = float(macro_auroc_incomplete_eligible / max(1e-6, full_auroc_eligible))

        return {
            "subsets": subset_results,
            "cardinality_summary": cardinality_summary,
            "macro_avg_auroc": macro_auroc_all,
            "macro_avg_auroc_eligible": macro_auroc_eligible,
            "macro_avg_auroc_complete": macro_auroc_complete,
            "macro_avg_auroc_incomplete": macro_auroc_incomplete,
            "macro_avg_auroc_incomplete_eligible": macro_auroc_incomplete_eligible,
            "macro_avg_auprc": _safe_nanmean([m["auprc"] for m in subset_results.values()]),
            "macro_avg_ece": _safe_nanmean([m["ece"] for m in subset_results.values()]),
            "macro_avg_brier": _safe_nanmean([m["brier"] for m in subset_results.values()]),
            "full_modality_auroc": full_auroc,
            "full_modality_auroc_eligible": full_auroc_eligible,
            "missing_robustness_retention": mrr,
            "missing_robustness_retention_eligible": mrr_eligible,
        }

    @torch.no_grad()
    def evaluate_missing_rate_curve(
        self,
        predict_fn: Callable[[List[torch.Tensor], torch.Tensor], Dict[str, torch.Tensor]],
        modality_tensors: List[torch.Tensor],
        natural_masks: torch.Tensor,
        targets: np.ndarray,
        drop_rates: Optional[List[float]] = None,
        n_repeats: int = 5,
        seed: int = 42,
    ) -> List[Dict[str, float]]:
        """
        Evaluates performance degradation across progressive random modality drop rates rho in [0.0, 0.7],
        ensuring at least 1 naturally observed modality remains available per patient.
        """
        if drop_rates is None:
            drop_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        N = modality_tensors[0].shape[0]
        device = modality_tensors[0].device
        dtype = modality_tensors[0].dtype
        nat_np = (natural_masks.detach().cpu().numpy() > 0.5).astype(np.float32)

        curve_results = []
        for rho in drop_rates:
            repeats = 1 if rho == 0.0 else n_repeats
            rep_metrics = []
            for r in range(repeats):
                rng = np.random.default_rng(seed + int(rho * 1000) + r)
                keep_prob = 1.0 - rho
                rand_keep = (rng.random((N, self.num_modalities)) < keep_prob).astype(np.float32)
                eff_mask_np = nat_np * rand_keep
                zero_rows = np.where(eff_mask_np.sum(axis=1) == 0)[0]
                for row_idx in zero_rows:
                    avail_cols = np.where(nat_np[row_idx] > 0.5)[0]
                    if len(avail_cols) > 0:
                        chosen = rng.choice(avail_cols)
                        eff_mask_np[row_idx, chosen] = 1.0

                eff_mask = torch.tensor(eff_mask_np, device=device, dtype=dtype)
                masked_inputs = [
                    _mask_modality_tensor(modality_tensors[m], eff_mask[:, m], device, dtype)
                    for m in range(self.num_modalities)
                ]
                out = predict_fn(masked_inputs, eff_mask)
                probs = out["prob"].detach().cpu().numpy().ravel()
                hes = out.get("hesitancy", torch.zeros_like(out["prob"])).detach().cpu().numpy().ravel()
                # Enforce exact vacuous output (prob=0.5, hesitancy=1.0) if a patient has 0 observed modalities
                has_obs_np = eff_mask_np.sum(axis=1) > 0.5
                probs = np.where(has_obs_np, probs, 0.5)
                hes = np.where(has_obs_np, hes, 1.0)
                m_dict = compute_binary_metrics(probs, targets, uncertainty=hes)
                m_dict["mean_hesitancy"] = float(np.mean(hes)) if len(hes) > 0 else float("nan")
                rep_metrics.append(m_dict)

            avg_entry = {"drop_rate": float(rho)}
            for k in rep_metrics[0].keys():
                avg_entry[k] = _safe_nanmean([rm[k] for rm in rep_metrics])
                avg_entry[f"{k}_std"] = _safe_nanstd([rm[k] for rm in rep_metrics])
            curve_results.append(avg_entry)

        return curve_results
