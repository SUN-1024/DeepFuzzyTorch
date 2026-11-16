"""
Cross-Modality Evidence Conflict Injector & Safety-Critical Referral Evaluator.

Simulates controlled inter-modality contradictions (opposite-class donor feature interpolation,
biomarker spike, or sign inversion on observed modalities within the same cross-validation fold)
and evaluates:
  1. Monotonic increase of internal Intuitionistic Fuzzy Conflict C_O and fused hesitancy pi_fused.
  2. Unsupervised Conflict Anomaly Detection AUROC (distinguishing clean vs. conflicted patients
     among eligible multi-modal recipients via combined conflict/hesitancy).
  3. Safety-critical Selective Clinical Referral curves (referring top-tau% highest-uncertainty
     patients to confirmatory clinical testing).
"""

from __future__ import annotations
from decimal import Decimal
import math
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .metrics import compute_binary_metrics


def _frexp_prod64(*factors: object) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes overflow-free factored product `(mantissa_f64, exp2_i64)` across float64 factors."""
    ts = [torch.as_tensor(f, dtype=torch.float64, device="cpu") for f in factors]
    b_sh = torch.broadcast_shapes(*(t.shape for t in ts))
    m_acc = torch.ones(b_sh, dtype=torch.float64)
    e_acc = torch.zeros(b_sh, dtype=torch.int64)
    is_zero = torch.zeros(b_sh, dtype=torch.bool)
    for t in ts:
        te = t.expand(b_sh)
        is_zero |= te == 0.0
        mt, et = torch.frexp(te)
        m_prod, en = torch.frexp(m_acc * mt)
        m_acc = m_prod
        e_acc = e_acc + et.to(dtype=torch.int64) + en.to(dtype=torch.int64)
    if bool(is_zero.any()):
        m_acc = torch.where(is_zero, torch.zeros_like(m_acc), m_acc)
        e_acc = torch.where(is_zero, torch.zeros_like(e_acc), e_acc)
    return m_acc, e_acc


def _frexp_sum_clamp64(terms: List[Tuple[torch.Tensor, torch.Tensor]], fmax: float) -> torch.Tensor:
    """Sums factored `(mantissa_f64, exp2_i64)` terms without overflow and clamps to `[-fmax, fmax]`."""
    b_sh = torch.broadcast_shapes(*(m.shape for m, _ in terms))
    exps_b = [(m.expand(b_sh), e.expand(b_sh)) for m, e in terms]
    max_e = torch.full(b_sh, -(1 << 30), dtype=torch.int64)
    for m_i, e_i in exps_b:
        nz = m_i != 0.0
        if bool(nz.any()):
            max_e = torch.where(nz, torch.maximum(max_e, e_i), max_e)
    any_nz = max_e > -(1 << 29)
    s_acc = torch.zeros(b_sh, dtype=torch.float64)
    for m_i, e_i in exps_b:
        nz = m_i != 0.0
        if not bool(nz.any()):
            continue
        shift = torch.where(nz, e_i - max_e, torch.zeros_like(max_e)).clamp(min=-1075, max=0).to(dtype=torch.int32)
        term_v = torch.where(nz, torch.ldexp(m_i, shift), torch.zeros_like(m_i))
        s_acc = s_acc + term_v
    sm, sde = torch.frexp(s_acc)
    nz_s = any_nz & (sm != 0.0)
    fin_e = max_e + sde.to(dtype=torch.int64)
    safe_e = fin_e.clamp(min=-1080, max=1024).to(dtype=torch.int32)
    raw = torch.ldexp(sm, safe_e)
    fmax_f = float(fmax)
    sat = torch.copysign(torch.full_like(raw, fmax_f), sm)
    over = nz_s & ((fin_e > 1024) | (~torch.isfinite(raw)) | (raw.abs() > fmax_f))
    under = (~nz_s) | (fin_e < -1080)
    return torch.where(
        under,
        torch.zeros_like(raw),
        torch.where(over, sat, raw.clamp(-fmax_f, fmax_f)),
    )


def _safe_unscale_clamp64(u_raw: torch.Tensor, scale64: torch.Tensor, fmax: float) -> torch.Tensor:
    """Scales normalized float64 values by `scale64` and clamps cleanly within `[-fmax, fmax]`."""
    fmax_t = torch.full_like(u_raw, float(fmax))
    prod = u_raw * scale64
    sat = torch.copysign(fmax_t, u_raw)
    return torch.where(
        u_raw == 0.0,
        torch.zeros_like(u_raw),
        torch.where(torch.isfinite(prod), prod.clamp(-fmax, fmax), sat),
    )


def _safe_observed_std64_factored(
    x_full: torch.Tensor, row_indices: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes factored column-wise finite sample std (ddof=1) as `(std_u, col_scale)` on CPU float64."""
    out_shape = (1, *x_full.shape[1:])
    if len(row_indices) < 2 or x_full.is_complex() or x_full.dtype == torch.bool:
        return (
            torch.zeros(out_shape, dtype=torch.float64),
            torch.ones(out_shape, dtype=torch.float64),
        )
    idx_t = torch.as_tensor(row_indices, device=x_full.device, dtype=torch.long)
    sub = x_full[idx_t].detach().to("cpu", dtype=torch.float64)
    fin_m = torch.isfinite(sub)
    cnt = fin_m.sum(dim=0, keepdim=True)
    col_max = torch.where(fin_m, sub.abs(), torch.zeros_like(sub)).amax(dim=0, keepdim=True)
    safe_col_max = torch.where(col_max > 0.0, col_max, torch.ones_like(col_max))
    sub_u = torch.where(fin_m, sub / safe_col_max, torch.zeros_like(sub))
    mean_u = sub_u.sum(dim=0, keepdim=True) / cnt.clamp(min=1).to(dtype=torch.float64)
    diff_u = torch.where(fin_m, sub_u - mean_u, torch.zeros_like(sub_u))
    raw_scale = diff_u.abs().amax(dim=0, keepdim=True)
    scale = torch.where(raw_scale > 0.0, raw_scale, torch.ones_like(raw_scale))
    var_u = ((diff_u / scale) ** 2).sum(dim=0, keepdim=True) / (cnt - 1).clamp(min=1).to(dtype=torch.float64)
    std_u = scale * torch.sqrt(var_u)
    valid = (cnt >= 2) & (col_max > 0.0)
    return (
        torch.where(valid, std_u, torch.zeros_like(std_u)),
        torch.where(valid, safe_col_max, torch.ones_like(safe_col_max)),
    )


def _safe_observed_std64(x_full: torch.Tensor, row_indices: np.ndarray) -> torch.Tensor:
    """Computes column-wise finite sample std (ddof=1) in float64 without overflow or undefined <2-sample inflation."""
    std_u, col_scale = _safe_observed_std64_factored(x_full, row_indices)
    return _safe_unscale_clamp64(std_u, col_scale, float(torch.finfo(torch.float64).max))


def _safe_observed_std(x_full: torch.Tensor, row_indices: np.ndarray) -> torch.Tensor:
    """Computes column-wise finite sample std (ddof=1) over observed rows without NaN/overflow leakage."""
    std64 = _safe_observed_std64(x_full, row_indices)
    if not x_full.is_floating_point():
        return std64.to(device=x_full.device, dtype=torch.float32)
    fmax = float(torch.finfo(x_full.dtype).max)
    return std64.clamp(max=fmax).to(device=x_full.device, dtype=x_full.dtype)


class CrossModalityConflictInjector:
    """
    Standardized Cross-Modality Conflict Stress-Test & Selective Referral Evaluator.
    """

    def __init__(self, num_modalities: int, modality_names: Optional[List[str]] = None):
        self.num_modalities = num_modalities
        self.modality_names = modality_names or [f"Modality_{i}" for i in range(num_modalities)]

    def inject_conflict(
        self,
        modality_tensors: List[torch.Tensor],
        targets: np.ndarray,
        corrupted_modality_idx: int = 0,
        conflict_fraction: float = 0.5,
        severity_gamma: float = 1.0,
        mode: str = "opposite_swap",
        natural_masks: Optional[torch.Tensor] = None,
        fold_ids: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> Tuple[List[torch.Tensor], np.ndarray]:
        """
        Injects contradiction into `corrupted_modality_idx` for a subset of eligible samples
        where `corrupted_modality_idx` (and at least one other modality) is observed.

        Modes:
          - 'opposite_swap': Interpolates modality features with an observed opposite-class
            donor from the same fold, scaled by severity_gamma:
                x_m' = (1 - gamma) * x_m + gamma * (2.0 * x_m^{(opposite_donor)} - x_m)
          - 'sign_inversion': Reflects standardized feature vector:
                x_m' = (1 - 2.0 * gamma) * x_m
          - 'biomarker_spike': Adds directional opposite-polarity shift of magnitude 3.0 * gamma * std(x_m).

        Returns:
            corrupted_tensors: List of M tensors [N, D_m]
            is_conflicted: Binary indicator array [N] (1 = conflict injected, 0 = clean)
        """
        rng = np.random.default_rng(seed)
        N = modality_tensors[0].shape[0]
        y = np.asarray(targets).ravel().astype(int)

        if natural_masks is not None:
            mask_np = (natural_masks.detach().cpu().numpy() > 0.5).astype(int)
            has_target_mod = mask_np[:, corrupted_modality_idx] == 1
            has_other_mod = (mask_np.sum(axis=1) - mask_np[:, corrupted_modality_idx]) >= 1
            eligible_recipients = np.where(has_target_mod & has_other_mod)[0]
            observed_mod_pool = np.where(has_target_mod)[0]
        else:
            eligible_recipients = np.arange(N)
            observed_mod_pool = np.arange(N)

        if mode not in ("opposite_swap", "sign_inversion", "biomarker_spike"):
            raise ValueError(f"Unsupported conflict injection mode: {mode!r}")

        n_corrupt = int(round(len(eligible_recipients) * conflict_fraction))
        corrupt_indices = (
            rng.choice(eligible_recipients, size=n_corrupt, replace=False)
            if (n_corrupt > 0 and len(eligible_recipients) > 0)
            else np.array([], dtype=int)
        )
        is_conflicted = np.zeros(N, dtype=int)
        out_tensors = [t.clone() for t in modality_tensors]
        if n_corrupt == 0 or severity_gamma <= 0.0:
            return out_tensors, is_conflicted
        is_conflicted[corrupt_indices] = 1

        x_mod = out_tensors[corrupted_modality_idx]
        pos_obs = observed_mod_pool[y[observed_mod_pool] == 1]
        neg_obs = observed_mod_pool[y[observed_mod_pool] == 0]

        if mode == "opposite_swap":
            if len(pos_obs) == 0 or len(neg_obs) == 0:
                is_conflicted[:] = 0
                return out_tensors, is_conflicted
            donor_indices = np.arange(N)
            for idx in corrupt_indices:
                base_pool = neg_obs if y[idx] == 1 else pos_obs
                if fold_ids is not None:
                    same_fold_pool = base_pool[fold_ids[base_pool] == fold_ids[idx]]
                    if len(same_fold_pool) == 0:
                        is_conflicted[idx] = 0
                        continue
                    pool = same_fold_pool
                else:
                    pool = base_pool
                cand_x = x_mod[pool]
                ref_x = x_mod[idx].unsqueeze(0)
                reduce_dims = tuple(range(1, cand_x.ndim))
                if cand_x.is_floating_point() or cand_x.is_complex():
                    same_nan = torch.isnan(cand_x) & torch.isnan(ref_x)
                    diff_t = torch.any((cand_x != ref_x) & ~same_nan, dim=reduce_dims)
                    finite_cand = torch.isfinite(cand_x).all(dim=reduce_dims)
                    valid_fin = pool[(diff_t & finite_cand).detach().cpu().numpy()]
                    if bool(torch.isfinite(ref_x).all()):
                        valid_pool = valid_fin
                    else:
                        valid_pool = valid_fin if len(valid_fin) > 0 else pool[diff_t.detach().cpu().numpy()]
                else:
                    diff_t = torch.any(cand_x != ref_x, dim=reduce_dims)
                    valid_pool = pool[diff_t.detach().cpu().numpy()]
                if len(valid_pool) == 0:
                    is_conflicted[idx] = 0
                    continue
                donor_indices[idx] = rng.choice(valid_pool)
            active_idx = np.where(is_conflicted == 1)[0]
            if len(active_idx) == 0:
                return out_tensors, is_conflicted
            active_t = torch.as_tensor(active_idx, device=x_mod.device, dtype=torch.long)
            donor_t = torch.as_tensor(donor_indices[active_idx], device=x_mod.device, dtype=torch.long)
            x_sel = x_mod[active_t]
            d_sel = modality_tensors[corrupted_modality_idx][donor_t]
            gamma_f = float(severity_gamma)
            # Strong opposite-class donor interpolation scaled by severity_gamma on selected rows only
            cand_sel = x_sel * (1.0 - gamma_f) + (2.0 * d_sel - x_sel) * gamma_f
            if x_sel.is_floating_point() and not bool(torch.isfinite(cand_sel).all()):
                fmax = float(torch.finfo(x_mod.dtype).max)
                x64 = x_sel.detach().to("cpu", dtype=torch.float64)
                d64 = d_sel.detach().to("cpu", dtype=torch.float64)
                x64_f = torch.where(torch.isfinite(x64), x64, torch.zeros_like(x64))
                d64_f = torch.where(torch.isfinite(d64), d64, torch.zeros_like(d64))
                cand_safe = _frexp_sum_clamp64(
                    [
                        _frexp_prod64(x64_f),
                        _frexp_prod64(-2.0, gamma_f, x64_f),
                        _frexp_prod64(2.0, gamma_f, d64_f),
                    ],
                    fmax,
                ).to(device=x_mod.device, dtype=x_mod.dtype)
                fix_m = (~torch.isfinite(cand_sel)) & torch.isfinite(x_sel) & torch.isfinite(d_sel)
                cand_sel = torch.where(fix_m, cand_safe, cand_sel)
            out_tensors[corrupted_modality_idx][active_t] = cand_sel
        elif mode == "sign_inversion":
            active_idx = np.where(is_conflicted == 1)[0]
            if len(active_idx) > 0:
                active_t = torch.as_tensor(active_idx, device=x_mod.device, dtype=torch.long)
                x_sel = x_mod[active_t]
                gamma_f = float(severity_gamma)
                cand_sel = x_sel * (1.0 - 2.0 * gamma_f)
                if x_sel.is_floating_point() and not bool(torch.isfinite(cand_sel).all()):
                    fmax = float(torch.finfo(x_mod.dtype).max)
                    x64 = x_sel.detach().to("cpu", dtype=torch.float64)
                    x64_f = torch.where(torch.isfinite(x64), x64, torch.zeros_like(x64))
                    cand_safe = _frexp_sum_clamp64(
                        [
                            _frexp_prod64(x64_f),
                            _frexp_prod64(-2.0, gamma_f, x64_f),
                        ],
                        fmax,
                    ).to(device=x_mod.device, dtype=x_mod.dtype)
                    fix_m = (~torch.isfinite(cand_sel)) & torch.isfinite(x_sel)
                    cand_sel = torch.where(fix_m, cand_safe, cand_sel)
                out_tensors[corrupted_modality_idx][active_t] = cand_sel
        else:
            active_idx = np.where(is_conflicted == 1)[0]
            if len(active_idx) > 0:
                gamma_f = float(severity_gamma)
                if fold_ids is not None:
                    std_u_sel = torch.zeros((len(active_idx), *x_mod.shape[1:]), dtype=torch.float64)
                    col_scale_sel = torch.ones((len(active_idx), *x_mod.shape[1:]), dtype=torch.float64)
                    act_folds = fold_ids[active_idx]
                    obs_folds = fold_ids[observed_mod_pool]
                    for f_val in np.unique(act_folds):
                        f_obs_rows = observed_mod_pool[obs_folds == f_val]
                        f_std_u, f_col_sc = _safe_observed_std64_factored(x_mod, f_obs_rows)
                        f_mask = torch.as_tensor(act_folds == f_val, dtype=torch.bool)
                        std_u_sel[f_mask] = f_std_u
                        col_scale_sel[f_mask] = f_col_sc
                else:
                    std_u_g, col_scale_g = _safe_observed_std64_factored(x_mod, observed_mod_pool)
                    std_u_sel = std_u_g.expand((len(active_idx), *x_mod.shape[1:]))
                    col_scale_sel = col_scale_g.expand((len(active_idx), *x_mod.shape[1:]))
                active_t = torch.as_tensor(active_idx, device=x_mod.device, dtype=torch.long)
                x_sel = x_mod[active_t]
                if x_sel.is_floating_point():
                    fmax = float(torch.finfo(x_mod.dtype).max)
                    x64 = x_sel.detach().to("cpu", dtype=torch.float64)
                    sign64 = torch.tensor(
                        np.where(y[active_idx] == 1, -1.0, 1.0), dtype=torch.float64
                    ).view((-1,) + (1,) * (x64.ndim - 1)).expand_as(x64)
                    x64_s = torch.where(torch.isfinite(x64), x64, torch.zeros_like(x64))
                    cand_fin = _frexp_sum_clamp64(
                        [
                            _frexp_prod64(x64_s),
                            _frexp_prod64(3.0, sign64, std_u_sel, col_scale_sel, gamma_f),
                        ],
                        fmax,
                    ).to(device=x_mod.device, dtype=x_mod.dtype)
                    cand_sel = torch.where(torch.isfinite(x_sel), cand_fin, x_sel)
                else:
                    sign_sel = torch.tensor(
                        np.where(y[active_idx] == 1, -1.0, 1.0),
                        device=x_mod.device,
                        dtype=torch.float32,
                    ).view((-1,) + (1,) * (x_sel.ndim - 1))
                    std_sel = _safe_unscale_clamp64(
                        std_u_sel, col_scale_sel, float(torch.finfo(torch.float32).max)
                    ).to(device=x_mod.device, dtype=torch.float32)
                    cand_sel = (x_sel.to(dtype=torch.float32) + sign_sel * (3.0 * gamma_f) * std_sel).to(
                        dtype=x_mod.dtype
                    )
                out_tensors[corrupted_modality_idx][active_t] = cand_sel

        orig_mod = modality_tensors[corrupted_modality_idx]
        new_mod = out_tensors[corrupted_modality_idx]
        for idx in corrupt_indices:
            if is_conflicted[idx] == 1:
                o_row = orig_mod[idx]
                n_row = new_mod[idx]
                if orig_mod.is_floating_point() or orig_mod.is_complex():
                    same_nan = torch.isnan(o_row) & torch.isnan(n_row)
                    changed = bool(torch.any((n_row != o_row) & ~same_nan).item())
                else:
                    changed = bool(torch.any(n_row != o_row).item())
                if not changed:
                    is_conflicted[idx] = 0

        return out_tensors, is_conflicted

    @staticmethod
    def _compute_detection_auroc(
        is_conflicted: np.ndarray,
        eligible_mask: np.ndarray,
        score: np.ndarray,
    ) -> float:
        """Compute unsupervised conflict-detection AUROC on eligible multi-modal subjects."""
        conf_elig = (is_conflicted == 1) & eligible_mask
        clean_elig = (is_conflicted == 0) & eligible_mask
        if (
            np.any(conf_elig)
            and np.any(clean_elig)
            and bool(np.all(np.isfinite(score[eligible_mask])))
        ):
            return float(roc_auc_score(is_conflicted[eligible_mask], score[eligible_mask]))
        return float("nan")

    @staticmethod
    def _compute_referral_curve(
        probs: np.ndarray,
        targets: np.ndarray,
        score: np.ndarray,
        referral_rates: List[float],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate selective performance when referring top-tau% highest-score samples."""
        order = np.argsort(score, kind="mergesort")
        n_total = len(targets)
        referral_curve: Dict[str, Dict[str, float]] = {}
        seen_ref_keys: Dict[str, object] = {}
        for tau in referral_rates:
            tau_f = float(tau)
            if math.isfinite(tau_f):
                clean_tau = 0.0 if tau_f == 0.0 else tau_f
                pct_dec = (Decimal(str(clean_tau)) * Decimal(100)).normalize()
                if pct_dec == Decimal(0):
                    pct_str = "0"
                elif Decimal("1e-9") <= abs(pct_dec) < Decimal("1e12"):
                    pct_str = format(pct_dec, "f")
                    if "." in pct_str:
                        pct_str = pct_str.rstrip("0").rstrip(".") or "0"
                else:
                    pct_str = str(pct_dec)
            else:
                pct_str = repr(tau_f)
            tau_tok: object = "nan" if math.isnan(tau_f) else tau_f
            ref_key = f"refer_{pct_str}pct"
            if ref_key in seen_ref_keys and seen_ref_keys[ref_key] != tau_tok:
                suffix = 2
                alt_key = f"{ref_key}_{suffix}"
                while alt_key in seen_ref_keys and seen_ref_keys[alt_key] != tau_tok:
                    suffix += 1
                    alt_key = f"{ref_key}_{suffix}"
                ref_key = alt_key
            seen_ref_keys[ref_key] = tau_tok

            if n_total == 0 or not math.isfinite(tau_f) or tau_f >= 1.0:
                keep_n = 0
            else:
                keep_n = min(n_total, max(0, int(round(n_total * (1.0 - tau_f)))))
            realized_coverage = float(keep_n / n_total) if n_total > 0 else 0.0
            req_cov = float(max(0.0, min(1.0, 1.0 - tau_f))) if math.isfinite(tau_f) else float("nan")
            if keep_n == 0:
                referral_curve[ref_key] = {
                    "coverage": realized_coverage,
                    "requested_coverage": req_cov,
                    "n_retained": 0,
                    "auroc": float("nan"),
                    "acc": float("nan"),
                    "brier": float("nan"),
                }
                continue
            keep_idx = order[:keep_n]
            sub_m = compute_binary_metrics(probs[keep_idx], targets[keep_idx])
            referral_curve[ref_key] = {
                "coverage": realized_coverage,
                "requested_coverage": req_cov,
                "n_retained": int(keep_n),
                "auroc": sub_m["auroc"],
                "acc": sub_m["acc"],
                "brier": sub_m["brier"],
            }
        return referral_curve

    @torch.no_grad()
    def evaluate_conflict_sweep(
        self,
        predict_fn: Callable[[List[torch.Tensor], torch.Tensor], Dict[str, torch.Tensor]],
        modality_tensors: List[torch.Tensor],
        natural_masks: torch.Tensor,
        targets: np.ndarray,
        corrupted_modality_idx: int = 1,
        severities: Optional[List[float]] = None,
        referral_rates: Optional[List[float]] = None,
        fold_ids: Optional[np.ndarray] = None,
        conflict_fraction: float = 0.30,
        seed: int = 42,
    ) -> Dict[str, object]:
        """
        Sweeps conflict severity gamma from 0.0 to 1.0 on the exact same recipient cohort
        and evaluates:
          1. Monotonic empirical trend of internal conflict C_O and hesitancy pi_fused.
          2. Separated and composite Unsupervised Conflict Detection AUROCs:
             pure C_O (`conflict_detection_auroc_conflict_only`),
             pure pi_fused (`conflict_detection_auroc_hes_only`), and
             equal-weight composite 0.5*(C_O + pi_fused) (`conflict_detection_auroc`).
          3. Safety-critical Selective Clinical Referral curves under each triage score.
        """
        if severities is None:
            severities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        if referral_rates is None:
            referral_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

        mask_np = (natural_masks.detach().cpu().numpy() > 0.5).astype(int)
        eligible_mask = (mask_np[:, corrupted_modality_idx] == 1) & (
            (mask_np.sum(axis=1) - mask_np[:, corrupted_modality_idx]) >= 1
        )

        sweep_records = []
        for gamma in severities:
            # Keep fixed `seed` across all severities so the exact same recipients & donors are compared (M8)
            corrupted_inputs, is_conflicted = self.inject_conflict(
                modality_tensors=modality_tensors,
                targets=targets,
                corrupted_modality_idx=corrupted_modality_idx,
                conflict_fraction=conflict_fraction if gamma > 0 else 0.0,
                severity_gamma=gamma,
                mode="opposite_swap",
                natural_masks=natural_masks,
                fold_ids=fold_ids,
                seed=seed,
            )
            out = predict_fn(corrupted_inputs, natural_masks)
            probs = out["prob"].detach().cpu().numpy().ravel()
            hes = out.get("hesitancy", 1.0 - 2.0 * (out["prob"] - 0.5).abs()).detach().cpu().numpy().ravel()
            conflict_score = out.get("conflict", torch.zeros_like(out["prob"])).detach().cpu().numpy().ravel()

            combined_anomaly_score = 0.5 * hes + 0.5 * conflict_score
            conf_elig_mask = (is_conflicted == 1) & eligible_mask
            clean_elig_mask = (is_conflicted == 0) & eligible_mask
            hes_conflicted = (
                float(np.mean(hes[conf_elig_mask])) if np.any(conf_elig_mask) else float("nan")
            )
            hes_clean = (
                float(np.mean(hes[clean_elig_mask])) if np.any(clean_elig_mask) else float("nan")
            )
            conf_conflicted = (
                float(np.mean(conflict_score[conf_elig_mask])) if np.any(conf_elig_mask) else float("nan")
            )
            conf_clean = (
                float(np.mean(conflict_score[clean_elig_mask])) if np.any(clean_elig_mask) else float("nan")
            )

            detect_auroc = self._compute_detection_auroc(
                is_conflicted, eligible_mask, combined_anomaly_score
            )
            detect_auroc_conflict_only = self._compute_detection_auroc(
                is_conflicted, eligible_mask, conflict_score
            )
            detect_auroc_hes_only = self._compute_detection_auroc(
                is_conflicted, eligible_mask, hes
            )

            base_metrics = compute_binary_metrics(probs, targets, uncertainty=combined_anomaly_score)
            conflict_only_metrics = compute_binary_metrics(probs, targets, uncertainty=conflict_score)
            hes_only_metrics = compute_binary_metrics(probs, targets, uncertainty=hes)

            referral_curve = self._compute_referral_curve(
                probs, targets, combined_anomaly_score, referral_rates
            )
            referral_curve_conflict_only = self._compute_referral_curve(
                probs, targets, conflict_score, referral_rates
            )
            referral_curve_hes_only = self._compute_referral_curve(
                probs, targets, hes, referral_rates
            )

            sweep_records.append({
                "severity_gamma": float(gamma),
                "conflict_fraction": float(conflict_fraction if gamma > 0 else 0.0),
                "n_eligible_recipients": int(np.sum(eligible_mask)),
                "n_conflicted": int(np.sum(is_conflicted)),
                "overall_auroc": base_metrics["auroc"],
                "overall_acc": base_metrics["acc"],
                "overall_brier": base_metrics["brier"],
                "aurc": base_metrics["aurc"],
                "aurc_conflict_only": conflict_only_metrics["aurc"],
                "aurc_hes_only": hes_only_metrics["aurc"],
                "mean_hesitancy": float(np.mean(hes)) if len(hes) > 0 else float("nan"),
                "hesitancy_clean": hes_clean,
                "hesitancy_conflicted": hes_conflicted,
                "conflict_clean": conf_clean,
                "conflict_conflicted": conf_conflicted,
                "conflict_detection_auroc": detect_auroc,
                "conflict_detection_auroc_conflict_only": detect_auroc_conflict_only,
                "conflict_detection_auroc_hes_only": detect_auroc_hes_only,
                "referral_curve": referral_curve,
                "referral_curve_conflict_only": referral_curve_conflict_only,
                "referral_curve_hes_only": referral_curve_hes_only,
            })

        return {
            "corrupted_modality": self.modality_names[corrupted_modality_idx],
            "sweep_records": sweep_records,
        }
