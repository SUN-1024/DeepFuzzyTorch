"""
Clinical, Calibration, Selective Prediction & Statistical Evaluation Metrics for DeepFuzzyTorch.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) with equal-width probability bins."""
    probs = np.clip(
        np.nan_to_num(np.asarray(probs, dtype=np.float64).ravel(), nan=0.5, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    targets = np.nan_to_num(
        np.asarray(targets, dtype=np.float64).ravel(), nan=0.0, posinf=1.0, neginf=0.0
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(probs)
    if n_total == 0:
        return float("nan")
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)
        if np.any(mask):
            acc_bin = np.mean(targets[mask])
            conf_bin = np.mean(probs[mask])
            ece += (np.sum(mask) / n_total) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_aurc(
    probs: np.ndarray,
    targets: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Area Under the Risk-Coverage Curve (AURC) and Accuracy-Rejection Curve (ARC)
    following the discrete Riemann-sum selective risk convention (Geifman & El-Yaniv, 2017):
        AURC = (1 / N) * sum_{k=1}^N E[R_hat(k / N)],
    using the cancellation-free default margin uncertainty `2 * min(p, 1 - p)` when
    `uncertainty is None`, 0-1 errors evaluated at `threshold`, and averaging empirical 0-1
    errors within each tied-uncertainty block so that AURC is invariant to sample/bootstrap
    permutation order.
    """
    probs = np.clip(
        np.nan_to_num(np.asarray(probs, dtype=np.float64).ravel(), nan=0.5, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    targets = np.nan_to_num(
        np.asarray(targets, dtype=np.float64).ravel(), nan=0.0, posinf=1.0, neginf=0.0
    )
    thr_raw = float(threshold)
    thr = 0.5 if np.isnan(thr_raw) else thr_raw
    preds = (probs >= thr).astype(np.float64)
    errors = (preds != targets).astype(np.float64)

    if uncertainty is None:
        uncertainty = 2.0 * np.minimum(probs, 1.0 - probs)
    else:
        uncertainty = np.nan_to_num(
            np.asarray(uncertainty, dtype=np.float64).ravel(), nan=1.0, posinf=1.0, neginf=0.0
        )

    n = len(errors)
    if n == 0:
        return float("nan"), np.array([]), np.array([]), np.array([])

    order = np.argsort(uncertainty, kind="mergesort")
    u_sorted = uncertainty[order]
    e_sorted = errors[order]
    expected_errors = np.empty_like(e_sorted)
    _, idx_start, counts = np.unique(u_sorted, return_index=True, return_counts=True)
    for st, cnt in zip(idx_start, counts):
        expected_errors[st : st + cnt] = np.mean(e_sorted[st : st + cnt])

    coverages = np.arange(1, n + 1, dtype=np.float64) / n
    cum_risks = np.cumsum(expected_errors) / np.arange(1, n + 1, dtype=np.float64)
    cum_accs = 1.0 - cum_risks
    aurc = float(np.mean(cum_risks))
    return aurc, coverages, cum_risks, cum_accs


def _sanitize_targets(targets: np.ndarray) -> np.ndarray:
    """Flattens targets to finite float64."""
    return np.nan_to_num(
        np.asarray(targets, dtype=np.float64).ravel(), nan=0.0, posinf=1.0, neginf=0.0
    )


def _sanitize_probs_and_scores(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns consistent ranking-preserving finite scores and [0, 1]-clamped probabilities.
    Neither output collapses tail probabilities into [1e-6, 1 - 1e-6].
    """
    raw = np.asarray(probs, dtype=np.float64).ravel()
    max_f64 = np.finfo(np.float64).max
    scores = np.nan_to_num(raw, nan=0.5, posinf=max_f64, neginf=-max_f64)
    probs_01 = np.clip(
        np.nan_to_num(raw, nan=0.5, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    return scores, probs_01


def _sanitize_uncertainty(uncertainty: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if uncertainty is None:
        return None
    return np.nan_to_num(
        np.asarray(uncertainty, dtype=np.float64).ravel(), nan=1.0, posinf=1.0, neginf=0.0
    )


def _eval_single_binary_metric(
    scores: np.ndarray,
    probs_01: np.ndarray,
    y_int: np.ndarray,
    metric_name: str,
    threshold: float = 0.5,
    n_bins: int = 10,
    uncertainty: Optional[np.ndarray] = None,
) -> float:
    """Evaluates a single scalar metric using unified score and [0, 1] probability arrays."""
    if len(y_int) == 0:
        if metric_name in ("auroc", "auprc", "acc", "bal_acc", "f1", "sensitivity", "specificity", "brier", "ece", "aurc"):
            return float("nan")
        raise ValueError(f"Unsupported metric_name: {metric_name!r}")
    has_both = len(np.unique(y_int)) > 1
    if metric_name == "auroc":
        return float(roc_auc_score(y_int, scores)) if has_both else float("nan")
    if metric_name == "auprc":
        return float(average_precision_score(y_int, scores)) if has_both else float("nan")
    preds = (probs_01 >= threshold).astype(int)
    if metric_name == "acc":
        return float(accuracy_score(y_int, preds))
    if metric_name == "bal_acc":
        acc = float(accuracy_score(y_int, preds))
        return float(balanced_accuracy_score(y_int, preds)) if has_both else acc
    if metric_name == "f1":
        return float(f1_score(y_int, preds, zero_division=0))
    if metric_name == "sensitivity":
        pos_mask = y_int == 1
        return float(np.mean(preds[pos_mask] == 1)) if np.any(pos_mask) else float("nan")
    if metric_name == "specificity":
        neg_mask = y_int == 0
        return float(np.mean(preds[neg_mask] == 0)) if np.any(neg_mask) else float("nan")
    if metric_name == "brier":
        return float(brier_score_loss(y_int, probs_01))
    if metric_name == "ece":
        return compute_ece(probs_01, y_int, n_bins=n_bins)
    if metric_name == "aurc":
        aurc_val, _, _, _ = compute_aurc(
            probs_01, y_int, uncertainty=uncertainty, threshold=threshold
        )
        return float(aurc_val)
    raise ValueError(f"Unsupported metric_name: {metric_name!r}")


def compute_binary_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes comprehensive discriminative, calibration, and selective-prediction metrics."""
    scores, probs_01 = _sanitize_probs_and_scores(probs)
    y_int = _sanitize_targets(targets).astype(int)
    unc_arr = _sanitize_uncertainty(uncertainty)
    thr_raw = float(threshold)
    thr = 0.5 if np.isnan(thr_raw) else thr_raw
    if len(y_int) == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "acc": float("nan"),
            "bal_acc": float("nan"),
            "f1": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "aurc": float("nan"),
        }
    preds = (probs_01 >= thr).astype(int)
    has_both = len(np.unique(y_int)) > 1

    auroc = float(roc_auc_score(y_int, scores)) if has_both else float("nan")
    auprc = float(average_precision_score(y_int, scores)) if has_both else float("nan")
    acc = float(accuracy_score(y_int, preds))
    bal_acc = float(balanced_accuracy_score(y_int, preds)) if has_both else acc
    f1 = float(f1_score(y_int, preds, zero_division=0))
    brier = float(brier_score_loss(y_int, probs_01))
    ece = compute_ece(probs_01, y_int, n_bins=10)
    aurc, _, _, _ = compute_aurc(probs_01, y_int, uncertainty=unc_arr, threshold=thr)

    pos_mask = y_int == 1
    neg_mask = y_int == 0
    sens = float(np.mean(preds[pos_mask] == 1)) if np.any(pos_mask) else float("nan")
    spec = float(np.mean(preds[neg_mask] == 0)) if np.any(neg_mask) else float("nan")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "acc": acc,
        "bal_acc": bal_acc,
        "f1": f1,
        "sensitivity": sens,
        "specificity": spec,
        "brier": brier,
        "ece": ece,
        "aurc": aurc,
    }


def bootstrap_confidence_interval(
    probs: np.ndarray,
    targets: np.ndarray,
    metric_name: str = "auroc",
    n_bootstraps: int = 1000,
    seed: int = 42,
    uncertainty: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Computes point estimate and 95% class-stratified non-parametric bootstrap confidence interval."""
    scores, probs_01 = _sanitize_probs_and_scores(probs)
    y_int = _sanitize_targets(targets).astype(int)
    unc_arr = _sanitize_uncertainty(uncertainty)
    thr_raw = float(threshold)
    thr = 0.5 if np.isnan(thr_raw) else thr_raw
    rng = np.random.default_rng(seed)

    point = _eval_single_binary_metric(
        scores, probs_01, y_int, metric_name, threshold=thr, uncertainty=unc_arr
    )
    if len(y_int) == 0 or np.isnan(point):
        return float("nan"), float("nan"), float("nan")
    if n_bootstraps <= 0:
        return float(point), float("nan"), float("nan")

    neg_idx = np.where(y_int == 0)[0]
    pos_idx = np.where(y_int == 1)[0]
    has_both = len(neg_idx) > 0 and len(pos_idx) > 0
    all_idx = neg_idx if len(neg_idx) > 0 else (pos_idx if len(pos_idx) > 0 else np.arange(len(y_int)))

    vals = []
    for _ in range(n_bootstraps):
        if has_both:
            b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            idx = np.concatenate([b_neg, b_pos])
        else:
            idx = rng.choice(all_idx, size=len(all_idx), replace=True)
        unc_b = unc_arr[idx] if unc_arr is not None else None
        val = _eval_single_binary_metric(
            scores[idx], probs_01[idx], y_int[idx], metric_name, threshold=thr, uncertainty=unc_b
        )
        if not np.isnan(val):
            vals.append(float(val))

    if not vals:
        return float(point), float("nan"), float("nan")
    low, high = np.percentile(vals, [2.5, 97.5])
    return float(point), float(low), float(high)


def paired_bootstrap_pvalue(
    probs_ours: np.ndarray,
    probs_base: np.ndarray,
    targets: np.ndarray,
    metric_name: str = "auroc",
    higher_is_better: bool = True,
    n_bootstraps: int = 1000,
    seed: int = 42,
    uncertainty_ours: Optional[np.ndarray] = None,
    uncertainty_base: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> float:
    """
    Two-sided class-stratified paired bootstrap hypothesis test comparing `probs_ours`
    against `probs_base`. Uses exact array equality so small ranking separations (e.g.
    0.5 +/- 1e-13) are never collapsed to p = 1.0; returns NaN if the metric is undefined
    on the cohort and p = 1.0 for identical predictions or zero observed differences.
    """
    scores_ours, probs_ours_01 = _sanitize_probs_and_scores(probs_ours)
    scores_base, probs_base_01 = _sanitize_probs_and_scores(probs_base)
    y_int = _sanitize_targets(targets).astype(int)
    unc_ours_c = _sanitize_uncertainty(uncertainty_ours)
    unc_base_c = _sanitize_uncertainty(uncertainty_base)
    thr_raw = float(threshold)
    thr = 0.5 if np.isnan(thr_raw) else thr_raw

    if len(y_int) == 0:
        return float("nan")

    m0_ours = _eval_single_binary_metric(
        scores_ours, probs_ours_01, y_int, metric_name, threshold=thr, uncertainty=unc_ours_c
    )
    m0_base = _eval_single_binary_metric(
        scores_base, probs_base_01, y_int, metric_name, threshold=thr, uncertainty=unc_base_c
    )
    obs_diff = (m0_ours - m0_base) if higher_is_better else (m0_base - m0_ours)
    if np.isnan(obs_diff):
        return float("nan")

    same_unc = (
        (unc_ours_c is None and unc_base_c is None)
        or (
            unc_ours_c is not None
            and unc_base_c is not None
            and np.array_equal(unc_ours_c, unc_base_c)
        )
    )
    if metric_name in ("auroc", "auprc"):
        if np.array_equal(scores_ours, scores_base):
            return 1.0
    elif metric_name == "aurc":
        if np.array_equal(probs_ours_01, probs_base_01) and same_unc:
            return 1.0
    elif np.array_equal(probs_ours_01, probs_base_01):
        return 1.0

    tie_tol = 1e-15
    if abs(obs_diff) <= tie_tol:
        return 1.0
    if n_bootstraps <= 0:
        return float("nan")

    neg_idx = np.where(y_int == 0)[0]
    pos_idx = np.where(y_int == 1)[0]
    has_both = len(neg_idx) > 0 and len(pos_idx) > 0
    all_idx = neg_idx if len(neg_idx) > 0 else (pos_idx if len(pos_idx) > 0 else np.arange(len(y_int)))

    rng = np.random.default_rng(seed)

    diffs = []
    for _ in range(n_bootstraps):
        if has_both:
            b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            idx = np.concatenate([b_neg, b_pos])
        else:
            idx = rng.choice(all_idx, size=len(all_idx), replace=True)
        u_o = unc_ours_c[idx] if unc_ours_c is not None else None
        u_b = unc_base_c[idx] if unc_base_c is not None else None
        m_ours = _eval_single_binary_metric(
            scores_ours[idx], probs_ours_01[idx], y_int[idx], metric_name, threshold=thr, uncertainty=u_o
        )
        m_base = _eval_single_binary_metric(
            scores_base[idx], probs_base_01[idx], y_int[idx], metric_name, threshold=thr, uncertainty=u_b
        )
        diff = (m_ours - m_base) if higher_is_better else (m_base - m_ours)
        if not np.isnan(diff):
            diffs.append(float(diff))

    if not diffs:
        return float("nan")
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    n_valid = float(len(diffs_arr))
    # Split exact ties evenly between tails so identical distributions yield p = 1.0
    n_neg = float(np.sum(diffs_arr < -tie_tol)) + 0.5 * float(np.sum(np.abs(diffs_arr) <= tie_tol))
    n_pos = float(np.sum(diffs_arr > tie_tol)) + 0.5 * float(np.sum(np.abs(diffs_arr) <= tie_tol))
    p_tail = min(n_neg, n_pos) / n_valid
    return float(min(1.0, max(1.0 / (n_valid + 1.0), 2.0 * p_tail)))


def compute_logrank_test(
    times_a: np.ndarray,
    events_a: np.ndarray,
    times_b: np.ndarray,
    events_b: np.ndarray,
) -> Tuple[float, float]:
    """
    Computes the Mantel-Haenszel two-sample log-rank statistic over risk sets at all
    distinct observed event times t_{(1)} < ... < t_{(K)} and its asymptotic
    chi-squared (df = 1) p-value.
    """
    t_a = np.asarray(times_a, dtype=np.float64).ravel()
    e_a = np.asarray(events_a, dtype=np.int32).ravel()
    t_b = np.asarray(times_b, dtype=np.float64).ravel()
    e_b = np.asarray(events_b, dtype=np.int32).ravel()

    all_t = np.concatenate([t_a, t_b])
    all_e = np.concatenate([e_a, e_b])
    distinct_event_times = np.sort(np.unique(all_t[all_e == 1]))
    if len(distinct_event_times) == 0:
        return 0.0, 1.0

    obs_minus_exp = 0.0
    var_sum = 0.0
    for t_k in distinct_event_times:
        n_a = float(np.sum(t_a >= t_k))
        n_b = float(np.sum(t_b >= t_k))
        n_total = n_a + n_b
        if n_total <= 1.0:
            continue
        d_a = float(np.sum((t_a == t_k) & (e_a == 1)))
        d_b = float(np.sum((t_b == t_k) & (e_b == 1)))
        d_total = d_a + d_b
        exp_a = d_total * (n_a / n_total)
        var_a = (n_a * n_b * d_total * (n_total - d_total)) / (n_total * n_total * (n_total - 1.0))
        obs_minus_exp += d_a - exp_a
        var_sum += var_a

    if var_sum <= 1e-12:
        return 0.0, 1.0
    chisq = float((obs_minus_exp * obs_minus_exp) / var_sum)
    p_val = float(chi2.sf(chisq, df=1))
    return chisq, p_val


def compute_multigroup_logrank_test(
    times: np.ndarray,
    events: np.ndarray,
    groups: np.ndarray,
) -> Tuple[float, float]:
    """
    Computes the K-sample omnibus Mantel-Haenszel log-rank statistic across K >= 2 risk
    strata and its asymptotic chi-squared p-value using the numerical rank (eff_df <= K - 1)
    and Moore-Penrose pseudoinverse of the risk-set covariance matrix.
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    e = np.asarray(events, dtype=np.int32).ravel()
    g = np.asarray(groups).ravel()
    unique_groups = np.unique(g)
    k_groups = len(unique_groups)
    if k_groups < 2:
        return 0.0, 1.0

    distinct_event_times = np.sort(np.unique(t[e == 1]))
    if len(distinct_event_times) == 0:
        return 0.0, 1.0

    obs_minus_exp = np.zeros(k_groups - 1, dtype=np.float64)
    cov_mat = np.zeros((k_groups - 1, k_groups - 1), dtype=np.float64)

    for t_k in distinct_event_times:
        at_risk = t >= t_k
        n_total = float(np.sum(at_risk))
        if n_total <= 1.0:
            continue
        is_event = (t == t_k) & (e == 1)
        d_total = float(np.sum(is_event))
        if d_total <= 0.0:
            continue

        n_vec = np.array([float(np.sum(at_risk & (g == grp))) for grp in unique_groups[: k_groups - 1]])
        d_vec = np.array([float(np.sum(is_event & (g == grp))) for grp in unique_groups[: k_groups - 1]])
        p_vec = n_vec / n_total
        exp_vec = d_total * p_vec
        obs_minus_exp += d_vec - exp_vec

        factor = d_total * (n_total - d_total) / (n_total - 1.0)
        cov_k = factor * (np.diag(p_vec) - np.outer(p_vec, p_vec))
        cov_mat += cov_k

    eff_df = int(np.linalg.matrix_rank(cov_mat, tol=1e-10, hermitian=True))
    if eff_df <= 0:
        return 0.0, 1.0
    try:
        pinv_cov = np.linalg.pinv(cov_mat, rcond=1e-10, hermitian=True)
        chisq = float(obs_minus_exp @ pinv_cov @ obs_minus_exp)
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    p_val = float(chi2.sf(max(0.0, chisq), df=eff_df))
    return chisq, p_val


def compute_decision_curve_analysis(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Computes Vickers & Elkin Decision Curve Analysis (DCA) Net Benefit curves:
        NB(p_t) = TP / N - (FP / N) * (p_t / (1 - p_t))
    """
    probs = np.nan_to_num(
        np.asarray(probs, dtype=np.float64).ravel(), nan=0.5, posinf=1.0, neginf=0.0
    )
    targets = np.nan_to_num(
        np.asarray(targets, dtype=np.float64).ravel(), nan=0.0, posinf=1.0, neginf=0.0
    )
    thresholds_arr = (
        np.linspace(0.05, 0.80, 76)
        if thresholds is None
        else np.asarray(thresholds, dtype=np.float64).ravel()
    )
    if len(targets) == 0 or len(probs) == 0:
        nan_arr = np.full(thresholds_arr.shape, np.nan, dtype=np.float64)
        return {
            "thresholds": thresholds_arr,
            "net_benefit_model": nan_arr,
            "net_benefit_all": nan_arr.copy(),
            "net_benefit_none": nan_arr.copy(),
        }

    n = float(len(targets))
    prevalence = float(np.mean(targets))
    nb_model = []
    nb_all = []
    nb_none = np.zeros_like(thresholds_arr)

    for pt in thresholds_arr:
        preds = (probs >= pt).astype(int)
        tp = float(np.sum((preds == 1) & (targets == 1)))
        fp = float(np.sum((preds == 1) & (targets == 0)))
        exchange_rate = pt / max(1e-6, 1.0 - pt)
        nb_m = (tp / n) - (fp / n) * exchange_rate
        nb_a = prevalence - (1.0 - prevalence) * exchange_rate
        nb_model.append(nb_m)
        nb_all.append(nb_a)

    return {
        "thresholds": thresholds_arr,
        "net_benefit_model": np.asarray(nb_model),
        "net_benefit_all": np.asarray(nb_all),
        "net_benefit_none": nb_none,
    }
