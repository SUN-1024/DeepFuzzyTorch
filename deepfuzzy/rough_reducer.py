"""
GPU-Accelerated Pointwise T-Norm Soft-Lukasiewicz Fuzzy-Rough Feature Selector & Reduct Analyzer.

Implements granular attribute subset selection grounded in Dubois-Prade Fuzzy-Rough Set Theory:
  - Given unlabeled training features X in R^{N x D}, standardizes each attribute k strictly
    unsupervised via robust median and 95%-span dispersion s_k = 4 * MAD_k (with SD fallback
    for sparse/binary columns), yielding per-attribute Laplace similarity:
        R_k(x_i, x_j) = exp( - |x_{i,k} - x_{j,k}| / (sigma * s_k) ) in (0, 1].
  - Aggregates multi-attribute indiscernibility via a genuine pointwise T-norm over k in B
    (default Godel T-norm R_B(x_i, x_j) = min_{k in B} R_k(x_i, x_j); also supports product
    and Lukasiewicz T-norms), guaranteeing exact granular refinement:
        B subseteq B'  ===>  R_{B'}(x_i, x_j) <= R_B(x_i, x_j)  pointwise for all (i, j).
  - Evaluates the Lukasiewicz implicator I(R_B(x_i, x_j), y_same_{i,j}) = min(1, 1 - R_B + y_same)
    and its monotone shifted LogSumExp soft-infimum lower approximation:
        R_low_B(D)(x_i) = -tau * ln( (1 / (N - 1)) * sum_{j != i} exp( -I_{i,j} / tau ) ),
    which has strictly positive partial derivatives everywhere and therefore guarantees
    monotonically non-decreasing positive-region dependency gamma_B(D) (0 violations) and
    monotonically non-increasing opposing-class boundary BND_B(D) = 1 - gamma_B(D).
  - Separates genuine discriminative signal from high-dimensional geometric distance concentration
    via a label-permutation null baseline gamma_B^{null}(D), ranking forward-greedy candidates
    by chance-adjusted excess dependency while preserving exact Dubois-Prade backward minimality
    pruning down to size >= 1.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class FuzzyRoughFeatureReducer:
    """
    Pointwise T-Norm Soft-Lukasiewicz Fuzzy-Rough Granular Feature Selector & Reduct Analyzer.
    """

    def __init__(
        self,
        max_features: int = 25,
        sigma: float = 0.85,
        soft_temp: float = 0.08,
        max_samples: int = 4096,
        subsample_size: Optional[int] = None,
        tnorm: str = "godel",
        n_null_permutations: int = 12,
        n_stability_bootstraps: Optional[int] = None,
        redundancy_tol: float = 1e-5,
        seed: int = 42,
    ):
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError(f"sigma must be finite and > 0, got {sigma}")
        if soft_temp <= 0.0 or not np.isfinite(soft_temp):
            raise ValueError(f"soft_temp must be finite and > 0, got {soft_temp}")
        tnorm_norm = str(tnorm).strip().lower()
        if tnorm_norm not in ("godel", "product", "lukasiewicz"):
            raise ValueError(
                f"tnorm must be one of ('godel', 'product', 'lukasiewicz'), got {tnorm!r}"
            )

        self.max_features = int(max_features)
        self.sigma = float(sigma)
        self.soft_temp = float(soft_temp)
        effective_max = int(subsample_size) if subsample_size is not None else int(max_samples)
        self.max_samples = max(2, effective_max)
        self.subsample_size = self.max_samples
        self.tnorm = tnorm_norm
        self.n_null_permutations = max(0, int(n_null_permutations))
        self.n_stability_bootstraps = (
            None if n_stability_bootstraps is None else max(1, int(n_stability_bootstraps))
        )
        self.redundancy_tol = float(redundancy_tol)
        self.seed = int(seed)

        self.selected_indices_: List[int] = []
        self.selected_names_: List[str] = []
        self.minimal_reduct_indices_: List[int] = []
        self.minimal_reduct_names_: List[str] = []
        self.dependency_curve_: List[float] = []
        self.null_dependency_curve_: List[float] = []
        self.excess_dependency_curve_: List[float] = []
        self.normalized_excess_curve_: List[float] = []
        self.boundary_curve_: List[float] = []
        self.feature_scores_: Dict[str, float] = {}
        self.univariate_scores_: np.ndarray = np.array([], dtype=np.float32)
        self.univariate_excess_scores_: np.ndarray = np.array([], dtype=np.float32)
        self.feature_medians_: np.ndarray = np.array([], dtype=np.float32)
        self.feature_scales_: np.ndarray = np.array([], dtype=np.float32)

    @staticmethod
    def _robust_standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unsupervised column-wise standardization using median and 4 * MAD (95% Gaussian span),
        falling back to 4 * SD for sparse/binary attributes where MAD == 0, and 4.0 for
        constant or all-missing attributes. Emits zero NumPy warnings on all-NaN/Inf slices.
        """
        X_f64 = np.asarray(X, dtype=np.float64)
        N, D = X_f64.shape
        finite_mask = np.isfinite(X_f64)
        medians = np.zeros((1, D), dtype=np.float64)
        scales = np.full((1, D), 4.0, dtype=np.float64)

        for j in range(D):
            valid = X_f64[finite_mask[:, j], j]
            if valid.size == 0:
                continue
            med_j = float(np.median(valid))
            medians[0, j] = med_j
            mad_j = 1.4826 * float(np.median(np.abs(valid - med_j)))
            if mad_j > 1e-6:
                scales[0, j] = 4.0 * mad_j
            else:
                std_j = float(np.std(valid))
                if std_j > 1e-6:
                    scales[0, j] = 4.0 * std_j
                else:
                    scales[0, j] = 4.0

        X_clean = np.where(finite_mask, X_f64, medians)
        X_norm = (X_clean - medians) / scales
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return X_norm, medians.astype(np.float32), scales.astype(np.float32)

    def _apply_standardization(self, X: np.ndarray) -> np.ndarray:
        X_f64 = np.asarray(X, dtype=np.float64)
        if self.feature_medians_.size == X_f64.shape[1] and self.feature_scales_.size == X_f64.shape[1]:
            med = self.feature_medians_.astype(np.float64)
            sc = self.feature_scales_.astype(np.float64)
            finite_mask = np.isfinite(X_f64)
            X_clean = np.where(finite_mask, X_f64, med)
            X_norm = (X_clean - med) / np.maximum(sc, 1e-8)
            return np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_norm, _, _ = self._robust_standardize_fit(X_f64)
        return X_norm

    def _feature_similarity(self, col_vec: torch.Tensor) -> torch.Tensor:
        """Single-attribute Laplace similarity R_k(x_i, x_j) = exp(-|z_{i,k} - z_{j,k}| / sigma)."""
        dist = (col_vec - col_vec.T).abs()
        return torch.exp(-dist / self.sigma).clamp(0.0, 1.0)

    def _combine_tnorm(self, R_cur: torch.Tensor, R_cand: torch.Tensor) -> torch.Tensor:
        """Pointwise T-norm update guaranteeing R_{B union {k}}(x_i, x_j) <= R_B(x_i, x_j)."""
        if self.tnorm == "godel":
            return torch.minimum(R_cur, R_cand)
        if self.tnorm == "product":
            return (R_cur * R_cand).clamp(0.0, 1.0)
        # Lukasiewicz T-norm: max(0, R_cur + R_cand - 1)
        return (R_cur + R_cand - 1.0).clamp(0.0, 1.0)

    @torch.no_grad()
    def compute_indiscernibility(
        self,
        X: np.ndarray,
        subset: List[int],
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """
        Computes the exact pointwise T-norm fuzzy indiscernibility matrix R_B(x_i, x_j) in [0, 1]^{N x N}
        for attribute subset B = `subset` (returns all-ones matrix when B is empty).
        When `device` is Apple MPS and `dtype` is float64 (unsupported on Metal), computation runs
        in float64 on CPU and transfers the validated result back to MPS as float32.
        """
        dev_obj = torch.device(device)
        mps_f64 = (dev_obj.type == "mps" and dtype == torch.float64)
        compute_dev = torch.device("cpu") if mps_f64 else dev_obj
        out_dtype = torch.float32 if mps_f64 else dtype

        n_samples = X.shape[0]
        if len(subset) == 0:
            return torch.ones((n_samples, n_samples), device=dev_obj, dtype=out_dtype)

        X_norm = self._apply_standardization(X)
        Xt = torch.tensor(X_norm, device=compute_dev, dtype=dtype)
        R_mat = torch.ones((n_samples, n_samples), device=compute_dev, dtype=dtype)
        for idx in subset:
            col = Xt[:, int(idx) : int(idx) + 1]
            R_k = torch.exp(-(col - col.T).abs() / self.sigma).clamp(0.0, 1.0)
            R_mat = self._combine_tnorm(R_mat, R_k)
        if mps_f64:
            return R_mat.to(dtype=torch.float32).to(device=dev_obj)
        return R_mat

    @staticmethod
    def _stratified_subsample_indices(
        y: np.ndarray, target_n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draws a class-stratified subsample without replacement strictly inside the given fold."""
        n_total = len(y)
        if target_n >= n_total:
            return np.arange(n_total, dtype=np.int64)
        y_bin = (y >= 0.5).astype(np.int32)
        idx0 = np.where(y_bin == 0)[0]
        idx1 = np.where(y_bin == 1)[0]
        if len(idx0) == 0 or len(idx1) == 0:
            return np.sort(rng.choice(n_total, size=target_n, replace=False))
        n0 = int(round(target_n * (len(idx0) / float(n_total))))
        n0 = max(1, min(len(idx0), min(target_n - 1, n0)))
        n1 = target_n - n0
        if n1 > len(idx1):
            n1 = len(idx1)
            n0 = min(len(idx0), target_n - n1)
        s0 = rng.choice(idx0, size=n0, replace=False)
        s1 = rng.choice(idx1, size=n1, replace=False)
        return np.sort(np.concatenate([s0, s1]))

    @staticmethod
    def _build_view_context(
        Xt_v: torch.Tensor,
        y_v: np.ndarray,
        soft_temp: float,
        n_null_perms: int,
        rng: np.random.Generator,
    ) -> Dict[str, object]:
        """Precomputes label matrices and permutation null targets for one evaluation view."""
        device = Xt_v.device
        n_v = Xt_v.shape[0]
        is_binary = bool(np.all((y_v == 0.0) | (y_v == 1.0)))
        y_cols = [y_v.astype(np.float32)]
        for _ in range(n_null_perms):
            y_cols.append(rng.permutation(y_v).astype(np.float32))
        y_mat_np = np.column_stack(y_cols)  # [Nv, 1 + P]

        f64_device = torch.device("cpu") if device.type == "mps" else device
        eye_bool = torch.eye(n_v, device=device, dtype=torch.bool)
        eye_bool_64 = (
            eye_bool
            if f64_device == device
            else torch.eye(n_v, device=f64_device, dtype=torch.bool)
        )
        Xt_64 = (Xt_v if f64_device == device else Xt_v.to(device=f64_device)).to(dtype=torch.float64)
        if is_binary:
            Y1 = torch.tensor(y_mat_np, device=device, dtype=torch.float32)
            Y0 = 1.0 - Y1
            n1 = Y1.sum(dim=0, keepdim=True)
            n0 = Y0.sum(dim=0, keepdim=True)
            same_cnt = Y1 * (n1 - 1.0) + Y0 * (n0 - 1.0)
            Y1_64 = (
                Y1[:, 0:1] if f64_device == device else Y1[:, 0:1].to(device=f64_device)
            ).to(dtype=torch.float64)
            Y0_64 = 1.0 - Y1_64
            same_cnt_64 = (
                same_cnt[:, 0:1]
                if f64_device == device
                else same_cnt[:, 0:1].to(device=f64_device)
            ).to(dtype=torch.float64)
            return {
                "device": device,
                "f64_device": f64_device,
                "Xt": Xt_v,
                "Xt_64": Xt_64,
                "N": n_v,
                "is_binary": True,
                "eye_bool": eye_bool,
                "eye_bool_64": eye_bool_64,
                "Y1": Y1,
                "Y0": Y0,
                "same_cnt": same_cnt,
                "Y1_64": Y1_64,
                "Y0_64": Y0_64,
                "same_cnt_64": same_cnt_64,
                "e_m1_32": float(np.exp(-1.0 / max(soft_temp, 1e-4))),
                "e_m1_64": float(np.exp(-1.0 / max(soft_temp, 1e-6))),
            }

        # General continuous fuzzy membership path y in [0, 1]
        y_t = torch.tensor(y_mat_np, device=device, dtype=torch.float32)  # [Nv, 1+P]
        return {
            "device": device,
            "f64_device": f64_device,
            "Xt": Xt_v,
            "Xt_64": Xt_64,
            "N": n_v,
            "is_binary": False,
            "eye_bool": eye_bool,
            "eye_bool_64": eye_bool_64,
            "y_mat": y_t,
            "y_obs_64": (
                y_t[:, 0:1] if f64_device == device else y_t[:, 0:1].to(device=f64_device)
            ).to(dtype=torch.float64),
        }

    def _eval_view_from_R(
        self, R_mat: torch.Tensor, ctx: Dict[str, object], exact_f64_obs: bool = False
    ) -> Tuple[float, float]:
        """
        Evaluates monotone shifted LogSumExp soft-infimum positive-region dependency gamma_B(D)
        and mean permutation-null dependency gamma_B^{null}(D) on a single view.
        """
        n_v = int(ctx["N"])  # type: ignore[arg-type]
        if n_v <= 1:
            return 0.0, 0.0

        tau = self.soft_temp
        denom = float(n_v - 1)
        view_dev: torch.device = ctx["device"]  # type: ignore[assignment]
        f64_dev: torch.device = ctx["f64_device"]  # type: ignore[assignment]
        eye_bool: torch.Tensor = ctx["eye_bool"]  # type: ignore[assignment]
        eye_bool_64: torch.Tensor = ctx["eye_bool_64"]  # type: ignore[assignment]

        if bool(ctx["is_binary"]):
            if exact_f64_obs:
                R64 = (R_mat if R_mat.device == f64_dev else R_mat.to(device=f64_dev)).to(dtype=torch.float64)
                # Shift by u_max = 0 or exact row max to avoid underflow even for tiny tau -> 0+
                # Log-domain exact evaluation: log((same_cnt * exp(-1/tau) + sum_{opp} exp((R-1)/tau)) / (N-1))
                Y1_64: torch.Tensor = ctx["Y1_64"]  # type: ignore[assignment]
                Y0_64: torch.Tensor = ctx["Y0_64"]  # type: ignore[assignment]
                # Exact pairwise implication I_ij = 1 if y_i == y_j else (1 - R_ij)
                same_pair = (Y1_64 @ Y1_64.T + Y0_64 @ Y0_64.T) > 0.5
                imp_64 = torch.where(same_pair, torch.ones_like(R64), (1.0 - R64).clamp(0.0, 1.0))
                u_64 = (-imp_64 / tau).masked_fill(eye_bool_64, -float("inf"))
                lse_64 = torch.logsumexp(u_64, dim=1) - np.log(denom)
                lower_64 = (-tau * lse_64).clamp(0.0, 1.0)
                gamma_obs = float(lower_64.mean().item())
            else:
                gamma_obs = 0.0

            R32 = (R_mat if R_mat.device == view_dev else R_mat.to(dtype=torch.float32).to(device=view_dev)).to(dtype=torch.float32)
            if tau >= 0.015:
                E_shift = torch.exp((R32 - 1.0) / tau)
                E_shift = E_shift.masked_fill(eye_bool, 0.0)
                Y1: torch.Tensor = ctx["Y1"]  # type: ignore[assignment]
                Y0: torch.Tensor = ctx["Y0"]  # type: ignore[assignment]
                same_cnt: torch.Tensor = ctx["same_cnt"]  # type: ignore[assignment]
                e_m1 = float(ctx["e_m1_32"])  # type: ignore[arg-type]
                opp_sum = (E_shift @ Y0) * Y1 + (E_shift @ Y1) * Y0
                avg_exp = (same_cnt * e_m1 + opp_sum) / denom
                lowers = (-tau * torch.log(avg_exp.clamp(min=1e-38))).clamp(0.0, 1.0)
                gammas = lowers.mean(dim=0)
                if not exact_f64_obs:
                    gamma_obs = float(gammas[0].item())
                gamma_null = (
                    float(gammas[1:].mean().item()) if gammas.numel() > 1 else 0.0
                )
                return gamma_obs, gamma_null

            # Small-tau underflow-safe logsumexp fallback
            Y1_s: torch.Tensor = ctx["Y1"]  # type: ignore[assignment]
            gammas_list = []
            start_col = 1 if exact_f64_obs else 0
            for c_idx in range(start_col, Y1_s.shape[1]):
                yc = Y1_s[:, c_idx : c_idx + 1]
                same_c = (yc @ yc.T + (1.0 - yc) @ (1.0 - yc).T) > 0.5
                imp_c = torch.where(same_c, torch.ones_like(R32), (1.0 - R32).clamp(0.0, 1.0))
                u_c = (-imp_c / tau).masked_fill(eye_bool, -float("inf"))
                low_c = (-tau * (torch.logsumexp(u_c, dim=1) - float(np.log(denom)))).clamp(0.0, 1.0)
                gammas_list.append(float(low_c.mean().item()))
            if not exact_f64_obs:
                gamma_obs = gammas_list[0]
                null_vals = gammas_list[1:]
            else:
                null_vals = gammas_list
            gamma_null = float(np.mean(null_vals)) if len(null_vals) > 0 else 0.0
            return gamma_obs, gamma_null

        # Continuous soft-label evaluation
        if exact_f64_obs:
            R64 = (R_mat if R_mat.device == f64_dev else R_mat.to(device=f64_dev)).to(dtype=torch.float64)
            y_obs_64: torch.Tensor = ctx["y_obs_64"]  # type: ignore[assignment]
            y_same_64 = (1.0 - (y_obs_64 - y_obs_64.T).abs()).clamp(0.0, 1.0)
            imp_64 = (1.0 - R64 + y_same_64).clamp(0.0, 1.0)
            u_64 = (-imp_64 / tau).masked_fill(eye_bool_64, -float("inf"))
            low_64 = (-tau * (torch.logsumexp(u_64, dim=1) - np.log(denom))).clamp(0.0, 1.0)
            gamma_obs = float(low_64.mean().item())
        else:
            gamma_obs = 0.0

        y_mat: torch.Tensor = ctx["y_mat"]  # type: ignore[assignment]
        R32 = (R_mat if R_mat.device == view_dev else R_mat.to(dtype=torch.float32).to(device=view_dev)).to(dtype=torch.float32)
        g_vals = []
        start_col = 1 if exact_f64_obs else 0
        for c_idx in range(start_col, y_mat.shape[1]):
            yc = y_mat[:, c_idx : c_idx + 1]
            y_same = (1.0 - (yc - yc.T).abs()).clamp(0.0, 1.0)
            imp = (1.0 - R32 + y_same).clamp(0.0, 1.0)
            u = (-imp / tau).masked_fill(eye_bool, -float("inf"))
            low = (-tau * (torch.logsumexp(u, dim=1) - float(np.log(denom)))).clamp(0.0, 1.0)
            g_vals.append(float(low.mean().item()))
        if not exact_f64_obs:
            gamma_obs = g_vals[0]
            null_vals = g_vals[1:]
        else:
            null_vals = g_vals
        gamma_null = float(np.mean(null_vals)) if len(null_vals) > 0 else 0.0
        return gamma_obs, gamma_null

    @torch.no_grad()
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> "FuzzyRoughFeatureReducer":
        """
        Computes pointwise T-norm Fuzzy-Rough dependency across all D attributes, ranks
        forward-greedy subsets up to `max_features` via null-calibrated excess dependency,
        and performs backward minimality pruning down to size >= 1.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array X [N, D], got shape {X_arr.shape}")
        y_raw = np.asarray(y, dtype=np.float32).ravel()
        y_arr = np.clip(np.nan_to_num(y_raw, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        N, D = X_arr.shape
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"Feat_{j}" for j in range(D)]
        )
        if len(names) < D:
            names = names + [f"Feat_{j}" for j in range(len(names), D)]
        names = names[:D]

        self.feature_scores_ = {}
        self.dependency_curve_ = []
        self.null_dependency_curve_ = []
        self.excess_dependency_curve_ = []
        self.normalized_excess_curve_ = []
        self.boundary_curve_ = []

        if D == 0 or N == 0:
            self.selected_indices_ = []
            self.selected_names_ = []
            self.minimal_reduct_indices_ = []
            self.minimal_reduct_names_ = []
            self.univariate_scores_ = np.zeros(D, dtype=np.float32)
            self.univariate_excess_scores_ = np.zeros(D, dtype=np.float32)
            return self

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Unsupervised robust standardization strictly on the input training matrix X_arr
        X_norm_all, medians, scales = self._robust_standardize_fit(X_arr)
        self.feature_medians_ = medians
        self.feature_scales_ = scales

        rng = np.random.default_rng(self.seed)
        if self.n_stability_bootstraps is None:
            n_views = 5 if N > self.max_samples else 1
        else:
            n_views = self.n_stability_bootstraps

        view_indices: List[np.ndarray] = []
        if n_views == 1 and N <= self.max_samples:
            view_indices.append(np.arange(N, dtype=np.int64))
        else:
            view_n = (
                self.max_samples
                if N > self.max_samples
                else max(2, int(round(0.85 * N)))
            )
            for _ in range(n_views):
                view_indices.append(self._stratified_subsample_indices(y_arr, view_n, rng))

        views = [
            self._build_view_context(
                torch.tensor(X_norm_all[idx_v], device=device, dtype=torch.float32),
                y_arr[idx_v],
                self.soft_temp,
                self.n_null_permutations,
                rng,
            )
            for idx_v in view_indices
        ]

        # Step 1: Univariate screening across all D attributes (capped at 768 rows/view for speed when D is large)
        uni_cap = 768 if D > 128 else self.max_samples
        uni_views = []
        for idx_v, v_ctx in zip(view_indices, views):
            if len(idx_v) > uni_cap:
                sub_local = self._stratified_subsample_indices(y_arr[idx_v], uni_cap, rng)
                uni_views.append(
                    self._build_view_context(
                        v_ctx["Xt"][sub_local],  # type: ignore[index]
                        y_arr[idx_v][sub_local],
                        self.soft_temp,
                        min(self.n_null_permutations, 8),
                        rng,
                    )
                )
            else:
                uni_views.append(v_ctx)

        univ_gammas = np.zeros(D, dtype=np.float32)
        univ_excess = np.zeros(D, dtype=np.float32)
        for j in range(D):
            g_acc = 0.0
            e_acc = 0.0
            for u_ctx in uni_views:
                Xt_u: torch.Tensor = u_ctx["Xt"]  # type: ignore[assignment]
                col = Xt_u[:, j : j + 1]
                Rj = self._feature_similarity(col)
                go, gn = self._eval_view_from_R(Rj, u_ctx, exact_f64_obs=False)
                g_acc += go
                e_acc += (go - gn) / max(1e-6, 1.0 - gn)
            g_mean = g_acc / float(len(uni_views))
            e_mean = e_acc / float(len(uni_views))
            univ_gammas[j] = float(g_mean)
            univ_excess[j] = float(e_mean)
            self.feature_scores_[names[j]] = float(g_mean)

        self.univariate_scores_ = univ_gammas
        self.univariate_excess_scores_ = univ_excess

        # Step 2: Candidate pool & per-view similarity cache for forward-greedy T-norm construction
        pool_size = min(D, max(self.max_features * 3, 60))
        # Rank candidate pool by null-calibrated excess dependency (breaking exact ties by raw gamma)
        rank_key = univ_excess + 1e-4 * univ_gammas
        candidate_pool = [int(i) for i in np.argsort(-rank_key, kind="mergesort")[:pool_size]]

        dev_obj = torch.device(device)
        f64_dev = torch.device("cpu") if dev_obj.type == "mps" else dev_obj
        R_cur_32 = [
            torch.ones((int(v["N"]), int(v["N"])), device=dev_obj, dtype=torch.float32)  # type: ignore[arg-type]
            for v in views
        ]
        R_cur_64 = [
            torch.ones((int(v["N"]), int(v["N"])), device=f64_dev, dtype=torch.float64)  # type: ignore[arg-type]
            for v in views
        ]

        # Precompute float32 candidate similarity matrices only when memory budget is modest
        max_view_n = max(int(v["N"]) for v in views)  # type: ignore[arg-type]
        use_cache = (len(views) * len(candidate_pool) * max_view_n * max_view_n * 4) <= 300_000_000
        cand_cache_32: Optional[List[Dict[int, torch.Tensor]]] = None
        if use_cache:
            cand_cache_32 = []
            for v in views:
                Xt_v: torch.Tensor = v["Xt"]  # type: ignore[assignment]
                cand_cache_32.append(
                    {cand: self._feature_similarity(Xt_v[:, cand : cand + 1]) for cand in candidate_pool}
                )

        selected: List[int] = []
        selected_set = set()
        k_target = min(self.max_features, len(candidate_pool))

        for _ in range(k_target):
            best_feat = -1
            best_obj = -1e9

            for cand in candidate_pool:
                if cand in selected_set:
                    continue
                obj_sum = 0.0
                for v_idx, v_ctx in enumerate(views):
                    if cand_cache_32 is not None:
                        R_cand_32 = cand_cache_32[v_idx][cand]
                    else:
                        Xt_v = v_ctx["Xt"]  # type: ignore[assignment]
                        R_cand_32 = self._feature_similarity(Xt_v[:, cand : cand + 1])
                    R_try = self._combine_tnorm(R_cur_32[v_idx], R_cand_32)
                    go, gn = self._eval_view_from_R(R_try, v_ctx, exact_f64_obs=False)
                    norm_exc = (go - gn) / max(1e-6, 1.0 - gn)
                    obj_sum += norm_exc + 1e-4 * go

                obj_mean = obj_sum / float(len(views))
                if obj_mean > best_obj:
                    best_obj = obj_mean
                    best_feat = cand

            if best_feat < 0:
                break

            selected.append(best_feat)
            selected_set.add(best_feat)

            # Exact float64 T-norm update & positive-region evaluation (guarantees 0 monotonicity violations)
            g_obs_acc = 0.0
            g_null_acc = 0.0
            for v_idx, v_ctx in enumerate(views):
                Xt_v64: torch.Tensor = v_ctx["Xt_64"]  # type: ignore[assignment]
                col64 = Xt_v64[:, best_feat : best_feat + 1]
                R_best_64 = torch.exp(-(col64 - col64.T).abs() / self.sigma).clamp(0.0, 1.0)
                R_cur_64[v_idx] = self._combine_tnorm(R_cur_64[v_idx], R_best_64)
                R_cur_32[v_idx] = (
                    R_cur_64[v_idx].to(dtype=torch.float32)
                    if f64_dev == dev_obj
                    else R_cur_64[v_idx].to(dtype=torch.float32).to(device=dev_obj)
                )

                go_64, gn_32 = self._eval_view_from_R(R_cur_64[v_idx], v_ctx, exact_f64_obs=True)
                g_obs_acc += go_64
                g_null_acc += gn_32

            gamma_step = g_obs_acc / float(len(views))
            null_step = g_null_acc / float(len(views))
            exc_step = gamma_step - null_step
            norm_exc_step = exc_step / max(1e-6, 1.0 - null_step)
            bnd_step = max(0.0, 1.0 - gamma_step)

            self.dependency_curve_.append(float(gamma_step))
            self.null_dependency_curve_.append(float(null_step))
            self.excess_dependency_curve_.append(float(exc_step))
            self.normalized_excess_curve_.append(float(norm_exc_step))
            self.boundary_curve_.append(float(bnd_step))

        self.selected_indices_ = [int(i) for i in selected]
        self.selected_names_ = [names[i] for i in self.selected_indices_]

        # Step 3: Exact Dubois-Prade backward minimality pruning (down to size >= 1)
        pruned = list(self.selected_indices_)
        if len(pruned) > 1 and len(self.dependency_curve_) > 0:
            full_gamma = self.dependency_curve_[-1]
            # Precompute float64 per-feature similarity matrices ONLY for the small selected subset (<= max_features)
            sel_R64: List[Dict[int, torch.Tensor]] = []
            for v_ctx in views:
                Xt_v64 = v_ctx["Xt_64"]  # type: ignore[assignment]
                sel_R64.append(
                    {
                        idx_s: torch.exp(
                            -(Xt_v64[:, idx_s : idx_s + 1] - Xt_v64[:, idx_s : idx_s + 1].T).abs()
                            / self.sigma
                        ).clamp(0.0, 1.0)
                        for idx_s in pruned
                    }
                )

            for cand in reversed(list(self.selected_indices_)):
                if len(pruned) <= 1:
                    break
                trial_subset = [idx_s for idx_s in pruned if idx_s != cand]
                g_rem_acc = 0.0
                for v_idx, v_ctx in enumerate(views):
                    n_v = int(v_ctx["N"])  # type: ignore[arg-type]
                    R_rem = torch.ones((n_v, n_v), device=f64_dev, dtype=torch.float64)
                    for idx_s in trial_subset:
                        R_rem = self._combine_tnorm(R_rem, sel_R64[v_idx][idx_s])
                    go_rem, _ = self._eval_view_from_R(R_rem, v_ctx, exact_f64_obs=True)
                    g_rem_acc += go_rem
                gamma_rem = g_rem_acc / float(len(views))
                if gamma_rem >= full_gamma - self.redundancy_tol:
                    pruned = trial_subset

        self.minimal_reduct_indices_ = [int(i) for i in pruned]
        self.minimal_reduct_names_ = [names[i] for i in self.minimal_reduct_indices_]
        return self

    def transform(self, X: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Selects the top-k Fuzzy-Rough attributes."""
        if k is None:
            idx = self.selected_indices_
        else:
            idx = self.selected_indices_[:k]
        return np.asarray(X)[:, idx]

    def get_reduct_summary(self) -> Dict[str, object]:
        return {
            "selected_indices": [int(i) for i in self.selected_indices_],
            "selected_names": list(self.selected_names_),
            "minimal_reduct_indices": [int(i) for i in self.minimal_reduct_indices_],
            "minimal_reduct_names": list(self.minimal_reduct_names_),
            "dependency_curve": [float(v) for v in self.dependency_curve_],
            "null_dependency_curve": [float(v) for v in self.null_dependency_curve_],
            "excess_dependency_curve": [float(v) for v in self.excess_dependency_curve_],
            "boundary_curve": [float(v) for v in self.boundary_curve_],
            "top_univariate_scores": {
                str(k): float(v)
                for k, v in sorted(self.feature_scores_.items(), key=lambda x: -x[1])[:30]
            },
        }
