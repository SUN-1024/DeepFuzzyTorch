#!/usr/bin/env python3
"""Track 6: Computational Complexity, Wall-Clock Runtime & Peak Memory Benchmark.

Empirically benchmarks:
1. Sample/Batch scaling: N in {64, 256, 1024, 4096} (fixed D=100 per modality, M=4)
2. Feature dimension scaling: D in {100, 1000, 5000} (fixed N=256, M=4; plus N=1024 reducer scaling)
3. Modality count scaling: M in {2, 4, 8, 16} (fixed N=256, D=100)

Models compared:
- `UnifiedDeepFuzzyModel` (ours: full end-to-end differentiable model)
- `UnifiedDeepFuzzyModel_PostReduct20` (ours after K*=20 fuzzy-rough reduction for high-D inputs)
- `FuzzyRoughFeatureReducer` (pointwise T-norm rough-set reducer fit wall-clock time & memory)
- `EarlyConcat` (`EarlyConcatMLP`)
- `GatedAttention` (`GatedAttentionFusion`)
- `EvidentialDL` (`EvidentialDL` with Dirichlet KL annealing)
- `RCML` (`RCMLBaseline`, AAAI 2024 reliable conflictive multi-view learning)
- `QMF` (`QMFBaseline`, ICML 2023 quality-aware multimodal fusion)
- `DrFuse` (`DrFuseBaseline`, AAAI 2024 disentangled representation fusion)

Saves standalone results to `results/runtime_vram_benchmark.json` and embeds into
`results/benchmark_results.json["track6_runtime_complexity"]` if present.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import sys
import tempfile
import threading
import time
import weakref
from typing import Any, Dict, List

if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
    try:
        tempfile.gettempdir()
    except FileNotFoundError:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deepfuzzy import IntuitionisticFuzzyLoss, UnifiedDeepFuzzyModel
from deepfuzzy.baselines import (
    DrFuseBaseline,
    EarlyConcatMLP,
    EvidentialDL,
    GatedAttentionFusion,
    QMFBaseline,
    RCMLBaseline,
)
from deepfuzzy.rough_reducer import FuzzyRoughFeatureReducer


class PeakTensorMemoryTracker(TorchDispatchMode):
    """Exact live PyTorch tensor byte tracker (Tensor -> StorageImpl -> Buffer two-layer tracking)."""

    _RESIZE_LOCK = threading.RLock()
    _ACTIVE_TRACKERS: List["PeakTensorMemoryTracker"] = []
    _ORIG_UNTYPED_RESIZE: Any = None

    def __init__(self, pinned_tensors: Any = ()) -> None:
        super().__init__()
        self._tensor_sid: Dict[int, int] = {}
        self._sid_refs: Dict[int, int] = {}
        self._sid_info: Dict[int, Tuple[int, int]] = {}
        self._live_wrefs: Dict[int, weakref.ref] = {}
        self._finalizers: Dict[int, weakref.finalize] = {}
        self._finalized_tids: set[int] = set()
        self.pinned_bytes: int = 0
        self.cur_bytes: int = 0
        self.peak_bytes: int = 0
        self._pin_objects(pinned_tensors)
        self.pinned_bytes = self.cur_bytes

    def _recompute_union_bytes(self) -> int:
        intervals = [
            (ptr, ptr + nb)
            for ptr, nb in self._sid_info.values()
            if ptr and nb > 0
        ]
        if not intervals:
            self.cur_bytes = 0
            return 0
        intervals.sort()
        total = 0
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                total += cur_e - cur_s
                cur_s, cur_e = s, e
        total += cur_e - cur_s
        self.cur_bytes = total
        if total > self.peak_bytes:
            self.peak_bytes = total
        return total

    def _update_sid(self, sid: int, ptr: int, nbytes: int) -> None:
        eff_ptr = ptr if (ptr and nbytes > 0) else 0
        eff_nb = nbytes if eff_ptr else 0
        if self._sid_info.get(sid, (0, 0)) == (eff_ptr, eff_nb):
            return
        self._sid_info[sid] = (eff_ptr, eff_nb)
        self._recompute_union_bytes()

    def _dec_sid(self, sid: int) -> None:
        if not sid:
            return
        cnt = self._sid_refs.get(sid, 0) - 1
        if cnt <= 0:
            self._sid_refs.pop(sid, None)
            self._sid_info.pop(sid, None)
            self._recompute_union_bytes()
        else:
            self._sid_refs[sid] = cnt

    def _inc_sid(self, sid: int, ptr: int, nbytes: int) -> None:
        if not sid:
            return
        if sid not in self._sid_refs:
            self._sid_refs[sid] = 1
        else:
            self._sid_refs[sid] += 1
        self._update_sid(sid, ptr, nbytes)

    def _pin_objects(self, obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            self._track_tensor(obj)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                self._pin_objects(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                self._pin_objects(item)

    @staticmethod
    def _make_gc_cb(
        tracker_wref: "weakref.ref[PeakTensorMemoryTracker]",
        t_id: int,
    ):
        def _on_gc() -> None:
            tr = tracker_wref()
            if tr is None:
                return
            tr._finalized_tids.discard(t_id)
            tr._finalizers.pop(t_id, None)
            tr._live_wrefs.pop(t_id, None)
            cur_s = tr._tensor_sid.pop(t_id, 0)
            tr._dec_sid(cur_s)

        return _on_gc

    def _track_tensor(self, obj: torch.Tensor) -> None:
        tid = id(obj)
        try:
            st = obj.untyped_storage()
            sid = int(st._cdata)
            ptr = int(st.data_ptr())
            nbytes = int(st.nbytes())
        except Exception:
            return
        prev_sid = self._tensor_sid.get(tid)
        if prev_sid is None:
            self._tensor_sid[tid] = sid
            self._inc_sid(sid, ptr, nbytes)
        elif prev_sid != sid:
            self._dec_sid(prev_sid)
            self._tensor_sid[tid] = sid
            self._inc_sid(sid, ptr, nbytes)
        else:
            self._update_sid(sid, ptr, nbytes)

        self._live_wrefs[tid] = weakref.ref(obj)
        if tid not in self._finalized_tids:
            self._finalized_tids.add(tid)
            self._finalizers[tid] = weakref.finalize(obj, self._make_gc_cb(weakref.ref(self), tid))

    def _release_all(self) -> None:
        for fin in list(self._finalizers.values()):
            try:
                fin.detach()
            except Exception:
                pass
        self._finalizers.clear()
        self._finalized_tids.clear()
        self._live_wrefs.clear()
        self._tensor_sid.clear()
        self._sid_refs.clear()
        self._sid_info.clear()

    def _sync_live_storages(self) -> None:
        for tid, wref in list(self._live_wrefs.items()):
            obj = wref()
            if obj is not None:
                self._track_tensor(obj)

    def __enter__(self):
        ret = super().__enter__()
        cls = type(self)
        with cls._RESIZE_LOCK:
            if not cls._ACTIVE_TRACKERS:
                orig = torch.UntypedStorage.resize_
                cls._ORIG_UNTYPED_RESIZE = orig

                def _tracked_untyped_resize(storage_self, *args, **kwargs):
                    res = orig(storage_self, *args, **kwargs)
                    with cls._RESIZE_LOCK:
                        trackers = list(cls._ACTIVE_TRACKERS)
                    for tr in trackers:
                        try:
                            tr._sync_live_storages()
                        except Exception:
                            pass
                    return res

                torch.UntypedStorage.resize_ = _tracked_untyped_resize  # type: ignore[method-assign]
            cls._ACTIVE_TRACKERS.append(self)
        return ret

    def __exit__(self, exc_type, exc_val, exc_tb):
        cls = type(self)
        try:
            self._sync_live_storages()
        finally:
            with cls._RESIZE_LOCK:
                try:
                    cls._ACTIVE_TRACKERS.remove(self)
                except ValueError:
                    pass
                if not cls._ACTIVE_TRACKERS and cls._ORIG_UNTYPED_RESIZE is not None:
                    torch.UntypedStorage.resize_ = cls._ORIG_UNTYPED_RESIZE  # type: ignore[method-assign]
                    cls._ORIG_UNTYPED_RESIZE = None
            self._release_all()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def _refresh(self, obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            self._track_tensor(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._refresh(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                self._refresh(item)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self._refresh(args)
        if kwargs:
            self._refresh(kwargs)
        self._refresh(out)
        return out


def _measure_single_step_peak_mb(
    step_fn, model: torch.nn.Module, inputs: List[torch.Tensor], device: torch.device
) -> float:
    """Measure exact peak live tensor memory (parameters + buffers + inputs + live activations/gradients) in MiB."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        step_fn()
        torch.cuda.synchronize(device)
        return round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 3)

    gc.collect()
    pinned = (list(model.parameters()), list(model.buffers()), inputs)
    with PeakTensorMemoryTracker(pinned_tensors=pinned) as tracker:
        step_fn()
    return round(tracker.peak_bytes / (1024.0 * 1024.0), 3)


def _build_model(name: str, modality_dims: List[int], embed_dim: int = 48) -> torch.nn.Module:
    if name == "UnifiedDeepFuzzyModel":
        return UnifiedDeepFuzzyModel(
            modality_dims=modality_dims,
            embed_dim=embed_dim,
            num_classes=1,
            num_tsk_concepts=8,
            num_tsk_rules=6,
        )
    if name == "EarlyConcat":
        return EarlyConcatMLP(modality_dims=modality_dims, embed_dim=embed_dim)
    if name == "GatedAttention":
        return GatedAttentionFusion(modality_dims=modality_dims, embed_dim=embed_dim)
    if name == "EvidentialDL":
        return EvidentialDL(modality_dims=modality_dims, embed_dim=embed_dim)
    if name == "RCML":
        return RCMLBaseline(modality_dims=modality_dims, embed_dim=embed_dim)
    if name == "QMF":
        return QMFBaseline(modality_dims=modality_dims, embed_dim=embed_dim)
    if name == "DrFuse":
        return DrFuseBaseline(modality_dims=modality_dims, embed_dim=embed_dim)
    raise ValueError(f"Unknown model: {name}")


def _benchmark_neural_config(
    model_name: str,
    N: int,
    D: int,
    M: int,
    device: torch.device,
    n_warmup: int = 5,
    n_repeats: int = 30,
) -> Dict[str, Any]:
    torch.manual_seed(42)
    modality_dims = [D] * M
    model = _build_model(model_name, modality_dims, embed_dim=48).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    fuzzy_crit = IntuitionisticFuzzyLoss() if model_name == "UnifiedDeepFuzzyModel" else None

    xs = [torch.randn(N, D, device=device, dtype=torch.float32) for _ in range(M)]
    mask = (torch.rand(N, M, device=device) > 0.15).float()
    mask[:, 0] = 1.0
    yb = (torch.rand(N, 1, device=device) > 0.5).float()

    def fwd_bwd_step() -> None:
        if model_name == "UnifiedDeepFuzzyModel":
            out = model(xs, modality_masks=mask)
            loss, _ = fuzzy_crit(
                out["fused_ifn"],
                yb,
                modality_ifns=out["modality_ifns"],
                modality_masks=mask,
                final_prob=out["prob"],
                rough_boundary=out["rough_boundary"],
                gate_sparsity_l1=out["gate_sparsity_l1"],
            )
        else:
            out = model(xs, modality_masks=mask)
            prob = out["prob"].clamp(1e-6, 1.0 - 1e-6)
            loss = F.binary_cross_entropy(prob, yb)
            if "modality_probs" in out:
                aux = 0.0
                for m_i, mp in enumerate(out["modality_probs"]):
                    obs_m = mask[:, m_i : m_i + 1]
                    aux = aux + (
                        F.binary_cross_entropy(mp.clamp(1e-6, 1.0 - 1e-6), yb, reduction="none")
                        * obs_m
                    ).mean()
                loss = loss + 0.2 * (aux / M)
            if hasattr(model, "compute_kl_loss"):
                loss = loss + 0.2 * model.compute_kl_loss(yb, epoch=5, annealing_step=10)
            if "disentangle_loss" in out:
                loss = loss + 0.1 * out["disentangle_loss"]
        loss.backward()

    def infer_step() -> None:
        with torch.no_grad():
            _ = model(xs, modality_masks=mask)

    model.train()
    for _ in range(n_warmup):
        opt.zero_grad(set_to_none=True)
        fwd_bwd_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    train_times_ms: List[float] = []
    for _ in range(n_repeats):
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fwd_bwd_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_times_ms.append((time.perf_counter() - t0) * 1000.0)

    opt.zero_grad(set_to_none=True)
    peak_mem_mb = _measure_single_step_peak_mb(fwd_bwd_step, model, [*xs, mask, yb], device)
    opt.zero_grad(set_to_none=True)

    model.eval()
    for _ in range(n_warmup):
        infer_step()
    infer_times_ms: List[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        infer_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_times_ms.append((time.perf_counter() - t0) * 1000.0)

    n_params = int(sum(p.numel() for p in model.parameters()))
    return {
        "model": model_name,
        "N": N,
        "D_per_modality": D,
        "M_modalities": M,
        "n_params": n_params,
        "fwd_bwd_mean_ms": round(float(np.mean(train_times_ms)), 3),
        "fwd_bwd_std_ms": round(float(np.std(train_times_ms, ddof=1)), 3),
        "fwd_bwd_median_ms": round(float(np.median(train_times_ms)), 3),
        "infer_mean_ms": round(float(np.mean(infer_times_ms)), 3),
        "infer_std_ms": round(float(np.std(infer_times_ms, ddof=1)), 3),
        "peak_memory_mib": peak_mem_mb,
    }


def _benchmark_reducer_config(
    N: int,
    D: int,
    max_features: int = 20,
    n_null_permutations: int = 8,
    n_stability_bootstraps: int = 1,
    n_repeats: int = 3,
) -> Dict[str, Any]:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((N, D)).astype(np.float32)
    signal = X[:, : min(10, D)].sum(axis=1)
    y = (signal + 0.5 * rng.standard_normal(N) > 0.0).astype(np.int64)

    fit_times_s: List[float] = []
    peak_mb = 0.0
    for r_i in range(n_repeats):
        reducer = FuzzyRoughFeatureReducer(
            max_features=max_features,
            sigma=0.85,
            max_samples=4096,
            n_null_permutations=n_null_permutations,
            n_stability_bootstraps=n_stability_bootstraps,
            seed=42 + r_i,
        )
        t0 = time.perf_counter()
        reducer.fit(X, y)
        fit_times_s.append(time.perf_counter() - t0)

    gc.collect()
    red_mem = FuzzyRoughFeatureReducer(
        max_features=max_features,
        sigma=0.85,
        max_samples=4096,
        n_null_permutations=n_null_permutations,
        n_stability_bootstraps=n_stability_bootstraps,
        seed=42,
    )
    x_pin = torch.from_numpy(np.ascontiguousarray(X))
    y_pin = torch.from_numpy(np.ascontiguousarray(y))
    with PeakTensorMemoryTracker(pinned_tensors=(x_pin, y_pin)) as tracker:
        red_mem.fit(X, y)
    peak_mb = round(tracker.peak_bytes / (1024.0 * 1024.0), 3)

    return {
        "N": N,
        "D_candidates": D,
        "max_features_selected": max_features,
        "n_null_permutations": n_null_permutations,
        "n_stability_bootstraps": n_stability_bootstraps,
        "fit_mean_sec": round(float(np.mean(fit_times_s)), 4),
        "fit_std_sec": round(float(np.std(fit_times_s, ddof=1)) if len(fit_times_s) > 1 else 0.0, 4),
        "fit_mean_ms": round(float(np.mean(fit_times_s) * 1000.0), 2),
        "peak_memory_mib": peak_mb,
    }


def main() -> None:
    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw_info = {
        "device": str(device),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count": os.cpu_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    print(f"Running Track 6 Runtime & Peak Memory Benchmark on {hw_info['device']} ({hw_info['processor']})...")

    models_core = [
        "UnifiedDeepFuzzyModel",
        "EarlyConcat",
        "GatedAttention",
        "EvidentialDL",
        "RCML",
        "QMF",
        "DrFuse",
    ]

    # 1. Sweep N in {64, 256, 1024, 4096} (D=100, M=4)
    sweep_N_neural: List[Dict[str, Any]] = []
    for N in [64, 256, 1024, 4096]:
        for m_name in models_core:
            row = _benchmark_neural_config(
                m_name, N=N, D=100, M=4, device=device, n_warmup=5, n_repeats=30
            )
            sweep_N_neural.append(row)
        print(
            f"  [Sweep N={N:4d}, D=100, M=4] DeepFuzzy={sweep_N_neural[-7]['fwd_bwd_mean_ms']:.2f}ms "
            f"({sweep_N_neural[-7]['peak_memory_mib']:.1f}MiB) | EarlyConcat={sweep_N_neural[-6]['fwd_bwd_mean_ms']:.2f}ms | "
            f"RCML={sweep_N_neural[-3]['fwd_bwd_mean_ms']:.2f}ms"
        )

    # 2. Sweep D in {100, 1000, 5000} (N=256, M=4) + PostReduct(K*=20)
    sweep_D_neural: List[Dict[str, Any]] = []
    for D in [100, 1000, 5000]:
        for m_name in models_core:
            row = _benchmark_neural_config(
                m_name, N=256, D=D, M=4, device=device, n_warmup=5, n_repeats=30
            )
            sweep_D_neural.append(row)
        post_red = _benchmark_neural_config(
            "UnifiedDeepFuzzyModel", N=256, D=min(D, 20), M=4, device=device, n_warmup=5, n_repeats=30
        )
        post_red["model"] = "UnifiedDeepFuzzyModel_PostReduct20"
        post_red["raw_D_before_reduct"] = D
        sweep_D_neural.append(post_red)
        print(
            f"  [Sweep D={D:4d}, N=256, M=4] DeepFuzzy(raw)={sweep_D_neural[-8]['fwd_bwd_mean_ms']:.2f}ms | "
            f"DeepFuzzy(PostReduct20)={post_red['fwd_bwd_mean_ms']:.2f}ms | EarlyConcat={sweep_D_neural[-7]['fwd_bwd_mean_ms']:.2f}ms"
        )

    # 3. Sweep M in {2, 4, 8, 16} (N=256, D=100) — verifies O(N * M * log M * C) Choquet + O(N * M^2 * C) conflict scaling
    sweep_M_neural: List[Dict[str, Any]] = []
    for M in [2, 4, 8, 16]:
        for m_name in models_core:
            row = _benchmark_neural_config(
                m_name, N=256, D=100, M=M, device=device, n_warmup=5, n_repeats=30
            )
            sweep_M_neural.append(row)
        print(
            f"  [Sweep M={M:2d}, N=256, D=100] DeepFuzzy={sweep_M_neural[-7]['fwd_bwd_mean_ms']:.2f}ms "
            f"(params={sweep_M_neural[-7]['n_params']}) | GatedAttn={sweep_M_neural[-5]['fwd_bwd_mean_ms']:.2f}ms | "
            f"RCML={sweep_M_neural[-3]['fwd_bwd_mean_ms']:.2f}ms"
        )

    # 4. FuzzyRoughFeatureReducer scaling across N in {64, 256, 1024, 4096} and D in {100, 1000, 5000}
    reducer_sweep_N: List[Dict[str, Any]] = []
    for N in [64, 256, 1024, 4096]:
        n_rep = 2 if N == 4096 else 3
        r_row = _benchmark_reducer_config(
            N=N, D=100, max_features=20, n_null_permutations=8, n_stability_bootstraps=1, n_repeats=n_rep
        )
        reducer_sweep_N.append(r_row)
        print(
            f"  [Reducer Sweep N={N:4d}, D=100] fit={r_row['fit_mean_sec']:.3f}s ({r_row['peak_memory_mib']:.1f}MiB)"
        )

    reducer_sweep_D: List[Dict[str, Any]] = []
    for D in [100, 1000, 5000]:
        n_rep = 2 if D == 5000 else 3
        r_row = _benchmark_reducer_config(
            N=256, D=D, max_features=20, n_null_permutations=8, n_stability_bootstraps=1, n_repeats=n_rep
        )
        reducer_sweep_D.append(r_row)
        print(
            f"  [Reducer Sweep D={D:4d}, N=256] fit={r_row['fit_mean_sec']:.3f}s ({r_row['peak_memory_mib']:.1f}MiB)"
        )

    reducer_tcga_scale = _benchmark_reducer_config(
        N=1024, D=1000, max_features=20, n_null_permutations=8, n_stability_bootstraps=1, n_repeats=2
    )
    print(
        f"  [Reducer N=1024, D=1000] fit={reducer_tcga_scale['fit_mean_sec']:.3f}s ({reducer_tcga_scale['peak_memory_mib']:.1f}MiB)"
    )

    payload = {
        "hardware_environment": hw_info,
        "memory_unit": "MiB",
        "theoretical_complexity": {
            "FuzzyRoughFeatureReducer_fit": "O(K_max * D * N^2) offline once per training fold (sub-quadratic O(K_max * D * S^2) when N > max_samples=S)",
            "IFS_Encoder_and_RoughBottleneck": "O(N * sum_m D_m * d) linear in sample size and input dimension",
            "CertaintyOrdered_IFSChoquetFusion": "O(N * M * log M * C + N * M^2 * C) via certainty-ordered Choquet capacity + pairwise conflict tensor (avoids exponential O(2^M) subset enumeration)",
            "DeepConceptTSKHead": "O(N * C * R) linear in concepts C and rules R",
        },
        "sweep_N_D100_M4": sweep_N_neural,
        "sweep_D_N256_M4": sweep_D_neural,
        "sweep_M_N256_D100": sweep_M_neural,
        "reducer_sweep_N_D100": reducer_sweep_N,
        "reducer_sweep_D_N256": reducer_sweep_D,
        "reducer_N1024_D1000": reducer_tcga_scale,
        "total_benchmark_wall_sec": round(time.perf_counter() - t_start, 2),
    }

    out_path = os.path.join(ROOT, "results/runtime_vram_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved runtime & memory benchmark to {out_path} in {payload['total_benchmark_wall_sec']:.1f}s")

    bench_json_path = os.path.join(ROOT, "results/benchmark_results.json")
    if os.path.exists(bench_json_path):
        with open(bench_json_path, "r") as f:
            bdata = json.load(f)
        bdata["track6_runtime_complexity"] = payload
        with open(bench_json_path, "w") as f:
            json.dump(bdata, f, indent=2)
        print(f"Embedded `track6_runtime_complexity` into {bench_json_path}")


if __name__ == "__main__":
    main()
