"""
DeepFuzzyTorch: Core Differentiable Fuzzy, Intuitionistic, Hesitant, Rough & TSK Operators
==========================================================================================
Target Venues (HMC 331 High-Quality Consensus Catalog):
  1. Primary Target: INFORMATION FUSION (Tier D-30, IF 14-18)
  2. Dual-Track Bio-Software Targets: GENOME BIOLOGY (Tier D-28) & NUCLEIC ACIDS RESEARCH (Tier E-123)

Mathematical Guarantees Implemented & Verified:
  - Theorem 1 (Simplex Closure): mu + nu + pi = 1 with mu, nu, pi >= 0 in exact real arithmetic
    for all inputs (including malformed/non-finite inputs sanitized onto the closed simplex).
  - Theorem 2 (Exact Missing-Modality Vacuity Degeneration): Missing modality (o_m = 0)
    collapses to the vacuous IFN <0, 0, 1>, receives zero Choquet capacity increment w_m = 0,
    does not dilute observed-pair conflict C_O, and yields exact algebraic equivalence to the
    |O|-modality operator (preserving canonical relative order among active modalities) for all
    conflict penalties gamma >= 0.
  - Theorem 3 (Partial-Order Monotonicity & Idempotency): Under Atanassov partial order
    (mu_m <= mu'_m, nu_m >= nu'_m) on a fixed active set O, at fixed modality certainties r_j and
    frozen conflict discount d > 0 on the open simplex interior where pi_hat_k > 0 for all k in O,
    d(mu_fused)/d(mu_m) > 0 and d(nu_fused)/d(nu_m) > 0.
  - Theorem 4 (Conflict-Driven Epistemic Hesitancy Inflation): On a fixed active set O, since
    Choquet weights w_m depend solely on observed non-vacuity r_m = 1 - pi_m and Sugeno densities
    g_m^0, dw_m/dC_O = 0 and d(log pi_fused)/dC_O > 0 holds for all gamma > 0 under fixed
    non-vacuity masses r_m = mu_m + nu_m in (0, 1).
  - Theorem 5 (Projected Soft-Lukasiewicz Fuzzy-Rough Inclusion):
    0 <= R_low(A)(x) <= A(x) <= R_up(A)(x) <= 1 and BND_R(A)(x) >= 0 for all beta in [0, 1].
"""

from __future__ import annotations
import atexit
from collections import deque
from dataclasses import dataclass
import math
import re
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import warnings
import weakref
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


_HALF_PREC_DTYPES = (torch.float16, torch.bfloat16)
_FP16_GRAD_MAX = 8000.0
_FP16_MULTI_BUDGET = 32000.0
_FP16_BWD_ROOT_MAX = 48000.0
_FP16_OVERFLOW_ROOT_KEY = -1
_BF16_GRAD_MAX = 3.3e37
_HALF_PREC_TLS = threading.local()
_FP16_LOAD_LOCK = threading.Lock()
# Retained as empty exports for backward compatibility; root accounting is scoped
# strictly to each active autograd execution pass so live unselected graphs cannot interfere.
_FP16_ACTIVE_ROOT_LOAD: Dict[int, float] = {}
_FP16_BWD_PENDING_TOKENS: Dict[int, List[object]] = {}
_FP16_BWD_USED_BUDGET: Dict[object, Dict[int, Dict[str, object]]] = {}
_FP16_BWD_CB_QUEUED: Dict[object, bool] = {}
_FP16_BWD_PASS_META: Dict[object, Dict[str, object]] = {}

_TEARDOWN_ANCHOR_STEP = 500
_PENDING_TEARDOWN_ANCHORS: deque = deque()
_TEARDOWN_DRAIN_ARMED = False


def _drain_teardown_anchors() -> None:
    """Iteratively frees chunked C++ autograd Node milestones outside ~Node() to avoid C-stack overflow."""
    global _TEARDOWN_DRAIN_ARMED
    if getattr(_HALF_PREC_TLS, "draining_anchors", False):
        return
    _HALF_PREC_TLS.draining_anchors = True
    try:
        while True:
            try:
                batch = _PENDING_TEARDOWN_ANCHORS.popleft()
            except IndexError:
                _TEARDOWN_DRAIN_ARMED = False
                if not _PENDING_TEARDOWN_ANCHORS:
                    break
                continue
            while batch:
                item = batch.pop()
                del item
    finally:
        _HALF_PREC_TLS.draining_anchors = False
        if _PENDING_TEARDOWN_ANCHORS and not _TEARDOWN_DRAIN_ARMED:
            _TEARDOWN_DRAIN_ARMED = True
            _DeferredAnchorDrain()


atexit.register(_drain_teardown_anchors)


class _DeferredAnchorDrain:
    """Self-cyclic handle collected by cyclic GC after Node::~Node() finishes next_edges_ teardown."""

    __slots__ = ("_self",)

    def __init__(self) -> None:
        self._self = self

    def __del__(self) -> None:
        try:
            _drain_teardown_anchors()
        except Exception:
            pass


class _AnchorSentinel:
    """Stores child milestone autograd nodes on parent.metadata and defers their destruction."""

    __slots__ = ("children",)

    def __init__(self) -> None:
        self.children: List[object] = []

    def __del__(self) -> None:
        global _TEARDOWN_DRAIN_ARMED
        ch = self.children
        self.children = []
        if ch:
            _PENDING_TEARDOWN_ANCHORS.append(ch)
            if not _TEARDOWN_DRAIN_ARMED:
                _TEARDOWN_DRAIN_ARMED = True
                try:
                    _DeferredAnchorDrain()
                except Exception:
                    pass


def _link_teardown_milestone(
    mile_node: object, cur_node: object, single_child: bool = False
) -> object:
    """Attaches `cur_node` as a deferred teardown child of `mile_node` (bounding C-stack recursion)."""
    if mile_node is None or cur_node is None or mile_node is cur_node:
        return cur_node
    cur_md = getattr(cur_node, "metadata", None)
    mile_md = getattr(mile_node, "metadata", None)
    if not isinstance(cur_md, dict) or not isinstance(mile_md, dict):
        return cur_node
    if not cur_md.get("_df_anchored", False):
        sent = mile_md.get("_df_sentinel")
        if not isinstance(sent, _AnchorSentinel):
            sent = _AnchorSentinel()
            mile_md["_df_sentinel"] = sent
            if single_child:
                mile_md["_df_sentinel_single"] = True
        elif not single_child:
            mile_md["_df_sentinel_single"] = False
        sent.children.append(cur_node)
        cur_md["_df_anchored"] = True
    return cur_node


def _step_teardown_milestone(
    mile_node: object,
    dist: int,
    cur_node: object,
    step_inc: int = 1,
    single_child: bool = False,
) -> Tuple[object, int]:
    nd = dist + step_inc
    if nd < _TEARDOWN_ANCHOR_STEP or mile_node is None or cur_node is None or mile_node is cur_node:
        return mile_node, nd
    cur_md = getattr(cur_node, "metadata", None)
    mile_md = getattr(mile_node, "metadata", None)
    if not isinstance(cur_md, dict) or not isinstance(mile_md, dict):
        return mile_node, nd
    if not cur_md.get("_df_anchored", False):
        _link_teardown_milestone(mile_node, cur_node, single_child=single_child)
        return cur_node, 0
    if isinstance(cur_md.get("_df_sentinel"), _AnchorSentinel):
        if single_child or cur_md.get("_df_sentinel_single", False):
            return cur_node, 0
    return mile_node, nd


def _flush_fp16_bwd_pass(pass_key: object) -> None:
    metas_to_clean: List[Dict[str, object]] = []
    with _FP16_LOAD_LOCK:
        if isinstance(pass_key, int):
            _FP16_BWD_PENDING_TOKENS.pop(pass_key, None)
            _FP16_BWD_USED_BUDGET.pop(pass_key, None)
            _FP16_BWD_CB_QUEUED.pop(pass_key, None)
            m0 = _FP16_BWD_PASS_META.pop(pass_key, None)
            if isinstance(m0, dict):
                metas_to_clean.append(m0)
            stale_keys = [
                k for k in list(_FP16_BWD_USED_BUDGET.keys())
                if isinstance(k, tuple) and k and k[0] == pass_key
            ]
            for sk in stale_keys:
                _FP16_BWD_USED_BUDGET.pop(sk, None)
                _FP16_BWD_CB_QUEUED.pop(sk, None)
                ms = _FP16_BWD_PASS_META.pop(sk, None)
                if isinstance(ms, dict):
                    metas_to_clean.append(ms)
            stale_cb = [
                k for k in list(_FP16_BWD_CB_QUEUED.keys())
                if isinstance(k, tuple) and k and k[0] == pass_key
            ]
            for sk in stale_cb:
                _FP16_BWD_CB_QUEUED.pop(sk, None)
                ms2 = _FP16_BWD_PASS_META.pop(sk, None)
                if isinstance(ms2, dict):
                    metas_to_clean.append(ms2)
            stale_meta = [
                k for k in list(_FP16_BWD_PASS_META.keys())
                if isinstance(k, tuple) and k and k[0] == pass_key
            ]
            for sk in stale_meta:
                ms3 = _FP16_BWD_PASS_META.pop(sk, None)
                if isinstance(ms3, dict):
                    metas_to_clean.append(ms3)
        else:
            _FP16_BWD_USED_BUDGET.pop(pass_key, None)
            _FP16_BWD_CB_QUEUED.pop(pass_key, None)
            ms4 = _FP16_BWD_PASS_META.pop(pass_key, None)
            if isinstance(ms4, dict):
                metas_to_clean.append(ms4)
    for meta in metas_to_clean:
        for hdict_key in ("hooked_py_fns", "pre_hooked_py_fns"):
            hooks = meta.get(hdict_key)
            if isinstance(hooks, dict):
                for h in hooks.values():
                    if hasattr(h, "remove"):
                        try:
                            h.remove()
                        except Exception:
                            pass
                hooks.clear()
        inner_map = meta.get("inner_ckpt_hooks_by_pid")
        if isinstance(inner_map, dict):
            for h_list in inner_map.values():
                if isinstance(h_list, list):
                    for h in h_list:
                        if hasattr(h, "remove"):
                            try:
                                h.remove()
                            except Exception:
                                pass
                    h_list.clear()
            inner_map.clear()
        orig_fns = meta.get("orig_ckpt_run_fns")
        if isinstance(orig_fns, list):
            for ctx_o, fn_o in orig_fns:
                try:
                    if hasattr(ctx_o, "run_function"):
                        ctx_o.run_function = fn_o
                except Exception:
                    pass
            orig_fns.clear()
        for map_key in (
            "vjp_chain_cache",
            "pre_scale_by_pid",
            "ckpt_cast_caps_by_pid",
            "ckpt_slot_state_by_pid",
        ):
            m_obj = meta.get(map_key)
            if isinstance(m_obj, dict):
                m_obj.clear()
        nl_pids = meta.get("nonlinear_ckpt_pids")
        if isinstance(nl_pids, set):
            nl_pids.clear()
    _drain_teardown_anchors()


def _work_dtype_for(device: Optional[torch.device] = None) -> torch.dtype:
    if device is not None and device.type == "mps":
        return torch.float32
    return torch.float64


def _active_autocast_dtype(device: Optional[torch.device]) -> Optional[torch.dtype]:
    dev_type = device.type if device is not None else "cpu"
    try:
        if torch.is_autocast_enabled(dev_type):
            dt = torch.get_autocast_dtype(dev_type)
            if dt in _HALF_PREC_DTYPES:
                return dt
    except (RuntimeError, TypeError, AttributeError):
        pass
    return None


def _disable_autocast_ctx(device: Optional[torch.device]):
    dev_type = device.type if device is not None else "cpu"
    return torch.autocast(device_type=dev_type, enabled=False)


def _clamp_low_prec_grad(
    grad: torch.Tensor,
    orig_dtype: torch.dtype,
    fp16_max: float = _FP16_GRAD_MAX,
) -> torch.Tensor:
    if orig_dtype == torch.float16:
        lim = min(_FP16_GRAD_MAX, float(fp16_max))
        return grad.clamp(-lim, lim)
    if orig_dtype == torch.bfloat16:
        return grad.clamp(-_BF16_GRAD_MAX, _BF16_GRAD_MAX)
    return grad


_ELEMENTWISE_VJP_PREFIXES = (
    "ReluBackward",
    "AbsBackward",
    "ClampBackward",
    "LeakyReluBackward",
    "GeluBackward",
    "SiluBackward",
    "SigmoidBackward",
    "TanhBackward",
    "SinBackward",
    "CosBackward",
    "ExpBackward",
    "LogBackward",
    "NegBackward",
    "CloneBackward",
    "AliasBackward",
    "ContiguousBackward",
    "ToCopyBackward",
    "AddBackward",
    "SubBackward",
    "RsubBackward",
    "MulBackward",
    "DivBackward",
)


_MEMORY_ORDER_VJP_PREFIXES = _ELEMENTWISE_VJP_PREFIXES + (
    "ViewBackward",
    "UnsafeViewBackward",
    "ReshapeAliasBackward",
    "UnsqueezeBackward",
    "SqueezeBackward",
)
_MAX_VJP_NODES = 131072


_MAX_CHAIN_HOPS = 2_097_152

_SAVED_TENSOR_CLS = getattr(getattr(torch._C, "_autograd", None), "SavedTensor", ())
_PY_BWD_CLS = getattr(torch.autograd.function, "BackwardCFunction", None)
_RAW_SAVED_ATTR_CACHE: Dict[type, Tuple[str, ...]] = {}
_ISOMETRIC_OR_TERNARY_CHAIN_PREFIXES = (
    "ReluBackward",
    "ClampBackward",
    "NegBackward",
    "CloneBackward",
    "AliasBackward",
    "ContiguousBackward",
    "ToCopyBackward",
    "AddBackward",
    "SubBackward",
    "RsubBackward",
    "MulBackward",
    "DivBackward",
    "ViewBackward",
    "UnsafeViewBackward",
    "ReshapeAliasBackward",
    "UnsqueezeBackward",
    "SqueezeBackward",
)


def _raw_val_has_unpack_hook(raw_v: object) -> bool:
    if _SAVED_TENSOR_CLS and isinstance(raw_v, _SAVED_TENSOR_CLS):
        return getattr(raw_v, "unpack_hook", None) is not None
    if isinstance(raw_v, (tuple, list)):
        return any(_raw_val_has_unpack_hook(el) for el in raw_v)
    return False


def _node_has_saved_unpack_hook(node: object) -> bool:
    if node is None:
        return False
    cls = type(node)
    if cls.__name__ == "AccumulateGrad" or (_PY_BWD_CLS is not None and isinstance(node, _PY_BWD_CLS)):
        return False
    attrs = _RAW_SAVED_ATTR_CACHE.get(cls)
    if attrs is None:
        attrs = tuple(a for a in dir(cls) if a.startswith("_raw_saved_"))
        _RAW_SAVED_ATTR_CACHE[cls] = attrs
    for attr in attrs:
        try:
            raw_v = getattr(node, attr, None)
        except Exception:
            return True
        if _raw_val_has_unpack_hook(raw_v):
            return True
    return False


_NONFP16_LEAF_KEY_OFFSET = 1 << 61


def _fp16_leaf_bwd_key(var: torch.Tensor) -> int:
    return id(var) if var.dtype == torch.float16 else -(_NONFP16_LEAF_KEY_OFFSET + id(var))


def _is_one_shot_bwd_node(node: object) -> bool:
    if node is None:
        return False
    if _PY_BWD_CLS is not None and isinstance(node, _PY_BWD_CLS):
        return True
    return _node_has_saved_unpack_hook(node)


def _is_custom_py_bwd_node(node: object) -> bool:
    if node is None:
        return False
    return bool(
        _PY_BWD_CLS is not None
        and isinstance(node, _PY_BWD_CLS)
        and type(node).__name__ != "CheckpointFunctionBackward"
    )


def _is_linear_checkpoint_bwd_node(node: object) -> bool:
    if node is None:
        return False
    if type(node).__name__ == "CheckpointFunctionBackward":
        return True
    if _PY_BWD_CLS is not None and isinstance(node, _PY_BWD_CLS):
        return False
    return _node_has_saved_unpack_hook(node)


def _safe_saved_attr(node: object, saved_attr: str) -> Any:
    if node is None:
        return None
    cls = type(node)
    raw_attr = "_raw" + saved_attr
    if hasattr(cls, raw_attr):
        try:
            raw_v = getattr(node, raw_attr, None)
        except Exception:
            return None
        if _raw_val_has_unpack_hook(raw_v):
            return None
        if _SAVED_TENSOR_CLS and isinstance(raw_v, _SAVED_TENSOR_CLS):
            return raw_v.data
    return getattr(node, saved_attr, None)


def _collapsible_chain_step(node: object, slot_nr: int):
    if node is None or slot_nr != 0:
        return None
    if _is_one_shot_bwd_node(node):
        return None
    cname = type(node).__name__
    if (
        cname == "AccumulateGrad"
        or _is_fp16_stop_fn(cname)
        or _is_fp16_cast_leaf_boundary(node, cname)
    ):
        return None
    if not cname.startswith(_ISOMETRIC_OR_TERNARY_CHAIN_PREFIXES):
        return None
    in_metas = getattr(node, "_input_metadata", None)
    if not in_metas or len(in_metas) != 1:
        return None
    m0 = in_metas[0]
    sh0 = getattr(m0, "shape", None)
    dt0 = getattr(m0, "dtype", None)
    dev0 = getattr(m0, "device", None)
    if sh0 is None or dt0 is None or not dt0.is_floating_point:
        return None
    sh0_t = tuple(int(d) for d in sh0)
    n0 = math.prod(sh0_t)
    if n0 <= 0:
        return None
    active = [
        (ei, nf, int(onr))
        for ei, (nf, onr) in enumerate(node.next_functions)
        if nf is not None
    ]
    if len(active) != 1:
        return None
    ei, ch, ch_onr = active[0]
    ch_sh, ch_dt = _node_shape_dtype(ch, ch_onr)
    if ch_sh is None or ch_dt is None or not ch_dt.is_floating_point or ch_dt != dt0:
        return None
    if math.prod(int(d) for d in ch_sh) != n0:
        return None
    if cname.startswith(("AddBackward", "SubBackward", "RsubBackward", "MulBackward", "DivBackward")):
        if sh0_t != ch_sh:
            return None
        if cname.startswith("MulBackward"):
            c_val = _finite_scalar_const(
                _safe_saved_attr(node, "_saved_other" if ei == 0 else "_saved_self")
            )
            if c_val is None or abs(c_val) != 1.0:
                return None
        elif cname.startswith("DivBackward"):
            if ei != 0:
                return None
            c_val = _finite_scalar_const(_safe_saved_attr(node, "_saved_other"))
            if c_val is None or abs(c_val) != 1.0:
                return None
        else:
            raw_a = _safe_saved_attr(node, "_saved_alpha")
            a_val = 1.0 if raw_a is None else float(raw_a)
            need_edge = 0 if cname.startswith("RsubBackward") else 1
            eff_a = a_val if ei == need_edge else 1.0
            if not math.isfinite(eff_a) or abs(eff_a) != 1.0:
                return None
    return ei, ch, ch_onr, sh0_t, ch_sh, dt0, (dev0 or torch.device("cpu")), cname


def _collapse_vjp_mem_chain(
    start_fn: object,
    start_out_nr: int,
    chain_cache: Optional[Dict[Tuple[int, int], Tuple[object, int, torch.Tensor]]] = None,
    chain_lock: Optional[threading.Lock] = None,
    mile_node: Optional[object] = None,
    mile_dist: int = 0,
) -> Tuple[object, int, Optional[torch.Tensor]]:
    """
    Collapses non-amplifying linear chains (length >= 2) of C++ memory-order elementwise/view autograd nodes
    via a single bounded probe pass, returning `(tail_fn, tail_out_nr, diag_f32_or_None)`.
    """
    if start_fn is None or int(start_out_nr) != 0:
        return start_fn, int(start_out_nr), None

    key0 = (id(start_fn), int(start_out_nr))
    if chain_cache is not None:
        if chain_lock is not None:
            with chain_lock:
                hit0 = chain_cache.get(key0)
        else:
            hit0 = chain_cache.get(key0)
        if hit0 is not None:
            return hit0[0], hit0[1], hit0[2]

    def _collapse_from_anchor(
        anchor_fn: object,
        anchor_out_nr: int,
        anchor_mile: object,
        anchor_mdist: int,
    ) -> Tuple[object, int, Optional[torch.Tensor]]:
        key_a = (id(anchor_fn), int(anchor_out_nr))
        s0 = _collapsible_chain_step(anchor_fn, int(anchor_out_nr))
        if s0 is None:
            return anchor_fn, int(anchor_out_nr), None
        s1 = _collapsible_chain_step(s0[1], s0[2])
        s_hit0 = None
        if s1 is None and chain_cache is not None:
            sk_a = (id(s0[1]), int(s0[2]))
            if chain_lock is not None:
                with chain_lock:
                    s_hit0 = chain_cache.get(sk_a)
            else:
                s_hit0 = chain_cache.get(sk_a)
        if s1 is None and s_hit0 is None:
            return anchor_fn, int(anchor_out_nr), None

        sh_init = s0[3]
        dev_init = s0[6]
        u_f32 = torch.ones(sh_init, device=dev_init, dtype=torch.float32)
        scale_m = 0.5
        exp2_acc = 1
        cur: object = anchor_fn
        cur_onr = int(anchor_out_nr)
        tail_sh = sh_init
        mile = anchor_mile
        m_dist = int(anchor_mdist)
        hops = 0
        zero_short = False
        suffix_composed: Optional[Tuple[object, int, torch.Tensor]] = None

        try:
            with torch.no_grad(), _disable_autocast_ctx(dev_init):
                while hops < _MAX_CHAIN_HOPS:
                    if hops > 0 and chain_cache is not None:
                        ckey = (id(cur), int(cur_onr))
                        if chain_lock is not None:
                            with chain_lock:
                                c_hit = chain_cache.get(ckey)
                        else:
                            c_hit = chain_cache.get(ckey)
                        if c_hit is not None:
                            if m_dist > 0 and mile is not None:
                                _link_teardown_milestone(mile, c_hit[0], single_child=True)
                            suffix_composed = c_hit
                            break

                    info = _collapsible_chain_step(cur, cur_onr)
                    if info is None:
                        break
                    ei, ch, ch_onr, _, ch_sh, dt0, dev0, cname = info
                    if zero_short and not cname.startswith(("DivBackward", "LogBackward")):
                        mile, m_dist = _step_teardown_milestone(mile, m_dist, ch, single_child=True)
                        cur, cur_onr, tail_sh = ch, ch_onr, ch_sh
                        hops += 1
                        continue

                    u_max = float(u_f32.abs().max().item())
                    if u_max == 0.0 and not cname.startswith(("DivBackward", "LogBackward")):
                        zero_short = True
                        mile, m_dist = _step_teardown_milestone(mile, m_dist, ch, single_child=True)
                        cur, cur_onr, tail_sh = ch, ch_onr, ch_sh
                        hops += 1
                        continue
                    if u_max > 0.0 and (u_max > 64.0 or u_max < (1.0 / 64.0)):
                        u_f32 = u_f32 / u_max
                        m_f, e_f = math.frexp(u_max)
                        m_p, e_p = math.frexp(scale_m * m_f)
                        scale_m = m_p
                        exp2_acc += e_f + e_p

                    outs = cur(u_f32.to(device=dev0, dtype=dt0))
                    go = outs[ei] if isinstance(outs, tuple) else outs
                    if go is None or not bool(torch.isfinite(go).all()):
                        return anchor_fn, int(anchor_out_nr), None
                    go_f32 = go.detach().to(dtype=torch.float32)
                    if u_max > 0.0 and bool((go_f32.abs() > u_f32.reshape(go_f32.shape).abs() * (1.0 + 1e-5)).any()):
                        break
                    u_f32 = go_f32
                    mile, m_dist = _step_teardown_milestone(mile, m_dist, ch, single_child=True)
                    cur, cur_onr, tail_sh = ch, ch_onr, ch_sh
                    hops += 1
        except Exception:
            return anchor_fn, int(anchor_out_nr), None

        if suffix_composed is not None:
            c_tail, c_onr, c_diag = suffix_composed
            if abs(exp2_acc) > 120:
                return anchor_fn, int(anchor_out_nr), None
            factor = 0.0 if zero_short else (math.ldexp(scale_m, exp2_acc) if exp2_acc >= -145 else 0.0)
            diag_f32 = u_f32.reshape(c_diag.shape) * c_diag * factor
            cur, cur_onr = c_tail, int(c_onr)
        else:
            if hops < 2:
                return anchor_fn, int(anchor_out_nr), None
            if zero_short:
                diag_f32 = torch.zeros(tail_sh, device=dev_init, dtype=torch.float32)
            else:
                if exp2_acc > 120:
                    return anchor_fn, int(anchor_out_nr), None
                factor = math.ldexp(scale_m, exp2_acc) if exp2_acc >= -145 else 0.0
                diag_f32 = (u_f32 * factor).reshape(tail_sh)

        if not bool(torch.isfinite(diag_f32).all()):
            return anchor_fn, int(anchor_out_nr), None

        if chain_cache is not None:
            res_entry = (cur, int(cur_onr), diag_f32)
            if chain_lock is not None:
                with chain_lock:
                    chain_cache[key_a] = res_entry
            else:
                chain_cache[key_a] = res_entry
        return cur, int(cur_onr), diag_f32

    step0 = _collapsible_chain_step(start_fn, 0)
    if step0 is None:
        return start_fn, int(start_out_nr), None
    base_mile = mile_node if mile_node is not None else start_fn
    base_mdist = int(mile_dist) if mile_node is not None else 0

    if chain_cache is not None:
        # Pre-cache the suffix 8 hops downstream so shared deep chains behind per-branch wrappers are evaluated once.
        w_fn, w_onr = step0[1], step0[2]
        w_mile, w_dist = _step_teardown_milestone(base_mile, base_mdist, w_fn, single_child=True)
        for k_ahead in range(1, 9):
            w_key = (id(w_fn), int(w_onr))
            if chain_lock is not None:
                with chain_lock:
                    w_hit = chain_cache.get(w_key)
            else:
                w_hit = chain_cache.get(w_key)
            if w_hit is not None:
                break
            w_info = _collapsible_chain_step(w_fn, int(w_onr))
            if w_info is None:
                break
            if k_ahead == 8:
                if _collapsible_chain_step(w_info[1], w_info[2]) is not None:
                    _collapse_from_anchor(w_fn, int(w_onr), w_mile, w_dist)
                break
            w_fn, w_onr = w_info[1], w_info[2]
            w_mile, w_dist = _step_teardown_milestone(w_mile, w_dist, w_fn, single_child=True)

    return _collapse_from_anchor(start_fn, int(start_out_nr), base_mile, base_mdist)


def _eval_fp16_vjp_to_roots(
    start_fn: object,
    start_out_nr: int,
    g_f32: torch.Tensor,
    chain_cache: Optional[Dict[Tuple[int, int], Tuple[object, int, torch.Tensor]]] = None,
    chain_lock: Optional[threading.Lock] = None,
) -> Tuple[
    Optional[Dict[int, torch.Tensor]],
    bool,
    Optional[Callable[[torch.Tensor], Optional[Dict[int, torch.Tensor]]]],
    List[object],
    Optional[Callable[[Dict[int, torch.Tensor]], Optional[torch.Tensor]]],
]:
    """
    Evaluates exact float32 VJP of `g_f32` through C++ autograd nodes back to FP16 roots,
    intermediate FP16 input slots, or one-shot backward boundaries without invoking custom
    Python `backward()` or unpacking saved-tensor hooks eagerly.
    Returns `(root_deltas_or_None, is_mem_order_bijection, vjp_eval_fn_or_None, py_fn_nodes, alpha_pullback_fn_or_None)`.
    """
    if start_fn is None:
        return None, False, None, [], None
    g_numel = int(g_f32.numel())
    max_abs = float(g_f32.abs().max().item()) if g_numel > 0 else 0.0
    if max_abs == 0.0:
        return {}, True, None, [], None

    eff_start_fn = start_fn
    eff_out_nr = int(start_out_nr)

    nodes: Dict[int, object] = {id(eff_start_fn): eff_start_fn}
    indeg: Dict[int, int] = {id(eff_start_fn): 0}
    vjp_mile: Dict[int, Tuple[int, object]] = {id(eff_start_fn): (0, eff_start_fn)}
    edge_diags: Dict[Tuple[int, int], Tuple[object, int, torch.Tensor]] = {}
    py_nodes_by_id: Dict[int, object] = {}
    queue: List[object] = [eff_start_fn]
    qi = 0
    is_mem_order = True
    while qi < len(queue):
        cur = queue[qi]
        qi += 1
        cid = id(cur)
        d_cur, m_cur = vjp_mile.get(cid, (0, cur))
        cname = type(cur).__name__
        if cname == "AccumulateGrad" or _is_fp16_stop_fn(cname):
            continue
        is_py_fn = _is_one_shot_bwd_node(cur)
        if is_py_fn:
            py_nodes_by_id[cid] = cur
            in_m_py = getattr(cur, "_input_metadata", None)
            if (
                len(py_nodes_by_id) > 1
                or not in_m_py
                or len(in_m_py) != 1
                or getattr(in_m_py[0], "shape", None) is None
                or math.prod(int(d) for d in in_m_py[0].shape) != g_numel
            ):
                is_mem_order = False
            continue
        if is_mem_order:
            if not cname.startswith(_MEMORY_ORDER_VJP_PREFIXES):
                is_mem_order = False
            else:
                in_metas_chk = getattr(cur, "_input_metadata", None)
                if not in_metas_chk:
                    is_mem_order = False
                else:
                    for m_chk in in_metas_chk:
                        sh_m = getattr(m_chk, "shape", None)
                        if sh_m is None or math.prod(int(d) for d in sh_m) != g_numel:
                            is_mem_order = False
                            break

        for edge_idx, (nf_raw, onr_raw) in enumerate(cur.next_functions):
            if nf_raw is None:
                continue
            nf, onr, e_diag = _collapse_vjp_mem_chain(
                nf_raw,
                int(onr_raw),
                chain_cache=chain_cache,
                chain_lock=chain_lock,
                mile_node=m_cur,
                mile_dist=d_cur,
            )
            if e_diag is not None:
                edge_diags[(cid, edge_idx)] = (nf, onr, e_diag)
            if is_mem_order:
                ch_sh, _ = _node_shape_dtype(nf, int(onr))
                if ch_sh is None or math.prod(int(d) for d in ch_sh) != g_numel:
                    is_mem_order = False
            nid = id(nf)
            indeg[nid] = indeg.get(nid, 0) + 1
            if nid not in nodes:
                if len(nodes) >= _MAX_VJP_NODES:
                    return None, False, None, [], None
                nodes[nid] = nf
                m_nf, d_nf = _step_teardown_milestone(m_cur, d_cur, nf)
                vjp_mile[nid] = (d_nf, m_nf)
                queue.append(nf)

    # Precompute static topological order so VJP evaluation and O(1) pullback reuse it.
    indeg_rem = dict(indeg)
    topo_order: List[object] = [eff_start_fn] if indeg_rem.get(id(eff_start_fn), 0) == 0 else []
    ti = 0
    while ti < len(topo_order):
        cur = topo_order[ti]
        ti += 1
        cid = id(cur)
        cname = type(cur).__name__
        if (
            cname == "AccumulateGrad"
            or _is_fp16_stop_fn(cname)
            or _is_one_shot_bwd_node(cur)
        ):
            continue
        for edge_idx, (nf_raw, _) in enumerate(cur.next_functions):
            if nf_raw is None:
                continue
            ed = edge_diags.get((cid, edge_idx))
            nf = ed[0] if ed is not None else nf_raw
            nid = id(nf)
            if nid not in nodes:
                continue
            indeg_rem[nid] -= 1
            if indeg_rem[nid] == 0:
                topo_order.append(nf)

    if len(topo_order) != len(nodes):
        return None, False, None, [], None

    def _run_topo(vec_f32: torch.Tensor) -> Optional[Dict[int, torch.Tensor]]:
        if vec_f32.numel() == 0:
            return {}
        v_max = float(vec_f32.abs().max().item())
        if v_max == 0.0:
            return {}
        probe_scale = max(v_max, 1e-12)
        init_vec = vec_f32.detach() / probe_scale
        slots: Dict[int, Dict[int, torch.Tensor]] = {
            id(eff_start_fn): {int(eff_out_nr): init_vec}
        }
        r_deltas: Dict[int, torch.Tensor] = {}
        try:
            with torch.no_grad(), _disable_autocast_ctx(vec_f32.device):
                for cur in topo_order:
                    cid = id(cur)
                    cname = type(cur).__name__
                    c_slots = slots.pop(cid, None)
                    if cname == "AccumulateGrad":
                        var = getattr(cur, "variable", None)
                        if isinstance(var, torch.Tensor) and c_slots and 0 in c_slots:
                            rid = _fp16_leaf_bwd_key(var)
                            d_f32 = c_slots[0].to(dtype=torch.float32) * probe_scale
                            r_deltas[rid] = d_f32 if rid not in r_deltas else (r_deltas[rid] + d_f32)
                        continue
                    if _is_one_shot_bwd_node(cur):
                        if c_slots:
                            for si, sv in c_slots.items():
                                key_py = -(cid * 1024 + int(si) + 16)
                                d_f32 = sv.to(dtype=torch.float32) * probe_scale
                                r_deltas[key_py] = d_f32 if key_py not in r_deltas else (r_deltas[key_py] + d_f32)
                        continue
                    if _is_fp16_stop_fn(cname):
                        if c_slots:
                            sv0 = c_slots[0] if 0 in c_slots else next(iter(c_slots.values()))
                            d_f32 = sv0.to(dtype=torch.float32) * probe_scale
                            r_deltas[cid] = d_f32 if cid not in r_deltas else (r_deltas[cid] + d_f32)
                        continue
                    is_cast_leaf = _is_fp16_cast_leaf_boundary(cur, cname)
                    if is_cast_leaf and c_slots:
                        sv0 = c_slots[0] if 0 in c_slots else next(iter(c_slots.values()))
                        d_f32 = sv0.to(dtype=torch.float32) * probe_scale
                        r_deltas[cid] = d_f32 if cid not in r_deltas else (r_deltas[cid] + d_f32)
                    if not c_slots:
                        continue
                    in_metas = getattr(cur, "_input_metadata", None)
                    for si, sv in c_slots.items():
                        m_chk = in_metas[si] if (in_metas and si < len(in_metas)) else None
                        if (
                            getattr(m_chk, "dtype", None) == torch.float16
                            and not (is_cast_leaf and int(si) == 0)
                            and _collapsible_chain_step(cur, int(si)) is None
                        ):
                            inter_key = -(cid * 1024 + int(si) + 512)
                            d_i32 = sv.to(dtype=torch.float32) * probe_scale
                            r_deltas[inter_key] = (
                                d_i32 if inter_key not in r_deltas else (r_deltas[inter_key] + d_i32)
                            )
                    n_in = max(len(in_metas) if in_metas else 0, max(c_slots.keys()) + 1)
                    call_args: List[Optional[torch.Tensor]] = []
                    for si in range(n_in):
                        sv = c_slots.get(si)
                        m_si = in_metas[si] if (in_metas and si < len(in_metas)) else None
                        dt_si = getattr(m_si, "dtype", None) or torch.float16
                        dev_si = getattr(m_si, "device", None) or vec_f32.device
                        if sv is not None:
                            call_args.append(sv.to(device=dev_si, dtype=dt_si))
                        elif m_si is not None and getattr(m_si, "shape", None) is not None:
                            z_sh = tuple(int(d) for d in m_si.shape)
                            call_args.append(torch.zeros(z_sh, device=dev_si, dtype=dt_si))
                        else:
                            call_args.append(None)
                    outs = cur(*call_args)
                    if not isinstance(outs, tuple):
                        outs = (outs,)
                    for edge_idx, ((nf_raw, out_nr_raw), go) in enumerate(zip(cur.next_functions, outs)):
                        if nf_raw is None or go is None:
                            continue
                        ed = edge_diags.get((cid, edge_idx))
                        if ed is not None:
                            nf_t, onr, e_diag = ed
                            nid = id(nf_t)
                            if nid not in nodes:
                                continue
                            go32 = go.detach().to(dtype=torch.float32).reshape(e_diag.shape) * e_diag
                        else:
                            nid = id(nf_raw)
                            if nid not in nodes:
                                continue
                            onr = int(out_nr_raw)
                            go32 = go.detach().to(dtype=torch.float32)
                        nf_slots = slots.setdefault(nid, {})
                        nf_slots[onr] = go32 if onr not in nf_slots else (nf_slots[onr] + go32)
        except Exception:
            return None

        for d_v in r_deltas.values():
            if not bool(torch.isfinite(d_v).all()):
                return None
        return r_deltas

    def _pullback_alpha_graph(alpha_by_rid: Dict[int, torch.Tensor]) -> Optional[torch.Tensor]:
        dev = g_f32.device
        node_alpha: Dict[Tuple[int, int], torch.Tensor] = {}

        def _get_edge_in_alpha(edge_idx: int, cid_cur: int, nf_raw: object, onr_raw: int) -> Optional[torch.Tensor]:
            ed = edge_diags.get((cid_cur, edge_idx))
            if ed is not None:
                nf_t, onr_t, e_diag = ed
                ch_a = node_alpha.get((id(nf_t), int(onr_t)))
                raw_sh, _ = _node_shape_dtype(nf_raw, int(onr_raw))
                tgt_sh = raw_sh if raw_sh is not None else tuple(int(d) for d in e_diag.shape)
                if ch_a is None:
                    return torch.ones(tgt_sh, device=dev, dtype=torch.float32)
                ch_loc = ch_a.to(device=dev, dtype=torch.float32).reshape(e_diag.shape)
                eff_a = torch.where(e_diag.to(device=dev).abs() > 0.0, ch_loc, torch.ones_like(ch_loc))
                return eff_a.reshape(tgt_sh)
            if nf_raw is None:
                return None
            ch_a = node_alpha.get((id(nf_raw), int(onr_raw)))
            if ch_a is not None:
                return ch_a.to(device=dev, dtype=torch.float32)
            raw_sh, _ = _node_shape_dtype(nf_raw, int(onr_raw))
            if raw_sh is not None:
                return torch.ones(raw_sh, device=dev, dtype=torch.float32)
            return None

        try:
            with torch.no_grad():
                for cur in reversed(topo_order):
                    cid = id(cur)
                    cname = type(cur).__name__
                    if cname == "AccumulateGrad":
                        var = getattr(cur, "variable", None)
                        if isinstance(var, torch.Tensor):
                            av = alpha_by_rid.get(_fp16_leaf_bwd_key(var))
                            if av is not None:
                                node_alpha[(cid, 0)] = av.to(device=dev, dtype=torch.float32).reshape(tuple(var.shape))
                        continue
                    if _is_one_shot_bwd_node(cur):
                        in_metas = getattr(cur, "_input_metadata", None)
                        n_py = len(in_metas) if in_metas else 1
                        for si in range(max(1, n_py)):
                            av = alpha_by_rid.get(-(cid * 1024 + si + 16))
                            if av is not None:
                                node_alpha[(cid, si)] = av.to(device=dev, dtype=torch.float32)
                        continue
                    if _is_fp16_stop_fn(cname):
                        av = alpha_by_rid.get(cid)
                        if av is not None:
                            node_alpha[(cid, 0)] = av.to(device=dev, dtype=torch.float32)
                        continue

                    in_metas = getattr(cur, "_input_metadata", None)
                    sh0_raw = getattr(in_metas[0], "shape", None) if in_metas else None
                    sh0 = tuple(int(d) for d in sh0_raw) if sh0_raw is not None else None
                    edge_alphas: List[Optional[torch.Tensor]] = [
                        _get_edge_in_alpha(ei, cid, nf_r, int(onr_r))
                        for ei, (nf_r, onr_r) in enumerate(cur.next_functions)
                    ]
                    active_ea = [ea for ei, (nf_r, _), ea in zip(range(len(edge_alphas)), cur.next_functions, edge_alphas) if nf_r is not None]
                    if any(ea is None for ea in active_ea):
                        return None
                    in_a0 = active_ea[0] if len(active_ea) == 1 else (edge_alphas[0] if edge_alphas else None)

                    if cname.startswith(("SplitWithSizesBackward", "SplitBackward", "UnbindBackward")):
                        if in_a0 is None:
                            return None
                        dim_v = _safe_saved_attr(cur, "_saved_dim")
                        if dim_v is None:
                            return None
                        d_i = int(dim_v)
                        if cname.startswith("SplitWithSizesBackward"):
                            sp_sz = _safe_saved_attr(cur, "_saved_split_sizes")
                            if sp_sz is None:
                                return None
                            chunks = torch.split(in_a0, [int(s) for s in sp_sz], dim=d_i)
                        elif cname.startswith("SplitBackward"):
                            sp_s = _safe_saved_attr(cur, "_saved_split_size")
                            if sp_s is None:
                                return None
                            chunks = torch.split(in_a0, int(sp_s), dim=d_i)
                        else:
                            chunks = torch.unbind(in_a0, dim=d_i)
                        for si, ch_t in enumerate(chunks):
                            node_alpha[(cid, si)] = ch_t
                    elif cname.startswith("CatBackward"):
                        if sh0 is None:
                            return None
                        dim_v = _safe_saved_attr(cur, "_saved_dim")
                        if dim_v is None:
                            return None
                        d_i = int(dim_v) % max(1, len(sh0))
                        none_idxs = [ei for ei, (nf_r, _) in enumerate(cur.next_functions) if nf_r is None]
                        if len(none_idxs) > 1:
                            return None
                        parts: List[Optional[torch.Tensor]] = list(edge_alphas)
                        if len(none_idxs) == 1:
                            known_w = sum(int(p.shape[d_i]) for p in parts if p is not None)
                            rem_w = int(sh0[d_i]) - known_w
                            if rem_w < 0:
                                return None
                            m_sh = list(sh0)
                            m_sh[d_i] = rem_w
                            parts[none_idxs[0]] = torch.ones(tuple(m_sh), device=dev, dtype=torch.float32)
                        if any(p is None for p in parts):
                            return None
                        node_alpha[(cid, 0)] = torch.cat([p for p in parts if p is not None], dim=d_i)
                    elif cname.startswith("StackBackward"):
                        if sh0 is None:
                            return None
                        dim_v = _safe_saved_attr(cur, "_saved_dim")
                        if dim_v is None:
                            return None
                        d_i = int(dim_v) % max(1, len(sh0))
                        item_sh = tuple(int(sz) for ax, sz in enumerate(sh0) if ax != d_i)
                        parts_st = [
                            p if p is not None else torch.ones(item_sh, device=dev, dtype=torch.float32)
                            for p in edge_alphas
                        ]
                        node_alpha[(cid, 0)] = torch.stack(parts_st, dim=d_i)
                    elif cname.startswith("ExpandBackward"):
                        if in_a0 is None or sh0 is None:
                            return None
                        node_alpha[(cid, 0)] = torch.broadcast_to(in_a0, sh0).clone()
                    elif cname.startswith("RepeatBackward"):
                        if in_a0 is None:
                            return None
                        reps = _safe_saved_attr(cur, "_saved_repeats")
                        if reps is None:
                            return None
                        r_tup = tuple(int(r) for r in reps)
                        src_a = in_a0
                        while src_a.ndim < len(r_tup):
                            src_a = src_a.unsqueeze(0)
                        node_alpha[(cid, 0)] = src_a.repeat(*r_tup)
                    elif cname.startswith("SliceBackward"):
                        if in_a0 is None:
                            return None
                        d_v = _safe_saved_attr(cur, "_saved_dim")
                        s_v = _safe_saved_attr(cur, "_saved_start")
                        e_v = _safe_saved_attr(cur, "_saved_end")
                        st_v = _safe_saved_attr(cur, "_saved_step")
                        if None in (d_v, s_v, e_v, st_v):
                            return None
                        node_alpha[(cid, 0)] = torch.ops.aten.slice.Tensor(
                            in_a0, int(d_v), int(s_v), int(e_v), int(st_v)
                        )
                    elif cname.startswith("SelectBackward"):
                        if in_a0 is None:
                            return None
                        d_v = _safe_saved_attr(cur, "_saved_dim")
                        i_v = _safe_saved_attr(cur, "_saved_index")
                        if d_v is None or i_v is None:
                            return None
                        node_alpha[(cid, 0)] = torch.ops.aten.select.int(in_a0, int(d_v), int(i_v))
                    elif cname.startswith("IndexBackward"):
                        if in_a0 is None:
                            return None
                        raw_idxs = _safe_saved_attr(cur, "_saved_indices")
                        if not isinstance(raw_idxs, (tuple, list)):
                            return None
                        node_alpha[(cid, 0)] = torch.ops.aten.index.Tensor(
                            in_a0, [None if t is None else t.to(device=dev) for t in raw_idxs]
                        )
                    elif cname.startswith("IndexSelectBackward"):
                        if in_a0 is None:
                            return None
                        d_v = _safe_saved_attr(cur, "_saved_dim")
                        idx_t = _safe_saved_attr(cur, "_saved_index")
                        if d_v is None or not isinstance(idx_t, torch.Tensor):
                            return None
                        node_alpha[(cid, 0)] = torch.index_select(
                            in_a0, int(d_v), idx_t.to(device=dev, dtype=torch.long)
                        )
                    elif cname.startswith("GatherBackward"):
                        if in_a0 is None:
                            return None
                        d_v = _safe_saved_attr(cur, "_saved_dim")
                        idx_t = _safe_saved_attr(cur, "_saved_index")
                        if d_v is None or not isinstance(idx_t, torch.Tensor):
                            return None
                        node_alpha[(cid, 0)] = torch.gather(
                            in_a0, int(d_v), idx_t.to(device=dev, dtype=torch.long)
                        )
                    elif cname.startswith("PermuteBackward"):
                        if in_a0 is None:
                            return None
                        dims_v = _safe_saved_attr(cur, "_saved_dims")
                        if dims_v is None:
                            return None
                        node_alpha[(cid, 0)] = in_a0.permute(*[int(d) for d in dims_v])
                    elif cname.startswith("TransposeBackward"):
                        if in_a0 is None:
                            return None
                        d0_v = _safe_saved_attr(cur, "_saved_dim0")
                        d1_v = _safe_saved_attr(cur, "_saved_dim1")
                        if d0_v is None or d1_v is None:
                            return None
                        node_alpha[(cid, 0)] = in_a0.transpose(int(d0_v), int(d1_v))
                    elif cname.startswith("TBackward"):
                        if in_a0 is None:
                            return None
                        node_alpha[(cid, 0)] = in_a0.t()
                    elif cname.startswith("FlipBackward"):
                        if in_a0 is None:
                            return None
                        dims_v = _safe_saved_attr(cur, "_saved_dims")
                        if dims_v is None:
                            return None
                        node_alpha[(cid, 0)] = in_a0.flip(dims=[int(d) for d in dims_v])
                    elif cname.startswith(("SumBackward", "MeanBackward")):
                        if in_a0 is None or sh0 is None:
                            return None
                        if math.prod(sh0) == 1:
                            node_alpha[(cid, 0)] = in_a0.amin().reshape(sh0)
                        else:
                            dims_v = _safe_saved_attr(cur, "_saved_dim")
                            if dims_v is None:
                                node_alpha[(cid, 0)] = in_a0.amin().reshape(sh0)
                            else:
                                kdim_v = bool(_safe_saved_attr(cur, "_saved_keepdim"))
                                d_tup = tuple(int(d) for d in dims_v)
                                node_alpha[(cid, 0)] = torch.amin(in_a0, dim=d_tup, keepdim=kdim_v).reshape(sh0)
                    elif cname.startswith(_MEMORY_ORDER_VJP_PREFIXES):
                        if sh0 is None or not active_ea:
                            return None
                        if len(active_ea) == 1:
                            a0 = active_ea[0]
                            assert a0 is not None
                            if int(a0.numel()) == math.prod(sh0):
                                node_alpha[(cid, 0)] = a0.reshape(sh0)
                            else:
                                node_alpha[(cid, 0)] = torch.broadcast_to(a0, sh0).clone()
                        else:
                            acc_a: Optional[torch.Tensor] = None
                            for ea in active_ea:
                                assert ea is not None
                                b_a = ea.reshape(sh0) if int(ea.numel()) == math.prod(sh0) else torch.broadcast_to(ea, sh0)
                                acc_a = b_a.clone() if acc_a is None else torch.minimum(acc_a, b_a)
                            assert acc_a is not None
                            node_alpha[(cid, 0)] = acc_a
                    else:
                        if sh0 is not None and all(
                            ea is not None and bool((ea >= 1.0 - 1e-6).all()) for ea in active_ea
                        ):
                            node_alpha[(cid, 0)] = torch.ones(sh0, device=dev, dtype=torch.float32)
                        else:
                            return None

                    if _is_fp16_cast_leaf_boundary(cur, cname) and cid in alpha_by_rid:
                        a_cast = alpha_by_rid[cid].to(device=dev, dtype=torch.float32)
                        prev_0 = node_alpha.get((cid, 0))
                        node_alpha[(cid, 0)] = (
                            a_cast
                            if prev_0 is None
                            else torch.minimum(prev_0, a_cast.reshape(prev_0.shape))
                        )
                    n_out_slots = len(in_metas) if in_metas else 1
                    for si in range(max(1, n_out_slots)):
                        inter_key = -(cid * 1024 + si + 512)
                        if inter_key in alpha_by_rid:
                            a_int = alpha_by_rid[inter_key].to(device=dev, dtype=torch.float32)
                            prev_s = node_alpha.get((cid, si))
                            node_alpha[(cid, si)] = (
                                a_int
                                if prev_s is None
                                else torch.minimum(prev_s, a_int.reshape(prev_s.shape))
                            )
        except Exception:
            return None

        root_a = node_alpha.get((id(eff_start_fn), int(eff_out_nr)))
        if root_a is None or int(root_a.numel()) != g_numel:
            return None
        return root_a.reshape(g_f32.shape).clamp(0.0, 1.0)

    root_deltas = _run_topo(g_f32)
    if root_deltas is None:
        return None, False, None, [], None
    if is_mem_order and len(root_deltas) == 1:
        only_d = next(iter(root_deltas.values()))
        is_mem_order = int(only_d.numel()) == g_numel
    else:
        is_mem_order = False
    return root_deltas, is_mem_order, _run_topo, list(py_nodes_by_id.values()), _pullback_alpha_graph


def _align_bwd_state_tensor(
    prev: Optional[torch.Tensor],
    ref: torch.Tensor,
) -> Optional[torch.Tensor]:
    if prev is None:
        return torch.zeros_like(ref)
    if prev.device != ref.device:
        prev = prev.to(device=ref.device)
    if prev.shape == ref.shape:
        return prev
    if prev.numel() == ref.numel():
        return prev.reshape(ref.shape)
    return None


def _apply_telescoping_deltas(
    bwd_state: Dict[int, Dict[str, object]],
    root_deltas: Dict[int, torch.Tensor],
    g_f32: torch.Tensor,
    fp16_lim: float,
    elementwise_single: bool,
    vjp_eval_fn: Optional[Callable[[torch.Tensor], Optional[Dict[int, torch.Tensor]]]] = None,
    alpha_pullback_fn: Optional[Callable[[Dict[int, torch.Tensor]], Optional[torch.Tensor]]] = None,
    max_emit_abs: Optional[float] = None,
) -> Optional[Tuple[torch.Tensor, Union[torch.Tensor, float]]]:
    """
    Updates per-root `(target, emitted)` running accumulators so canceling forward/backward
    contributions telescope back cleanly to ~0 while emitted gradients never exceed +/-eff_max.
    Returns `(g_emit_detached_f32, scale_factor_detached)`.
    """
    phys_cap: Optional[float] = min(float(fp16_lim), float(max_emit_abs)) if max_emit_abs is not None else None
    g_abs_t = g_f32.abs() if phys_cap is not None else None
    phys_exceeded = bool((g_abs_t > phys_cap).any().item()) if (phys_cap is not None and g_abs_t is not None and g_f32.numel() > 0) else False
    alpha_phys_g: Optional[torch.Tensor] = (
        torch.where(
            g_abs_t > phys_cap,  # type: ignore[operator]
            float(phys_cap) / g_abs_t.clamp(min=1e-12),  # type: ignore[arg-type, union-attr]
            torch.ones_like(g_f32),
        )
        if phys_exceeded and phys_cap is not None
        else None
    )

    def _entry_in_box_mask(
        e_prev_t: torch.Tensor, delta_t: torch.Tensor, eff_m: float, e_goal_t: torch.Tensor
    ) -> torch.Tensor:
        e_nf = e_prev_t + delta_t
        tol = 1e-3 * torch.maximum(torch.ones_like(e_goal_t), e_goal_t.abs())
        return (e_prev_t.abs() <= eff_m) & (e_nf.abs() <= eff_m) & ((e_nf - e_goal_t).abs() <= tol)

    def _compute_entry_alpha(
        e_prev_t: torch.Tensor, delta_t: torch.Tensor, desired_t: torch.Tensor, eff_m: float, e_goal_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_box = _entry_in_box_mask(e_prev_t, delta_t, eff_m, e_goal_t)
        if bool(in_box.all()):
            return in_box, torch.ones_like(delta_t)
        nz = delta_t.abs() > 0.0
        safe_d = torch.where(nz, delta_t, torch.ones_like(delta_t))
        pos_d = delta_t > 0.0
        cap_room = torch.where(
            pos_d,
            torch.clamp(eff_m - e_prev_t, min=0.0) / safe_d,
            torch.clamp(-eff_m - e_prev_t, max=0.0) / safe_d,
        )
        targ_ratio = torch.where(
            pos_d,
            torch.clamp(desired_t, min=0.0) / safe_d,
            torch.clamp(desired_t, max=0.0) / safe_d,
        )
        alpha_t = torch.where(
            in_box | (~nz),
            torch.ones_like(delta_t),
            torch.minimum(targ_ratio, cap_room).clamp(0.0, 1.0),
        )
        return in_box, alpha_t

    entries: List[
        Tuple[int, Dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]
    ] = []
    for rid_k, delta_r in root_deltas.items():
        ent = bwd_state.get(rid_k)
        if ent is None:
            ent = {"target": None, "emitted": None, "abs_used": 0.0}
            bwd_state[rid_k] = ent
        t_prev = _align_bwd_state_tensor(ent.get("target"), delta_r)  # type: ignore[arg-type]
        e_prev = _align_bwd_state_tensor(ent.get("emitted"), delta_r)  # type: ignore[arg-type]
        if t_prev is None or e_prev is None:
            return None
        default_cap = 1e30 if int(rid_k) <= -_NONFP16_LEAF_KEY_OFFSET else _FP16_BWD_ROOT_MAX
        base_cap = float(ent.get("cap", default_cap))
        eff_max = max(0.0, base_cap - float(ent["abs_used"]))
        t_new = t_prev + delta_r
        cancel_mask = ((t_prev * delta_r) < 0.0) & (
            t_new.abs() <= 1.5e-3 * torch.minimum(t_prev.abs(), delta_r.abs())
        )
        if bool(cancel_mask.any().item()):
            t_new = torch.where(cancel_mask, torch.zeros_like(t_new), t_new)
        e_goal = t_new.clamp(-eff_max, eff_max)
        entries.append((rid_k, ent, delta_r, t_new, e_prev, e_goal - e_prev, eff_max, e_goal))

    if not entries:
        if alpha_phys_g is not None:
            return (g_f32 * alpha_phys_g).clamp(-fp16_lim, fp16_lim), alpha_phys_g.detach()
        return g_f32.clamp(-fp16_lim, fp16_lim), 1.0

    # Fast Path 1: single root with 1-to-1 memory-order bijection (same-shape OR view/reshape/squeeze/unsqueeze).
    if elementwise_single and len(entries) == 1 and entries[0][2].numel() == g_f32.numel():
        _, ent0, delta_r, t_new, e_prev, desired_inc, eff_max, e_goal = entries[0]
        in_box, alpha_r = _compute_entry_alpha(e_prev, delta_r, desired_inc, eff_max, e_goal)
        if bool(in_box.all()) and alpha_phys_g is None:
            ent0["target"] = t_new
            ent0["emitted"] = e_prev + delta_r
            return g_f32.clamp(-fp16_lim, fp16_lim), 1.0
        alpha_g = alpha_r.reshape(g_f32.shape).detach()
        if alpha_phys_g is not None:
            alpha_g = torch.minimum(alpha_g, alpha_phys_g)
            alpha_r = alpha_g.reshape(delta_r.shape)
        g_emit = (g_f32 * alpha_g).clamp(-fp16_lim, fp16_lim)
        ent0["target"] = t_new
        e_next = e_prev + alpha_r * delta_r
        if bool((e_prev.abs() <= eff_max + 1e-3).all()):
            e_next = e_next.clamp(-eff_max, eff_max)
        ent0["emitted"] = e_next
        return g_emit, alpha_g

    alpha_entries: List[
        Tuple[int, Dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]
    ] = []
    all_in_box = True
    for rid_k, ent, delta_r, t_new, e_prev, desired_inc, eff_max, e_goal in entries:
        in_box_k, alpha_k = _compute_entry_alpha(e_prev, delta_r, desired_inc, eff_max, e_goal)
        if not bool(in_box_k.all()):
            all_in_box = False
        alpha_entries.append((rid_k, ent, delta_r, t_new, e_prev, eff_max, in_box_k, alpha_k))

    if all_in_box and alpha_phys_g is None:
        for _, ent, delta_r, t_new, e_prev, _, _, _ in alpha_entries:
            ent["target"] = t_new
            ent["emitted"] = e_prev + delta_r
        return g_f32.clamp(-fp16_lim, fp16_lim), 1.0

    # Fast Path 2: exact O(1) graph pullback + per-coordinate / disjoint-slice pullback via static VJP evaluator.
    g_numel = int(g_f32.numel())
    if vjp_eval_fn is not None and 1 < g_numel <= 1_048_576:
        def _try_accept_alpha_g(
            cand_ag: torch.Tensor, allow_upstream_throttle: bool = True
        ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            cand_cl = cand_ag.detach().clamp(0.0, 1.0)
            if alpha_phys_g is not None:
                cand_cl = torch.minimum(cand_cl, alpha_phys_g)
                allow_upstream_throttle = True
            chk = vjp_eval_fn(g_f32 * cand_cl)
            if chk is None:
                return None
            proposed_emitted: List[Tuple[Dict[str, object], torch.Tensor, torch.Tensor]] = []
            for rid_k, ent, delta_r, t_new, e_prev, eff_max, _, alpha_k in alpha_entries:
                chk_d = chk.get(rid_k)
                if chk_d is None:
                    chk_d = torch.zeros_like(delta_r)
                if chk_d.shape != delta_r.shape:
                    return None
                want_d = alpha_k * delta_r
                tol_d = 1e-2 * torch.maximum(torch.ones_like(want_d), delta_r.abs())
                if allow_upstream_throttle:
                    pos_m = delta_r > 0.0
                    neg_m = delta_r < 0.0
                    ok_dir = torch.where(
                        pos_m,
                        (chk_d >= -tol_d) & (chk_d <= want_d + tol_d),
                        torch.where(
                            neg_m,
                            (chk_d <= tol_d) & (chk_d >= want_d - tol_d),
                            chk_d.abs() <= tol_d,
                        ),
                    )
                    if not bool(ok_dir.all()):
                        return None
                else:
                    if not bool(((chk_d - want_d).abs() <= tol_d).all()):
                        return None
                e_next = e_prev + chk_d
                if bool((e_prev.abs() <= eff_max + 1e-3).all()):
                    if not bool((e_next.abs() <= eff_max + max(1e-1, 1e-2 * eff_max)).all()):
                        return None
                    e_next = e_next.clamp(-eff_max, eff_max)
                proposed_emitted.append((ent, t_new, e_next))
            for ent, t_new, e_next in proposed_emitted:
                ent["target"] = t_new
                ent["emitted"] = e_next
            return (g_f32 * cand_cl).clamp(-fp16_lim, fp16_lim), cand_cl

        if alpha_pullback_fn is not None:
            cand_pb = alpha_pullback_fn({rid_k: alpha_k for rid_k, _, _, _, _, _, _, alpha_k in alpha_entries})
            if cand_pb is not None and cand_pb.shape == g_f32.shape:
                accepted = _try_accept_alpha_g(cand_pb, allow_upstream_throttle=True)
                if accepted is not None:
                    return accepted

        if len(alpha_entries) == 1:
            alpha0 = alpha_entries[0][7]
            if alpha0.numel() == g_numel:
                accepted = _try_accept_alpha_g(alpha0.reshape(g_f32.shape), allow_upstream_throttle=False)
                if accepted is not None:
                    return accepted
            else:
                try:
                    accepted = _try_accept_alpha_g(
                        torch.broadcast_to(alpha0, g_f32.shape).clone(), allow_upstream_throttle=False
                    )
                    if accepted is not None:
                        return accepted
                except RuntimeError:
                    pass
                if alpha0.ndim < g_f32.ndim and tuple(g_f32.shape[: alpha0.ndim]) == tuple(alpha0.shape):
                    try:
                        view_sh = tuple(alpha0.shape) + (1,) * (g_f32.ndim - alpha0.ndim)
                        accepted = _try_accept_alpha_g(
                            alpha0.reshape(view_sh).expand(g_f32.shape).clone(), allow_upstream_throttle=False
                        )
                        if accepted is not None:
                            return accepted
                    except RuntimeError:
                        pass
                if alpha0.numel() > 1 and g_numel % alpha0.numel() == 0:
                    try:
                        rep_cand = (
                            alpha0.reshape(-1, 1)
                            .expand(alpha0.numel(), g_numel // alpha0.numel())
                            .reshape(g_f32.shape)
                            .clone()
                        )
                        accepted = _try_accept_alpha_g(rep_cand, allow_upstream_throttle=False)
                        if accepted is not None:
                            return accepted
                    except RuntimeError:
                        pass

        # Disjoint coordinate routing probe (`transpose`, `permute`, `t`, `cat`, `stack`, `slice`, `select`).
        u_map = vjp_eval_fn(torch.ones_like(g_f32))
        idx_f = torch.arange(g_numel, device=g_f32.device, dtype=torch.float32)
        lo_map = vjp_eval_fn(((idx_f % 1024.0) + 1.0).reshape(g_f32.shape))
        hi_map: Optional[Dict[int, torch.Tensor]] = (
            vjp_eval_fn(torch.div(idx_f, 1024, rounding_mode="floor").reshape(g_f32.shape))
            if g_numel > 1024
            else {}
        )
        if u_map is not None and lo_map is not None and hi_map is not None:
            cand_flat_cpu = torch.ones(g_numel, dtype=torch.float32)
            mapped_any = False
            for rid_k, _, delta_r, _, _, _, _, alpha_k in alpha_entries:
                u_r = u_map.get(rid_k)
                lo_r = lo_map.get(rid_k)
                if u_r is None or lo_r is None or u_r.shape != delta_r.shape:
                    continue
                hi_r = hi_map.get(rid_k)
                nz_u = u_r.abs() > 1e-12
                safe_u = torch.where(nz_u, u_r, torch.ones_like(u_r))
                k_lo = torch.round(lo_r / safe_u).to(dtype=torch.long) - 1
                k_hi = (
                    torch.round(hi_r / safe_u).to(dtype=torch.long)
                    if (hi_r is not None and g_numel > 1024)
                    else torch.zeros_like(k_lo)
                )
                k_idx = k_hi * 1024 + k_lo
                valid_m = nz_u & (k_idx >= 0) & (k_idx < g_numel)
                if bool(valid_m.any()):
                    cand_flat_cpu.scatter_reduce_(
                        0,
                        k_idx[valid_m].detach().to("cpu", dtype=torch.long).reshape(-1),
                        alpha_k[valid_m].detach().to("cpu", dtype=torch.float32).reshape(-1),
                        reduce="amin",
                        include_self=True,
                    )
                    mapped_any = True
            if mapped_any:
                accepted = _try_accept_alpha_g(
                    cand_flat_cpu.to(device=g_f32.device).reshape(g_f32.shape),
                    allow_upstream_throttle=True,
                )
                if accepted is not None:
                    return accepted

        # Disjoint batch-slice support fallback (e.g. per-sample linear projections).
        if g_f32.ndim >= 1 and 1 < int(g_f32.shape[0]) <= 64:
            n_b = int(g_f32.shape[0])
            slice_ok = True
            cand_b = torch.ones_like(g_f32)
            seen_supp: Dict[int, torch.Tensor] = {}
            for bi in range(n_b):
                bm = torch.zeros_like(g_f32)
                bm[bi] = 1.0
                b_map = vjp_eval_fn(g_f32 * bm)
                if b_map is None:
                    slice_ok = False
                    break
                b_alpha = 1.0
                for rid_k, _, delta_r, _, _, _, _, alpha_k in alpha_entries:
                    bd = b_map.get(rid_k)
                    if bd is None or bd.shape != delta_r.shape:
                        continue
                    supp = bd.abs() > 1e-12
                    if not bool(supp.any()):
                        continue
                    prev_s = seen_supp.get(rid_k)
                    if prev_s is not None and bool((prev_s & supp).any()):
                        slice_ok = False
                        break
                    seen_supp[rid_k] = supp if prev_s is None else (prev_s | supp)
                    b_alpha = min(b_alpha, float(alpha_k[supp].min().item()))
                if not slice_ok:
                    break
                cand_b[bi] = max(0.0, min(1.0, b_alpha))
            if slice_ok:
                accepted = _try_accept_alpha_g(cand_b, allow_upstream_throttle=True)
                if accepted is not None:
                    return accepted

    alpha = 1.0
    for _, _, delta_r, _, _, _, in_box_k, alpha_k in alpha_entries:
        need_cap = (~in_box_k) & (delta_r.abs() > 0.0)
        if bool(need_cap.any()):
            alpha = min(alpha, float(alpha_k[need_cap].min().item()))
    if phys_exceeded and phys_cap is not None and g_abs_t is not None and g_f32.numel() > 0:
        g_abs_max = float(g_abs_t.max().item())
        if g_abs_max > float(phys_cap):
            alpha = min(alpha, float(phys_cap) / g_abs_max)
    alpha = max(0.0, min(1.0, alpha))

    g_emit = g_f32 if alpha >= 1.0 else (g_f32 * float(alpha))
    for _, ent, delta_r, t_new, e_prev, eff_max, _, _ in alpha_entries:
        ent["target"] = t_new
        e_next = e_prev + float(alpha) * delta_r
        if bool((e_prev.abs() <= eff_max + 1e-3).all()):
            e_next = e_next.clamp(-eff_max, eff_max)
        ent["emitted"] = e_next
    return g_emit.clamp(-fp16_lim, fp16_lim), float(alpha)


_CKPT_HOOK_LOCK = threading.RLock()
_CKPT_HOOK_ORIG: Any = None
_CKPT_HOOK_ACTIVE = 0
_CKPT_HOOK_TLS = threading.local()
_CKPT_ISOMETRIC_CHAIN_PREFIXES = (
    "CloneBackward",
    "AliasBackward",
    "ContiguousBackward",
    "ViewBackward",
    "UnsafeViewBackward",
    "ReshapeAliasBackward",
    "UnsqueezeBackward",
    "SqueezeBackward",
    "TransposeBackward",
    "PermuteBackward",
    "TBackward",
    "NegBackward",
)
_CKPT_INNER_SKIP_PREFIXES = _MEMORY_ORDER_VJP_PREFIXES + (
    "ToCopyBackward",
    "SliceBackward",
    "SelectBackward",
    "CatBackward",
    "StackBackward",
    "SplitBackward",
    "SplitWithSizesBackward",
    "TransposeBackward",
    "PermuteBackward",
    "TBackward",
    "ExpandBackward",
    "IndexBackward",
    "GatherBackward",
)


class _CkptHookDetector:
    """Detects any Tensor.register_hook call executed inside a recomputed checkpoint subgraph."""

    def __init__(self) -> None:
        self.saw_tensor_hook = False
        self.hooked_node_ids: set[int] = set()

    def __enter__(self):
        global _CKPT_HOOK_ORIG, _CKPT_HOOK_ACTIVE
        stack = getattr(_CKPT_HOOK_TLS, "stack", None)
        if stack is None:
            stack = []
            _CKPT_HOOK_TLS.stack = stack
        stack.append(self)
        with _CKPT_HOOK_LOCK:
            if _CKPT_HOOK_ACTIVE == 0:
                orig_fn = torch.Tensor.register_hook
                _CKPT_HOOK_ORIG = orig_fn

                def _tracked_register_hook(tensor_self, hook, *args, **kwargs):
                    res = orig_fn(tensor_self, hook, *args, **kwargs)
                    stk = getattr(_CKPT_HOOK_TLS, "stack", None)
                    if stk:
                        gf = getattr(tensor_self, "grad_fn", None)
                        gf_id = id(gf) if gf is not None else None
                        for det in stk:
                            det.saw_tensor_hook = True
                            if gf_id is not None:
                                det.hooked_node_ids.add(gf_id)
                    return res

                torch.Tensor.register_hook = _tracked_register_hook  # type: ignore[method-assign]
            _CKPT_HOOK_ACTIVE += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CKPT_HOOK_ORIG, _CKPT_HOOK_ACTIVE
        stack = getattr(_CKPT_HOOK_TLS, "stack", None)
        if stack:
            try:
                stack.remove(self)
            except ValueError:
                pass
        with _CKPT_HOOK_LOCK:
            _CKPT_HOOK_ACTIVE = max(0, _CKPT_HOOK_ACTIVE - 1)
            if _CKPT_HOOK_ACTIVE == 0 and _CKPT_HOOK_ORIG is not None:
                torch.Tensor.register_hook = _CKPT_HOOK_ORIG  # type: ignore[method-assign]
                _CKPT_HOOK_ORIG = None
        return False


class _CkptLinearGradReplace(torch.autograd.Function):
    """Injects adjusted cotangents onto linear checkpoint outputs without distorting user tensor hooks."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scaled_val_f32: torch.Tensor, scale_down: float, need_sub: bool):
        ctx.scaled_val = scaled_val_f32.detach()
        ctx.scale_down = float(scale_down)
        ctx.need_sub = bool(need_sub)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if grad_out is None:
            return None, None, None, None
        if grad_out.requires_grad:
            if not ctx.need_sub:
                res = (
                    (grad_out.to(dtype=torch.float32) * ctx.scale_down)
                    .clamp(-65504.0, 65504.0)
                    .to(dtype=grad_out.dtype)
                )
            else:
                nz_g = grad_out.detach().abs() > 1e-12
                if bool(nz_g.any()):
                    go_live_f32 = grad_out.to(dtype=torch.float32)
                    ratio = torch.where(
                        nz_g,
                        ctx.scaled_val / torch.where(nz_g, go_live_f32.detach(), torch.ones_like(ctx.scaled_val)),
                        torch.zeros_like(ctx.scaled_val),
                    ).detach()
                    res = (go_live_f32 * ratio).clamp(-65504.0, 65504.0).to(dtype=grad_out.dtype)
                else:
                    res = ctx.scaled_val.to(dtype=grad_out.dtype)
            return res, None, None, None
        return ctx.scaled_val.to(dtype=grad_out.dtype, device=grad_out.device), None, None, None


def _collect_ckpt_tensors(obj: Any, out: List[torch.Tensor]) -> None:
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _collect_ckpt_tensors(item, out)
    elif isinstance(obj, dict):
        for item in obj.values():
            _collect_ckpt_tensors(item, out)


def _inspect_recomputed_ckpt_subgraph(
    raw_outputs: Any,
    det_args: Tuple[Any, ...],
    det: _CkptHookDetector,
) -> Tuple[bool, Dict[int, float], List[object]]:
    """Classifies whether a recomputed checkpoint subgraph has hooks/custom ops and extracts FP16 cast caps."""
    out_tensors: List[torch.Tensor] = []
    _collect_ckpt_tensors(raw_outputs, out_tensors)

    is_nonlinear = bool(det.saw_tensor_hook)
    for t_o in out_tensors:
        if getattr(t_o, "_backward_hooks", None):
            is_nonlinear = True
            break

    det_input_edge_by_id: Dict[int, int] = {}
    tensor_slot_idx = 0
    for arg in det_args:
        if isinstance(arg, torch.Tensor):
            if arg.requires_grad:
                det_input_edge_by_id[id(arg)] = tensor_slot_idx
            tensor_slot_idx += 1

    visited: Dict[int, object] = {}
    queue: List[object] = []
    for t_o in out_tensors:
        gf = getattr(t_o, "grad_fn", None) if t_o.requires_grad else None
        if gf is not None and id(gf) not in visited:
            visited[id(gf)] = gf
            queue.append(gf)

    children_to_parents: Dict[int, List[Tuple[object, str]]] = {}
    acc_nodes_for_inputs: Dict[int, object] = {}
    inner_mm_nodes: List[object] = []

    qi = 0
    while qi < len(queue) and len(visited) < _MAX_VJP_NODES:
        cur = queue[qi]
        qi += 1
        cname = type(cur).__name__
        if cname == "AccumulateGrad":
            var = getattr(cur, "variable", None)
            if getattr(var, "_backward_hooks", None):
                is_nonlinear = True
            if isinstance(var, torch.Tensor) and var.dtype != torch.float16:
                edge_idx = det_input_edge_by_id.get(id(var))
                if edge_idx is not None:
                    acc_nodes_for_inputs[edge_idx] = cur
            continue

        is_one_shot = _is_one_shot_bwd_node(cur)
        if is_one_shot:
            is_nonlinear = True
        elif (not cname.startswith(_CKPT_INNER_SKIP_PREFIXES)) and _has_fp16_grad_input(cur):
            inner_mm_nodes.append(cur)

        for nf, _ in getattr(cur, "next_functions", ()):
            if nf is None:
                continue
            nid = id(nf)
            children_to_parents.setdefault(nid, []).append((cur, cname))
            if nid not in visited:
                visited[nid] = nf
                queue.append(nf)

    ckpt_cast_caps: Dict[int, float] = {}
    for edge_idx, acc_node in acc_nodes_for_inputs.items():
        all_cast_bound = True
        cast_leaf_ids: set[int] = set()
        walk_q: List[object] = [acc_node]
        walk_seen: set[int] = {id(acc_node)}
        wi = 0
        while wi < len(walk_q):
            c_node = walk_q[wi]
            wi += 1
            parents = children_to_parents.get(id(c_node), ())
            if not parents:
                all_cast_bound = False
                break
            for p_node, p_cname in parents:
                if _is_fp16_cast_leaf_boundary(p_node, p_cname):
                    cast_leaf_ids.add(id(p_node))
                    continue
                if p_cname.startswith(_CKPT_ISOMETRIC_CHAIN_PREFIXES):
                    pid_n = id(p_node)
                    if pid_n not in walk_seen:
                        walk_seen.add(pid_n)
                        walk_q.append(p_node)
                    continue
                all_cast_bound = False
                break
            if not all_cast_bound:
                break
        if all_cast_bound and cast_leaf_ids:
            ckpt_cast_caps[edge_idx] = float(len(cast_leaf_ids)) * _FP16_BWD_ROOT_MAX

    return is_nonlinear, ckpt_cast_caps, inner_mm_nodes


def _make_inner_ckpt_hooks(node: object, safe_peak: float = 256.0) -> Tuple[Any, Any]:
    """Safe power-of-2 FP16 local scaling hooks for internal C++ matmul ops inside non-linear checkpoints."""
    state: Dict[str, float] = {"pre_s": 1.0}

    def _in_pre(grad_outputs):
        if not isinstance(grad_outputs, tuple) or not grad_outputs:
            return grad_outputs
        max_abs = 0.0
        has_fp16 = False
        for go in grad_outputs:
            if (
                isinstance(go, torch.Tensor)
                and go.dtype == torch.float16
                and go.numel() > 0
                and bool(torch.isfinite(go).all())
            ):
                has_fp16 = True
                m = float(go.detach().abs().max().item())
                if m > max_abs:
                    max_abs = m
        if not has_fp16 or max_abs <= safe_peak:
            state["pre_s"] = 1.0
            return grad_outputs
        k = int(math.ceil(math.log2(max_abs / safe_peak)))
        if k <= 0:
            state["pre_s"] = 1.0
            return grad_outputs
        scale_down = math.ldexp(1.0, -k)
        state["pre_s"] = math.ldexp(1.0, k)
        new_gos = list(grad_outputs)
        for si, go in enumerate(grad_outputs):
            if (
                isinstance(go, torch.Tensor)
                and go.dtype == torch.float16
                and go.numel() > 0
                and bool(torch.isfinite(go).all())
            ):
                new_gos[si] = (go.to(dtype=torch.float32) * scale_down).clamp(-65504.0, 65504.0).to(dtype=torch.float16)
        return tuple(new_gos)

    def _in_post(grad_inputs, grad_outputs):
        pre_s = state.get("pre_s", 1.0)
        if pre_s == 1.0 or not isinstance(grad_inputs, tuple):
            return grad_inputs
        new_gis = list(grad_inputs)
        changed = False
        for idx, gi in enumerate(grad_inputs):
            if (
                not isinstance(gi, torch.Tensor)
                or not gi.is_floating_point()
                or gi.numel() == 0
                or not bool(torch.isfinite(gi).all())
            ):
                continue
            if gi.dtype == torch.float16:
                new_gis[idx] = (gi.to(dtype=torch.float32) * pre_s).clamp(-65504.0, 65504.0).to(dtype=torch.float16)
            elif gi.dtype == torch.float64:
                new_gis[idx] = gi * pre_s
            else:
                new_gis[idx] = (gi.to(dtype=torch.float32) * pre_s).to(dtype=gi.dtype)
            changed = True
        return tuple(new_gis) if changed else grad_inputs

    return _in_pre, _in_post


def _register_fp16_py_fn_hooks_locked(
    py_nodes: List[object],
    pass_key: object,
) -> None:
    """Registers at most one pass-scoped pre-hook and post-hook per one-shot backward boundary node (caller holds _FP16_LOAD_LOCK)."""
    if not py_nodes:
        return
    pass_meta = _FP16_BWD_PASS_META.setdefault(pass_key, {})
    pre_hooked = pass_meta.setdefault("pre_hooked_py_fns", {})
    hooked = pass_meta.setdefault("hooked_py_fns", {})
    for py_node in py_nodes:
        pid = id(py_node)
        if _is_linear_checkpoint_bwd_node(py_node) and pid not in pre_hooked and hasattr(py_node, "register_prehook"):
            try:
                pre_hooked[pid] = py_node.register_prehook(_make_py_fn_pre_hook(py_node, pass_key))
            except Exception:
                pass
        if pid not in hooked and hasattr(py_node, "register_hook"):
            try:
                hooked[pid] = py_node.register_hook(_make_py_fn_post_hook(py_node, pass_key))
            except Exception:
                pass


def _make_py_fn_pre_hook(py_node: object, pass_key: object):
    """Defers scaling for CheckpointFunctionBackward until its recomputed graph is inspected, and scales non-ckpt boundaries directly."""
    ran = [False]
    py_name = type(py_node).__name__
    is_mem_local = py_name.startswith(_MEMORY_ORDER_VJP_PREFIXES)
    safe_peak = 8192.0 if is_mem_local else 2048.0

    def _pre_hook(grad_outputs):
        if ran[0]:
            return grad_outputs
        ran[0] = True
        if not isinstance(grad_outputs, tuple) or not grad_outputs:
            return grad_outputs

        pid = id(py_node)
        with _FP16_LOAD_LOCK:
            bwd_state = _FP16_BWD_USED_BUDGET.get(pass_key)
            if bwd_state is None:
                return grad_outputs
            slot_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
            for si in range(len(grad_outputs)):
                ent = bwd_state.get(-(pid * 1024 + si + 16))
                if isinstance(ent, dict):
                    t_raw = ent.get("target")
                    e_raw = ent.get("emitted")
                    if isinstance(t_raw, torch.Tensor) and isinstance(e_raw, torch.Tensor):
                        slot_states.append((t_raw.detach(), e_raw.detach()))
                        continue
                slot_states.append(None)

        go_f32_slots: Dict[int, torch.Tensor] = {}
        need_sub_slots: Dict[int, bool] = {}
        need_sub = False
        max_abs_all = 0.0
        for si, go in enumerate(grad_outputs):
            if (
                not isinstance(go, torch.Tensor)
                or go.dtype != torch.float16
                or go.numel() == 0
                or not bool(torch.isfinite(go).all())
            ):
                continue
            go_cand = go.detach().to(dtype=torch.float32)
            go_f32 = go_cand
            slot_need_sub = False
            st = slot_states[si] if si < len(slot_states) else None
            if st is not None:
                t_raw, e_raw = st
                t_al = _align_bwd_state_tensor(t_raw, go_cand)
                e_al = _align_bwd_state_tensor(e_raw, go_cand)
                if t_al is not None and e_al is not None:
                    miss_cand = t_al - e_al
                    miss_max = float(miss_cand.abs().max().item())
                    if miss_max > 1e-6:
                        e_max = float(e_al.abs().max().item())
                        if e_max > 1e-12 and float((go_cand - e_al).abs().max().item()) <= max(64.0, 2e-2 * e_max):
                            go_f32 = t_al
                        elif e_max <= 1e-12 and float(go_cand.abs().max().item()) <= 1e-6:
                            go_f32 = t_al
                        else:
                            go_f32 = go_cand + miss_cand
                        need_sub = True
                        slot_need_sub = True
            go_f32_slots[si] = go_f32
            need_sub_slots[si] = slot_need_sub
            slot_max = float(go_f32.abs().max().item())
            if slot_max > max_abs_all:
                max_abs_all = slot_max

        orig_run_fn = getattr(py_node, "run_function", None) if py_name == "CheckpointFunctionBackward" else None
        if py_name == "CheckpointFunctionBackward" and callable(orig_run_fn):
            with _FP16_LOAD_LOCK:
                pass_meta = _FP16_BWD_PASS_META.get(pass_key)
                if pass_meta is None:
                    return grad_outputs
                pass_meta.setdefault("orig_ckpt_run_fns", []).append((py_node, orig_run_fn))

            def _wrapped_run_fn(*det_args, **det_kwargs):
                try:
                    with torch.enable_grad(), _CkptHookDetector() as det:
                        raw_outputs = orig_run_fn(*det_args, **det_kwargs)
                finally:
                    py_node.run_function = orig_run_fn  # type: ignore[attr-defined]

                is_nonlinear, ckpt_cast_caps, inner_mm_nodes = _inspect_recomputed_ckpt_subgraph(
                    raw_outputs, det_args, det
                )
                with _FP16_LOAD_LOCK:
                    pm_now = _FP16_BWD_PASS_META.get(pass_key)
                    if pm_now is None:
                        return raw_outputs
                    if ckpt_cast_caps:
                        pm_now.setdefault("ckpt_cast_caps_by_pid", {})[pid] = ckpt_cast_caps
                    if is_nonlinear:
                        pm_now.setdefault("nonlinear_ckpt_pids", set()).add(pid)
                        if inner_mm_nodes:
                            h_list = pm_now.setdefault("inner_ckpt_hooks_by_pid", {}).setdefault(pid, [])
                            for mm_node in inner_mm_nodes:
                                p_h, q_h = _make_inner_ckpt_hooks(mm_node)
                                if hasattr(mm_node, "register_prehook"):
                                    try:
                                        h_list.append(mm_node.register_prehook(p_h))
                                    except Exception:
                                        pass
                                if hasattr(mm_node, "register_hook"):
                                    try:
                                        h_list.append(mm_node.register_hook(q_h))
                                    except Exception:
                                        pass
                        return raw_outputs

                    if not go_f32_slots or (not need_sub and max_abs_all <= 2048.0):
                        return raw_outputs

                    pre_s = 1.0
                    scale_down = 1.0
                    if max_abs_all > safe_peak:
                        k = int(math.ceil(math.log2(max_abs_all / safe_peak)))
                        if k > 0:
                            scale_down = math.ldexp(1.0, -k)
                            pre_s = math.ldexp(1.0, k)
                    pm_now.setdefault("pre_scale_by_pid", {})[pid] = pre_s

                if isinstance(raw_outputs, torch.Tensor):
                    if raw_outputs.requires_grad and 0 in go_f32_slots:
                        scaled_v = (go_f32_slots[0] * scale_down).clamp(-65504.0, 65504.0)
                        return _CkptLinearGradReplace.apply(raw_outputs, scaled_v, scale_down, need_sub_slots[0])
                    return raw_outputs
                if isinstance(raw_outputs, (tuple, list)):
                    wrapped_items = list(raw_outputs)
                    for si, item in enumerate(raw_outputs):
                        if isinstance(item, torch.Tensor) and item.requires_grad and si in go_f32_slots:
                            scaled_v = (go_f32_slots[si] * scale_down).clamp(-65504.0, 65504.0)
                            wrapped_items[si] = _CkptLinearGradReplace.apply(
                                item, scaled_v, scale_down, need_sub_slots[si]
                            )
                    return type(raw_outputs)(wrapped_items)
                return raw_outputs

            py_node.run_function = _wrapped_run_fn  # type: ignore[attr-defined]
            return grad_outputs

        if not go_f32_slots:
            return grad_outputs
        if not need_sub and (is_mem_local or max_abs_all <= 2048.0):
            return grad_outputs

        pre_s = 1.0
        scale_down = 1.0
        if max_abs_all > safe_peak:
            k = int(math.ceil(math.log2(max_abs_all / safe_peak)))
            if k > 0:
                scale_down = math.ldexp(1.0, -k)
                pre_s = math.ldexp(1.0, k)

        with _FP16_LOAD_LOCK:
            pass_meta = _FP16_BWD_PASS_META.get(pass_key)
            if pass_meta is None:
                return grad_outputs
            pass_meta.setdefault("pre_scale_by_pid", {})[pid] = pre_s

        new_gos = list(grad_outputs)
        for si, go_f32 in go_f32_slots.items():
            orig_go = grad_outputs[si]
            scaled_val = (go_f32 * scale_down).clamp(-65504.0, 65504.0)
            if orig_go.requires_grad:
                if not need_sub:
                    new_gos[si] = (
                        (orig_go.to(dtype=torch.float32) * scale_down)
                        .clamp(-65504.0, 65504.0)
                        .to(dtype=orig_go.dtype)
                    )
                else:
                    nz_g = orig_go.detach().abs() > 1e-12
                    if bool(nz_g.any()):
                        go_live_f32 = orig_go.to(dtype=torch.float32)
                        ratio = torch.where(
                            nz_g,
                            scaled_val / torch.where(nz_g, go_live_f32.detach(), torch.ones_like(scaled_val)),
                            torch.zeros_like(scaled_val),
                        ).detach()
                        new_gos[si] = (go_live_f32 * ratio).clamp(-65504.0, 65504.0).to(dtype=orig_go.dtype)
                    else:
                        new_gos[si] = scaled_val.to(dtype=orig_go.dtype)
            else:
                new_gos[si] = scaled_val.to(dtype=orig_go.dtype)
        return tuple(new_gos)

    return _pre_hook


def _make_py_fn_post_hook(py_node: object, pass_key: object):
    """Telescopes float16 and mixed-dtype checkpoint gradients emitted by a single native invocation of a one-shot backward node."""
    ran = [False]
    py_name = type(py_node).__name__
    is_ckpt_node = _is_linear_checkpoint_bwd_node(py_node)
    is_ckpt_fn = py_name == "CheckpointFunctionBackward"

    def _reconstruct_true_gi(
        raw_gi_f32: torch.Tensor,
        edge_idx: int,
        go0_f32: Optional[torch.Tensor],
        missed0_f32: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if go0_f32 is None or missed0_f32 is None or go0_f32.numel() == 0:
            return raw_gi_f32
        true_go0 = go0_f32 + missed0_f32
        nz0 = go0_f32.abs() > 1e-12
        act = (missed0_f32.abs() > 1e-6) | nz0
        if not bool(act.any().item()):
            return raw_gi_f32
        safe_go0 = torch.where(nz0, go0_f32, torch.ones_like(go0_f32))
        gain0 = torch.where(nz0, true_go0 / safe_go0, torch.ones_like(go0_f32))
        act_gains = gain0[act]
        g_min = float(act_gains.min().item())
        g_max = float(act_gains.max().item())
        if (g_max - g_min) <= 1e-4 * max(1.0, abs(g_min), abs(g_max)):
            g_sc = float(act_gains.mean().item())
            return raw_gi_f32 * g_sc
        if raw_gi_f32.numel() == go0_f32.numel() and (
            py_name.startswith(_MEMORY_ORDER_VJP_PREFIXES + ("SqrtBackward", "PowBackward0", "ReciprocalBackward"))
            or py_name == "CheckpointFunctionBackward"
        ):
            return raw_gi_f32 * gain0.reshape(raw_gi_f32.shape)
        if raw_gi_f32.ndim >= 1 and go0_f32.ndim >= 1 and int(raw_gi_f32.shape[0]) == int(go0_f32.shape[0]):
            if py_name.startswith("MmBackward") and edge_idx != 0:
                return raw_gi_f32
            if py_name.startswith("AddmmBackward") and edge_idx != 1:
                return raw_gi_f32
            bsz = int(go0_f32.shape[0])
            g2 = gain0.reshape(bsz, -1)
            a2 = act.reshape(bsz, -1)
            row_gains = torch.ones((bsz,), device=raw_gi_f32.device, dtype=torch.float32)
            row_ok = True
            for bi in range(bsz):
                m_b = a2[bi]
                if not bool(m_b.any().item()):
                    continue
                vals_b = g2[bi][m_b]
                vb_min = float(vals_b.min().item())
                vb_max = float(vals_b.max().item())
                if (vb_max - vb_min) > 1e-4 * max(1.0, abs(vb_min), abs(vb_max)):
                    row_ok = False
                    break
                row_gains[bi] = float(vals_b.mean().item())
            if row_ok:
                return raw_gi_f32 * row_gains.view((bsz,) + (1,) * (raw_gi_f32.ndim - 1))
        return raw_gi_f32

    def _hook(grad_inputs, grad_outputs):
        if ran[0]:
            return grad_inputs
        ran[0] = True

        t_in_0: Optional[torch.Tensor] = None
        e_in_0: Optional[torch.Tensor] = None
        pre_s: Optional[float] = None
        ckpt_cast_caps: Dict[int, float] = {}
        is_nl = False
        pid = id(py_node)
        with _FP16_LOAD_LOCK:
            bwd_state = _FP16_BWD_USED_BUDGET.get(pass_key)
            if bwd_state is None:
                return grad_inputs
            pass_meta = _FP16_BWD_PASS_META.setdefault(pass_key, {})
            inner_map = pass_meta.get("inner_ckpt_hooks_by_pid")
            if isinstance(inner_map, dict):
                h_list = inner_map.pop(pid, None)
                if isinstance(h_list, list):
                    for h in h_list:
                        if hasattr(h, "remove"):
                            try:
                                h.remove()
                            except Exception:
                                pass
                    h_list.clear()
            nl_pids = pass_meta.get("nonlinear_ckpt_pids")
            if isinstance(nl_pids, set):
                is_nl = pid in nl_pids
                nl_pids.discard(pid)
            caps_map = pass_meta.get("ckpt_cast_caps_by_pid")
            if isinstance(caps_map, dict):
                raw_caps = caps_map.pop(pid, None)
                if isinstance(raw_caps, dict):
                    ckpt_cast_caps = raw_caps
            chain_cache = pass_meta.setdefault("vjp_chain_cache", {})
            chain_lock = pass_meta.setdefault("chain_lock", threading.Lock())
            pre_map = pass_meta.get("pre_scale_by_pid")
            if isinstance(pre_map, dict):
                pre_s_val = pre_map.pop(pid, None)
                if pre_s_val is not None:
                    pre_s = float(pre_s_val)
            if is_ckpt_node and pre_s is None and not is_nl:
                ent_0 = bwd_state.get(-(pid * 1024 + 16))
                if isinstance(ent_0, dict):
                    t_raw = ent_0.get("target")
                    e_raw = ent_0.get("emitted")
                    if isinstance(t_raw, torch.Tensor) and isinstance(e_raw, torch.Tensor):
                        t_in_0 = t_raw.detach()
                        e_in_0 = e_raw.detach()

        if not isinstance(grad_inputs, tuple):
            return grad_inputs

        go0_f32: Optional[torch.Tensor] = None
        missed0_f32: Optional[torch.Tensor] = None
        if (
            is_ckpt_node
            and not is_nl
            and t_in_0 is not None
            and e_in_0 is not None
            and isinstance(grad_outputs, tuple)
            and len(grad_outputs) >= 1
            and isinstance(grad_outputs[0], torch.Tensor)
            and grad_outputs[0].numel() > 0
            and bool(torch.isfinite(grad_outputs[0]).all())
        ):
            go0_cand = grad_outputs[0].detach().to(dtype=torch.float32)
            t_al = _align_bwd_state_tensor(t_in_0, go0_cand)
            e_al = _align_bwd_state_tensor(e_in_0, go0_cand)
            if t_al is not None and e_al is not None:
                miss_cand = t_al - e_al
                if float(miss_cand.abs().max().item()) > 1e-6:
                    e_max = float(e_al.abs().max().item())
                    if e_max > 1e-12 and float((go0_cand - e_al).abs().max().item()) <= max(64.0, 2e-2 * e_max):
                        go0_f32 = e_al
                        missed0_f32 = miss_cand
                    else:
                        go0_f32 = go0_cand
                        missed0_f32 = t_al - go0_cand

        nxts = getattr(py_node, "next_functions", ())
        new_gis = list(grad_inputs)
        changed = False
        for idx, ((nf, onr), gi) in enumerate(zip(nxts, grad_inputs)):
            if (
                nf is None
                or not isinstance(gi, torch.Tensor)
                or not gi.is_floating_point()
                or gi.numel() == 0
                or not bool(torch.isfinite(gi).all())
            ):
                continue

            is_fp16_edge = gi.dtype == torch.float16
            if not is_fp16_edge and not is_ckpt_fn:
                if pre_s is not None and pre_s != 1.0:
                    if gi.dtype == torch.float64:
                        new_gis[idx] = gi * float(pre_s)
                    else:
                        new_gis[idx] = (gi.to(dtype=torch.float32) * float(pre_s)).to(dtype=gi.dtype)
                    changed = True
                continue

            if not is_fp16_edge and gi.dtype == torch.float64:
                raw_gi_chk = gi.detach().to(dtype=torch.float32)
                if not bool(torch.isfinite(raw_gi_chk).all()):
                    if pre_s is not None and pre_s != 1.0:
                        new_gis[idx] = gi * float(pre_s)
                        changed = True
                    continue

            gi_abs_max = float(gi.detach().abs().max().item())
            if pre_s is not None:
                raw_gi_f32 = gi.detach().to(dtype=torch.float32)
                true_gi_f32 = raw_gi_f32 * float(pre_s)
                reconstructed = bool((true_gi_f32 - raw_gi_f32).abs().max().item() > 1e-6)
            else:
                raw_gi_f32 = (
                    gi.detach().to(dtype=torch.float32).clamp(-_FP16_BWD_ROOT_MAX, _FP16_BWD_ROOT_MAX)
                    if is_fp16_edge
                    else gi.detach().to(dtype=torch.float32)
                )
                true_gi_f32 = (
                    raw_gi_f32
                    if is_nl
                    else _reconstruct_true_gi(raw_gi_f32, idx, go0_f32, missed0_f32)
                )
                reconstructed = bool((true_gi_f32 - raw_gi_f32).abs().max().item() > 1e-6)

            sub_deltas, sub_elem_same, sub_eval_fn, sub_py_nodes, sub_alpha_pb = _eval_fp16_vjp_to_roots(
                nf,
                int(onr),
                true_gi_f32,
                chain_cache=chain_cache,
                chain_lock=chain_lock,
            )

            cast_cap = ckpt_cast_caps.get(idx) if not is_fp16_edge else None
            edge_lim = _FP16_BWD_ROOT_MAX if is_fp16_edge else 1e30
            if cast_cap is not None:
                eff_max_emit: Optional[float] = float(cast_cap)
            elif is_fp16_edge and (reconstructed or pre_s is not None):
                eff_max_emit = _FP16_BWD_ROOT_MAX
            else:
                eff_max_emit = None

            with _FP16_LOAD_LOCK:
                bwd_now = _FP16_BWD_USED_BUDGET.get(pass_key)
                if bwd_now is None:
                    return grad_inputs
                if sub_py_nodes:
                    _register_fp16_py_fn_hooks_locked(sub_py_nodes, pass_key)

                if cast_cap is not None and sub_deltas and type(nf).__name__ == "AccumulateGrad":
                    for rk in sub_deltas:
                        if int(rk) <= -_NONFP16_LEAF_KEY_OFFSET:
                            ent_r = bwd_now.get(rk)
                            if ent_r is None:
                                ent_r = {"target": None, "emitted": None, "abs_used": 0.0}
                                bwd_now[rk] = ent_r
                            prev_cap = float(ent_r.get("cap", 1e30))
                            ent_r["cap"] = min(prev_cap, float(cast_cap))

                done_v = False
                g_emit_det = true_gi_f32.clamp(-edge_lim, edge_lim)
                scale_det: Union[torch.Tensor, float] = 1.0
                if sub_deltas is not None:
                    if not sub_deltas:
                        done_v = True
                        g_emit_det = true_gi_f32.clamp(-edge_lim, edge_lim)
                    else:
                        res_sub = _apply_telescoping_deltas(
                            bwd_now,
                            sub_deltas,
                            true_gi_f32,
                            edge_lim,
                            elementwise_single=sub_elem_same,
                            vjp_eval_fn=sub_eval_fn,
                            alpha_pullback_fn=sub_alpha_pb,
                            max_emit_abs=eff_max_emit,
                        )
                        if res_sub is not None:
                            g_emit_det, scale_det = res_sub
                            done_v = True

                if not done_v:
                    nf_key = -(id(nf) * 1024 + int(onr) + 16)
                    ent_nf = bwd_now.get(nf_key)
                    if ent_nf is None:
                        ent_nf = {"target": None, "emitted": None, "abs_used": 0.0}
                        bwd_now[nf_key] = ent_nf
                    nf_base_cap = float(cast_cap) if cast_cap is not None else edge_lim
                    lim = max(0.0, nf_base_cap - float(ent_nf["abs_used"]))
                    g_abs_max = float(true_gi_f32.abs().max().item()) if true_gi_f32.numel() > 0 else 0.0
                    em_abs = min(g_abs_max, lim)
                    if em_abs > 0.0:
                        ent_nf["abs_used"] = min(nf_base_cap, float(ent_nf["abs_used"]) + em_abs)
                    g_emit_det = true_gi_f32.clamp(-lim, lim)
                    scale_det = 0.0 if (g_abs_max > 0.0 and lim < g_abs_max) else 1.0

            if (
                done_v
                and pre_s is None
                and gi_abs_max <= edge_lim
                and (
                    (not reconstructed and not isinstance(scale_det, torch.Tensor) and float(scale_det) >= 1.0)
                    or bool(torch.allclose(g_emit_det, raw_gi_f32, atol=1e-3, rtol=1e-4))
                )
            ):
                continue

            if not gi.requires_grad:
                new_gi = g_emit_det.to(dtype=gi.dtype)
            elif reconstructed:
                nz_r = raw_gi_f32.abs() > 1e-12
                eff_scale = torch.where(
                    nz_r,
                    g_emit_det / torch.where(nz_r, raw_gi_f32, torch.ones_like(raw_gi_f32)),
                    torch.zeros_like(raw_gi_f32),
                ).detach()
                g_live = (
                    gi.to(dtype=torch.float32)
                    if (pre_s is not None or not is_fp16_edge)
                    else gi.to(dtype=torch.float32).clamp(-edge_lim, edge_lim)
                )
                new_gi = (g_live * eff_scale).clamp(-edge_lim, edge_lim).to(dtype=gi.dtype)
            else:
                g_live = (
                    gi.to(dtype=torch.float32).clamp(-edge_lim, edge_lim)
                    if is_fp16_edge
                    else gi.to(dtype=torch.float32)
                )
                if isinstance(scale_det, torch.Tensor):
                    new_gi = (
                        (g_live * scale_det.detach().clamp(0.0, 1.0))
                        .clamp(-edge_lim, edge_lim)
                        .to(dtype=gi.dtype)
                    )
                elif float(scale_det) >= 1.0:
                    new_gi = g_live.clamp(-edge_lim, edge_lim).to(dtype=gi.dtype)
                else:
                    s_cl = min(1.0, max(0.0, float(scale_det)))
                    new_gi = (
                        (g_live * s_cl)
                        .clamp(-edge_lim, edge_lim)
                        .to(dtype=gi.dtype)
                    )
            new_gis[idx] = new_gi
            changed = True

        return tuple(new_gis) if changed else grad_inputs

    return _hook


class _SafeCastToWorkDtype(torch.autograd.Function):
    """Promotes low-precision tensors cleanly while bounding half-precision backward gradients."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        work_dtype: torch.dtype,
        fp16_max: float = _FP16_GRAD_MAX,
        track_load: bool = True,
    ) -> torch.Tensor:
        ctx.orig_dtype = x.dtype
        ctx.fp16_max = float(fp16_max)
        ctx._is_direct_leaf = x.grad_fn is None
        if track_load and x.dtype == torch.float16 and x.requires_grad:
            specs, max_inter_gain = _trace_fp16_tensor_roots_with_meta(x)
            if not specs:
                zs_fan = float(max(_zero_stride_fanin(x), 1))
                specs = [(id(x), zs_fan, 1, 1.0, int(x.numel()), zs_fan == 1.0)]
                max_inter_gain = 1.0
            ctx._root_specs = specs
            ctx._max_inter_gain = float(max(1.0, max_inter_gain))
        else:
            ctx._root_specs = ()
            ctx._max_inter_gain = 1.0
        return x.to(dtype=work_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if grad_output is None:
            return None, None, None, None
        orig_dtype = ctx.orig_dtype
        max_inter_gain = float(getattr(ctx, "_max_inter_gain", 1.0))
        fp16_lim = min(_FP16_GRAD_MAX, float(ctx.fp16_max))
        if max_inter_gain > 1.0:
            fp16_lim = min(fp16_lim, _FP16_BWD_ROOT_MAX / max_inter_gain)

        def _emit_grad_out(
            g_emit_det: torch.Tensor, scale_det: Union[torch.Tensor, float]
        ) -> Tuple[torch.Tensor, None, None, None]:
            if not grad_output.requires_grad:
                return g_emit_det.to(dtype=orig_dtype), None, None, None
            g_live = grad_output.to(dtype=torch.float32).clamp(-fp16_lim, fp16_lim)
            if isinstance(scale_det, torch.Tensor):
                g_out = (g_live * scale_det.detach()).clamp(-fp16_lim, fp16_lim).to(dtype=orig_dtype)
            elif float(scale_det) >= 1.0:
                g_out = g_live.clamp(-fp16_lim, fp16_lim).to(dtype=orig_dtype)
            else:
                g_out = (g_live * float(scale_det)).clamp(-fp16_lim, fp16_lim).to(dtype=orig_dtype)
            return g_out, None, None, None

        if orig_dtype == torch.float16:
            root_specs = getattr(ctx, "_root_specs", ())
            if root_specs:
                tid = threading.get_ident()
                gtid = (
                    int(torch._C._current_graph_task_id())
                    if hasattr(torch._C, "_current_graph_task_id")
                    else -1
                )
                pass_key = (tid, gtid)
                cb_registered = True
                with _FP16_LOAD_LOCK:
                    need_cb = not _FP16_BWD_CB_QUEUED.get(pass_key, False)
                    if need_cb:
                        _FP16_BWD_CB_QUEUED[pass_key] = True
                if need_cb:
                    try:
                        def _bwd_done_cb(pk=pass_key):
                            _flush_fp16_bwd_pass(pk)

                        weakref.finalize(_bwd_done_cb, _flush_fp16_bwd_pass, pass_key)
                        torch.autograd.Variable._execution_engine.queue_callback(_bwd_done_cb)
                    except Exception:
                        cb_registered = False
                        _flush_fp16_bwd_pass(pass_key)

                if not cb_registered:
                    for _, w_abs, _, _, _, _ in root_specs:
                        fp16_lim = min(fp16_lim, _FP16_MULTI_BUDGET / max(1.0, float(w_abs)))
                    grad_output = _clamp_low_prec_grad(grad_output, orig_dtype, fp16_max=fp16_lim)
                    return grad_output.to(dtype=orig_dtype), None, None, None

                if grad_output.numel() == 0 or bool(torch.isnan(grad_output).any()):
                    grad_output = _clamp_low_prec_grad(grad_output, orig_dtype, fp16_max=fp16_lim)
                    return grad_output.to(dtype=orig_dtype), None, None, None

                g_f32 = grad_output.detach().to(dtype=torch.float32).clamp(-fp16_lim, fp16_lim)
                g_numel = int(g_f32.numel())
                all_sign_known = all(sp in (-1, 1) and sc > 0.0 for _, _, sp, sc, _, _ in root_specs)

                can_fast_case_a = False
                if len(root_specs) == 1:
                    _, _, sp0, sc0, _, same0 = root_specs[0]
                    can_fast_case_a = bool(same0 and sp0 in (-1, 1) and sc0 > 0.0)
                can_scalar_signed = all_sign_known and all(
                    (rn == 1) or (g_numel == 1 and same_s)
                    for _, _, _, _, rn, same_s in root_specs
                )
                is_direct_leaf = bool(getattr(ctx, "_is_direct_leaf", False))
                use_direct_leaf_fast = is_direct_leaf and (can_fast_case_a or can_scalar_signed)

                vjp_deltas: Optional[Dict[int, torch.Tensor]] = None
                vjp_elem_same = False
                vjp_eval_fn: Optional[Callable[[torch.Tensor], Optional[Dict[int, torch.Tensor]]]] = None
                vjp_py_nodes: List[object] = []
                vjp_alpha_pb: Optional[Callable[[Dict[int, torch.Tensor]], Optional[torch.Tensor]]] = None
                if not use_direct_leaf_fast:
                    ctx_nxts = getattr(ctx, "next_functions", None)
                    if ctx_nxts:
                        nf0, onr0 = ctx_nxts[0]
                        if nf0 is not None:
                            with _FP16_LOAD_LOCK:
                                p_meta = _FP16_BWD_PASS_META.setdefault(pass_key, {})
                                c_cache = p_meta.setdefault("vjp_chain_cache", {})
                                c_lock = p_meta.setdefault("chain_lock", threading.Lock())
                            vjp_deltas, vjp_elem_same, vjp_eval_fn, vjp_py_nodes, vjp_alpha_pb = _eval_fp16_vjp_to_roots(
                                nf0, int(onr0), g_f32, chain_cache=c_cache, chain_lock=c_lock
                            )

                with _FP16_LOAD_LOCK:
                    bwd_state = _FP16_BWD_USED_BUDGET.setdefault(pass_key, {})
                    if vjp_py_nodes:
                        _register_fp16_py_fn_hooks_locked(vjp_py_nodes, pass_key)

                    if not use_direct_leaf_fast and vjp_deltas is not None:
                        if not vjp_deltas:
                            return _emit_grad_out(g_f32, 1.0)
                        res_v = _apply_telescoping_deltas(
                            bwd_state,
                            vjp_deltas,
                            g_f32,
                            fp16_lim,
                            elementwise_single=vjp_elem_same,
                            vjp_eval_fn=vjp_eval_fn,
                            alpha_pullback_fn=vjp_alpha_pb,
                        )
                        if res_v is not None:
                            return _emit_grad_out(*res_v)

                    # Case A: single root, exact elementwise correspondence & known Jacobian sign.
                    if can_fast_case_a:
                        rid0, _, sp0, sc0, _, _ = root_specs[0]
                        s_mult = float(sp0) * float(sc0)
                        res_a = _apply_telescoping_deltas(
                            bwd_state,
                            {rid0: g_f32 * s_mult},
                            g_f32,
                            fp16_lim,
                            elementwise_single=True,
                        )
                        if res_a is not None:
                            return _emit_grad_out(*res_a)

                    # Case B: all reached roots are scalar-compatible (root numel == 1, or scalar grad).
                    if can_scalar_signed:
                        g_sum = g_f32.sum().view(1)
                        scalar_deltas = {
                            rid_k: g_sum * (float(sp_k) * float(sc_k))
                            for rid_k, _, sp_k, sc_k, _, _ in root_specs
                        }
                        res_b = _apply_telescoping_deltas(
                            bwd_state,
                            scalar_deltas,
                            g_f32,
                            fp16_lim,
                            elementwise_single=False,
                        )
                        if res_b is not None:
                            return _emit_grad_out(*res_b)

                    # Case C: conservative unsigned accounting fallback for huge/untraced subgraphs.
                    g_abs_max = float(
                        torch.where(torch.isfinite(g_f32), g_f32.abs(), torch.full_like(g_f32, float("inf")))
                        .max()
                        .item()
                    )
                    rid_eff: List[Tuple[int, Dict[str, object], float]] = []
                    for rid_k, w_abs, _, _, _, _ in root_specs:
                        w_eff = max(1.0, float(w_abs))
                        ent_k = bwd_state.get(rid_k)
                        if ent_k is None:
                            ent_k = {"target": None, "emitted": None, "abs_used": 0.0}
                            bwd_state[rid_k] = ent_k
                        prev_em = ent_k.get("emitted")
                        em_peak = (
                            float(prev_em.abs().max().item())
                            if isinstance(prev_em, torch.Tensor) and prev_em.numel() > 0
                            else 0.0
                        )
                        cur_used = min(_FP16_BWD_ROOT_MAX, float(ent_k["abs_used"]) + min(_FP16_BWD_ROOT_MAX, em_peak))
                        rem_budget = max(0.0, _FP16_BWD_ROOT_MAX - cur_used)
                        fp16_lim = min(fp16_lim, rem_budget / w_eff)
                        rid_eff.append((rid_k, ent_k, w_eff))
                    emitted_abs = min(g_abs_max, fp16_lim)
                    if emitted_abs > 0.0:
                        for _, ent_k, w_eff in rid_eff:
                            consumed = w_eff * emitted_abs
                            if consumed > 0.0:
                                ent_k["abs_used"] = min(
                                    _FP16_BWD_ROOT_MAX,
                                    float(ent_k["abs_used"]) + consumed,
                                )
        grad_output = _clamp_low_prec_grad(grad_output, orig_dtype, fp16_max=fp16_lim)
        return grad_output.to(dtype=orig_dtype), None, None, None


class _WorkToLowPrec(torch.autograd.Function):
    """Clamps forward values to target half-precision finite range before casting."""

    @staticmethod
    def forward(ctx, x_work: torch.Tensor, lp_dtype: torch.dtype) -> torch.Tensor:
        ctx.work_dtype = x_work.dtype
        ctx._out_numel = int(x_work.numel())
        if lp_dtype in _HALF_PREC_DTYPES:
            fmax = float(torch.finfo(lp_dtype).max)
            return x_work.clamp(-fmax, fmax).to(dtype=lp_dtype)
        return x_work.to(dtype=lp_dtype)

    @staticmethod
    def backward(ctx, grad_lp: torch.Tensor):
        if grad_lp is None:
            return None, None
        return grad_lp.to(dtype=ctx.work_dtype), None


class _RecoverWorkFromLowPrec(torch.autograd.Function):
    """Recovers exact working-precision tensor while routing bounded gradient through x_lp."""

    @staticmethod
    def forward(ctx, x_lp: torch.Tensor, x_work: torch.Tensor) -> torch.Tensor:
        ctx.lp_dtype = x_lp.dtype
        ctx.work_dtype = x_work.dtype
        return x_work.clone()

    @staticmethod
    def backward(ctx, grad_work: torch.Tensor):
        if grad_work is None:
            return None, None
        g_lp = _clamp_low_prec_grad(grad_work, ctx.lp_dtype).to(dtype=ctx.lp_dtype)
        g_res = grad_work - g_lp.to(dtype=ctx.work_dtype)
        return g_lp, g_res


def _has_valid_work_snap(x: torch.Tensor) -> bool:
    w = getattr(x, "_work_val", None)
    snap = getattr(x, "_work_snap", None)
    if not isinstance(w, torch.Tensor) or not isinstance(snap, torch.Tensor):
        return False
    if x.requires_grad and x.grad_fn is None:
        return False
    if snap.shape != x.shape or snap.device != x.device or snap.dtype != x.dtype:
        return False
    if w.shape != x.shape or w.device != x.device:
        return False
    return bool(torch.equal(x.detach(), snap))


def _bridge_work_via_lp(
    u_work: torch.Tensor, orig_dtype: Optional[torch.dtype]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bridges working-precision intermediate through a finite lp_dtype tensor preserving C++ autograd."""
    if orig_dtype not in _HALF_PREC_DTYPES or not u_work.dtype.is_floating_point:
        return u_work, u_work
    if u_work.dtype == orig_dtype and _has_valid_work_snap(u_work):
        return u_work._work_val, u_work  # type: ignore[attr-defined]
    u_lp = _WorkToLowPrec.apply(u_work, orig_dtype)
    u_use = _RecoverWorkFromLowPrec.apply(u_lp, u_work)
    u_lp._work_val = u_use
    u_lp._work_snap = u_lp.detach().clone()
    return u_use, u_lp


def _zero_stride_fanin(t: torch.Tensor) -> int:
    if t.layout != torch.strided or t.numel() <= 1:
        return 1
    fan = 1
    for sz, st in zip(t.shape, t.stride()):
        if st == 0 and sz > 1:
            fan *= int(sz)
    return fan


_STOP_FN_NAMES = (
    "AccumulateGrad",
    "_WorkToLowPrecBackward",
    "_SafeCastToWorkDtypeBackward",
    "_RecoverWorkFromLowPrecBackward",
)


def _is_fp16_stop_fn(cur_name: str) -> bool:
    return cur_name in _STOP_FN_NAMES


def _has_fp16_grad_input(fn: object) -> bool:
    metas = getattr(fn, "_input_metadata", None)
    if not metas:
        return False
    return any(getattr(m, "dtype", None) == torch.float16 for m in metas)


_UNARY_SAME_SHAPE_PASS = (
    "CloneBackward",
    "AliasBackward",
    "ContiguousBackward",
    "NegBackward",
)
_UNARY_VIEW_BIJECTIVE_PASS = (
    "ViewBackward",
    "UnsafeViewBackward",
    "ReshapeAliasBackward",
    "UnsqueezeBackward",
    "SqueezeBackward",
)
_UNARY_PERM_PASS = (
    "TransposeBackward",
    "PermuteBackward",
    "TBackward",
)
_UNARY_BIJECTIVE_PASS = _UNARY_VIEW_BIJECTIVE_PASS + _UNARY_PERM_PASS
_AFFINE_PRESERVE_OPS = (
    "CloneBackward",
    "AliasBackward",
    "ContiguousBackward",
    "ToCopyBackward",
)
_AFFINE_RESHAPE_SIGN_OPS = (
    "ViewBackward",
    "UnsafeViewBackward",
    "ReshapeAliasBackward",
    "UnsqueezeBackward",
    "SqueezeBackward",
    "TransposeBackward",
    "PermuteBackward",
    "TBackward",
    "FlipBackward",
    "ExpandBackward",
    "RepeatBackward",
    "SelectBackward",
    "SliceBackward",
    "SplitBackward",
    "SplitWithSizesBackward",
    "UnbindBackward",
    "IndexBackward",
    "IndexSelectBackward",
    "GatherBackward",
    "CatBackward",
    "StackBackward",
    "SumBackward",
)


def _node_shape_dtype(
    node: object, slot_nr: int = 0
) -> Tuple[Optional[Tuple[int, ...]], Optional[torch.dtype]]:
    cname = type(node).__name__
    if cname == "AccumulateGrad":
        var = getattr(node, "variable", None)
        if isinstance(var, torch.Tensor):
            return tuple(int(d) for d in var.shape), var.dtype
        return None, None
    metas = getattr(node, "_input_metadata", None)
    if metas and slot_nr < len(metas):
        m = metas[slot_nr]
        sh = getattr(m, "shape", None)
        dt = getattr(m, "dtype", None)
        return (tuple(int(d) for d in sh) if sh is not None else None), dt
    return None, None


def _finite_scalar_const(val: object) -> Optional[float]:
    if isinstance(val, (int, float)):
        fv = float(val)
        return fv if math.isfinite(fv) and fv != 0.0 else None
    if isinstance(val, torch.Tensor) and val.numel() == 1:
        fv = float(val.detach().item())
        return fv if math.isfinite(fv) and fv != 0.0 else None
    return None


def _peek_fp16_cast_return_onr(
    cur: object, max_hops: int = 4096
) -> Optional[Tuple[object, int, int]]:
    """
    If `cur` is a ToCopyBackward(fp16 -> non-fp16) that returns to float16 purely via
    shape-preserving single-child ToCopyBackward casts (e.g. `x.float().half()`),
    returns `(target_fp16_node, target_output_nr, hops)`; otherwise returns `None`.
    """
    cname = type(cur).__name__
    if not (cname.startswith("ToCopyBackward") and _has_fp16_grad_input(cur)):
        return None
    cur_sh, _ = _node_shape_dtype(cur, 0)
    p = cur
    mile = cur
    m_dist = 0
    hops = 0
    seen: set[int] = {id(cur)}
    while hops < max_hops:
        active = [(nf, onr) for nf, onr in p.next_functions if nf is not None]
        if len(active) != 1:
            return None
        nf, onr = active[0]
        onr_i = int(onr)
        nid = id(nf)
        if nid in seen:
            return None
        seen.add(nid)
        hops += 1
        mile, m_dist = _step_teardown_milestone(mile, m_dist, nf, single_child=True)
        nf_name = type(nf).__name__
        if nf_name == "AccumulateGrad":
            var = getattr(nf, "variable", None)
            if isinstance(var, torch.Tensor) and var.dtype == torch.float16:
                v_sh = tuple(int(d) for d in var.shape)
                if cur_sh is None or v_sh == cur_sh:
                    return nf, onr_i, hops
            return None
        nf_sh, nf_dt = _node_shape_dtype(nf, onr_i)
        if cur_sh is not None and nf_sh is not None and nf_sh != cur_sh:
            return None
        if nf_dt == torch.float16 or _has_fp16_grad_input(nf):
            return nf, onr_i, hops
        if not nf_name.startswith("ToCopyBackward"):
            return None
        if nf_dt not in (torch.float32, torch.float64, torch.bfloat16):
            return None
        p = nf
    return None


def _peek_fp16_cast_return(cur: object, max_hops: int = 4096) -> Optional[Tuple[object, int]]:
    res = _peek_fp16_cast_return_onr(cur, max_hops=max_hops)
    return (res[0], res[2]) if res is not None else None


_BOUNDED_NONAFFINE_UNARY_OPS = (
    "ReluBackward",
    "AbsBackward",
    "ClampBackward",
    "SigmoidBackward",
    "TanhBackward",
    "SinBackward",
    "CosBackward",
)


def _has_long_bijective_tail(start: object, min_hops: int = 16) -> bool:
    """Returns True if `start` leads a single-child unary/view/perm pass chain of >= min_hops."""
    cur = start
    seen: set[int] = set()
    hops = 0
    prefixes = _UNARY_SAME_SHAPE_PASS + _UNARY_BIJECTIVE_PASS
    while cur is not None and hops < min_hops:
        cid = id(cur)
        if cid in seen:
            return False
        seen.add(cid)
        cname = type(cur).__name__
        if (
            cname == "AccumulateGrad"
            or _is_fp16_stop_fn(cname)
            or _is_custom_py_bwd_node(cur)
        ):
            return False
        if not cname.startswith(prefixes):
            return False
        active = [nf for nf, _ in cur.next_functions if nf is not None]
        if len(active) != 1:
            return False
        cur = active[0]
        hops += 1
    return hops >= min_hops


def _advance_unary_chain_with_perm(
    start_node: object,
    unary_cache: Dict[Tuple[int, int], Tuple[object, int, float, float, bool, bool, bool]],
    hop_budget: List[int],
    start_onr: int = 0,
) -> Tuple[object, int, float, float, bool, bool, bool]:
    """
    Collapses single-child pass-through unary, bounded non-affine elementwise, round-trip dtype
    cast, contiguous view, and long (>=16 hop) permutation chains in linear hops, returning:
      (terminal_node, terminal_onr, signed_scale_product, max_prefix_fp16_abs_gain,
       all_same_shape, sign_known, crossed_perm)
    """
    cur = start_node
    cur_onr = int(start_onr)
    mile = start_node
    m_dist = 0
    trail: List[Tuple[Tuple[int, int], float, bool, bool, bool, bool]] = []
    seen: set[Tuple[int, int]] = set()
    tail_scale = 1.0
    tail_peak = 0.0
    tail_same = True
    tail_sign_ok = True
    tail_perm = False
    while cur is not None:
        cid = id(cur)
        ckey = (cid, cur_onr)
        cached = unary_cache.get(ckey)
        if cached is not None:
            if cur is not start_node and m_dist > 0 and mile is not None:
                _link_teardown_milestone(mile, cur, single_child=True)
            cur, cur_onr, tail_scale, tail_peak, tail_same, tail_sign_ok, tail_perm = cached
            break
        if ckey in seen or hop_budget[0] <= 0:
            break
        cur_name = type(cur).__name__
        if (
            cur_name == "AccumulateGrad"
            or _is_fp16_stop_fn(cur_name)
            or _is_custom_py_bwd_node(cur)
        ):
            break
        if cur_onr == 0 and cur_name.startswith("ToCopyBackward"):
            peek = _peek_fp16_cast_return_onr(cur)
            if peek is not None:
                nf_ret, onr_ret, p_hops = peek
                seen.add(ckey)
                trail.append((ckey, 1.0, True, False, True, False))
                hop_budget[0] -= p_hops
                if p_hops >= _TEARDOWN_ANCHOR_STEP or m_dist + p_hops >= _TEARDOWN_ANCHOR_STEP:
                    mile = _link_teardown_milestone(mile, nf_ret, single_child=True)
                    m_dist = 0
                else:
                    m_dist += p_hops
                cur, cur_onr = nf_ret, onr_ret
                continue
            active_tc = [(ei, nf, onr) for ei, (nf, onr) in enumerate(cur.next_functions) if nf is not None]
            if len(active_tc) == 1:
                _, nf_tc, onr_tc = active_tc[0]
                cur_sh, cur_dt = _node_shape_dtype(cur, 0)
                nxt_sh, nxt_dt = _node_shape_dtype(nf_tc, int(onr_tc))
                if cur_dt == torch.float16 and nxt_dt == torch.float16 and cur_sh is not None and cur_sh == nxt_sh:
                    seen.add(ckey)
                    trail.append((ckey, 1.0, True, True, True, False))
                    hop_budget[0] -= 1
                    mile, m_dist = _step_teardown_milestone(mile, m_dist, nf_tc, single_child=True)
                    cur, cur_onr = nf_tc, int(onr_tc)
                    continue
            break
        if cur_onr != 0:
            break
        active = [(ei, nf, onr) for ei, (nf, onr) in enumerate(cur.next_functions) if nf is not None]
        if len(active) != 1:
            break
        ei, nf0, onr0 = active[0]
        step_scale: Optional[float] = None
        step_same = False
        step_sign_ok = True
        step_perm = False
        is_fp16_step = False
        if cur_name.startswith(_UNARY_SAME_SHAPE_PASS):
            step_scale = -1.0 if cur_name.startswith("NegBackward") else 1.0
            step_same = True
            _, cur_dt = _node_shape_dtype(cur, 0)
            is_fp16_step = (cur_dt == torch.float16)
        elif cur_name.startswith(_UNARY_VIEW_BIJECTIVE_PASS):
            step_scale = 1.0
            step_same = False
            _, cur_dt = _node_shape_dtype(cur, 0)
            is_fp16_step = (cur_dt == torch.float16)
        elif cur_name.startswith(_UNARY_PERM_PASS):
            if len(trail) >= 16 or _has_long_bijective_tail(cur, max(1, 16 - len(trail))):
                step_scale = 1.0
                step_same = False
                step_perm = True
                _, cur_dt = _node_shape_dtype(cur, 0)
                is_fp16_step = (cur_dt == torch.float16)
        else:
            cur_sh, cur_dt = _node_shape_dtype(cur, 0)
            nxt_sh, nxt_dt = _node_shape_dtype(nf0, int(onr0))
            if (
                cur_dt in (torch.float16, torch.float32, torch.float64)
                and cur_dt == nxt_dt
                and cur_sh is not None
                and cur_sh == nxt_sh
            ):
                is_fp16_step = (cur_dt == torch.float16)
                if cur_name.startswith(_BOUNDED_NONAFFINE_UNARY_OPS):
                    step_scale = 1.0
                    step_same = True
                    step_sign_ok = False
                elif cur_name.startswith("MulBackward"):
                    c_val = _finite_scalar_const(
                        _safe_saved_attr(cur, "_saved_other" if ei == 0 else "_saved_self")
                    )
                    if c_val is not None:
                        step_scale = c_val
                        step_same = True
                elif cur_name.startswith("DivBackward") and ei == 0:
                    c_val = _finite_scalar_const(_safe_saved_attr(cur, "_saved_other"))
                    if c_val is not None:
                        step_scale = 1.0 / c_val
                        step_same = True
                elif cur_name.startswith("AddBackward"):
                    raw_a = _safe_saved_attr(cur, "_saved_alpha") if ei == 1 else 1.0
                    alpha = float(raw_a) if raw_a is not None else 1.0
                    if math.isfinite(alpha) and alpha != 0.0:
                        step_scale = alpha
                        step_same = True
                elif cur_name.startswith("SubBackward"):
                    raw_a = _safe_saved_attr(cur, "_saved_alpha") if ei == 1 else 1.0
                    alpha = float(raw_a) if raw_a is not None else 1.0
                    if math.isfinite(alpha) and alpha != 0.0:
                        step_scale = -alpha if ei == 1 else alpha
                        step_same = True
                elif cur_name.startswith("RsubBackward"):
                    raw_a = _safe_saved_attr(cur, "_saved_alpha") if ei == 0 else 1.0
                    alpha = float(raw_a) if raw_a is not None else 1.0
                    if math.isfinite(alpha) and alpha != 0.0:
                        step_scale = -alpha if ei == 0 else alpha
                        step_same = True
        if step_scale is None:
            break
        seen.add(ckey)
        trail.append((ckey, step_scale, step_same, is_fp16_step, step_sign_ok, step_perm))
        hop_budget[0] -= 1
        mile, m_dist = _step_teardown_milestone(mile, m_dist, nf0, single_child=True)
        cur, cur_onr = nf0, int(onr0)

    acc_scale = tail_scale
    acc_peak = tail_peak
    acc_same = tail_same
    acc_sign_ok = tail_sign_ok
    acc_perm = tail_perm
    for ckey, step_scale, step_same, is_fp16_step, step_sign_ok, step_perm in reversed(trail):
        s_abs = abs(step_scale)
        if is_fp16_step:
            acc_peak = s_abs * max(1.0, acc_peak)
        else:
            acc_peak = s_abs * acc_peak
        acc_scale = step_scale * acc_scale
        acc_same = acc_same and step_same
        acc_sign_ok = acc_sign_ok and step_sign_ok
        acc_perm = acc_perm or step_perm
        unary_cache[ckey] = (cur, cur_onr, acc_scale, acc_peak, acc_same, acc_sign_ok, acc_perm)
    return cur, cur_onr, acc_scale, acc_peak, acc_same, acc_sign_ok, acc_perm


def _advance_unary_chain(
    start_node: object,
    unary_cache: Dict[int, Tuple[object, float, float, bool, bool]],
    hop_budget: List[int],
) -> Tuple[object, float, float, bool, bool]:
    ext_cache: Dict[Tuple[int, int], Tuple[object, int, float, float, bool, bool, bool]] = {
        (k, 0): (v[0], 0, v[1], v[2], v[3], v[4], False)
        for k, v in unary_cache.items()
    }
    cur, _, sc, pk, sm, sok, _ = _advance_unary_chain_with_perm(start_node, ext_cache, hop_budget, 0)
    for (k, onr), v in ext_cache.items():
        if onr == 0:
            unary_cache[k] = (v[0], v[2], v[3], v[4], v[5])
    return cur, sc, pk, sm, sok


def _is_fp16_cast_leaf_boundary(cur: object, cur_name: str) -> bool:
    if not (cur_name.startswith("ToCopyBackward") and _has_fp16_grad_input(cur)):
        return False
    if _peek_fp16_cast_return_onr(cur) is not None:
        return False
    nxt_fns = [nf for nf, _ in cur.next_functions if nf is not None]
    return (not nxt_fns) or not all(
        _has_fp16_grad_input(nf)
        or (
            type(nf).__name__ == "AccumulateGrad"
            and getattr(getattr(nf, "variable", None), "dtype", None) == torch.float16
        )
        for nf in nxt_fns
    )


def _normalize_signed_int(val: Any) -> Optional[int]:
    if val is None or isinstance(val, bool):
        return None
    try:
        iv = int(val)
    except Exception:
        return None
    if iv >= (1 << 63):
        iv -= (1 << 64)
    return iv


def _extract_index_backward_specs(
    cur: object,
    self_sh: Tuple[int, ...],
    max_elems: int = 4_194_304,
    max_bool_elems: int = 67_108_864,
) -> Optional[Tuple[List[int], List[int], List[torch.Tensor], Tuple[int, ...]]]:
    """
    Validates `IndexBackward._saved_indices` against `self_sh` and returns
    `(sub_dims, indexed_dims, flat_cols, b_sh)` in O(output_numel), or `None`
    if uninspectable/exceeds limits. `flat_cols == []` means no tensor indices.
    """
    idxs = _safe_saved_attr(cur, "_saved_indices")
    if idxs is None:
        return None
    sub_dims: List[int] = []
    indexed_dims: List[int] = []
    int_idxs: List[torch.Tensor] = []
    dim_i = 0
    for item in idxs:
        if item is None:
            dim_i += 1
            if dim_i > len(self_sh):
                return None
        elif isinstance(item, torch.Tensor):
            if getattr(item, "is_meta", False):
                return None
            if item.dtype == torch.bool:
                if item.numel() > max_bool_elems or dim_i + item.ndim > len(self_sh):
                    return None
                nz = item.detach().to(device="cpu").nonzero(as_tuple=True)
                if nz and nz[0].numel() > max_elems:
                    return None
                for off in range(item.ndim):
                    sub_dims.append(int(self_sh[dim_i + off]))
                    indexed_dims.append(dim_i + off)
                dim_i += item.ndim
                int_idxs.extend(t.to(dtype=torch.int64) for t in nz)
            else:
                if item.numel() > max_elems or dim_i >= len(self_sh):
                    return None
                d_len = int(self_sh[dim_i])
                sub_dims.append(d_len)
                indexed_dims.append(dim_i)
                dim_i += 1
                idx_cpu = item.detach().to(device="cpu", dtype=torch.int64)
                if d_len > 0 and idx_cpu.numel() > 0:
                    idx_cpu = torch.where(idx_cpu < 0, idx_cpu + d_len, idx_cpu)
                    if bool((idx_cpu < 0).any().item()) or bool((idx_cpu >= d_len).any().item()):
                        return None
                int_idxs.append(idx_cpu)
        else:
            return None

    if not int_idxs:
        return [], [], [], ()
    if any(d <= 0 for d in sub_dims):
        return [], [], [], ()
    b_sh = torch.broadcast_shapes(*(t.shape for t in int_idxs))
    b_numel = math.prod(b_sh)
    if b_numel > max_elems:
        return None
    if b_numel == 0:
        return [], [], [], ()
    cols = [t.expand(b_sh).reshape(-1) for t in int_idxs]
    return sub_dims, indexed_dims, cols, tuple(int(d) for d in b_sh)


def _exact_index_max_dup(
    cur: object,
    cur_name: str,
    out_sh: Optional[Tuple[int, ...]],
    self_sh: Optional[Tuple[int, ...]],
    fallback_mult: float,
) -> float:
    """
    Computes exact maximum backward gradient scatter multiplicity into `self` for
    `IndexBackward`, `IndexSelectBackward`, and `GatherBackward` in O(accessed_elems),
    independent of `self` tensor size.
    """
    if self_sh is None:
        return fallback_mult
    if any(int(d) == 0 for d in self_sh) or (out_sh is not None and any(int(d) == 0 for d in out_sh)):
        return 1.0
    max_elems = 4_194_304
    max_bool_elems = 67_108_864
    try:
        if cur_name.startswith("IndexSelectBackward"):
            d_v = _normalize_signed_int(_safe_saved_attr(cur, "_saved_dim"))
            idx_t = _safe_saved_attr(cur, "_saved_index")
            if d_v is None or not isinstance(idx_t, torch.Tensor) or getattr(idx_t, "is_meta", False):
                return fallback_mult
            if idx_t.numel() == 0:
                return 1.0
            if idx_t.numel() > max_elems or not self_sh:
                return fallback_mult
            dim = d_v + len(self_sh) if d_v < 0 else d_v
            if dim < 0 or dim >= len(self_sh):
                return fallback_mult
            dim_len = int(self_sh[dim])
            if dim_len <= 0:
                return 1.0
            idx_cpu = idx_t.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
            idx_cpu = torch.where(idx_cpu < 0, idx_cpu + dim_len, idx_cpu)
            if bool((idx_cpu < 0).any().item()) or bool((idx_cpu >= dim_len).any().item()):
                return fallback_mult
            _, counts = torch.unique(idx_cpu, return_counts=True)
            return float(max(1, int(counts.max().item())))

        if cur_name.startswith("GatherBackward"):
            d_v = _normalize_signed_int(_safe_saved_attr(cur, "_saved_dim"))
            idx_t = _safe_saved_attr(cur, "_saved_index")
            if d_v is None or not isinstance(idx_t, torch.Tensor) or getattr(idx_t, "is_meta", False):
                return fallback_mult
            if idx_t.numel() == 0:
                return 1.0
            if idx_t.numel() > max_elems or not self_sh:
                return fallback_mult
            dim = d_v + len(self_sh) if d_v < 0 else d_v
            if dim < 0 or dim >= len(self_sh) or idx_t.ndim != len(self_sh):
                return fallback_mult
            dim_len = int(self_sh[dim])
            if dim_len <= 0:
                return 1.0
            m_d = int(idx_t.shape[dim])
            if m_d <= 1:
                return 1.0
            idx_cpu = idx_t.detach().to(device="cpu", dtype=torch.int64)
            idx_cpu = torch.where(idx_cpu < 0, idx_cpu + dim_len, idx_cpu)
            if bool((idx_cpu < 0).any().item()) or bool((idx_cpu >= dim_len).any().item()):
                return fallback_mult
            idx_2d = idx_cpu.movedim(dim, -1).reshape(-1, m_d).contiguous()
            sorted_idx, _ = torch.sort(idx_2d, dim=-1)
            same_prev = torch.zeros_like(sorted_idx, dtype=torch.bool)
            same_prev[:, 1:] = sorted_idx[:, 1:] == sorted_idx[:, :-1]
            starts = (~same_prev).reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            ends = torch.cat([starts[1:], torch.tensor([int(sorted_idx.numel())], dtype=torch.int64)])
            return float(max(1, int((ends - starts).max().item())))

        if cur_name.startswith("IndexBackward"):
            parsed = _extract_index_backward_specs(cur, self_sh, max_elems=max_elems, max_bool_elems=max_bool_elems)
            if parsed is None:
                return fallback_mult
            sub_dims, _, cols, _ = parsed
            if not cols:
                return 1.0
            if len(cols) == 1:
                _, counts = torch.unique(cols[0], return_counts=True)
                return float(max(1, int(counts.max().item())))
            if math.prod(sub_dims) <= (1 << 62):
                lin = cols[-1].clone()
                stride = int(sub_dims[-1])
                for j in range(len(cols) - 2, -1, -1):
                    lin = lin + cols[j] * stride
                    stride *= int(sub_dims[j])
                _, counts = torch.unique(lin, return_counts=True)
            else:
                coords = torch.stack(cols, dim=1)
                _, counts = torch.unique(coords, dim=0, return_counts=True)
            return float(max(1, int(counts.max().item())))
    except Exception:
        return fallback_mult
    return fallback_mult


_MAX_GAIN_DENSE = 262_144
_MAX_GAIN_SPARSE = 262_144
_MAX_GAIN_BOXES = 64
_MAX_SLAB_CELLS = 16_384
_MAX_SLAB_PERIOD = 256


def _canon_slice(s: int, e: int, st: int, dim_len: int) -> Optional[Tuple[int, int, int]]:
    if dim_len <= 0 or st <= 0:
        return None
    s_c = 0 if s < 0 else s
    e_c = dim_len if e > dim_len else e
    if s_c >= e_c:
        return None
    cnt = 1 + (e_c - 1 - s_c) // st
    last = s_c + (cnt - 1) * st
    return (s_c, last + 1, 1 if cnt == 1 else st)


def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def _ap_intersect(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    """Exact O(1) intersection test for two canonical 1D arithmetic progressions."""
    s1, e1, st1 = a
    s2, e2, st2 = b
    lo = s1 if s1 >= s2 else s2
    hi = e1 if e1 <= e2 else e2
    if lo >= hi:
        return False
    if st1 == 1 and st2 == 1:
        return True
    if st1 == 1:
        return (lo + ((s2 - lo) % st2)) < hi
    if st2 == 1:
        return (lo + ((s1 - lo) % st1)) < hi
    g_d, u, _ = _egcd(st1, st2)
    diff = s2 - s1
    if diff % g_d != 0:
        return False
    m2 = st2 // g_d
    lcm = (st1 // g_d) * st2
    x0 = s1 + st1 * (((diff // g_d) * u) % m2)
    return (lo + ((x0 - lo) % lcm)) < hi


def _gain_zero(sh: Tuple[int, ...]) -> Dict[str, Any]:
    return {"kind": "boxes", "shape": sh, "items": []}


def _gain_full(sh: Tuple[int, ...], val: float = 1.0) -> Optional[Dict[str, Any]]:
    fv = abs(float(val))
    if not math.isfinite(fv):
        return None
    if fv == 0.0 or any(int(d) <= 0 for d in sh):
        return _gain_zero(sh)
    return {"kind": "boxes", "shape": sh, "items": [(tuple((0, int(d), 1) for d in sh), fv)]}


def _gain_is_zero(g: Dict[str, Any]) -> bool:
    k = g["kind"]
    if k == "boxes":
        return not g["items"]
    if k == "sparse":
        return int(g["vals"].numel()) == 0
    if k == "dense":
        return True if g["t"].numel() == 0 else not bool((g["t"] > 0.0).any().item())
    return _gain_is_zero(g["boxes"]) and _gain_is_zero(g["sparse"])


def _coalesce_boxes(
    sh: Tuple[int, ...], items: List[Tuple[Tuple[Tuple[int, int, int], ...], float]]
) -> List[Tuple[Tuple[Tuple[int, int, int], ...], float]]:
    if not items:
        return []
    acc: Dict[Tuple[Tuple[int, int, int], ...], float] = {}
    for box, val in items:
        if val > 0.0:
            acc[box] = acc.get(box, 0.0) + float(val)
    cur = list(acc.items())
    rank = len(sh)
    if rank > 0 and len(cur) > 1:
        changed = True
        while changed and len(cur) > 1:
            changed = False
            for d in range(rank):
                groups: Dict[Tuple[Any, ...], List[Tuple[int, int, Tuple[Tuple[int, int, int], ...], float]]] = {}
                passthrough: List[Tuple[Tuple[Tuple[int, int, int], ...], float]] = []
                for box, val in cur:
                    bs, be, bst = box[d]
                    if bst != 1:
                        passthrough.append((box, val))
                        continue
                    key = (box[:d] + box[d + 1 :], val)
                    groups.setdefault(key, []).append((bs, be, box, val))
                next_cur = list(passthrough)
                for (_, val), segs in groups.items():
                    if len(segs) == 1:
                        next_cur.append((segs[0][2], val))
                        continue
                    segs.sort(key=lambda it: (it[0], it[1]))
                    ms, me, mbox, _ = segs[0]
                    for s_i, e_i, box_i, _ in segs[1:]:
                        if s_i == me:
                            me = e_i
                            changed = True
                        else:
                            nb = mbox[:d] + ((ms, me, 1),) + mbox[d + 1 :]
                            next_cur.append((nb, val))
                            ms, me, mbox = s_i, e_i, box_i
                    nb = mbox[:d] + ((ms, me, 1),) + mbox[d + 1 :]
                    next_cur.append((nb, val))
                cur = next_cur
    return cur


def _boxes_max(
    sh: Tuple[int, ...], items: List[Tuple[Tuple[Tuple[int, int, int], ...], float]]
) -> float:
    """Computes exact maximum over overlapping boxes using components and compressed slabs."""
    if not items:
        return 0.0
    if len(sh) == 0:
        return float(sum(v for _, v in items))
    if len(items) == 1:
        return float(max(0.0, items[0][1]))

    n = len(items)
    adj: List[List[int]] = [[] for _ in range(n)]
    boxes_only = [b for b, _ in items]
    for i in range(n):
        bi = boxes_only[i]
        for j in range(i + 1, n):
            bj = boxes_only[j]
            if all(_ap_intersect(ad, bd) for ad, bd in zip(bi, bj)):
                adj[i].append(j)
                adj[j].append(i)

    seen = [False] * n
    best = 0.0
    rank = len(sh)
    for start in range(n):
        if seen[start]:
            continue
        comp: List[int] = [start]
        seen[start] = True
        qi = 0
        while qi < len(comp):
            u = comp[qi]
            qi += 1
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    comp.append(w)
        if len(comp) == 1:
            best = max(best, float(items[comp[0]][1]))
            continue
        comp_items = [items[i] for i in comp]
        comp_sum = float(sum(v for _, v in comp_items))
        if comp_sum <= best:
            continue

        grid_ok = True
        axes_pts: List[List[int]] = []
        total_cells = 1
        for d in range(rank):
            triples = {b[d] for b, _ in comp_items}
            if len(triples) == 1:
                s0 = next(iter(triples))[0]
                pts_d = [s0]
            else:
                period = 1
                for _, _, st in triples:
                    period = math.lcm(period, st)
                    if period > _MAX_SLAB_PERIOD:
                        grid_ok = False
                        break
                if not grid_ok:
                    break
                cuts = sorted({c for s, e, _ in triples for c in (s, e)})
                pts_d = []
                for a_c, b_c in zip(cuts[:-1], cuts[1:]):
                    if a_c >= b_c:
                        continue
                    lim_c = b_c if b_c < a_c + period else (a_c + period)
                    for x_c in range(a_c, lim_c):
                        if any(s <= x_c < e and ((x_c - s) % st == 0) for s, e, st in triples):
                            pts_d.append(x_c)
                            if total_cells * len(pts_d) > _MAX_SLAB_CELLS:
                                grid_ok = False
                                break
                    if not grid_ok:
                        break
            if not grid_ok or not pts_d:
                break
            total_cells *= len(pts_d)
            axes_pts.append(pts_d)

        if not grid_ok:
            best = max(best, comp_sum)
            continue
        if len(axes_pts) < rank:
            continue
        if total_cells == 1:
            p0 = tuple(pts[0] for pts in axes_pts)
            val_0 = sum(
                v
                for b, v in comp_items
                if all(s <= c < e and ((c - s) % st == 0) for c, (s, e, st) in zip(p0, b))
            )
            best = max(best, float(val_0))
            continue

        slab_shape = tuple(len(p) for p in axes_pts)
        grid = torch.zeros(slab_shape, dtype=torch.float64)
        for b, v in comp_items:
            ax_idxs = [
                torch.tensor(
                    [i for i, x_c in enumerate(axes_pts[d]) if s <= x_c < e and ((x_c - s) % st == 0)],
                    dtype=torch.long,
                )
                for d, (s, e, st) in enumerate(b)
            ]
            if any(ix.numel() == 0 for ix in ax_idxs):
                continue
            mg = torch.meshgrid(*ax_idxs, indexing="ij")
            grid[mg] += float(v)
        best = max(best, float(grid.max().item()))
    return best


def _coalesce_sparse(
    sh: Tuple[int, ...], coords: torch.Tensor, vals: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if vals.numel() == 0:
        return torch.empty((0, len(sh)), dtype=torch.int64), torch.empty((0,), dtype=torch.float64)
    pos = vals > 0.0
    if not bool(pos.all().item()):
        coords = coords[pos]
        vals = vals[pos]
    if vals.numel() <= 1:
        return coords, vals
    rank = len(sh)
    if rank == 0:
        return torch.empty((1, 0), dtype=torch.int64), vals.sum().view(1)
    if math.prod(sh) <= (1 << 62):
        lin = coords[:, -1].clone()
        stride = max(1, int(sh[-1]))
        for j in range(rank - 2, -1, -1):
            lin = lin + coords[:, j] * stride
            stride *= max(1, int(sh[j]))
        _, sort_idx = torch.sort(lin)
        s_coords = coords[sort_idx]
        s_vals = vals[sort_idx]
        _, inv, counts = torch.unique_consecutive(lin[sort_idx], return_inverse=True, return_counts=True)
        n_u = int(counts.numel())
        if n_u == int(s_vals.numel()):
            return s_coords, s_vals
        first_mask = torch.ones_like(inv, dtype=torch.bool)
        if inv.numel() > 1:
            first_mask[1:] = inv[1:] != inv[:-1]
        u_coords = s_coords[first_mask]
    else:
        u_coords, inv = torch.unique(coords, dim=0, return_inverse=True)
        n_u = int(u_coords.shape[0])
        s_vals = vals
    u_vals = torch.zeros((n_u,), dtype=torch.float64)
    u_vals.index_add_(0, inv, s_vals)
    return u_coords, u_vals


def _gain_to_dense(g: Dict[str, Any]) -> Dict[str, Any]:
    k = g["kind"]
    sh = g["shape"]
    if k == "dense":
        return g
    t = torch.zeros(sh, dtype=torch.float64)
    if k == "boxes":
        for box, val in g["items"]:
            if len(sh) == 0:
                t = t + float(val)
            else:
                sl = tuple(slice(s, e, st) for s, e, st in box)
                t[sl] += float(val)
    elif k == "sparse":
        coords, vals = _coalesce_sparse(sh, g["coords"], g["vals"])
        if vals.numel() > 0:
            if len(sh) == 0:
                t = t + float(vals.sum().item())
            else:
                t.index_put_(tuple(coords[:, d] for d in range(len(sh))), vals, accumulate=True)
    else:
        t = _gain_to_dense(g["boxes"])["t"] + _gain_to_dense(g["sparse"])["t"]
    return {"kind": "dense", "shape": sh, "t": t}


def _dense_to_sparse(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = g["t"]
    sh = g["shape"]
    if t.numel() == 0:
        return _gain_zero(sh)
    nz = torch.nonzero(t > 0.0, as_tuple=False)
    if nz.shape[0] == 0:
        return _gain_zero(sh)
    if nz.shape[0] > _MAX_GAIN_SPARSE:
        return None
    vals = t[tuple(nz[:, d] for d in range(len(sh)))].to(dtype=torch.float64)
    return {"kind": "sparse", "shape": sh, "coords": nz.to(dtype=torch.int64), "vals": vals}


def _gain_scale(g: Optional[Dict[str, Any]], factor: float) -> Optional[Dict[str, Any]]:
    if g is None:
        return None
    f = abs(float(factor))
    if not math.isfinite(f):
        return None
    if f == 0.0:
        return _gain_zero(g["shape"])
    if f == 1.0:
        return g
    k = g["kind"]
    if k == "dense":
        return {"kind": "dense", "shape": g["shape"], "t": g["t"] * f}
    if k == "boxes":
        return {"kind": "boxes", "shape": g["shape"], "items": [(b, v * f) for b, v in g["items"] if v * f > 0.0]}
    if k == "sparse":
        return {"kind": "sparse", "shape": g["shape"], "coords": g["coords"], "vals": g["vals"] * f}
    return {
        "kind": "mixed",
        "shape": g["shape"],
        "boxes": _gain_scale(g["boxes"], f),
        "sparse": _gain_scale(g["sparse"], f),
    }


def _gain_max(g: Dict[str, Any]) -> float:
    k = g["kind"]
    if k == "dense":
        t = g["t"]
        return float(max(0.0, t.max().item())) if t.numel() > 0 else 0.0
    if k == "boxes":
        return _boxes_max(g["shape"], g["items"])
    if k == "sparse":
        _, vals = _coalesce_sparse(g["shape"], g["coords"], g["vals"])
        return float(max(0.0, vals.max().item())) if vals.numel() > 0 else 0.0
    best = _gain_max(g["boxes"])
    coords, vals = _coalesce_sparse(g["shape"], g["sparse"]["coords"], g["sparse"]["vals"])
    if vals.numel() > 0:
        tot = vals.clone()
        for box, bval in g["boxes"]["items"]:
            hit = torch.ones((coords.shape[0],), dtype=torch.bool)
            for d, (s, e, st) in enumerate(box):
                cd = coords[:, d]
                hit &= (cd >= s) & (cd < e)
                if st > 1:
                    hit &= ((cd - s) % st) == 0
            tot[hit] += float(bval)
        best = max(best, float(tot.max().item()))
    return best


def _gain_add(g1: Optional[Dict[str, Any]], g2: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if g1 is None or g2 is None:
        return None
    if _gain_is_zero(g1):
        return g2
    if _gain_is_zero(g2):
        return g1
    sh = g1["shape"]
    k1, k2 = g1["kind"], g2["kind"]
    if k1 == "dense" or k2 == "dense":
        if math.prod(sh) <= _MAX_GAIN_DENSE:
            return {"kind": "dense", "shape": sh, "t": _gain_to_dense(g1)["t"] + _gain_to_dense(g2)["t"]}
        s1 = _dense_to_sparse(g1) if k1 == "dense" else g1
        s2 = _dense_to_sparse(g2) if k2 == "dense" else g2
        return _gain_add(s1, s2)
    if k1 == "boxes" and k2 == "boxes":
        merged = _coalesce_boxes(sh, g1["items"] + g2["items"])
        if len(merged) <= _MAX_GAIN_BOXES:
            return {"kind": "boxes", "shape": sh, "items": merged}
        if math.prod(sh) <= _MAX_GAIN_DENSE:
            return _gain_to_dense({"kind": "boxes", "shape": sh, "items": merged})
        m_val = _boxes_max(sh, merged)
        if not merged or m_val <= 0.0:
            return _gain_zero(sh)
        bbox = tuple(
            (min(b[d][0] for b, _ in merged), max(b[d][1] for b, _ in merged), 1)
            for d in range(len(sh))
        )
        return {"kind": "boxes", "shape": sh, "items": [(bbox, m_val)]}
    if k1 == "sparse" and k2 == "sparse":
        cat_c = torch.cat([g1["coords"], g2["coords"]], dim=0)
        cat_v = torch.cat([g1["vals"], g2["vals"]], dim=0)
        if cat_v.numel() > _MAX_GAIN_SPARSE:
            cat_c, cat_v = _coalesce_sparse(sh, cat_c, cat_v)
        if cat_v.numel() <= _MAX_GAIN_SPARSE:
            return {"kind": "sparse", "shape": sh, "coords": cat_c, "vals": cat_v}
        if math.prod(sh) <= _MAX_GAIN_DENSE:
            t = torch.zeros(sh, dtype=torch.float64)
            t.index_put_(tuple(cat_c[:, d] for d in range(len(sh))), cat_v, accumulate=True)
            return {"kind": "dense", "shape": sh, "t": t}
        return None
    if k1 == "mixed" or k2 == "mixed":
        b1 = g1["boxes"] if k1 == "mixed" else (g1 if k1 == "boxes" else _gain_zero(sh))
        b2 = g2["boxes"] if k2 == "mixed" else (g2 if k2 == "boxes" else _gain_zero(sh))
        s1 = g1["sparse"] if k1 == "mixed" else (g1 if k1 == "sparse" else _gain_zero(sh))
        s2 = g2["sparse"] if k2 == "mixed" else (g2 if k2 == "sparse" else _gain_zero(sh))
        if k1 == "sparse":
            s1 = g1
        if k2 == "sparse":
            s2 = g2
        b_m = _gain_add(b1, b2)
        if _gain_is_zero(s1):
            s_m = s2 if s2["kind"] == "sparse" else None
        elif _gain_is_zero(s2):
            s_m = s1 if s1["kind"] == "sparse" else None
        else:
            s_m = _gain_add(s1, s2)
        if b_m is None or s_m is None:
            return None
        if s_m["kind"] != "sparse" or _gain_is_zero(s_m):
            return b_m if _gain_is_zero(s_m) else _gain_add(b_m, s_m)
        if _gain_is_zero(b_m):
            return s_m
        return {"kind": "mixed", "shape": sh, "boxes": b_m, "sparse": s_m}
    bg = g1 if k1 == "boxes" else g2
    sg = g2 if k1 == "boxes" else g1
    return {"kind": "mixed", "shape": sh, "boxes": bg, "sparse": sg}


def _map_gain(g: Optional[Dict[str, Any]], fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if g is None:
        return None
    if g["kind"] != "mixed":
        return fn(g)
    bm = fn(g["boxes"])
    sm = fn(g["sparse"])
    return _gain_add(bm, sm)


def _gain_slice_embed(
    g: Optional[Dict[str, Any]], dim: int, start: int, end: int, step: int, target_sh: Tuple[int, ...]
) -> Optional[Dict[str, Any]]:
    def _do(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if _gain_is_zero(sub):
            return _gain_zero(target_sh)
        s, e, st = slice(start, end, step).indices(target_sh[dim])
        expected_len = max(0, (e - s + st - 1) // st) if st > 0 else 0
        if expected_len != sub["shape"][dim]:
            return None
        k = sub["kind"]
        if k == "dense":
            if math.prod(target_sh) <= _MAX_GAIN_DENSE:
                out = torch.zeros(target_sh, dtype=torch.float64)
                sl = [slice(None)] * len(target_sh)
                sl[dim] = slice(s, e, st)
                out[tuple(sl)] = sub["t"]
                return {"kind": "dense", "shape": target_sh, "t": out}
            sp = _dense_to_sparse(sub)
            return _gain_slice_embed(sp, dim, start, end, step, target_sh) if sp is not None else None
        if k == "boxes":
            new_items: List[Tuple[Tuple[Tuple[int, int, int], ...], float]] = []
            for box, val in sub["items"]:
                bs, be, bst = box[dim]
                new_d = _canon_slice(s + bs * st, s + (be - 1) * st + 1, bst * st, target_sh[dim])
                if new_d is not None:
                    new_items.append((box[:dim] + (new_d,) + box[dim + 1 :], val))
            return {"kind": "boxes", "shape": target_sh, "items": new_items}
        nc = sub["coords"].clone()
        nc[:, dim] = s + st * nc[:, dim]
        return {"kind": "sparse", "shape": target_sh, "coords": nc, "vals": sub["vals"]}
    return _map_gain(g, _do)


def _gain_select_embed(
    g: Optional[Dict[str, Any]], dim: int, index: int, target_sh: Tuple[int, ...]
) -> Optional[Dict[str, Any]]:
    def _do(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if _gain_is_zero(sub):
            return _gain_zero(target_sh)
        k = sub["kind"]
        if k == "dense":
            if math.prod(target_sh) <= _MAX_GAIN_DENSE:
                out = torch.zeros(target_sh, dtype=torch.float64)
                out.select(dim, index).copy_(sub["t"])
                return {"kind": "dense", "shape": target_sh, "t": out}
            sp = _dense_to_sparse(sub)
            return _gain_select_embed(sp, dim, index, target_sh) if sp is not None else None
        if k == "boxes":
            pt = (index, index + 1, 1)
            return {
                "kind": "boxes",
                "shape": target_sh,
                "items": [(b[:dim] + (pt,) + b[dim:], v) for b, v in sub["items"]],
            }
        c = sub["coords"]
        col = torch.full((c.shape[0], 1), index, dtype=torch.int64)
        nc = torch.cat([c[:, :dim], col, c[:, dim:]], dim=1)
        return {"kind": "sparse", "shape": target_sh, "coords": nc, "vals": sub["vals"]}
    return _map_gain(g, _do)


def _gain_split_axis(
    g: Optional[Dict[str, Any]], dim: int, start: int, length: int
) -> Optional[Dict[str, Any]]:
    def _do(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sh = sub["shape"]
        out_sh = sh[:dim] + (length,) + sh[dim + 1 :]
        if length == 0 or _gain_is_zero(sub):
            return _gain_zero(out_sh)
        k = sub["kind"]
        if k == "dense":
            return {"kind": "dense", "shape": out_sh, "t": sub["t"].narrow(dim, start, length).clone()}
        if k == "boxes":
            end = start + length
            new_items: List[Tuple[Tuple[Tuple[int, int, int], ...], float]] = []
            for box, val in sub["items"]:
                bs, be, bst = box[dim]
                lo = bs if bs >= start else start
                hi = be if be <= end else end
                if lo >= hi:
                    continue
                first = lo + ((bs - lo) % bst)
                if first >= hi:
                    continue
                new_d = _canon_slice(first - start, hi - start, bst, length)
                if new_d is not None:
                    new_items.append((box[:dim] + (new_d,) + box[dim + 1 :], val))
            return {"kind": "boxes", "shape": out_sh, "items": new_items}
        c = sub["coords"]
        m = (c[:, dim] >= start) & (c[:, dim] < start + length)
        nc = c[m].clone()
        nc[:, dim] -= start
        return {"kind": "sparse", "shape": out_sh, "coords": nc, "vals": sub["vals"][m]}
    return _map_gain(g, _do)


def _gain_cat_chunks(
    chunks: List[Dict[str, Any]], dim: int, target_sh: Tuple[int, ...]
) -> Optional[Dict[str, Any]]:
    if not chunks:
        return _gain_zero(target_sh)
    if all(_gain_is_zero(c) for c in chunks):
        return _gain_zero(target_sh)
    if math.prod(target_sh) <= _MAX_GAIN_DENSE and any(c["kind"] == "dense" for c in chunks):
        return {
            "kind": "dense",
            "shape": target_sh,
            "t": torch.cat([_gain_to_dense(c)["t"] for c in chunks], dim=dim),
        }
    acc: Optional[Dict[str, Any]] = _gain_zero(target_sh)
    off = 0
    for ch in chunks:
        clen = int(ch["shape"][dim])
        if not _gain_is_zero(ch):
            def _shift(sub: Dict[str, Any], o: int = off) -> Optional[Dict[str, Any]]:
                if _gain_is_zero(sub):
                    return _gain_zero(target_sh)
                if sub["kind"] == "dense":
                    sp = _dense_to_sparse(sub)
                    return _shift(sp, o) if sp is not None else None
                if sub["kind"] == "boxes":
                    ni = []
                    for box, val in sub["items"]:
                        bs, be, bst = box[dim]
                        ni.append((box[:dim] + ((bs + o, be + o, bst),) + box[dim + 1 :], val))
                    return {"kind": "boxes", "shape": target_sh, "items": ni}
                nc = sub["coords"].clone()
                nc[:, dim] += o
                return {"kind": "sparse", "shape": target_sh, "coords": nc, "vals": sub["vals"]}
            shifted = _map_gain(ch, _shift)
            acc = _gain_add(acc, shifted)
            if acc is None:
                return None
        off += clen
    return acc


def _gain_permute(g: Optional[Dict[str, Any]], fwd_dims: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
    def _do(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        old_sh = sub["shape"]
        rank = len(old_sh)
        if len(fwd_dims) != rank:
            return None
        inv_perm = [0] * rank
        new_sh_l = [0] * rank
        for k_ax, d_in in enumerate(fwd_dims):
            inv_perm[d_in] = k_ax
            new_sh_l[d_in] = int(old_sh[k_ax])
        new_sh = tuple(new_sh_l)
        if _gain_is_zero(sub):
            return _gain_zero(new_sh)
        k = sub["kind"]
        if k == "dense":
            return {"kind": "dense", "shape": new_sh, "t": sub["t"].permute(inv_perm).contiguous()}
        if k == "boxes":
            ni = []
            for box, val in sub["items"]:
                nb = [box[inv_perm[d]] for d in range(rank)]
                ni.append((tuple(nb), val))
            return {"kind": "boxes", "shape": new_sh, "items": ni}
        c = sub["coords"]
        nc = torch.stack([c[:, inv_perm[d]] for d in range(rank)], dim=1)
        return {"kind": "sparse", "shape": new_sh, "coords": nc, "vals": sub["vals"]}
    return _map_gain(g, _do)


def _gain_flip(g: Optional[Dict[str, Any]], flip_dims: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
    def _do(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sh = sub["shape"]
        if _gain_is_zero(sub):
            return sub
        k = sub["kind"]
        if k == "dense":
            return {"kind": "dense", "shape": sh, "t": torch.flip(sub["t"], flip_dims).contiguous()}
        dim_set = set(flip_dims)
        if k == "boxes":
            ni = []
            for box, val in sub["items"]:
                nb = list(box)
                for d in dim_set:
                    bs, be, bst = nb[d]
                    dl = int(sh[d])
                    nb[d] = (dl - be, dl - bs, bst)
                ni.append((tuple(nb), val))
            return {"kind": "boxes", "shape": sh, "items": ni}
        nc = sub["coords"].clone()
        for d in dim_set:
            nc[:, d] = (int(sh[d]) - 1) - nc[:, d]
        return {"kind": "sparse", "shape": sh, "coords": nc, "vals": sub["vals"]}
    return _map_gain(g, _do)


def _gain_reshape(g: Optional[Dict[str, Any]], target_sh: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
    def _do(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        old_sh = sub["shape"]
        if old_sh == target_sh:
            return sub
        if math.prod(old_sh) != math.prod(target_sh):
            return None
        if _gain_is_zero(sub):
            return _gain_zero(target_sh)
        k = sub["kind"]
        if k == "dense":
            if math.prod(target_sh) <= _MAX_GAIN_DENSE:
                return {"kind": "dense", "shape": target_sh, "t": sub["t"].reshape(target_sh)}
            return None
        if k == "boxes":
            if len(sub["items"]) == 1 and all(
                b == (0, int(dl), 1) for b, dl in zip(sub["items"][0][0], old_sh)
            ):
                return _gain_full(target_sh, sub["items"][0][1])
            old_non1 = [(i, int(d)) for i, d in enumerate(old_sh) if int(d) != 1]
            tgt_non1 = [(j, int(d)) for j, d in enumerate(target_sh) if int(d) != 1]
            if [d for _, d in old_non1] == [d for _, d in tgt_non1]:
                ni = []
                for box, val in sub["items"]:
                    nb = [(0, 1, 1)] * len(target_sh)
                    for (oi, _), (tj, _) in zip(old_non1, tgt_non1):
                        nb[tj] = box[oi]
                    ni.append((tuple(nb), val))
                return {"kind": "boxes", "shape": target_sh, "items": ni}
            if math.prod(target_sh) <= _MAX_GAIN_DENSE:
                return {"kind": "dense", "shape": target_sh, "t": _gain_to_dense(sub)["t"].reshape(target_sh)}
            if len(old_sh) == 1 and len(target_sh) >= 1:
                stride = math.prod(target_sh[1:]) if len(target_sh) > 1 else 1
                ni = []
                for (bs, be, bst), val in sub["items"]:
                    if bst == 1 and bs % stride == 0 and be % stride == 0:
                        nb = ((bs // stride, be // stride, 1),) + tuple((0, int(d), 1) for d in target_sh[1:])
                        ni.append((nb, val))
                    else:
                        return None
                return {"kind": "boxes", "shape": target_sh, "items": ni}
            return None
        if math.prod(old_sh) > (1 << 62):
            return None
        c = sub["coords"]
        flat = c[:, -1].clone() if old_sh else torch.zeros((c.shape[0],), dtype=torch.int64)
        stride = max(1, int(old_sh[-1])) if old_sh else 1
        for j in range(len(old_sh) - 2, -1, -1):
            flat = flat + c[:, j] * stride
            stride *= max(1, int(old_sh[j]))
        rem = flat
        cols_new: List[torch.Tensor] = [torch.empty(0, dtype=torch.int64)] * len(target_sh)
        for j in range(len(target_sh) - 1, -1, -1):
            dj = int(target_sh[j])
            cols_new[j] = rem % dj
            rem = rem // dj
        nc = torch.stack(cols_new, dim=1) if target_sh else torch.empty((c.shape[0], 0), dtype=torch.int64)
        return {"kind": "sparse", "shape": target_sh, "coords": nc, "vals": sub["vals"]}
    return _map_gain(g, _do)


def _gain_sum_to_shape(g: Optional[Dict[str, Any]], target_sh: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
    def _do(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        old_sh = sub["shape"]
        if old_sh == target_sh:
            return sub
        lead = len(old_sh) - len(target_sh)
        if lead < 0:
            return None
        red_trail = []
        for idx_k, t_d in enumerate(target_sh):
            o_d = int(old_sh[lead + idx_k])
            if t_d != o_d:
                if t_d == 1:
                    red_trail.append(idx_k)
                else:
                    return None
        if _gain_is_zero(sub):
            return _gain_zero(target_sh)
        k = sub["kind"]
        if k == "dense":
            t = sub["t"]
            if lead > 0:
                t = t.sum(dim=tuple(range(lead)))
            if red_trail:
                t = t.sum(dim=tuple(red_trail), keepdim=True)
            t = t.reshape(target_sh)
            if math.prod(target_sh) <= _MAX_GAIN_DENSE:
                return {"kind": "dense", "shape": target_sh, "t": t}
            return None
        if k == "boxes":
            ni = []
            for box, val in sub["items"]:
                mult = 1
                for ld in range(lead):
                    bs, be, bst = box[ld]
                    mult *= 1 + (be - 1 - bs) // bst
                nb = list(box[lead:])
                for rd in red_trail:
                    bs, be, bst = nb[rd]
                    mult *= 1 + (be - 1 - bs) // bst
                    nb[rd] = (0, 1, 1)
                ni.append((tuple(nb), float(val) * float(mult)))
            merged = _coalesce_boxes(target_sh, ni)
            if len(merged) <= _MAX_GAIN_BOXES:
                return {"kind": "boxes", "shape": target_sh, "items": merged}
            return _gain_add({"kind": "boxes", "shape": target_sh, "items": merged}, _gain_zero(target_sh))
        nc = sub["coords"][:, lead:].clone()
        for rd in red_trail:
            nc[:, rd] = 0
        uc, uv = _coalesce_sparse(target_sh, nc, sub["vals"])
        return {"kind": "sparse", "shape": target_sh, "coords": uc, "vals": uv}
    return _map_gain(g, _do)


def _gain_repeat_reduce(
    g: Optional[Dict[str, Any]], target_sh: Tuple[int, ...], reps: Tuple[int, ...]
) -> Optional[Dict[str, Any]]:
    def _do(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        old_sh = sub["shape"]
        lead = len(reps) - len(target_sh)
        if lead < 0 or len(old_sh) != len(reps):
            return None
        if _gain_is_zero(sub):
            return _gain_zero(target_sh)
        if all(int(r) == 1 for r in reps):
            return _gain_reshape(sub, target_sh)
        k = sub["kind"]
        if k == "dense" or (k == "boxes" and math.prod(old_sh) <= _MAX_GAIN_DENSE):
            t = _gain_to_dense(sub)["t"]
            p_sh = (1,) * lead + target_sh
            inter_sh = tuple(d for pair in zip(reps, p_sh) for d in pair)
            sum_dims = tuple(range(0, len(inter_sh), 2))
            res = t.reshape(inter_sh).sum(dim=sum_dims).reshape(target_sh)
            if math.prod(target_sh) <= _MAX_GAIN_DENSE:
                return {"kind": "dense", "shape": target_sh, "t": res}
            return _dense_to_sparse({"kind": "dense", "shape": target_sh, "t": res})
        if k == "sparse":
            nc = sub["coords"][:, lead:].clone()
            for j, td in enumerate(target_sh):
                if int(td) > 0:
                    nc[:, j] = nc[:, j] % int(td)
            uc, uv = _coalesce_sparse(target_sh, nc, sub["vals"])
            return {"kind": "sparse", "shape": target_sh, "coords": uc, "vals": uv}
        p_sh = (1,) * lead + target_sh
        items_out = list(sub["items"])
        for ax in range(lead, len(reps)):
            r_ax = int(reps[ax])
            d_ax = int(p_sh[ax])
            if r_ax <= 1 or d_ax <= 0:
                continue
            next_items = []
            for box, val in items_out:
                bs, be, bst = box[ax]
                if d_ax == 1:
                    cnt = 1 + (be - 1 - bs) // bst
                    next_items.append((box[:ax] + ((0, 1, 1),) + box[ax + 1 :], val * float(cnt)))
                elif bst == 1 and (be - bs) <= d_ax:
                    s_m = bs % d_ax
                    span = be - bs
                    if s_m + span <= d_ax:
                        next_items.append((box[:ax] + ((s_m, s_m + span, 1),) + box[ax + 1 :], val))
                    else:
                        next_items.append((box[:ax] + ((s_m, d_ax, 1),) + box[ax + 1 :], val))
                        next_items.append((box[:ax] + ((0, s_m + span - d_ax, 1),) + box[ax + 1 :], val))
                else:
                    copies = (be - bs + d_ax - 1) // d_ax
                    next_items.append((box[:ax] + ((0, d_ax, 1),) + box[ax + 1 :], val * float(copies)))
                if len(next_items) > _MAX_GAIN_BOXES:
                    return None
            items_out = next_items
        # Drop leading repeated dimensions by multiplying element count
        final_items = []
        for box, val in items_out:
            mult = 1
            for ld in range(lead):
                bs, be, bst = box[ld]
                mult *= 1 + (be - 1 - bs) // bst
            final_items.append((box[lead:], float(val) * float(mult)))
        return {"kind": "boxes", "shape": target_sh, "items": _coalesce_boxes(target_sh, final_items)}
    return _map_gain(g, _do)


def _gain_expand_reduced(
    g: Optional[Dict[str, Any]],
    red_dims: Tuple[int, ...],
    keepdim: bool,
    target_sh: Tuple[int, ...],
) -> Optional[Dict[str, Any]]:
    def _do(sub: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if _gain_is_zero(sub):
            return _gain_zero(target_sh)
        k = sub["kind"]
        if k == "dense":
            if math.prod(target_sh) <= _MAX_GAIN_DENSE:
                t = sub["t"]
                if not keepdim:
                    unsq = [int(target_sh[d]) if d not in red_dims else 1 for d in range(len(target_sh))]
                    t = t.reshape(tuple(unsq))
                return {"kind": "dense", "shape": target_sh, "t": t.expand(target_sh).clone()}
            t = sub["t"]
            t_min = float(t.min().item())
            t_max = float(t.max().item())
            if t_min == t_max:
                return _gain_full(target_sh, t_max)
            return None
        if k == "boxes":
            red_set = set(red_dims)
            ni = []
            for box, val in sub["items"]:
                if keepdim:
                    nb = tuple((0, int(target_sh[d]), 1) if d in red_set else box[d] for d in range(len(target_sh)))
                else:
                    it = iter(box)
                    nb = tuple((0, int(target_sh[d]), 1) if d in red_set else next(it) for d in range(len(target_sh)))
                ni.append((nb, val))
            return {"kind": "boxes", "shape": target_sh, "items": ni}
        return None
    return _map_gain(g, _do)


def _scatter_index_gain(
    cur: object,
    cur_name: str,
    cur_gain: Optional[Dict[str, Any]],
    sh_cur: Optional[Tuple[int, ...]],
    self_sh: Optional[Tuple[int, ...]],
) -> Optional[Dict[str, Any]]:
    """Propagates coordinate gain backwards through IndexSelect/Gather/Index into `self_sh`."""
    if cur_gain is None or self_sh is None:
        return None
    if _gain_is_zero(cur_gain):
        return _gain_zero(self_sh)
    if any(int(d) == 0 for d in self_sh) or (sh_cur is not None and any(int(d) == 0 for d in sh_cur)):
        return _gain_zero(self_sh)
    max_elems = 4_194_304
    max_bool_elems = 67_108_864
    sh_out = cur_gain["shape"]
    try:
        if cur_name.startswith("IndexSelectBackward"):
            d_v = _normalize_signed_int(_safe_saved_attr(cur, "_saved_dim"))
            idx_t = _safe_saved_attr(cur, "_saved_index")
            if d_v is None or not isinstance(idx_t, torch.Tensor) or getattr(idx_t, "is_meta", False) or not self_sh:
                return None
            if idx_t.numel() > max_elems:
                return None
            dim = d_v + len(self_sh) if d_v < 0 else d_v
            if dim < 0 or dim >= len(self_sh) or idx_t.numel() != sh_out[dim]:
                return None
            dim_len = int(self_sh[dim])
            idx_cpu = idx_t.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
            idx_cpu = torch.where(idx_cpu < 0, idx_cpu + dim_len, idx_cpu)
            if bool((idx_cpu < 0).any().item()) or bool((idx_cpu >= dim_len).any().item()):
                return None
            if math.prod(sh_out) <= _MAX_GAIN_DENSE:
                g_out = _gain_to_dense(cur_gain)["t"]
                if math.prod(self_sh) <= _MAX_GAIN_DENSE:
                    acc = torch.zeros(self_sh, dtype=torch.float64)
                    acc.index_add_(dim, idx_cpu, g_out)
                    return {"kind": "dense", "shape": self_sh, "t": acc}
                nz = torch.nonzero(g_out > 0.0, as_tuple=True)
                if not nz or nz[0].numel() == 0:
                    return _gain_zero(self_sh)
                vals = g_out[nz].to(dtype=torch.float64)
                cols_l = list(nz)
                cols_l[dim] = idx_cpu[cols_l[dim]]
                uc, uv = _coalesce_sparse(self_sh, torch.stack(cols_l, dim=1), vals)
                return {"kind": "sparse", "shape": self_sh, "coords": uc, "vals": uv}
            if int(idx_cpu.numel()) <= _MAX_GAIN_BOXES:
                acc_g: Optional[Dict[str, Any]] = _gain_zero(self_sh)
                for p_i in range(int(idx_cpu.numel())):
                    dst_i = int(idx_cpu[p_i].item())
                    sl_g = _gain_split_axis(cur_gain, dim, p_i, 1)
                    if sl_g is None or _gain_is_zero(sl_g):
                        continue
                    # Replace axis dim (0,1,1) -> (dst_i, dst_i+1, 1) directly via select+select_embed
                    sq = _gain_reshape(
                        sl_g, sl_g["shape"][:dim] + sl_g["shape"][dim + 1 :]
                    ) if False else None
                    # Shift box dim directly via split+box/sparse translation
                    def _reloc(sub: Dict[str, Any], d_dst: int = dst_i) -> Optional[Dict[str, Any]]:
                        if sub["kind"] == "boxes":
                            ni = [
                                (b[:dim] + ((d_dst, d_dst + 1, 1),) + b[dim + 1 :], v)
                                for b, v in sub["items"]
                            ]
                            return {"kind": "boxes", "shape": self_sh, "items": ni}
                        if sub["kind"] == "sparse":
                            nc = sub["coords"].clone()
                            nc[:, dim] = d_dst
                            return {"kind": "sparse", "shape": self_sh, "coords": nc, "vals": sub["vals"]}
                        return None
                    acc_g = _gain_add(acc_g, _map_gain(sl_g, _reloc))
                    if acc_g is None:
                        return None
                return acc_g
            return None

        if cur_name.startswith("GatherBackward"):
            d_v = _normalize_signed_int(_safe_saved_attr(cur, "_saved_dim"))
            idx_t = _safe_saved_attr(cur, "_saved_index")
            if d_v is None or not isinstance(idx_t, torch.Tensor) or getattr(idx_t, "is_meta", False) or not self_sh:
                return None
            if idx_t.numel() > max_elems or tuple(idx_t.shape) != sh_out or idx_t.ndim != len(self_sh):
                return None
            dim = d_v + len(self_sh) if d_v < 0 else d_v
            if dim < 0 or dim >= len(self_sh):
                return None
            dim_len = int(self_sh[dim])
            idx_cpu = idx_t.detach().to(device="cpu", dtype=torch.int64)
            idx_cpu = torch.where(idx_cpu < 0, idx_cpu + dim_len, idx_cpu)
            if bool((idx_cpu < 0).any().item()) or bool((idx_cpu >= dim_len).any().item()):
                return None
            if math.prod(sh_out) <= _MAX_GAIN_DENSE:
                g_out = _gain_to_dense(cur_gain)["t"]
                if math.prod(self_sh) <= _MAX_GAIN_DENSE:
                    acc = torch.zeros(self_sh, dtype=torch.float64)
                    acc.scatter_add_(dim, idx_cpu, g_out)
                    return {"kind": "dense", "shape": self_sh, "t": acc}
                nz = torch.nonzero(g_out > 0.0, as_tuple=True)
                if not nz or nz[0].numel() == 0:
                    return _gain_zero(self_sh)
                vals = g_out[nz].to(dtype=torch.float64)
                cols_l = list(nz)
                cols_l[dim] = idx_cpu[nz]
                uc, uv = _coalesce_sparse(self_sh, torch.stack(cols_l, dim=1), vals)
                return {"kind": "sparse", "shape": self_sh, "coords": uc, "vals": uv}
            return None

        if cur_name.startswith("IndexBackward"):
            parsed = _extract_index_backward_specs(cur, self_sh, max_elems=max_elems, max_bool_elems=max_bool_elems)
            if parsed is None:
                return None
            _, indexed_dims, cols, b_sh = parsed
            if not cols:
                return cur_gain if sh_out == self_sh else None
            idx_set = set(indexed_dims)
            non_idx_dims = [d for d in range(len(self_sh)) if d not in idx_set]
            non_idx_sh = tuple(self_sh[d] for d in non_idx_dims)
            is_contig = (
                len(indexed_dims) <= 1
                or indexed_dims == list(range(indexed_dims[0], indexed_dims[0] + len(indexed_dims)))
            )
            b_pos = sum(1 for d in range(indexed_dims[0]) if d not in idx_set) if is_contig else 0
            expected_out_sh = non_idx_sh[:b_pos] + b_sh + non_idx_sh[b_pos:]
            if sh_out != expected_out_sh:
                return None
            b_numel = math.prod(b_sh) if b_sh else 1
            if math.prod(sh_out) <= _MAX_GAIN_DENSE:
                g_out = _gain_to_dense(cur_gain)["t"]
                n_b = len(b_sh)
                if n_b > 0 and b_pos > 0:
                    perm = (
                        list(range(b_pos, b_pos + n_b))
                        + list(range(0, b_pos))
                        + list(range(b_pos + n_b, g_out.ndim))
                    )
                    g_moved = g_out.permute(perm)
                else:
                    g_moved = g_out
                g_flat = g_moved.reshape((b_numel,) + non_idx_sh)
                nz = torch.nonzero(g_flat > 0.0, as_tuple=True)
                if not nz or nz[0].numel() == 0:
                    return _gain_zero(self_sh)
                bi = nz[0]
                vals = g_flat[nz].to(dtype=torch.float64)
                coords = torch.empty((vals.numel(), len(self_sh)), dtype=torch.int64)
                for m, d_idx in enumerate(indexed_dims):
                    coords[:, d_idx] = cols[m][bi]
                for r, d_non in enumerate(non_idx_dims):
                    coords[:, d_non] = nz[1 + r]
                if math.prod(self_sh) <= _MAX_GAIN_DENSE:
                    acc = torch.zeros(self_sh, dtype=torch.float64)
                    acc.index_put_(tuple(coords[:, d] for d in range(len(self_sh))), vals, accumulate=True)
                    return {"kind": "dense", "shape": self_sh, "t": acc}
                uc, uv = _coalesce_sparse(self_sh, coords, vals)
                return {"kind": "sparse", "shape": self_sh, "coords": uc, "vals": uv}
            if b_numel <= _MAX_GAIN_BOXES and cur_gain["kind"] == "boxes":
                n_b = len(b_sh)
                acc_b: Optional[Dict[str, Any]] = _gain_zero(self_sh)
                for bi_i in range(b_numel):
                    rem = bi_i
                    b_coords = [0] * n_b
                    for k_b in range(n_b - 1, -1, -1):
                        dk = int(b_sh[k_b])
                        b_coords[k_b] = rem % dk
                        rem //= dk
                    ni_boxes = []
                    for box, val in cur_gain["items"]:
                        ok = True
                        for k_b, bc in enumerate(b_coords):
                            bs, be, bst = box[b_pos + k_b]
                            if not (bs <= bc < be and ((bc - bs) % bst) == 0):
                                ok = False
                                break
                        if not ok:
                            continue
                        new_b: List[Optional[Tuple[int, int, int]]] = [None] * len(self_sh)
                        for m, d_idx in enumerate(indexed_dims):
                            dv = int(cols[m][bi_i].item())
                            new_b[d_idx] = (dv, dv + 1, 1)
                        non_slice = list(box[:b_pos]) + list(box[b_pos + n_b :])
                        for d_non, seg in zip(non_idx_dims, non_slice):
                            new_b[d_non] = seg
                        ni_boxes.append((tuple(seg for seg in new_b if seg is not None), val))
                    if ni_boxes:
                        acc_b = _gain_add(acc_b, {"kind": "boxes", "shape": self_sh, "items": ni_boxes})
                        if acc_b is None:
                            return None
                return acc_b
            return None
    except Exception:
        return None
    return None


def _trace_fp16_tensor_roots_with_meta(
    t: torch.Tensor,
) -> Tuple[List[Tuple[int, float, int, float, int, bool]], float]:
    """
    Traces float16 backward graph roots from tensor `t` without sharing global overflow keys
    across unrelated graphs. Returns:
      (list of (rid, w_abs, sign_parity, signed_scale, root_numel, same_shape), max_intermediate_gain).
    """
    _drain_teardown_anchors()
    zs_fan = float(_zero_stride_fanin(t))
    t_shape = tuple(int(d) for d in t.shape)
    t_stride = tuple(t.stride()) if t.layout == torch.strided else ()
    t_numel = int(t.numel())
    gf = t.grad_fn
    if gf is None:
        return [(id(t), zs_fan, 1, 1.0, t_numel, zs_fan == 1.0)], 1.0
    gf_onr = int(getattr(t, "output_nr", 0))
    cached_t = getattr(t, "_df_root_trace_cache", None)
    if (
        isinstance(cached_t, tuple)
        and len(cached_t) == 6
        and cached_t[0] is gf
        and cached_t[1] == t_shape
        and cached_t[2] == t_stride
        and cached_t[3] == t.dtype
    ):
        return list(cached_t[4]), float(cached_t[5])

    max_nodes = 65536
    hop_budget = [4_194_304]
    unary_cache: Dict[Tuple[int, int], Tuple[object, int, float, float, bool, bool, bool]] = {}
    nodes: Dict[int, object] = {id(gf): gf}
    indeg: Dict[int, int] = {id(gf): 0}
    bfs_mile: Dict[int, Tuple[int, object]] = {id(gf): (0, gf)}
    queue: List[object] = [gf]
    overflowed = False
    idx = 0
    while idx < len(queue):
        cur = queue[idx]
        idx += 1
        d_cur, m_cur = bfs_mile.get(id(cur), (0, cur))
        cur_name = type(cur).__name__
        if (
            cur_name == "AccumulateGrad"
            or _is_fp16_stop_fn(cur_name)
            or _is_custom_py_bwd_node(cur)
        ):
            continue
        for nxt_raw, raw_onr in cur.next_functions:
            if nxt_raw is None:
                continue
            m_raw, d_raw = _step_teardown_milestone(m_cur, d_cur, nxt_raw)
            nxt, _, _, _, _, _, _ = _advance_unary_chain_with_perm(
                nxt_raw, unary_cache, hop_budget, int(raw_onr)
            )
            if nxt is not nxt_raw:
                m_nxt, d_nxt = _step_teardown_milestone(
                    m_raw,
                    min(_TEARDOWN_ANCHOR_STEP - 1, d_raw + (_TEARDOWN_ANCHOR_STEP // 2)),
                    nxt,
                )
            else:
                m_nxt, d_nxt = m_raw, d_raw
            nid = id(nxt)
            indeg[nid] = indeg.get(nid, 0) + 1
            if nid not in nodes:
                if len(nodes) >= max_nodes or hop_budget[0] <= 0:
                    bfs_mile.setdefault(nid, (d_nxt, m_nxt))
                    overflowed = True
                    break
                nodes[nid] = nxt
                bfs_mile[nid] = (d_nxt, m_nxt)
                queue.append(nxt)
        if overflowed:
            break

    if overflowed:
        # Sound disjoint fallback: preserve every root already discovered AND walk every frontier branch.
        fallback_w = max(zs_fan, float(max(1, t_numel)))
        spine_roots: Dict[int, Tuple[float, int]] = {}
        stack: List[Tuple[object, int, object]] = (
            [(q_nd, *bfs_mile.get(id(q_nd), (0, q_nd))) for q_nd in queue[max(0, idx - 1) :]]
            if idx > 0
            else [(gf, 0, gf)]
        )

        def _maybe_record_node(nd: object) -> bool:
            nd_name = type(nd).__name__
            nd_id = id(nd)
            if nd_name == "AccumulateGrad":
                var = getattr(nd, "variable", None)
                if isinstance(var, torch.Tensor) and var.dtype == torch.float16:
                    spine_roots[id(var)] = (fallback_w, int(var.numel()))
                return True
            if _is_fp16_stop_fn(nd_name):
                out_n = getattr(nd, "_out_numel", None)
                r_n = int(out_n) if isinstance(out_n, int) and out_n > 0 else t_numel
                spine_roots[nd_id] = (fallback_w, r_n)
                return True
            if _is_custom_py_bwd_node(nd):
                in_m = getattr(nd, "_input_metadata", None)
                b_shape = getattr(in_m[0], "shape", None) if in_m else None
                r_n = int(math.prod(b_shape)) if b_shape is not None else t_numel
                spine_roots[-(nd_id * 1024 + 16)] = (fallback_w, r_n)
                return True
            if _is_fp16_cast_leaf_boundary(nd, nd_name):
                in_m = getattr(nd, "_input_metadata", None)
                b_shape = getattr(in_m[0], "shape", None) if in_m else None
                r_n = int(math.prod(b_shape)) if b_shape is not None else t_numel
                spine_roots[nd_id] = (fallback_w, r_n)
                return False
            return False

        for nd in nodes.values():
            if not _maybe_record_node(nd):
                d_nd, m_nd = bfs_mile.get(id(nd), (0, nd))
                for nf, onr in nd.next_functions:
                    if nf is not None and id(nf) not in nodes:
                        m_nf, d_nf = _step_teardown_milestone(m_nd, d_nd, nf)
                        stack.append((nf, d_nf, m_nf))

        visited_sp: set[int] = set()
        sp_hops = 2_000_000
        while stack and sp_hops > 0:
            sp_hops -= 1
            cur_s, d_s, m_s = stack.pop()
            csid = id(cur_s)
            if csid in visited_sp:
                continue
            visited_sp.add(csid)
            if _maybe_record_node(cur_s):
                continue
            nxt_list = [(nf, onr) for nf, onr in cur_s.next_functions if nf is not None]
            if not nxt_list:
                spine_roots[csid] = (fallback_w, t_numel)
                continue
            if len(nxt_list) == 1:
                nf0, onr0 = nxt_list[0]
                m_nf, d_nf = _step_teardown_milestone(m_s, d_s, nf0, single_child=True)
                nxt0, _, _, _, _, _, _ = _advance_unary_chain_with_perm(
                    nf0, unary_cache, hop_budget, int(onr0)
                )
                if nxt0 is not nf0:
                    m_nf, d_nf = _step_teardown_milestone(
                        m_nf,
                        min(_TEARDOWN_ANCHOR_STEP - 1, d_nf + (_TEARDOWN_ANCHOR_STEP // 2)),
                        nxt0,
                        single_child=True,
                    )
                if id(nxt0) not in visited_sp:
                    stack.append((nxt0, d_nf, m_nf))
            else:
                for nf, onr in nxt_list:
                    m_nf, d_nf = _step_teardown_milestone(m_s, d_s, nf)
                    nxt_c, _, _, _, _, _, _ = _advance_unary_chain_with_perm(
                        nf, unary_cache, hop_budget, int(onr)
                    )
                    if nxt_c is not nf:
                        m_nf, d_nf = _step_teardown_milestone(
                            m_nf,
                            min(_TEARDOWN_ANCHOR_STEP - 1, d_nf + (_TEARDOWN_ANCHOR_STEP // 2)),
                            nxt_c,
                        )
                    if id(nxt_c) not in visited_sp:
                        stack.append((nxt_c, d_nf, m_nf))

        for rem, _, _ in stack:
            if id(rem) not in visited_sp and not _maybe_record_node(rem):
                spine_roots[id(rem)] = (fallback_w, t_numel)

        if not spine_roots:
            spine_roots[id(gf)] = (fallback_w, t_numel)
        return [(rid, w_v, 0, 1.0, r_n, False) for rid, (w_v, r_n) in spine_roots.items()], 1.0

    weights: Dict[int, float] = {id(gf): 1.0}
    signed_weights: Dict[int, float] = {id(gf): 1.0}
    sign_ok: Dict[int, bool] = {id(gf): True}
    shape_ok: Dict[int, bool] = {id(gf): (zs_fan == 1.0)}
    expected_numel: Dict[int, Optional[int]] = {id(gf): t_numel}
    expected_shape: Dict[int, Optional[Tuple[int, ...]]] = {id(gf): t_shape}
    slot_gains: Dict[Tuple[int, int], Optional[Dict[str, Any]]] = {
        (id(gf), gf_onr): _gain_full(t_shape, 1.0)
    }
    reached_slots: set[Tuple[int, int]] = {(id(gf), gf_onr)}
    topo: List[object] = [gf] if indeg.get(id(gf), 0) == 0 else []
    t_idx = 0
    results_map: Dict[int, Tuple[float, float, bool, int, bool]] = {}
    max_inter_gain = 1.0

    def _record_root(rid: int, w_val: float, sw_val: float, sk: bool, r_n: int, same_f: bool) -> None:
        prev = results_map.get(rid)
        if prev is None:
            results_map[rid] = (w_val, sw_val, sk, r_n, same_f)
        else:
            results_map[rid] = (
                prev[0] + w_val,
                prev[1] + sw_val,
                prev[2] and sk,
                r_n,
                prev[4] and same_f,
            )

    def _assemble_split_gain(
        cur: object, cur_name: str, cid: int, child_sh: Tuple[int, ...]
    ) -> Optional[Dict[str, Any]]:
        dim_v = _normalize_signed_int(_safe_saved_attr(cur, "_saved_dim"))
        if dim_v is None or not child_sh:
            return None
        d_i = dim_v + len(child_sh) if dim_v < 0 else dim_v
        if d_i < 0 or d_i >= len(child_sh):
            return None
        dim_len = int(child_sh[d_i])
        chunk_lens: List[int] = []
        is_unbind = cur_name.startswith("UnbindBackward")
        if cur_name.startswith("SplitWithSizesBackward"):
            sp_sz = _safe_saved_attr(cur, "_saved_split_sizes")
            if sp_sz is None:
                return None
            chunk_lens = [max(0, int(s)) for s in sp_sz]
        elif cur_name.startswith("SplitBackward"):
            sp_s = _safe_saved_attr(cur, "_saved_split_size")
            if sp_s is None or int(sp_s) <= 0:
                return None
            step_s = int(sp_s)
            rem = dim_len
            while rem > 0:
                take = step_s if rem >= step_s else rem
                chunk_lens.append(take)
                rem -= take
            if not chunk_lens:
                chunk_lens = [0]
        else:
            chunk_lens = [1] * dim_len
        if sum(chunk_lens) != dim_len:
            return None
        parts: List[Dict[str, Any]] = []
        for si, clen in enumerate(chunk_lens):
            c_sh = (
                child_sh[:d_i] + child_sh[d_i + 1 :]
                if is_unbind
                else (child_sh[:d_i] + (clen,) + child_sh[d_i + 1 :])
            )
            tgt_c_sh = child_sh[:d_i] + (1,) + child_sh[d_i + 1 :] if is_unbind else c_sh
            if (cid, si) in reached_slots:
                sg = slot_gains.get((cid, si))
                if sg is None or sg["shape"] != c_sh:
                    return None
                if is_unbind:
                    sg = _gain_reshape(sg, tgt_c_sh)
                    if sg is None:
                        return None
                parts.append(sg)
            else:
                parts.append(_gain_zero(tgt_c_sh))
        return _gain_cat_chunks(parts, d_i, child_sh)

    while t_idx < len(topo):
        cur = topo[t_idx]
        t_idx += 1
        cid = id(cur)
        cur_name = type(cur).__name__
        w_cur = weights.get(cid, 1.0)
        sw_cur = signed_weights.get(cid, 1.0)
        sk_cur = sign_ok.get(cid, True)
        shk_cur = shape_ok.get(cid, False)
        n_cur = expected_numel.get(cid)
        sh_cur = expected_shape.get(cid)

        cur_gain: Optional[Dict[str, Any]] = None
        is_split_op = cur_name.startswith(("SplitWithSizesBackward", "SplitBackward", "UnbindBackward"))
        if is_split_op:
            active_split = [(ei, nf, onr) for ei, (nf, onr) in enumerate(cur.next_functions) if nf is not None]
            if len(active_split) == 1:
                _, nf_sp, onr_sp = active_split[0]
                ch_sp_sh, _ = _node_shape_dtype(nf_sp, int(onr_sp))
                if ch_sp_sh is None:
                    ss_sp = _safe_saved_attr(cur, "_saved_self_sym_sizes")
                    if ss_sp is not None:
                        ch_sp_sh = tuple(max(0, int(d)) for d in ss_sp)
                if ch_sp_sh is not None:
                    cur_gain = _assemble_split_gain(cur, cur_name, cid, ch_sp_sh)
        else:
            r_onrs = [onr for s_cid, onr in reached_slots if s_cid == cid]
            if len(r_onrs) == 1:
                cur_gain = slot_gains.get((cid, r_onrs[0]))
                if cur_gain is not None:
                    sh_cur = cur_gain["shape"]

        if cur_gain is not None:
            w_cur = max(0.0, _gain_max(cur_gain))

        _, cur_dt = _node_shape_dtype(cur, 0)
        if cur_dt == torch.float16:
            max_inter_gain = max(max_inter_gain, w_cur)

        if cur_name == "AccumulateGrad":
            var = getattr(cur, "variable", None)
            if isinstance(var, torch.Tensor) and var.dtype == torch.float16:
                v_numel = max(1, int(var.numel()))
                var_ratio = float(max(1, (t_numel + v_numel - 1) // v_numel)) if t_numel > v_numel else 1.0
                same_v = bool(shk_cur and tuple(var.shape) == t_shape and zs_fan == 1.0)
                root_w = max(w_cur, zs_fan) if cur_gain is not None else max(w_cur, zs_fan, var_ratio)
                _record_root(id(var), root_w, sw_cur, sk_cur, int(var.numel()), same_v)
            continue
        if _is_fp16_stop_fn(cur_name):
            out_n = getattr(cur, "_out_numel", None)
            out_ratio = (
                float(max(1, (t_numel + out_n - 1) // out_n))
                if (isinstance(out_n, int) and out_n > 0 and t_numel > out_n)
                else 1.0
            )
            r_n = int(out_n) if isinstance(out_n, int) and out_n > 0 else t_numel
            stop_w = max(w_cur, zs_fan) if cur_gain is not None else max(w_cur, zs_fan, out_ratio)
            _record_root(cid, stop_w, sw_cur, sk_cur, r_n, False)
            continue
        if _is_custom_py_bwd_node(cur):
            in_m = getattr(cur, "_input_metadata", None)
            b_shape = getattr(in_m[0], "shape", None) if in_m else None
            r_n = int(math.prod(b_shape)) if b_shape is not None else t_numel
            _record_root(-(cid * 1024 + 16), max(w_cur, zs_fan), sw_cur, False, r_n, False)
            continue
        if _is_fp16_cast_leaf_boundary(cur, cur_name):
            in_m = getattr(cur, "_input_metadata", None)
            b_shape = getattr(in_m[0], "shape", None) if in_m else None
            b_shape_tup = tuple(int(d) for d in b_shape) if b_shape is not None else None
            r_n = int(math.prod(b_shape_tup)) if b_shape_tup is not None else t_numel
            same_b = bool(shk_cur and b_shape_tup is not None and b_shape_tup == t_shape and zs_fan == 1.0)
            _record_root(cid, max(w_cur, zs_fan), sw_cur, sk_cur, r_n, same_b)

        in_metas = getattr(cur, "_input_metadata", None)
        next_mult = 1.0
        next_n: Optional[int] = n_cur
        if cur_name.startswith("ExpandBackward"):
            ss = _safe_saved_attr(cur, "_saved_self_sym_sizes")
            if ss is not None:
                base_n = 1
                for d in ss:
                    base_n *= max(1, int(d))
                ref_n = n_cur if n_cur is not None else t_numel
                next_mult = float(max(1, (ref_n + base_n - 1) // base_n))
                next_n = base_n
        elif cur_name.startswith("RepeatBackward"):
            reps = _safe_saved_attr(cur, "_saved_repeats")
            if reps is not None:
                r_prod = 1
                for r in reps:
                    r_prod *= max(1, int(r))
                next_mult = float(r_prod)
                if n_cur is not None:
                    next_n = max(1, n_cur // max(1, r_prod))
        elif cur_name.startswith(("IndexBackward", "IndexSelectBackward", "GatherBackward")):
            ref_n = n_cur if n_cur is not None else t_numel
            fallback_mult = float(max(1, ref_n))
            ss = _safe_saved_attr(cur, "_saved_self_sym_sizes")
            self_sh: Optional[Tuple[int, ...]] = None
            if ss is not None:
                self_sh = tuple(max(0, int(d)) for d in ss)
                base_n = 1
                for d in self_sh:
                    base_n *= max(1, d)
                next_n = base_n
            else:
                saved_s = _safe_saved_attr(cur, "_saved_self")
                if isinstance(saved_s, torch.Tensor):
                    self_sh = tuple(int(d) for d in saved_s.shape)
                    next_n = max(1, int(saved_s.numel()))
                elif cur.next_functions:
                    nf0_i, onr0_i = cur.next_functions[0]
                    if nf0_i is not None:
                        self_sh, _ = _node_shape_dtype(nf0_i, int(onr0_i))
                    next_n = int(math.prod(self_sh)) if self_sh is not None else None
                else:
                    next_n = None
            next_mult = _exact_index_max_dup(cur, cur_name, sh_cur, self_sh, fallback_mult)
        elif cur_name.startswith(("CatBackward", "StackBackward", "SliceBackward", "SelectBackward")) or is_split_op:
            next_n = None

        cat_widths: Optional[List[Optional[int]]] = None
        cat_dim: Optional[int] = None
        if cur_name.startswith("CatBackward") and sh_cur is not None and len(sh_cur) > 0:
            cd_raw = _normalize_signed_int(_safe_saved_attr(cur, "_saved_dim"))
            if cd_raw is not None:
                cat_dim = cd_raw + len(sh_cur) if cd_raw < 0 else cd_raw
                if 0 <= cat_dim < len(sh_cur):
                    cw: List[Optional[int]] = []
                    none_i: List[int] = []
                    for ei_c, (nf_c, onr_c) in enumerate(cur.next_functions):
                        if nf_c is None:
                            cw.append(None)
                            none_i.append(ei_c)
                        else:
                            sh_c, _ = _node_shape_dtype(nf_c, int(onr_c))
                            if sh_c is not None and cat_dim < len(sh_c):
                                cw.append(int(sh_c[cat_dim]))
                            else:
                                cw.append(None)
                    if len(none_i) == 1 and all(w is not None or idx_c == none_i[0] for idx_c, w in enumerate(cw)):
                        known_w = sum(w for w in cw if w is not None)
                        rem_w = int(sh_cur[cat_dim]) - known_w
                        if rem_w >= 0:
                            cw[none_i[0]] = rem_w
                    cat_widths = cw

        active_edges = [(ei, nf, onr) for ei, (nf, onr) in enumerate(cur.next_functions) if nf is not None]
        for edge_idx, nxt_raw, raw_onr in active_edges:
            ch_sh, edge_dt = _node_shape_dtype(nxt_raw, int(raw_onr))
            e_meta = in_metas[edge_idx] if (in_metas is not None and edge_idx < len(in_metas)) else None
            e_shape_raw = getattr(e_meta, "shape", None) if e_meta is not None else None
            e_shape_meta = tuple(int(d) for d in e_shape_raw) if e_shape_raw is not None else None
            e_shape = ch_sh if ch_sh is not None else e_shape_meta

            e_sign = 0
            e_scale = 1.0
            e_same = False
            e_mult = next_mult
            if cur_name.startswith(_AFFINE_PRESERVE_OPS):
                e_sign = 1
                e_same = e_shape is None or (sh_cur is not None and e_shape == sh_cur)
            elif cur_name.startswith("NegBackward"):
                e_sign = -1
                e_same = e_shape is None or (sh_cur is not None and e_shape == sh_cur)
            elif cur_name.startswith(_AFFINE_RESHAPE_SIGN_OPS):
                e_sign = 1
                e_same = False
            elif cur_name.startswith("MeanBackward"):
                ss_m = _safe_saved_attr(cur, "_saved_self_sym_sizes")
                self_m_sh = tuple(max(0, int(d)) for d in ss_m) if ss_m is not None else ch_sh
                if self_m_sh is not None and sh_cur is not None:
                    self_n_m = math.prod(self_m_sh)
                    out_n_m = math.prod(sh_cur)
                    if self_n_m > 0 and out_n_m > 0 and self_n_m % out_n_m == 0:
                        e_sign = 1
                        e_scale = 1.0 / float(self_n_m // out_n_m)
                        e_mult = next_mult * e_scale
            elif cur_name.startswith("AddBackward"):
                raw_a = _safe_saved_attr(cur, "_saved_alpha") if edge_idx == 1 else 1.0
                alpha = float(raw_a) if raw_a is not None else 1.0
                if alpha != 0.0 and math.isfinite(alpha):
                    e_sign = 1 if alpha > 0.0 else -1
                    e_scale = abs(alpha)
                    e_same = (sh_cur is not None and e_shape is not None and e_shape == sh_cur)
                    e_mult = next_mult * e_scale
            elif cur_name.startswith("SubBackward"):
                raw_a = _safe_saved_attr(cur, "_saved_alpha") if edge_idx == 1 else 1.0
                alpha = float(raw_a) if raw_a is not None else 1.0
                if alpha != 0.0 and math.isfinite(alpha):
                    s_base = -1 if edge_idx == 1 else 1
                    e_sign = s_base if alpha > 0.0 else -s_base
                    e_scale = abs(alpha)
                    e_same = (sh_cur is not None and e_shape is not None and e_shape == sh_cur)
                    e_mult = next_mult * e_scale
            elif cur_name.startswith("RsubBackward"):
                raw_a = _safe_saved_attr(cur, "_saved_alpha") if edge_idx == 0 else 1.0
                alpha = float(raw_a) if raw_a is not None else 1.0
                if alpha != 0.0 and math.isfinite(alpha):
                    s_base = -1 if edge_idx == 0 else 1
                    e_sign = s_base if alpha > 0.0 else -s_base
                    e_scale = abs(alpha)
                    e_same = (sh_cur is not None and e_shape is not None and e_shape == sh_cur)
                    e_mult = next_mult * e_scale
            elif cur_name.startswith("MulBackward") and len(active_edges) == 1:
                c_val = _finite_scalar_const(
                    _safe_saved_attr(cur, "_saved_other" if edge_idx == 0 else "_saved_self")
                )
                if c_val is not None:
                    e_sign = 1 if c_val > 0.0 else -1
                    e_scale = abs(c_val)
                    e_same = (sh_cur is not None and e_shape is not None and e_shape == sh_cur)
                    e_mult = next_mult * e_scale
            elif cur_name.startswith("DivBackward") and len(active_edges) == 1 and edge_idx == 0:
                c_val = _finite_scalar_const(_safe_saved_attr(cur, "_saved_other"))
                if c_val is not None:
                    e_sign = 1 if c_val > 0.0 else -1
                    e_scale = 1.0 / abs(c_val)
                    e_same = (sh_cur is not None and e_shape is not None and e_shape == sh_cur)
                    e_mult = next_mult * e_scale

            # Coordinate gain edge transform
            edge_gain: Optional[Dict[str, Any]] = None
            if cur_gain is not None and ch_sh is not None:
                if is_split_op:
                    edge_gain = cur_gain if cur_gain["shape"] == ch_sh else None
                elif cur_name.startswith((_AFFINE_PRESERVE_OPS + ("NegBackward",))):
                    edge_gain = cur_gain if cur_gain["shape"] == ch_sh else None
                elif cur_name.startswith(("AddBackward", "SubBackward", "RsubBackward")):
                    if e_sign != 0:
                        edge_gain = _gain_sum_to_shape(_gain_scale(cur_gain, e_scale), ch_sh)
                elif cur_name.startswith(("MulBackward", "DivBackward")):
                    if e_sign != 0 and cur_gain["shape"] == ch_sh:
                        edge_gain = _gain_scale(cur_gain, e_scale)
                elif cur_name.startswith("CatBackward"):
                    if cat_widths is not None and cat_dim is not None and edge_idx < len(cat_widths):
                        ew = cat_widths[edge_idx]
                        if ew is not None:
                            pre = cat_widths[:edge_idx]
                            suf = cat_widths[edge_idx + 1 :]
                            off: Optional[int] = None
                            if all(w is not None for w in pre):
                                off = sum(int(w) for w in pre if w is not None)
                            elif all(w is not None for w in suf):
                                off = int(cur_gain["shape"][cat_dim]) - ew - sum(int(w) for w in suf if w is not None)
                            if off is not None and off >= 0:
                                edge_gain = _gain_split_axis(cur_gain, cat_dim, off, ew)
                elif cur_name.startswith("StackBackward"):
                    sd_raw = _normalize_signed_int(_safe_saved_attr(cur, "_saved_dim"))
                    if sd_raw is not None and len(cur_gain["shape"]) > 0:
                        sd_i = sd_raw + len(cur_gain["shape"]) if sd_raw < 0 else sd_raw
                        if 0 <= sd_i < len(cur_gain["shape"]) and edge_idx < cur_gain["shape"][sd_i]:
                            sp_g = _gain_split_axis(cur_gain, sd_i, edge_idx, 1)
                            if sp_g is not None:
                                edge_gain = _gain_reshape(sp_g, ch_sh)
                elif cur_name.startswith("SliceBackward"):
                    sl_d = _normalize_signed_int(_safe_saved_attr(cur, "_saved_dim"))
                    sl_s = _normalize_signed_int(_safe_saved_attr(cur, "_saved_start"))
                    sl_e = _normalize_signed_int(_safe_saved_attr(cur, "_saved_end"))
                    sl_st = _normalize_signed_int(_safe_saved_attr(cur, "_saved_step"))
                    if None not in (sl_d, sl_s, sl_e, sl_st) and ch_sh:
                        sl_di = sl_d + len(ch_sh) if sl_d < 0 else sl_d
                        if 0 <= sl_di < len(ch_sh) and sl_st > 0:
                            edge_gain = _gain_slice_embed(cur_gain, sl_di, sl_s, sl_e, sl_st, ch_sh)
                elif cur_name.startswith("SelectBackward"):
                    sel_d = _normalize_signed_int(_safe_saved_attr(cur, "_saved_dim"))
                    sel_i = _normalize_signed_int(_safe_saved_attr(cur, "_saved_index"))
                    if sel_d is not None and sel_i is not None and ch_sh:
                        sel_di = sel_d + len(ch_sh) if sel_d < 0 else sel_d
                        if 0 <= sel_di < len(ch_sh):
                            sel_ii = sel_i + int(ch_sh[sel_di]) if sel_i < 0 else sel_i
                            if 0 <= sel_ii < int(ch_sh[sel_di]):
                                edge_gain = _gain_select_embed(cur_gain, sel_di, sel_ii, ch_sh)
                elif cur_name.startswith(("IndexBackward", "IndexSelectBackward", "GatherBackward")):
                    edge_gain = _scatter_index_gain(cur, cur_name, cur_gain, sh_cur, ch_sh)
                elif cur_name.startswith(_UNARY_VIEW_BIJECTIVE_PASS):
                    edge_gain = _gain_reshape(cur_gain, ch_sh)
                elif cur_name.startswith("TransposeBackward"):
                    d0 = _normalize_signed_int(_safe_saved_attr(cur, "_saved_dim0"))
                    d1 = _normalize_signed_int(_safe_saved_attr(cur, "_saved_dim1"))
                    rk = len(cur_gain["shape"])
                    if d0 is not None and d1 is not None and rk > 0:
                        d0i = d0 + rk if d0 < 0 else d0
                        d1i = d1 + rk if d1 < 0 else d1
                        if 0 <= d0i < rk and 0 <= d1i < rk:
                            perm_t = list(range(rk))
                            perm_t[d0i], perm_t[d1i] = perm_t[d1i], perm_t[d0i]
                            edge_gain = _gain_permute(cur_gain, tuple(perm_t))
                elif cur_name.startswith("PermuteBackward"):
                    p_dims = _safe_saved_attr(cur, "_saved_dims")
                    rk = len(cur_gain["shape"])
                    if p_dims is not None and len(p_dims) == rk:
                        norm_p = []
                        ok_p = True
                        for dv in p_dims:
                            dvi = _normalize_signed_int(dv)
                            if dvi is None:
                                ok_p = False
                                break
                            if dvi < 0:
                                dvi += rk
                            norm_p.append(dvi)
                        if ok_p:
                            edge_gain = _gain_permute(cur_gain, tuple(norm_p))
                elif cur_name.startswith("TBackward"):
                    if len(cur_gain["shape"]) == 2:
                        edge_gain = _gain_permute(cur_gain, (1, 0))
                    else:
                        edge_gain = cur_gain if cur_gain["shape"] == ch_sh else None
                elif cur_name.startswith("FlipBackward"):
                    f_dims = _safe_saved_attr(cur, "_saved_dims")
                    rk = len(cur_gain["shape"])
                    if f_dims is not None and rk > 0:
                        norm_f = []
                        ok_f = True
                        for dv in f_dims:
                            dvi = _normalize_signed_int(dv)
                            if dvi is None:
                                ok_f = False
                                break
                            if dvi < 0:
                                dvi += rk
                            if not (0 <= dvi < rk):
                                ok_f = False
                                break
                            norm_f.append(dvi)
                        if ok_f:
                            edge_gain = _gain_flip(cur_gain, tuple(norm_f))
                elif cur_name.startswith("ExpandBackward"):
                    edge_gain = _gain_sum_to_shape(cur_gain, ch_sh)
                elif cur_name.startswith("RepeatBackward"):
                    reps = _safe_saved_attr(cur, "_saved_repeats")
                    if reps is not None:
                        edge_gain = _gain_repeat_reduce(cur_gain, ch_sh, tuple(int(r) for r in reps))
                elif cur_name.startswith(("SumBackward", "MeanBackward")):
                    if cur_name.startswith("SumBackward") or e_sign != 0:
                        g_in_s = _gain_scale(cur_gain, e_scale) if cur_name.startswith("MeanBackward") else cur_gain
                        if cur_name.startswith(("SumBackward0", "MeanBackward0")) or _safe_saved_attr(cur, "_saved_dim") is None:
                            red_dims = tuple(range(len(ch_sh)))
                            kd = False
                        else:
                            raw_sd = _safe_saved_attr(cur, "_saved_dim")
                            kd_raw = _safe_saved_attr(cur, "_saved_keepdim")
                            kd = bool(kd_raw) if kd_raw is not None else False
                            red_l = []
                            for dv in raw_sd:
                                dvi = _normalize_signed_int(dv)
                                if dvi is None:
                                    red_l = []
                                    break
                                if dvi < 0:
                                    dvi += len(ch_sh)
                                red_l.append(dvi)
                            red_dims = tuple(sorted(set(red_l)))
                        if len(red_dims) > 0:
                            edge_gain = _gain_expand_reduced(g_in_s, red_dims, kd, ch_sh)
                elif cur_name.startswith(_BOUNDED_NONAFFINE_UNARY_OPS):
                    if cur_gain["shape"] == ch_sh:
                        edge_gain = cur_gain

            nxt, nxt_onr, chain_scale, chain_peak, chain_same, chain_sign_ok, crossed_perm = (
                _advance_unary_chain_with_perm(nxt_raw, unary_cache, hop_budget, int(raw_onr))
            )
            nxt_gain: Optional[Dict[str, Any]] = None
            if edge_gain is not None and not crossed_perm:
                nxt_sh, _ = _node_shape_dtype(nxt, nxt_onr)
                scaled_eg = _gain_scale(edge_gain, abs(chain_scale))
                if scaled_eg is not None:
                    if nxt_sh is None or nxt_sh == scaled_eg["shape"]:
                        nxt_gain = scaled_eg
                    elif not chain_same:
                        nxt_gain = _gain_reshape(scaled_eg, nxt_sh)

            if edge_dt == torch.float16:
                edge_peak_w = _gain_max(edge_gain) if edge_gain is not None else (w_cur * e_mult)
                max_inter_gain = max(max_inter_gain, edge_peak_w * max(1.0, chain_peak))

            nid = id(nxt)
            if nid not in nodes:
                continue
            weights[nid] = weights.get(nid, 0.0) + w_cur * e_mult * abs(chain_scale)
            s_key = (nid, nxt_onr)
            if s_key not in reached_slots:
                reached_slots.add(s_key)
                slot_gains[s_key] = nxt_gain
            else:
                slot_gains[s_key] = _gain_add(slot_gains.get(s_key), nxt_gain)

            if sk_cur and e_sign != 0 and chain_sign_ok:
                signed_weights[nid] = signed_weights.get(nid, 0.0) + sw_cur * float(e_sign) * e_scale * chain_scale
                sign_ok[nid] = sign_ok.get(nid, True)
            else:
                sign_ok[nid] = False
            nxt_same = bool(shk_cur and e_same and chain_same)
            shape_ok[nid] = shape_ok.get(nid, True) and nxt_same
            nxt_shape = e_shape if nxt_same else (e_shape if nxt is nxt_raw else None)
            if nid not in expected_shape:
                expected_shape[nid] = nxt_shape
            elif expected_shape[nid] != nxt_shape:
                expected_shape[nid] = None
            nxt_n_val = int(math.prod(e_shape)) if e_shape is not None else next_n
            if nid not in expected_numel:
                expected_numel[nid] = nxt_n_val
            elif expected_numel[nid] != nxt_n_val:
                expected_numel[nid] = None
            indeg[nid] -= 1
            if indeg[nid] == 0:
                topo.append(nxt)

    out_specs: List[Tuple[int, float, int, float, int, bool]] = []
    for rid, (w_abs, sw_val, sk_val, r_n, same_f) in results_map.items():
        if sk_val and abs(sw_val) > 0.0 and math.isfinite(sw_val):
            s_par = 1 if sw_val > 0.0 else -1
            s_scale = abs(sw_val)
        else:
            s_par = 0
            s_scale = 1.0
        out_specs.append((rid, w_abs, s_par, s_scale, r_n, same_f))
    try:
        t._df_root_trace_cache = (gf, t_shape, t_stride, t.dtype, tuple(out_specs), float(max_inter_gain))  # type: ignore[attr-defined]
    except Exception:
        pass
    return out_specs, max_inter_gain


def _trace_fp16_tensor_roots(
    t: torch.Tensor,
) -> List[Tuple[int, float, int, float, int, bool]]:
    specs, _ = _trace_fp16_tensor_roots_with_meta(t)
    return specs


def _accumulate_fp16_tensor_fanin(t: torch.Tensor, root_hits: Dict[int, float]) -> None:
    specs, max_inter_gain = _trace_fp16_tensor_roots_with_meta(t)
    for rid, w_abs, _, _, _, _ in specs:
        root_hits[rid] = root_hits.get(rid, 0.0) + max(w_abs, max_inter_gain)


def _fp16_cap_for_inputs(obj) -> float:
    """Caps float16 backward casts accounting for views, expands, repeats, and multi-path shared roots."""
    root_hits: Dict[int, float] = {}

    def _visit(x):
        if isinstance(x, torch.Tensor):
            if x.dtype == torch.float16:
                _accumulate_fp16_tensor_fanin(x, root_hits)
        elif isinstance(x, IntuitionisticFuzzyTensor):
            _visit(x.mu)
            _visit(x.nu)
            _visit(x.pi)
        elif isinstance(x, dict):
            for v in x.values():
                _visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _visit(v)

    _visit(obj)
    max_fanin = max(root_hits.values()) if root_hits else 1.0
    eff_fanin = max(1.0, float(max_fanin))
    if eff_fanin <= 1.0:
        return _FP16_GRAD_MAX
    return min(_FP16_GRAD_MAX, _FP16_MULTI_BUDGET / eff_fanin)


def _promote_low_prec_cached(
    x: Optional[torch.Tensor],
    work_dtype: Optional[torch.dtype],
    cache: Dict[int, torch.Tensor],
    fp16_max: float = _FP16_GRAD_MAX,
) -> Optional[torch.Tensor]:
    if x is None or work_dtype is None:
        return x
    if x.dtype in _HALF_PREC_DTYPES and x.dtype != work_dtype:
        key = id(x)
        if key in cache:
            return cache[key]
        if _has_valid_work_snap(x):
            w_val: torch.Tensor = x._work_val  # type: ignore[attr-defined]
            w = w_val if w_val.dtype == work_dtype else w_val.to(dtype=work_dtype)
        else:
            w = _SafeCastToWorkDtype.apply(x, work_dtype, float(fp16_max), torch.is_grad_enabled())
        cache[key] = w
        return w
    return x


def _sanitize_half_prec_ifn_for_work(
    mu: torch.Tensor,
    nu: torch.Tensor,
    pi: Optional[torch.Tensor],
    work_dtype: torch.dtype,
    cache: Dict[int, torch.Tensor],
    fp16_max: float = _FP16_GRAD_MAX,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Promotes (mu, nu, pi), recovering _work_val or rescaling ULP-valid simplex triples before subtraction."""
    raw_lp_dtypes = [t.dtype for t in (mu, nu, pi) if t is not None and t.dtype in _HALF_PREC_DTYPES]
    all_have_work = (
        (mu.dtype not in _HALF_PREC_DTYPES or _has_valid_work_snap(mu))
        and (nu.dtype not in _HALF_PREC_DTYPES or _has_valid_work_snap(nu))
        and (pi is None or pi.dtype not in _HALF_PREC_DTYPES or _has_valid_work_snap(pi))
    )
    mu_w = _promote_low_prec_cached(mu, work_dtype, cache, fp16_max=fp16_max)
    nu_w = _promote_low_prec_cached(nu, work_dtype, cache, fp16_max=fp16_max)
    pi_w = _promote_low_prec_cached(pi, work_dtype, cache, fp16_max=fp16_max) if pi is not None else None
    assert mu_w is not None and nu_w is not None
    if (not all_have_work) and raw_lp_dtypes and pi_w is not None:
        lp_eps = max(float(torch.finfo(dt).eps) for dt in raw_lp_dtypes)
        mu_f = mu_w.detach()
        nu_f = nu_w.detach()
        pi_f = pi_w.detach()
        tot = mu_w + nu_w + pi_w
        tot_d = tot.detach()
        ulp_ok = (
            torch.isfinite(mu_f)
            & torch.isfinite(nu_f)
            & torch.isfinite(pi_f)
            & (mu_f >= 0.0)
            & (mu_f <= 1.0)
            & (nu_f >= 0.0)
            & (nu_f <= 1.0)
            & (pi_f >= 0.0)
            & (pi_f <= 1.0)
            & (tot_d > 0.0)
            & (tot_d != 1.0)
            & ((tot_d - 1.0).abs() <= 2.0 * lp_eps)
        )
        if bool(ulp_ok.any()):
            safe_tot = torch.where(ulp_ok, tot, torch.ones_like(tot))
            mu_w = mu_w / safe_tot
            nu_w = nu_w / safe_tot
            pi_w = pi_w / safe_tot
    return mu_w, nu_w, pi_w


def _infer_fp_dtype(obj, module: Optional[nn.Module] = None) -> Tuple[Optional[torch.dtype], torch.device]:
    fp_dtype: Optional[torch.dtype] = None
    first_device: Optional[torch.device] = None

    def _visit(x):
        nonlocal fp_dtype, first_device
        if isinstance(x, torch.Tensor):
            if first_device is None:
                first_device = x.device
            if x.dtype.is_floating_point:
                fp_dtype = x.dtype if fp_dtype is None else torch.promote_types(fp_dtype, x.dtype)
        elif isinstance(x, IntuitionisticFuzzyTensor):
            _visit(x.mu)
            _visit(x.nu)
            _visit(x.pi)
        elif isinstance(x, dict):
            for v in x.values():
                _visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _visit(v)

    _visit(obj)
    if module is not None:
        for p in module.parameters():
            if first_device is None:
                first_device = p.device
            if p.dtype.is_floating_point:
                fp_dtype = p.dtype if fp_dtype is None else torch.promote_types(fp_dtype, p.dtype)
        for b in module.buffers():
            if first_device is None:
                first_device = b.device
            if b.dtype.is_floating_point:
                fp_dtype = b.dtype if fp_dtype is None else torch.promote_types(fp_dtype, b.dtype)
    return fp_dtype, (first_device or torch.device("cpu"))


def _collect_lp_info(obj, module: Optional[nn.Module] = None) -> Tuple[Optional[torch.dtype], torch.device]:
    lp_dtype: Optional[torch.dtype] = None
    first_device: Optional[torch.device] = None

    def _visit(x):
        nonlocal lp_dtype, first_device
        if isinstance(x, torch.Tensor):
            if first_device is None:
                first_device = x.device
            if lp_dtype is None and x.dtype in _HALF_PREC_DTYPES:
                lp_dtype = x.dtype
        elif isinstance(x, IntuitionisticFuzzyTensor):
            _visit(x.mu)
            _visit(x.nu)
            _visit(x.pi)
        elif isinstance(x, dict):
            for v in x.values():
                _visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _visit(v)

    _visit(obj)
    if module is not None:
        for p in module.parameters():
            if first_device is None:
                first_device = p.device
            if lp_dtype is None and p.dtype in _HALF_PREC_DTYPES:
                lp_dtype = p.dtype
            if lp_dtype is not None:
                break
        if lp_dtype is None:
            for b in module.buffers():
                if first_device is None:
                    first_device = b.device
                if b.dtype in _HALF_PREC_DTYPES:
                    lp_dtype = b.dtype
                    break
    dev = first_device or torch.device("cpu")
    if lp_dtype is not None:
        return lp_dtype, dev
    if _active_autocast_dtype(dev) is not None:
        in_fp_dt, _ = _infer_fp_dtype(obj, module)
        return (in_fp_dt or torch.float32), dev
    return None, dev


def _promote_tree_to_work(
    obj,
    work_dtype: torch.dtype,
    cache: Dict[int, torch.Tensor],
    fp16_max: float = _FP16_GRAD_MAX,
):
    if isinstance(obj, torch.Tensor):
        if obj.dtype in _HALF_PREC_DTYPES:
            return _promote_low_prec_cached(obj, work_dtype, cache, fp16_max=fp16_max)
        if obj.dtype.is_floating_point and obj.dtype != work_dtype:
            return obj.to(dtype=work_dtype)
        return obj
    if isinstance(obj, IntuitionisticFuzzyTensor):
        mu_w, nu_w, pi_w = _sanitize_half_prec_ifn_for_work(
            obj.mu, obj.nu, obj.pi, work_dtype, cache, fp16_max=fp16_max
        )
        return IntuitionisticFuzzyTensor(mu=mu_w, nu=nu_w, pi=pi_w)
    if isinstance(obj, dict):
        return {k: _promote_tree_to_work(v, work_dtype, cache, fp16_max=fp16_max) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_promote_tree_to_work(v, work_dtype, cache, fp16_max=fp16_max) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_promote_tree_to_work(v, work_dtype, cache, fp16_max=fp16_max) for v in obj)
    return obj


def _cast_tree_from_work(obj, orig_dtype: torch.dtype):
    if isinstance(obj, torch.Tensor):
        if obj.dtype == orig_dtype or not obj.dtype.is_floating_point:
            return obj
        return _bridge_work_via_lp(obj, orig_dtype)[1] if orig_dtype in _HALF_PREC_DTYPES else obj.to(dtype=orig_dtype)
    if isinstance(obj, IntuitionisticFuzzyTensor):
        return IntuitionisticFuzzyTensor(
            mu=_cast_tree_from_work(obj.mu, orig_dtype),
            nu=_cast_tree_from_work(obj.nu, orig_dtype),
            pi=_cast_tree_from_work(obj.pi, orig_dtype),
        )
    if isinstance(obj, dict):
        return {k: _cast_tree_from_work(v, orig_dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cast_tree_from_work(v, orig_dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cast_tree_from_work(v, orig_dtype) for v in obj)
    return obj


def _run_module_in_work_prec(
    module: nn.Module,
    orig_dtype: torch.dtype,
    device: torch.device,
    args: tuple,
    kwargs: Optional[Dict[str, object]] = None,
    gate_return_dict: bool = False,
):
    work_dtype = _work_dtype_for(device)
    cache: Dict[int, torch.Tensor] = {}
    params_and_buffers: Dict[str, torch.Tensor] = {}
    for name, p in module.named_parameters():
        if p.dtype in _HALF_PREC_DTYPES:
            params_and_buffers[name] = _promote_low_prec_cached(p, work_dtype, cache)  # type: ignore[assignment]
        elif p.dtype.is_floating_point and p.dtype != work_dtype:
            params_and_buffers[name] = p.to(dtype=work_dtype)
        else:
            params_and_buffers[name] = p
    for name, b in module.named_buffers():
        if b.dtype in _HALF_PREC_DTYPES:
            params_and_buffers[name] = _promote_low_prec_cached(b, work_dtype, cache)  # type: ignore[assignment]
        elif b.dtype.is_floating_point and b.dtype != work_dtype:
            params_and_buffers[name] = b.to(dtype=work_dtype)
        else:
            params_and_buffers[name] = b

    fp16_in_cap = _fp16_cap_for_inputs((args, kwargs)) if orig_dtype == torch.float16 else _FP16_GRAD_MAX
    args_w = _promote_tree_to_work(args, work_dtype, cache, fp16_max=fp16_in_cap)
    kwargs_w = _promote_tree_to_work(kwargs or {}, work_dtype, cache, fp16_max=fp16_in_cap)

    prev_active = getattr(_HALF_PREC_TLS, "active", False)
    prev_dtype = getattr(_HALF_PREC_TLS, "orig_dtype", None)
    prev_gate = getattr(_HALF_PREC_TLS, "gate_return_dict", False)
    _HALF_PREC_TLS.active = True
    _HALF_PREC_TLS.orig_dtype = orig_dtype
    _HALF_PREC_TLS.gate_return_dict = gate_return_dict
    try:
        with _disable_autocast_ctx(device):
            out_w = functional_call(module, params_and_buffers, args_w, kwargs_w)
    finally:
        _HALF_PREC_TLS.active = prev_active
        _HALF_PREC_TLS.orig_dtype = prev_dtype
        _HALF_PREC_TLS.gate_return_dict = prev_gate
    return _cast_tree_from_work(out_w, orig_dtype)


def _clamp_preserve_boundary_grad(
    t: torch.Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> torch.Tensor:
    """Clamps `t` onto `[min_val, max_val]` while preserving unit derivative at feasible closed boundaries."""
    cond = torch.isfinite(t)
    if min_val is not None:
        cond = cond & (t >= min_val)
    if max_val is not None:
        cond = cond & (t <= max_val)
    return torch.where(cond, t, t.clamp(min=min_val, max=max_val))


@dataclass
class IntuitionisticFuzzyTensor:
    """
    Differentiable container for an Atanassov Intuitionistic Fuzzy Number (IFN) batch
    <mu, nu, pi> on the Intuitionistic Fuzzy Simplex (up to target-dtype IEEE-754 rounding):
        mu + nu + pi = 1,   mu, nu, pi in [0, 1].
    """

    mu: torch.Tensor
    nu: torch.Tensor
    pi: torch.Tensor

    def validate(self, atol: float = 1e-4) -> bool:
        """Verifies finite simplex membership with strict absolute tolerance (`rtol=0.0`)."""
        if atol < 0.0:
            return False
        if not (torch.isfinite(self.mu).all() and torch.isfinite(self.nu).all() and torch.isfinite(self.pi).all()):
            return False
        in_bounds = (
            (self.mu >= -atol).all()
            and (self.nu >= -atol).all()
            and (self.pi >= -atol).all()
            and (self.mu <= 1.0 + atol).all()
            and (self.nu <= 1.0 + atol).all()
            and (self.pi <= 1.0 + atol).all()
        )
        closure = torch.allclose(
            self.mu + self.nu + self.pi,
            torch.ones_like(self.mu),
            rtol=0.0,
            atol=atol,
        )
        return bool(in_bounds and closure)

    def expected_probability(self) -> torch.Tensor:
        """Pignistic / intuitionistic score probability P(y=1) = mu + 0.5 * pi."""
        with _disable_autocast_ctx(self.mu.device):
            if (
                self.mu.dtype in _HALF_PREC_DTYPES
                or self.nu.dtype in _HALF_PREC_DTYPES
                or self.pi.dtype in _HALF_PREC_DTYPES
            ):
                orig_dt, dev = _collect_lp_info((self.mu, self.nu, self.pi))
                work_dt = _work_dtype_for(dev)
                cache: Dict[int, torch.Tensor] = {}
                fp16_cap = _fp16_cap_for_inputs((self.mu, self.nu, self.pi))
                mu_w, _, pi_w = _sanitize_half_prec_ifn_for_work(
                    self.mu, self.nu, self.pi, work_dt, cache, fp16_max=fp16_cap
                )
                assert pi_w is not None
                p_w = _clamp_preserve_boundary_grad(mu_w + 0.5 * pi_w, 0.0, 1.0)
                return _bridge_work_via_lp(p_w, orig_dt)[1]
            return _clamp_preserve_boundary_grad(self.mu + 0.5 * self.pi, 0.0, 1.0)

    def certainty(self) -> torch.Tensor:
        """Observed non-vacuity mass r = 1 - pi = mu + nu."""
        with _disable_autocast_ctx(self.mu.device):
            if (
                self.mu.dtype in _HALF_PREC_DTYPES
                or self.nu.dtype in _HALF_PREC_DTYPES
                or self.pi.dtype in _HALF_PREC_DTYPES
            ):
                orig_dt, dev = _collect_lp_info((self.mu, self.nu, self.pi))
                work_dt = _work_dtype_for(dev)
                cache: Dict[int, torch.Tensor] = {}
                fp16_cap = _fp16_cap_for_inputs((self.mu, self.nu, self.pi))
                _, _, pi_w = _sanitize_half_prec_ifn_for_work(
                    self.mu, self.nu, self.pi, work_dt, cache, fp16_max=fp16_cap
                )
                assert pi_w is not None
                return _bridge_work_via_lp((1.0 - pi_w).clamp(0.0, 1.0), orig_dt)[1]
            return (1.0 - self.pi).clamp(0.0, 1.0)

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.mu
        yield self.nu
        yield self.pi


class IntuitionisticFuzzyEncoder(nn.Module):
    """
    Projects an arbitrary feature tensor x in R^{B x D} into a valid Atanassov
    Intuitionistic Fuzzy Number (IFN) triple <mu, nu, pi> on the 2-simplex:
        mu(x) + nu(x) + pi(x) = 1,   mu, nu, pi in [0, 1].

    When a modality is unobserved (modality_mask == 0), its positive and negative
    evidences are identically zeroed out, collapsing the representation to the exact
    vacuous IFN state <0, 0, 1> (pure epistemic ignorance) without imputation bias.
    """

    def __init__(
        self,
        in_dim: Optional[int] = None,
        num_classes: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        prior_vacuity: float = 1.2,
        in_features: Optional[int] = None,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        feat_dim = in_dim if in_dim is not None else in_features
        if feat_dim is None:
            raise ValueError("Either in_dim or in_features must be specified.")
        if out_dim is not None:
            num_classes = out_dim
        self.in_dim = feat_dim
        self.in_features = feat_dim
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Direct linear skip connection preserves 1D/low-D quantitative biomarkers
        self.skip_pos = nn.Linear(feat_dim, num_classes)
        self.skip_neg = nn.Linear(feat_dim, num_classes)
        self.pos_evidence_head = nn.Linear(hidden_dim, num_classes)
        self.neg_evidence_head = nn.Linear(hidden_dim, num_classes)
        self.prior_hesitancy = nn.Parameter(torch.ones(1, num_classes) * float(prior_vacuity))

    def _forward_high_prec_f64(
        self,
        x_clean: torch.Tensor,
        m_bin: Optional[torch.Tensor],
        out_dtype: torch.dtype,
    ) -> IntuitionisticFuzzyTensor:
        """Overflow-safe float64 fallback for extreme-magnitude inputs or linear head sums."""
        dev = x_clean.device

        def _to_f64(t: torch.Tensor) -> torch.Tensor:
            if t.device.type == "mps":
                return t.to("cpu").to(dtype=torch.float64)
            return t.to(dtype=torch.float64)

        def _from_f64(t64: torch.Tensor) -> torch.Tensor:
            if dev.type == "mps":
                return t64.to(dtype=out_dtype).to(device=dev)
            return t64.to(device=dev, dtype=out_dtype)

        def _add_elem_ls(
            u1: torch.Tensor, ls1: torch.Tensor, u2: torch.Tensor, ls2: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            lmax = torch.maximum(ls1, ls2).detach()
            u_sum = u1 * torch.exp((ls1 - lmax).detach()) + u2 * torch.exp((ls2 - lmax).detach())
            peak = u_sum.detach().abs()
            zero_m = peak <= 0.0
            safe_p = torch.where(zero_m, torch.ones_like(peak), peak)
            shift = torch.clamp(torch.log(safe_p), min=-lmax).detach()
            new_ls = torch.where(zero_m, torch.zeros_like(lmax), lmax + shift).detach()
            half_dl = (0.5 * (lmax - new_ls)).detach()
            rest_dl = ((lmax - new_ls) - half_dl).detach()
            u_out = torch.where(zero_m, torch.zeros_like(u_sum), (u_sum * torch.exp(half_dl)) * torch.exp(rest_dl))
            return u_out, new_ls

        def _renorm_elem_ls(u: torch.Tensor, log_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            if u.numel() == 0:
                return u, torch.zeros_like(log_s)
            peak = u.detach().abs()
            zero_m = peak <= 0.0
            safe_p = torch.where(zero_m, torch.ones_like(peak), peak)
            shift = torch.clamp(torch.log(safe_p), min=-log_s.detach()).detach()
            new_ls = torch.where(zero_m, torch.zeros_like(log_s), log_s.detach() + shift).detach()
            half_dl = (0.5 * (log_s.detach() - new_ls)).detach()
            rest_dl = ((log_s.detach() - new_ls) - half_dl).detach()
            u_out = torch.where(zero_m, torch.zeros_like(u), (u * torch.exp(half_dl)) * torch.exp(rest_dl))
            return u_out, new_ls

        def _linear_elem_ls(
            u: torch.Tensor,
            log_s: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            w64 = _to_f64(weight)
            b_sz, d_in = u.shape[0], u.shape[-1]
            d_out = w64.shape[0]
            if d_in == 0:
                prod_u = torch.zeros((b_sz, d_out), device=u.device, dtype=u.dtype)
                prod_ls = torch.zeros_like(prod_u)
            else:
                w_fin = torch.where(torch.isfinite(w64), w64, torch.zeros_like(w64))
                uw, lw = _renorm_elem_ls(w_fin, torch.zeros_like(w_fin))
                chunk_o = min(64, max(1, (2 * 1024 * 1024) // max(1, b_sz * d_in)))
                u_chunks: List[torch.Tensor] = []
                ls_chunks: List[torch.Tensor] = []
                nz_u = u.detach() != 0.0
                nz_w = uw.detach() != 0.0
                for o0 in range(0, d_out, chunk_o):
                    o1 = min(d_out, o0 + chunk_o)
                    uw_c = uw[o0:o1]
                    lw_c = lw[o0:o1]
                    act = nz_u.unsqueeze(1) & nz_w[o0:o1].unsqueeze(0)
                    pair_ls = torch.where(
                        act,
                        log_s.detach().unsqueeze(1) + lw_c.detach().unsqueeze(0),
                        torch.full((b_sz, o1 - o0, d_in), -float("inf"), device=u.device, dtype=u.dtype),
                    )
                    max_ls_raw = pair_ls.amax(dim=-1)
                    has_act = torch.isfinite(max_ls_raw)
                    max_ls = torch.where(has_act, max_ls_raw, torch.zeros_like(max_ls_raw)).detach()
                    rel = torch.where(
                        act,
                        torch.exp((pair_ls - max_ls.unsqueeze(-1)).clamp(min=-800.0, max=0.0)),
                        torch.zeros_like(pair_ls),
                    )
                    u_raw = (u.unsqueeze(1) * uw_c.unsqueeze(0) * rel).sum(dim=-1)
                    u_c, ls_c = _renorm_elem_ls(u_raw, max_ls)
                    u_chunks.append(u_c)
                    ls_chunks.append(ls_c)
                prod_u = torch.cat(u_chunks, dim=-1)
                prod_ls = torch.cat(ls_chunks, dim=-1)
            if torch.isinf(w64).any() and d_in > 0:
                # Inf weight rows retain sign direction along non-zero u inputs at dominating log-scale
                w_inf_pos = torch.isposinf(w64).to(dtype=u.dtype)
                w_inf_neg = torch.isneginf(w64).to(dtype=u.dtype)
                inf_dir = F.linear(u, w_inf_pos - w_inf_neg)
                has_inf_w = inf_dir.abs() > 0.0
                prod_u = torch.where(has_inf_w, torch.sign(inf_dir), prod_u)
                prod_ls = torch.where(
                    has_inf_w,
                    torch.maximum(prod_ls + 50.0, torch.full_like(prod_ls, 1000.0)),
                    prod_ls,
                )
            if bias is not None:
                b64 = _to_f64(bias)
                b_fin = torch.where(torch.isfinite(b64), b64, torch.zeros_like(b64))
                bu_1, bls_1 = _renorm_elem_ls(b_fin, torch.zeros_like(b_fin))
                bu = bu_1.unsqueeze(0).expand_as(prod_u)
                bls = bls_1.unsqueeze(0).expand_as(prod_ls)
                u_sum, ls_sum = _add_elem_ls(prod_u, prod_ls, bu, bls)
                if torch.isposinf(b64).any() or torch.isneginf(b64).any():
                    b_pi = torch.isposinf(b64).unsqueeze(0).expand_as(u_sum)
                    b_ni = torch.isneginf(b64).unsqueeze(0).expand_as(u_sum)
                    u_sum = torch.where(b_pi, torch.ones_like(u_sum), torch.where(b_ni, -torch.ones_like(u_sum), u_sum))
                    ls_sum = torch.where(
                        b_pi | b_ni,
                        torch.maximum(ls_sum + 50.0, torch.full_like(ls_sum, 1000.0)),
                        ls_sum,
                    )
                return u_sum, ls_sum
            return prod_u, prod_ls

        x64 = _to_f64(x_clean)
        m_bin_w = _to_f64(m_bin) if m_bin is not None else None
        ph64 = _to_f64(self.prior_hesitancy)

        h64 = x64
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                h64 = F.linear(
                    h64,
                    _to_f64(layer.weight),
                    _to_f64(layer.bias) if layer.bias is not None else None,
                )
            elif isinstance(layer, nn.GELU):
                h64 = F.gelu(h64)
            elif isinstance(layer, nn.Dropout):
                h64 = F.dropout(h64, p=layer.p, training=self.training)
            else:
                h64 = _to_f64(layer(h64.to(dtype=layer.weight.dtype if hasattr(layer, "weight") else out_dtype)))

        p_head = F.linear(
            h64,
            _to_f64(self.pos_evidence_head.weight),
            _to_f64(self.pos_evidence_head.bias) if self.pos_evidence_head.bias is not None else None,
        )
        p_skip = F.linear(
            x64,
            _to_f64(self.skip_pos.weight),
            _to_f64(self.skip_pos.bias) if self.skip_pos.bias is not None else None,
        )
        n_head = F.linear(
            h64,
            _to_f64(self.neg_evidence_head.weight),
            _to_f64(self.neg_evidence_head.bias) if self.neg_evidence_head.bias is not None else None,
        )
        n_skip = F.linear(
            x64,
            _to_f64(self.skip_neg.weight),
            _to_f64(self.skip_neg.bias) if self.skip_neg.bias is not None else None,
        )
        pos_logit = p_head + p_skip
        neg_logit = n_head + n_skip

        def _finalize_scaled_s(
            alpha_s: torch.Tensor,
            beta_s: torch.Tensor,
            pi_0_s: torch.Tensor,
            m_row: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            total_s = (alpha_s + beta_s + pi_0_s).clamp(min=1e-300)
            mu_w = alpha_s / total_s
            nu_w = beta_s / total_s
            pi_raw_w = (pi_0_s / total_s).expand_as(alpha_s)
            pi_unnorm = (m_row * pi_raw_w + (1.0 - m_row)) if m_row is not None else pi_raw_w
            return mu_w, nu_w, pi_unnorm.clamp(0.0, 1.0)

        def _eval_elem_ls_rows(
            x_sub: torch.Tensor, m_sub: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Per-element log-scale evaluation (u * exp(log_s)) keeps hidden units & batch rows independent.
            u_x, ls_x = _renorm_elem_ls(x_sub, torch.zeros_like(x_sub))
            u_h, ls_h = u_x, ls_x
            for layer in self.backbone:
                if isinstance(layer, nn.Linear):
                    u_h, ls_h = _linear_elem_ls(u_h, ls_h, layer.weight, layer.bias)
                elif isinstance(layer, nn.GELU):
                    small_m = ls_h.detach() <= 500.0
                    z_fin = torch.where(small_m, u_h, torch.zeros_like(u_h)) * torch.exp(
                        ls_h.detach().clamp(min=-800.0, max=500.0)
                    )
                    g_small, ls_small = _renorm_elem_ls(F.gelu(z_fin), torch.zeros_like(ls_h))
                    u_large = torch.where(u_h.detach() > 0.0, u_h, torch.zeros_like(u_h))
                    ls_large = torch.where(u_h.detach() > 0.0, ls_h, torch.zeros_like(ls_h))
                    u_h = torch.where(small_m, g_small, u_large)
                    ls_h = torch.where(small_m, ls_small, ls_large).detach()
                elif isinstance(layer, nn.Dropout):
                    u_h, ls_h = _renorm_elem_ls(F.dropout(u_h, p=layer.p, training=self.training), ls_h)
                else:
                    raw_l = _to_f64(layer((u_h * torch.exp(ls_h.detach().clamp(max=50.0))).to(
                        dtype=layer.weight.dtype if hasattr(layer, "weight") else out_dtype
                    )))
                    u_h, ls_h = _renorm_elem_ls(raw_l, torch.zeros_like(raw_l))

            ph_u, ph_ls = _linear_elem_ls(
                u_h, ls_h, self.pos_evidence_head.weight, self.pos_evidence_head.bias
            )
            ps_u, ps_ls = _linear_elem_ls(
                u_x, ls_x, self.skip_pos.weight, self.skip_pos.bias
            )
            nh_u, nh_ls = _linear_elem_ls(
                u_h, ls_h, self.neg_evidence_head.weight, self.neg_evidence_head.bias
            )
            ns_u, ns_ls = _linear_elem_ls(
                u_x, ls_x, self.skip_neg.weight, self.skip_neg.bias
            )

            pos_u, pos_ls = _add_elem_ls(ph_u, ph_ls, ps_u, ps_ls)
            neg_u, neg_ls = _add_elem_ls(nh_u, nh_ls, ns_u, ns_ls)

            def _softplus_ls(u_in: torch.Tensor, ls_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                small_ls = ls_in <= 500.0
                z_small = torch.where(small_ls, u_in, torch.zeros_like(u_in)) * torch.exp(
                    ls_in.detach().clamp(min=-800.0, max=500.0)
                )
                sp_small = F.softplus(z_small, threshold=20.0)
                u_large = torch.clamp(u_in, min=0.0)
                ls_large = torch.where(u_in.detach() > 0.0, ls_in, torch.zeros_like(ls_in))
                return (
                    torch.where(small_ls, sp_small, u_large),
                    torch.where(small_ls, torch.zeros_like(ls_in), ls_large).detach(),
                )

            au, als = _softplus_ls(pos_u, pos_ls)
            bu, bls = _softplus_ls(neg_u, neg_ls)
            if m_sub is not None:
                au = au * m_sub
                bu = bu * m_sub
                obs_m = m_sub > 0.5
                als = torch.where(obs_m, als, torch.zeros_like(als)).detach()
                bls = torch.where(obs_m, bls, torch.zeros_like(bls)).detach()

            ph_fin_m = torch.isfinite(ph64)
            ph_pinf = torch.isposinf(ph64)
            ph_fin = torch.where(ph_fin_m, ph64, torch.full_like(ph64, -100.0))
            large_ph = ph_fin_m & (ph_fin.detach() > 500.0)
            s_ph = torch.where(large_ph, ph_fin.detach().abs().clamp(min=1.0), torch.ones_like(ph_fin))
            sp_ph = F.softplus(torch.where(large_ph, torch.zeros_like(ph_fin), ph_fin), threshold=20.0) + 1e-4
            u_ph_f = torch.where(large_ph, ph_fin / s_ph, sp_ph)
            ls_ph_f = torch.where(large_ph, torch.log(s_ph).detach(), torch.zeros_like(ph_fin)).detach()
            pi0_u = u_ph_f.expand_as(au)
            pi0_ls = ls_ph_f.expand_as(als)
            if bool(ph_pinf.any()):
                pinf_exp = ph_pinf.expand_as(au)
                pi0_u = torch.where(pinf_exp, torch.ones_like(pi0_u), pi0_u)
                pi0_ls = torch.where(pinf_exp, torch.maximum(als, bls).detach() + 100.0, pi0_ls).detach()

            ltot = torch.maximum(torch.maximum(als, bls), pi0_ls).detach()
            alpha_s = au * torch.exp((als - ltot).clamp(min=-800.0).detach())
            beta_s = bu * torch.exp((bls - ltot).clamp(min=-800.0).detach())
            pi_0_s = pi0_u * torch.exp((pi0_ls - ltot).clamp(min=-800.0).detach())
            max_fin = torch.maximum(torch.maximum(alpha_s, beta_s), pi_0_s).detach().clamp(min=1.0)
            return _finalize_scaled_s(alpha_s / max_fin, beta_s / max_fin, pi_0_s / max_fin, m_sub)

        ph_fin_ok = bool(torch.isfinite(ph64).all())
        if ph_fin_ok and x64.shape[0] > 0:
            r_dims_h = tuple(range(1, h64.ndim))
            h_row_fin = torch.isfinite(h64.detach()).all(dim=r_dims_h) if r_dims_h else torch.isfinite(h64.detach())
            row_fin64 = (
                h_row_fin
                & torch.isfinite(p_head.detach()).all(dim=-1)
                & torch.isfinite(p_skip.detach()).all(dim=-1)
                & torch.isfinite(n_head.detach()).all(dim=-1)
                & torch.isfinite(n_skip.detach()).all(dim=-1)
                & torch.isfinite(pos_logit.detach()).all(dim=-1)
                & torch.isfinite(neg_logit.detach()).all(dim=-1)
            )
        else:
            row_fin64 = torch.full((x64.shape[0],), ph_fin_ok, device=x64.device, dtype=torch.bool)

        if bool(row_fin64.all()):
            pi_0_w = F.softplus(ph64, threshold=20.0) + 1e-4
            alpha_w = F.softplus(pos_logit, threshold=20.0)
            beta_w = F.softplus(neg_logit, threshold=20.0)
            if m_bin_w is not None:
                alpha_w = alpha_w * m_bin_w
                beta_w = beta_w * m_bin_w
            max_e = torch.maximum(torch.maximum(alpha_w, beta_w), pi_0_w).detach().clamp(min=1.0)
            mu_w, nu_w, pi_w = _finalize_scaled_s(alpha_w / max_e, beta_w / max_e, pi_0_w / max_e, m_bin_w)
        elif x64.ndim >= 2 and x64.shape[0] > 1 and bool(row_fin64.any()):
            idx_fin = torch.where(row_fin64)[0]
            idx_ovf = torch.where(~row_fin64)[0]
            pi_0_f = F.softplus(ph64, threshold=20.0) + 1e-4
            alpha_f = F.softplus(pos_logit[idx_fin], threshold=20.0)
            beta_f = F.softplus(neg_logit[idx_fin], threshold=20.0)
            m_fin = m_bin_w[idx_fin] if m_bin_w is not None else None
            if m_fin is not None:
                alpha_f = alpha_f * m_fin
                beta_f = beta_f * m_fin
            max_e_f = torch.maximum(torch.maximum(alpha_f, beta_f), pi_0_f).detach().clamp(min=1.0)
            mu_f, nu_f, pi_f = _finalize_scaled_s(alpha_f / max_e_f, beta_f / max_e_f, pi_0_f / max_e_f, m_fin)
            m_ovf = m_bin_w[idx_ovf] if m_bin_w is not None else None
            mu_o, nu_o, pi_o = _eval_elem_ls_rows(x64[idx_ovf], m_ovf)
            mu_w = torch.zeros_like(pos_logit).index_copy(0, idx_fin, mu_f).index_copy(0, idx_ovf, mu_o)
            nu_w = torch.zeros_like(neg_logit).index_copy(0, idx_fin, nu_f).index_copy(0, idx_ovf, nu_o)
            pi_w = torch.zeros_like(pos_logit).index_copy(0, idx_fin, pi_f).index_copy(0, idx_ovf, pi_o)
        else:
            mu_w, nu_w, pi_w = _eval_elem_ls_rows(x64, m_bin_w)

        return IntuitionisticFuzzyTensor(mu=_from_f64(mu_w), nu=_from_f64(nu_w), pi=_from_f64(pi_w))

    def forward(
        self, x: torch.Tensor, modality_mask: Optional[torch.Tensor] = None
    ) -> IntuitionisticFuzzyTensor:
        """
        Args:
            x: Input tensor of shape [B, D]
            modality_mask: Optional binary observation indicator [B] or [B, 1]
                           (1 = observed, 0 = missing modality)
        Returns:
            IntuitionisticFuzzyTensor(mu, nu, pi) (also unpackable as mu, nu, pi)
        """
        if not getattr(_HALF_PREC_TLS, "active", False):
            lp_dt, dev = _collect_lp_info((x, modality_mask), self)
            if lp_dt is not None:
                return _run_module_in_work_prec(self, lp_dt, dev, (x, modality_mask))

        # Sanitize NaNs before any linear projection so masked NaNs never propagate (M5)
        x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if modality_mask is not None:
            m = modality_mask.to(device=x_clean.device, dtype=x_clean.dtype)
            if m.dim() == 1:
                m = m.unsqueeze(-1)
            m_bin = (m > 0.5).to(dtype=x_clean.dtype)
            x_clean = x_clean * m_bin
        else:
            m_bin = None

        ph_ok = bool(torch.isfinite(self.prior_hesitancy.detach()).all()) and (
            self.prior_hesitancy.numel() == 0
            or float(self.prior_hesitancy.detach().abs().max().item()) <= 1e15
        )
        if not ph_ok:
            return self._forward_high_prec_f64(x_clean, m_bin, x_clean.dtype)

        def _split_and_merge_rows(row_ok: torch.Tensor) -> IntuitionisticFuzzyTensor:
            if not bool(row_ok.any()) or x_clean.ndim < 2 or x_clean.shape[0] <= 1:
                return self._forward_high_prec_f64(x_clean, m_bin, x_clean.dtype)
            idx_ok = torch.where(row_ok)[0]
            idx_bad = torch.where(~row_ok)[0]
            m_ok = modality_mask[idx_ok] if modality_mask is not None else None
            m_bad = m_bin[idx_bad] if m_bin is not None else None
            ifn_ok = self.forward(x[idx_ok], m_ok)
            ifn_bad = self._forward_high_prec_f64(x_clean[idx_bad], m_bad, x_clean.dtype)

            def _merge(ta: torch.Tensor, tb: torch.Tensor) -> torch.Tensor:
                out = torch.zeros((x_clean.shape[0], *ta.shape[1:]), device=ta.device, dtype=ta.dtype)
                return out.index_copy(0, idx_ok, ta).index_copy(0, idx_bad, tb.to(device=ta.device, dtype=ta.dtype))

            return IntuitionisticFuzzyTensor(
                mu=_merge(ifn_ok.mu, ifn_bad.mu),
                nu=_merge(ifn_ok.nu, ifn_bad.nu),
                pi=_merge(ifn_ok.pi, ifn_bad.pi),
            )

        h = self.backbone(x_clean)
        r_dims_h = tuple(range(1, h.ndim))
        h_ok = torch.isfinite(h.detach()).all(dim=r_dims_h) if (h.ndim >= 2 and h.shape[0] > 0) else torch.isfinite(h.detach())
        if not bool(h_ok.all()):
            return _split_and_merge_rows(h_ok)

        p_head = self.pos_evidence_head(h)
        p_skip = self.skip_pos(x_clean)
        n_head = self.neg_evidence_head(h)
        n_skip = self.skip_neg(x_clean)
        if x_clean.ndim >= 2 and x_clean.shape[0] > 0:
            row_ok = (
                torch.isfinite(p_head.detach()).all(dim=-1)
                & torch.isfinite(p_skip.detach()).all(dim=-1)
                & torch.isfinite(n_head.detach()).all(dim=-1)
                & torch.isfinite(n_skip.detach()).all(dim=-1)
                & (p_head.detach().abs().amax(dim=-1) <= 1e15)
                & (p_skip.detach().abs().amax(dim=-1) <= 1e15)
                & (n_head.detach().abs().amax(dim=-1) <= 1e15)
                & (n_skip.detach().abs().amax(dim=-1) <= 1e15)
            )
            if not bool(row_ok.all()):
                return _split_and_merge_rows(row_ok)
        else:
            pre_ok = bool(
                torch.isfinite(p_head).all()
                and torch.isfinite(p_skip).all()
                and torch.isfinite(n_head).all()
                and torch.isfinite(n_skip).all()
            )
            pre_mags = [
                float(t.detach().abs().max().item())
                for t in (p_head, p_skip, n_head, n_skip)
                if t.numel() > 0
            ]
            if (not pre_ok) or (pre_mags and max(pre_mags) > 1e15):
                return self._forward_high_prec_f64(x_clean, m_bin, x_clean.dtype)

        alpha = F.softplus(p_head + p_skip)
        beta = F.softplus(n_head + n_skip)
        pi_0 = F.softplus(self.prior_hesitancy) + 1e-4

        if m_bin is not None:
            alpha = alpha * m_bin
            beta = beta * m_bin

        max_ev_raw = torch.maximum(alpha, beta).detach()
        large_ev = max_ev_raw > 1e15
        alpha_fast = torch.where(large_ev, torch.ones_like(alpha), alpha)
        beta_fast = torch.where(large_ev, torch.ones_like(beta), beta)
        total = (alpha_fast + beta_fast + pi_0).clamp(min=1e-8)
        mu_std = alpha_fast / total
        nu_std = beta_fast / total
        near_one = (mu_std + nu_std).detach() >= 0.999
        if getattr(_HALF_PREC_TLS, "active", False) or bool(near_one.any()) or bool(large_ev.any()):
            work_dt = _work_dtype_for(x_clean.device)
            alpha_w = alpha.to(dtype=work_dt)
            beta_w = beta.to(dtype=work_dt)
            pi_0_w = pi_0.to(dtype=work_dt)
            max_e = torch.maximum(torch.maximum(alpha_w, beta_w), pi_0_w).detach().clamp(min=1.0)
            alpha_s = alpha_w / max_e
            beta_s = beta_w / max_e
            pi_0_s = pi_0_w / max_e
            total_s = (alpha_s + beta_s + pi_0_s).clamp(min=1e-30)
            mu_w = (alpha_s / total_s).to(dtype=mu_std.dtype)
            nu_w = (beta_s / total_s).to(dtype=nu_std.dtype)
            pi_raw_w = (pi_0_s / total_s).expand_as(alpha_s)
            m_bin_w = m_bin.to(dtype=work_dt) if m_bin is not None else None
            pi_unnorm = (m_bin_w * pi_raw_w + (1.0 - m_bin_w)) if m_bin_w is not None else pi_raw_w
            pi_w = pi_unnorm.clamp(0.0, 1.0).to(dtype=mu_std.dtype)
            if getattr(_HALF_PREC_TLS, "active", False):
                mu = mu_w
                nu = nu_w
                pi = pi_w
            else:
                use_w = near_one | large_ev
                pi_std = (1.0 - mu_std - nu_std).clamp(0.0, 1.0)
                mu = torch.where(use_w, mu_w, mu_std)
                nu = torch.where(use_w, nu_w, nu_std)
                pi = torch.where(use_w, pi_w, pi_std)
        else:
            mu = mu_std
            nu = nu_std
            pi = (1.0 - mu_std - nu_std).clamp(0.0, 1.0)
        return IntuitionisticFuzzyTensor(mu=mu, nu=nu, pi=pi)


class IFSChoquetFusion(nn.Module):
    """
    Block 1: Non-Additive Monotone Choquet Intuitionistic Fuzzy Integral with
    Observed-Pair Conflict Discounting.

    Fuses M modalities represented as Intuitionistic Fuzzy Numbers
    {(mu_m, nu_m, pi_m)}_{m=1}^M via:
      1. Restricted Base Measure on Observed Modalities O = {m : 1 - pi_m > 0}:
            g_bar_m = (g_m^0 * r_m) / sum_{j=1}^M (g_j^0 * r_j),   r_m = 1 - pi_m.
      2. Descending Certainty Permutation sigma and Non-Additive Monotone Capacity:
            s_m = r_m,
            G(E_k) = (sum_{j=1}^k g_bar_{sigma(j)})^p,   p = exp(0.35 * tanh(eta)) > 0,
            w_{sigma(k)} = G(E_k) - G(E_{k-1}).
         Missing modalities have r_m = 0 => g_bar_m = 0 => w_m = 0 identically.
         Furthermore, since s_m = r_m = 1 - pi_m depends only on non-vacuity mass,
         under constant certainties r_j and frozen conflict discount d > 0 on the interior
         simplex (pi_hat_k > 0 for all active k in O), we have dw_m / dmu_m = 0 and
         dw_m / dnu_m = 0, guaranteeing strict partial-order monotonicity (Theorem 3).
         Equal positive certainties are resolved by canonical modality index order.
      3. Observed-Pair Intuitionistic Conflict:
            C_{m,n} = mu_m * nu_n + nu_m * mu_n,
            C_O = sum_{m != n} a_m a_n C_{m,n} / max(1, sum_{m != n} a_m a_n),
            d = exp(-gamma * C_O).
         Adding vacuous modalities <0, 0, 1> leaves O, C_O, and w_m 100% invariant (Theorem 2).
      4. Conflict-Discounted Capacity-Weighted Geometric-Hesitancy Intuitionistic Aggregation
         (Hesitancy-Preserving IFWA) in Log-Space:
            mu_hat_m = d * mu_m,   nu_hat_m = d * nu_m,   pi_hat_m = 1 - d * r_m,
            mu_fused = 1 - prod_{m=1}^M (1 - mu_hat_m)^{w_m},
            pi_fused = prod_{m=1}^M (pi_hat_m)^{w_m},
            nu_fused = prod_{m=1}^M (1 - mu_hat_m)^{w_m} - prod_{m=1}^M (pi_hat_m)^{w_m}.
          Unlike standard IFWA (Xu, 2007) which takes a geometric product of nu_hat_m (causing
          single-modality nu_hat_m = 0 to zero-veto all opposing evidence), pooling (1 - mu_hat_m)
          and pi_hat_m preserves opposing evidence and decouples pi_fused from evidence polarity.
          Boundary, active-set & tie geometry: (a) on any fixed active set O, when p != 1,
          argsort-ordered capacity increments are C^inf smooth on open strict-certainty chambers
          {r_{sigma(1)} > ... > r_{sigma(|O|)} > 0}; (b) activating/deactivating a near-zero
          modality changes |O| in C_O, producing an interior O_delta(gamma / |O|) jump on
          {r_m <= 1 - delta} and a closed-boundary Theta(gamma^{W_*}) jump when active
          boundary vertices r_j = 1 appear alongside C_O > 0 (vanishing from fused outputs
          at gamma = 0); (c) across a positive-certainty tie block B(r) = {j in O : r_j = r > 0}
          when p != 1, pi_fused is continuous (C^0) everywhere on fixed O, whereas (mu_fused, nu_fused)
          are continuous across B(r) iff all j in B(r) share equal mu_j or some active m* in O
          attains the absorbing state mu_hat_{m*} = 1 (with distinct mu_j weighted asymmetrically
          by index order on the non-absorbing domain); (d) on closed-simplex vertices, crisp
          positive support <1, 0, 0> acts as an absorbing upper vertex when w_m > 0, whereas
          crisp negative refutation <0, 1, 0> follows the hesitancy-preserving residual
          (1 - mu_fused) - pi_fused (reducing to 1 - mu_fused when pi_fused = 0). Malformed
          or non-finite inputs passed directly to IFSChoquetFusion are sanitized radially onto
          the closed simplex.
    """

    def __init__(
        self,
        num_modalities: int,
        num_classes: int = 1,
        conflict_penalty: float = 0.6,
        conflict_gamma: Optional[float] = None,
        uniform_choquet: bool = False,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        gamma_val = conflict_gamma if conflict_gamma is not None else conflict_penalty
        self.conflict_gamma = float(gamma_val)
        self.conflict_penalty = float(gamma_val)
        self.uniform_choquet = uniform_choquet

        # Learnable Sugeno density parameters theta_m and non-additive exponent eta
        self.fuzzy_density_logits = nn.Parameter(torch.zeros(num_modalities, num_classes))
        self.choquet_lambda_raw = nn.Parameter(torch.zeros(1, num_classes))

    @property
    def choquet_logits(self) -> nn.Parameter:
        return self.fuzzy_density_logits

    @property
    def synergy_eta(self) -> nn.Parameter:
        return self.choquet_lambda_raw

    @staticmethod
    def _sanitize_input_ifn_f64(
        mu_raw: torch.Tensor,
        nu_raw: torch.Tensor,
        pi_raw: Optional[torch.Tensor] = None,
        orig_input_dtype: Optional[torch.dtype] = None,
        work_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Maps arbitrary (possibly out-of-range, non-finite, or over-unit-sum) IFN inputs
        radially onto the closed intuitionistic simplex {mu, nu, pi >= 0, mu + nu + pi = 1}
        in working precision (float64, or float32 on MPS) up to IEEE-754 rounding, preserving
        feasible closed-boundary gradients and snapping any drifted pi entries to Sterbenz
        1 - max(mu, nu) - min(mu, nu) while preserving ULP-valid tiny hesitancy.
        """
        work_dt = work_dtype or _work_dtype_for(mu_raw.device)
        lp_dt = orig_input_dtype if orig_input_dtype is not None else (
            mu_raw.dtype if mu_raw.dtype.is_floating_point else None
        )
        mu_f = mu_raw.to(dtype=work_dt)
        nu_f = nu_raw.to(dtype=work_dt)
        mu_c = _clamp_preserve_boundary_grad(
            torch.nan_to_num(mu_f, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        )
        nu_c = _clamp_preserve_boundary_grad(
            torch.nan_to_num(nu_f, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        )
        s = mu_c + nu_c
        scale = torch.where(s > 1.0, s, torch.ones_like(s))
        mu_s = mu_c / scale
        nu_s = nu_c / scale
        pi_res = _clamp_preserve_boundary_grad(
            1.0 - torch.maximum(mu_s, nu_s) - torch.minimum(mu_s, nu_s), 0.0, 1.0
        )

        if pi_raw is not None:
            pi_f = pi_raw.to(dtype=work_dt)
            pi_c = _clamp_preserve_boundary_grad(
                torch.nan_to_num(pi_f, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0
            )
            in_dom = (
                torch.isfinite(mu_f)
                & torch.isfinite(nu_f)
                & torch.isfinite(pi_f)
                & (mu_f >= 0.0)
                & (mu_f <= 1.0)
                & (nu_f >= 0.0)
                & (nu_f <= 1.0)
                & (pi_f >= 0.0)
                & (pi_f <= 1.0)
            )
            if lp_dt is not None and lp_dt in (torch.float16, torch.bfloat16, torch.float32):
                in_eps = float(torch.finfo(lp_dt).eps)
                tot_f = mu_c + nu_c + pi_c
                tot_d = tot_f.detach()
                ulp_in_ok = (
                    in_dom
                    & (s <= 1.0 + 2.0 * in_eps)
                    & (tot_d > 0.0)
                    & (((tot_f).to(dtype=lp_dt) == 1.0) | (tot_d != 1.0))
                    & ((tot_d - 1.0).abs() <= 2.0 * in_eps)
                )
                if bool(ulp_in_ok.any()):
                    safe_tot = torch.where(ulp_in_ok, tot_f, torch.ones_like(tot_f))
                    mu_s = torch.where(ulp_in_ok, mu_c / safe_tot, mu_s)
                    nu_s = torch.where(ulp_in_ok, nu_c / safe_tot, nu_s)
                    pi_c_norm = torch.where(ulp_in_ok, pi_c / safe_tot, pi_c)
                    pi_res_norm = _clamp_preserve_boundary_grad(
                        1.0 - torch.maximum(mu_s, nu_s) - torch.minimum(mu_s, nu_s), 0.0, 1.0
                    )
                    pi_res = torch.where(
                        ulp_in_ok,
                        torch.where((mu_s + nu_s).detach() >= 0.999, pi_c_norm, pi_res_norm),
                        pi_res,
                    )
            else:
                ulp_in_ok = torch.zeros_like(in_dom, dtype=torch.bool)
            res_err = (mu_s + nu_s + pi_res - 1.0).abs()
            raw_err = (mu_s + nu_s + pi_c - 1.0).abs()
            in_eps_val = float(torch.finfo(lp_dt).eps) if (lp_dt is not None and lp_dt.is_floating_point) else 0.0
            valid_simplex = (
                in_dom
                & (~ulp_in_ok)
                & (s <= 1.0)
                & (
                    (((mu_s + nu_s + pi_c) == 1.0) & (raw_err <= res_err))
                    | (
                        (lp_dt is not None and lp_dt in (torch.float16, torch.bfloat16, torch.float32))
                        & ((mu_s + nu_s + pi_c).to(dtype=lp_dt or work_dt) == 1.0)
                        & (raw_err <= 2.0 * in_eps_val)
                    )
                )
            )
            pi_s = torch.where(ulp_in_ok, pi_res, torch.where(valid_simplex, pi_c, pi_res))
        else:
            pi_s = pi_res
        return mu_s, nu_s, pi_s

    def compute_pairwise_conflict(
        self, mu_stack: torch.Tensor, nu_stack: torch.Tensor, obs_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes observed-pair intuitionistic fuzzy conflict C_O in [0, 1] and the
        full pairwise conflict matrix C_mat of shape [B, M, M, C].
        """
        dev = mu_stack.device
        work_dt = _work_dtype_for(dev)
        with _disable_autocast_ctx(dev):
            orig_dtype = mu_stack.dtype
            if orig_dtype in _HALF_PREC_DTYPES or nu_stack.dtype in _HALF_PREC_DTYPES:
                cache: Dict[int, torch.Tensor] = {}
                fp16_cap = _fp16_cap_for_inputs((mu_stack, nu_stack, obs_mask))
                mu_w = _promote_low_prec_cached(mu_stack, work_dt, cache, fp16_max=fp16_cap)
                nu_w = _promote_low_prec_cached(nu_stack, work_dt, cache, fp16_max=fp16_cap)
                obs_w = _promote_low_prec_cached(obs_mask, work_dt, cache, fp16_max=fp16_cap)
                assert mu_w is not None and nu_w is not None and obs_w is not None
                avg_c, pair_c = self.compute_pairwise_conflict(mu_w, nu_w, obs_w)
                out_dt = orig_dtype if orig_dtype in _HALF_PREC_DTYPES else nu_stack.dtype
                return _bridge_work_via_lp(avg_c, out_dt)[1], _bridge_work_via_lp(pair_c, out_dt)[1]
            mu_d, nu_d, _ = self._sanitize_input_ifn_f64(mu_stack, nu_stack, work_dtype=work_dt)
            obs_d = torch.nan_to_num(
                obs_mask.to(dtype=work_dt), nan=0.0, posinf=1.0, neginf=0.0
            ).clamp(0.0, 1.0)

            mu_i = mu_d.unsqueeze(2)  # [B, M, 1, C]
            nu_i = nu_d.unsqueeze(2)
            mu_j = mu_d.unsqueeze(1)  # [B, 1, M, C]
            nu_j = nu_d.unsqueeze(1)

            pairwise_c = mu_i * nu_j + nu_i * mu_j  # [B, M, M, C]
            M = mu_stack.shape[1]
            if M <= 1:
                z = torch.zeros(
                    mu_stack.shape[0], mu_stack.shape[2], device=mu_stack.device, dtype=orig_dtype
                )
                return z, pairwise_c.to(dtype=orig_dtype)

            off_diag = (1.0 - torch.eye(M, device=mu_stack.device, dtype=work_dt)).view(1, M, M, 1)
            # Sort by canonical observed certainty order before summing so trailing vacuous zeros
            # are accumulated at the tail (IEEE-754 x + 0.0 == x invariance across float32/float64).
            score_d = (mu_d + nu_d).clamp(0.0, 1.0) * obs_d
            sort_idx = torch.argsort(score_d, dim=1, descending=True, stable=True)
            mu_s = torch.gather(mu_d, dim=1, index=sort_idx)
            nu_s = torch.gather(nu_d, dim=1, index=sort_idx)
            obs_s = torch.gather(obs_d, dim=1, index=sort_idx)
            pair_c_s = (mu_s.unsqueeze(2) * nu_s.unsqueeze(1) + nu_s.unsqueeze(2) * mu_s.unsqueeze(1)) * (
                obs_s.unsqueeze(2) * obs_s.unsqueeze(1) * off_diag
            )
            pair_obs_s = obs_s.unsqueeze(2) * obs_s.unsqueeze(1) * off_diag
            active_pairs = torch.cumsum(torch.cumsum(pair_obs_s, dim=1)[:, -1, :, :], dim=1)[:, -1, :]
            conflict_num = torch.cumsum(torch.cumsum(pair_c_s, dim=1)[:, -1, :, :], dim=1)[:, -1, :]
            avg_conflict = torch.where(
                active_pairs > 0.5,
                conflict_num / active_pairs.clamp(min=1.0),
                torch.zeros_like(conflict_num),
            )
            return avg_conflict.to(dtype=orig_dtype), (pairwise_c * off_diag).to(dtype=orig_dtype)

    def _fuse_tensors(
        self,
        mu_list: List[torch.Tensor],
        nu_list: List[torch.Tensor],
        pi_list: List[torch.Tensor],
    ) -> Tuple[IntuitionisticFuzzyTensor, Dict[str, torch.Tensor]]:
        dev = mu_list[0].device
        work_dt = _work_dtype_for(dev)
        is_f32_work = work_dt == torch.float32
        sub_thresh = 1e-34 if is_f32_work else 1e-300
        shift_lim = 60 if is_f32_work else 600
        sub_pow = 20 if is_f32_work else 100
        max_lt_one = (1.0 - 2.0**-24) if is_f32_work else (1.0 - 2.0**-53)
        min_pos = 1e-35 if is_f32_work else 1e-300
        tiny_lin = 1e-7 if is_f32_work else 1e-16

        with _disable_autocast_ctx(dev):
            orig_dtype = mu_list[0].dtype
            lp_cache: Dict[int, torch.Tensor] = {}
            any_input_lp = any(t.dtype in _HALF_PREC_DTYPES for lst in (mu_list, nu_list, pi_list) for t in lst)
            raw_input_dt = next(
                (t.dtype for lst in (mu_list, nu_list, pi_list) for t in lst if t.dtype in _HALF_PREC_DTYPES),
                None,
            )
            if any_input_lp:
                fp16_fuse_cap = _fp16_cap_for_inputs((mu_list, nu_list, pi_list))
                mu_work_list, nu_work_list, pi_work_list = [], [], []
                for m_t, n_t, p_t in zip(mu_list, nu_list, pi_list):
                    mw, nw, pw = _sanitize_half_prec_ifn_for_work(
                        m_t, n_t, p_t, work_dt, lp_cache, fp16_max=fp16_fuse_cap
                    )
                    assert pw is not None
                    mu_work_list.append(mw)
                    nu_work_list.append(nw)
                    pi_work_list.append(pw)
                mu_list, nu_list, pi_list = mu_work_list, nu_work_list, pi_work_list

            squeeze_last = mu_list[0].dim() == 1
            if squeeze_last:
                mu_list = [t.unsqueeze(-1) for t in mu_list]
                nu_list = [t.unsqueeze(-1) for t in nu_list]
                pi_list = [t.unsqueeze(-1) for t in pi_list]

            mu_raw = torch.stack(mu_list, dim=1)
            nu_raw = torch.stack(nu_list, dim=1)
            pi_raw = torch.stack(pi_list, dim=1)

            mu_stack, nu_stack, pi_input_stack = self._sanitize_input_ifn_f64(
                mu_raw,
                nu_raw,
                pi_raw,
                orig_input_dtype=(raw_input_dt or orig_dtype),
                work_dtype=work_dt,
            )
            pi_res_stack = _clamp_preserve_boundary_grad(
                1.0 - torch.maximum(mu_stack, nu_stack) - torch.minimum(mu_stack, nu_stack), 0.0, 1.0
            )
            pi_stack = torch.where((mu_stack + nu_stack).detach() >= 0.999, pi_input_stack, pi_res_stack)

            # Non-vacuity mass r_m = mu_m + nu_m = 1 - pi_m in [0, 1]; exact O = {m : mu_m > 0 or nu_m > 0}
            r_stack = _clamp_preserve_boundary_grad(mu_stack + nu_stack, 0.0, 1.0)  # [B, M, C]
            obs_mask = ((mu_stack > 0.0) | (nu_stack > 0.0)).to(dtype=work_dt)  # [B, M, C]

            # 1. Observed-pair conflict C_O (invariant to vacuous modalities <0, 0, 1>)
            avg_conflict_in, pairwise_c_in = self.compute_pairwise_conflict(mu_stack, nu_stack, obs_mask)
            avg_conflict = avg_conflict_in.to(dtype=work_dt)

            # 2. Non-Additive Monotone Choquet Capacity Increment Weights w_m
            C = mu_stack.shape[2]
            score_m = r_stack * obs_mask
            sort_idx = torch.argsort(score_m, dim=1, descending=True, stable=True)  # [B, M, C]
            obs_sorted = torch.gather(obs_mask, dim=1, index=sort_idx)

            if self.uniform_choquet:
                w_sum_obs = torch.cumsum(obs_sorted, dim=1)[:, -1:, :]
                has_any_obs = (w_sum_obs > 0.0).squeeze(1)
                w_sorted = torch.where(
                    w_sum_obs > 0.0,
                    obs_sorted / w_sum_obs.clamp(min=1.0),
                    torch.zeros_like(obs_sorted),
                )
                weights = torch.zeros_like(w_sorted)
                weights.scatter_(dim=1, index=sort_idx, src=w_sorted)
            else:
                logits_prm = _promote_low_prec_cached(self.fuzzy_density_logits, work_dt, lp_cache)
                assert logits_prm is not None
                logits = logits_prm.to(dtype=work_dt)
                if logits.shape[1] != C:
                    logits = logits[:, :1].expand(self.num_modalities, C)
                base_g = (F.softplus(logits) + 1e-3).unsqueeze(0).expand_as(r_stack)
                eta_prm = _promote_low_prec_cached(self.choquet_lambda_raw, work_dt, lp_cache)
                assert eta_prm is not None
                eta = eta_prm.to(dtype=work_dt)
                if eta.shape[1] != C:
                    eta = eta[:, :1].expand(1, C)
                p_exp_raw = torch.exp(torch.tanh(eta) * 0.35).unsqueeze(0).expand(mu_stack.shape[0], 1, C)

                # Scale-safe non-vacuity mass via dynamic power-of-two frexp/ldexp rescaling
                r_max_det = r_stack.detach().max(dim=1, keepdim=True).values
                sub_r = r_max_det < sub_thresh
                _, exp_r = torch.frexp(r_max_det)
                neg_exp = torch.where(r_max_det > 0.0, -exp_r, torch.zeros_like(exp_r))
                shift1 = (neg_exp // 2).clamp(-shift_lim, shift_lim)
                shift2 = (neg_exp - shift1).clamp(-shift_lim, shift_lim)
                scale1 = torch.ldexp(torch.ones_like(r_max_det), shift1)
                scale2 = torch.ldexp(torch.ones_like(r_max_det), shift2)
                sub_dn = 2.0 ** (-sub_pow)
                sub_up = 2.0 ** sub_pow
                scale2_bwd = torch.where(sub_r, scale2 * sub_dn, scale2)
                r_fwd = ((mu_stack * scale1) * scale2) + ((nu_stack * scale1) * scale2)
                r_bwd = ((mu_stack * scale1) * scale2_bwd) + ((nu_stack * scale1) * scale2_bwd)
                r_for_g = r_fwd.detach() + (r_bwd - r_bwd.detach())
                base_g_bwd = torch.where(sub_r, base_g * sub_dn, base_g)
                base_g_eff = base_g.detach() + (base_g_bwd - base_g_bwd.detach())
                p_exp_bwd = torch.where(sub_r, p_exp_raw * sub_dn, p_exp_raw)
                p_exp = p_exp_raw.detach() + (p_exp_bwd - p_exp_bwd.detach())
                unnorm_g = base_g_eff * r_for_g * obs_mask  # [B, M, C], identically 0 when missing
                unnorm_sorted = torch.gather(unnorm_g, dim=1, index=sort_idx)  # [B, M, C]
                sum_g = torch.cumsum(unnorm_sorted, dim=1)[:, -1:, :]  # [B, 1, C]
                has_any_obs = (sum_g > 0.0).squeeze(1)  # [B, C]
                safe_sum_g = torch.where(sum_g > 0.0, sum_g, torch.ones_like(sum_g))
                g_sorted = torch.where(
                    sum_g > 0.0,
                    unnorm_sorted / safe_sum_g,
                    torch.zeros_like(unnorm_sorted),
                )

                cum_g = _clamp_preserve_boundary_grad(torch.cumsum(g_sorted, dim=1), 0.0, 1.0)  # [B, M, C]
                cap_cur = torch.where(
                    cum_g > 0.0,
                    _clamp_preserve_boundary_grad(cum_g, min_val=min_pos).pow(p_exp),
                    torch.zeros_like(cum_g),
                )
                cap_prev = torch.cat([torch.zeros_like(cap_cur[:, :1, :]), cap_cur[:, :-1, :]], dim=1)
                w_sorted = _clamp_preserve_boundary_grad(cap_cur - cap_prev, min_val=0.0) * obs_sorted
                w_sum = torch.cumsum(w_sorted, dim=1)[:, -1:, :]
                safe_w_sum = torch.where(w_sum > 0.0, w_sum, torch.ones_like(w_sum))
                w_sorted = torch.where(
                    w_sum > 0.0,
                    w_sorted / safe_w_sum,
                    torch.zeros_like(w_sorted),
                )
                w_bwd_boost = torch.where(sub_r, w_sorted * sub_up, w_sorted)
                w_sorted = w_sorted.detach() + (w_bwd_boost - w_bwd_boost.detach())

                # Scatter Choquet capacity increments back to original modality positions
                weights = torch.zeros_like(w_sorted)
                weights.scatter_(dim=1, index=sort_idx, src=w_sorted)
                weights = weights * obs_mask  # Guarantee exact 0.0 weight for missing modalities

            # 3. Conflict-discounted IFN components: d = exp(-gamma * C_O)
            discount = torch.exp(-self.conflict_gamma * avg_conflict).unsqueeze(1)  # [B, 1, C]
            mu_hat_exact = _clamp_preserve_boundary_grad(mu_stack * discount, 0.0, 1.0)
            nu_hat_exact = _clamp_preserve_boundary_grad(nu_stack * discount, 0.0, 1.0)
            pi_hat_sterbenz = _clamp_preserve_boundary_grad(
                1.0 - torch.maximum(mu_hat_exact, nu_hat_exact) - torch.minimum(mu_hat_exact, nu_hat_exact),
                0.0,
                1.0,
            )
            pi_hat_disc = _clamp_preserve_boundary_grad(pi_stack * discount + (1.0 - discount), 0.0, 1.0)
            pi_hat_exact = torch.where(
                (mu_hat_exact + nu_hat_exact).detach() >= 0.999, pi_hat_disc, pi_hat_sterbenz
            )
            mu_hat = _clamp_preserve_boundary_grad(mu_hat_exact, 0.0, max_lt_one)
            nu_hat = _clamp_preserve_boundary_grad(nu_hat_exact, 0.0, max_lt_one)

            # 4. Exact IFWA aggregation in working-precision log-space in canonical sorted order
            mu_hat_sorted = torch.gather(mu_hat, dim=1, index=sort_idx)
            nu_hat_sorted = torch.gather(nu_hat, dim=1, index=sort_idx)
            mu_hat_exact_sorted = torch.gather(mu_hat_exact, dim=1, index=sort_idx)
            pi_hat_exact_sorted = torch.gather(pi_hat_exact, dim=1, index=sort_idx)
            one_minus_mu_hat = _clamp_preserve_boundary_grad(1.0 - mu_hat_sorted, min_val=min_pos)
            q_nu_sorted = _clamp_preserve_boundary_grad(nu_hat_sorted / one_minus_mu_hat, 0.0, max_lt_one)
            sterbenz_pi_hat_sorted = _clamp_preserve_boundary_grad(
                1.0 - torch.maximum(mu_hat_sorted, nu_hat_sorted) - torch.minimum(mu_hat_sorted, nu_hat_sorted),
                min_val=0.0,
            )
            eff_pi_hat_sorted = torch.where(
                (mu_hat_sorted + nu_hat_sorted).detach() >= 0.999,
                pi_hat_exact_sorted,
                sterbenz_pi_hat_sorted,
            )
            one_minus_q_sorted = _clamp_preserve_boundary_grad(
                eff_pi_hat_sorted / one_minus_mu_hat,
                min_val=min_pos,
                max_val=1.0,
            )

            log1p_neg_mu = torch.where(
                mu_hat_sorted < tiny_lin,
                -mu_hat_sorted,
                torch.log1p(-mu_hat_sorted),
            )
            log1p_neg_q = torch.where(
                q_nu_sorted <= 0.5,
                torch.where(
                    q_nu_sorted < tiny_lin,
                    -q_nu_sorted,
                    torch.log1p(-q_nu_sorted),
                ),
                torch.log(one_minus_q_sorted),
            )

            # Subnormal-safe weighted accumulation across modalities
            sub_mu = mu_hat_sorted.detach().max(dim=1, keepdim=True).values < sub_thresh
            mu_term_in = torch.where(sub_mu, log1p_neg_mu * (2.0**sub_pow), log1p_neg_mu)
            log_one_minus_mu_acc = torch.cumsum(w_sorted * mu_term_in, dim=1)[:, -1, :]
            log_one_minus_mu = torch.where(
                sub_mu.squeeze(1), log_one_minus_mu_acc * (2.0**(-sub_pow)), log_one_minus_mu_acc
            )

            neg_log1p_neg_q = -log1p_neg_q
            sub_q = neg_log1p_neg_q.detach().max(dim=1, keepdim=True).values < sub_thresh
            q_term_in = torch.where(sub_q, neg_log1p_neg_q * (2.0**sub_pow), neg_log1p_neg_q)
            delta_log_nu_acc = _clamp_preserve_boundary_grad(
                torch.cumsum(w_sorted * q_term_in, dim=1)[:, -1, :], min_val=0.0
            )
            delta_log_nu = torch.where(
                sub_q.squeeze(1), delta_log_nu_acc * (2.0**(-sub_pow)), delta_log_nu_acc
            )

            one_minus_mu_fused = _clamp_preserve_boundary_grad(torch.exp(log_one_minus_mu), 0.0, 1.0)
            mu_fused_raw = torch.where(
                log_one_minus_mu > -tiny_lin,
                _clamp_preserve_boundary_grad(-log_one_minus_mu, 0.0, 1.0),
                _clamp_preserve_boundary_grad(-torch.expm1(log_one_minus_mu), 0.0, 1.0),
            )
            nu_fused_small_q = _clamp_preserve_boundary_grad(
                one_minus_mu_fused
                * torch.where(
                    delta_log_nu < tiny_lin,
                    _clamp_preserve_boundary_grad(delta_log_nu, 0.0, 1.0),
                    _clamp_preserve_boundary_grad(-torch.expm1(-delta_log_nu), 0.0, 1.0),
                ),
                0.0,
                1.0,
            )
            pi_fused_large_q = _clamp_preserve_boundary_grad(one_minus_mu_fused * torch.exp(-delta_log_nu), 0.0, 1.0)
            nu_fused_large_q = _clamp_preserve_boundary_grad(one_minus_mu_fused - pi_fused_large_q, 0.0, 1.0)
            nu_fused_raw = torch.where(delta_log_nu <= 0.7, nu_fused_small_q, nu_fused_large_q)

            any_mu_one = ((w_sorted.detach() > 0.0) & (mu_hat_exact_sorted.detach() >= 1.0)).any(dim=1)
            any_pi_zero = ((w_sorted.detach() > 0.0) & (pi_hat_exact_sorted.detach() <= 0.0)).any(dim=1)
            mu_fused_multi = torch.where(any_mu_one, torch.ones_like(mu_fused_raw), mu_fused_raw)
            nu_fused_multi = torch.where(
                any_mu_one,
                torch.zeros_like(nu_fused_raw),
                torch.where(
                    any_pi_zero,
                    _clamp_preserve_boundary_grad(1.0 - mu_fused_multi, 0.0, 1.0),
                    nu_fused_raw,
                ),
            )
            pi_fused_sterbenz = _clamp_preserve_boundary_grad(
                1.0 - torch.maximum(mu_fused_multi, nu_fused_multi) - torch.minimum(mu_fused_multi, nu_fused_multi),
                0.0,
                1.0,
            )
            pi_fused_cand = torch.where(
                (mu_fused_multi + nu_fused_multi).detach() >= 0.999,
                pi_fused_large_q,
                pi_fused_sterbenz,
            )
            pi_fused_multi = torch.where(
                any_mu_one | any_pi_zero,
                torch.zeros_like(mu_fused_multi),
                pi_fused_cand,
            )

            single_obs = obs_mask.sum(dim=1) == 1.0
            mu_single = (mu_stack * obs_mask).sum(dim=1)
            nu_single = (nu_stack * obs_mask).sum(dim=1)
            pi_single_in = (pi_input_stack * obs_mask).sum(dim=1)
            pi_single_sterbenz = _clamp_preserve_boundary_grad(
                1.0 - torch.maximum(mu_single, nu_single) - torch.minimum(mu_single, nu_single),
                0.0,
                1.0,
            )
            pi_single_tan = pi_single_in.detach() + (pi_single_sterbenz - pi_single_sterbenz.detach())
            pi_single = torch.where((mu_single + nu_single).detach() >= 0.999, pi_single_in, pi_single_tan)

            mu_fused_w = torch.where(single_obs, mu_single, mu_fused_multi)
            nu_fused_w = torch.where(single_obs, nu_single, nu_fused_multi)
            pi_fused_w = torch.where(single_obs, pi_single, pi_fused_multi)

            if orig_dtype in _HALF_PREC_DTYPES:
                mu_w = torch.where(has_any_obs, mu_fused_w, torch.zeros_like(mu_fused_w)) + 0.0
                nu_w = torch.where(has_any_obs, nu_fused_w, torch.zeros_like(nu_fused_w)) + 0.0
                pi_w = torch.where(has_any_obs, pi_fused_w, torch.ones_like(mu_w)) + 0.0
                prob_w = _clamp_preserve_boundary_grad(mu_w + 0.5 * pi_w, 0.0, 1.0)
                conf_w = avg_conflict
                pair_w = pairwise_c_in.to(dtype=work_dt)
                wts_w = weights
                if squeeze_last:
                    mu_w = mu_w.squeeze(-1)
                    nu_w = nu_w.squeeze(-1)
                    pi_w = pi_w.squeeze(-1)
                    prob_w = prob_w.squeeze(-1)
                    conf_w = conf_w.squeeze(-1)
                    pair_w = pair_w.squeeze(-1)
                    wts_w = wts_w.squeeze(-1)
                _, mu_fused = _bridge_work_via_lp(mu_w, orig_dtype)
                _, nu_fused = _bridge_work_via_lp(nu_w, orig_dtype)
                _, pi_fused = _bridge_work_via_lp(pi_w, orig_dtype)
                _, prob = _bridge_work_via_lp(prob_w, orig_dtype)
                _, conflict_out = _bridge_work_via_lp(conf_w, orig_dtype)
                _, pairwise_out = _bridge_work_via_lp(pair_w, orig_dtype)
                _, weights_out = _bridge_work_via_lp(wts_w, orig_dtype)
            else:
                mu_fused = (torch.where(has_any_obs, mu_fused_w, torch.zeros_like(mu_fused_w)) + 0.0).to(dtype=orig_dtype)
                nu_fused = (torch.where(has_any_obs, nu_fused_w, torch.zeros_like(nu_fused_w)) + 0.0).to(dtype=orig_dtype)
                pi_fused = (
                    torch.where(
                        has_any_obs,
                        pi_fused_w.to(dtype=orig_dtype),
                        torch.ones_like(mu_fused),
                    )
                    + 0.0
                )
                weights_out = weights.to(dtype=orig_dtype)
                conflict_out = avg_conflict.to(dtype=orig_dtype)
                pairwise_out = pairwise_c_in.to(dtype=orig_dtype)

                prob = _clamp_preserve_boundary_grad(mu_fused + 0.5 * pi_fused, 0.0, 1.0)
                if squeeze_last:
                    mu_fused = mu_fused.squeeze(-1)
                    nu_fused = nu_fused.squeeze(-1)
                    pi_fused = pi_fused.squeeze(-1)
                    prob = prob.squeeze(-1)
                    conflict_out = conflict_out.squeeze(-1)
                    pairwise_out = pairwise_out.squeeze(-1)
                    weights_out = weights_out.squeeze(-1)

            fused_ifn = IntuitionisticFuzzyTensor(mu=mu_fused, nu=nu_fused, pi=pi_fused)
            aux = {
                "fused_ifn": fused_ifn,
                "mu_fused": mu_fused,
                "nu_fused": nu_fused,
                "pi_fused": pi_fused,
                "prob": prob,
                "conflict": conflict_out,
                "conflict_score": conflict_out,
                "pairwise_conflict": pairwise_out,
                "weights": weights_out,
                "choquet_weights": weights_out,
            }
            return fused_ifn, aux

    def forward(
        self,
        ifns_or_mu_list: List[Union[IntuitionisticFuzzyTensor, torch.Tensor]],
        nu_list: Optional[List[torch.Tensor]] = None,
        pi_list: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple[IntuitionisticFuzzyTensor, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Supports both calling conventions:
          1. `fused_ifn, aux = fusion([ifn1, ifn2, ...])`
          2. `out_dict = fusion(mu_list, nu_list, pi_list)`
        """
        if nu_list is None and pi_list is None:
            mus = [x.mu for x in ifns_or_mu_list]  # type: ignore[union-attr]
            nus = [x.nu for x in ifns_or_mu_list]  # type: ignore[union-attr]
            pis = [x.pi for x in ifns_or_mu_list]  # type: ignore[union-attr]
            return self._fuse_tensors(mus, nus, pis)
        else:
            assert nu_list is not None and pi_list is not None
            _, aux = self._fuse_tensors(
                ifns_or_mu_list, nu_list, pi_list  # type: ignore[arg-type]
            )
            return aux


class HesitantFuzzyAggregator(nn.Module):
    """
    Block 2: Torra Hesitant Fuzzy Set (HFS) Multi-Foundation-Model / Multi-Prompt
    Consensus Aggregator.

    Given P foundation models or prompt templates producing membership values
    H(x) = {h_1(x), ..., h_P(x)} in [0, 1]^P, computes:
      1. Hesitant Fuzzy Epistemic Disagreement (Xu & Xia, 2012; Liao & Xu, 2013):
            pi_HFS_raw(x) = min(1, 2 * sqrt(Var_p(h_p(x)))),
            pi_HFS(x)     = min(0.95, 2 * sqrt(Var_p(h_p(x))))  (capped at 0.95 on the
                            induced IFN simplex so support/refutation polarity is preserved).
      2. Median-Concordant Consensus Aggregation:
            c_p(x) = exp(-kappa * |h_p(x) - median(H(x))|),
            omega_p(x) = Softmax_p( alpha_p + beta_cert * |2 h_p(x) - 1| + log(c_p(x)) ),
            mu_HFS(x) = sum_{p=1}^P omega_p(x) h_p(x).
    """

    def __init__(
        self,
        num_sources: Optional[int] = None,
        concordance_kappa: float = 3.5,
        num_prompts: Optional[int] = None,
    ):
        super().__init__()
        n_src = num_sources if num_sources is not None else num_prompts
        if n_src is None:
            raise ValueError("Either num_sources or num_prompts must be specified.")
        self.num_sources = n_src
        self.num_prompts = n_src
        self.concordance_kappa = concordance_kappa
        self.prompt_reliability = nn.Parameter(torch.zeros(1, n_src))
        self.certainty_temp = nn.Parameter(torch.tensor(-1.5))

    def compute_hesitant_entropy(self, h_set: torch.Tensor) -> torch.Tensor:
        """
        Computes raw hesitant fuzzy epistemic dispersion in [0, 1] (note: `forward`
        additionally caps induced IFN hesitancy `ifn.pi` at 0.95):
            pi_HFS_raw(x) = min(1, 2 * sqrt(Var_p(h_p(x)))).
        """
        if not getattr(_HALF_PREC_TLS, "active", False) and (
            h_set.dtype in _HALF_PREC_DTYPES or _active_autocast_dtype(h_set.device) is not None
        ):
            work_dt = _work_dtype_for(h_set.device)
            with _disable_autocast_ctx(h_set.device):
                if h_set.dtype in _HALF_PREC_DTYPES:
                    fp16_cap = _fp16_cap_for_inputs(h_set)
                    h_w = _promote_low_prec_cached(h_set, work_dt, {}, fp16_max=fp16_cap)
                    assert h_w is not None
                    return _bridge_work_via_lp(self.compute_hesitant_entropy(h_w), h_set.dtype)[1]
                h_w = h_set.to(dtype=work_dt)
                return self.compute_hesitant_entropy(h_w).to(dtype=h_set.dtype)
        with _disable_autocast_ctx(h_set.device):
            h = torch.nan_to_num(h_set, nan=0.5, posinf=0.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)
            std_h = torch.sqrt(h.var(dim=1, unbiased=False) + 1e-12)
            return (2.0 * std_h).clamp(0.0, 1.0)

    def forward(self, h_set: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_set: Membership predictions from P experts/models [B, P] or [B, P, L] in [0, 1].
        """
        if not getattr(_HALF_PREC_TLS, "active", False):
            lp_dt, dev = _collect_lp_info(h_set, self)
            if lp_dt is not None:
                return _run_module_in_work_prec(self, lp_dt, dev, (h_set,))
        is_2d = h_set.dim() == 2
        if is_2d:
            h_clean = (
                torch.nan_to_num(h_set, nan=0.5, posinf=0.0, neginf=0.0)
                .clamp(1e-6, 1.0 - 1e-6)
                .unsqueeze(-1)
            )  # [B, P, 1]
        else:
            h_clean = torch.nan_to_num(h_set, nan=0.5, posinf=0.0, neginf=0.0).clamp(
                1e-6, 1.0 - 1e-6
            )  # [B, P, L]

        std_h = torch.sqrt(h_clean.var(dim=1, unbiased=False) + 1e-12)  # [B, L]
        h_entropy = (2.0 * std_h).clamp(0.0, 1.0)  # [B, L]
        h_var = h_clean.var(dim=1, unbiased=False)  # [B, L]

        certainty = (2.0 * h_clean - 1.0).abs()  # [B, P, L]
        median_h = h_clean.median(dim=1, keepdim=True).values  # [B, 1, L]
        log_concordance = -self.concordance_kappa * (h_clean - median_h).abs()  # [B, P, L]

        rel = self.prompt_reliability.unsqueeze(-1)  # [1, P, 1]
        logits = rel + F.softplus(self.certainty_temp) * certainty + log_concordance
        weights = F.softmax(logits, dim=1)  # [B, P, L]

        mu_hfs = (weights * h_clean).sum(dim=1)  # [B, L]
        pi_hfs = h_entropy.clamp(0.0, 0.95)
        rem = 1.0 - pi_hfs
        mu_out = mu_hfs * rem
        nu_out = (1.0 - mu_hfs) * rem
        ifn = IntuitionisticFuzzyTensor(mu=mu_out, nu=nu_out, pi=pi_hfs)

        return {
            "consensus_mu": mu_hfs,
            "consensus_prob": mu_hfs,
            "hfs_entropy": h_entropy,
            "hesitant_entropy": h_entropy,
            "hesitant_variance": h_var,
            "prompt_weights": weights.squeeze(-1) if is_2d else weights,
            "mu": mu_out,
            "nu": nu_out,
            "pi": pi_hfs,
            "ifn": ifn,
        }


class DifferentiableFuzzyRoughLayer(nn.Module):
    """
    Block 3A: Differentiable Projected Soft-Lukasiewicz Fuzzy-Rough Approximation Layer
    for Explicit Target Concept Membership A in [0, 1]^{B x N x C}.

    Guarantees Theorem 5:
        0 <= R_low(A) <= A <= R_up(A) <= 1   and   BND_R(A) = R_up(A) - R_low(A) >= 0.
    """

    def __init__(
        self,
        feat_dim: int,
        proj_dim: int = 16,
        soft_temp: float = 0.1,
        smooth_beta: float = 0.5,
    ):
        super().__init__()
        if soft_temp <= 0.0:
            raise ValueError(f"soft_temp must be strictly positive (> 0), got {soft_temp}")
        if not (0.0 <= smooth_beta <= 1.0):
            raise ValueError(f"smooth_beta must lie in [0, 1], got {smooth_beta}")
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim
        self.soft_temp = float(soft_temp)
        self.smooth_beta = float(smooth_beta)
        self.proj = nn.Linear(feat_dim, proj_dim)
        self.bandwidth = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        tokens: torch.Tensor,
        soft_labels: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: [B, N, D]
            soft_labels: [B, N] or [B, N, C] in [0, 1]
            token_mask: Optional binary mask [B, N]
        """
        if not getattr(_HALF_PREC_TLS, "active", False):
            lp_dt, dev = _collect_lp_info((tokens, soft_labels, token_mask), self)
            if lp_dt is not None:
                return _run_module_in_work_prec(self, lp_dt, dev, (tokens, soft_labels, token_mask))
        x_clean = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        squeeze_c = soft_labels.dim() == 2
        soft_labels_3d = soft_labels.unsqueeze(-1) if squeeze_c else soft_labels
        A = torch.nan_to_num(soft_labels_3d, nan=0.5, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        B, N, _ = x_clean.shape

        if token_mask is not None:
            t_mask = (token_mask.to(device=x_clean.device, dtype=x_clean.dtype) > 0.5).to(dtype=x_clean.dtype)
            x_clean = x_clean * t_mask.unsqueeze(-1)
        else:
            t_mask = torch.ones(B, N, device=x_clean.device, dtype=x_clean.dtype)

        z = self.proj(x_clean)  # [B, N, P]
        diff = (z.unsqueeze(2) - z.unsqueeze(1)).abs().mean(dim=-1)  # [B, N, N]
        sigma = F.softplus(self.bandwidth) + 0.1
        R = torch.exp(-diff / sigma)  # [B, N, N] in (0, 1], with R_{i,i} = 1

        eye = torch.eye(N, device=x_clean.device, dtype=x_clean.dtype).unsqueeze(0)
        pair_valid = (t_mask.unsqueeze(2) * t_mask.unsqueeze(1)).clamp(0.0, 1.0)
        R = R * torch.maximum(pair_valid, eye)

        # Broadcast over C concept channels: R_exp [B, N, N, 1], A_j [B, 1, N, C]
        R_exp = R.unsqueeze(-1)
        A_j = A.unsqueeze(1).expand(B, N, N, A.shape[-1])
        implication = torch.clamp(1.0 - R_exp + A_j, min=0.0, max=1.0)
        tnorm = torch.clamp(R_exp + A_j - 1.0, min=0.0, max=1.0)

        valid_j = torch.maximum(t_mask.unsqueeze(1).expand(B, N, N), eye).unsqueeze(-1)
        imp_masked = torch.where(valid_j > 0.5, implication, torch.ones_like(implication))
        tnorm_masked = torch.where(valid_j > 0.5, tnorm, torch.zeros_like(tnorm))

        hard_min = imp_masked.min(dim=2).values  # [B, N, C]
        hard_max = tnorm_masked.max(dim=2).values  # [B, N, C]

        tau_safe = max(float(self.soft_temp), 1e-30)
        valid_mask = valid_j > 0.5
        if tau_safe >= 1e-4 and x_clean.dtype not in (torch.float16, torch.bfloat16):
            neg_inf = torch.tensor(-torch.inf, device=x_clean.device, dtype=x_clean.dtype)
            logits_low = torch.where(valid_mask, -implication / tau_safe, neg_inf)
            logits_up = torch.where(valid_mask, tnorm / tau_safe, neg_inf)
            w_low = F.softmax(logits_low, dim=2)
            w_up = F.softmax(logits_up, dim=2)
            soft_low = (w_low * implication).sum(dim=2)
            soft_up = (w_up * tnorm).sum(dim=2)
        else:
            work_dt = _work_dtype_for(x_clean.device)
            hard_min_w = hard_min.to(dtype=work_dt)
            hard_max_w = hard_max.to(dtype=work_dt)
            delta_low_w = torch.where(valid_mask, implication.to(dtype=work_dt) - hard_min_w.unsqueeze(2), 0.0)
            delta_up_w = torch.where(valid_mask, tnorm.to(dtype=work_dt) - hard_max_w.unsqueeze(2), 0.0)
            neg_inf_w = torch.tensor(-torch.inf, device=x_clean.device, dtype=work_dt)
            logits_low_w = torch.where(valid_mask, -delta_low_w / tau_safe, neg_inf_w)
            logits_up_w = torch.where(valid_mask, delta_up_w / tau_safe, neg_inf_w)
            w_low_w = torch.where(valid_mask, F.softmax(logits_low_w, dim=2), 0.0)
            w_up_w = torch.where(valid_mask, F.softmax(logits_up_w, dim=2), 0.0)
            soft_low = (hard_min_w + (w_low_w * delta_low_w).sum(dim=2)).to(dtype=x_clean.dtype)
            soft_up = (hard_max_w + (w_up_w * delta_up_w).sum(dim=2)).to(dtype=x_clean.dtype)

        blend_low = (1.0 - self.smooth_beta) * hard_min + self.smooth_beta * soft_low
        blend_up = (1.0 - self.smooth_beta) * hard_max + self.smooth_beta * soft_up

        lower_approx = torch.minimum(A, blend_low).clamp(0.0, 1.0)
        upper_approx = torch.maximum(A, blend_up).clamp(0.0, 1.0)
        boundary = (upper_approx - lower_approx).clamp(min=0.0)
        dependency_gamma = lower_approx.mean(dim=(1, 2))
        if squeeze_c:
            lower_approx = lower_approx.squeeze(-1)
            upper_approx = upper_approx.squeeze(-1)
            boundary = boundary.squeeze(-1)

        return {
            "lower_approx": lower_approx,
            "upper_approx": upper_approx,
            "boundary_region": boundary,
            "dependency_gamma": dependency_gamma,
            "similarity_matrix": R,
        }


class FuzzyRoughAttentionGate(nn.Module):
    """
    Block 3B: Differentiable Projected Soft-Lukasiewicz Fuzzy-Rough Token Gate.

    Computes the fuzzy indiscernibility similarity relation R_B(x_i, x_j) and the
    projected Soft-Lukasiewicz lower and upper approximations:
        I_{i,j} = min(1, 1 - R_{i,j} + A_j),
        T_{i,j} = max(0, R_{i,j} + A_j - 1),
        R_low(A)_i = min(A_i, (1 - beta) min_j I_{i,j} + beta sum_j omega_{i,j}^- I_{i,j}),
        R_up(A)_i  = max(A_i, (1 - beta) max_j T_{i,j} + beta sum_j omega_{i,j}^+ T_{i,j}).
    By construction (Theorem 5), 0 <= R_low(A)_i <= A_i <= R_up(A)_i <= 1 holds
    identically for all beta in [0, 1] and temperature tau > 0.
    """

    def __init__(
        self,
        embed_dim: int,
        soft_temp: float = 0.15,
        smooth_beta: float = 0.5,
        num_granules: int = 5,
    ):
        super().__init__()
        if soft_temp <= 0.0:
            raise ValueError(f"soft_temp must be strictly positive (> 0), got {soft_temp}")
        if not (0.0 <= smooth_beta <= 1.0):
            raise ValueError(f"smooth_beta must lie in [0, 1], got {smooth_beta}")
        self.embed_dim = embed_dim
        self.soft_temp = float(soft_temp)
        self.num_granules = max(3, num_granules)
        self.beta_raw = nn.Parameter(torch.logit(torch.tensor(float(smooth_beta)).clamp(1e-3, 1.0 - 1e-3)))
        self.feature_weights = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.bandwidth = nn.Parameter(torch.tensor(0.0))
        self.target_membership_head = nn.Linear(embed_dim, self.num_granules)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward_dict(
        self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if not getattr(_HALF_PREC_TLS, "active", False):
            lp_dt, dev = _collect_lp_info((x, token_mask), self)
            if lp_dt is not None:
                return _run_module_in_work_prec(
                    self, lp_dt, dev, (x,), {"token_mask": token_mask}, gate_return_dict=True
                )
        x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        B, N, _ = x_clean.shape

        if token_mask is not None:
            t_mask = (token_mask.to(device=x_clean.device, dtype=x_clean.dtype) > 0.5).to(dtype=x_clean.dtype)
            x_clean = x_clean * t_mask.unsqueeze(-1)
        else:
            t_mask = torch.ones(B, N, device=x_clean.device, dtype=x_clean.dtype)

        w = F.softplus(self.feature_weights)
        x_weighted = x_clean * w  # [B, N, D]

        # Per-token internal granular memberships a_{b,n,g} in (0, 1), g = 0..G-1
        # Operating strictly within each token preserves exact unimodal invariance when other tokens are masked
        granule_probs = torch.sigmoid(self.target_membership_head(x_weighted))  # [B, N, G]
        A = granule_probs[:, :, 0] * t_mask  # Anchor concept membership A_{b,n} in [0, 1)
        A_exp = A.unsqueeze(-1)  # [B, N, 1]

        gamma_fr = 0.25 * torch.sigmoid(self.bandwidth) + 0.05  # in (0.05, 0.30) < 1
        diff_g = (granule_probs - A_exp).abs()  # [B, N, G], with diff_g[:, :, 0] == 0
        R = torch.exp(-gamma_fr * diff_g)       # [B, N, G] in (0, 1], with R[:, :, 0] == 1

        implication = torch.clamp(1.0 - R + granule_probs, min=0.0, max=1.0)  # I_{b,n,g}
        tnorm = torch.clamp(R + granule_probs - 1.0, min=0.0, max=1.0)        # T_{b,n,g}

        hard_min = implication.min(dim=-1).values
        hard_max = tnorm.max(dim=-1).values

        tau_safe = max(float(self.soft_temp), 1e-30)
        if tau_safe >= 1e-4 and x_clean.dtype not in (torch.float16, torch.bfloat16):
            w_low = F.softmax(-implication / tau_safe, dim=-1)
            w_up = F.softmax(tnorm / tau_safe, dim=-1)
            soft_low = (w_low * implication).sum(dim=-1)
            soft_up = (w_up * tnorm).sum(dim=-1)
        else:
            work_dt = _work_dtype_for(x_clean.device)
            hard_min_w = hard_min.to(dtype=work_dt)
            hard_max_w = hard_max.to(dtype=work_dt)
            delta_low_w = implication.to(dtype=work_dt) - hard_min_w.unsqueeze(-1)
            delta_up_w = tnorm.to(dtype=work_dt) - hard_max_w.unsqueeze(-1)
            w_low_w = F.softmax(-delta_low_w / tau_safe, dim=-1)
            w_up_w = F.softmax(delta_up_w / tau_safe, dim=-1)
            soft_low = (hard_min_w + (w_low_w * delta_low_w).sum(dim=-1)).to(dtype=x_clean.dtype)
            soft_up = (hard_max_w + (w_up_w * delta_up_w).sum(dim=-1)).to(dtype=x_clean.dtype)

        beta = torch.sigmoid(self.beta_raw)
        blend_low = (1.0 - beta) * hard_min + beta * soft_low
        blend_up = (1.0 - beta) * hard_max + beta * soft_up

        # Exact Theorem 5 projection: 0 <= R_low(A) <= A <= R_up(A) <= 1
        lower_approx = (torch.minimum(A, blend_low).clamp(0.0, 1.0)) * t_mask
        upper_approx = (torch.maximum(A, blend_up).clamp(0.0, 1.0)) * t_mask
        boundary = (upper_approx - lower_approx).clamp(min=0.0) * t_mask

        pos_weight = lower_approx.unsqueeze(-1)
        bnd_weight = boundary.unsqueeze(-1)
        refined = x_clean * (1.0 + 0.25 * pos_weight - 0.15 * bnd_weight) + 0.10 * self.out_proj(x_clean) * bnd_weight
        refined = refined * t_mask.unsqueeze(-1)

        return {
            "refined_features": refined,
            "target_membership": A,
            "lower_approx": lower_approx,
            "upper_approx": upper_approx,
            "boundary_region": boundary,
            "similarity_matrix": R,
        }

    def forward(
        self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        if getattr(_HALF_PREC_TLS, "active", False) and getattr(_HALF_PREC_TLS, "gate_return_dict", False):
            return self.forward_dict(x, token_mask=token_mask)
        if not getattr(_HALF_PREC_TLS, "active", False):
            lp_dt, dev = _collect_lp_info((x, token_mask), self)
            if lp_dt is not None:
                return _run_module_in_work_prec(self, lp_dt, dev, (x,), {"token_mask": token_mask})
        out = self.forward_dict(x, token_mask=token_mask)
        return out["refined_features"], out["boundary_region"].unsqueeze(-1)


class DeepConceptTSKHead(nn.Module):
    """
    Block 4: White-Box Concept-Bottleneck Deep Takagi-Sugeno-Kang (TSK) Fuzzy Rule Head.

    Evaluates R interpretable IF-THEN rules over C measured concept dimensions z in R^C
    with optional binary observation mask m_z in {0, 1}^C (C_obs = max(1, sum_c m_{z,c})):
        phi_r(z, m_z) = exp( -1/(2 C_obs) * sum_{c=1}^C m_{z,c} g_{r,c} ((z_c - c_{r,c}) / sigma_{r,c})^2 ),
        phi_bar_r(z, m_z) = phi_r(z, m_z) / sum_{j=1}^R phi_j(z, m_z),
        y_{r,k}(z, m_z) = w_{r,0,k} + sum_{c=1}^C m_{z,c} g_{r,c} w_{r,c,k} z_c,
        f_{TSK,k}(z, m_z) = sum_{r=1}^R phi_bar_r(z, m_z) y_{r,k}(z, m_z).

    Supports complete structured/text rule parameter export across both binary (out_dim == 1)
    and multi-class (out_dim > 1) heads, with exact floating-point prediction reconstruction
    from in-memory tensors and empirical O(max_r ||w_r||_1 * 1e-8) reconstruction discrepancy
    from 8-decimal text rules on standardized bounded concepts with widths >= 0.20, rather
    than a uniform bound over unbounded R^C. Supports missing-concept invariance
    (extending Theorem 2 end-to-end: exact `0.00` invariance across arbitrary
    missing-input fill values at fixed tensor width, and algebraic equivalence in real arithmetic
    to the reduced observed-concept submodel within IEEE-754 float32 reduction-order rounding).
    """

    def __init__(
        self,
        num_concepts: int = 8,
        num_rules: int = 8,
        out_dim: int = 1,
        num_classes: Optional[int] = None,
        concept_names: Optional[List[str]] = None,
    ):
        super().__init__()
        k_out = num_classes if num_classes is not None else out_dim
        self.num_concepts = num_concepts
        self.num_rules = num_rules
        self.out_dim = k_out
        self.num_classes = k_out
        raw_names = list(concept_names) if concept_names is not None else [f"Concept_{i+1}" for i in range(num_concepts)]
        if len(raw_names) < num_concepts:
            raw_names = raw_names + [f"Concept_{i+1}" for i in range(len(raw_names), num_concepts)]
        self.concept_names = raw_names[:num_concepts]

        self.centers = nn.Parameter(torch.randn(num_rules, num_concepts) * 0.5)
        self.log_widths = nn.Parameter(torch.zeros(num_rules, num_concepts))
        self.antecedent_gates = nn.Parameter(torch.ones(num_rules, num_concepts) * 1.0)
        self.consequent_weights = nn.Parameter(torch.randn(num_rules, num_concepts, k_out) * 0.1)
        self.consequent_bias = nn.Parameter(torch.zeros(num_rules, k_out))

    def forward(
        self,
        z_concepts: torch.Tensor,
        concept_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            z_concepts: Measured concept vector [B, C]
            concept_mask: Optional binary observation indicator [B, C] (1 = observed, 0 = missing)
        Returns:
            logits: [B, out_dim]
            aux_dict: Dict with logits, rule_firing, rule_outputs, gate_sparsity_l1, effective_centers
        """
        if not getattr(_HALF_PREC_TLS, "active", False):
            lp_dt, dev = _collect_lp_info((z_concepts, concept_mask), self)
            if lp_dt is not None:
                return _run_module_in_work_prec(self, lp_dt, dev, (z_concepts, concept_mask))
        z_clean = torch.nan_to_num(z_concepts, nan=0.0, posinf=0.0, neginf=0.0)
        c = self.centers.unsqueeze(0)    # [1, R, C]
        sigma = F.softplus(self.log_widths).unsqueeze(0) + 0.2  # [1, R, C]
        gates = torch.sigmoid(self.antecedent_gates).unsqueeze(0)  # [1, R, C]

        if concept_mask is not None:
            m_bin = (
                torch.nan_to_num(
                    concept_mask.to(device=z_clean.device, dtype=z_clean.dtype),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                > 0.5
            ).to(dtype=z_clean.dtype)
            z_eff = z_clean * m_bin
            m_exp = m_bin.unsqueeze(1)  # [B, 1, C]
            dist_sq = gates * (((z_eff.unsqueeze(1) - c) / sigma).pow(2)) * m_exp  # [B, R, C]
            active_c = m_exp.sum(dim=-1).clamp(min=1.0)  # [B, 1]
            gated_dist = dist_sq.sum(dim=-1) / active_c  # [B, R], HTSK 1/C_obs scaling
        else:
            z_eff = z_clean
            dist_sq = gates * (((z_eff.unsqueeze(1) - c) / sigma).pow(2))  # [B, R, C]
            gated_dist = dist_sq.mean(dim=-1)  # [B, R], HTSK 1/C scaling

        norm_firing = F.softmax(-0.5 * gated_dist, dim=-1)  # [B, R], exact partition of unity

        # Gated consequent weights: g_{r,c} * w_{r,c} (matches Eq. 13 and exported rules)
        eff_consequent = self.consequent_weights * torch.sigmoid(self.antecedent_gates).unsqueeze(-1)  # [R, C, K]
        rule_outputs = torch.einsum("bc,rck->brk", z_eff, eff_consequent) + self.consequent_bias.unsqueeze(0)
        logits = torch.einsum("br,brk->bk", norm_firing, rule_outputs)  # [B, K]

        gate_l1 = torch.sigmoid(self.antecedent_gates).mean()
        aux = {
            "logits": logits,
            "rule_firing": norm_firing,
            "rule_outputs": rule_outputs,
            "gate_sparsity_l1": gate_l1,
            "effective_centers": (torch.sigmoid(self.antecedent_gates) * self.centers).detach(),
        }
        return logits, aux

    @torch.no_grad()
    def export_structured_rules(
        self, concept_names: Optional[List[str]] = None
    ) -> Dict[str, object]:
        """
        Exports complete structured rule parameters (centers, sigmas, gates, consequent_bias,
        effective_consequent_weights, distance_scale, out_dim, and per-rule dictionaries) capable
        of reconstructing forward() TSK head logit predictions within float32 precision across
        both out_dim == 1 and out_dim > 1.
        """
        names = concept_names or self.concept_names
        if len(names) < self.num_concepts:
            names = list(names) + [f"Concept_{i+1}" for i in range(len(names), self.num_concepts)]
        names = names[: self.num_concepts]

        c_np = self.centers.detach().cpu().numpy()  # [R, C]
        s_np = (F.softplus(self.log_widths) + 0.2).detach().cpu().numpy()  # [R, C]
        g_np = torch.sigmoid(self.antecedent_gates).detach().cpu().numpy()  # [R, C]
        eff_w_np = (
            self.consequent_weights * torch.sigmoid(self.antecedent_gates).unsqueeze(-1)
        ).detach().cpu().numpy()  # [R, C, K]
        b_np = self.consequent_bias.detach().cpu().numpy()  # [R, K]

        rule_list = []
        for r in range(self.num_rules):
            antecedents = []
            consequents: Dict[str, Union[float, List[float]]] = {}
            for idx, name in enumerate(names):
                center_val = float(c_np[r, idx])
                width_val = float(s_np[r, idx])
                gate_val = float(g_np[r, idx])
                if self.out_dim == 1:
                    weight_val: Union[float, List[float]] = float(eff_w_np[r, idx, 0])
                else:
                    weight_val = [float(eff_w_np[r, idx, k]) for k in range(self.out_dim)]
                if center_val < -0.25:
                    ling = "LOW"
                elif center_val > 0.25:
                    ling = "HIGH"
                else:
                    ling = "MEDIUM"
                antecedents.append({
                    "concept": name,
                    "linguistic_term": ling,
                    "center": center_val,
                    "width": width_val,
                    "gate": gate_val,
                })
                consequents[name] = weight_val
            bias_entry: Union[float, List[float]] = (
                float(b_np[r, 0])
                if self.out_dim == 1
                else [float(b_np[r, k]) for k in range(self.out_dim)]
            )
            rule_list.append({
                "rule_index": r + 1,
                "bias": bias_entry,
                "antecedents": antecedents,
                "consequents": consequents,
            })

        return {
            "concept_names": names,
            "out_dim": int(self.out_dim),
            "centers": c_np,
            "sigmas": s_np,
            "gates": g_np,
            "consequent_bias": b_np,
            "consequent_weights": self.consequent_weights.detach().cpu().numpy(),
            "effective_consequent_weights": eff_w_np,
            "distance_scale": 1.0 / float(self.num_concepts),
            "rules": rule_list,
        }

    @torch.no_grad()
    def export_clinical_rules(
        self,
        concept_names: Optional[List[str]] = None,
        gate_threshold: float = 0.0,
        precision: int = 8,
    ) -> List[str]:
        """
        Exports human-readable IF-THEN fuzzy rules including all active concepts
        (default gate_threshold=0.0 exports all concepts without truncation;
        default precision=8 formats parameters to 8 decimal places for high-precision
        text reconstruction within ~1e-7 of the float32 TSK head logit).
        Supports both scalar (`out_dim == 1`) and multi-class (`out_dim > 1`) consequents.
        """
        structured = self.export_structured_rules(concept_names)
        k_out = int(structured["out_dim"])  # type: ignore[arg-type]
        rules_str = []
        for r_dict in structured["rules"]:  # type: ignore[index]
            ant_parts = []
            for ant in r_dict["antecedents"]:
                if ant["gate"] >= gate_threshold:
                    ant_parts.append(
                        f"{ant['concept']} is {ant['linguistic_term']} "
                        f"(c={ant['center']:+.{precision}f}, s={ant['width']:.{precision}f}, g={ant['gate']:.{precision}f})"
                    )
            ant_str = " AND ".join(ant_parts) if ant_parts else "TRUE"

            if k_out == 1:
                cons_parts = [f"{float(r_dict['bias']):+.{precision}f}"]
                for ant in r_dict["antecedents"]:
                    w_val = float(r_dict["consequents"][ant["concept"]])
                    cons_parts.append(f"{w_val:+.{precision}f}*{ant['concept']}")
                cons_str = " ".join(cons_parts)
                rules_str.append(f"Rule {r_dict['rule_index']}: IF {ant_str} THEN Logit = {cons_str}")
            else:
                class_clauses = []
                bias_list = r_dict["bias"]
                for k in range(k_out):
                    cons_parts_k = [f"{float(bias_list[k]):+.{precision}f}"]
                    for ant in r_dict["antecedents"]:
                        w_vec = r_dict["consequents"][ant["concept"]]
                        cons_parts_k.append(f"{float(w_vec[k]):+.{precision}f}*{ant['concept']}")
                    class_clauses.append(f"Logit[{k}] = " + " ".join(cons_parts_k))
                rules_str.append(
                    f"Rule {r_dict['rule_index']}: IF {ant_str} THEN " + " ; ".join(class_clauses)
                )
        return rules_str

    @staticmethod
    def evaluate_exported_rules_text(
        rules_str: List[str],
        z_concepts: Union[np.ndarray, torch.Tensor],
        concept_names: Optional[List[str]] = None,
        concept_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Parses the human-readable rule strings exported by `export_clinical_rules`
        (when `gate_threshold=0.0`) directly via regular expressions and reconstructs
        the TSK head logit predictions on `z_concepts` [B, C].
        Returns shape `(B,)` when `out_dim == 1` and `(B, out_dim)` when `out_dim > 1`.
        """
        if isinstance(z_concepts, torch.Tensor):
            z_np = z_concepts.detach().cpu().numpy().astype(np.float64)
        else:
            z_np = np.asarray(z_concepts, dtype=np.float64)
        z_np = np.nan_to_num(z_np, nan=0.0, posinf=0.0, neginf=0.0)
        B, C = z_np.shape
        R = len(rules_str)
        if R == 0:
            raise ValueError("rules_str must contain at least one exported TSK rule.")

        c_mat = np.zeros((R, C), dtype=np.float64)
        s_mat = np.ones((R, C), dtype=np.float64)
        g_mat = np.zeros((R, C), dtype=np.float64)

        num_pat = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
        ant_pattern = re.compile(
            rf"(?:^Rule\s+\d+:\s+IF\s+|\s+AND\s+)(.+?)\s+is\s+(?:LOW|MEDIUM|HIGH)\s+"
            rf"\(c=({num_pat}),\s*s=({num_pat}),\s*g=({num_pat})\)"
        )
        bias_pattern = re.compile(rf"^\s*({num_pat})(?:\s+|$)")
        cons_term_pattern = re.compile(rf"({num_pat})\*(.+?)(?=\s+{num_pat}\*|$)")
        multi_clause_pattern = re.compile(r"Logit\[(\d+)\]\s*=\s*(.+?)(?=\s*;\s*Logit\[\d+\]\s*=|$)")

        is_multiclass = " THEN Logit[" in rules_str[0]
        if is_multiclass:
            first_clauses = multi_clause_pattern.findall(rules_str[0].split(" THEN ", 1)[1])
            K = len(first_clauses)
            if K < 1:
                raise ValueError(f"Cannot parse multi-class TSK consequents in: {rules_str[0]}")
        else:
            K = 1

        w_mat = np.zeros((R, C, K), dtype=np.float64)
        b_mat = np.zeros((R, K), dtype=np.float64)

        inferred_names: List[str] = list(concept_names) if concept_names is not None else []
        for r_idx, line in enumerate(rules_str):
            if is_multiclass:
                if " THEN " not in line:
                    raise ValueError(f"Cannot parse TSK rule string: {line}")
                if_part, then_part = line.split(" THEN ", 1)
                clauses = multi_clause_pattern.findall(then_part.strip())
                if len(clauses) != K:
                    raise ValueError(
                        f"Inconsistent consequent class count in rule {r_idx + 1}: got {len(clauses)}, expected {K}."
                    )
                parsed_clauses = [(int(k_str), expr.strip()) for k_str, expr in clauses]
            else:
                if " THEN Logit = " not in line:
                    raise ValueError(f"Cannot parse TSK rule string: {line}")
                if_part, then_part = line.split(" THEN Logit = ", 1)
                parsed_clauses = [(0, then_part.strip())]

            ant_matches = [(m[0].strip(), m[1], m[2], m[3]) for m in ant_pattern.findall(if_part.strip())]

            for k_idx, then_clean in parsed_clauses:
                if not (0 <= k_idx < K):
                    raise ValueError(f"Class index {k_idx} out of bounds for K={K} in: {line}")
                bias_match = bias_pattern.match(then_clean)
                if bias_match is None:
                    raise ValueError(f"Cannot parse consequent bias in TSK rule string: {line}")
                b_mat[r_idx, k_idx] = float(bias_match.group(1))
                then_rest = then_clean[bias_match.end() :]
                cons_matches = [(m[0], m[1].strip()) for m in cons_term_pattern.findall(then_rest)]

                if not inferred_names:
                    ant_name_set = {m[0] for m in ant_matches}
                    if cons_matches:
                        inferred_names = [
                            (
                                nm[:-1]
                                if idx_m == len(cons_matches) - 1
                                and nm.endswith(".")
                                and nm not in ant_name_set
                                and (not ant_name_set or nm[:-1] in ant_name_set)
                                else nm
                            )
                            for idx_m, (_, nm) in enumerate(cons_matches)
                        ]
                    elif ant_matches:
                        inferred_names = [m[0] for m in ant_matches]
                if len(inferred_names) != C:
                    raise ValueError(
                        f"Concept count mismatch in TSK rule parser: parsed {len(inferred_names)} concepts, expected C={C}."
                    )
                name_to_idx = {nm: idx for idx, nm in enumerate(inferred_names)}

                for w_str, cname in cons_matches:
                    if cname not in name_to_idx and cname.endswith(".") and cname[:-1] in name_to_idx:
                        cname = cname[:-1]
                    if cname not in name_to_idx:
                        raise ValueError(f"Unknown consequent concept {cname!r} in TSK rule string: {line}")
                    idx = name_to_idx[cname]
                    w_mat[r_idx, idx, k_idx] = float(w_str)

            name_to_idx = {nm: idx for idx, nm in enumerate(inferred_names)}
            for cname, c_str, s_str, g_str in ant_matches:
                if cname not in name_to_idx:
                    raise ValueError(f"Unknown antecedent concept {cname!r} in TSK rule string: {line}")
                idx = name_to_idx[cname]
                c_mat[r_idx, idx] = float(c_str)
                s_mat[r_idx, idx] = float(s_str)
                g_mat[r_idx, idx] = float(g_str)

        if concept_mask is not None:
            if isinstance(concept_mask, torch.Tensor):
                m_np = concept_mask.detach().cpu().numpy().astype(np.float64)
            else:
                m_np = np.asarray(concept_mask, dtype=np.float64)
            m_bin = (np.nan_to_num(m_np, nan=0.0, posinf=0.0, neginf=0.0) > 0.5).astype(np.float64)
            z_eff = z_np * m_bin
            dist_sq = (
                g_mat[None, :, :]
                * (((z_eff[:, None, :] - c_mat[None, :, :]) / s_mat[None, :, :]) ** 2)
                * m_bin[:, None, :]
            )
            active_c = np.maximum(1.0, np.sum(m_bin, axis=-1, keepdims=True))
            gated_dist = np.sum(dist_sq, axis=-1) / active_c
        else:
            z_eff = z_np
            dist_sq = g_mat[None, :, :] * (((z_eff[:, None, :] - c_mat[None, :, :]) / s_mat[None, :, :]) ** 2)
            gated_dist = np.mean(dist_sq, axis=-1)

        logits_fire = -0.5 * gated_dist
        logits_fire = logits_fire - np.max(logits_fire, axis=-1, keepdims=True)
        exp_fire = np.exp(logits_fire)
        norm_fire = exp_fire / np.sum(exp_fire, axis=-1, keepdims=True)

        rule_out = np.einsum("bc,rck->brk", z_eff, w_mat) + b_mat[None, :, :]
        pred_logit = np.einsum("br,brk->bk", norm_fire, rule_out)
        if not is_multiclass and K == 1:
            return pred_logit[:, 0].astype(np.float32)
        return pred_logit.astype(np.float32)


class PythagoreanFuzzyTTALoss(nn.Module):
    """
    Block 5: Yager Pythagorean Fuzzy Set (PFS)-inspired Test-Time Adaptation Loss.

    Sanitizes input IFNs against NaN/+-Inf, applies hesitancy pre-shrinkage
    s_P = 1 - 0.5 * clamp(pi, 0, 0.99), projects (mu * s_P, nu * s_P) onto the
    Pythagorean Fuzzy disk mu_P(x)^2 + nu_P(x)^2 <= 1, and computes strictly bounded
    Pythagorean quadratic hesitancy:
        q_P(x)  = clamp(1 - mu_P(x)^2 - nu_P(x)^2, 0, 1),
        pi_P(x) = clamp(sqrt(q_P(x) + 1e-7), 0, 1) in [0, 1].
    Note: for valid IFN inputs inside the intuitionistic simplex Delta_IFS
    (mu + nu <= 1), s_P <= 1 guarantees mu_P^2 + nu_P^2 <= (mu_P + nu_P)^2 <= 1,
    so disk projection activates only on out-of-simplex inputs; pi_P thus acts
    as a PFS-inspired quadratic residual regularizer within Delta_IFS.
    Minimizes q_P(x) on reliable test samples gated by detached reliability mask
    pi_P(x) <= tau_P (default 0.55, with all-sample fallback if empty), regularized
    by marginal class-diversity negative entropy to prevent one-class collapse:
        L_{PFS-TTA} = E_{pi_P <= tau_P}[q_P(x)] + lambda_div * sum_k p_bar_k log(p_bar_k).
    """

    def __init__(self, hesitancy_threshold: float = 0.55, diversity_weight: float = 0.25):
        super().__init__()
        self.hesitancy_threshold = hesitancy_threshold
        self.diversity_weight = diversity_weight

    def forward(
        self,
        ifn_or_mu: Union[IntuitionisticFuzzyTensor, torch.Tensor],
        nu_c: Optional[torch.Tensor] = None,
        pi_c: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not getattr(_HALF_PREC_TLS, "active", False):
            lp_dt, dev = _collect_lp_info((ifn_or_mu, nu_c, pi_c), self)
            if lp_dt is not None:
                return _run_module_in_work_prec(self, lp_dt, dev, (ifn_or_mu, nu_c, pi_c))
        if isinstance(ifn_or_mu, IntuitionisticFuzzyTensor):
            mu_raw, nu_raw, pi_raw = ifn_or_mu.mu, ifn_or_mu.nu, ifn_or_mu.pi
        else:
            assert nu_c is not None and pi_c is not None
            mu_raw, nu_raw, pi_raw = ifn_or_mu, nu_c, pi_c

        mu_in = torch.nan_to_num(mu_raw, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        nu_in = torch.nan_to_num(nu_raw, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        pi_in = torch.nan_to_num(pi_raw, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        scale = 1.0 - 0.5 * pi_in.clamp(0.0, 0.99)
        mu_p = (mu_in * scale).clamp(0.0, 1.0)
        nu_p = (nu_in * scale).clamp(0.0, 1.0)

        norm_sq = (mu_p.pow(2) + nu_p.pow(2)).clamp(min=0.0)
        over_mask = (norm_sq > 1.0).to(dtype=mu_in.dtype)
        denom = torch.sqrt(norm_sq + 1e-7)
        mu_p = torch.where(over_mask > 0.5, mu_p / denom, mu_p)
        nu_p = torch.where(over_mask > 0.5, nu_p / denom, nu_p)

        pi_p_sq = (1.0 - mu_p.pow(2) - nu_p.pow(2)).clamp(min=0.0, max=1.0)
        pi_p = torch.sqrt(pi_p_sq + 1e-7).clamp(min=0.0, max=1.0)

        reliable_mask = (pi_p <= self.hesitancy_threshold).to(dtype=mu_in.dtype).detach()
        if reliable_mask.sum() < 1.0:
            reliable_mask = torch.ones_like(pi_p)

        hes_loss = (reliable_mask * pi_p_sq).sum() / (reliable_mask.sum() + 1e-7)

        # Collapse-prevention marginal class diversity regularizer
        prob_pos = (mu_p.pow(2) + 0.5 * pi_p_sq).clamp(1e-5, 1.0 - 1e-5)
        mean_pos = prob_pos.mean()
        mean_neg = 1.0 - mean_pos
        div_loss = mean_pos * torch.log(mean_pos) + mean_neg * torch.log(mean_neg)

        loss = hes_loss + self.diversity_weight * div_loss
        return {
            "tta_loss": loss,
            "mu_p": mu_p,
            "nu_p": nu_p,
            "pi_p": pi_p,
            "reliable_fraction": reliable_mask.mean(),
        }


class IntuitionisticFuzzyLoss(nn.Module):
    """
    Unified Evidential Intuitionistic Fuzzy Training Loss (matches Manuscript Eq. 16):
      1. Log-safe Binary Cross-Entropy (straight-through [eps_BCE, 1-eps_BCE] BCE clamp
         evaluated in float32/float64 precision with eps_BCE=1e-4 for float16 and 1e-6 for
         bfloat16/float32/float64, plus float16-range gradient guarding, preserving finite tail
         gradients into prob / TSK head) + exact unclipped Brier Risk on [0, 1]
      2. Direct Intuitionistic Simplex Evidence Alignment:
            y * (1 - mu_fused)^2 + (1 - y) * (1 - nu_fused)^2
      3. Epistemic Hesitancy-to-Error Calibration on unclipped [0, 1] probability:
            (pi_fused - |sg(P(y=1)) - y|)^2
      4. False-Evidence Penalty on Misclassified Samples & Auxiliary Per-Modality Supervision
      5. TSK Rule Antecedent Sparsity Regularization ||g||_1 and Fuzzy-Rough Boundary Contraction
    """

    def __init__(
        self,
        lambda_dir: float = 0.25,
        lambda_hes_cal: float = 0.20,
        lambda_conflict_reg: float = 0.15,
        lambda_sparse: float = 0.02,
        lambda_rough_bnd: float = 0.05,
        lambda_aux_mod: float = 0.20,
        false_evidence_weight: Optional[float] = None,
        modality_aux_weight: Optional[float] = None,
        rough_boundary_weight: Optional[float] = None,
    ):
        super().__init__()
        self.lambda_dir = lambda_dir
        self.lambda_hes_cal = lambda_hes_cal
        self.lambda_conflict_reg = (
            false_evidence_weight if false_evidence_weight is not None else lambda_conflict_reg
        )
        self.lambda_sparse = lambda_sparse
        self.lambda_rough_bnd = (
            rough_boundary_weight if rough_boundary_weight is not None else lambda_rough_bnd
        )
        self.lambda_aux_mod = (
            modality_aux_weight if modality_aux_weight is not None else lambda_aux_mod
        )

    @staticmethod
    def _sanitize_ifn_components(
        mu: torch.Tensor,
        nu: torch.Tensor,
        pi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects arbitrary/nonfinite IFN tensors onto the closed simplex cleanly up to target IEEE-754 rounding."""
        orig_dtype = mu.dtype if mu.dtype.is_floating_point else torch.float32
        work_dt = _work_dtype_for(mu.device)
        mu_f = mu.to(dtype=work_dt)
        nu_f = nu.to(dtype=work_dt)
        mu_c = _clamp_preserve_boundary_grad(
            torch.nan_to_num(mu_f, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        )
        nu_c = _clamp_preserve_boundary_grad(
            torch.nan_to_num(nu_f, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        )
        s = mu_c + nu_c
        scale = torch.where(s > 1.0, s, torch.ones_like(s))
        mu_norm_w = mu_c / scale
        nu_norm_w = nu_c / scale
        orig_eps = float(torch.finfo(orig_dtype).eps)
        if pi is not None:
            pi_f = pi.to(dtype=work_dt)
            pi_c_w = _clamp_preserve_boundary_grad(
                torch.nan_to_num(pi_f, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0
            )
            in_dom_box = (
                torch.isfinite(mu_f)
                & torch.isfinite(nu_f)
                & torch.isfinite(pi_f)
                & (mu_f >= 0.0)
                & (mu_f <= 1.0)
                & (nu_f >= 0.0)
                & (nu_f <= 1.0)
                & (pi_f >= 0.0)
                & (pi_f <= 1.0)
            )
            tot_c_w = mu_c + nu_c + pi_c_w
            tot_d_w = tot_c_w.detach()
            ulp_loss_ok = (
                in_dom_box
                & (s <= 1.0 + 2.0 * orig_eps)
                & (tot_d_w > 0.0)
                & ((tot_d_w - 1.0).abs() <= 2.0 * orig_eps)
            )
            safe_tot_w = torch.where(ulp_loss_ok, tot_c_w, torch.ones_like(tot_c_w))
            mu_norm_w = torch.where(ulp_loss_ok, mu_c / safe_tot_w, mu_norm_w)
            nu_norm_w = torch.where(ulp_loss_ok, nu_c / safe_tot_w, nu_norm_w)
            pi_norm_w = torch.where(ulp_loss_ok, pi_c_w / safe_tot_w, pi_c_w)
            mu_out = mu_norm_w.to(dtype=orig_dtype)
            nu_out = nu_norm_w.to(dtype=orig_dtype)
            mu_out_w = mu_out.to(dtype=work_dt)
            nu_out_w = nu_out.to(dtype=work_dt)
            pi_res_out = _clamp_preserve_boundary_grad(
                1.0 - torch.maximum(mu_out_w, nu_out_w) - torch.minimum(mu_out_w, nu_out_w),
                0.0,
                1.0,
            ).to(dtype=orig_dtype)
            pi_clean_out = pi_norm_w.to(dtype=orig_dtype)
            res_err = (mu_out_w + nu_out_w + pi_res_out.to(dtype=work_dt) - 1.0).abs()
            raw_err = (mu_out_w + nu_out_w + pi_clean_out.to(dtype=work_dt) - 1.0).abs()
            in_dom = in_dom_box & (s <= 1.0)
            use_raw_pi = (
                ulp_loss_ok
                | (
                    in_dom
                    & (
                        (((mu_out + nu_out + pi_clean_out) == 1.0) & (raw_err <= res_err))
                        | (((mu_out + nu_out + pi_clean_out) == 1.0) & (raw_err <= 2.0 * orig_eps))
                    )
                )
            )
            pi_out = torch.where(use_raw_pi, pi_clean_out, pi_res_out)
        else:
            mu_out = mu_norm_w.to(dtype=orig_dtype)
            nu_out = nu_norm_w.to(dtype=orig_dtype)
            mu_out_w = mu_out.to(dtype=work_dt)
            nu_out_w = nu_out.to(dtype=work_dt)
            pi_out = _clamp_preserve_boundary_grad(
                1.0 - torch.maximum(mu_out_w, nu_out_w) - torch.minimum(mu_out_w, nu_out_w),
                0.0,
                1.0,
            ).to(dtype=orig_dtype)
        return mu_out, nu_out, pi_out

    @staticmethod
    def _sanitize_prob_unit_and_bce(
        raw_prob: torch.Tensor,
        eps_bce: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sanitizes probability onto [0, 1] without interior tail clipping and returns straight-through BCE prob."""
        raw_p = raw_prob.view(-1, 1)
        in_dom = torch.isfinite(raw_p) & (raw_p >= 0.0) & (raw_p <= 1.0)
        p_clean = torch.nan_to_num(raw_p, nan=0.5, posinf=1.0, neginf=0.0)
        prob = torch.where(in_dom, p_clean, p_clean.clamp(0.0, 1.0).detach())
        prob_bce = prob + (prob.clamp(eps_bce, 1.0 - eps_bce) - prob).detach()
        return prob, prob_bce

    @staticmethod
    def _promote_low_prec(x: Optional[torch.Tensor], work_dtype: Optional[torch.dtype]) -> Optional[torch.Tensor]:
        if x is None or work_dtype is None:
            return x
        if x.dtype in (torch.float16, torch.bfloat16) and x.dtype != work_dtype:
            return _SafeCastToWorkDtype.apply(x, work_dtype, _FP16_GRAD_MAX, torch.is_grad_enabled())
        return x

    def forward(
        self,
        ifn_or_out: Union[IntuitionisticFuzzyTensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        modality_ifns: Optional[List[IntuitionisticFuzzyTensor]] = None,
        modality_masks: Optional[torch.Tensor] = None,
        final_prob: Optional[torch.Tensor] = None,
        rough_boundary: Optional[torch.Tensor] = None,
        gate_sparsity_l1: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(ifn_or_out, dict):
            raw_mu = ifn_or_out["mu_fused"]
            raw_nu = ifn_or_out["nu_fused"]
            raw_pi = ifn_or_out.get("pi_fused", None)
            raw_prob = ifn_or_out.get("prob", None)
            modality_ifns = ifn_or_out.get("modality_ifns", modality_ifns)
            rough_boundary = ifn_or_out.get("rough_boundary", rough_boundary)
            gate_sparsity_l1 = ifn_or_out.get("gate_sparsity_l1", gate_sparsity_l1)
        else:
            raw_mu = ifn_or_out.mu
            raw_nu = ifn_or_out.nu
            raw_pi = ifn_or_out.pi
            raw_prob = final_prob

        probe_tensors: List[Optional[torch.Tensor]] = [
            raw_mu,
            raw_nu,
            raw_pi,
            raw_prob,
            rough_boundary,
            gate_sparsity_l1,
        ]
        if modality_ifns is not None:
            for m_ifn in modality_ifns:
                probe_tensors.extend([m_ifn.mu, m_ifn.nu, m_ifn.pi])
        active_fp_dtypes = [t.dtype for t in probe_tensors if t is not None and t.dtype.is_floating_point]
        out_dtype = active_fp_dtypes[0] if active_fp_dtypes else torch.float32
        for dt in active_fp_dtypes[1:]:
            out_dtype = torch.promote_types(out_dtype, dt)
        any_fp16 = any(dt == torch.float16 for dt in active_fp_dtypes)
        any_low_prec = any(dt in (torch.float16, torch.bfloat16) for dt in active_fp_dtypes)
        first_fp_dev = next(
            (t.device for t in probe_tensors if isinstance(t, torch.Tensor) and t.dtype.is_floating_point),
            torch.device("cpu"),
        )
        ac_dt = _active_autocast_dtype(first_fp_dev)
        work_dtype = _work_dtype_for(first_fp_dev) if (any_low_prec or ac_dt is not None) else None
        eps_bce = 1e-4 if any_fp16 else 1e-6
        fp16_loss_cap = _fp16_cap_for_inputs(probe_tensors) if any_fp16 else _FP16_GRAD_MAX
        lp_cache: Dict[int, torch.Tensor] = {}

        with _disable_autocast_ctx(first_fp_dev):
            def _promote_loss_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if t is None or work_dtype is None:
                    return t
                if t.dtype in _HALF_PREC_DTYPES:
                    return _promote_low_prec_cached(t, work_dtype, lp_cache, fp16_max=fp16_loss_cap)
                if t.dtype.is_floating_point and t.dtype != work_dtype:
                    return t.to(dtype=work_dtype)
                return t

            if work_dtype is not None:
                mu_in, nu_in, pi_in = _sanitize_half_prec_ifn_for_work(
                    raw_mu, raw_nu, raw_pi, work_dtype, lp_cache, fp16_max=fp16_loss_cap
                )
                mu_in = _promote_loss_tensor(mu_in)
                nu_in = _promote_loss_tensor(nu_in)
                pi_in = _promote_loss_tensor(pi_in)
                assert mu_in is not None and nu_in is not None
                mu, nu, pi = self._sanitize_ifn_components(mu_in, nu_in, pi_in)
            else:
                mu, nu, pi = self._sanitize_ifn_components(raw_mu, raw_nu, raw_pi)

            if raw_prob is not None:
                prob, prob_bce = self._sanitize_prob_unit_and_bce(
                    _promote_loss_tensor(raw_prob), eps_bce=eps_bce
                )
            else:
                prob, prob_bce = self._sanitize_prob_unit_and_bce(mu + 0.5 * pi, eps_bce=eps_bce)

            mu = mu.view(-1, 1)
            nu = nu.view(-1, 1)
            pi = pi.view(-1, 1)

            targets_clean = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)
            y = targets_clean.view(-1, 1).to(dtype=prob.dtype, device=prob.device).clamp(0.0, 1.0)
            if sample_weights is not None:
                w_work = _clamp_preserve_boundary_grad(
                    torch.nan_to_num(
                        sample_weights.view(-1, 1).to(dtype=_work_dtype_for(prob.device), device=prob.device),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ),
                    min_val=0.0,
                )
                if w_work.numel() > 0:
                    w_max = w_work.detach().max()
                    if w_max > 0.0:
                        w_scaled = w_work / w_max
                        w = (w_scaled / w_scaled.mean()).to(dtype=prob.dtype)
                    else:
                        w = (w_work * 0.0).to(dtype=prob.dtype)
                else:
                    w = w_work.to(dtype=prob.dtype)
            else:
                w = torch.ones_like(y)

            bce = (w * F.binary_cross_entropy(prob_bce, y, reduction="none")).mean()
            brier = (w * (prob - y).pow(2)).mean()
            dir_loss = (w * (y * (1.0 - mu).pow(2) + (1.0 - y) * (1.0 - nu).pow(2))).mean()

            # Epistemic hesitancy calibration: pi_fused tracks empirical residual error |sg(p) - y|
            abs_err = (prob.detach() - y).abs()
            hes_cal_loss = (w * (pi - abs_err).pow(2)).mean()

            # Penalize false-class evidence (y * nu + (1 - y) * mu)
            false_ev = y * nu + (1.0 - y) * mu
            ev_reg = (w * false_ev.pow(2)).mean()

            loss = (
                bce
                + 0.35 * brier
                + self.lambda_dir * dir_loss
                + self.lambda_hes_cal * hes_cal_loss
                + self.lambda_conflict_reg * ev_reg
            )

            if modality_ifns is not None and len(modality_ifns) > 0:
                aux_loss = torch.tensor(0.0, device=prob.device, dtype=prob.dtype)
                M = len(modality_ifns)
                for m_idx, m_ifn in enumerate(modality_ifns):
                    if work_dtype is not None:
                        mm_in, mn_in, mp_in = _sanitize_half_prec_ifn_for_work(
                            m_ifn.mu, m_ifn.nu, m_ifn.pi, work_dtype, lp_cache, fp16_max=fp16_loss_cap
                        )
                        mm_in = _promote_loss_tensor(mm_in)
                        mn_in = _promote_loss_tensor(mn_in)
                        mp_in = _promote_loss_tensor(mp_in)
                        assert mm_in is not None and mn_in is not None
                        m_mu, _, m_pi = self._sanitize_ifn_components(mm_in, mn_in, mp_in)
                    else:
                        m_mu, _, m_pi = self._sanitize_ifn_components(m_ifn.mu, m_ifn.nu, m_ifn.pi)
                    m_mu = m_mu.view(-1, 1)
                    m_pi = m_pi.view(-1, 1)
                    obs_weight = (1.0 - m_pi).detach() * w
                    if modality_masks is not None:
                        m_mask = torch.nan_to_num(
                            modality_masks[:, m_idx : m_idx + 1].to(device=prob.device, dtype=prob.dtype),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        ).clamp(0.0, 1.0)
                        obs_weight = obs_weight * m_mask
                    _, m_prob_bce = self._sanitize_prob_unit_and_bce(m_mu + 0.5 * m_pi, eps_bce=eps_bce)
                    aux_bce = F.binary_cross_entropy(m_prob_bce, y, reduction="none")
                    aux_loss = aux_loss + (obs_weight * aux_bce).mean()
                loss = loss + (self.lambda_aux_mod / max(1, M)) * aux_loss

            if gate_sparsity_l1 is not None:
                g_in = _promote_loss_tensor(gate_sparsity_l1)
                assert g_in is not None
                g_clean = torch.nan_to_num(g_in, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                loss = loss + self.lambda_sparse * g_clean
            if rough_boundary is not None:
                rb_in = _promote_loss_tensor(rough_boundary)
                assert rb_in is not None
                rb_clean = torch.nan_to_num(rb_in, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                loss = loss + self.lambda_rough_bnd * rb_clean.mean()

            if loss.dtype != out_dtype:
                if out_dtype in _HALF_PREC_DTYPES:
                    loss = _WorkToLowPrec.apply(loss, out_dtype)
                else:
                    loss = loss.to(dtype=out_dtype)

            loss_dict = {
                "total_loss": float(loss.detach().item()),
                "bce": float(bce.detach().item()),
                "brier": float(brier.detach().item()),
                "hes_cal": float(hes_cal_loss.detach().item()),
                "ev_reg": float(ev_reg.detach().item()),
            }
            return loss, loss_dict


class UnifiedDeepFuzzyModel(nn.Module):
    """
    End-to-End Unified Differentiable Intuitionistic, Fuzzy-Rough, and TSK Fusion Architecture.

    Integrates:
      1. Per-modality feature projection + Soft-Lukasiewicz Fuzzy-Rough token refinement
         (masked by observed modalities).
      2. Per-modality IntuitionisticFuzzyEncoder with exact missing-modality vacuity <0, 0, 1>
         and FuzzyRoughAttentionGate certainty modulation.
      3. Non-additive certainty-ordered IFSChoquetFusion with observed-pair conflict discounting.
      4. White-Box DeepConceptTSKHead operating on measured, observation-masked concepts.
    """

    def __init__(
        self,
        modality_dims: List[int],
        embed_dim: int = 32,
        num_concepts: Optional[int] = None,
        num_rules: Optional[int] = None,
        num_tsk_concepts: int = 8,
        num_tsk_rules: int = 6,
        num_classes: int = 1,
        prior_vacuity: float = 1.2,
        conflict_penalty: float = 0.55,
        dropout: float = 0.15,
        use_rough_bottleneck: bool = True,
        use_feature_gates: Optional[bool] = None,
        use_tsk_head: bool = True,
        disable_vacuity_mask: bool = False,
        uniform_choquet: bool = False,
        concept_names: Optional[List[str]] = None,
    ):
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                f"UnifiedDeepFuzzyModel supports binary endpoints (num_classes=1); got num_classes={num_classes}."
            )
        c_dim = num_concepts if num_concepts is not None else num_tsk_concepts
        r_dim = num_rules if num_rules is not None else num_tsk_rules

        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)
        self.embed_dim = embed_dim
        self.num_concepts = c_dim
        self.num_rules = r_dim
        self.num_classes = num_classes
        self.use_rough_bottleneck = use_rough_bottleneck
        self.use_feature_gates = use_rough_bottleneck if use_feature_gates is None else use_feature_gates
        self.use_tsk_head = use_tsk_head
        self.disable_vacuity_mask = disable_vacuity_mask
        self.uniform_choquet = uniform_choquet

        self.feature_gates = nn.ParameterList([
            nn.Parameter(torch.ones(1, d)) for d in modality_dims
        ])
        self.modality_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for d in modality_dims
        ])

        self.rough_gate = FuzzyRoughAttentionGate(embed_dim=embed_dim, soft_temp=0.15)
        self.ifs_encoders = nn.ModuleList([
            IntuitionisticFuzzyEncoder(
                in_dim=embed_dim + d,
                num_classes=1,
                hidden_dim=embed_dim,
                dropout=dropout,
                prior_vacuity=prior_vacuity,
            )
            for d in modality_dims
        ])

        self.choquet_fusion = IFSChoquetFusion(
            num_modalities=self.num_modalities,
            num_classes=1,
            conflict_penalty=conflict_penalty,
            uniform_choquet=uniform_choquet,
        )

        self.tsk_head = DeepConceptTSKHead(
            num_concepts=c_dim,
            num_rules=r_dim,
            out_dim=1,
            concept_names=concept_names,
        )
        self.tsk_blend_logit = nn.Parameter(torch.tensor(-0.4))

        # Track whether custom non-certainty clinical concept names were supplied for slots [M:]
        if concept_names is not None and c_dim > self.num_modalities:
            tail_names = self.tsk_head.concept_names[self.num_modalities : c_dim]
            self._has_custom_clinical_concepts = any(
                ("certainty" not in nm.lower()) and (re.match(r"^concept_\d+$", nm.lower()) is None)
                for nm in tail_names
            )
        else:
            self._has_custom_clinical_concepts = False

    @property
    def rough_gates(self) -> nn.ParameterList:
        return self.feature_gates

    def forward(
        self,
        modality_inputs: List[torch.Tensor],
        modality_masks: Optional[torch.Tensor] = None,
        clinical_concepts: Optional[torch.Tensor] = None,
        clinical_concept_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if not getattr(_HALF_PREC_TLS, "active", False):
            lp_dt, dev = _collect_lp_info(
                (modality_inputs, modality_masks, clinical_concepts, clinical_concept_masks),
                self,
            )
            if lp_dt is not None:
                return _run_module_in_work_prec(
                    self,
                    lp_dt,
                    dev,
                    (modality_inputs, modality_masks, clinical_concepts, clinical_concept_masks),
                )

        lp_dt = getattr(_HALF_PREC_TLS, "orig_dtype", None) if getattr(_HALF_PREC_TLS, "active", False) else None
        B = modality_inputs[0].shape[0]
        device = modality_inputs[0].device
        dtype = modality_inputs[0].dtype

        if modality_masks is None:
            modality_masks = torch.ones(B, self.num_modalities, device=device, dtype=dtype)
        else:
            modality_masks = modality_masks.to(device=device, dtype=dtype)

        eff_masks = (
            torch.ones_like(modality_masks)
            if self.disable_vacuity_mask
            else (modality_masks > 0.5).to(dtype=dtype)
        )

        raw_cleaned = []
        proj_tokens = []
        for m, x_m in enumerate(modality_inputs):
            x_m_dev = torch.nan_to_num(x_m.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
            m_col = eff_masks[:, m : m + 1]
            if self.use_feature_gates:
                g_m = torch.sigmoid(self.feature_gates[m])
                x_gated = x_m_dev * g_m * m_col
            else:
                x_gated = x_m_dev * m_col
            raw_cleaned.append(x_gated)
            tok_m = self.modality_projectors[m](x_gated) * m_col
            proj_tokens.append(tok_m)

        token_stack = torch.stack(proj_tokens, dim=1)  # [B, M, D]
        if self.use_rough_bottleneck:
            rough_out = self.rough_gate.forward_dict(token_stack, token_mask=eff_masks)
            refined_tokens = rough_out["refined_features"]
            rough_boundary = rough_out["boundary_region"]
            lower_approx = rough_out["lower_approx"]
            upper_approx = rough_out["upper_approx"]
        else:
            refined_tokens = token_stack
            rough_boundary = torch.zeros(B, self.num_modalities, device=device, dtype=dtype)
            lower_approx = torch.zeros(B, self.num_modalities, device=device, dtype=dtype)
            upper_approx = torch.ones(B, self.num_modalities, device=device, dtype=dtype)

        if lp_dt in _HALF_PREC_DTYPES:
            rough_boundary_use, rough_boundary_ret = _bridge_work_via_lp(rough_boundary, lp_dt)
        else:
            rough_boundary_use, rough_boundary_ret = rough_boundary, rough_boundary

        modality_ifns: List[IntuitionisticFuzzyTensor] = []
        mu_list, nu_list, pi_list = [], [], []
        mu_list_w, nu_list_w, pi_list_w = [], [], []
        for m in range(self.num_modalities):
            mask_m = None if self.disable_vacuity_mask else eff_masks[:, m]
            combined_m = torch.cat([refined_tokens[:, m, :], raw_cleaned[m]], dim=-1)
            ifn_m_raw = self.ifs_encoders[m](combined_m, modality_mask=mask_m)
            if self.use_rough_bottleneck:
                fr_cert = (1.0 - 0.12 * rough_boundary_use[:, m : m + 1]).clamp(0.75, 1.0)
                mu_m = ifn_m_raw.mu * fr_cert
                nu_m = ifn_m_raw.nu * fr_cert
                pi_m_cand = (ifn_m_raw.pi * fr_cert + (1.0 - fr_cert)).clamp(0.0, 1.0)
                if lp_dt in _HALF_PREC_DTYPES:
                    pi_m = pi_m_cand
                else:
                    pi_m_std = (1.0 - mu_m - nu_m).clamp(0.0, 1.0)
                    pi_m = torch.where((mu_m + nu_m).detach() >= 0.999, pi_m_cand, pi_m_std)
            else:
                mu_m, nu_m, pi_m = ifn_m_raw.mu, ifn_m_raw.nu, ifn_m_raw.pi

            if lp_dt in _HALF_PREC_DTYPES:
                mu_w, mu_lp = _bridge_work_via_lp(mu_m, lp_dt)
                nu_w, nu_lp = _bridge_work_via_lp(nu_m, lp_dt)
                pi_w, pi_lp = _bridge_work_via_lp(pi_m, lp_dt)
                modality_ifns.append(IntuitionisticFuzzyTensor(mu=mu_lp, nu=nu_lp, pi=pi_lp))
                mu_list.append(mu_lp)
                nu_list.append(nu_lp)
                pi_list.append(pi_lp)
                mu_list_w.append(mu_w)
                nu_list_w.append(nu_w)
                pi_list_w.append(pi_w)
            else:
                ifn_m = (
                    IntuitionisticFuzzyTensor(mu=mu_m, nu=nu_m, pi=pi_m)
                    if self.use_rough_bottleneck
                    else ifn_m_raw
                )
                modality_ifns.append(ifn_m)
                mu_list.append(ifn_m.mu)
                nu_list.append(ifn_m.nu)
                pi_list.append(ifn_m.pi)

        fused_ifn, fusion_out = self.choquet_fusion(modality_ifns)  # type: ignore[misc]
        mu_comb = fused_ifn.mu
        nu_comb = fused_ifn.nu
        pi_comb = fused_ifn.pi
        ifs_prob = fused_ifn.expected_probability()

        # Measured Concept Vector z in R^C and Observation Mask m_z in {0, 1}^C for White-Box TSK Rules
        if lp_dt in _HALF_PREC_DTYPES:
            mod_net_scores = torch.cat(
                [mu_list_w[m] - nu_list_w[m] for m in range(self.num_modalities)], dim=-1
            )
            pi_comb_use = _promote_low_prec_cached(pi_comb, dtype, {})
            ifs_prob_use = _promote_low_prec_cached(ifs_prob, dtype, {})
            assert pi_comb_use is not None and ifs_prob_use is not None
        elif lp_dt is not None and lp_dt != dtype:
            mod_net_scores = torch.cat(
                [mu_list[m] - nu_list[m] for m in range(self.num_modalities)], dim=-1
            )
            ifs_prob_lp = ifs_prob.to(dtype=lp_dt)
            ifs_prob_use = ifs_prob_lp.to(dtype=dtype) + (
                ifs_prob.detach() - ifs_prob_lp.detach().to(dtype=dtype)
            )
            ifs_prob = ifs_prob_lp
            pi_comb_use = pi_comb
        else:
            mod_net_scores = torch.cat(
                [mu_list[m] - nu_list[m] for m in range(self.num_modalities)], dim=-1
            )  # [B, M]
            pi_comb_use = pi_comb
            ifs_prob_use = ifs_prob
        mod_obs_mask = eff_masks  # [B, M]
        if clinical_concepts is not None:
            clin_dev = clinical_concepts.to(device=device, dtype=dtype)
            clin_finite = torch.isfinite(clin_dev).to(dtype=dtype)
            c_clean = torch.nan_to_num(clin_dev, nan=0.0, posinf=0.0, neginf=0.0)
            if self.disable_vacuity_mask:
                clin_obs = torch.ones_like(clin_finite)
            else:
                clin_obs = eff_masks[:, -1:] * clin_finite
                if clinical_concept_masks is not None:
                    ccm_dev = clinical_concept_masks.to(device=device, dtype=dtype)
                    if ccm_dev.shape != clin_dev.shape:
                        raise ValueError(
                            f"clinical_concept_masks shape {tuple(ccm_dev.shape)} must match "
                            f"clinical_concepts shape {tuple(clin_dev.shape)}"
                        )
                    ccm_clean = torch.nan_to_num(ccm_dev, nan=0.0, posinf=0.0, neginf=0.0)
                    clin_obs = clin_obs * (ccm_clean > 0.5).to(dtype=dtype)
            c_clean = c_clean * clin_obs
            z_raw = torch.cat([mod_net_scores, c_clean], dim=-1)
            mask_raw = torch.cat([mod_obs_mask, clin_obs], dim=-1)
        else:
            if self.use_tsk_head and self._has_custom_clinical_concepts:
                warnings.warn(
                    f"Custom clinical concept_names {self.tsk_head.concept_names[self.num_modalities:]} "
                    "were provided to UnifiedDeepFuzzyModel, but clinical_concepts=None in forward(); "
                    "slots [M:] are populated with modality certainties (1 - pi_m) rather than clinical biomarkers.",
                    UserWarning,
                    stacklevel=2,
                )
            if lp_dt in _HALF_PREC_DTYPES:
                mod_cert = torch.cat(
                    [mu_list_w[m] + nu_list_w[m] for m in range(self.num_modalities)], dim=-1
                )
            else:
                mod_cert = torch.cat([1.0 - pi_list[m] for m in range(self.num_modalities)], dim=-1)
            z_raw = torch.cat([mod_net_scores, mod_cert], dim=-1)
            mask_raw = torch.cat([mod_obs_mask, mod_obs_mask], dim=-1)

        if z_raw.shape[1] >= self.num_concepts:
            z_concepts = z_raw[:, : self.num_concepts]
            concept_mask = mask_raw[:, : self.num_concepts]
        else:
            pad_len = self.num_concepts - z_raw.shape[1]
            pad = torch.zeros(B, pad_len, device=device, dtype=dtype)
            z_concepts = torch.cat([z_raw, pad], dim=-1)
            pad_mask = (
                torch.ones(B, pad_len, device=device, dtype=dtype)
                if self.disable_vacuity_mask
                else torch.zeros(B, pad_len, device=device, dtype=dtype)
            )
            concept_mask = torch.cat([mask_raw, pad_mask], dim=-1)

        if lp_dt in _HALF_PREC_DTYPES:
            z_use, z_ret = _bridge_work_via_lp(z_concepts, lp_dt)
        elif lp_dt is not None and lp_dt != dtype:
            z_ret = z_concepts.to(dtype=lp_dt)
            z_use = z_ret.to(dtype=dtype) + (z_concepts.detach() - z_ret.detach().to(dtype=dtype))
        else:
            z_use, z_ret = z_concepts, z_concepts

        if self.use_tsk_head:
            tsk_logits_w, tsk_out = self.tsk_head(
                z_use, concept_mask=None if self.disable_vacuity_mask else concept_mask
            )
            if lp_dt in _HALF_PREC_DTYPES:
                tsk_logits_use, tsk_logits = _bridge_work_via_lp(tsk_logits_w, lp_dt)
                tsk_prob = torch.sigmoid(tsk_logits_use)
                alpha = torch.sigmoid(self.tsk_blend_logit) * (1.0 - pi_comb_use)
                final_prob_w = _clamp_preserve_boundary_grad(
                    (1.0 - alpha) * ifs_prob_use + alpha * tsk_prob, 0.0, 1.0
                )
                _, final_prob = _bridge_work_via_lp(final_prob_w, lp_dt)
            elif lp_dt is not None and lp_dt != dtype:
                tsk_logits = tsk_logits_w.to(dtype=lp_dt)
                tsk_logits_use = tsk_logits.to(dtype=dtype) + (
                    tsk_logits_w.detach() - tsk_logits.detach().to(dtype=dtype)
                )
                tsk_prob = torch.sigmoid(tsk_logits_use)
                alpha = torch.sigmoid(self.tsk_blend_logit) * (1.0 - pi_comb_use)
                final_prob = _clamp_preserve_boundary_grad(
                    (1.0 - alpha) * ifs_prob_use + alpha * tsk_prob, 0.0, 1.0
                ).to(dtype=lp_dt)
            else:
                tsk_logits = tsk_logits_w
                tsk_prob = torch.sigmoid(tsk_logits)
                alpha = torch.sigmoid(self.tsk_blend_logit) * (1.0 - pi_comb)
                final_prob = _clamp_preserve_boundary_grad(
                    (1.0 - alpha) * ifs_prob + alpha * tsk_prob, 0.0, 1.0
                )
            gate_l1 = tsk_out["gate_sparsity_l1"]
            rule_firing = tsk_out["rule_firing"]
        else:
            final_prob = ifs_prob
            gate_l1 = torch.tensor(0.0, device=device, dtype=dtype)
            rule_firing = torch.zeros(B, self.num_rules, device=device, dtype=dtype)
            if lp_dt in _HALF_PREC_DTYPES:
                eps_logit = 1e-4 if lp_dt == torch.float16 else 1e-6
                p_st = ifs_prob_use + (ifs_prob_use.clamp(eps_logit, 1.0 - eps_logit) - ifs_prob_use).detach()
                _, tsk_logits = _bridge_work_via_lp(torch.logit(p_st), lp_dt)
            elif lp_dt is not None and lp_dt != dtype:
                eps_logit = 1e-4 if lp_dt == torch.float16 else 1e-6
                p_st = ifs_prob_use + (ifs_prob_use.clamp(eps_logit, 1.0 - eps_logit) - ifs_prob_use).detach()
                tsk_logits = torch.logit(p_st).to(dtype=lp_dt)
            else:
                p_st = ifs_prob + (ifs_prob.clamp(1e-6, 1.0 - 1e-6) - ifs_prob).detach()
                tsk_logits = torch.logit(p_st)

        return {
            "fused_ifn": fused_ifn,
            "modality_ifns": modality_ifns,
            "prob": final_prob,
            "ifs_prob": ifs_prob,
            "mu_fused": mu_comb,
            "nu_fused": nu_comb,
            "pi_fused": pi_comb,
            "hesitancy": pi_comb,
            "conflict": fusion_out["conflict"],
            "pairwise_conflict": fusion_out["pairwise_conflict"],
            "choquet_weights": fusion_out["weights"],
            "modality_mus": mu_list,
            "modality_nus": nu_list,
            "modality_pis": pi_list,
            "rough_boundary": rough_boundary_ret,
            "lower_approx": lower_approx,
            "upper_approx": upper_approx,
            "rule_firing": rule_firing,
            "tsk_logits": tsk_logits,
            "z_concepts": z_ret,
            "concept_mask": concept_mask,
            "gate_sparsity_l1": gate_l1,
        }
