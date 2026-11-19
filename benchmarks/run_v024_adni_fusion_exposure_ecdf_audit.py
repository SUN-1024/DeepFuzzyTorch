#!/usr/bin/env python3
"""Fold-streamed ADNI fusion, training-exposure, and nested ECDF audit.

The two input archives are read-only.  The JSON contains aggregate information
only; patient-level predictions and calibration scores are stored in a separate
permission-restricted NPZ.  Full execution requires the benchmark dependencies,
ADNI data, the existing result archives, and local TabPFN V2 model weights.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import atexit
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold

from deepfuzzy import (CombinatorialMissingModalityBenchmark,
                       CrossModalityConflictInjector, FuzzyRoughFeatureReducer,
                       UnifiedDeepFuzzyModel, bootstrap_confidence_interval,
                       compute_binary_metrics, paired_bootstrap_pvalue)
from deepfuzzy.baselines import QMFBaseline
from benchmarks.run_full_benchmark import (fit_strict_preprocessor,
    load_adni_cohort1, resolve_data_paths, set_all_seeds,
    train_deepfuzzy_or_baseline, clopper_pearson_ci, compute_bh_fdr_map)

MODALITY_NAMES = ["MRI", "Plasma", "FDG", "Clin"]
CONCEPT_NAMES = ["MRI_NetSupport", "Plasma_NetSupport", "FDG_NetSupport", "Clin_NetSupport",
                 "APOE4", "MMSE", "ADAS13", "CDRSB"]
OLD_NEURAL = ["DeepFuzzyTorch", "FUML", "QMF", "DrFuse", "DDEF"]
NEW_NEURAL = ["FUML_Faithful", "RFUML_Faithful", "QMF_Faithful", "MUSE_BatchGraph"]
TABULAR = ["XGBoost", "CatBoost", "TabPFN"]
MODDROP = [x + "_ModDrop" for x in TABULAR]
MODELS = OLD_NEURAL + TABULAR + NEW_NEURAL + MODDROP
DUAL = "DualStream_ModDropTabPFN_DeepFuzzyTorch"
FIELDS = ("prob", "hesitancy", "conflict")
SCORES = ("comp", "conf", "hes")


def _clean_inputs(xs: Sequence[torch.Tensor], mask: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    mask = (mask > 0.5).to(dtype=torch.float32)
    clean = [torch.where(mask[:, i:i + 1] > 0, torch.nan_to_num(x.float()),
                         torch.zeros_like(x, dtype=torch.float32)) for i, x in enumerate(xs)]
    return clean, mask


def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


class FUMLFaithfulBaseline(nn.Module):
    """Gaussian prototype possibilities and conflict-tolerant DP-style fusion."""
    def __init__(self, modality_dims: Sequence[int], embed_dim: int = 40,
                 dropout: float = 0.2, n_prototypes: int = 8):
        super().__init__()
        self.num_modalities = len(modality_dims)
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(d, embed_dim), nn.GELU(),
            nn.LayerNorm(embed_dim), nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim),
            nn.GELU()) for d in modality_dims])
        self.centers = nn.Parameter(torch.randn(self.num_modalities, n_prototypes, embed_dim) * 0.1)
        self.log_sigmas = nn.Parameter(torch.zeros(self.num_modalities, n_prototypes, embed_dim))
        self.heads = nn.ModuleList([nn.Linear(n_prototypes, 2) for _ in modality_dims])

    def discount(self, pi: torch.Tensor, mask: torch.Tensor,
                 credibility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return pi, torch.zeros_like(mask)

    def forward(self, modality_inputs: Sequence[torch.Tensor],
                modality_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        xs, mask = _clean_inputs(modality_inputs, modality_masks)
        ps = []
        for m, x in enumerate(xs):
            z = self.encoders[m](x)
            sigma = self.log_sigmas[m].clamp(-4, 4).exp()
            sim = torch.exp(-0.5 * (((z[:, None, :] - self.centers[m]) / sigma) ** 2).mean(-1))
            raw = torch.sigmoid(self.heads[m](sim)).clamp_min(1e-6)
            p = raw / raw.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
            ps.append(torch.where(mask[:, m:m + 1] > 0, p, torch.ones_like(p)))
        pi = torch.stack(ps, dim=1)
        necessity = 1 - pi.flip(-1)
        credibility = 0.5 * (pi + necessity)
        discounted, d = self.discount(pi, mask, credibility)
        necessity_dis = 1 - discounted.flip(-1)
        reliability = necessity_dis.max(dim=-1).values * mask
        all_missing = mask.sum(dim=1, keepdim=True) == 0
        # Normal Dubois-Prade possibilistic conjunctive core over active views (missing = vacuous (1,1)).
        hard_min = discounted.min(dim=1).values
        soft_w = torch.softmax((-discounted).masked_fill(mask[:, :, None] == 0, -1e4) / 0.15, dim=1)
        soft_min = (soft_w * discounted).sum(dim=1)
        pi_conj = hard_min.detach() + soft_min - soft_min.detach()
        conflict = (1 - pi_conj.max(dim=-1, keepdim=True).values).clamp(0, 1)
        conflict = torch.where(all_missing, torch.zeros_like(conflict), conflict)
        fused = (pi_conj + conflict).clamp(0, 1)
        fused = torch.where(all_missing, torch.ones_like(fused), fused)
        nf = 1 - fused.flip(-1)
        cr = 0.5 * (fused + nf)
        prob = (cr[:, :1] / cr.sum(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(0, 1)
        prob = torch.where(all_missing, torch.full_like(prob, 0.5), prob)
        hes = (1 - nf.max(dim=-1, keepdim=True).values).clamp(0, 1)
        hes = torch.where(all_missing, torch.ones_like(hes), hes)
        return {"prob": prob, "hesitancy": hes, "conflict": conflict,
                "fused_credibility": cr, "fused_necessity": nf,
                "view_possibility": pi, "discounted_possibility": discounted,
                "fused_possibility": fused, "view_reliability": reliability,
                "view_credibility": credibility, "view_disagreement": d}


class RFUMLFaithfulBaseline(FUMLFaithfulBaseline):
    def __init__(self, modality_dims: Sequence[int], embed_dim: int = 40, dropout: float = 0.2):
        super().__init__(modality_dims, embed_dim, dropout)
        self.conflict_weight = nn.Parameter(torch.tensor(0.0))
        self.conflict_bias = nn.Parameter(torch.tensor(-1.0))

    def discount(self, pi: torch.Tensor, mask: torch.Tensor,
                 credibility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = credibility[:, :, 0]
        pair = (v[:, :, None] - v[:, None, :]).abs()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        eye = torch.eye(mask.shape[1], device=mask.device, dtype=mask.dtype)[None, :, :]
        pair_mask = pair_mask * (1 - eye)
        count = pair_mask.sum(dim=-1)
        d = (pair * pair_mask).sum(dim=-1) / count.clamp_min(1)
        alpha = torch.sigmoid(F.softplus(self.conflict_weight) * d + self.conflict_bias)
        alpha = alpha * (count > 0).to(alpha.dtype) * mask
        return (1 - alpha[:, :, None]) * pi + alpha[:, :, None], d


class QMFFaithfulBaseline(QMFBaseline):
    """QMF architecture exposing its quality logits for history ranking."""
    def forward(self, modality_inputs: Sequence[torch.Tensor],
                modality_masks: torch.Tensor) -> Dict[str, Any]:
        xs, mask = _clean_inputs(modality_inputs, modality_masks)
        diffs, qualities, views = [], [], []
        for m, x in enumerate(xs):
            logits = torch.nan_to_num(self.encoders[m](x), nan=0.0, posinf=20, neginf=-20).clamp(-20, 20)
            diff = logits[:, 0] - logits[:, 1]
            q = F.softplus(self.energy_scale_raw[m]) * torch.logsumexp(logits, dim=-1)
            q = q + F.softplus(self.margin_scale_raw[m]) * diff.abs()
            diffs.append(diff)
            qualities.append(q)
            views.append(torch.sigmoid(diff))
        q = torch.stack(qualities, dim=1)
        logits = torch.stack(diffs, dim=1)
        views = torch.stack(views, dim=1)
        w = torch.softmax(q.masked_fill(mask == 0, -1e9), dim=1) * mask
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-7)
        prob = torch.sigmoid((w * logits).sum(dim=1, keepdim=True))
        empty = mask.sum(dim=1, keepdim=True) == 0
        prob = torch.where(empty, torch.full_like(prob, 0.5), prob)
        conflict = (2 * torch.sqrt((w * (views - prob).square()).sum(dim=1, keepdim=True) + 1e-8)).clamp(0, 1)
        conflict = torch.where(empty, torch.zeros_like(conflict), conflict)
        hes = (0.5 * _entropy(prob) + 0.5 * conflict).clamp(0, 1)
        hes = torch.where(empty, torch.ones_like(hes), hes)
        return {"prob": prob, "hesitancy": hes, "conflict": conflict,
                "q_m": q, "diff_logits": logits, "view_prob": views}


class MUSEBatchGraphBaseline(nn.Module):
    """Two-layer, inductive patient-modality graph constructed within each batch."""
    def __init__(self, modality_dims: Sequence[int], embed_dim: int = 40, dropout: float = 0.2):
        super().__init__()
        self.projectors = nn.ModuleList([nn.Sequential(nn.Linear(d, embed_dim), nn.GELU(),
            nn.LayerNorm(embed_dim), nn.Dropout(dropout)) for d in modality_dims])
        self.prototype = nn.Parameter(torch.randn(len(modality_dims), embed_dim) * 0.05)
        self.patient_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(2)])
        self.modality_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(2)])
        self.edge_layers = nn.ModuleList([nn.Linear(2 * embed_dim, embed_dim) for _ in range(2)])
        self.readout_gate = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, 1)
        self.view_head = nn.Linear(embed_dim, 1)

    def forward(self, modality_inputs: Sequence[torch.Tensor],
                modality_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        xs, mask = _clean_inputs(modality_inputs, modality_masks)
        edge = torch.stack([proj(x) for proj, x in zip(self.projectors, xs)], dim=1)
        edge = edge * mask[:, :, None]
        patient = edge.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        nodes = self.prototype
        for patient_layer, modality_layer, edge_layer in zip(self.patient_layers, self.modality_layers, self.edge_layers):
            msgs = edge + patient_layer(patient)[:, None, :]
            agg = (msgs * mask[:, :, None]).sum(0) / mask.sum(0)[:, None].clamp_min(1)
            nodes = torch.tanh(modality_layer(nodes) + agg)
            context = nodes[None, :, :].expand(edge.shape[0], -1, -1)
            edge = torch.tanh(edge_layer(torch.cat([edge, context], dim=-1))) * mask[:, :, None]
            patient = edge.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        gate = torch.sigmoid(self.readout_gate(edge)).squeeze(-1) * mask
        gate = gate / gate.sum(1, keepdim=True).clamp_min(1e-7)
        prob = torch.sigmoid(self.classifier((gate[:, :, None] * edge).sum(1)))
        views = torch.sigmoid(self.view_head(edge)).squeeze(-1)
        disagreement = 2 * torch.sqrt((gate * (views - prob).square()).sum(1, keepdim=True) + 1e-8)
        empty = mask.sum(1, keepdim=True) == 0
        prob = torch.where(empty, torch.full_like(prob, 0.5), prob)
        conflict = torch.where(empty, torch.zeros_like(disagreement), disagreement.clamp(0, 1))
        hes = torch.where(empty, torch.ones_like(prob), _entropy(prob))
        return {"prob": prob, "hesitancy": hes, "conflict": conflict}


def build_tabular_matrix(xs: Sequence[np.ndarray], mask: np.ndarray) -> np.ndarray:
    mask = (np.asarray(mask) > 0.5).astype(np.float32)
    parts = [np.where(mask[:, i:i + 1] > 0, np.asarray(x, dtype=np.float32), np.nan)
             for i, x in enumerate(xs)]
    return np.concatenate(parts + [mask], axis=1).astype(np.float32)


def augment_training(xs: Sequence[np.ndarray], natural: np.ndarray, y: np.ndarray,
                     seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    natural = (np.asarray(natural) > 0.5).astype(np.float32)
    rng = np.random.default_rng(seed)
    masks = [natural]
    for _ in range(4):
        aug = natural * (rng.random(natural.shape) >= 0.20)
        lost = (aug.sum(axis=1) == 0) & (natural.sum(axis=1) > 0)
        aug[lost] = natural[lost]
        masks.append(aug.astype(np.float32))
    matrix = np.concatenate([build_tabular_matrix(xs, m) for m in masks], axis=0)
    return matrix, np.tile(np.asarray(y, dtype=np.int64), 5), np.stack(masks)


# Same isolated length-prefixed pickle protocol and hyperparameters as the Phase-1 worker.
# A release command prevents hundreds of nested fitted estimators accumulating in the child.
_WORKER_SOURCE = r'''
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import pickle, struct, sys
out = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
models = {}
serial = 0
def readn(n):
    data = bytearray()
    while len(data) < n:
        part = sys.stdin.buffer.read(n - len(data))
        if not part:
            return None
        data.extend(part)
    return bytes(data)
while True:
    h = readn(4)
    if h is None:
        break
    raw = readn(struct.unpack("!I", h)[0])
    if raw is None:
        break
    try:
        cmd = pickle.loads(raw)
        if cmd[0] == "close":
            result = None
        elif cmd[0] == "fit":
            _, name, x, y, seed = cmd
            if name == "XGBoost":
                model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
                    min_child_weight=3, eval_metric="logloss", random_state=seed, n_jobs=1)
            elif name == "CatBoost":
                model = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.04,
                    l2_leaf_reg=4.0, nan_mode="Min", verbose=0, random_seed=seed, thread_count=1)
            else:
                raise ValueError("Unknown model " + name)
            model.fit(x, y)
            serial += 1
            models[serial] = model
            result = serial
        elif cmd[0] == "predict":
            result = np.asarray(models[cmd[1]].predict_proba(cmd[2]), dtype=np.float32)
        elif cmd[0] == "release":
            models.pop(cmd[1], None)
            result = None
        else:
            raise ValueError("Unknown worker command")
        response = ("ok", result)
    except Exception as exc:
        response = ("error", repr(exc))
    data = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
    out.write(struct.pack("!I", len(data)) + data)
    if cmd[0] == "close":
        break
'''


class _GBDTWorkerClient:
    def __init__(self) -> None:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.proc = subprocess.Popen([sys.executable, "-u", "-c", _WORKER_SOURCE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        atexit.register(self.close)

    def call(self, payload: Any) -> Any:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("GBDT worker pipes unavailable")
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self.proc.stdin.write(struct.pack("!I", len(raw)) + raw)
            self.proc.stdin.flush()
            def readn(n: int) -> bytes:
                data = bytearray()
                while len(data) < n:
                    part = self.proc.stdout.read(n - len(data))
                    if not part:
                        raise RuntimeError("GBDT worker exited or closed its protocol pipe")
                    data.extend(part)
                return bytes(data)
            size = struct.unpack("!I", readn(4))[0]
            status, result = pickle.loads(readn(size))
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("GBDT worker communication failed") from exc
        if status != "ok":
            raise RuntimeError("GBDT worker error: " + str(result))
        return result

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.call(("close",))
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
                self.proc.wait()


class ConstantProbability:
    def __init__(self, p: float):
        self.p = float(p)
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(len(x), self.p, dtype=np.float32)
        return np.column_stack([1 - p, p])


class _SmokeTabular:
    """Exercises tabular masks and augmentation without optional model packages."""
    def __init__(self, seed: int):
        self.model = LogisticRegression(max_iter=200, random_state=seed)
    def fit(self, x: np.ndarray, y: np.ndarray) -> "_SmokeTabular":
        self.model.fit(np.nan_to_num(x, nan=0.0), y)
        return self
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.nan_to_num(x, nan=0.0))


class TabularModel:
    def __init__(self, name: str, xs: Sequence[np.ndarray], mask: np.ndarray, y: np.ndarray,
                 seed: int, augment_seed: int, device: str, inner: bool,
                 worker: Optional[_GBDTWorkerClient], smoke: bool):
        self.worker = None
        self.mid = None
        self.model = None
        matrix = build_tabular_matrix(xs, mask)
        labels = np.asarray(y, dtype=np.int64)
        if name.endswith("_ModDrop"):
            matrix, labels, _ = augment_training(xs, mask, y, augment_seed)
        if len(labels) == 0 or np.unique(labels).size < 2:
            self.model = ConstantProbability(float(labels.mean()) if len(labels) else 0.5)
        elif smoke:
            self.model = _SmokeTabular(seed).fit(matrix, labels)
        elif name.startswith(("XGBoost", "CatBoost")):
            if worker is None:
                raise RuntimeError("GBDT subprocess was not initialized")
            self.worker = worker
            self.mid = worker.call(("fit", name.split("_")[0], matrix, labels, int(seed)))
        else:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            self.model = TabPFNClassifier.create_default_for_version(ModelVersion.V2,
                device=device, n_estimators=4 if inner else 8, fit_mode="fit_with_cache", random_state=seed)
            self.model.fit(matrix, labels)

    def predict(self, xs: Sequence[np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
        n = len(mask)
        p = np.full(n, 0.5, dtype=np.float32)
        active = np.asarray(mask).sum(axis=1) > 0
        if np.any(active):
            x = build_tabular_matrix([a[active] for a in xs], mask[active])
            proba = (self.worker.call(("predict", self.mid, x)) if self.worker is not None
                     else self.model.predict_proba(x))
            if np.asarray(proba).shape != (len(x), 2):
                raise RuntimeError("Unexpected tabular probability shape")
            p[active] = np.asarray(proba)[:, 1]
        pp = np.clip(p, 1e-6, 1 - 1e-6)
        h = -(pp * np.log2(pp) + (1 - pp) * np.log2(1 - pp)).astype(np.float32)
        h[~active] = 1.0
        return {"prob": p, "hesitancy": h, "conflict": np.zeros(n, dtype=np.float32)}

    def close(self) -> None:
        if self.worker is not None and self.mid is not None:
            self.worker.call(("release", self.mid))
            self.mid = None
        self.model = None


def clear_cache() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def fit_new(name: str, xs: Sequence[np.ndarray], masks: np.ndarray, y: np.ndarray,
            epochs: int, lr: float, embed_dim: int, seed: int) -> nn.Module:
    set_all_seeds(seed)
    dims = [x.shape[1] for x in xs]
    cls = {"FUML_Faithful": FUMLFaithfulBaseline, "RFUML_Faithful": RFUMLFaithfulBaseline,
           "QMF_Faithful": QMFFaithfulBaseline, "MUSE_BatchGraph": MUSEBatchGraphBaseline}[name]
    model = cls(dims, embed_dim=embed_dim, dropout=0.2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tx = [torch.as_tensor(x, dtype=torch.float32) for x in xs]
    tm = torch.as_tensor(masks, dtype=torch.float32)
    ty = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    history = torch.zeros((len(y), len(xs)), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    batch_size = min(128, max(32, len(y) // 6))
    model.train()
    for ep in range(epochs):
        for start in range(0, len(y), batch_size):
            idx = rng.permutation(len(y)) if start == 0 else perm
            if start == 0:
                perm = idx
            ids = perm[start:start + batch_size]
            bm = tm[ids].clone()
            if ep > 5:
                dropped = bm * torch.from_numpy((rng.random(bm.shape) > 0.20).astype(np.float32))
                dropped[dropped.sum(1) == 0] = bm[dropped.sum(1) == 0]
                bm = dropped
            by = ty[ids]
            opt.zero_grad()
            out = model([x[ids] for x in tx], modality_masks=bm)
            p = out["prob"].clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(p, by)
            if name in ("FUML_Faithful", "RFUML_Faithful"):
                cr = out["fused_credibility"][:, :1]
                pos_possibility = 1 - out["fused_necessity"][:, 1:2]
                loss = loss + F.mse_loss(cr, by) + F.mse_loss(pos_possibility, by)
                view_cr = out["view_credibility"][:, :, 0]
                weight = (1 - out["view_disagreement"].detach()).clamp_min(0.1) * bm
                loss = loss + 0.25 * ((view_cr - by).square() * weight).sum() / weight.sum().clamp_min(1)
            elif name == "QMF_Faithful":
                vp = out["view_prob"].clamp(1e-6, 1 - 1e-6)
                vloss = F.binary_cross_entropy(vp, by.expand_as(vp), reduction="none")
                loss = loss + 0.25 * (vloss * bm).sum() / bm.sum().clamp_min(1)
                h = history[ids]
                dh = h[:, :, None] - h[:, None, :]
                dq = out["q_m"][:, :, None] - out["q_m"][:, None, :]
                valid = (dh.abs() > 0.02) & (bm[:, :, None] > 0) & (bm[:, None, :] > 0)
                if bool(valid.any()):
                    loss = loss + 0.15 * F.relu(0.05 - dh.sign() * dq)[valid].mean()
                with torch.no_grad():
                    current = torch.exp(-vloss.detach())
                    history[ids] = torch.where(bm > 0, 0.9 * h + 0.1 * current, h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        scheduler.step()
    model.eval()
    return model


class Predictor:
    def __init__(self, model: Any, tabular: bool, concepts: bool = False):
        self.model, self.tabular, self.concepts = model, tabular, concepts

    @torch.no_grad()
    def __call__(self, xs: Sequence[torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        xs, mask = _clean_inputs(xs, mask)
        if self.tabular:
            out = self.model.predict([x.cpu().numpy() for x in xs], mask.cpu().numpy())
            return {k: torch.as_tensor(out[k], dtype=torch.float32).reshape(-1, 1) for k in FIELDS}
        if isinstance(self.model, UnifiedDeepFuzzyModel) and self.concepts:
            out = self.model(xs, modality_masks=mask, clinical_concepts=xs[-1])
        else:
            out = self.model(xs, modality_masks=mask)
        return {k: out[k].detach().cpu().reshape(-1, 1) for k in FIELDS}


def fit_model(name: str, xs: Sequence[np.ndarray], mask: np.ndarray, y: np.ndarray,
              cohort: int, seed: int, augment_seed: int, device: str,
              inner: bool, worker: Optional[_GBDTWorkerClient], smoke: bool) -> Predictor:
    if not len(y) or np.unique(y).size < 2:
        model = TabularModel(name, xs, mask, y, seed, augment_seed, device, inner, worker, smoke)
        return Predictor(model, True)
    if name in TABULAR + MODDROP:
        return Predictor(TabularModel(name, xs, mask, y, seed, augment_seed, device, inner, worker, smoke), True)
    epochs, lr, dim = ((28, 1.8e-3, 40) if cohort == 1 else (50, 2.5e-3, 32))
    if smoke:
        epochs, dim = 2, 8
    if name in NEW_NEURAL:
        model = fit_new(name, xs, mask, y, epochs, lr, dim, seed)
    else:
        model = train_deepfuzzy_or_baseline(name, list(xs), mask, y, epochs=epochs,
            lr=lr, embed_dim=dim, concept_names=CONCEPT_NAMES if cohort == 1 else None,
            seed=seed, device=torch.device("cpu"))
    return Predictor(model, False, cohort == 1)


def release_model(predictor: Predictor) -> None:
    if predictor.tabular:
        predictor.model.close()
    predictor.model = None
    clear_cache()


def _folds(y: np.ndarray, n: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y).ravel()
    if len(y) < n:
        raise ValueError("Too few patients for patient-disjoint folds")
    _, counts = np.unique(y, return_counts=True)
    if len(counts) > 1 and counts.min() >= n:
        splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y))
    # A stratified split is undefined for a single class or a class with < n members.
    splitter = KFold(n_splits=n, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y))))


def _reduce_mri(raw_tr: np.ndarray, raw_va: np.ndarray, y: np.ndarray,
                names: Sequence[str], seed: int, max_features: int) -> np.ndarray:
    if not len(y) or np.unique(y).size < 2:
        return np.arange(min(24, raw_tr.shape[1]), dtype=np.int64)
    standardized, _, _, _ = fit_strict_preprocessor([raw_tr], [raw_va])
    red = FuzzyRoughFeatureReducer(max_features=max_features, sigma=0.85, max_samples=4096,
        n_null_permutations=8, n_stability_bootstraps=5, seed=seed)
    red.fit(standardized[0], y, feature_names=list(names), device="cpu")
    return np.asarray(red.selected_indices_[:24], dtype=np.int64)


def _select_raw(raw: Sequence[np.ndarray], ids: np.ndarray, indices: np.ndarray) -> List[np.ndarray]:
    return [raw[0][ids][:, indices]] + [x[ids] for x in raw[1:]]


def _disc_for_rows(df: pd.DataFrame, tr: np.ndarray, rows: np.ndarray,
                   cohort: int) -> np.ndarray:
    clinical = ((df["DX_bl"] != "CN") | (df["CDRSB"] >= 0.5)).to_numpy()
    plasma = "plasma_ptau181_bl" if cohort == 1 else "pT217_F"
    has = ((df[plasma].notna() & (df["blood_is_bl_visit"].fillna(0) == 1)).to_numpy()
           if cohort == 1 else np.ones(len(df), dtype=bool))
    pv = df[plasma].to_numpy()
    hv = df["Hippocampus_ICV_ratio"].to_numpy()
    pool = pv[tr][has[tr]]
    with np.errstate(all="ignore"):
        pmed = float(np.nanmedian(pool)) if np.any(np.isfinite(pool)) else float("nan")
        hmed = float(np.nanmedian(hv[tr])) if np.any(np.isfinite(hv[tr])) else float("nan")
    pp, hh, cc = pv[rows] >= pmed, hv[rows] <= hmed, clinical[rows]
    return has[rows] & ((pp & ~hh & ~cc) | (~pp & hh & cc))


def _make_fids(folds: Sequence[Tuple[np.ndarray, np.ndarray]], n: int) -> np.ndarray:
    ids = np.full(n, -1, dtype=np.int32)
    for f, (tr, va) in enumerate(folds):
        if np.intersect1d(tr, va).size or np.any(ids[va] != -1):
            raise RuntimeError("Invalid patient-disjoint outer split")
        ids[va] = f
    if np.any(ids < 0):
        raise RuntimeError("Outer folds did not cover every patient")
    return ids


def _check_equal(name: str, a: Any, b: Any) -> None:
    if not np.array_equal(np.asarray(a).ravel(), np.asarray(b).ravel()):
        raise RuntimeError(name + " does not match the read-only replay archive")


def load_real() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    results = ROOT / "results"
    with (results / "benchmark_results.json").open("r", encoding="utf-8") as stream:
        master = json.load(stream)
    # Load only replay anchors; the legacy archive may also contain unrelated
    # object arrays from other benchmark tracks.
    with np.load(results / "oof_predictions_and_splits.npz", allow_pickle=True) as archive:
        required = ["adni1_y", "adni2_y", "adni1_fold_ids", "adni2_fold_ids",
                    "adni1_real_discordant_mask", "adni2_real_discordant_mask",
                    "adni2_nonoverlap_c1plasma_mask"]
        optional = (["adni1_rids"] + ["adni1_fold%d_mri_selected_idx" % f for f in range(5)]
                    + ["adni1_prob_" + name for name in OLD_NEURAL + TABULAR])
        archived = {k: np.array(archive[k], copy=True) for k in required}
        archived.update({k: np.array(archive[k], copy=True) for k in optional if k in archive.files})
    adni_dir = resolve_data_paths()["adni_dir"]
    adni1 = load_adni_cohort1(adni_dir)
    raw1 = [np.asarray(x, dtype=np.float32) for x in adni1["X_modalities"]]
    y1_original = np.asarray(adni1["y_amyloid"]).ravel()
    df1 = adni1["df_amyloid"].copy().reset_index(drop=True)
    df2 = pd.read_csv(Path(adni_dir) / "cohort2_adni_ptau217_gfap.csv").copy()
    df2 = df2.assign(PTGENDER_num=(df2["PTGENDER"] == "Male").astype(float))
    df2 = df2[df2["amyloid_gold_pos"].notna()].copy().reset_index(drop=True)
    y2_original = df2["amyloid_gold_pos"].to_numpy()
    mri2 = ["Ventricles_ICV_ratio", "Hippocampus_ICV_ratio", "WholeBrain_ICV_ratio",
            "Entorhinal_ICV_ratio", "Fusiform_ICV_ratio", "MidTemp_ICV_ratio"]
    mri2 += sorted(c for c in df2.columns if c.startswith("FSX_ST"))
    plasma2 = ["pT217_F", "log_pT217_F", "GFAP_Q", "log_GFAP_Q", "ptau217_gfap_ratio"]
    clin2 = ["APOE4", "MMSE", "ADAS13", "CDRSB", "LDELTOTAL", "RAVLT_immediate", "FAQ", "AGE", "PTGENDER_num"]
    fdg = df2[["FDG_suvr"]].to_numpy(dtype=np.float32, copy=True)
    fdg[df2["FDG_is_bl_visit"].fillna(0).to_numpy() != 1] = np.nan
    raw2 = [df2[mri2].to_numpy(dtype=np.float32), df2[plasma2].to_numpy(dtype=np.float32),
            fdg, df2[clin2].to_numpy(dtype=np.float32)]
    if len(y1_original) != 1137 or len(y2_original) != 289:
        raise RuntimeError("Unexpected ADNI cohort sizes")
    _check_equal("Cohort 1 labels", y1_original, archived["adni1_y"])
    _check_equal("Cohort 2 labels", y2_original, archived["adni2_y"])
    if not np.isin(y1_original, [0, 1]).all() or not np.isin(y2_original, [0, 1]).all():
        raise RuntimeError("ADNI targets must be binary")
    # The reference training code receives float32 labels; keeping that dtype
    # also preserves its seed-dependent initialization arithmetic exactly.
    y1 = y1_original.astype(np.float32)
    y2 = y2_original.astype(np.float32)
    folds1, folds2 = _folds(y1, 5, 42), _folds(y2, 5, 42)
    fids1, fids2 = _make_fids(folds1, len(y1)), _make_fids(folds2, len(y2))
    _check_equal("Cohort 1 fold IDs", fids1, archived["adni1_fold_ids"])
    _check_equal("Cohort 2 fold IDs", fids2, archived["adni2_fold_ids"])
    if "adni1_rids" in archived:
        _check_equal("Cohort 1 participant ordering", adni1["rids"], archived["adni1_rids"])
    disc1, disc2 = np.zeros(len(y1), bool), np.zeros(len(y2), bool)
    for tr, va in folds1:
        disc1[va] = _disc_for_rows(df1, tr, va, 1)
    for tr, va in folds2:
        disc2[va] = _disc_for_rows(df2, tr, va, 2)
    has_pl1 = (df1["plasma_ptau181_bl"].notna() & (df1["blood_is_bl_visit"].fillna(0) == 1)).to_numpy()
    unique2 = ~df2["RID"].isin(set(df1.loc[has_pl1, "RID"].tolist())).to_numpy()
    for key, actual in [("adni1_real_discordant_mask", disc1), ("adni2_real_discordant_mask", disc2),
                        ("adni2_nonoverlap_c1plasma_mask", unique2)]:
        _check_equal(key, actual, archived[key].astype(bool))
    if int(disc1.sum() + (disc2 & unique2).sum()) != 198:
        raise RuntimeError("Expected 198 deduplicated discordant participants")
    selected = master["track1_adni_cohort1"]["rough_summary"]["per_fold_selected_indices"]
    if len(selected) != 5:
        raise RuntimeError("Expected five archived MRI selections")
    sel1 = []
    for f, (tr, va) in enumerate(folds1):
        recorded = np.asarray(selected[f], dtype=np.int64)
        if len(recorded) != 24 or len(np.unique(recorded)) != 24 or np.any(recorded < 0) or np.any(recorded >= raw1[0].shape[1]):
            raise RuntimeError("Invalid archived MRI selection")
        replay = _reduce_mri(raw1[0][tr], raw1[0][va], y1[tr], adni1["mri_cols"], 42 + f, 30)
        _check_equal("Cohort 1 MRI selection in fold " + str(f), replay, recorded)
        key = "adni1_fold%d_mri_selected_idx" % f
        if key in archived:
            _check_equal(key, recorded, archived[key])
        sel1.append(recorded)
    sel2 = [_reduce_mri(raw2[0][tr], raw2[0][va], y2[tr], mri2, 100 + f, 24)
            for f, (tr, va) in enumerate(folds2)]
    same = np.asarray(adni1["same_visit_bl_ref_mask"], dtype=bool)
    if int(same.sum()) != 924:
        raise RuntimeError("Expected 924 same-visit reference patients")
    c1 = dict(raw=raw1, y=y1, df=df1, folds=folds1, fids=fids1, selections=sel1,
              mri_cols=adni1["mri_cols"], same=same, disc=disc1)
    c2 = dict(raw=raw2, y=y2, df=df2, folds=folds2, fids=fids2, selections=sel2,
              mri_cols=mri2, disc=disc2, unique=unique2)
    return c1, c2, master, archived


def synthetic_cohort(seed: int, n: int = 48) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2).astype(np.int64)
    rng.shuffle(y)
    raw = [rng.normal(size=(n, d)).astype(np.float32) for d in (6, 2, 1, 5)]
    for m in range(4):
        raw[m][:, 0] += 0.8 * y
    for m in (1, 2):
        raw[m][rng.random(n) < 0.15] = np.nan
    df = pd.DataFrame({"DX_bl": np.where(rng.random(n) < 0.5, "CN", "MCI"),
        "CDRSB": rng.choice([0.0, 0.5], n), "plasma_ptau181_bl": rng.normal(size=n),
        "pT217_F": rng.normal(size=n), "blood_is_bl_visit": np.ones(n),
        "Hippocampus_ICV_ratio": rng.normal(size=n)})
    folds = _folds(y, 2, 42)
    same = rng.random(n) < 0.75
    return dict(raw=raw, y=y, df=df, folds=folds, fids=_make_fids(folds, n),
        selections=[np.arange(6) for _ in folds], mri_cols=["MRI%d" % i for i in range(6)],
        same=same, disc=np.arange(n) % 3 == 0, unique=np.arange(n) % 4 != 0)


def prepare_outer(cohort: Dict[str, Any], real: bool, number: int) -> None:
    raw, y, folds = cohort["raw"], cohort["y"], cohort["folds"]
    n = len(y)
    dims = [len(cohort["selections"][0])] + [x.shape[1] for x in raw[1:]]
    full_x = [torch.zeros((n, d), dtype=torch.float32) for d in dims]
    full_mask = torch.zeros((n, 4), dtype=torch.float32)
    outer = []
    for f, (tr, va) in enumerate(folds):
        sel = cohort["selections"][f]
        xt, xv, mt, mv = fit_strict_preprocessor(_select_raw(raw, tr, sel), _select_raw(raw, va, sel))
        for m in range(4):
            full_x[m][va] = torch.from_numpy(xv[m])
        full_mask[va] = torch.from_numpy(mv)
        pool = _disc_for_rows(cohort["df"], tr, tr, number)
        outer.append(dict(xt=xt, xv=xv, mt=mt, mv=mv, pool=pool))
    cohort["outer"] = outer
    cohort["full_x"], cohort["full_mask"] = full_x, full_mask
    if number == 1:
        complete = (full_mask.numpy() > 0.5).all(axis=1)
        eligible = (full_mask[:, 1] > 0.5) & ((full_mask.sum(1) - full_mask[:, 1]) >= 1)
        cohort["complete"] = complete
        if real and (int(complete.sum()) != 854 or int(eligible.sum()) != 872 or
                     int((complete & cohort["same"]).sum()) != 852):
            raise RuntimeError("Complete-case, same-visit, or conflict-eligible count drifted")


def ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if not len(reference):
        return np.full(len(values), np.nan)
    if not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("ECDF inputs must be finite")
    return 0.5 * ((reference[:, None] < values).mean(0) + (reference[:, None] <= values).mean(0))


def score_values(out: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    h, c = out["hesitancy"], out["conflict"]
    return {"comp": 0.5 * (h + c), "conf": c, "hes": h}


def predict_np(pred: Predictor, xs: Sequence[np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    out = pred([torch.as_tensor(x, dtype=torch.float32) for x in xs],
               torch.as_tensor(mask, dtype=torch.float32))
    vals = {k: out[k].numpy().ravel() for k in FIELDS}
    if any(not np.isfinite(v).all() for v in vals.values()):
        raise RuntimeError("Model produced a nonfinite prediction")
    return vals


def _empty_output(n: int) -> Dict[str, np.ndarray]:
    return {k: np.full(n, np.nan, dtype=np.float32) for k in FIELDS}


def validate_padding(pred: Predictor, xs: Sequence[torch.Tensor], natural: torch.Tensor,
                     bench: CombinatorialMissingModalityBenchmark) -> Dict[str, Any]:
    checks, largest = 0, 0.0
    for label, bits, _ in bench.enumerate_subsets():
        mask = natural * torch.tensor(bits, dtype=natural.dtype).reshape(1, 4)
        reference_x = [torch.where(mask[:, m:m + 1] > 0, x, torch.zeros_like(x))
                       for m, x in enumerate(xs)]
        reference = pred(reference_x, mask)
        for fill in (0.0, -999.0, 777.0, float("nan")):
            padded = [torch.where(mask[:, m:m + 1] > 0, x, torch.full_like(x, fill))
                      for m, x in enumerate(xs)]
            test = pred(padded, mask)
            for field in FIELDS:
                a, b = test[field].numpy(), reference[field].numpy()
                diff = float(np.max(np.abs(a - b))) if len(a) else 0.0
                if not np.isfinite(diff) or diff != 0 or not np.array_equal(a.view(np.uint8), b.view(np.uint8)):
                    raise RuntimeError("Padding invariance failed for %s %s %s: %s" % (label, fill, field, diff))
                largest = max(largest, diff)
                checks += 1
    empty = pred([torch.full_like(x, float("nan")) for x in xs], torch.zeros_like(natural))
    for key, value in [("prob", 0.5), ("hesitancy", 1.0), ("conflict", 0.0)]:
        if not bool(torch.all(empty[key] == value)):
            raise RuntimeError("All-missing prediction is not vacuous for " + key)
    return {"comparisons": checks, "max_padding_diff": largest,
            "fill_values": [0.0, -999.0, 777.0, "NaN"],
            "outputs_checked": list(FIELDS), "all_missing_vacuous": True,
            "bitwise_equality_required": True}


def _save_output(dst: Dict[str, np.ndarray], ids: np.ndarray, out: Dict[str, torch.Tensor]) -> None:
    for key in FIELDS:
        value = out[key].detach().cpu().numpy().ravel()
        if len(value) != len(ids) or not np.isfinite(value).all():
            raise RuntimeError("Invalid fold prediction for " + key)
        dst[key][ids] = value


def _inner_prepared(cohort: Dict[str, Any], f: int, number: int, smoke: bool) -> List[Dict[str, Any]]:
    tr = cohort["folds"][f][0]
    y, raw = cohort["y"], cohort["raw"]
    prepared = []
    for j, (it, iv) in enumerate(_folds(y[tr], 3, 3000 + f)):
        tr_ids, va_ids = tr[it], tr[iv]
        if np.intersect1d(tr_ids, va_ids).size:
            raise RuntimeError("Inner training and validation overlap")
        if smoke:
            sel = cohort["selections"][f]
        else:
            sel = _reduce_mri(raw[0][tr_ids], raw[0][va_ids], y[tr_ids], cohort["mri_cols"],
                              (2000 if number == 1 else 3000) + f * 10 + j,
                              30 if number == 1 else 24)
        xt, xv, mt, mv = fit_strict_preprocessor(_select_raw(raw, tr_ids, sel), _select_raw(raw, va_ids, sel))
        prepared.append(dict(it=it, iv=iv, xt=xt, xv=xv, mt=mt, mv=mv, sel=sel))
    return prepared


def _nested_for_model(cohort: Dict[str, Any], f: int, number: int, name: str,
                      inner_data: Sequence[Dict[str, Any]], worker: Optional[_GBDTWorkerClient],
                      device: str, smoke: bool) -> Dict[str, np.ndarray]:
    tr = cohort["folds"][f][0]
    y = cohort["y"]
    inner_out = _empty_output(len(tr))
    seen = np.zeros(len(tr), dtype=np.int32)
    for j, part in enumerate(inner_data):
        it, iv = part["it"], part["iv"]
        pred = fit_model(name, part["xt"], part["mt"], y[tr[it]], number,
            seed=(42 if number == 1 else 100) + f + 1000 + j,
            augment_seed=(42 if number == 1 else 100) + f + 1000 + j,
            device=device, inner=True, worker=worker, smoke=smoke)
        try:
            vals = predict_np(pred, part["xv"], part["mv"])
            for key in FIELDS:
                inner_out[key][iv] = vals[key]
            seen[iv] += 1
        finally:
            release_model(pred)
    if not np.all(seen == 1) or any(not np.isfinite(v).all() for v in inner_out.values()):
        raise RuntimeError("Inner OOF predictions must cover each outer-training patient once")
    return inner_out


def _batch_coupling(pred: Predictor, xs: Sequence[torch.Tensor], mask: torch.Tensor) -> float:
    full = pred(xs, mask)["prob"].numpy().ravel()
    single = np.concatenate([pred([x[i:i + 1] for x in xs], mask[i:i + 1])["prob"].numpy().ravel()
                             for i in range(len(mask))]) if len(mask) else np.array([])
    return float(np.max(np.abs(full - single))) if len(full) else 0.0


def run_cohort(cohort: Dict[str, Any], number: int, models: Sequence[str],
               worker: Optional[_GBDTWorkerClient], device: str, smoke: bool) -> Dict[str, Any]:
    raw, y, folds = cohort["raw"], cohort["y"], cohort["folds"]
    n = len(y)
    bench = CombinatorialMissingModalityBenchmark(MODALITY_NAMES)
    subsets = bench.enumerate_subsets()
    full_x, full_mask = cohort["full_x"], cohort["full_mask"]
    same = cohort.get("same", np.ones(n, dtype=bool))
    injector = CrossModalityConflictInjector(4, MODALITY_NAMES)
    corrupted = None
    if number == 1:
        corrupted, _ = injector.inject_conflict(full_x, y, corrupted_modality_idx=1,
            conflict_fraction=0.30, severity_gamma=0.8, mode="opposite_swap",
            natural_masks=full_mask, fold_ids=cohort["fids"], seed=42)
    records: Dict[str, Any] = {}
    for name in models:
        rec = {"clean": _empty_output(n),
               "inner_scores": {s: np.full((len(folds), n), np.nan, dtype=np.float32) for s in SCORES},
               "rank_inner": {s: np.full(n, np.nan, dtype=np.float64) for s in SCORES},
               "rank_outer": {s: np.full(n, np.nan, dtype=np.float64) for s in SCORES},
               "padding": {"comparisons": 0, "max_padding_diff": 0.0}, "coupling": 0.0}
        if number == 1:
            rec["subset_full"] = {label: _empty_output(n) for label, _, _ in subsets}
            rec["subset_same"] = {label: _empty_output(n) for label, _, _ in subsets}
            rec["corrupt"] = _empty_output(n)
        records[name] = rec
    fold_metadata = []
    for f, (tr, va) in enumerate(folds):
        print("[C%d fold %d/%d] preparing strict inner folds" % (number, f + 1, len(folds)), flush=True)
        outer = cohort["outer"][f]
        inner_data = _inner_prepared(cohort, f, number, smoke)
        fold_metadata.append({"fold": f, "n_train": len(tr), "n_validation": len(va),
            "n_discordant_calibration": int(outer["pool"].sum()),
            "mri_selected_indices": [int(x) for x in cohort["selections"][f]],
            "inner_folds": [{"fold": j, "n_train": len(d["it"]), "n_validation": len(d["iv"]),
                             "mri_selected_indices": [int(x) for x in d["sel"]]}
                            for j, d in enumerate(inner_data)]})
        for name in models:
            print("  " + name, flush=True)
            rec = records[name]
            inner_out = _nested_for_model(cohort, f, number, name, inner_data, worker, device, smoke)
            inscores = score_values(inner_out)
            for s in SCORES:
                rec["inner_scores"][s][f, tr] = inscores[s]
            pred = fit_model(name, outer["xt"], outer["mt"], y[tr], number,
                seed=(42 if number == 1 else 100) + f,
                augment_seed=(42 if number == 1 else 100) + f,
                device=device, inner=False, worker=worker, smoke=smoke)
            try:
                train_pred = predict_np(pred, outer["xt"], outer["mt"])
                val_pred = predict_np(pred, outer["xv"], outer["mv"])
                for key in FIELDS:
                    rec["clean"][key][va] = val_pred[key]
                trscore, vascore = score_values(train_pred), score_values(val_pred)
                pool = outer["pool"]
                for s in SCORES:
                    rec["rank_inner"][s][va] = ecdf(inscores[s][pool], vascore[s])
                    rec["rank_outer"][s][va] = ecdf(trscore[s][pool], vascore[s])
                if number == 1:
                    vx, vm = [x[va] for x in full_x], full_mask[va]
                    same_ids = va[same[va]]
                    for label, bits, _ in subsets:
                        eff = vm * torch.tensor(bits, dtype=torch.float32).reshape(1, 4)
                        _save_output(rec["subset_full"][label], va, pred(vx, eff))
                        if len(same_ids):
                            same_eff = full_mask[same_ids] * torch.tensor(bits, dtype=torch.float32).reshape(1, 4)
                            _save_output(rec["subset_same"][label], same_ids,
                                         pred([x[same_ids] for x in full_x], same_eff))
                    _save_output(rec["corrupt"], va, pred([x[va] for x in corrupted], vm))
                    if name in NEW_NEURAL + MODDROP:
                        pad = validate_padding(pred, vx, vm, bench)
                        rec["padding"]["comparisons"] += pad["comparisons"]
                        rec["padding"]["max_padding_diff"] = max(rec["padding"]["max_padding_diff"], pad["max_padding_diff"])
                    if name in ("MUSE_BatchGraph", "DeepFuzzyTorch"):
                        rec["coupling"] = max(rec["coupling"], _batch_coupling(pred, vx, vm))
            finally:
                release_model(pred)
    cohort["fold_metadata"] = fold_metadata
    return records


def replay_dual(cohort: Dict[str, Any], device: str, smoke: bool = False) -> Dict[str, Any]:
    # Run sequentially after r1/r2 have freed models, keeping num_threads=1 for TabPFN.
    from benchmarks.run_tabpfn_deepfuzzy_plugin import fit_unimodal_deepfuzzy, fit_dual_stream
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        y, raw, folds = cohort["y"], cohort["raw"], cohort["folds"]
        n = len(y)
        bench = CombinatorialMissingModalityBenchmark(MODALITY_NAMES)
        rec = {"clean": _empty_output(n), "subset_same":
               {label: _empty_output(n) for label, _, _ in bench.enumerate_subsets()}}
        for f, (tr, va) in enumerate(folds):
            print("[Dual stream] fold %d/%d on %s" % (f + 1, len(folds), device), flush=True)
            raw_tr, raw_va = [x[tr] for x in raw], [x[va] for x in raw]
            base = fit_unimodal_deepfuzzy(raw_tr, raw_va, y[tr], cohort["selections"][f],
                42 + f, 28, 2, 8, device, fit_strict_preprocessor)
            dual = None
            try:
                dual, _ = fit_dual_stream(raw_tr, y[tr], base, 42 + f, 28, 2, 8,
                                          device, fit_strict_preprocessor)
                clean = dual.predict(base.clean_val, base.natural_val)
                _save_output(rec["clean"], va, clean)
                local_same = np.flatnonzero(cohort["same"][va])
                if len(local_same):
                    same_ids = va[local_same]
                    for label, bits, _ in bench.enumerate_subsets():
                        mask = base.natural_val[local_same] * np.asarray(bits, dtype=np.float32)
                        out = dual.predict([x[local_same] for x in base.clean_val], mask)
                        _save_output(rec["subset_same"][label], same_ids, out)
            finally:
                del dual, base
                clear_cache()
        return rec
    finally:
        torch.set_num_threads(prev_threads)


def subset_metrics(cohort: Dict[str, Any], rec: Dict[str, Any], same_only: bool) -> Dict[str, Any]:
    bench = CombinatorialMissingModalityBenchmark(MODALITY_NAMES)
    selection = np.flatnonzero(cohort["same"]) if same_only else np.arange(len(cohort["y"]))
    source = rec["subset_same" if same_only else "subset_full"]
    subsets = bench.enumerate_subsets()
    index = [0]
    def dispatch(xs: List[torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if index[0] >= len(subsets):
            raise RuntimeError("Unexpected extra subset call")
        label, bits, _ = subsets[index[0]]
        index[0] += 1
        expected = cohort["full_mask"][selection] * torch.tensor(bits, dtype=torch.float32).reshape(1, 4)
        if not torch.equal(mask, expected):
            raise RuntimeError("Subset dispatcher mask mismatch")
        vals = source[label]
        if any(not np.isfinite(vals[k][selection]).all() for k in FIELDS):
            raise RuntimeError("Incomplete subset OOF predictions")
        return {k: torch.from_numpy(vals[k][selection].reshape(-1, 1)) for k in FIELDS}
    result = bench.evaluate_all_subsets(dispatch, [x[selection] for x in cohort["full_x"]],
                                        cohort["full_mask"][selection], cohort["y"][selection])
    if index[0] != 15:
        raise RuntimeError("Expected 15 combinatorial subsets")
    label = "+".join(MODALITY_NAMES)
    return {"n": int(len(selection)),
        "complete_4mod_n": int(result["subsets"][label]["n_complete_subset"]),
        "complete_4mod_auroc": result["subsets"][label]["auroc_complete_subset"],
        "macro_15subset_auroc": result["macro_avg_auroc"],
        "macro_15subset_auroc_eligible": result["macro_avg_auroc_eligible"],
        "mrr_14incomplete": result["missing_robustness_retention"],
        "mrr_14incomplete_eligible": result["missing_robustness_retention_eligible"],
        "single_modality_auroc": {m: result["subsets"][m]["auroc"] for m in MODALITY_NAMES},
        "subsets": result["subsets"]}


def conflict_metrics(cohort: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    injector = CrossModalityConflictInjector(4, MODALITY_NAMES)
    def dispatch(xs: List[torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: torch.from_numpy(rec["corrupt"][k].reshape(-1, 1)) for k in FIELDS}
    return injector.evaluate_conflict_sweep(dispatch, cohort["full_x"], cohort["full_mask"],
        cohort["y"], corrupted_modality_idx=1, severities=[0.8], referral_rates=[0.0, 0.2],
        fold_ids=cohort["fids"], conflict_fraction=0.30, seed=42)


def metric_entry(p: np.ndarray, h: np.ndarray, y: np.ndarray, n_boot: int) -> Dict[str, Any]:
    result: Dict[str, Any] = compute_binary_metrics(p, y, uncertainty=h)
    result["n"] = int(len(y))
    result["confidence_intervals_95"] = {}
    for metric in ("auroc", "brier"):
        _, lo, hi = bootstrap_confidence_interval(p, y, metric, n_bootstraps=n_boot, seed=42,
                                                  uncertainty=h)
        result["confidence_intervals_95"][metric] = [float(lo), float(hi)]
    return result


def comparison_tables(cohort: Dict[str, Any], recs: Dict[str, Any], n_boot: int) -> Dict[str, Any]:
    y = cohort["y"]
    masks = {"full": np.ones(len(y), dtype=bool)}
    if "complete" in cohort:
        masks.update({"same_visit_bl_924": cohort["same"], "complete_4mod": cohort["complete"]})
    result = {}
    for group, keep in masks.items():
        table = {}
        for name, rec in recs.items():
            p, h = rec["clean"]["prob"][keep], rec["clean"]["hesitancy"][keep]
            table[name] = metric_entry(p, h, y[keep], n_boot)
            for baseline in ("DeepFuzzyTorch", "TabPFN"):
                if baseline in recs:
                    ref = recs[baseline]["clean"]["prob"][keep]
                    pval = (1.0 if name == baseline else
                        paired_bootstrap_pvalue(ref, p, y[keep], "auroc", n_bootstraps=n_boot, seed=42))
                    table[name]["p_vs_" + baseline] = pval
        for baseline in ("DeepFuzzyTorch", "TabPFN"):
            key = "p_vs_" + baseline
            pmap = {name: row[key] for name, row in table.items() if key in row}
            qmap = compute_bh_fdr_map(pmap, exclude_keys=(baseline,))
            for name in pmap:
                table[name]["q_bh_vs_" + baseline] = qmap[name] if np.isfinite(pmap[name]) else float("nan")
        result[group] = table
    return result


def paired_aurc(p_a: np.ndarray, u_a: np.ndarray, p_b: np.ndarray, u_b: np.ndarray,
                y: np.ndarray, n_boot: int, seed: int = 42) -> float:
    if not len(y):
        return float("nan")
    rng = np.random.default_rng(seed)
    diff = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        a = compute_binary_metrics(p_a[idx], y[idx], uncertainty=u_a[idx])["aurc"]
        b = compute_binary_metrics(p_b[idx], y[idx], uncertainty=u_b[idx])["aurc"]
        diff[i] = b - a
    return float(min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))))


def matched_coverage(c1: Dict[str, Any], c2: Dict[str, Any], r1: Dict[str, Any],
                     r2: Dict[str, Any], n_boot: int, smoke: bool) -> Dict[str, Any]:
    mask1 = c1["disc"]
    mask2 = c2["disc"] & c2["unique"]
    y = np.concatenate([c1["y"][mask1], c2["y"][mask2]])
    n = len(y)
    if not smoke and n != 198:
        raise RuntimeError("Wrong deduplicated discordant sample size")
    if n == 0:
        return {"n": 0, "n_c1": 0, "n_c2_nonoverlap": 0, "requested_grid": [],
                "models_by_calibration": {}, "unavailable": "No discordant participants"}
    requested = [("100%", 198), ("90%", 178), ("81.3%", 161),
                 ("80%", 158), ("70%", 139), ("60%", 119)]
    counts = [(key, max(1, min(n, int(round(k * n / 198)) if smoke else k))) for key, k in requested]
    rows: Dict[str, Any] = {}
    names = list(r1)
    variants = [(m, m, "comp") for m in names]
    if "DeepFuzzyTorch" in names:
        variants.append(("DeepFuzzyTorch_pure_C_O", "DeepFuzzyTorch", "conf"))
    def combined(model: str, field: str, score: Optional[str] = None) -> np.ndarray:
        a, b = r1[model][field], r2[model][field]
        if score is not None:
            a, b = a[score], b[score]
        return np.concatenate([a[mask1], b[mask2]])
    refp = combined("DeepFuzzyTorch", "clean", "prob")
    for source in ("rank_inner", "rank_outer"):
        refu = combined("DeepFuzzyTorch", source, "comp")
        table = {}
        for label, name, score in variants:
            p = combined(name, "clean", "prob")
            ranks = combined(name, source, score)
            if not np.isfinite(ranks).all():
                table[label] = {"unavailable": "Empty discordant outer-training calibration pool"}
                continue
            raw1, raw2 = score_values(r1[name]["clean"])[score], score_values(r2[name]["clean"])[score]
            tie = np.concatenate([raw1[mask1], raw2[mask2]])
            order = np.lexsort((tie, ranks))
            pts = {}
            for key, k in counts:
                ids = order[:k]
                met = compute_binary_metrics(p[ids], y[ids])
                correct = int(np.sum((p[ids] >= 0.5) == y[ids]))
                lo, hi = clopper_pearson_ci(correct, k)
                pts[key] = {"n_retained": k, "coverage": k / n, "k_correct": correct,
                            "accuracy": met["acc"], "accuracy_ci_95_clopper_pearson": [lo, hi],
                            "auroc": met["auroc"], "f1": met["f1"], "brier": met["brier"]}
            inner = combined(name, "rank_inner", score)
            outer = combined(name, "rank_outer", score)
            rho = (float(spearmanr(inner, outer).statistic)
                   if np.isfinite(inner).all() and np.isfinite(outer).all() and
                   np.unique(inner).size > 1 and np.unique(outer).size > 1 else float("nan"))
            aurc = compute_binary_metrics(p, y, uncertainty=ranks)["aurc"]
            pval = (1.0 if label == "DeepFuzzyTorch" else
                    paired_aurc(refp, refu, p, ranks, y, n_boot) if np.isfinite(refu).all() else float("nan"))
            table[label] = {"aurc": aurc, "p_aurc_vs_DeepFuzzyTorch_composite": pval,
                            "spearman_inner_vs_outer_ranks": rho, "points": pts}
        pmap = {k: v["p_aurc_vs_DeepFuzzyTorch_composite"] for k, v in table.items()
                if "p_aurc_vs_DeepFuzzyTorch_composite" in v}
        qmap = compute_bh_fdr_map(pmap, exclude_keys=("DeepFuzzyTorch",))
        for key in pmap:
            table[key]["q_bh_aurc_vs_DeepFuzzyTorch_composite"] = qmap[key] if np.isfinite(pmap[key]) else float("nan")
        rows[source] = table
    return {"n": n, "n_c1": int(mask1.sum()), "n_c2_nonoverlap": int(mask2.sum()),
            "requested_grid": [{"label": k, "n_retained": v} for k, v in counts],
            "models_by_calibration": rows}


def _archive_arrays(cohorts: Sequence[Dict[str, Any]], all_records: Sequence[Dict[str, Any]],
                    dual: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for number, (cohort, records) in enumerate(zip(cohorts, all_records), 1):
        prefix = "adni%d" % number
        arrays[prefix + "_y"] = cohort["y"].astype(np.int8)
        arrays[prefix + "_fold_ids"] = cohort["fids"].astype(np.int8)
        arrays[prefix + "_discordant_mask"] = cohort["disc"].astype(np.uint8)
        if number == 1:
            arrays[prefix + "_same_visit_mask"] = cohort["same"].astype(np.uint8)
            arrays[prefix + "_complete_mask"] = cohort["complete"].astype(np.uint8)
        else:
            arrays[prefix + "_nonoverlap_mask"] = cohort["unique"].astype(np.uint8)
        for name, rec in records.items():
            for key in FIELDS:
                arrays["%s_%s_%s" % (prefix, name, key)] = rec["clean"][key]
            for score in SCORES:
                for kind in ("inner_scores", "rank_inner", "rank_outer"):
                    arrays["%s_%s_%s_%s" % (prefix, name, kind, score)] = rec[kind][score]
            if number == 1:
                for kind in ("subset_full", "subset_same"):
                    for label, values in rec[kind].items():
                        for field in FIELDS:
                            arrays["%s_%s_%s_%s_%s" % (prefix, name, kind, label.replace("+", "_"), field)] = values[field]
                for field in FIELDS:
                    arrays["%s_%s_corrupt_%s" % (prefix, name, field)] = rec["corrupt"][field]
    if dual is not None:
        for key in FIELDS:
            arrays["adni1_%s_%s" % (DUAL, key)] = dual["clean"][key]
        for label, values in dual["subset_same"].items():
            for key in FIELDS:
                arrays["adni1_%s_subset_same_%s_%s" % (DUAL, label.replace("+", "_"), key)] = values[key]
    if any(np.asarray(v).dtype.kind not in "biuf" for v in arrays.values()):
        raise RuntimeError("Restricted archive may contain only numeric arrays")
    return arrays


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        raise RuntimeError("JSON cannot contain raw arrays or participant-level vectors")
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError("Unsupported JSON value: " + type(value).__name__)


def write_reports(audit: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    results = ROOT / "results"
    if not results.is_dir():
        raise RuntimeError("Required existing results directory is missing")
    npz_path = results / "v024_adni_oof_arrays.npz"
    json_path = results / "v024_adni_fusion_exposure_ecdf_audit.json"
    with tempfile.TemporaryDirectory(prefix=".v024_", dir=results) as tmp:
        stage_npz, stage_json = Path(tmp) / "archive.npz", Path(tmp) / "audit.json"
        fd = os.open(stage_npz, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(stage_npz.read_bytes()).hexdigest()
        audit["out_of_fold_archive"] = {"path": "results/v024_adni_oof_arrays.npz", "sha256": digest,
            "permissions": "0600", "restriction": "ADNI DUA-restricted participant-level data"}
        contents = json.dumps(_json_safe(audit), indent=2, allow_nan=False) + "\n"
        with stage_json.open("x", encoding="utf-8") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        # Each rename is atomic; the JSON is the last-published commit marker.
        # Preserve the previous archive if publishing the JSON encounters an error.
        backup = Path(tmp) / "previous.npz"
        had_previous = npz_path.exists()
        if had_previous:
            os.link(npz_path, backup)
        os.replace(stage_npz, npz_path)
        try:
            os.replace(stage_json, json_path)
        except BaseException:
            if had_previous:
                os.replace(backup, npz_path)
            else:
                npz_path.unlink(missing_ok=True)
            raise


def verify_replay(records: Dict[str, Any], y: np.ndarray, archived: Dict[str, np.ndarray]) -> Dict[str, Any]:
    checks = {}
    for name in OLD_NEURAL + TABULAR:
        key = "adni1_prob_" + name
        if key not in archived:
            checks[name] = {"archived_probabilities_present": False}
            continue
        a = compute_binary_metrics(np.asarray(archived[key]).ravel(), y)["auroc"]
        b = compute_binary_metrics(records[name]["clean"]["prob"], y)["auroc"]
        if not np.isfinite(a) or not np.isfinite(b) or abs(a - b) > 5e-4:
            raise RuntimeError("Archived clean OOF AUROC replay mismatch for %s: %s vs %s" % (name, a, b))
        checks[name] = {"archived_probabilities_present": True, "archived_auroc": a,
                        "replayed_auroc": b, "absolute_difference": abs(a - b)}
    return checks


def verify_dual_archive(y: np.ndarray, fids: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    path = ROOT / "results/tabpfn_plugin_oof_arrays.npz"
    info: Dict[str, Any] = {"archive_present": path.is_file(),
                            "audit_json_present": (ROOT / "results/tabpfn_deepfuzzy_plugin_audit.json").is_file()}
    if info["audit_json_present"]:
        with (ROOT / "results/tabpfn_deepfuzzy_plugin_audit.json").open("r", encoding="utf-8") as stream:
            json.load(stream)
    if not path.is_file():
        return None, info
    with np.load(path, allow_pickle=False) as archive:
        _check_equal("Dual-stream labels", y, archive["y"])
        _check_equal("Dual-stream fold IDs", fids, archive["fold_ids"])
        p = np.asarray(archive[DUAL + "_prob"], dtype=np.float32).ravel()
    auroc = compute_binary_metrics(p, y)["auroc"]
    if not np.isfinite(auroc) or abs(auroc - 0.8349) > 5e-4:
        raise RuntimeError("Archived dual-stream OOF AUROC is not within 5e-4 of 0.8349")
    info["archived_clean_auroc"] = auroc
    return p, info


def run(smoke: bool, skip_dual: bool) -> None:
    torch.set_num_threads(1 if smoke else 4)
    set_all_seeds(42)
    if smoke:
        c1, c2 = synthetic_cohort(11), synthetic_cohort(12)
        master, archived = {}, {}
        names = ["DeepFuzzyTorch", "TabPFN"] + NEW_NEURAL + MODDROP
        device = "cpu"
        dual_reference, dual_info = None, {"not_run_in_smoke": True}
        print("[Smoke] Synthetic tabular stand-ins exercise augmentation and scoring; optional model packages are not loaded.", flush=True)
    else:
        c1, c2, master, archived = load_real()
        names = MODELS
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dual_reference, dual_info = verify_dual_archive(c1["y"], c1["fids"])
    prepare_outer(c1, not smoke, 1)
    prepare_outer(c2, not smoke, 2)
    worker = None if smoke else _GBDTWorkerClient()
    try:
        r1 = run_cohort(c1, 1, names, worker, device, smoke)
        r2 = run_cohort(c2, 2, names, worker, device, smoke)
    finally:
        if worker is not None:
            worker.close()
    checks = {} if smoke else verify_replay(r1, c1["y"], archived)
    dual = None
    if not smoke and not skip_dual:
        dual = replay_dual(c1, device)
        if dual_reference is not None:
            replay_auc = compute_binary_metrics(dual["clean"]["prob"], c1["y"])["auroc"]
            reference_auc = compute_binary_metrics(dual_reference, c1["y"])["auroc"]
            if not np.isfinite(replay_auc) or abs(replay_auc - reference_auc) > 5e-4:
                raise RuntimeError("Dual-stream replay does not match archived AUROC")
            dual_info["replayed_clean_auroc"] = replay_auc
    elif skip_dual:
        dual_info["skipped_by_cli"] = True
    n_boot = 5 if smoke else 1000
    same_table = {name: subset_metrics(c1, rec, True) for name, rec in r1.items()}
    if dual is not None:
        same_table[DUAL] = subset_metrics(c1, dual, True)
    complete = c1["complete"]
    same = c1["same"]
    audit = {
        "configuration": {"outer_folds": len(c1["folds"]), "outer_seed": 42, "inner_folds": 3,
            "inner_seed": "3000 + outer fold", "inner_calibration": "patient-disjoint inner out-of-fold",
            "calibration_pool": "outer-training discordance using outer-training medians",
            "tabpfn_version": "V2", "tabpfn_outer_estimators": 8, "tabpfn_inner_estimators": 4,
            "tabpfn_device": device, "dual_stream_device": device, "training_drop_probability": 0.20,
            "augmentation": "one natural plus four independently masked copies", "bootstrap_replicates": n_boot},
        "cohorts": {"adni1": {"n": len(c1["y"]), "n_same_visit": int(same.sum()),
            "n_complete_4mod": int(complete.sum()), "n_same_visit_complete_4mod": int((same & complete).sum()),
            "n_same_visit_incomplete_4mod": int((same & ~complete).sum()),
            "n_other_visit_complete_4mod": int((~same & complete).sum()),
            "n_other_visit_incomplete_4mod": int((~same & ~complete).sum()),
            "n_eligible_conflict": int(((c1["full_mask"][:, 1] > 0) & (c1["full_mask"].sum(1) - c1["full_mask"][:, 1] >= 1)).sum())},
            "adni2": {"n": len(c2["y"])}, "folds_c1": c1["fold_metadata"], "folds_c2": c2["fold_metadata"]},
        "replay_anchors": checks, "dual_stream_anchor": dual_info,
        "c1_metrics": comparison_tables(c1, r1, n_boot),
        "c2_metrics": comparison_tables(c2, r2, n_boot),
        "c1_15_subset_full": {name: subset_metrics(c1, rec, False) for name, rec in r1.items()},
        "same_visit_bl_924_15_subset": same_table,
        "c1_gamma_0_8_conflict": {name: conflict_metrics(c1, rec) for name, rec in r1.items()},
        "padding_invariance": {name: dict(rec["padding"], fill_values=[0.0, -999.0, 777.0, "NaN"],
            outputs_checked=list(FIELDS), bitwise_equality_required=True) for name, rec in r1.items() if name in NEW_NEURAL + MODDROP},
        "max_batch_coupling_prob_diff": {name: r1[name]["coupling"] for name in ("MUSE_BatchGraph", "DeepFuzzyTorch")},
        "batch_coupling_note": "Maximum absolute float32 difference; sample-independent models can exhibit numerical roundoff when batch shapes change.",
        "deduplicated_discordant_matched_coverage": matched_coverage(c1, c2, r1, r2, n_boot, smoke),
    }
    if smoke:
        if len(same_table) != len(names) or any(len(row["subsets"]) != 15 for row in same_table.values()):
            raise RuntimeError("Smoke test did not exercise all same-visit subsets")
        for name in NEW_NEURAL + MODDROP:
            if r1[name]["padding"]["comparisons"] != len(c1["folds"]) * 180:
                raise RuntimeError("Smoke test padding checks incomplete for " + name)
        if not np.allclose(ecdf(np.array([0.0, 1.0, 1.0]), np.array([1.0])), [2.0 / 3.0]):
            raise RuntimeError("ECDF mid-rank calculation failed")
        # Exercise the degenerate training fallback and exact all-missing contract.
        p = fit_model("TabPFN_ModDrop", c1["outer"][0]["xt"], c1["outer"][0]["mt"],
            np.ones(len(c1["outer"][0]["xt"][0])), 1, 42, 42, "cpu", True, None, True)
        try:
            out = predict_np(p, c1["outer"][0]["xv"], np.zeros_like(c1["outer"][0]["mv"]))
            if not (np.all(out["prob"] == 0.5) and np.all(out["hesitancy"] == 1) and np.all(out["conflict"] == 0)):
                raise RuntimeError("Degenerate all-missing fallback failed")
        finally:
            release_model(p)
        _json_safe(audit)
        print("[SMOKE OK] A1, matched dropout exposure, same-visit 15 subsets, inner-OOF ECDF, padding, and degenerate inputs", flush=True)
        return
    if any(r1[n]["padding"]["comparisons"] != 900 for n in NEW_NEURAL + MODDROP):
        raise RuntimeError("Expected 900 padding comparisons per new model")
    arrays = _archive_arrays([c1, c2], [r1, r2], dual)
    write_reports(audit, arrays)
    print("[Done] Wrote results/v024_adni_fusion_exposure_ecdf_audit.json and restricted OOF archive", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Run a synthetic, no-results-write end-to-end check")
    parser.add_argument("--skip-dualstream-924", action="store_true", help="Skip the expensive dual-stream same-visit replay")
    args = parser.parse_args()
    run(args.smoke_test, args.skip_dualstream_924)


if __name__ == "__main__":
    main()
