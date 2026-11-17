#!/usr/bin/env python3
"""ADNI Cohort 1 TabPFN/DeepFuzzy plug-in audit.

Run from the Fuzzy repository root.  This script reads the existing ADNI data,
benchmark JSON, and NPZ archive; it never updates those files.  A successful
non-smoke run creates the two plug-in reports and a DUA-restricted OOF archive
in results/ and refuses to overwrite them.  --smoke-test uses synthetic data and
writes no reports.

Outer models use the archived 24 MRI indices.  To keep the stacked training
features genuinely out of sample, inner models refit preprocessing and select
MRI features using their own training patients.  Their unimodal OOF features
are themselves cross-fitted.  This nested work is intentionally substantial.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Importing repository modules must not create __pycache__ in the repository.
sys.dont_write_bytecode = True

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from deepfuzzy import (
    CombinatorialMissingModalityBenchmark,
    CrossModalityConflictInjector,
    DeepConceptTSKHead,
    FuzzyRoughFeatureReducer,
    IntuitionisticFuzzyLoss,
    UnifiedDeepFuzzyModel,
    bootstrap_confidence_interval,
    compute_binary_metrics,
    paired_bootstrap_pvalue,
)

MODALITY_NAMES = ["MRI", "Plasma", "FDG", "Clin"]
CONCEPT_NAMES = [
    "MRI_NetSupport", "Plasma_NetSupport", "FDG_NetSupport", "Clin_NetSupport",
    "APOE4", "MMSE", "ADAS13", "CDRSB",
]
VARIANT_1 = "UnimodalTabPFN_DeepFuzzyTorch"
VARIANT_2 = "DualStream_ModDropTabPFN_DeepFuzzyTorch"
AUDIT_NAME = "tabpfn_deepfuzzy_plugin_audit.json"
SUMMARY_NAME = "TABPFN_DEEPFUZZY_PLUGIN_SUMMARY.md"
OOF_ARCHIVE_NAME = "tabpfn_plugin_oof_arrays.npz"
OOF_ARCHIVE_NOTE = (
    "Participant-level ADNI OOF predictions and split IDs archived locally and withheld "
    "from public redistribution under the ADNI Data Use Agreement."
)


def require_finite(name: str, value: Any) -> None:
    if not np.isfinite(np.asarray(value)).all():
        raise RuntimeError(f"{name} contains NaN or infinity")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def clear_device_cache(device: str) -> None:
    gc.collect()
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def checked_split(train: np.ndarray, valid: np.ndarray, n: int, where: str) -> None:
    train = np.asarray(train, dtype=np.int64)
    valid = np.asarray(valid, dtype=np.int64)
    if (len(np.unique(train)) != len(train) or len(np.unique(valid)) != len(valid)
            or np.intersect1d(train, valid).size != 0):
        raise RuntimeError(f"{where}: training and validation indices overlap or repeat")
    if (np.any(train < 0) or np.any(valid < 0) or np.any(train >= n)
            or np.any(valid >= n) or len(train) + len(valid) != n):
        raise RuntimeError(f"{where}: split does not partition the patient set")


def splits3(y: np.ndarray, seed: int, where: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Three patient-disjoint folds, including small/one-class fallback cases."""
    yy = np.asarray(y, dtype=np.int64).ravel()
    if len(yy) < 3:
        return []
    counts = np.bincount(yy, minlength=2)
    if counts.max() < 3:
        splitter = KFold(n_splits=3, shuffle=True, random_state=seed)
        pairs = list(splitter.split(np.zeros(len(yy))))
    else:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        pairs = list(splitter.split(np.zeros(len(yy)), yy))
    seen = np.zeros(len(yy), dtype=np.int32)
    for tr, va in pairs:
        checked_split(tr, va, len(yy), where)
        seen[va] += 1
    if not np.all(seen == 1):
        raise RuntimeError(f"{where}: each patient must appear in validation exactly once")
    return pairs


def logit(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def clean_inputs(xs: Sequence[np.ndarray], mask: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape[1] != 4 or len(xs) != 4:
        raise ValueError("Expected four modalities and an N x 4 mask")
    require_finite("modality mask", mask)
    mask = (mask > 0.5).astype(np.float32)
    clean = []
    for m, x in enumerate(xs):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or len(x) != len(mask):
            raise ValueError(f"Invalid input shape for {MODALITY_NAMES[m]}")
        obs = mask[:, m] > 0.5
        require_finite(f"observed {MODALITY_NAMES[m]} values", x[obs])
        clean.append(np.where(obs[:, None], x, np.float32(0.0)).astype(np.float32))
    return clean, mask


def _build_tabular_matrix(xs: Sequence[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Phase-1 tabular representation: masked NaNs plus observation bits."""
    clean, mask = clean_inputs(xs, mask)
    parts = [np.where(mask[:, m:m + 1] > 0.5, clean[m], np.nan).astype(np.float32)
             for m in range(4)]
    mat = np.concatenate(parts + [mask], axis=1)
    for m, x in enumerate(clean):
        start = sum(a.shape[1] for a in clean[:m])
        part = mat[:, start:start + x.shape[1]]
        if not np.isfinite(part[mask[:, m] > 0.5]).all():
            raise RuntimeError(f"Observed {MODALITY_NAMES[m]} features are nonfinite")
        if not np.isnan(part[mask[:, m] <= 0.5]).all():
            raise RuntimeError(f"Missing {MODALITY_NAMES[m]} features must be NaN in TabPFN matrix")
    return mat


class ConstantProbability:
    def __init__(self, rate: float):
        self.rate = float(rate)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(len(x), self.rate, dtype=np.float64)
        return np.column_stack([1.0 - p, p])


def fit_pfn(x: np.ndarray, y: np.ndarray, n_estimators: int, device: str, seed: int):
    yy = np.asarray(y, dtype=np.int64).ravel()
    if len(yy) != len(x):
        raise ValueError("TabPFN training features and labels have different lengths")
    if len(yy) == 0 or len(np.unique(yy)) < 2:
        # An empty inner-training set has no estimable rate.  Never use its validation labels.
        return ConstantProbability(float(yy.mean()) if len(yy) else 0.5)
    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device=device, n_estimators=n_estimators,
        fit_mode="fit_with_cache", random_state=seed,
    )
    clf.fit(x, yy)
    return clf


def predict_pfn(clf: Any, x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.empty(0, dtype=np.float32)
    probs = np.asarray(clf.predict_proba(x))
    if probs.shape != (len(x), 2):
        raise RuntimeError(f"Unexpected TabPFN probability shape: {probs.shape}")
    p = probs[:, 1].astype(np.float32)
    require_finite("TabPFN probabilities", p)
    if np.any((p < 0.0) | (p > 1.0)):
        raise RuntimeError("TabPFN produced probabilities outside [0, 1]")
    return p


def choose_inner_mri(raw_mri: np.ndarray, y: np.ndarray, seed: int, preprocessor: Any) -> np.ndarray:
    """Fit MRI selection without consulting the corresponding inner holdout."""
    d = raw_mri.shape[1]
    if d <= 24:
        return np.arange(d, dtype=np.int64)
    # Selection needs only the training partition; keep the expensive stability
    # audit out of nested cross-fitting.  Outer selection is always the archive.
    tr, _, _, _ = preprocessor([raw_mri], [raw_mri[:0]])
    require_finite("inner MRI preprocessing", tr[0])
    reducer = FuzzyRoughFeatureReducer(
        max_features=24, sigma=0.85, max_samples=4096,
        n_null_permutations=0, n_stability_bootstraps=1, seed=seed,
    )
    reducer.fit(tr[0], y, device="cpu")
    idx = np.asarray(reducer.selected_indices_[:24], dtype=np.int64)
    if len(idx) != 24 or len(np.unique(idx)) != 24 or np.any(idx < 0) or np.any(idx >= d):
        raise RuntimeError("Inner MRI selector did not return 24 valid distinct features")
    return idx


def selected_raw(raw: Sequence[np.ndarray], idx: np.ndarray) -> List[np.ndarray]:
    return [np.asarray(raw[0])[:, idx]] + [np.asarray(a) for a in raw[1:]]


def preprocess_pair(raw_tr: Sequence[np.ndarray], raw_va: Sequence[np.ndarray],
                    idx: np.ndarray, preprocessor: Any):
    x_tr, x_va, m_tr, m_va = preprocessor(selected_raw(raw_tr, idx), selected_raw(raw_va, idx))
    x_tr, m_tr = clean_inputs(x_tr, m_tr)
    x_va, m_va = clean_inputs(x_va, m_va)
    # The OOF split is restricted to observed MRI rows.  Its membership must
    # not depend on a selector fitted with those same patients' labels.
    for raw, mask in ((raw_tr, m_tr), (raw_va, m_va)):
        independent = (~np.isnan(raw[0])).mean(axis=1) >= 0.5
        if not np.array_equal(independent, mask[:, 0] > 0.5):
            raise RuntimeError("MRI feature selection changes the observed-row mask; "
                               "patient-disjoint unimodal OOF cannot be guaranteed")
    return x_tr, x_va, m_tr, m_va


def oof_unimodal(raw_tr: Sequence[np.ndarray], x_tr: Sequence[np.ndarray],
                  m_tr: np.ndarray, y: np.ndarray, seed: int, n_estimators: int,
                  device: str, preprocessor: Any) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    n = len(y)
    out, audits = [], []
    for m in range(4):
        obs = np.flatnonzero(m_tr[:, m] > 0.5)
        p = np.full(n, np.nan, dtype=np.float32)
        pair_list = splits3(y[obs], seed + 101 * (m + 1), f"unimodal {m}")
        fallbacks = 0
        if len(obs) and not pair_list:
            # Each observed patient is predicted from all other observed patients.
            for j, patient in enumerate(obs):
                others = np.delete(obs, j)
                p[patient] = float(np.mean(y[others])) if len(others) else 0.5
                fallbacks += 1
        for inner_i, (it, iv) in enumerate(pair_list):
            tr_ids, va_ids = obs[it], obs[iv]
            if np.intersect1d(tr_ids, va_ids).size:
                raise RuntimeError("Unimodal inner training/validation overlap")
            if len(np.unique(y[tr_ids])) < 2:
                p[va_ids] = float(np.mean(y[tr_ids])) if len(tr_ids) else 0.5
                fallbacks += 1
                continue
            if m == 0:
                inner_sel = choose_inner_mri(raw_tr[0][tr_ids], y[tr_ids],
                                             seed + 1000 + inner_i, preprocessor)
                raw_a = raw_tr[0][tr_ids][:, inner_sel]
                raw_b = raw_tr[0][va_ids][:, inner_sel]
            else:
                raw_a, raw_b = raw_tr[m][tr_ids], raw_tr[m][va_ids]
            aa, bb, ma, mb = preprocessor([raw_a], [raw_b])
            if m == 0 and (not np.all(ma[:, 0] > 0.5) or not np.all(mb[:, 0] > 0.5)):
                raise RuntimeError("Nested MRI selection changed an observed-row mask")
            require_finite("unimodal inner train", aa[0])
            require_finite("unimodal inner validation", bb[0])
            clf = fit_pfn(aa[0], y[tr_ids], n_estimators, device,
                          seed + 17 * m + inner_i)
            p[va_ids] = predict_pfn(clf, bb[0])
            del clf
        require_finite(f"{MODALITY_NAMES[m]} OOF probabilities", p[obs])
        p[m_tr[:, m] <= 0.5] = 0.5
        out.append(p)
        audits.append({"modality": MODALITY_NAMES[m], "n_observed": int(len(obs)),
                       "n_inner_folds": len(pair_list), "base_rate_fallbacks": fallbacks})
    return out, audits


def augmented(xs: Sequence[np.ndarray], mask: np.ndarray, uni: Sequence[np.ndarray],
              stats: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
    result = []
    for m in range(4):
        ze = np.column_stack([logit(uni[m]), entropy(uni[m])])
        ze = ((ze - stats[m][0]) / stats[m][1]).astype(np.float32)
        value = np.concatenate([ze, xs[m]], axis=1)
        value = np.where(mask[:, m:m + 1] > 0.5, value, np.float32(0.0)).astype(np.float32)
        require_finite(f"augmented {MODALITY_NAMES[m]}", value)
        result.append(value)
    return result


def train_deepfuzzy(xs: Sequence[np.ndarray], clin: np.ndarray, mask: np.ndarray,
                    y: np.ndarray, epochs: int, seed: int) -> UnifiedDeepFuzzyModel:
    set_seed(seed)
    model = UnifiedDeepFuzzyModel(
        modality_dims=[a.shape[1] for a in xs], embed_dim=40, num_classes=1,
        prior_vacuity=0.45, conflict_penalty=0.55, use_rough_bottleneck=True,
        use_feature_gates=True, use_tsk_head=True, num_tsk_concepts=8,
        num_tsk_rules=6, concept_names=CONCEPT_NAMES, dropout=0.18,
    )
    # Match the benchmark's supervised initialization, using only this training set.
    with torch.no_grad():
        rel = []
        centered = y - np.mean(y)
        for m, x in enumerate(xs):
            obs = mask[:, m] > 0.5
            yc = y[obs] - np.mean(y[obs]) if np.any(obs) else centered
            x_used = x[obs] if np.any(obs) else x
            corr = np.abs((x_used * yc[:, None]).mean(axis=0))
            rel.append(float(np.mean(np.sort(corr)[-min(5, len(corr)):])))
            norm = (corr - corr.mean()) / (corr.std() + 1e-5)
            model.rough_gates[m].copy_(torch.tensor(norm * 0.45, dtype=torch.float32))
        rel = np.asarray(rel, dtype=np.float32)
        rel_z = (rel - rel.mean()) / (rel.std() + 1e-5)
        model.choquet_fusion.fuzzy_density_logits.copy_(torch.tensor(rel_z[:, None] * 0.85))
    criterion = IntuitionisticFuzzyLoss(
        lambda_dir=0.15, lambda_hes_cal=0.10, false_evidence_weight=0.20,
        modality_aux_weight=0.75, rough_boundary_weight=0.04,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt = [torch.tensor(x, dtype=torch.float32) for x in xs]
    ct = torch.tensor(clin, dtype=torch.float32)
    mt = torch.tensor(mask, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    n = len(y)
    batch_size = min(128, max(32, n // 6))
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            ids = order[start:start + batch_size]
            bm = mt[ids].clone()
            if ep > 5:
                draw = torch.tensor((rng.random(bm.shape) > 0.20).astype(np.float32))
                dropped = bm * draw
                dropped[dropped.sum(dim=1) < 0.5] = bm[dropped.sum(dim=1) < 0.5]
                bm = dropped
            bx = [xt[m][ids] * bm[:, m:m + 1] for m in range(4)]
            c_in = ct[ids] * bm[:, -1:]
            opt.zero_grad()
            out = model(bx, modality_masks=bm, clinical_concepts=c_in)
            weights = torch.ones_like(yt[ids])
            weights = weights / (weights.mean() + 1e-7)
            loss, _ = criterion(
                out["fused_ifn"], yt[ids], modality_ifns=out["modality_ifns"],
                modality_masks=bm, final_prob=out["prob"],
                rough_boundary=out["rough_boundary"], gate_sparsity_l1=out["gate_sparsity_l1"],
                sample_weights=weights,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite DeepFuzzy training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
    model.eval()
    return model.cpu()


class FittedUnimodalDeepFuzzy:
    """Fold model accepting unaugmented, already standardized modality arrays."""
    def __init__(self, model: UnifiedDeepFuzzyModel, uni_models: Sequence[Any],
                 stats: Sequence[Tuple[np.ndarray, np.ndarray]],
                 x_val: Sequence[np.ndarray], mask_val: np.ndarray,
                 x_train: Sequence[np.ndarray], mask_train: np.ndarray,
                 fit_audit: Dict[str, Any]):
        self.model = model
        self.uni_models = list(uni_models)
        self.stats = list(stats)
        self.clean_val = [np.array(a, dtype=np.float32, copy=True) for a in x_val]
        self.natural_val = np.array(mask_val, dtype=np.float32, copy=True)
        self.x_train = list(x_train)
        self.mask_train = mask_train
        self.fit_audit = fit_audit
        self.clean_uni: List[np.ndarray] = []
        for m in range(4):
            obs = self.natural_val[:, m] > 0.5
            p = np.full(len(mask_val), 0.5, dtype=np.float32)
            p[obs] = predict_pfn(self.uni_models[m], self.clean_val[m][obs])
            self.clean_uni.append(p)

    def probabilities(self, xs: Sequence[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
        out = []
        for m in range(4):
            obs = mask[:, m] > 0.5
            p = np.full(len(mask), 0.5, dtype=np.float32)
            # A cache hit requires the same patient position, an observed natural
            # modality, and exactly the clean observed values.  Changed inputs go
            # through live inference; unavailable fills are never inspected.
            if len(mask) == len(self.natural_val):
                same = (self.natural_val[:, m] > 0.5) & np.all(xs[m] == self.clean_val[m], axis=1)
                hit = obs & same
                p[hit] = self.clean_uni[m][hit]
            else:
                hit = np.zeros(len(mask), dtype=bool)
            live = obs & ~hit
            if np.any(live):
                p[live] = predict_pfn(self.uni_models[m], xs[m][live])
            out.append(p)
        return out

    @torch.no_grad()
    def predict(self, xs: Sequence[np.ndarray], mask: np.ndarray) -> Dict[str, Any]:
        xs, mask = clean_inputs(xs, mask)
        p_uni = self.probabilities(xs, mask)
        aug = augmented(xs, mask, p_uni, self.stats)
        tensors = [torch.from_numpy(a) for a in aug]
        mt = torch.from_numpy(mask)
        # Crucial: clinical concepts are the original standardized clinical data,
        # never the two-column-augmented clinical branch.
        clin = torch.from_numpy(xs[-1]) * mt[:, -1:]
        out = self.model(tensors, modality_masks=mt, clinical_concepts=clin)
        no_obs = mt.sum(dim=1) == 0
        if bool(no_obs.any()):
            out["prob"][no_obs] = 0.5
            out["hesitancy"][no_obs] = 1.0
            out["conflict"][no_obs] = 0.0
        for key in ("prob", "hesitancy", "conflict", "tsk_logits", "z_concepts"):
            if not bool(torch.isfinite(out[key]).all()):
                raise RuntimeError(f"Nonfinite DeepFuzzy output: {key}")
        expected_concepts = clin[:, :4]
        if not torch.equal(out["z_concepts"][:, 4:8], expected_concepts):
            raise RuntimeError("Clinical TSK concepts lost APOE4/MMSE/ADAS13/CDRSB alignment")
        for m in range(4):
            missing = mt[:, m] <= 0.5
            if bool(missing.any()):
                ifn = out["modality_ifns"][m]
                if (not bool(torch.all(ifn.mu[missing] == 0))
                        or not bool(torch.all(ifn.nu[missing] == 0))
                        or not bool(torch.all(ifn.pi[missing] == 1))):
                    raise RuntimeError(f"Missing {MODALITY_NAMES[m]} IFN is not exactly vacuous")
                if not bool(torch.all(out["choquet_weights"][missing, m] == 0)):
                    raise RuntimeError(f"Missing {MODALITY_NAMES[m]} has nonzero Choquet weight")
        return out


def fit_unimodal_deepfuzzy(raw_tr: Sequence[np.ndarray], raw_va: Sequence[np.ndarray],
                           y_tr: np.ndarray, sel: np.ndarray, seed: int, epochs: int,
                           inner_estimators: int, outer_estimators: int, device: str,
                           preprocessor: Any) -> FittedUnimodalDeepFuzzy:
    x_tr, x_va, m_tr, m_va = preprocess_pair(raw_tr, raw_va, sel, preprocessor)
    uni_oof, oof_audit = oof_unimodal(raw_tr, x_tr, m_tr, y_tr, seed,
                                      inner_estimators, device, preprocessor)
    stats = []
    models = []
    for m in range(4):
        obs = m_tr[:, m] > 0.5
        ze = np.column_stack([logit(uni_oof[m][obs]), entropy(uni_oof[m][obs])])
        mean = ze.mean(axis=0) if len(ze) else np.zeros(2)
        std = ze.std(axis=0) if len(ze) else np.ones(2)
        std = np.where(std < 1e-6, 1.0, std)
        require_finite("unimodal feature statistics", np.concatenate([mean, std]))
        stats.append((mean, std))
        models.append(fit_pfn(x_tr[m][obs], y_tr[obs], outer_estimators,
                              device, seed + 2000 + m))
    aug_tr = augmented(x_tr, m_tr, uni_oof, stats)
    model = train_deepfuzzy(aug_tr, x_tr[-1], m_tr, y_tr, epochs, seed)
    return FittedUnimodalDeepFuzzy(
        model, models, stats, x_va, m_va, x_tr, m_tr,
        {"unimodal_oof": oof_audit, "unimodal_feature_stats": [
            {"mean": mean.tolist(), "std": std.tolist()} for mean, std in stats]},
    )


def view_randomness(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random((2, n, 4)), rng.random((2, n, 4))


def make_views(natural: np.ndarray, draw: np.ndarray, tie: np.ndarray) -> List[np.ndarray]:
    natural = (np.asarray(natural) > 0.5).astype(np.float32)
    if draw.shape != (2, len(natural), 4) or tie.shape != draw.shape:
        raise ValueError("Invalid modality-dropout random draws")
    views = [natural.copy()]
    for i in range(2):
        view = natural * (draw[i] >= 0.35).astype(np.float32)
        empty = (view.sum(axis=1) == 0) & (natural.sum(axis=1) > 0)
        if np.any(empty):
            scores = np.where(natural[empty] > 0, tie[i, empty], -1.0)
            rows = np.flatnonzero(empty)
            view[rows, np.argmax(scores, axis=1)] = 1.0
        views.append(view)
    clinical_dropped = natural.copy()
    clinical_dropped[natural[:, :3].sum(axis=1) > 0, 3] = 0.0
    views.append(clinical_dropped)
    for view in views:
        if np.any(view > natural) or np.any((natural.sum(axis=1) > 0) & (view.sum(axis=1) == 0)):
            raise RuntimeError("Dropout views must be nonempty subsets of observed modalities")
    return views


def clin_dropped_mask(natural: np.ndarray) -> np.ndarray:
    mask = np.array(natural, dtype=np.float32, copy=True)
    mask[mask[:, :3].sum(axis=1) > 0, 3] = 0.0
    return mask


def fit_joint(xs: Sequence[np.ndarray], natural: np.ndarray, y: np.ndarray,
              draws: np.ndarray, ties: np.ndarray, n_estimators: int,
              device: str, seed: int):
    views = make_views(natural, draws, ties)
    matrix = np.concatenate([_build_tabular_matrix(xs, view) for view in views], axis=0)
    y_views = np.tile(np.asarray(y, dtype=np.int64), 4)
    return fit_pfn(matrix, y_views, n_estimators, device, seed)


def joint_predict(clf: Any, xs: Sequence[np.ndarray], mask: np.ndarray) -> np.ndarray:
    xs, mask = clean_inputs(xs, mask)
    p = np.full(len(mask), 0.5, dtype=np.float32)
    active = mask.sum(axis=1) > 0
    if np.any(active):
        # Never ask TabPFN to evaluate a patient with no active modality.
        p[active] = predict_pfn(clf, _build_tabular_matrix([x[active] for x in xs], mask[active]))
    return p


def blender_features(df_prob: np.ndarray, joint_prob: np.ndarray,
                     conflict: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z_df, z_joint = logit(df_prob), logit(joint_prob)
    conflict = np.asarray(conflict, dtype=np.float64).ravel()
    features = np.column_stack([z_df, z_joint, (mask.sum(axis=1) / 4.0) * z_joint,
                                conflict * (z_df - z_joint)])
    require_finite("blender features", features)
    return features


class FittedDualStream:
    def __init__(self, base: FittedUnimodalDeepFuzzy, joint: Any, blender: Any):
        self.base, self.joint, self.blender = base, joint, blender
        self.joint_cache: Dict[bytes, np.float32] = {}

    def joint_prob(self, xs: Sequence[np.ndarray], mask: np.ndarray) -> np.ndarray:
        xs, mask = clean_inputs(xs, mask)
        p = np.full(len(mask), 0.5, dtype=np.float32)
        missing: Dict[bytes, List[int]] = {}
        for i in np.flatnonzero(mask.sum(axis=1) > 0):
            key = mask[i].tobytes() + b"".join(x[i].tobytes() for x in xs)
            if key in self.joint_cache:
                p[i] = self.joint_cache[key]
            else:
                missing.setdefault(key, []).append(int(i))
        if missing:
            keys = list(missing)
            ids = np.array([missing[k][0] for k in keys], dtype=np.int64)
            pred = joint_predict(self.joint, [x[ids] for x in xs], mask[ids])
            for key, value in zip(keys, pred):
                self.joint_cache[key] = value
                p[missing[key]] = value
        return p

    @torch.no_grad()
    def predict(self, xs: Sequence[np.ndarray], mask: np.ndarray) -> Dict[str, Any]:
        xs, mask = clean_inputs(xs, mask)
        base_out = self.base.predict(xs, mask)
        p_joint = self.joint_prob(xs, mask)
        feat = blender_features(base_out["prob"].numpy().ravel(), p_joint,
                                base_out["conflict"].numpy().ravel(), mask)
        p = np.asarray(self.blender.predict_proba(feat))[:, 1].astype(np.float32)
        p[mask.sum(axis=1) == 0] = 0.5
        require_finite("dual-stream probabilities", p)
        return {"prob": torch.from_numpy(p[:, None]),
                "hesitancy": base_out["hesitancy"], "conflict": base_out["conflict"]}


def fit_dual_stream(raw_tr: Sequence[np.ndarray], y_tr: np.ndarray,
                    outer_base: FittedUnimodalDeepFuzzy, seed: int, epochs: int,
                    inner_estimators: int, outer_estimators: int, device: str,
                    preprocessor: Any) -> Tuple[FittedDualStream, Dict[str, Any]]:
    n = len(y_tr)
    draws, ties = view_randomness(n, seed)
    features = np.full((2, n, 4), np.nan, dtype=np.float64)
    seen = np.zeros(n, dtype=np.int32)
    fold_audit = []
    pairs = splits3(y_tr, seed + 3001, "dual-stream meta OOF")
    if len(pairs) != 3:
        raise RuntimeError("Dual-stream training requires three patient-disjoint inner folds")
    for j, (tr, va) in enumerate(pairs):
        tr_raw = [x[tr] for x in raw_tr]
        va_raw = [x[va] for x in raw_tr]
        inner_sel = choose_inner_mri(tr_raw[0], y_tr[tr], seed + 4000 + j, preprocessor)
        meta_inner_est = 1
        meta_outer_est = min(2, inner_estimators)
        meta_epochs = 16 if epochs > 16 else epochs
        base = fit_unimodal_deepfuzzy(
            tr_raw, va_raw, y_tr[tr], inner_sel, seed + 5000 + j * 101,
            meta_epochs, meta_inner_est, meta_outer_est, device, preprocessor,
        )
        joint = fit_joint(base.x_train, base.mask_train, y_tr[tr], draws[:, tr], ties[:, tr],
                          meta_outer_est, device, seed + 6000 + j)
        natural = base.natural_val
        for view_i, vm in enumerate((natural, clin_dropped_mask(natural))):
            df_out = base.predict(base.clean_val, vm)
            jp = joint_predict(joint, base.clean_val, vm)
            features[view_i, va] = blender_features(
                df_out["prob"].numpy().ravel(), jp, df_out["conflict"].numpy().ravel(), vm)
        seen[va] += 1
        fold_audit.append({"inner_fold": j, "n_train_patients": len(tr),
                           "n_validation_patients": len(va),
                           "mri_selected_indices": inner_sel.tolist(),
                           "unimodal_oof": base.fit_audit["unimodal_oof"]})
        del base, joint
        clear_device_cache(device)
    if not np.all(seen == 1):
        raise RuntimeError("Each blender patient must receive exactly one held-out prediction per view")
    require_finite("patient-disjoint blender OOF features", features)
    blender = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                 random_state=seed)
    blender.fit(features.reshape(2 * n, 4), np.tile(y_tr.astype(np.int64), 2))
    joint = fit_joint(outer_base.x_train, outer_base.mask_train, y_tr, draws, ties,
                      outer_estimators, device, seed + 7000)
    return FittedDualStream(outer_base, joint, blender), {
        "meta_folds": fold_audit, "meta_rows": 2 * n,
        "views_per_joint_training_patient": 4, "drop_probability": 0.35,
        "blender_coefficients": blender.coef_.tolist(),
        "blender_intercept": blender.intercept_.tolist(),
    }


class FoldPredictAdapter:
    """Adapt a fold-level model returning numpy/torch outputs to tensor predict_fn(xs, mask)."""
    def __init__(self, model: Any):
        self.model = model

    @torch.no_grad()
    def __call__(self, modality_inputs_raw4_tensors: List[torch.Tensor],
                 mask_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        xs = [t.detach().cpu().numpy() for t in modality_inputs_raw4_tensors]
        mask = mask_tensor.detach().cpu().numpy()
        out = self.model.predict(xs, mask)
        return {k: out[k].detach().cpu() for k in ("prob", "hesitancy", "conflict")}


def batch_fingerprint(xs: Sequence[torch.Tensor], mask: torch.Tensor) -> bytes:
    mask_bin = (mask.detach().cpu() > 0.5).to(dtype=torch.float32).contiguous()
    h_parts = [mask_bin.numpy().tobytes()]
    for m, x in enumerate(xs):
        xm = torch.where(
            mask_bin[:, m:m + 1] > 0.5,
            x.detach().cpu().to(dtype=torch.float32),
            torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32),
        ).contiguous()
        h_parts.append(xm.numpy().tobytes())
    return b"|".join(h_parts)


class PrecomputedOOFDispatcher:
    """Serve exact fold-streamed OOF tensors matching Masked benchmark/injector calls."""
    def __init__(self, expected_n: int):
        self.expected_n = expected_n
        self.table: Dict[bytes, Dict[str, torch.Tensor]] = {}

    def register(self, xs: Sequence[torch.Tensor], mask: torch.Tensor,
                 out: Dict[str, torch.Tensor]) -> None:
        key = batch_fingerprint(xs, mask)
        self.table[key] = {k: out[k].detach().cpu().clone() for k in ("prob", "hesitancy", "conflict")}

    @torch.no_grad()
    def __call__(self, modality_inputs_raw4_tensors: List[torch.Tensor],
                 mask_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if len(modality_inputs_raw4_tensors) != 4 or len(mask_tensor) != self.expected_n:
            raise ValueError("Precomputed OOF dispatch requires full cohort order")
        key = batch_fingerprint(modality_inputs_raw4_tensors, mask_tensor)
        if key not in self.table:
            raise RuntimeError("Precomputed OOF cache miss for unexpected evaluation input")
        return self.table[key]


def validate_padding(predict_fn: Any, xs: List[torch.Tensor], natural: torch.Tensor,
                     benchmark: CombinatorialMissingModalityBenchmark) -> Dict[str, Any]:
    largest = 0.0
    checks = 0
    for label, bits, _ in benchmark.enumerate_subsets():
        mask = natural * torch.tensor(bits, dtype=natural.dtype).view(1, 4)
        observed = [torch.where(mask[:, m:m + 1] > 0.5, xs[m], torch.zeros_like(xs[m]))
                    for m in range(4)]
        reference = predict_fn(observed, mask)
        for fill in (0.0, -999.0, 777.0, float("nan")):
            filled = [torch.where(mask[:, m:m + 1] > 0.5, xs[m],
                                  torch.full_like(xs[m], fill)) for m in range(4)]
            test = predict_fn(filled, mask)
            for key in ("prob", "hesitancy", "conflict"):
                delta = float(torch.max(torch.abs(test[key] - reference[key])).item())
                if not np.isfinite(delta):
                    raise RuntimeError(f"Nonfinite padding difference: {label}, {key}")
                largest = max(largest, delta)
                test_bytes = test[key].detach().cpu().contiguous().numpy().view(np.uint8)
                ref_bytes = reference[key].detach().cpu().contiguous().numpy().view(np.uint8)
                if (delta > 1e-6 or delta != 0.0
                        or not np.array_equal(test_bytes, ref_bytes)):
                    raise RuntimeError(f"Padding invariance failed: {label}, {fill}, {key}, diff={delta}")
                checks += 1
    empty_mask = torch.zeros_like(natural)
    empty = [torch.full_like(x, float("nan")) for x in xs]
    vacuous = predict_fn(empty, empty_mask)
    if (not bool(torch.all(vacuous["prob"] == 0.5))
            or not bool(torch.all(vacuous["hesitancy"] == 1.0))
            or not bool(torch.all(vacuous["conflict"] == 0.0))):
        raise RuntimeError("All-missing patients must receive exact vacuous predictions")
    return {"max_padding_diff": largest, "fill_values": [0.0, -999.0, 777.0, "NaN"],
            "outputs_checked": ["prob", "hesitancy", "conflict"], "comparisons": checks,
            "threshold": 1e-6, "exact_zero_and_bitwise_equality_required": True,
            "all_modalities_missing_is_vacuous": True}


def get_metrics(p: np.ndarray, y: np.ndarray, h: np.ndarray, n_boot: int = 1000) -> Dict[str, Any]:
    require_finite("metric probabilities", p)
    require_finite("metric hesitancy", h)
    result = compute_binary_metrics(p, y, uncertainty=h)
    _, low, high = bootstrap_confidence_interval(p, y, "auroc", n_bootstraps=n_boot, seed=42)
    result["auroc_ci_95"] = [float(low), float(high)]
    return result


def evaluate_rules(base: FittedUnimodalDeepFuzzy) -> Dict[str, Any]:
    rules = base.model.tsk_head.export_clinical_rules(
        concept_names=CONCEPT_NAMES, gate_threshold=0.0, precision=8)
    out = base.predict(base.clean_val, base.natural_val)
    z = out["z_concepts"].numpy()
    cm = out["concept_mask"].numpy()
    logits = out["tsk_logits"].numpy().ravel()
    recon = DeepConceptTSKHead.evaluate_exported_rules_text(
        rules, z, concept_names=CONCEPT_NAMES, concept_mask=cm).ravel()
    require_finite("text rule reconstruction", recon)
    error = float(np.max(np.abs(logits - recon)))
    if error > 1e-5:
        raise RuntimeError(f"Eight-decimal TSK rule reconstruction error is too large: {error}")
    return {"outer_fold": 0, "n_validation": len(z), "precision_decimal_places": 8,
            "concept_names": CONCEPT_NAMES, "rules": rules,
            "reconstruction_max_abs_error": error}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def format_number(x: Any) -> str:
    if x is None or not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.4f}"


def summary_markdown(audit: Dict[str, Any]) -> str:
    info = audit["cohort"]
    lines = ["# TabPFN–DeepFuzzy plug-in audit", "",
             f"ADNI Cohort 1: N={info['n']}; same-visit reference n={info['n_same_visit']}; "
             f"complete four-modality n={info['n_complete']}. Five archived outer folds and "
             "archived fold-specific MRI indices are used.", "",
             "| Model | AUROC (95% CI) | AUPRC | Acc | F1 | Brier | ECE | AURC | "
             "Same-visit AUROC | Complete AUROC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in audit["models"].items():
        met = result["full_metrics"]
        ci = met["auroc_ci_95"]
        lines.append(f"| {name} | {format_number(met['auroc'])} "
                     f"({format_number(ci[0])}–{format_number(ci[1])}) | "
                     f"{format_number(met['auprc'])} | {format_number(met['acc'])} | "
                     f"{format_number(met['f1'])} | {format_number(met['brier'])} | "
                     f"{format_number(met['ece'])} | {format_number(met['aurc'])} | "
                     f"{format_number(result['same_visit_auroc'])} | "
                     f"{format_number(result['complete_auroc'])} |")
    lines += ["", "| Plug-in | p vs DeepFuzzyTorch | p vs TabPFN v2 | "
              "15-subset macro AUROC | 14-subset MRR | MaxPaddingDiff | "
              "Conflict-only detection AUROC γ=0.8 | 20% composite referral AUROC γ=0.8 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in (VARIANT_1, VARIANT_2):
        model = audit["models"][name]
        subset = model["combinatorial_missingness"]
        rec = next(r for r in model["conflict_sweep"]["sweep_records"]
                   if abs(r["severity_gamma"] - 0.8) < 1e-8)
        lines.append(f"| {name} | {format_number(model['paired_bootstrap_p_vs_DeepFuzzyTorch'])} | "
                     f"{format_number(model['paired_bootstrap_p_vs_TabPFN'])} | "
                     f"{format_number(subset['macro_avg_auroc'])} | "
                     f"{format_number(subset['missing_robustness_retention'])} | "
                     f"{model['padding_invariance']['max_padding_diff']:.1f} | "
                     f"{format_number(rec['conflict_detection_auroc_conflict_only'])} | "
                     f"{format_number(rec['referral_curve']['refer_20pct']['auroc'])} |")
    lines += ["", "| Modality subset | " + VARIANT_1 + " full / eligible AUROC | "
              + VARIANT_2 + " full / eligible AUROC |",
              "|---|---:|---:|"]
    s1 = audit["models"][VARIANT_1]["combinatorial_missingness"]["subsets"]
    s2 = audit["models"][VARIANT_2]["combinatorial_missingness"]["subsets"]
    for name in s1:
        lines.append(f"| {name} | {format_number(s1[name]['auroc'])} / "
                     f"{format_number(s1[name]['auroc_eligible'])} | "
                     f"{format_number(s2[name]['auroc'])} / "
                     f"{format_number(s2[name]['auroc_eligible'])} |")
    rules = audit["white_box_tsk"]
    lines += ["", f"Fold-0 TSK text reconstruction maximum absolute error: "
              f"{rules['reconstruction_max_abs_error']:.3g}.", "",
              "The blender uses patient-disjoint inner predictions for both streams. "
              "Inner preprocessing and MRI selection are refitted on inner training patients. "
              "The dual stream retains the first variant's hesitancy and conflict; its final "
              "probability is the logistic blend.", "",
              "The conflict experiment swaps observed plasma within outer validation folds; "
              "the referral score is 0.5 × (hesitancy + conflict). The JSON report includes "
              "all subset results, sweep records, fold audits, and six 8-decimal rules. "
              "Participant-level out-of-fold outputs are stored separately in DUA-restricted "
              "`results/tabpfn_plugin_oof_arrays.npz`."]
    return "\n".join(lines) + "\n"


def write_new_reports(results_dir: Path, audit: Dict[str, Any]) -> Dict[str, Any]:
    """Stage complete files, publish the JSON last, and roll back on failure."""
    paths = [results_dir / OOF_ARCHIVE_NAME, results_dir / SUMMARY_NAME,
             results_dir / AUDIT_NAME]
    for path in paths:
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing file: {path}")
    oof = audit["out_of_fold"]
    arrays = {name: np.asarray(oof[name]) for name in ("y", "fold_ids")}
    for variant in (VARIANT_1, VARIANT_2):
        for field in ("prob", "hesitancy", "conflict"):
            arrays[f"{variant}_{field}"] = np.asarray(oof["models"][variant][field])
    n = int(audit["cohort"]["n"])
    for name, array in arrays.items():
        if array.shape != (n,) or array.dtype.kind not in "biuf":
            raise ValueError(f"Invalid OOF array {name}: expected {n} numeric values")
        require_finite(f"OOF array {name}", array)

    # Temporary files share the destination filesystem, so each hard link is an
    # atomic, no-overwrite publication. The JSON is the final commit marker.
    with tempfile.TemporaryDirectory(prefix=".tabpfn_plugin_", dir=results_dir) as tmp:
        staged = [Path(tmp) / path.name for path in paths]
        fd = os.open(staged[0], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(staged[0].read_bytes()).hexdigest()
        safe = json_safe({key: value for key, value in audit.items() if key != "out_of_fold"})
        safe["out_of_fold_archive"] = {
            "path": f"results/{OOF_ARCHIVE_NAME}",
            "sha256": digest,
            "n_participants": n,
            "note": OOF_ARCHIVE_NOTE,
        }
        contents = (summary_markdown(safe), json.dumps(safe, indent=2, allow_nan=False) + "\n")
        for path, content in zip(staged[1:], contents):
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        created: List[Path] = []
        try:
            for source, destination in zip(staged, paths):
                os.link(source, destination)
                created.append(destination)
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
    return safe


def load_archives(root: Path, y: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    results = root / "results"
    paths = [results / "benchmark_results.json", results / "oof_predictions_and_splits.npz"]
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Required read-only benchmark archive is missing: {path}")
    with paths[0].open("r", encoding="utf-8") as f:
        master = json.load(f)
    with np.load(paths[1], allow_pickle=False) as npz:
        required = ["adni1_fold_ids", "adni1_y"]
        for name in ("DeepFuzzyTorch", "TabPFN"):
            required.extend([f"adni1_{kind}_{name}" for kind in ("prob", "hes", "conf")])
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise RuntimeError("Missing archived arrays (run the existing Phase-1 benchmark first): "
                               + ", ".join(missing))
        arrays = {key: np.array(npz[key], copy=True) for key in required}
        optional = ["adni1_rids"] + [f"adni1_fold{i}_mri_selected_idx" for i in range(5)]
        for key in optional:
            if key in npz.files:
                arrays[key] = np.array(npz[key], copy=True)
    if not np.array_equal(np.asarray(arrays["adni1_y"]).ravel(), y):
        raise RuntimeError("ADNI target ordering does not match the archived targets")
    for name in ("DeepFuzzyTorch", "TabPFN"):
        for kind in ("prob", "hes", "conf"):
            arr = np.asarray(arrays[f"adni1_{kind}_{name}"]).ravel()
            if len(arr) != len(y):
                raise RuntimeError(f"Archived {name} {kind} has wrong length")
            require_finite(f"archived {name} {kind}", arr)
        p = np.asarray(arrays[f"adni1_prob_{name}"]).ravel()
        if np.any((p < 0) | (p > 1)):
            raise RuntimeError(f"Archived {name} probabilities are outside [0, 1]")
    return master, arrays


def synthetic_data() -> Dict[str, Any]:
    rng = np.random.default_rng(2026)
    n = 60
    y = np.tile(np.array([0, 1], dtype=np.int64), n // 2)
    rng.shuffle(y)
    dims = [6, 2, 1, 8]
    xs = []
    for m, d in enumerate(dims):
        x = rng.normal(size=(n, d)).astype(np.float32)
        x[:, 0] += (y * 2 - 1).astype(np.float32) * (0.4 + 0.1 * m)
        missing = rng.random(n) < [0.08, 0.15, 0.20, 0.05][m]
        x[missing] = np.nan
        xs.append(x)
    # Ensure at least one observed modality for every synthetic patient.
    none = np.all(np.column_stack([np.isnan(x).all(axis=1) for x in xs]), axis=1)
    xs[-1][none] = 0.0
    return {"X_modalities": xs, "y_amyloid": y,
            "same_visit_bl_ref_mask": np.ones(n, dtype=bool),
            "modality_names": MODALITY_NAMES}


def run(smoke: bool) -> Optional[Dict[str, Any]]:
    root = Path.cwd()
    # The benchmark module owns the requested loader and preprocessing contract.
    # It also sets Torch threads on import, so restore the requested CPU setting.
    from benchmarks.run_full_benchmark import fit_strict_preprocessor, load_adni_cohort1, resolve_data_paths
    torch.set_num_threads(1)
    set_seed(42)
    tabpfn_dev = os.environ.get("TABPFN_DEVICE")
    if not tabpfn_dev:
        tabpfn_dev = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    epochs = 4 if smoke else 28
    inner_estimators = 1 if smoke else 2
    outer_estimators = 2 if smoke else 8
    n_folds = 2 if smoke else 5
    if smoke:
        data = synthetic_data()
        master: Dict[str, Any] = {}
        archived: Dict[str, np.ndarray] = {}
    else:
        results_dir = root / "results"
        for output in (results_dir / AUDIT_NAME, results_dir / SUMMARY_NAME,
                       results_dir / OOF_ARCHIVE_NAME):
            if output.exists():
                raise RuntimeError(f"Refusing to overwrite existing file: {output}")
        data_path = resolve_data_paths()["adni_dir"]
        data = load_adni_cohort1(data_path)
    raw = [np.asarray(x, dtype=np.float32) for x in data["X_modalities"]]
    y_raw = np.asarray(data["y_amyloid"]).ravel()
    require_finite("ADNI targets", y_raw)
    if not np.all(np.isin(y_raw, [0, 1])):
        raise ValueError("The cohort must have binary labels 0 and 1")
    y = y_raw.astype(np.int64)
    n = len(y)
    expected_dims = [6, 2, 1, 8] if smoke else [320, 2, 1, 17]
    if len(raw) != 4 or [a.shape for a in raw] != [(n, d) for d in expected_dims]:
        raise RuntimeError(f"Unexpected cohort or modality shapes; expected N x {expected_dims}")
    if not smoke and n != 1137:
        raise RuntimeError(f"Expected ADNI Cohort 1 N=1137, got {n}")
    if any(np.isinf(a).any() for a in raw):
        raise RuntimeError("Raw modalities contain infinity; NaNs alone denote missingness")
    same = np.asarray(data["same_visit_bl_ref_mask"], dtype=bool)
    if same.shape != (n,) or (not smoke and int(same.sum()) != 924):
        raise RuntimeError("Same-visit reference mask does not have the expected n=924")
    if not smoke:
        master, archived = load_archives(root, y)
        if "adni1_rids" in archived:
            if "rids" not in data or not np.array_equal(
                    np.asarray(data["rids"]).ravel(), np.asarray(archived["adni1_rids"]).ravel()):
                raise RuntimeError("ADNI patient ordering does not match archived patient IDs")
        try:
            selected = master["track1_adni_cohort1"]["rough_summary"]["per_fold_selected_indices"]
            main_table = master["track1_adni_cohort1"]["main_table"]
            for name in ("DeepFuzzyTorch", "TabPFN"):
                _ = main_table[name]["auroc"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Benchmark JSON lacks archived MRI selections or baseline metrics") from exc
    else:
        selected = [list(range(expected_dims[0])) for _ in range(n_folds)]
    if len(selected) != n_folds:
        raise RuntimeError("Archive must contain one MRI selection per outer fold")
    indices = []
    for fold, values in enumerate(selected):
        a = np.asarray(values)
        expected_k = expected_dims[0] if smoke else 24
        if (a.ndim != 1 or len(a) != expected_k or not np.issubdtype(a.dtype, np.integer)
                or len(np.unique(a)) != expected_k or np.any(a < 0) or np.any(a >= expected_dims[0])):
            raise RuntimeError(f"Invalid archived MRI indices for fold {fold}")
        npz_key = f"adni1_fold{fold}_mri_selected_idx"
        if npz_key in archived and not np.array_equal(a, archived[npz_key].ravel()):
            raise RuntimeError(f"JSON and NPZ MRI selections disagree for fold {fold}")
        indices.append(a.astype(np.int64))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(skf.split(raw[0], y))
    fids = np.full(n, -1, dtype=np.int32)
    seen = np.zeros(n, dtype=np.int32)
    for fold, (tr, va) in enumerate(folds):
        checked_split(tr, va, n, f"outer fold {fold}")
        fids[va] = fold
        seen[va] += 1
    if not np.all(seen == 1):
        raise RuntimeError("Outer validation folds do not cover every patient once")
    if not smoke:
        archive_fids = np.asarray(archived["adni1_fold_ids"]).ravel()
        if archive_fids.shape != (n,) or not np.array_equal(fids, archive_fids):
            raise RuntimeError("Generated StratifiedKFold IDs differ from the archive")
        for name in ("DeepFuzzyTorch", "TabPFN"):
            p = archived[f"adni1_prob_{name}"].ravel()
            au = compute_binary_metrics(p, y)["auroc"]
            if abs(float(main_table[name]["auroc"]) - au) > 5e-4:
                raise RuntimeError(f"Archived JSON and NPZ AUROC disagree for {name}")
    dims = [len(indices[0]), 2, 1, expected_dims[-1]]
    full_x = [torch.zeros(n, d, dtype=torch.float32) for d in dims]
    full_mask = torch.zeros(n, 4, dtype=torch.float32)
    for fold, (tr, va) in enumerate(folds):
        tr_raw, va_raw = [x[tr] for x in raw], [x[va] for x in raw]
        _, xv, mt, mv = preprocess_pair(tr_raw, va_raw, indices[fold], fit_strict_preprocessor)
        # The outer MRI mask defines the observed rows used for uni-OOF splitting.
        # If it depends on the supervised archived selection, an inner patient's
        # label could influence that partition.  Reject such data rather than leak.
        for ids, mask in ((tr, mt), (va, mv)):
            independent = ((~np.isnan(raw[0][ids])).mean(axis=1) >= 0.5)
            if not np.array_equal(independent, mask[:, 0] > 0.5):
                raise RuntimeError("Archived MRI selection changes observed-row masks; "
                                   "cannot promise patient-disjoint inner OOF features")
        for m in range(4):
            full_x[m][va] = torch.from_numpy(xv[m])
        full_mask[va] = torch.from_numpy(mv)
    complete = np.all(full_mask.numpy() > 0.5, axis=1)
    eligible = ((full_mask[:, 1] > 0.5) & (full_mask.sum(dim=1) - full_mask[:, 1] >= 1)).numpy()
    if not smoke and int(complete.sum()) != 854:
        raise RuntimeError(f"Expected 854 complete four-modality patients; got {complete.sum()}")
    if not smoke and int(eligible.sum()) != 872:
        raise RuntimeError(f"Expected 872 eligible plasma conflict recipients; got {eligible.sum()}")
    print(f"[Plugin] N={n}, folds={n_folds}, TabPFN device={tabpfn_dev}, epochs={epochs}", flush=True)
    bench = CombinatorialMissingModalityBenchmark(MODALITY_NAMES)
    injector = CrossModalityConflictInjector(4, MODALITY_NAMES)
    severities = [0.8] if smoke else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Precompute all full-N masked subset and conflict inputs so every fold predicts
    # its held-out slice immediately and releases MPS KV caches before the next fold.
    subset_inputs: List[Tuple[str, List[torch.Tensor], torch.Tensor]] = []
    for sub_label, bits, _ in bench.enumerate_subsets():
        bit_t = torch.tensor(bits, dtype=full_mask.dtype).view(1, 4).expand(n, -1)
        eff_m = (full_mask > 0.5).to(dtype=full_mask.dtype) * bit_t
        sub_xs = [
            torch.where(eff_m[:, m:m + 1] > 0.5, full_x[m], torch.zeros_like(full_x[m]))
            for m in range(4)
        ]
        subset_inputs.append((sub_label, sub_xs, eff_m))

    conflict_inputs: List[Tuple[float, List[torch.Tensor]]] = []
    for gamma in severities:
        corr_xs, _ = injector.inject_conflict(
            modality_tensors=full_x, targets=y, corrupted_modality_idx=1,
            conflict_fraction=0.30 if gamma > 0 else 0.0, severity_gamma=gamma,
            mode="opposite_swap", natural_masks=full_mask, fold_ids=fids, seed=42,
        )
        conflict_inputs.append((gamma, corr_xs))

    def _empty_bufs() -> Dict[str, Dict[str, torch.Tensor]]:
        bufs: Dict[str, Dict[str, torch.Tensor]] = {
            "clean": {k: torch.zeros(n, 1) for k in ("prob", "hesitancy", "conflict")}
        }
        for sub_label, _, _ in subset_inputs:
            bufs[f"sub:{sub_label}"] = {k: torch.zeros(n, 1) for k in ("prob", "hesitancy", "conflict")}
        for gamma, _ in conflict_inputs:
            bufs[f"conf:{gamma}"] = {k: torch.zeros(n, 1) for k in ("prob", "hesitancy", "conflict")}
        return bufs

    bufs = {VARIANT_1: _empty_bufs(), VARIANT_2: _empty_bufs()}
    padding_stats = {
        VARIANT_1: {"max_diff": 0.0, "comparisons": 0},
        VARIANT_2: {"max_diff": 0.0, "comparisons": 0},
    }
    rules: Optional[Dict[str, Any]] = None
    fold_audits = []

    def _eval_fold_model(v_name: str, fold_model: Any, va_idx: np.ndarray) -> None:
        adapter = FoldPredictAdapter(fold_model)
        clean_out = adapter([full_x[m][va_idx] for m in range(4)], full_mask[va_idx])
        for k in ("prob", "hesitancy", "conflict"):
            bufs[v_name]["clean"][k][va_idx] = clean_out[k].reshape(-1, 1)
        for sub_label, sub_xs, eff_m in subset_inputs:
            s_out = adapter([sub_xs[m][va_idx] for m in range(4)], eff_m[va_idx])
            for k in ("prob", "hesitancy", "conflict"):
                bufs[v_name][f"sub:{sub_label}"][k][va_idx] = s_out[k].reshape(-1, 1)
        pad_res = validate_padding(adapter, [full_x[m][va_idx] for m in range(4)], full_mask[va_idx], bench)
        padding_stats[v_name]["max_diff"] = max(padding_stats[v_name]["max_diff"], float(pad_res["max_padding_diff"]))
        padding_stats[v_name]["comparisons"] += int(pad_res["comparisons"])
        for gamma, corr_xs in conflict_inputs:
            c_out = adapter([corr_xs[m][va_idx] for m in range(4)], full_mask[va_idx])
            for k in ("prob", "hesitancy", "conflict"):
                bufs[v_name][f"conf:{gamma}"][k][va_idx] = c_out[k].reshape(-1, 1)

    for fold, (tr, va) in enumerate(folds):
        print(f"[Plugin] Outer fold {fold + 1}/{n_folds}: fitting & evaluating nested models", flush=True)
        tr_raw, va_raw = [x[tr] for x in raw], [x[va] for x in raw]
        base = fit_unimodal_deepfuzzy(
            tr_raw, va_raw, y[tr], indices[fold], 42 + fold, epochs,
            inner_estimators, outer_estimators, tabpfn_dev, fit_strict_preprocessor,
        )
        if not np.array_equal(base.natural_val, full_mask[va].numpy()):
            raise RuntimeError(f"Fold {fold} validation masks drifted during fitting")
        for m in range(4):
            if not np.array_equal(base.clean_val[m], full_x[m][va].numpy()):
                raise RuntimeError(f"Fold {fold} validation preprocessing drifted for modality {m}")
        if fold == 0:
            # A changed observed validation input must take the live unimodal path.
            found = False
            for m in range(4):
                observed = np.flatnonzero(base.natural_val[:, m] > 0.5)
                if len(observed):
                    changed = [x.copy() for x in base.clean_val]
                    row = int(observed[0])
                    changed[m][row, 0] += np.float32(0.75)
                    before = base.uni_models[m]

                    class CountCalls:
                        def __init__(self, clf: Any):
                            self.clf, self.calls = clf, 0

                        def predict_proba(self, x: np.ndarray) -> np.ndarray:
                            self.calls += 1
                            return self.clf.predict_proba(x)

                    counted = CountCalls(before)
                    base.uni_models[m] = counted
                    try:
                        base.predict(changed, base.natural_val)
                    finally:
                        base.uni_models[m] = before
                    if counted.calls != 1:
                        raise RuntimeError("Changed observed input did not trigger live unimodal prediction")
                    found = True
                    break
            if not found:
                raise RuntimeError("No observed modality available for live-inference check")
            rules = evaluate_rules(base)

        _eval_fold_model(VARIANT_1, base, va)
        dual, dual_audit = fit_dual_stream(
            tr_raw, y[tr], base, 42 + fold, epochs,
            inner_estimators, outer_estimators, tabpfn_dev, fit_strict_preprocessor,
        )
        _eval_fold_model(VARIANT_2, dual, va)
        fold_audits.append({"fold": fold, "n_train": len(tr), "n_validation": len(va),
                            "mri_selected_indices": indices[fold].tolist(),
                            "variant1": base.fit_audit, "variant2": dual_audit})
        del base, dual
        clear_device_cache(tabpfn_dev)

    if rules is None:
        raise RuntimeError("Fold 0 TSK rule evaluation did not execute")
    output1 = bufs[VARIANT_1]["clean"]
    output2 = bufs[VARIANT_2]["clean"]
    for key in ("hesitancy", "conflict"):
        if not torch.equal(output1[key], output2[key]):
            raise RuntimeError(f"Dual stream did not retain Variant 1 {key} exactly")

    dispatchers: Dict[str, PrecomputedOOFDispatcher] = {}
    for v_name in (VARIANT_1, VARIANT_2):
        disp = PrecomputedOOFDispatcher(n)
        disp.register(full_x, full_mask, bufs[v_name]["clean"])
        for sub_label, sub_xs, eff_m in subset_inputs:
            disp.register(sub_xs, eff_m, bufs[v_name][f"sub:{sub_label}"])
        for gamma, corr_xs in conflict_inputs:
            disp.register(corr_xs, full_mask, bufs[v_name][f"conf:{gamma}"])
        dispatchers[v_name] = disp

    pred1, pred2 = dispatchers[VARIANT_1], dispatchers[VARIANT_2]
    model_results: Dict[str, Any] = {}
    patient_outputs: Dict[str, Any] = {}
    if not smoke:
        for name in ("DeepFuzzyTorch", "TabPFN"):
            p = archived[f"adni1_prob_{name}"].ravel()
            h = archived[f"adni1_hes_{name}"].ravel()
            track = master["track1_adni_cohort1"]
            model_results[name] = {
                "source": "read-only archived benchmark",
                "full_metrics": get_metrics(p, y, h),
                "same_visit_auroc": compute_binary_metrics(p[same], y[same])["auroc"],
                "complete_auroc": compute_binary_metrics(p[complete], y[complete])["auroc"],
                "archived_json_main_metrics": track["main_table"][name],
                "archived_json_combinatorial_missingness": track.get("combinatorial_15subsets", {}).get(name),
                "archived_json_conflict_sweep": track.get("conflict_results", {}).get(name),
            }
    for name, pred, output in ((VARIANT_1, pred1, output1), (VARIANT_2, pred2, output2)):
        print(f"[Plugin] Evaluating {name}: 15 subsets, padding, conflict", flush=True)
        p = output["prob"].numpy().ravel()
        h = output["hesitancy"].numpy().ravel()
        c = output["conflict"].numpy().ravel()
        subset = bench.evaluate_all_subsets(pred, full_x, full_mask, y)
        if len(subset["subsets"]) != 15:
            raise RuntimeError("Missing-modality benchmark did not produce all 15 subsets")
        padding = {
            "max_padding_diff": padding_stats[name]["max_diff"],
            "fill_values": [0.0, -999.0, 777.0, "NaN"],
            "outputs_checked": ["prob", "hesitancy", "conflict"],
            "comparisons": padding_stats[name]["comparisons"],
            "threshold": 1e-6,
            "exact_zero_and_bitwise_equality_required": True,
            "all_modalities_missing_is_vacuous": True,
        }
        conflict = injector.evaluate_conflict_sweep(
            predict_fn=pred, modality_tensors=full_x, natural_masks=full_mask, targets=y,
            corrupted_modality_idx=1, severities=severities,
            referral_rates=[0.0, 0.2] if smoke else None,
            fold_ids=fids, conflict_fraction=0.30, seed=42,
        )
        rec08 = next(r for r in conflict["sweep_records"] if abs(r["severity_gamma"] - 0.8) < 1e-8)
        if rec08["n_eligible_recipients"] != int(eligible.sum()):
            raise RuntimeError("Conflict eligible recipient count does not match natural masks")
        result = {"full_metrics": get_metrics(p, y, h, n_boot=50 if smoke else 1000),
                  "same_visit_auroc": compute_binary_metrics(p[same], y[same])["auroc"],
                  "complete_auroc": compute_binary_metrics(p[complete], y[complete])["auroc"],
                  "combinatorial_missingness": subset, "padding_invariance": padding,
                  "conflict_sweep": conflict}
        if not smoke:
            for baseline in ("DeepFuzzyTorch", "TabPFN"):
                result[f"paired_bootstrap_p_vs_{baseline}"] = paired_bootstrap_pvalue(
                    p, archived[f"adni1_prob_{baseline}"].ravel(), y,
                    "auroc", n_bootstraps=1000, seed=42)
        model_results[name] = result
        patient_outputs[name] = {"prob": p, "hesitancy": h, "conflict": c}
    audit = {
        "cohort": {"n": n, "n_same_visit": int(same.sum()), "n_complete": int(complete.sum()),
                   "n_conflict_eligible": int(eligible.sum()), "modality_names": MODALITY_NAMES,
                   "raw_modality_dimensions": expected_dims, "outer_modality_dimensions": dims},
        "configuration": {"outer_folds": n_folds, "outer_seed": 42, "inner_folds": 3,
                          "epochs": epochs, "inner_estimators": inner_estimators,
                          "meta_inner_estimators": 1, "meta_outer_estimators": min(2, inner_estimators),
                          "meta_epochs": 16 if epochs > 16 else epochs,
                          "outer_estimators": outer_estimators, "tabpfn_version": "V2",
                          "fit_mode": "fit_with_cache", "tabpfn_device": tabpfn_dev,
                          "deepfuzzy_device": "cpu", "smoke_test": smoke,
                          "inner_mri_selection": "fold-local fuzzy-rough, 24 features, one view, zero null permutations",
                          "outer_mri_selection": "archived per_fold_selected_indices" if not smoke else "all six features",
                          "meta_features": ["logit_DeepFuzzy", "logit_joint", "observed_fraction_times_logit_joint",
                                            "conflict_times_logit_difference"],
                          "blender_views": ["natural", "clinical-dropped when another modality is observed"]},
        "integrity": {"outer_fold_alignment_verified": not smoke, "patient_disjoint_splits_verified": True,
                      "observed_nan_hygiene_verified": True, "clinical_concepts_alignment_verified": True,
                      "missing_vacuity_and_choquet_weights_verified": True,
                      "changed_observed_unimodal_input_uses_live_prediction": True,
                      "dual_stream_hesitancy_and_conflict_identical": True},
        "folds": fold_audits, "models": model_results, "white_box_tsk": rules,
        "out_of_fold": {"y": y, "fold_ids": fids, "models": patient_outputs},
    }
    if smoke:
        print("[SMOKE OK] Both variants completed; MaxPaddingDiff == 0.0", flush=True)
    else:
        audit = write_new_reports(root / "results", audit)
        print(f"[Plugin] Wrote results/{AUDIT_NAME}, results/{SUMMARY_NAME}, "
              f"and DUA-restricted results/{OOF_ARCHIVE_NAME}", flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Run the synthetic, no-report test")
    args = parser.parse_args()
    start = time.monotonic()
    run(args.smoke_test)
    print(f"[Plugin] Elapsed: {time.monotonic() - start:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
