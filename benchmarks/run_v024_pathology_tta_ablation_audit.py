#!/usr/bin/env python3
"""Track-5 pathology TTA ablations, with source-only fitting and target-only shifts.

The default run requires the two benchmark datasets under data_benchmark/ and
writes only results/v024_pathology_tta_ablation_audit.json.  --smoke-test runs
the same models and corruption paths on synthetic data and writes no results.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from deepfuzzy.metrics import compute_binary_metrics
from deepfuzzy.operators import HesitantFuzzyAggregator, PythagoreanFuzzyTTALoss


FM_NAMES = ("uni2_h", "phikon_v2", "kaiko_vitl14", "phikon", "ctranspath", "plip")
N_FM = len(FM_NAMES)
SEED = 42
N_BOOTSTRAPS = 1000
N_TTA_STEPS = 25
ENTROPY_LIMIT = 0.4 * math.log(2.0)
SAR_RHO = 0.05
SAR_RECOVERY_ENTROPY = 0.20
EMA_DECAY = 0.9
EATA_COSINE_LIMIT = 0.90
EATA_FISHER_WEIGHT = 1.0
REFERENCE = "PFS_TTA_Full"
BASELINES = (
    "BestSingleFM", "MeanPool6FMs", "MedianPool6FMs", "Tent_TTA", "SAR_TTA",
    "EATA_TTA", "HesitantFuzzy_NoTTA", REFERENCE,
)
ABLATIONS = (
    "Abl_LinearIFS_q1", "Abl_wo_DiscordanceReg", "Abl_wo_SourceAnchor",
    "Abl_wo_Diversity", "Abl_wo_ReliableMask",
)
REGIMES = (
    "clean_ood", "plip_invert_neg2x", "single_fm_invert_macro",
    "dual_fm_gauss_noise", "temp_scale_miscalibration", "single_fm_dead_zero_macro",
)
ANCHORS = {
    "Camelyon17": {REFERENCE: 0.9925, "Tent_TTA": 0.9893},
    "NCT-CRC": {REFERENCE: 0.9804, "Tent_TTA": 0.9561},
}


def set_all_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_labels(y: np.ndarray, description: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) == 0 or not np.isfinite(y).all() or set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError(f"{description} must contain finite binary labels of both classes")
    return y


def load_dataset(dataset: str) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """Open an NPZ and identify the exact, disjoint Track-5 source/target splits."""
    data_dir = WORKSPACE / "data_benchmark" / "wsi_data" / "multi_fm_features"
    if dataset == "Camelyon17":
        metadata_path = data_dir / "camelyon17_metadata.csv"
        npz_path = data_dir / "camelyon17_6fms.npz"
        meta = pd.read_csv(metadata_path)
        y = np.asarray(meta["label"], dtype=np.float32)
        hospitals = sorted(meta["hospital"].unique())
        source = meta["hospital"].isin(hospitals[:3]).to_numpy()
        target = meta["hospital"].isin(hospitals[3:]).to_numpy()
        expected = (2500, 1500, 1000)
    elif dataset == "NCT-CRC":
        metadata_path = data_dir / "nct_crc_cross_cohort_metadata.csv"
        npz_path = data_dir / "nct_crc_cross_cohort_6fms.npz"
        meta = pd.read_csv(metadata_path)
        y = np.asarray(meta["is_tumor_or_stroma"], dtype=np.float32)
        source = (meta["cohort"] == "NCT_CRC_HE_Discovery").to_numpy()
        target = (meta["cohort"] == "CRC_VAL_HE_7K_External").to_numpy()
        expected = (2349, 999, 1350)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    actual = (len(y), int(source.sum()), int(target.sum()))
    if actual != expected or np.any(source & target) or not np.all(source | target):
        raise ValueError(f"{dataset}: unexpected total/source/target counts: {actual}, expected {expected}")
    _validate_labels(y[source], f"{dataset} source")
    _validate_labels(y[target], f"{dataset} target")
    return np.load(npz_path, allow_pickle=True), y, source, target


def synthetic_dataset() -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """120 cases, including 80 source cases; 68 features exercise the PCA branch."""
    rng = np.random.default_rng(SEED)
    y = np.tile(np.array([0.0, 1.0], dtype=np.float32), 60)
    source = np.arange(120) < 80
    target = ~source
    latent = 2.0 * y - 1.0
    shared = latent + rng.normal(0.0, 0.6, size=120)
    features = {}
    for k, fm in enumerate(FM_NAMES):
        x = rng.normal(0.0, 1.0, size=(120, 68))
        x[:, :12] += (0.40 + 0.035 * k) * shared[:, None]
        x[target, 12:24] += 0.25 * (k + 1) / N_FM
        features[fm] = x.astype(np.float32)
    return features, y, source, target


def fit_track5(
    features: object, y: np.ndarray, source: np.ndarray, target: np.ndarray
) -> Dict[str, object]:
    """Reproduce Track-5 standardization, OOF scoring, full fit, and HFS fit."""
    y_source = _validate_labels(y[source], "Source")
    y_target = _validate_labels(y[target], "Target")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = [np.zeros(len(y_source), dtype=np.float32) for _ in FM_NAMES]
    target_features = []
    classifiers = []
    source_aurocs = []
    dimensions = {}
    for k, fm in enumerate(FM_NAMES):
        x_all = np.asarray(features[fm]).astype(np.float32)
        if x_all.ndim != 2 or x_all.shape[0] != len(y) or not np.isfinite(x_all).all():
            raise ValueError(f"Invalid feature matrix for {fm}: {x_all.shape}")
        dimensions[fm] = int(x_all.shape[1])
        x_source, x_target = x_all[source], x_all[target]
        n_comp = min(64, x_source.shape[1])
        for fold, (s_tr, s_va) in enumerate(skf.split(x_source, y_source)):
            mu = x_source[s_tr].mean(axis=0, keepdims=True)
            sd = x_source[s_tr].std(axis=0, keepdims=True) + 1e-5
            x_tr = (x_source[s_tr] - mu) / sd
            x_va = (x_source[s_va] - mu) / sd
            if x_source.shape[1] > n_comp:
                pca = PCA(n_components=n_comp, random_state=SEED + fold)
                x_tr = pca.fit_transform(x_tr).astype(np.float32)
                x_va = pca.transform(x_va).astype(np.float32)
            clf = LogisticRegression(C=0.2, max_iter=300)
            clf.fit(x_tr, y_source[s_tr])
            oof[k][s_va] = clf.predict_proba(x_va)[:, 1]
        source_aurocs.append(float(compute_binary_metrics(oof[k], y_source)["auroc"]))
        mu = x_source.mean(axis=0, keepdims=True)
        sd = x_source.std(axis=0, keepdims=True) + 1e-5
        x_tr = (x_source - mu) / sd
        x_te = (x_target - mu) / sd
        if x_source.shape[1] > n_comp:
            pca = PCA(n_components=n_comp, random_state=SEED)
            x_tr = pca.fit_transform(x_tr).astype(np.float32)
            x_te = pca.transform(x_te).astype(np.float32)
        clf = LogisticRegression(C=0.2, max_iter=300)
        clf.fit(x_tr, y_source)
        target_features.append(x_te)
        classifiers.append(clf)

    set_all_seeds()
    aggregator = HesitantFuzzyAggregator(num_sources=N_FM, concordance_kappa=2.6)
    au_arr = np.asarray(source_aurocs, dtype=np.float32)
    rel_init = (au_arr - au_arr.mean()) / (au_arr.std() + 1e-4) * 0.25
    rel_init_t = torch.tensor(rel_init[None, :], dtype=torch.float32)
    with torch.no_grad():
        aggregator.prompt_reliability.copy_(rel_init_t)
        aggregator.certainty_temp.fill_(-2.5)
    oof_mat = np.stack(oof, axis=1)
    h_source = torch.tensor(oof_mat, dtype=torch.float32)
    y_source_t = torch.tensor(y_source, dtype=torch.float32).unsqueeze(-1)
    optimizer = torch.optim.Adam(aggregator.parameters(), lr=0.025)
    for _ in range(150):
        optimizer.zero_grad()
        mu_source = aggregator(h_source)["consensus_mu"].clamp(1e-6, 1.0 - 1e-6)
        loss = (
            F.binary_cross_entropy(mu_source, y_source_t)
            + 0.25 * (mu_source - y_source_t).pow(2).mean()
            + 0.25 * F.softplus(aggregator.certainty_temp).pow(2).mean()
            + 0.05 * (aggregator.prompt_reliability - rel_init_t).pow(2).mean()
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        source_hfs = aggregator(h_source)["consensus_mu"].view(-1).numpy()
        source_weights = F.softmax(aggregator.prompt_reliability, dim=1).view(-1).tolist()
    return {
        "target_features": target_features,
        "classifiers": classifiers,
        "source_oof": oof_mat,
        "y_source": y_source,
        "y_target": y_target,
        "source_aurocs": source_aurocs,
        "source_oof_hfs_auroc": float(compute_binary_metrics(source_hfs, y_source)["auroc"]),
        "selected_source_fm": FM_NAMES[int(np.argmax(source_aurocs))],
        "best_index": int(np.argmax(source_aurocs)),
        "fm_raw_dims": dimensions,
        "aggregator": aggregator,
        "source_weights": source_weights,
        "fisher": source_fisher(oof_mat, y_source),
    }


def source_fisher(oof: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Diagonal empirical Fisher: mean of per-case squared source BCE gradients."""
    probs = torch.tensor(oof, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
    logits = torch.logit(probs)
    p_ensemble = probs.mean(dim=1, keepdim=True).clamp(1e-6, 1.0 - 1e-6)
    target = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    bce_derivative = (p_ensemble - target) / (p_ensemble * (1.0 - p_ensemble))
    per_bias = bce_derivative * probs * (1.0 - probs) / N_FM
    per_scale = per_bias * logits
    return per_scale.pow(2).mean(dim=0, keepdim=True).detach(), per_bias.pow(2).mean(dim=0, keepdim=True).detach()


def target_probabilities(fit: Dict[str, object], regime: str, fm_index: Optional[int]) -> np.ndarray:
    features = fit["target_features"]
    classifiers = fit["classifiers"]
    predictions = []
    for k, fm in enumerate(FM_NAMES):
        x = features[k]
        if regime == "plip_invert_neg2x" and fm == "plip":
            x = -2.0 * x
        elif regime == "single_fm_invert_macro" and k == fm_index:
            x = -2.0 * x
        elif regime == "single_fm_dead_zero_macro" and k == fm_index:
            x = np.zeros_like(x)
        elif regime == "dual_fm_gauss_noise" and fm in ("plip", "ctranspath"):
            rng = np.random.default_rng(SEED + k)
            x = x + rng.normal(0.0, 3.0, size=x.shape)
        elif regime == "temp_scale_miscalibration":
            if fm == "uni2_h":
                x = 0.2 * x
            elif fm in ("plip", "phikon"):
                x = 5.0 * x
            elif fm == "kaiko_vitl14":
                x = x + 0.5
        predictions.append(classifiers[k].predict_proba(x)[:, 1])
    return np.stack(predictions, axis=1).astype(np.float32)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1.0 - 1e-6)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def _fm_dict(values: torch.Tensor) -> Dict[str, float]:
    return {fm: float(v) for fm, v in zip(FM_NAMES, values.detach().view(-1).tolist())}


def tent(logits: torch.Tensor) -> Tuple[np.ndarray, Dict[str, object]]:
    set_all_seeds()
    scale = nn.Parameter(torch.ones(1, N_FM))
    bias = nn.Parameter(torch.zeros(1, N_FM))
    opt = torch.optim.Adam([scale, bias], lr=0.03)
    for _ in range(N_TTA_STEPS):
        opt.zero_grad()
        ensemble = torch.sigmoid(logits * scale + bias).mean(dim=1).clamp(1e-6, 1.0 - 1e-6)
        anchor = 0.20 * ((scale - 1.0).pow(2).mean() + bias.pow(2).mean())
        (_entropy(ensemble).mean() + anchor).backward()
        opt.step()
    with torch.no_grad():
        pred = torch.sigmoid(logits * scale + bias).mean(dim=1).numpy().astype(np.float32)
    return pred, {"scale": _fm_dict(scale), "bias": _fm_dict(bias)}


def sar(logits: torch.Tensor) -> Tuple[np.ndarray, Dict[str, object]]:
    """Frozen-head SAR: SAM on mean-pool entropy and the Tent parameter anchor."""
    set_all_seeds()
    scale = nn.Parameter(torch.ones(1, N_FM))
    bias = nn.Parameter(torch.zeros(1, N_FM))
    params = [scale, bias]
    opt = torch.optim.Adam(params, lr=0.03)
    ema = None
    resets = 0
    selected_total = 0
    selected_second_total = 0
    empty_steps = 0

    def objective(first_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ent = _entropy(torch.sigmoid(logits * scale + bias).mean(dim=1))
        mask = (ent.detach() < ENTROPY_LIMIT)
        if first_mask is not None:
            mask = mask & first_mask
        # A batch with no reliable cases has no entropy contribution.
        ent_loss = ent[mask].mean() if bool(mask.any()) else ent.sum() * 0.0
        anchor = 0.20 * ((scale - 1.0).pow(2).mean() + bias.pow(2).mean())
        return ent_loss + anchor, ent[mask].detach(), mask

    for _ in range(N_TTA_STEPS):
        opt.zero_grad()
        first_loss, reliable, first_mask = objective()
        first_loss.backward()
        grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in params))
        perturbations = [SAR_RHO * p.grad.detach() / (grad_norm + 1e-12) for p in params]
        with torch.no_grad():
            for p, perturbation in zip(params, perturbations):
                p.add_(perturbation)
        opt.zero_grad()
        second_loss, reliable_second, _ = objective(first_mask)
        second_loss.backward()
        with torch.no_grad():
            for p, perturbation in zip(params, perturbations):
                p.sub_(perturbation)
        opt.step()
        selected_total += int(reliable.numel())
        selected_second_total += int(reliable_second.numel())
        if reliable_second.numel() == 0:
            empty_steps += 1
            continue
        value = float(reliable_second.mean())
        ema = value if ema is None else EMA_DECAY * ema + (1.0 - EMA_DECAY) * value
        if ema < SAR_RECOVERY_ENTROPY:
            with torch.no_grad():
                scale.fill_(1.0)
                bias.zero_()
            opt = torch.optim.Adam(params, lr=0.03)
            ema = None
            resets += 1
    with torch.no_grad():
        pred = torch.sigmoid(logits * scale + bias).mean(dim=1).numpy().astype(np.float32)
    return pred, {
        "scale": _fm_dict(scale), "bias": _fm_dict(bias), "recovery_resets": resets,
        "mean_reliable_count": selected_total / N_TTA_STEPS,
        "mean_second_pass_reliable_count": selected_second_total / N_TTA_STEPS,
        "empty_second_pass_steps": empty_steps,
        "final_entropy_ema": ema,
    }


def eata(logits: torch.Tensor, fisher: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[np.ndarray, Dict[str, object]]:
    """Frozen-head EATA with source OOF empirical Fisher and target diversity filtering."""
    set_all_seeds()
    scale = nn.Parameter(torch.ones(1, N_FM))
    bias = nn.Parameter(torch.zeros(1, N_FM))
    opt = torch.optim.Adam([scale, bias], lr=0.03)
    running_prob = None
    selected_total = 0
    empty_steps = 0
    for _ in range(N_TTA_STEPS):
        opt.zero_grad()
        ensemble = torch.sigmoid(logits * scale + bias).mean(dim=1)
        ent = _entropy(ensemble)
        probability_vectors = torch.stack((ensemble, 1.0 - ensemble), dim=1)
        mask = ent.detach() < ENTROPY_LIMIT
        if running_prob is not None:
            cosine = F.cosine_similarity(probability_vectors.detach(), running_prob.unsqueeze(0), dim=1)
            mask = mask & (cosine < EATA_COSINE_LIMIT)
        count = int(mask.sum())
        selected_total += count
        if count:
            weights = torch.exp(-(ent[mask].detach() - ENTROPY_LIMIT))
            entropy_loss = (weights * ent[mask]).mean()
            batch_prob = probability_vectors[mask].detach().mean(dim=0)
            running_prob = batch_prob if running_prob is None else EMA_DECAY * running_prob + (1.0 - EMA_DECAY) * batch_prob
        else:
            empty_steps += 1
            entropy_loss = ent.sum() * 0.0
        penalty = (fisher[0] * (scale - 1.0).pow(2)).sum() + (fisher[1] * bias.pow(2)).sum()
        (entropy_loss + EATA_FISHER_WEIGHT * penalty).backward()
        opt.step()
    with torch.no_grad():
        pred = torch.sigmoid(logits * scale + bias).mean(dim=1).numpy().astype(np.float32)
    return pred, {
        "scale": _fm_dict(scale), "bias": _fm_dict(bias),
        "mean_selected_count": selected_total / N_TTA_STEPS, "empty_selected_steps": empty_steps,
    }


def linear_ifs_tta_loss(ifn: object) -> torch.Tensor:
    """Track-5 preprocessing with the specified linear residual and diversity."""
    mu = torch.nan_to_num(ifn.mu, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    nu = torch.nan_to_num(ifn.nu, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    pi = torch.nan_to_num(ifn.pi, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    shrink = 1.0 - 0.5 * pi.clamp(0.0, 0.99)
    mu_p = (mu * shrink).clamp(0.0, 1.0)
    nu_p = (nu * shrink).clamp(0.0, 1.0)
    norm_sq = (mu_p.pow(2) + nu_p.pow(2)).clamp(min=0.0)
    denominator = torch.sqrt(norm_sq + 1e-7)
    mu_p = torch.where(norm_sq > 1.0, mu_p / denominator, mu_p)
    nu_p = torch.where(norm_sq > 1.0, nu_p / denominator, nu_p)
    q_ifs = (1.0 - mu_p - nu_p).clamp(0.0, 1.0)
    mask = (q_ifs <= 0.65).to(dtype=mu.dtype).detach()
    if mask.sum() < 1.0:
        mask = torch.ones_like(q_ifs)
    hes_loss = (mask * q_ifs).sum() / (mask.sum() + 1e-7)
    positive = (mu_p + 0.5 * q_ifs).clamp(1e-5, 1.0 - 1e-5).mean()
    diversity = positive * torch.log(positive) + (1.0 - positive) * torch.log(1.0 - positive)
    return hes_loss + 0.25 * diversity


def pfs_adapt(
    probabilities: torch.Tensor, logits: torch.Tensor, source_aggregator: HesitantFuzzyAggregator, method: str
) -> Tuple[np.ndarray, Dict[str, object]]:
    set_all_seeds()
    aggregator = HesitantFuzzyAggregator(num_sources=N_FM, concordance_kappa=2.6)
    aggregator.load_state_dict(source_aggregator.state_dict())
    scale = nn.Parameter(torch.ones(1, N_FM))
    threshold = 1.0 if method == "Abl_wo_ReliableMask" else 0.65
    diversity = 0.0 if method == "Abl_wo_Diversity" else 0.25
    loss_fn = PythagoreanFuzzyTTALoss(hesitancy_threshold=threshold, diversity_weight=diversity)
    opt = torch.optim.Adam([scale, aggregator.prompt_reliability], lr=0.04)
    median = probabilities.median(dim=1, keepdim=True).values.detach()
    discord = (probabilities - median).abs().mean(dim=0, keepdim=True).detach()
    rel_anchor = aggregator.prompt_reliability.detach().clone()
    for _ in range(N_TTA_STEPS):
        opt.zero_grad()
        calibrated = torch.sigmoid(logits * scale)
        out = aggregator(calibrated)
        tta_loss = linear_ifs_tta_loss(out["ifn"]) if method == "Abl_LinearIFS_q1" else loss_fn(out["ifn"])["tta_loss"]
        weights = F.softmax(aggregator.prompt_reliability, dim=1)
        concordance = (weights * discord).sum()
        concord_weight = 0.0 if method == "Abl_wo_DiscordanceReg" else 2.5
        anchor_weight = 0.0 if method == "Abl_wo_SourceAnchor" else 1.0
        anchor = 0.15 * (aggregator.prompt_reliability - rel_anchor).pow(2).mean() + 0.5 * (scale - 1.0).pow(2).mean()
        loss = 0.2 * tta_loss + concord_weight * concordance + anchor_weight * anchor
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = aggregator(torch.sigmoid(logits * scale))["consensus_mu"].view(-1).numpy().astype(np.float32)
        weights = F.softmax(aggregator.prompt_reliability, dim=1)
    return pred, {"scale": _fm_dict(scale), "reliability_weights": _fm_dict(weights),
                  "prompt_reliability": _fm_dict(aggregator.prompt_reliability)}


def run_methods(fit: Dict[str, object], regime: str, fm_index: Optional[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    mat = target_probabilities(fit, regime, fm_index)
    h_target = torch.tensor(mat, dtype=torch.float32)
    logits = torch.logit(h_target.clamp(1e-4, 1.0 - 1e-4))
    source_aggregator = fit["aggregator"]
    with torch.no_grad():
        hfs = source_aggregator(h_target)["consensus_mu"].view(-1).numpy().astype(np.float32)
    predictions = {
        "BestSingleFM": mat[:, fit["best_index"]].astype(np.float32),
        "MeanPool6FMs": np.mean(mat, axis=1).astype(np.float32),
        "MedianPool6FMs": np.median(mat, axis=1).astype(np.float32),
        "HesitantFuzzy_NoTTA": hfs,
    }
    parameters = {}
    predictions["Tent_TTA"], parameters["Tent_TTA"] = tent(logits)
    predictions["SAR_TTA"], parameters["SAR_TTA"] = sar(logits)
    predictions["EATA_TTA"], parameters["EATA_TTA"] = eata(logits, fit["fisher"])
    methods = (REFERENCE,) if regime == "single_fm_dead_zero_macro" else (REFERENCE,) + ABLATIONS
    for method in methods:
        predictions[method], parameters[method] = pfs_adapt(h_target, logits, source_aggregator, method)
    expected = set(BASELINES) if regime == "single_fm_dead_zero_macro" else set(BASELINES + ABLATIONS)
    if set(predictions) != expected:
        raise RuntimeError(f"{regime}: missing or unexpected methods")
    for name, prediction in predictions.items():
        if prediction.shape != (len(mat),) or not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
            raise RuntimeError(f"{regime}: invalid predictions from {name}")
    return predictions, parameters


def bootstrap_counts(y: np.ndarray) -> np.ndarray:
    """Match deepfuzzy.metrics' class-stratified seed-42 bootstrap draws."""
    negatives = np.flatnonzero(y == 0)
    positives = np.flatnonzero(y == 1)
    if not len(negatives) or not len(positives):
        raise ValueError("Both classes are required for AUROC bootstrap")
    rng = np.random.default_rng(SEED)
    counts = np.empty((N_BOOTSTRAPS, len(y)), dtype=np.int32)
    for b in range(N_BOOTSTRAPS):
        neg = rng.choice(negatives, size=len(negatives), replace=True)
        pos = rng.choice(positives, size=len(positives), replace=True)
        counts[b] = np.bincount(np.concatenate((neg, pos)), minlength=len(y))
    return counts


def bootstrap_aurocs(pred: np.ndarray, y: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Weighted rank AUROC, including exact half-credit for tied scores."""
    order = np.argsort(pred, kind="mergesort")
    ordered = pred[order]
    is_pos = (y[order] == 1)
    is_neg = ~is_pos
    starts = np.concatenate(([0], np.flatnonzero(ordered[1:] != ordered[:-1]) + 1))
    ends = np.concatenate((starts[1:] - 1, [len(y) - 1]))
    weights = counts[:, order]
    neg_cumulative = np.cumsum(weights * is_neg, axis=1, dtype=np.int32)[:, ends]
    neg_before = np.concatenate((np.zeros((len(counts), 1), dtype=np.int32), neg_cumulative[:, :-1]), axis=1)
    pos_groups = np.add.reduceat(weights * is_pos, starts, axis=1)
    concordant = (pos_groups * (neg_before + 0.5 * (neg_cumulative - neg_before))).sum(axis=1)
    return concordant / (int(is_pos.sum()) * int(is_neg.sum()))


def paired_pvalue(reference_point: float, point: float, reference_boot: np.ndarray, boot: np.ndarray) -> float:
    """Two-sided paired bootstrap, identical tie and floor conventions to Track 5."""
    tolerance = 1e-15
    if abs(reference_point - point) <= tolerance:
        return 1.0
    diffs = reference_boot - boot
    ties = 0.5 * float(np.sum(np.abs(diffs) <= tolerance))
    neg = float(np.sum(diffs < -tolerance)) + ties
    pos = float(np.sum(diffs > tolerance)) + ties
    return float(min(1.0, max(1.0 / (len(diffs) + 1.0), 2.0 * min(neg, pos) / len(diffs))))


def bh_fdr(pvalues: Dict[str, float]) -> Dict[str, float]:
    """BH across non-reference methods in one dataset/regime/comparison family."""
    ordered = sorted((p, name) for name, p in pvalues.items() if name != REFERENCE)
    output = {REFERENCE: 1.0}
    current = 1.0
    for i in range(len(ordered) - 1, -1, -1):
        p, name = ordered[i]
        current = min(current, p * len(ordered) / (i + 1), 1.0)
        output[name] = float(current)
    return output


def summarize_case(
    predictions: Dict[str, np.ndarray], parameters: Dict[str, object], y: np.ndarray, counts: np.ndarray
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    boots = {name: bootstrap_aurocs(pred, y, counts) for name, pred in predictions.items()}
    rows = {}
    for name, pred in predictions.items():
        all_metrics = compute_binary_metrics(pred, y)
        ci = np.percentile(boots[name], [2.5, 97.5])
        rows[name] = {key: float(all_metrics[key]) for key in ("auroc", "auprc", "brier", "ece")}
        rows[name]["auroc_ci"] = [float(ci[0]), float(ci[1])]
        if name in parameters:
            rows[name]["learned_tta_parameters"] = parameters[name]
    add_significance(rows, boots)
    return rows, boots


def add_significance(rows: Dict[str, object], boots: Dict[str, np.ndarray]) -> None:
    ref = rows[REFERENCE]["auroc"]
    pvalues = {name: paired_pvalue(ref, row["auroc"], boots[REFERENCE], boots[name]) for name, row in rows.items()}
    qvalues = bh_fdr(pvalues)
    for name, row in rows.items():
        row["p_value_vs_pfs_tta"] = pvalues[name]
        row["q_bh_vs_pfs_tta"] = qvalues[name]


def macro_summary(cases: Dict[str, object], bootstrap_cases: Dict[str, object]) -> Dict[str, object]:
    """Average case metrics and paired bootstrap AUROCs, not sample predictions."""
    methods = list(cases[FM_NAMES[0]])
    rows = {}
    boots = {}
    for name in methods:
        rows[name] = {
            key: float(np.mean([cases[fm][name][key] for fm in FM_NAMES]))
            for key in ("auroc", "auprc", "brier", "ece")
        }
        boots[name] = np.mean(np.stack([bootstrap_cases[fm][name] for fm in FM_NAMES]), axis=0)
        ci = np.percentile(boots[name], [2.5, 97.5])
        rows[name]["auroc_ci"] = [float(ci[0]), float(ci[1])]
    add_significance(rows, boots)
    return rows


def check_anchor(dataset: str, predictions: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, object]:
    checked = {}
    for method, expected in ANCHORS[dataset].items():
        actual = float(compute_binary_metrics(predictions[method], y)["auroc"])
        if abs(actual - expected) > 1e-4:
            raise RuntimeError(f"Track-5 anchor drift: {dataset} {method} AUROC {actual:.8f}, expected {expected:.4f} ± 0.0001")
        print(f"  anchor {method}: {actual:.6f} (expected {expected:.4f})", flush=True)
        checked[method] = {"actual_auroc": actual, "expected_auroc": expected, "absolute_tolerance": 1e-4}
    return checked


def audit_dataset(dataset: str, smoke: bool = False) -> Dict[str, object]:
    print(f"Fitting Track-5 source pipeline: {dataset}", flush=True)
    if smoke:
        features, y, source, target = synthetic_dataset()
    else:
        features, y, source, target = load_dataset(dataset)
    try:
        fit = fit_track5(features, y, source, target)
    finally:
        if not smoke:
            features.close()
    y_target = fit["y_target"]
    counts = bootstrap_counts(y_target)
    results = {}
    anchor_check = None
    # Anchor before spending time on the other regimes, and before any output file is written.
    run_order = ("plip_invert_neg2x",) + tuple(r for r in REGIMES if r != "plip_invert_neg2x")
    for regime in run_order:
        print(f"  {dataset}: {regime}", flush=True)
        macro = regime in ("single_fm_invert_macro", "single_fm_dead_zero_macro")
        if macro:
            cases = {}
            boots = {}
            for index, fm in enumerate(FM_NAMES):
                preds, params = run_methods(fit, regime, index)
                cases[fm], boots[fm] = summarize_case(preds, params, y_target, counts)
            results[regime] = {"per_fm": cases, "macro_average": macro_summary(cases, boots)}
        else:
            preds, params = run_methods(fit, regime, None)
            if regime == "plip_invert_neg2x" and not smoke:
                anchor_check = check_anchor(dataset, preds, y_target)
            rows, _ = summarize_case(preds, params, y_target, counts)
            results[regime] = {"methods": rows}
    if set(results) != set(REGIMES):
        raise RuntimeError(f"{dataset}: incomplete regime results")
    return {
        "n_source": int(source.sum()), "n_target": int(target.sum()),
        "fm_raw_dims": fit["fm_raw_dims"], "selected_source_fm": fit["selected_source_fm"],
        "source_cv_aurocs": dict(zip(FM_NAMES, fit["source_aurocs"])),
        "source_oof_hfs_auroc": fit["source_oof_hfs_auroc"],
        "source_hfs_reliability_weights": dict(zip(FM_NAMES, fit["source_weights"])),
        "source_hfs_certainty_temp": float(fit["aggregator"].certainty_temp.detach()),
        "source_oof_fisher_diagonal": {"scale": _fm_dict(fit["fisher"][0]), "bias": _fm_dict(fit["fisher"][1])},
        "plip_inversion_parity_anchor": anchor_check,
        "regimes": results,
    }


def configuration() -> Dict[str, object]:
    return {
        "seed": SEED, "foundation_models": list(FM_NAMES), "source_folds": 5,
        "pca": "min(64, raw_feature_dimension); bypass when raw dimension <= 64, as in Track 5",
        "logistic_regression": {"C": 0.2, "max_iter": 300},
        "source_hfs": {"steps": 150, "adam_lr": 0.025, "concordance_kappa": 2.6},
        "tta_steps": N_TTA_STEPS,
        "pfs": {"adam_lr": 0.04, "hesitancy_threshold": 0.65, "diversity_weight": 0.25},
        "tent_sar_eata_adam_lr": 0.03, "tent_sar_anchor": 0.20,
        "sar": {"rho": SAR_RHO, "entropy_limit": ENTROPY_LIMIT, "entropy_ema_decay": EMA_DECAY,
                "recovery_threshold": SAR_RECOVERY_ENTROPY, "readout": "mean",
                "second_pass": "re-filter first-pass reliable samples; recover using second-pass entropy EMA"},
        "eata": {"entropy_limit": ENTROPY_LIMIT, "cosine_limit": EATA_COSINE_LIMIT,
                 "probability_ema_decay": EMA_DECAY, "fisher_weight": EATA_FISHER_WEIGHT,
                 "fisher": "mean squared per-sample source OOF ensemble BCE gradients at scale=1, bias=0",
                 "entropy_weight": "exp(-(entropy - entropy_limit)), detached",
                 "readout": "mean"},
        "corruptions": {
            "dual_fm_gauss_noise": "N(0, 3^2) on ctranspath and plip; independent RNG seed 42 + FM index",
            "temp_scale_miscalibration": "uni2_h x0.2, phikon and plip x5, kaiko_vitl14 +0.5",
            "single_fm_dead_zero_macro": "zero target PCA features before logistic scoring; classifier intercept remains",
        },
        "bootstrap": {"replicates": N_BOOTSTRAPS, "seed": SEED, "interval": "class-stratified percentile 95% AUROC",
                      "p_value": "two-sided class-stratified paired AUROC bootstrap, matching Track 5"},
        "macro": "arithmetic mean across six per-FM metrics; bootstrap pairs the same target cases across all six shifts",
        "fdr_family": "non-reference methods within each dataset and regime (or per-FM breakdown); reference p=q=1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Run all paths on synthetic N=120 without data or output files")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.smoke_test:
        result = audit_dataset("synthetic", smoke=True)
        json.dumps(result, allow_nan=False)
        print("Smoke test passed: all six regimes, baseline methods, and PFS ablations.", flush=True)
        return
    result = {
        "audit": "v024_pathology_tta_ablation_audit", "configuration": configuration(),
        "datasets": {name: audit_dataset(name) for name in ("Camelyon17", "NCT-CRC")},
    }
    # Serialize before opening the output; fail rather than emitting non-standard JSON NaN/Infinity.
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output = WORKSPACE / "results" / "v024_pathology_tta_ablation_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=".v024_audit_", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Wrote aggregate audit: {output}", flush=True)


if __name__ == "__main__":
    main()
