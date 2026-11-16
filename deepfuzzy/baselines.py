"""
Competitive SOTA Multimodal & Uncertainty Baselines for Benchmarking against DeepFuzzyTorch.

Includes:
  1. EarlyConcatMLP: Feature-level concatenation with mask indicators + MLP.
  2. LateAvgEnsemble: Modality-specific encoders with mask-aware late probability averaging.
  3. GatedAttentionFusion: Cross-modal Transformer + Gated Attention with key-padding masks.
  4. DempsterShaferFusion: Classical Dempster-Shafer orthogonal combination rule.
  5. EvidentialDL (alias EvidentialDLBaseline): Trusted Multi-View Classification
     (TMC, Han et al., IEEE TPAMI 2023) with Sensoy/Han annealed Dirichlet KL loss.
  6. DeepTSKBaseline: Standalone High-Dimensional Takagi-Sugeno-Kang Fuzzy Rule System
     with configurable `embed_dim` and HTSK 1/D exponent scaling (Cui, Wu & Xu, IJCNN 2021).
  7. QMFBaseline: Quality-Aware Multimodal Fusion (Zhang et al., ICML 2023).
  8. RCMLBaseline: Reliable Conflictive Multi-View Learning (Xu et al., AAAI 2024)
     separating Dirichlet vacuity u and projective conflict C, with annealed KL loss.
  9. DrFuseBaseline: Disentangled Shared + Specific Multimodal Fusion with Missing-Modality
     Masking and Cross-Modal Inconsistency Scoring (Yao et al., AAAI 2024).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .operators import DeepConceptTSKHead


def _binary_entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p_clean = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0).clamp(min=eps, max=1.0 - eps)
    return -(p_clean * torch.log2(p_clean) + (1.0 - p_clean) * torch.log2(1.0 - p_clean))


def _prepare_masks_and_inputs(
    modality_inputs: List[torch.Tensor],
    modality_masks: Optional[torch.Tensor],
    num_modalities: int,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Sanitizes modality tensors (`NaN`, `+Inf`, `-Inf` -> `0.0`) and normalizes binary
    observation masks `[B, M]` (`1.0` = observed, `0.0` = missing), returning
    `(clean_inputs, binary_masks, all_missing_col)`.
    """
    B = modality_inputs[0].shape[0]
    device = modality_inputs[0].device
    dtype = modality_inputs[0].dtype
    if modality_masks is None:
        masks = torch.ones(B, num_modalities, device=device, dtype=dtype)
    else:
        m_raw = torch.nan_to_num(
            modality_masks.to(device=device, dtype=dtype), nan=0.0, posinf=1.0, neginf=0.0
        )
        masks = (m_raw > 0.5).to(dtype=dtype)

    clean_xs = [
        torch.nan_to_num(
            modality_inputs[m].to(device=device, dtype=dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        * masks[:, m : m + 1]
        for m in range(num_modalities)
    ]
    all_miss = (masks.sum(dim=1, keepdim=True) < 0.5).to(dtype=dtype)
    return clean_xs, masks, all_miss


def _dirichlet_kl_to_uniform(alpha: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes Sensoy et al. (NeurIPS 2018) / Han et al. (TPAMI 2023) Dirichlet KL divergence
    KL[Dir(alpha_tilde) || Dir(1)] after removing non-misleading target evidence:
        alpha_tilde = y_onehot + (1 - y_onehot) * alpha,
    where `alpha` has shape `[B, 2]` ordered as `[alpha_pos (y=1), alpha_neg (y=0)]`.
    """
    alpha_clean = torch.nan_to_num(alpha, nan=1.0, posinf=1e4, neginf=1.0).clamp(min=1.0, max=1e4)
    y_col = (
        torch.nan_to_num(
            targets.to(device=alpha_clean.device, dtype=alpha_clean.dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        .reshape(-1, 1)
        .clamp(0.0, 1.0)
    )
    y_oh = torch.cat([y_col, 1.0 - y_col], dim=-1)  # [B, 2]: col 0 is y=1, col 1 is y=0
    alpha_tilde = y_oh + (1.0 - y_oh) * alpha_clean
    k_classes = float(alpha_tilde.shape[-1])
    s_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    ones = torch.ones_like(alpha_tilde)

    ln_b = (
        torch.lgamma(s_tilde)
        - torch.lgamma(torch.full_like(s_tilde, k_classes))
        - torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
    )
    digamma_term = ((alpha_tilde - ones) * (torch.digamma(alpha_tilde) - torch.digamma(s_tilde))).sum(
        dim=-1, keepdim=True
    )
    return (ln_b + digamma_term).clamp(min=0.0)


class EarlyConcatMLP(nn.Module):
    """Baseline 1: Concatenates imputed modality vectors and observation masks into an MLP."""

    def __init__(self, modality_dims: List[int], embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)
        in_total = sum(modality_dims) + len(modality_dims)
        self.net = nn.Sequential(
            nn.LayerNorm(in_total),
            nn.Dropout(dropout),
            nn.Linear(in_total, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self, modality_inputs: List[torch.Tensor], modality_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        masked_xs, masks, all_miss = _prepare_masks_and_inputs(
            modality_inputs, modality_masks, self.num_modalities
        )
        feat = torch.cat(masked_xs + [masks], dim=-1)
        logit = torch.nan_to_num(self.net(feat), nan=0.0, posinf=20.0, neginf=-20.0)
        prob = torch.sigmoid(logit)
        prob = prob * (1.0 - all_miss) + 0.5 * all_miss
        hes = _binary_entropy(prob)
        mod_probs = [
            prob * masks[:, m : m + 1] + 0.5 * (1.0 - masks[:, m : m + 1])
            for m in range(self.num_modalities)
        ]
        return {
            "prob": prob,
            "hesitancy": hes,
            "conflict": (hes * 0.5).clamp(0.0, 1.0),
            "mod_probs": mod_probs,
            "modality_probs": mod_probs,
        }


class LateAvgEnsemble(nn.Module):
    """Baseline 2: Modality-specific deep encoders with mask-aware late probability averaging."""

    def __init__(self, modality_dims: List[int], embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 1),
            )
            for d in modality_dims
        ])

    def forward(
        self, modality_inputs: List[torch.Tensor], modality_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        masked_xs, masks, all_miss = _prepare_masks_and_inputs(
            modality_inputs, modality_masks, self.num_modalities
        )
        probs_m = []
        for m in range(self.num_modalities):
            m_col = masks[:, m : m + 1]
            logit_m = torch.nan_to_num(
                self.encoders[m](masked_xs[m]), nan=0.0, posinf=20.0, neginf=-20.0
            )
            p_m = torch.sigmoid(logit_m) * m_col + 0.5 * (1.0 - m_col)
            probs_m.append(p_m)

        p_stack = torch.stack(probs_m, dim=1)  # [B, M, 1]
        m_exp = masks.unsqueeze(-1)  # [B, M, 1]
        denom = m_exp.sum(dim=1).clamp(min=1.0)
        prob = (p_stack * m_exp).sum(dim=1) / denom
        prob = prob * (1.0 - all_miss) + 0.5 * all_miss

        var_m = ((p_stack - prob.unsqueeze(1)).pow(2) * m_exp).sum(dim=1) / denom
        conflict = torch.sqrt(var_m + 1e-8).clamp(0.0, 1.0) * (1.0 - all_miss)
        hes = (0.5 * _binary_entropy(prob) + 2.0 * conflict).clamp(0.0, 1.0)
        return {
            "prob": prob,
            "hesitancy": hes,
            "conflict": conflict,
            "mod_probs": probs_m,
            "modality_probs": probs_m,
        }


class GatedAttentionFusion(nn.Module):
    """Baseline 3: Cross-Modal Transformer + Gated Attention Pooling with Key-Padding Masking."""

    def __init__(self, modality_dims: List[int], embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)
        n_heads = 4 if embed_dim % 4 == 0 else (2 if embed_dim % 2 == 0 else 1)
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            for d in modality_dims
        ])
        self.mod_embeddings = nn.Parameter(torch.randn(1, len(modality_dims), embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, max(4, embed_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(4, embed_dim // 2), 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self, modality_inputs: List[torch.Tensor], modality_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        masked_xs, masks, all_miss = _prepare_masks_and_inputs(
            modality_inputs, modality_masks, self.num_modalities
        )
        tokens = []
        mod_probs = []
        for m in range(self.num_modalities):
            m_col = masks[:, m : m + 1]
            z_m = (self.projectors[m](masked_xs[m]) + self.mod_embeddings[:, m, :]) * m_col
            z_m = torch.nan_to_num(z_m, nan=0.0, posinf=0.0, neginf=0.0)
            tokens.append(z_m)
            logit_m = torch.nan_to_num(self.classifier(z_m), nan=0.0, posinf=20.0, neginf=-20.0)
            mod_probs.append(torch.sigmoid(logit_m) * m_col + 0.5 * (1.0 - m_col))
        x_seq = torch.stack(tokens, dim=1)  # [B, M, D]

        key_pad = masks < 0.5  # [B, M] bool
        all_missing_rows = key_pad.all(dim=1)
        if all_missing_rows.any():
            key_pad = key_pad.clone()
            key_pad[all_missing_rows, 0] = False

        attn_out, _ = self.attn(x_seq, x_seq, x_seq, key_padding_mask=key_pad)
        attn_out = (
            torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0) * masks.unsqueeze(-1)
        )
        x_seq = x_seq + attn_out

        gate_logits = torch.nan_to_num(
            self.gate_net(x_seq).squeeze(-1), nan=0.0, posinf=20.0, neginf=-20.0
        )
        masked_logits = gate_logits.masked_fill(key_pad, -1e9)
        weights = F.softmax(masked_logits, dim=-1).unsqueeze(-1) * masks.unsqueeze(-1)
        w_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-7)
        weights = weights / w_sum

        pooled = (weights * x_seq).sum(dim=1)
        logit = torch.nan_to_num(self.classifier(pooled), nan=0.0, posinf=20.0, neginf=-20.0)
        prob = torch.sigmoid(logit)
        prob = prob * (1.0 - all_miss) + 0.5 * all_miss
        hes = _binary_entropy(prob)
        return {
            "prob": prob,
            "hesitancy": hes,
            "conflict": (hes * 0.5).clamp(0.0, 1.0),
            "mod_probs": mod_probs,
            "modality_probs": mod_probs,
        }


class DempsterShaferFusion(nn.Module):
    """Baseline 4: Classical Dempster-Shafer Belief Combination Network."""

    def __init__(self, modality_dims: List[int], embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 2),
            )
            for d in modality_dims
        ])

    def forward(
        self, modality_inputs: List[torch.Tensor], modality_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        masked_xs, masks, _ = _prepare_masks_and_inputs(
            modality_inputs, modality_masks, self.num_modalities
        )
        B = masked_xs[0].shape[0]
        device = masked_xs[0].device
        dtype = masked_xs[0].dtype

        b_pos = torch.zeros(B, 1, device=device, dtype=dtype)
        b_neg = torch.zeros(B, 1, device=device, dtype=dtype)
        u_mass = torch.ones(B, 1, device=device, dtype=dtype)
        total_conflict = torch.zeros(B, 1, device=device, dtype=dtype)
        mod_probs = []

        for m in range(self.num_modalities):
            mask_m = masks[:, m : m + 1]
            raw_logits = torch.nan_to_num(
                self.encoders[m](masked_xs[m]), nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(max=20.0)
            raw = F.softplus(raw_logits)
            e_pos = raw[:, 0:1] * mask_m
            e_neg = raw[:, 1:2] * mask_m
            S = e_pos + e_neg + 2.0
            bm_pos = e_pos / S
            bm_neg = e_neg / S
            um = 2.0 / S
            mod_probs.append((e_pos + 1.0) / S)

            K = b_pos * bm_neg + b_neg * bm_pos
            norm = (1.0 - K).clamp(min=1e-4)
            new_pos = (b_pos * bm_pos + b_pos * um + u_mass * bm_pos) / norm
            new_neg = (b_neg * bm_neg + b_neg * um + u_mass * bm_neg) / norm
            new_u = (u_mass * um) / norm

            s_tot = (new_pos + new_neg + new_u).clamp(min=1e-6)
            b_pos, b_neg, u_mass = new_pos / s_tot, new_neg / s_tot, new_u / s_tot
            total_conflict = total_conflict + K

        prob = (b_pos + 0.5 * u_mass).clamp(0.0, 1.0)
        return {
            "prob": prob,
            "hesitancy": u_mass.clamp(0.0, 1.0),
            "conflict": (total_conflict / max(1, self.num_modalities - 1)).clamp(0.0, 1.0),
            "mod_probs": mod_probs,
            "modality_probs": mod_probs,
        }


class EvidentialDL(nn.Module):
    """
    Baseline 5: Trusted Multi-View Classification (TMC, Han et al., IEEE TPAMI 2023)
    with Dirichlet Subjective Logic Opinion Combination (M1 oplus M2):
        b_k = (b_k^{(1)} b_k^{(2)} + b_k^{(1)} u^{(2)} + b_k^{(2)} u^{(1)}) / (1 - C),
        u   = (u^{(1)} u^{(2)}) / (1 - C),   C = b_0^{(1)} b_1^{(2)} + b_1^{(1)} b_0^{(2)},
        S   = 2 / u,   alpha_k = b_k * S + 1.
    Includes `compute_kl_loss(targets, epoch, annealing_step=10)` for Sensoy/Han KL annealing.
    """

    def __init__(self, modality_dims: List[int], embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 2),
            )
            for d in modality_dims
        ])
        self._last_alpha: Optional[torch.Tensor] = None
        self._last_mod_alphas: Optional[List[torch.Tensor]] = None
        self._last_masks: Optional[torch.Tensor] = None

    def forward(
        self, modality_inputs: List[torch.Tensor], modality_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        masked_xs, masks, _ = _prepare_masks_and_inputs(
            modality_inputs, modality_masks, self.num_modalities
        )
        B = masked_xs[0].shape[0]
        device = masked_xs[0].device
        dtype = masked_xs[0].dtype

        b_pos = torch.zeros(B, 1, device=device, dtype=dtype)
        b_neg = torch.zeros(B, 1, device=device, dtype=dtype)
        u = torch.ones(B, 1, device=device, dtype=dtype)
        conflict_acc = torch.zeros(B, 1, device=device, dtype=dtype)
        mod_probs: List[torch.Tensor] = []
        mod_alphas: List[torch.Tensor] = []

        for m in range(self.num_modalities):
            mask_m = masks[:, m : m + 1]
            raw_logits = torch.nan_to_num(
                self.encoders[m](masked_xs[m]), nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(max=20.0)
            ev = F.softplus(raw_logits)
            ep = ev[:, 0:1] * mask_m
            en = ev[:, 1:2] * mask_m
            S_m = ep + en + 2.0
            bm_pos = ep / S_m
            bm_neg = en / S_m
            u_m = 2.0 / S_m
            mod_probs.append((ep + 1.0) / S_m)
            mod_alphas.append(torch.cat([ep + 1.0, en + 1.0], dim=-1))

            C = b_pos * bm_neg + b_neg * bm_pos
            norm = (1.0 - C).clamp(min=1e-4)
            b_pos_new = (b_pos * bm_pos + b_pos * u_m + u * bm_pos) / norm
            b_neg_new = (b_neg * bm_neg + b_neg * u_m + u * bm_neg) / norm
            u_new = (u * u_m) / norm

            tot = (b_pos_new + b_neg_new + u_new).clamp(min=1e-6)
            b_pos, b_neg, u = b_pos_new / tot, b_neg_new / tot, u_new / tot
            conflict_acc = conflict_acc + C

        S_comb = 2.0 / u.clamp(min=1e-4)
        alpha_pos = b_pos * S_comb + 1.0
        alpha_neg = b_neg * S_comb + 1.0
        alpha_fused = torch.cat([alpha_pos, alpha_neg], dim=-1)

        self._last_alpha = alpha_fused
        self._last_mod_alphas = mod_alphas
        self._last_masks = masks

        prob = (alpha_pos / (alpha_pos + alpha_neg)).clamp(0.0, 1.0)
        return {
            "prob": prob,
            "hesitancy": u.clamp(0.0, 1.0),
            "conflict": (conflict_acc / max(1, self.num_modalities - 1)).clamp(0.0, 1.0),
            "alpha": alpha_fused,
            "mod_alphas": mod_alphas,
            "mod_probs": mod_probs,
            "modality_probs": mod_probs,
        }

    def compute_kl_loss(
        self,
        targets: torch.Tensor,
        epoch: int = 10,
        annealing_step: int = 10,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes annealed Sensoy et al. (2018) / Han et al. (2023) Dirichlet KL divergence
        regularizing misleading evidence on fused and observed view-specific Dirichlet opinions.
        """
        coef = min(1.0, max(0.0, float(epoch) / float(max(1, annealing_step))))
        if alpha is not None:
            return coef * _dirichlet_kl_to_uniform(alpha, targets).mean()
        if self._last_alpha is None:
            raise RuntimeError("Call forward() before compute_kl_loss() or pass `alpha` explicitly.")

        kl_fused = _dirichlet_kl_to_uniform(self._last_alpha, targets).mean()
        if self._last_mod_alphas is not None and self._last_masks is not None:
            kl_views = torch.zeros_like(kl_fused)
            for m, alpha_m in enumerate(self._last_mod_alphas):
                m_col = self._last_masks[:, m : m + 1]
                denom_m = m_col.sum().clamp(min=1.0)
                kl_m = (_dirichlet_kl_to_uniform(alpha_m, targets) * m_col).sum() / denom_m
                kl_views = kl_views + kl_m / float(max(1, len(self._last_mod_alphas)))
            return coef * (kl_fused + 0.5 * kl_views)
        return coef * kl_fused


EvidentialDLBaseline = EvidentialDL


class DeepTSKBaseline(nn.Module):
    """
    Baseline 6: Standalone High-Dimensional Takagi-Sugeno-Kang (TSK) Fuzzy Rule Classifier
    with configurable `embed_dim` and HTSK 1/D mean-exponent scaling (Cui, Wu & Xu, IJCNN 2021)
    preventing Gaussian rule firing underflow on high-dimensional multimodal inputs.
    """

    def __init__(
        self,
        modality_dims: List[int],
        embed_dim: int = 64,
        num_concepts: int = 8,
        num_rules: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)
        self.num_concepts = num_concepts
        self.num_rules = num_rules
        self.htsk_exponent_scale = 1.0 / float(max(1, num_concepts))

        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            for d in modality_dims
        ])
        self.unimodal_heads = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in modality_dims])
        self.concept_extractor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim * len(modality_dims) + len(modality_dims), embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_concepts),
            nn.Tanh(),
        )
        # DeepConceptTSKHead applies exact HTSK 1/C mean-squared distance scaling so firing weights never underflow
        self.tsk = DeepConceptTSKHead(num_concepts=num_concepts, num_rules=num_rules, num_classes=1)

    def forward(
        self, modality_inputs: List[torch.Tensor], modality_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        masked_xs, masks, all_miss = _prepare_masks_and_inputs(
            modality_inputs, modality_masks, self.num_modalities
        )
        embeds = []
        mod_probs = []
        for m in range(self.num_modalities):
            m_col = masks[:, m : m + 1]
            z_m = torch.nan_to_num(
                self.modality_encoders[m](masked_xs[m]), nan=0.0, posinf=0.0, neginf=0.0
            ) * m_col
            embeds.append(z_m)
            logit_m = torch.nan_to_num(
                self.unimodal_heads[m](z_m), nan=0.0, posinf=20.0, neginf=-20.0
            )
            mod_probs.append(torch.sigmoid(logit_m) * m_col + 0.5 * (1.0 - m_col))

        concepts = self.concept_extractor(torch.cat(embeds + [masks], dim=-1))
        concepts = torch.nan_to_num(concepts, nan=0.0, posinf=0.0, neginf=0.0)
        tsk_logits, tsk_aux = self.tsk(concepts)
        tsk_logits = torch.nan_to_num(tsk_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        prob = torch.sigmoid(tsk_logits)
        prob = prob * (1.0 - all_miss) + 0.5 * all_miss

        p_stack = torch.stack(mod_probs, dim=1)  # [B, M, 1]
        m_exp = masks.unsqueeze(-1)
        denom = m_exp.sum(dim=1).clamp(min=1.0)
        p_mean = (p_stack * m_exp).sum(dim=1) / denom
        var_m = ((p_stack - p_mean.unsqueeze(1)).pow(2) * m_exp).sum(dim=1) / denom
        conflict = torch.sqrt(var_m + 1e-8).clamp(0.0, 1.0) * (1.0 - all_miss)
        hes = _binary_entropy(prob)
        return {
            "prob": prob,
            "hesitancy": hes,
            "conflict": conflict,
            "mod_probs": mod_probs,
            "modality_probs": mod_probs,
            "tsk_aux": tsk_aux,
        }


class QMFBaseline(nn.Module):
    """
    Baseline 7: Quality-Aware Multimodal Fusion (QMF, Zhang et al., ICML 2023).

    Dynamically weights unimodal logits by sample-adaptive modality energy / margin quality:
        q_m(x) = softplus(beta_m) * logsumexp(z^{(m)}) + softplus(gamma_m) * |z_pos^{(m)} - z_neg^{(m)}|,
        w_m(x) = m_m * exp(q_m(x)) / sum_n m_n * exp(q_n(x)),
        z_fused(x) = sum_m w_m(x) * (z_pos^{(m)} - z_neg^{(m)}).
    """

    def __init__(self, modality_dims: List[int], embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout * 0.5),
                nn.Linear(embed_dim, 2),
            )
            for d in modality_dims
        ])
        self.energy_scale_raw = nn.Parameter(torch.zeros(len(modality_dims)))
        self.margin_scale_raw = nn.Parameter(torch.zeros(len(modality_dims)))

    def forward(
        self, modality_inputs: List[torch.Tensor], modality_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        masked_xs, masks, all_miss = _prepare_masks_and_inputs(
            modality_inputs, modality_masks, self.num_modalities
        )
        diff_logits = []
        qualities = []
        mod_probs = []
        beta_e = F.softplus(self.energy_scale_raw)
        gamma_m = F.softplus(self.margin_scale_raw)

        for m in range(self.num_modalities):
            m_col = masks[:, m : m + 1]
            z2 = torch.nan_to_num(
                self.encoders[m](masked_xs[m]), nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(-20.0, 20.0)
            diff_m = z2[:, 0:1] - z2[:, 1:2]  # [B, 1]
            energy_m = torch.logsumexp(z2, dim=-1, keepdim=True)  # [B, 1]
            qual_m = beta_e[m] * energy_m + gamma_m[m] * diff_m.abs()
            diff_logits.append(diff_m)
            qualities.append(qual_m.squeeze(-1))
            mod_probs.append(torch.sigmoid(diff_m) * m_col + 0.5 * (1.0 - m_col))

        q_mat = torch.stack(qualities, dim=1)  # [B, M]
        key_pad = masks < 0.5
        all_missing_rows = key_pad.all(dim=1)
        if all_missing_rows.any():
            key_pad = key_pad.clone()
            key_pad[all_missing_rows, 0] = False

        q_masked = q_mat.masked_fill(key_pad, -1e9)
        w = F.softmax(q_masked, dim=-1) * masks  # [B, M]
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-7)

        diff_stack = torch.stack(diff_logits, dim=1).squeeze(-1)  # [B, M]
        fused_logit = (w * diff_stack).sum(dim=-1, keepdim=True)
        prob = torch.sigmoid(fused_logit)
        prob = prob * (1.0 - all_miss) + 0.5 * all_miss

        p_stack = torch.stack(mod_probs, dim=1).squeeze(-1)  # [B, M]
        var_w = (w * (p_stack - prob).pow(2)).sum(dim=-1, keepdim=True)
        conflict = (2.0 * torch.sqrt(var_w + 1e-8)).clamp(0.0, 1.0) * (1.0 - all_miss)
        hes = (0.5 * _binary_entropy(prob) + 0.5 * conflict).clamp(0.0, 1.0)
        hes = hes * (1.0 - all_miss) + 1.0 * all_miss

        return {
            "prob": prob,
            "hesitancy": hes,
            "conflict": conflict,
            "quality_weights": w,
            "mod_probs": mod_probs,
            "modality_probs": mod_probs,
        }


class RCMLBaseline(nn.Module):
    """
    Baseline 8: Reliable Conflictive Multi-View Learning — Adapted (`RCML (adapted)`,
    adapted from Xu et al., AAAI 2024).

    Methodological Adaptation Note:
    Retains RCML's view-specific Dirichlet evidence encoders, conflict-tolerant opinion
    averaging over observed views (`e_pool`, `alpha = e_pool + 1`, vacuity `u`), and
    pairwise projective conflict `C_{m,n} = |p^{(m)} - p^{(n)}| * (1 - u^{(m)}) * (1 - u^{(n)})`,
    while adapting the training objective in `benchmarks/run_full_benchmark.py` to binary
    clinical risk prediction via evidential MSE/Brier + annealed Dirichlet KL + auxiliary
    unimodal BCE instead of Xu et al.'s original evidential accuracy and cross-view consistency loss.
    """

    def __init__(self, modality_dims: List[int], embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 2),
            )
            for d in modality_dims
        ])
        self.view_logits = nn.Parameter(torch.zeros(len(modality_dims)))
        self._last_alpha: Optional[torch.Tensor] = None
        self._last_mod_alphas: Optional[List[torch.Tensor]] = None
        self._last_masks: Optional[torch.Tensor] = None

    def forward(
        self, modality_inputs: List[torch.Tensor], modality_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        masked_xs, masks, all_miss = _prepare_masks_and_inputs(
            modality_inputs, modality_masks, self.num_modalities
        )
        B = masked_xs[0].shape[0]
        device = masked_xs[0].device
        dtype = masked_xs[0].dtype

        ev_list: List[torch.Tensor] = []
        u_list: List[torch.Tensor] = []
        mod_probs: List[torch.Tensor] = []
        mod_alphas: List[torch.Tensor] = []
        rel_priors = F.softplus(self.view_logits)  # [M] > 0

        for m in range(self.num_modalities):
            m_col = masks[:, m : m + 1]
            raw_logits = torch.nan_to_num(
                self.encoders[m](masked_xs[m]), nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(max=20.0)
            ev_m = F.softplus(raw_logits) * m_col  # [B, 2]
            alpha_m = ev_m + 1.0
            s_m = alpha_m.sum(dim=-1, keepdim=True)
            u_m = 2.0 / s_m  # 1.0 when missing
            p_m = alpha_m[:, 0:1] / s_m  # 0.5 when missing

            ev_list.append(ev_m)
            u_list.append(u_m)
            mod_probs.append(p_m)
            mod_alphas.append(alpha_m)

        # Conflict-tolerant opinion averaging weighted by observed view reliability priors (Xu et al., AAAI 2024)
        ev_stack = torch.stack(ev_list, dim=1)  # [B, M, 2]
        w_view = masks * rel_priors.unsqueeze(0)  # [B, M]
        w_denom = w_view.sum(dim=1, keepdim=True).clamp(min=1e-6)
        w_norm = (w_view / w_denom) * (1.0 - all_miss)  # [B, M], exact opinion average
        e_pool = (ev_stack * w_norm.unsqueeze(-1)).sum(dim=1)  # [B, 2]

        alpha_fused = e_pool + 1.0  # [B, 2]
        s_fused = alpha_fused.sum(dim=-1, keepdim=True)
        vacuity = (2.0 / s_fused).clamp(0.0, 1.0)
        prob = (alpha_fused[:, 0:1] / s_fused).clamp(0.0, 1.0)
        prob = prob * (1.0 - all_miss) + 0.5 * all_miss

        # Pairwise projective conflict across observed view pairs (m < n)
        conflict_sum = torch.zeros(B, 1, device=device, dtype=dtype)
        pair_weight_sum = torch.zeros(B, 1, device=device, dtype=dtype)
        for m in range(self.num_modalities):
            for n in range(m + 1, self.num_modalities):
                pair_obs = masks[:, m : m + 1] * masks[:, n : n + 1]
                c_mn = (
                    (mod_probs[m] - mod_probs[n]).abs()
                    * (1.0 - u_list[m])
                    * (1.0 - u_list[n])
                )
                conflict_sum = conflict_sum + pair_obs * c_mn
                pair_weight_sum = pair_weight_sum + pair_obs

        has_pair = (pair_weight_sum > 0.5).to(dtype=dtype)
        conflict = (conflict_sum / pair_weight_sum.clamp(min=1.0)).clamp(0.0, 1.0) * has_pair
        # Total uncertainty combines epistemic vacuity u and inter-view dissonance C
        hesitancy = (vacuity + (1.0 - vacuity) * conflict).clamp(0.0, 1.0)

        self._last_alpha = alpha_fused
        self._last_mod_alphas = mod_alphas
        self._last_masks = masks

        return {
            "prob": prob,
            "hesitancy": hesitancy,
            "vacuity": vacuity,
            "conflict": conflict,
            "alpha": alpha_fused,
            "mod_alphas": mod_alphas,
            "mod_probs": mod_probs,
            "modality_probs": mod_probs,
        }

    def compute_kl_loss(
        self,
        targets: torch.Tensor,
        epoch: int = 10,
        annealing_step: int = 10,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes annealed Dirichlet KL divergence regularizing misleading evidence on both
        conflict-pooled and observed view-specific opinions (Xu et al., AAAI 2024).
        """
        coef = min(1.0, max(0.0, float(epoch) / float(max(1, annealing_step))))
        if alpha is not None:
            return coef * _dirichlet_kl_to_uniform(alpha, targets).mean()
        if self._last_alpha is None:
            raise RuntimeError("Call forward() before compute_kl_loss() or pass `alpha` explicitly.")

        kl_fused = _dirichlet_kl_to_uniform(self._last_alpha, targets).mean()
        if self._last_mod_alphas is not None and self._last_masks is not None:
            kl_views = torch.zeros_like(kl_fused)
            for m, alpha_m in enumerate(self._last_mod_alphas):
                m_col = self._last_masks[:, m : m + 1]
                denom_m = m_col.sum().clamp(min=1.0)
                kl_m = (_dirichlet_kl_to_uniform(alpha_m, targets) * m_col).sum() / denom_m
                kl_views = kl_views + kl_m / float(max(1, len(self._last_mod_alphas)))
            return coef * (kl_fused + 0.5 * kl_views)
        return coef * kl_fused


class DrFuseBaseline(nn.Module):
    """
    Baseline 9: Disentangled Shared + Modality-Specific Multimodal Fusion — Adapted
    (`DrFuse (adapted)`, adapted from Yao et al., AAAI 2024).

    Methodological Adaptation Note:
    Retains dual shared/distinctive encoders with cross-modal shared-representation cosine
    orthogonality penalty and missing-modality masking, while adapting the fusion backbone to
    tabular/biomedical modalities using masked-mean shared-token pooling + single-layer
    `nn.MultiheadAttention` across tokens instead of Yao et al.'s original multi-layer
    Transformer encoder, JSD alignment loss, logit pooling, and attention ranking loss.
    """

    def __init__(self, modality_dims: List[int], embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)
        n_heads = 4 if embed_dim % 4 == 0 else (2 if embed_dim % 2 == 0 else 1)

        self.shared_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            for d in modality_dims
        ])
        self.specific_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            for d in modality_dims
        ])
        self.unimodal_heads = nn.ModuleList([
            nn.Linear(embed_dim * 2, 1) for _ in modality_dims
        ])
        self.token_embeddings = nn.Parameter(torch.randn(1, len(modality_dims) + 1, embed_dim) * 0.02)
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.attn_pool = nn.Sequential(
            nn.Linear(embed_dim, max(4, embed_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(4, embed_dim // 2), 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self, modality_inputs: List[torch.Tensor], modality_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        masked_xs, masks, all_miss = _prepare_masks_and_inputs(
            modality_inputs, modality_masks, self.num_modalities
        )
        B = masked_xs[0].shape[0]
        device = masked_xs[0].device
        dtype = masked_xs[0].dtype

        shared_list: List[torch.Tensor] = []
        spec_list: List[torch.Tensor] = []
        mod_probs: List[torch.Tensor] = []
        disentangle_cos = torch.zeros(B, 1, device=device, dtype=dtype)

        for m in range(self.num_modalities):
            m_col = masks[:, m : m + 1]
            z_sh = torch.nan_to_num(
                self.shared_encoders[m](masked_xs[m]), nan=0.0, posinf=0.0, neginf=0.0
            ) * m_col
            z_sp = torch.nan_to_num(
                self.specific_encoders[m](masked_xs[m]), nan=0.0, posinf=0.0, neginf=0.0
            ) * m_col
            shared_list.append(z_sh)
            spec_list.append(z_sp)

            logit_m = torch.nan_to_num(
                self.unimodal_heads[m](torch.cat([z_sh, z_sp], dim=-1)),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
            mod_probs.append(torch.sigmoid(logit_m) * m_col + 0.5 * (1.0 - m_col))

            cos_m = F.cosine_similarity(z_sh, z_sp, dim=-1, eps=1e-6).abs().unsqueeze(-1)
            disentangle_cos = disentangle_cos + cos_m * m_col

        sh_stack = torch.stack(shared_list, dim=1)  # [B, M, D]
        m_exp = masks.unsqueeze(-1)  # [B, M, 1]
        obs_cnt = m_exp.sum(dim=1).clamp(min=1.0)
        z_shared_bar = (sh_stack * m_exp).sum(dim=1) / obs_cnt  # [B, D]

        # Tokens: [z_shared_bar, z_spec_1, ..., z_spec_M]
        tok_list = [z_shared_bar * (1.0 - all_miss) + self.token_embeddings[:, 0, :] * (1.0 - all_miss)]
        for m in range(self.num_modalities):
            m_col = masks[:, m : m + 1]
            tok_list.append((spec_list[m] + self.token_embeddings[:, m + 1, :]) * m_col)
        tok_seq = torch.stack(tok_list, dim=1)  # [B, M + 1, D]

        # Shared token (col 0) is active whenever >= 1 modality is observed
        shared_active = (1.0 - all_miss)  # [B, 1]
        tok_masks = torch.cat([shared_active, masks], dim=1)  # [B, M + 1]
        key_pad = tok_masks < 0.5
        all_missing_rows = key_pad.all(dim=1)
        if all_missing_rows.any():
            key_pad = key_pad.clone()
            key_pad[all_missing_rows, 0] = False

        attn_out, _ = self.fusion_attn(tok_seq, tok_seq, tok_seq, key_padding_mask=key_pad)
        attn_out = (
            torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0) * tok_masks.unsqueeze(-1)
        )
        tok_seq = tok_seq + attn_out

        pool_logits = torch.nan_to_num(
            self.attn_pool(tok_seq).squeeze(-1), nan=0.0, posinf=20.0, neginf=-20.0
        )
        pool_logits = pool_logits.masked_fill(key_pad, -1e9)
        pool_w = F.softmax(pool_logits, dim=-1).unsqueeze(-1) * tok_masks.unsqueeze(-1)
        pool_w = pool_w / pool_w.sum(dim=1, keepdim=True).clamp(min=1e-7)

        fused_repr = (pool_w * tok_seq).sum(dim=1)
        logit = torch.nan_to_num(
            self.classifier(fused_repr), nan=0.0, posinf=20.0, neginf=-20.0
        )
        prob = torch.sigmoid(logit)
        prob = prob * (1.0 - all_miss) + 0.5 * all_miss

        # Cross-modal inconsistency score: shared-subspace cosine divergence + unimodal probability spread
        cos_to_bar = F.cosine_similarity(
            sh_stack, z_shared_bar.unsqueeze(1).expand_as(sh_stack), dim=-1, eps=1e-6
        )  # [B, M]
        subspace_div = ((0.5 * (1.0 - cos_to_bar)).clamp(0.0, 1.0) * masks).sum(
            dim=1, keepdim=True
        ) / obs_cnt
        p_stack = torch.stack(mod_probs, dim=1)  # [B, M, 1]
        p_mean = (p_stack * m_exp).sum(dim=1) / obs_cnt
        prob_spread = (2.0 * torch.sqrt(((p_stack - p_mean.unsqueeze(1)).pow(2) * m_exp).sum(dim=1) / obs_cnt + 1e-8)).clamp(0.0, 1.0)
        multi_obs = (masks.sum(dim=1, keepdim=True) > 1.5).to(dtype=dtype)
        conflict = (0.5 * subspace_div + 0.5 * prob_spread).clamp(0.0, 1.0) * multi_obs
        hes = (0.5 * _binary_entropy(prob) + 0.5 * conflict).clamp(0.0, 1.0)
        hes = hes * (1.0 - all_miss) + 1.0 * all_miss

        return {
            "prob": prob,
            "hesitancy": hes,
            "conflict": conflict,
            "disentangle_loss": (disentangle_cos / obs_cnt).mean(),
            "mod_probs": mod_probs,
            "modality_probs": mod_probs,
        }


class _BaseEvidentialBaseline(nn.Module):
    """Shared Dirichlet KL regularization and pairwise conflict helper for evidential baselines."""

    def __init__(self, modality_dims: List[int], embed_dim: int = 64):
        super().__init__()
        self.modality_dims = list(modality_dims)
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)
        self._last_alpha: Optional[torch.Tensor] = None
        self._last_mod_alphas: Optional[List[torch.Tensor]] = None
        self._last_masks: Optional[torch.Tensor] = None
        self._last_extra_loss: Optional[torch.Tensor] = None

    @staticmethod
    def _pairwise_belief_conflict(
        b_pos_list: List[torch.Tensor],
        b_neg_list: List[torch.Tensor],
        masks: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Mean Dempster-Shafer pairwise conflict b_m^+ b_n^- + b_m^- b_n^+ over observed pairs."""
        c_sum = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        p_sum = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        M = len(b_pos_list)
        for m in range(M):
            for n in range(m + 1, M):
                pair_obs = masks[:, m : m + 1] * masks[:, n : n + 1]
                k_mn = b_pos_list[m] * b_neg_list[n] + b_neg_list[m] * b_pos_list[n]
                c_sum = c_sum + pair_obs * k_mn
                p_sum = p_sum + pair_obs
        has_pair = (p_sum > 0.5).to(dtype=dtype)
        return (c_sum / p_sum.clamp(min=1.0)).clamp(0.0, 1.0) * has_pair

    def compute_kl_loss(
        self,
        targets: torch.Tensor,
        epoch: int = 10,
        annealing_step: int = 10,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Annealed Dirichlet KL divergence over fused and observed view-specific opinions."""
        coef = min(1.0, max(0.0, float(epoch) / float(max(1, annealing_step))))
        if alpha is not None:
            return coef * _dirichlet_kl_to_uniform(alpha, targets).mean()
        if self._last_alpha is None:
            raise RuntimeError("Call forward() before compute_kl_loss() or pass `alpha` explicitly.")

        kl_fused = _dirichlet_kl_to_uniform(self._last_alpha, targets).mean()
        loss = kl_fused
        if self._last_mod_alphas is not None and self._last_masks is not None:
            kl_views = torch.zeros_like(kl_fused)
            for m, alpha_m in enumerate(self._last_mod_alphas):
                m_col = self._last_masks[:, m : m + 1]
                denom_m = m_col.sum().clamp(min=1.0)
                kl_m = (_dirichlet_kl_to_uniform(alpha_m, targets) * m_col).sum() / denom_m
                kl_views = kl_views + kl_m / float(max(1, len(self._last_mod_alphas)))
            loss = loss + 0.5 * kl_views
        total = coef * loss
        if self._last_extra_loss is not None:
            total = total + 0.10 * self._last_extra_loss
        return total


class DDEFBaseline(_BaseEvidentialBaseline):
    """
    Baseline 10: Feature-level adaptation of Dual-level Deep Evidential Fusion (DDEF;
    Shao et al., Information Fusion 2024).

    Combines modality-specific feature evidences with cross-modal context verification
    gates/heads to calibrate per-view Dirichlet opinions, then fuses observed views
    via Dempster-Shafer belief combination with exact vacuous identity under missingness.
    """

    def __init__(self, modality_dims: List[int], embed_dim: int = 64, dropout: float = 0.2):
        super().__init__(modality_dims=modality_dims, embed_dim=embed_dim)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout * 0.5),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            for d in modality_dims
        ])
        self.feat_ev_heads = nn.ModuleList([nn.Linear(embed_dim, 2) for _ in modality_dims])
        self.ver_gates = nn.ModuleList([nn.Linear(embed_dim * 3, 1) for _ in modality_dims])
        self.ver_heads = nn.ModuleList([nn.Linear(embed_dim * 2, 2) for _ in modality_dims])

    def forward(
        self, modality_inputs: List[torch.Tensor], modality_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        masked_xs, masks, all_miss = _prepare_masks_and_inputs(
            modality_inputs, modality_masks, self.num_modalities
        )
        B = masked_xs[0].shape[0]
        device = masked_xs[0].device
        dtype = masked_xs[0].dtype

        h_list: List[torch.Tensor] = []
        e_feat_list: List[torch.Tensor] = []
        p_feat_list: List[torch.Tensor] = []
        u_feat_list: List[torch.Tensor] = []
        for m in range(self.num_modalities):
            m_col = masks[:, m : m + 1]
            h_m = torch.nan_to_num(
                self.encoders[m](masked_xs[m]), nan=0.0, posinf=0.0, neginf=0.0
            ) * m_col
            raw_f = torch.nan_to_num(
                self.feat_ev_heads[m](h_m), nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(max=20.0)
            ef_m = F.softplus(raw_f) * m_col
            af_m = ef_m + 1.0
            sf_m = af_m.sum(dim=-1, keepdim=True)
            h_list.append(h_m)
            e_feat_list.append(ef_m)
            p_feat_list.append(af_m[:, 0:1] / sf_m)
            u_feat_list.append(2.0 / sf_m)

        h_stack = torch.stack(h_list, dim=1)  # [B, M, D]
        m_exp = masks.unsqueeze(-1)
        obs_cnt = m_exp.sum(dim=1).clamp(min=1.0)
        h_ctx = ((h_stack * m_exp).sum(dim=1) / obs_cnt) * (1.0 - all_miss)

        b_pos = torch.zeros(B, 1, device=device, dtype=dtype)
        b_neg = torch.zeros(B, 1, device=device, dtype=dtype)
        u = torch.ones(B, 1, device=device, dtype=dtype)

        mod_probs: List[torch.Tensor] = []
        mod_alphas: List[torch.Tensor] = []
        b_pos_list: List[torch.Tensor] = []
        b_neg_list: List[torch.Tensor] = []

        for m in range(self.num_modalities):
            m_col = masks[:, m : m + 1]
            h_m = h_list[m]
            gate_in = torch.cat([h_m, h_ctx, (h_m - h_ctx).abs()], dim=-1)
            ver_gate = torch.sigmoid(
                torch.nan_to_num(self.ver_gates[m](gate_in), nan=0.0, posinf=10.0, neginf=-10.0)
            )
            ev_ver_raw = torch.nan_to_num(
                self.ver_heads[m](torch.cat([h_m, h_ctx], dim=-1)), nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(max=20.0)
            e_ver_m = F.softplus(ev_ver_raw)
            ev_m = (ver_gate * e_feat_list[m] + e_ver_m) * m_col
            alpha_m = ev_m + 1.0
            s_m = alpha_m.sum(dim=-1, keepdim=True)
            bm_pos = ev_m[:, 0:1] / s_m
            bm_neg = ev_m[:, 1:2] / s_m
            u_m = 2.0 / s_m

            mod_probs.append((alpha_m[:, 0:1] / s_m) * m_col + 0.5 * (1.0 - m_col))
            mod_alphas.append(alpha_m)
            b_pos_list.append(bm_pos)
            b_neg_list.append(bm_neg)

            c_ds = b_pos * bm_neg + b_neg * bm_pos
            norm = (1.0 - c_ds).clamp(min=1e-4)
            b_pos_new = (b_pos * bm_pos + b_pos * u_m + u * bm_pos) / norm
            b_neg_new = (b_neg * bm_neg + b_neg * u_m + u * bm_neg) / norm
            u_new = (u * u_m) / norm
            tot = (b_pos_new + b_neg_new + u_new).clamp(min=1e-6)
            b_pos, b_neg, u = b_pos_new / tot, b_neg_new / tot, u_new / tot

        s_comb = 2.0 / u.clamp(min=1e-4)
        alpha_pos = b_pos * s_comb + 1.0
        alpha_neg = b_neg * s_comb + 1.0
        alpha_fused = torch.cat([alpha_pos, alpha_neg], dim=-1)

        vacuity = u.clamp(0.0, 1.0) * (1.0 - all_miss) + 1.0 * all_miss
        prob = (alpha_pos / (alpha_pos + alpha_neg)).clamp(0.0, 1.0)
        prob = prob * (1.0 - all_miss) + 0.5 * all_miss

        conflict = self._pairwise_belief_conflict(
            b_pos_list, b_neg_list, masks, B, device, dtype
        ) * (1.0 - all_miss)
        hesitancy = (vacuity + (1.0 - vacuity) * conflict).clamp(0.0, 1.0)

        ver_loss = torch.zeros((), device=device, dtype=dtype)
        for m in range(self.num_modalities):
            m_col = masks[:, m : m + 1]
            conf_w = (1.0 - u_feat_list[m]).detach()
            diff_sq = (p_feat_list[m] - mod_probs[m].detach()).pow(2) * conf_w
            ver_loss = ver_loss + (diff_sq * m_col).sum() / m_col.sum().clamp(min=1.0)
        ver_loss = ver_loss / float(max(1, self.num_modalities))

        self._last_alpha = alpha_fused
        self._last_mod_alphas = mod_alphas
        self._last_masks = masks
        self._last_extra_loss = ver_loss

        return {
            "prob": prob,
            "hesitancy": hesitancy,
            "vacuity": vacuity,
            "conflict": conflict,
            "alpha": alpha_fused,
            "mod_alphas": mod_alphas,
            "mod_probs": mod_probs,
            "modality_probs": mod_probs,
        }


class FUMLBaseline(_BaseEvidentialBaseline):
    """
    Baseline 11: Feature-level adaptation of Deep Fuzzy Multi-view Learning (FUML;
    Duan et al., ICML 2025).

    Maps per-modality representations through normalized Gaussian fuzzy membership
    granulation to parameterize view evidences and fuzzy reliability weights, then
    aggregates observed opinions with joint fuzzy-belief conflict quantification.
    """

    def __init__(
        self,
        modality_dims: List[int],
        embed_dim: int = 64,
        num_prototypes: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__(modality_dims=modality_dims, embed_dim=embed_dim)
        self.num_prototypes = num_prototypes
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(d, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            for d in modality_dims
        ])
        self.centers = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_prototypes, embed_dim) * 0.25)
            for _ in modality_dims
        ])
        self.log_sigmas = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_prototypes, embed_dim))
            for _ in modality_dims
        ])
        self.ev_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(embed_dim + num_prototypes, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 2),
            )
            for _ in modality_dims
        ])
        self.rel_heads = nn.ModuleList([
            nn.Linear(embed_dim + num_prototypes, 1) for _ in modality_dims
        ])

    def forward(
        self, modality_inputs: List[torch.Tensor], modality_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        masked_xs, masks, all_miss = _prepare_masks_and_inputs(
            modality_inputs, modality_masks, self.num_modalities
        )
        B = masked_xs[0].shape[0]
        device = masked_xs[0].device
        dtype = masked_xs[0].dtype

        ev_list: List[torch.Tensor] = []
        rel_list: List[torch.Tensor] = []
        fuzz_dist_list: List[torch.Tensor] = []
        b_pos_list: List[torch.Tensor] = []
        b_neg_list: List[torch.Tensor] = []
        mod_probs: List[torch.Tensor] = []
        mod_alphas: List[torch.Tensor] = []

        for m in range(self.num_modalities):
            m_col = masks[:, m : m + 1]
            h_m = torch.nan_to_num(
                self.encoders[m](masked_xs[m]), nan=0.0, posinf=0.0, neginf=0.0
            ) * m_col  # [B, D]
            sigma_m = F.softplus(self.log_sigmas[m]).clamp(min=0.25, max=10.0)
            sq_dist = ((h_m.unsqueeze(1) - self.centers[m]) / sigma_m).pow(2).mean(dim=-1)
            mu_fuzz = torch.exp(-0.5 * sq_dist.clamp(max=20.0)) * m_col  # [B, K]

            feat_cat = torch.cat([h_m, mu_fuzz], dim=-1)
            raw_ev = torch.nan_to_num(
                self.ev_heads[m](feat_cat), nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(max=20.0)
            ev_m = F.softplus(raw_ev) * m_col
            alpha_m = ev_m + 1.0
            s_m = alpha_m.sum(dim=-1, keepdim=True)
            rel_m = torch.sigmoid(
                torch.nan_to_num(self.rel_heads[m](feat_cat), nan=0.0, posinf=10.0, neginf=-10.0)
            ).clamp(min=0.05) * m_col

            mu_norm = mu_fuzz / mu_fuzz.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            ev_list.append(ev_m)
            rel_list.append(rel_m)
            fuzz_dist_list.append(mu_norm)
            b_pos_list.append(ev_m[:, 0:1] / s_m)
            b_neg_list.append(ev_m[:, 1:2] / s_m)
            mod_probs.append((alpha_m[:, 0:1] / s_m) * m_col + 0.5 * (1.0 - m_col))
            mod_alphas.append(alpha_m)

        rel_stack = torch.stack(rel_list, dim=1)  # [B, M, 1]
        ev_stack = torch.stack(ev_list, dim=1)  # [B, M, 2]
        m_exp = masks.unsqueeze(-1)
        obs_cnt = m_exp.sum(dim=1).clamp(min=1.0)
        rel_mean = ((rel_stack * m_exp).sum(dim=1) / obs_cnt).clamp(min=1e-4)  # [B, 1]
        rel_weight = (rel_stack / rel_mean.unsqueeze(1)).clamp(0.1, 2.5) * m_exp

        e_fused = (ev_stack * rel_weight).sum(dim=1) * (1.0 - all_miss)
        alpha_fused = e_fused + 1.0
        s_fused = alpha_fused.sum(dim=-1, keepdim=True)
        vacuity = (2.0 / s_fused).clamp(0.0, 1.0) * (1.0 - all_miss) + 1.0 * all_miss
        prob = (alpha_fused[:, 0:1] / s_fused).clamp(0.0, 1.0)
        prob = prob * (1.0 - all_miss) + 0.5 * all_miss

        c_belief = self._pairwise_belief_conflict(b_pos_list, b_neg_list, masks, B, device, dtype)
        fuzz_l1_sum = torch.zeros(B, 1, device=device, dtype=dtype)
        pair_cnt = torch.zeros(B, 1, device=device, dtype=dtype)
        for m in range(self.num_modalities):
            for n in range(m + 1, self.num_modalities):
                pair_obs = masks[:, m : m + 1] * masks[:, n : n + 1]
                d_fuzz = 0.5 * (fuzz_dist_list[m] - fuzz_dist_list[n]).abs().sum(dim=-1, keepdim=True)
                fuzz_l1_sum = fuzz_l1_sum + pair_obs * d_fuzz.clamp(0.0, 1.0)
                pair_cnt = pair_cnt + pair_obs
        has_pair = (pair_cnt > 0.5).to(dtype=dtype)
        c_fuzz = (fuzz_l1_sum / pair_cnt.clamp(min=1.0)).clamp(0.0, 1.0) * has_pair
        conflict = (0.5 * c_belief + 0.5 * c_fuzz).clamp(0.0, 1.0) * (1.0 - all_miss)
        hesitancy = (vacuity + (1.0 - vacuity) * conflict).clamp(0.0, 1.0)

        self._last_alpha = alpha_fused
        self._last_mod_alphas = mod_alphas
        self._last_masks = masks
        self._last_extra_loss = None

        return {
            "prob": prob,
            "hesitancy": hesitancy,
            "vacuity": vacuity,
            "conflict": conflict,
            "alpha": alpha_fused,
            "mod_alphas": mod_alphas,
            "mod_probs": mod_probs,
            "modality_probs": mod_probs,
        }
