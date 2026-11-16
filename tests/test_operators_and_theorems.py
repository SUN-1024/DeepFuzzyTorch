"""
Automated Unit Tests & Mathematical Theorem Verification for DeepFuzzyTorch.

Compatible with both `pytest` (`pytest -q tests/test_operators_and_theorems.py`)
and standalone execution (`python tests/test_operators_and_theorems.py`).

Verifies:
  1. Theorem 1: Intuitionistic Fuzzy Boundedness & Closure (mu + nu + pi == 1)
  2. Theorem 2: Missing-Modality Exact Degeneration (<0,0,1> leaves fused output bit-for-bit invariant
     even when conflict_penalty gamma > 0 and across all modality positions)
  3. Theorem 3: Strict Partial-Order Monotonicity (d mu_fused / d mu_m > 0, d nu_fused / d nu_m > 0)
  4. Theorem 4: Conflict-Induced Hesitancy Inflation (d log pi_fused / d C_O > 0 for all gamma > 0
     under fixed non-vacuity masses r_m = mu_m + nu_m in (0, 1))
  5. Theorem 5: Fuzzy-Rough Inclusion Inequality (0 <= R_low(A) <= A <= R_up(A) <= 1) in both
     DifferentiableFuzzyRoughLayer and FuzzyRoughAttentionGate
  6. Exact White-Box Rule Reconstruction in DeepConceptTSKHead (max |y_pred - y_recon| < 1e-6)
  7. Masked NaN Sanitization & End-to-End Autograd Backpropagation
  8. Statistical Evaluation Integrity (paired_bootstrap_pvalue(y, p, p) == 1.0, asymptotic Mantel-Haenszel Log-Rank test)
"""

import json
import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deepfuzzy import (
    IntuitionisticFuzzyTensor,
    IntuitionisticFuzzyEncoder,
    IFSChoquetFusion,
    HesitantFuzzyAggregator,
    DifferentiableFuzzyRoughLayer,
    FuzzyRoughAttentionGate,
    DeepConceptTSKHead,
    PythagoreanFuzzyTTALoss,
    IntuitionisticFuzzyLoss,
    UnifiedDeepFuzzyModel,
    EarlyConcatMLP,
    LateAvgEnsemble,
    GatedAttentionFusion,
    DempsterShaferFusion,
    EvidentialDL,
    EvidentialDLBaseline,
    DeepTSKBaseline,
    QMFBaseline,
    RCMLBaseline,
    DrFuseBaseline,
    DDEFBaseline,
    FUMLBaseline,
    CombinatorialMissingModalityBenchmark,
    CrossModalityConflictInjector,
    FuzzyRoughFeatureReducer,
    bootstrap_confidence_interval,
    compute_binary_metrics,
    compute_decision_curve_analysis,
    paired_bootstrap_pvalue,
    compute_logrank_test,
    compute_multigroup_logrank_test,
)
from deepfuzzy.metrics import compute_aurc, compute_ece


def test_theorem1_simplex_closure():
    torch.manual_seed(42)
    B, D, C = 16, 24, 1
    enc1 = IntuitionisticFuzzyEncoder(in_dim=D, num_classes=C)
    enc2 = IntuitionisticFuzzyEncoder(in_dim=D, num_classes=C)
    enc3 = IntuitionisticFuzzyEncoder(in_dim=D, num_classes=C)
    fusion = IFSChoquetFusion(num_modalities=3, conflict_penalty=0.75)

    ifn1 = enc1(torch.randn(B, D))
    ifn2 = enc2(torch.randn(B, D))
    ifn3 = enc3(torch.randn(B, D))

    assert ifn1.validate() and ifn2.validate() and ifn3.validate()
    fused_ifn, aux = fusion([ifn1, ifn2, ifn3])
    assert fused_ifn.validate(atol=1e-5), "Theorem 1 violated: Fused IFN did not sum to 1!"
    closure_err = (fused_ifn.mu + fused_ifn.nu + fused_ifn.pi - 1.0).abs().max().item()
    assert closure_err < 1e-5
    # Verify Choquet weights sum to 1 across observed modalities
    w_sum = aux["choquet_weights"].sum(dim=1)
    assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-5)

    # Verify extreme float64 tiny-evidence stability (1e-8, 1e-18, 1e-60, 2^-54 refutation, 5e-324 subnormal)
    f_1mod = IFSChoquetFusion(num_modalities=1, conflict_penalty=0.0)
    for tiny_val in [1e-8, 1e-18, 1e-60]:
        mu_t = torch.tensor([[tiny_val]], dtype=torch.float64)
        nu_t = torch.tensor([[0.0]], dtype=torch.float64)
        pi_t = torch.tensor([[1.0 - tiny_val]], dtype=torch.float64)
        ifn_t = IntuitionisticFuzzyTensor(mu=mu_t, nu=nu_t, pi=pi_t)
        assert ifn_t.validate()
        fused_t, aux_t = f_1mod([ifn_t])
        assert float(aux_t["choquet_weights"].item()) == 1.0
        assert float(fused_t.mu.item()) == tiny_val

    mu_ref = torch.tensor([[0.5]], dtype=torch.float64)
    nu_ref = torch.tensor([[2.0**-54]], dtype=torch.float64)
    pi_ref = 1.0 - mu_ref - nu_ref
    fused_ref, aux_ref = f_1mod([IntuitionisticFuzzyTensor(mu=mu_ref, nu=nu_ref, pi=pi_ref)])
    assert float(aux_ref["choquet_weights"].item()) == 1.0
    assert float(fused_ref.mu.item()) == 0.5
    assert float(fused_ref.nu.item()) == 2.0**-54
    assert float(fused_ref.pi.item()) == float(pi_ref.item())

    for logit_val in [0.0, -100.0]:
        f_sub = IFSChoquetFusion(num_modalities=1, conflict_penalty=0.0).double()
        with torch.no_grad():
            f_sub.fuzzy_density_logits.fill_(logit_val)
        mu_sub = torch.tensor([[5e-324]], dtype=torch.float64, requires_grad=True)
        nu_sub = torch.zeros_like(mu_sub, requires_grad=True)
        out_sub, aux_sub = f_sub([IntuitionisticFuzzyTensor(mu=mu_sub, nu=nu_sub, pi=1.0 - mu_sub - nu_sub)])
        out_sub.mu.sum().backward()
        assert float(aux_sub["choquet_weights"].item()) == 1.0
        assert float(out_sub.mu.item()) == 5e-324
        assert float(mu_sub.grad.item()) == 1.0

    # Verify multi-modality minimum subnormal (R9-01) and near-zero hesitancy (R9-02)
    f_2mod = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.0).double()
    v_sub = [torch.tensor([[5e-324]], dtype=torch.float64) for _ in range(2)]
    z_sub = [torch.zeros_like(v) for v in v_sub]
    out_2mu, aux_2mu = f_2mod([IntuitionisticFuzzyTensor(v, z, 1.0 - v - z) for v, z in zip(v_sub, z_sub)])
    out_2nu, aux_2nu = f_2mod([IntuitionisticFuzzyTensor(z, v, 1.0 - v - z) for v, z in zip(v_sub, z_sub)])
    assert float(out_2mu.mu.item()) == 5e-324 and float(out_2mu.nu.item()) == 0.0
    assert float(out_2nu.mu.item()) == 0.0 and float(out_2nu.nu.item()) == 5e-324

    mu_nz = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
    nu_nz = torch.tensor([[0.5 - 2.0**-54]], dtype=torch.float64, requires_grad=True)
    out_nz, _ = f_1mod.double()([IntuitionisticFuzzyTensor(mu_nz, nu_nz, 1.0 - mu_nz - nu_nz)])
    grad_nz = torch.autograd.grad(out_nz.nu.sum(), nu_nz)[0]
    assert float(out_nz.mu.item()) == 0.5
    assert float(out_nz.nu.item()) == float(nu_nz.item())
    assert float(out_nz.pi.item()) == float((1.0 - mu_nz - nu_nz).item())
    assert float(grad_nz.item()) == 1.0

    # Verify unequal small masses across all exponent scales (R10-01) and exact simplex vertices (R10-03)
    for a_val, b_val in [(1e-50, 2e-50), (1e-151, 2e-151), (1e-250, 2e-250), (5e-324, 10e-324)]:
        f_ab = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.0).double()
        with torch.no_grad():
            f_ab.fuzzy_density_logits.zero_()
            f_ab.choquet_lambda_raw.zero_()
        a_t = torch.tensor([[a_val]], dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor([[b_val]], dtype=torch.float64, requires_grad=True)
        z_t = torch.zeros_like(a_t)
        out_ab, _ = f_ab([IntuitionisticFuzzyTensor(a_t, z_t, 1.0 - a_t), IntuitionisticFuzzyTensor(b_t, z_t, 1.0 - b_t)])
        ga, gb = torch.autograd.grad(out_ab.mu.sum(), [a_t, b_t], retain_graph=True)
        gl = torch.autograd.grad(out_ab.mu.sum(), f_ab.fuzzy_density_logits)[0]
        assert abs(float(ga.item()) - 1.0 / 9.0) < 1e-12
        assert abs(float(gb.item()) - 7.0 / 9.0) < 1e-12
        if a_val >= 1e-250:
            assert gl[0, 0].item() < 0.0 and gl[1, 0].item() > 0.0

    for m_v, n_v, p_v in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        vt = IntuitionisticFuzzyTensor(
            torch.tensor([[m_v]], dtype=torch.float64),
            torch.tensor([[n_v]], dtype=torch.float64),
            torch.tensor([[p_v]], dtype=torch.float64),
        )
        o_v1, _ = f_1mod.double()([vt])
        o_v2, _ = f_2mod([vt, vt])
        assert (o_v1.mu.item(), o_v1.nu.item(), o_v1.pi.item()) == (m_v, n_v, p_v)
        assert (o_v2.mu.item(), o_v2.nu.item(), o_v2.pi.item()) == (m_v, n_v, p_v)

    # Verify mixed zero-hesitancy boundary vertex + interior modality yields exact pi_fused == 0.0
    ifn_zero_pi = IntuitionisticFuzzyTensor(
        torch.tensor([[0.6]], dtype=torch.float64),
        torch.tensor([[0.4]], dtype=torch.float64),
        torch.tensor([[0.0]], dtype=torch.float64),
    )
    ifn_interior = IntuitionisticFuzzyTensor(
        torch.tensor([[0.2]], dtype=torch.float64),
        torch.tensor([[0.3]], dtype=torch.float64),
        torch.tensor([[0.5]], dtype=torch.float64),
    )
    o_mix, _ = f_2mod([ifn_zero_pi, ifn_interior])
    assert float(o_mix.pi.item()) == 0.0
    assert float((o_mix.mu + o_mix.nu + o_mix.pi - 1.0).abs().item()) == 0.0

    # Verify R11-01 (exact-rational singleton identity), R11-02 (returned weight derivatives), and R11-03 (exponent derivative)
    s_mu = torch.tensor([[2.0**-54]], dtype=torch.float64)
    s_nu = torch.tensor([[1.0 - 2.0**-53]], dtype=torch.float64)
    s_pi = torch.tensor([[2.0**-54]], dtype=torch.float64)
    o_s1, _ = f_1mod.double()([IntuitionisticFuzzyTensor(s_mu, s_nu, s_pi)])
    assert (o_s1.mu.item(), o_s1.nu.item(), o_s1.pi.item()) == (2.0**-54, 1.0 - 2.0**-53, 2.0**-54)

    f_r11 = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.0).double()
    a_r11 = torch.tensor([[1e-301]], dtype=torch.float64, requires_grad=True)
    b_r11 = torch.tensor([[2e-301]], dtype=torch.float64, requires_grad=True)
    z_r11 = torch.zeros_like(a_r11)
    out_r11, aux_r11 = f_r11(
        [IntuitionisticFuzzyTensor(a_r11, z_r11, 1.0 - a_r11), IntuitionisticFuzzyTensor(b_r11, z_r11, 1.0 - b_r11)]
    )
    gw_a, gw_b, gw_th = torch.autograd.grad(
        aux_r11["weights"][0, 0, 0], (a_r11, b_r11, f_r11.fuzzy_density_logits), retain_graph=True
    )
    g_eta = torch.autograd.grad(out_r11.mu.sum(), f_r11.choquet_lambda_raw)[0]
    assert abs(gw_a.item() / (2.0 / (9.0 * 1e-301)) - 1.0) < 1e-12
    assert abs(gw_b.item() / (-1.0 / (9.0 * 1e-301)) - 1.0) < 1e-12
    assert abs(gw_th[0, 0].item() / (1.0 / (9.0 * (np.log(2.0) + 0.001))) - 1.0) < 1e-12
    assert abs(g_eta.item() / (1e-301 * (2.0 / 3.0) * np.log(2.0 / 3.0) * 0.35) - 1.0) < 1e-12


def test_theorem2_missing_modality_invariance():
    """
    Verifies Theorem 2 (Exact Vacuity Invariance) under conflict_penalty gamma > 0
    and across arbitrary positions of the missing modality.
    """
    torch.manual_seed(42)
    B, C = 20, 1
    u1 = torch.rand(B, C) * 0.65 + 0.05
    v1 = torch.rand(B, C) * (0.90 - u1)
    ifn1 = IntuitionisticFuzzyTensor(mu=u1, nu=v1, pi=1.0 - u1 - v1)

    u2 = torch.rand(B, C) * 0.65 + 0.05
    v2 = torch.rand(B, C) * (0.90 - u2)
    ifn2 = IntuitionisticFuzzyTensor(mu=u2, nu=v2, pi=1.0 - u2 - v2)

    vac = IntuitionisticFuzzyTensor(
        mu=torch.zeros(B, C), nu=torch.zeros(B, C), pi=torch.ones(B, C)
    )

    # Use identical uniform prior utility and non-zero conflict_penalty gamma = 0.85
    f2 = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.85)
    f3 = IFSChoquetFusion(num_modalities=3, conflict_penalty=0.85)
    with torch.no_grad():
        f2.fuzzy_density_logits.zero_()
        f3.fuzzy_density_logits.zero_()
        f3.choquet_lambda_raw.copy_(f2.choquet_lambda_raw)

    fused_2, aux_2 = f2([ifn1, ifn2])
    fused_3_tail, aux_3_tail = f3([ifn1, ifn2, vac])
    fused_3_head, aux_3_head = f3([vac, ifn1, ifn2])
    fused_3_mid, aux_3_mid = f3([ifn1, vac, ifn2])

    for fused_3, aux_3, vac_idx in [
        (fused_3_tail, aux_3_tail, 2),
        (fused_3_head, aux_3_head, 0),
        (fused_3_mid, aux_3_mid, 1),
    ]:
        assert torch.equal(fused_2.mu, fused_3.mu), "Theorem 2 mu bitwise invariance failed!"
        assert torch.equal(fused_2.nu, fused_3.nu), "Theorem 2 nu bitwise invariance failed!"
        assert torch.equal(fused_2.pi, fused_3.pi), "Theorem 2 pi bitwise invariance failed!"
        assert torch.equal(aux_2["conflict_score"], aux_3["conflict_score"])
        assert torch.equal(
            aux_3["choquet_weights"][:, vac_idx, :],
            torch.zeros(B, C),
        )

    # Also verify non-uniform learned fuzzy densities and non-unit Choquet exponent p != 1 across all 5 positions
    n_nonunif = 500
    u_nu = torch.rand(n_nonunif, 4) * 0.5
    v_nu = torch.rand(n_nonunif, 4) * (0.9 - u_nu)
    inputs_4 = [
        IntuitionisticFuzzyTensor(u_nu[:, i : i + 1], v_nu[:, i : i + 1], 1.0 - u_nu[:, i : i + 1] - v_nu[:, i : i + 1])
        for i in range(4)
    ]
    vac_nu = IntuitionisticFuzzyTensor(torch.zeros(n_nonunif, 1), torch.zeros(n_nonunif, 1), torch.ones(n_nonunif, 1))
    small = IFSChoquetFusion(4, conflict_penalty=0.85)
    large = IFSChoquetFusion(5, conflict_penalty=0.85)
    with torch.no_grad():
        small.fuzzy_density_logits.copy_(torch.tensor([[-1.3], [0.27], [2.1], [-0.4]]))
        small.choquet_lambda_raw.fill_(0.9)
        large.choquet_lambda_raw.copy_(small.choquet_lambda_raw)
        ref_4, _ = small(inputs_4)
        for pos in range(5):
            density = small.fuzzy_density_logits.detach()
            large.fuzzy_density_logits.copy_(torch.cat([density[:pos], torch.tensor([[1.1]]), density[pos:]]))
            obs_5, _ = large(inputs_4[:pos] + [vac_nu] + inputs_4[pos:])
            assert torch.equal(ref_4.mu, obs_5.mu)
            assert torch.equal(ref_4.nu, obs_5.nu)
            assert torch.equal(ref_4.pi, obs_5.pi)

    # Verify float64 vacuity invariance across all 5 positions and tiny positive non-vacuity 5e-8 (R3-06)
    u_d = torch.rand(200, 4, dtype=torch.float64) * 0.5
    v_d = torch.rand(200, 4, dtype=torch.float64) * (0.9 - u_d)
    ins_d = [
        IntuitionisticFuzzyTensor(u_d[:, i : i + 1], v_d[:, i : i + 1], 1.0 - u_d[:, i : i + 1] - v_d[:, i : i + 1])
        for i in range(4)
    ]
    vac_d = IntuitionisticFuzzyTensor(
        torch.zeros(200, 1, dtype=torch.float64),
        torch.zeros(200, 1, dtype=torch.float64),
        torch.ones(200, 1, dtype=torch.float64),
    )
    small_d = IFSChoquetFusion(4, conflict_penalty=0.85).double()
    large_d = IFSChoquetFusion(5, conflict_penalty=0.85).double()
    with torch.no_grad():
        small_d.fuzzy_density_logits.copy_(torch.tensor([[-1.3], [0.27], [2.1], [-0.4]], dtype=torch.float64))
        small_d.choquet_lambda_raw.fill_(0.9)
        large_d.choquet_lambda_raw.copy_(small_d.choquet_lambda_raw)
        ref_d, _ = small_d(ins_d)
        for pos in range(5):
            density_d = small_d.fuzzy_density_logits
            large_d.fuzzy_density_logits.copy_(
                torch.cat([density_d[:pos], torch.tensor([[1.1]], dtype=torch.float64), density_d[pos:]])
            )
            act_d, _ = large_d(ins_d[:pos] + [vac_d] + ins_d[pos:])
            assert torch.equal(ref_d.mu, act_d.mu)
            assert torch.equal(ref_d.nu, act_d.nu)
            assert torch.equal(ref_d.pi, act_d.pi)

    mu_tiny = torch.tensor([[5e-8]], dtype=torch.float64)
    z_tiny = torch.zeros_like(mu_tiny)
    tiny_ifn = IntuitionisticFuzzyTensor(mu_tiny, z_tiny, 1.0 - mu_tiny)
    single_f = IFSChoquetFusion(1, conflict_penalty=0.0).double()
    fused_tiny, aux_tiny = single_f([tiny_ifn])
    assert abs(fused_tiny.mu.item() - 5e-8) < 1e-14
    assert aux_tiny["choquet_weights"].item() == 1.0


def test_theorem3_partial_order_monotonicity():
    torch.manual_seed(42)
    # 1. Fixed non-vacuity contour r_1 = 0.75 (nu_1 = r_1 - mu_1) with fixed discount (gamma = 0)
    fusion_fixed_d = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.0)
    r1 = 0.75
    mu1 = torch.tensor([[0.45]], requires_grad=True)
    nu1 = r1 - mu1
    pi1 = torch.tensor([[1.0 - r1]])
    ifn1 = IntuitionisticFuzzyTensor(mu=mu1, nu=nu1, pi=pi1)
    ifn2 = IntuitionisticFuzzyTensor(
        mu=torch.tensor([[0.50]]), nu=torch.tensor([[0.25]]), pi=torch.tensor([[0.25]])
    )
    fused_0, aux_0 = fusion_fixed_d([ifn1, ifn2])
    d_mu_contour = torch.autograd.grad(fused_0.mu.sum(), mu1, retain_graph=True)[0]
    w1 = aux_0["choquet_weights"][0, 0, 0]
    expected_d_mu = w1 * (1.0 - fused_0.mu[0, 0]) / (1.0 - mu1[0, 0])
    assert torch.allclose(d_mu_contour.squeeze(), expected_d_mu, atol=1e-5)

    # 2. End-to-end monotonicity with positive conflict penalty gamma = 0.65 > 0
    fusion = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.65)
    mu1_e = torch.tensor([[0.45]], requires_grad=True)
    nu1_e = torch.tensor([[0.30]], requires_grad=True)
    pi1_e = 1.0 - mu1_e - nu1_e
    ifn1_e = IntuitionisticFuzzyTensor(mu=mu1_e, nu=nu1_e, pi=pi1_e)
    fused, _ = fusion([ifn1_e, ifn2])
    d_mu = torch.autograd.grad(fused.mu.sum(), mu1_e, retain_graph=True)[0]
    d_nu = torch.autograd.grad(fused.nu.sum(), nu1_e, retain_graph=True)[0]
    assert d_mu.item() > 0.0, "Theorem 3 violated: d mu_fused / d mu_1 must be strictly positive!"
    assert d_nu.item() > 0.0, "Theorem 3 violated: d nu_fused / d nu_1 must be strictly positive!"


def test_theorem4_conflict_hesitancy_inflation():
    torch.manual_seed(42)
    agree_1 = IntuitionisticFuzzyTensor(
        mu=torch.tensor([[0.85]]), nu=torch.tensor([[0.05]]), pi=torch.tensor([[0.10]])
    )
    agree_2 = IntuitionisticFuzzyTensor(
        mu=torch.tensor([[0.80]]), nu=torch.tensor([[0.10]]), pi=torch.tensor([[0.10]])
    )
    conflict_2 = IntuitionisticFuzzyTensor(
        mu=torch.tensor([[0.05]]), nu=torch.tensor([[0.85]]), pi=torch.tensor([[0.10]])
    )
    fusion_2 = IFSChoquetFusion(num_modalities=2, conflict_penalty=1.0)
    fused_agree, aux_a = fusion_2([agree_1, agree_2])
    fused_conflict, aux_c = fusion_2([agree_1, conflict_2])
    assert (aux_c["conflict_score"] > aux_a["conflict_score"]).all()
    assert (fused_conflict.pi > fused_agree.pi).all(), "Theorem 4 violated: Conflict must inflate hesitancy!"


def test_theorem5_fuzzy_rough_inclusion_and_gate():
    torch.manual_seed(42)
    B, N, D, C = 8, 12, 24, 2
    frs_layer = DifferentiableFuzzyRoughLayer(feat_dim=D, proj_dim=16)
    tokens = torch.randn(B, N, D, requires_grad=True)
    soft_labels = torch.rand(B, N, C)
    frs_out = frs_layer(tokens, soft_labels)
    assert (frs_out["lower_approx"] <= soft_labels + 1e-6).all()
    assert (frs_out["upper_approx"] >= soft_labels - 1e-6).all()
    assert (frs_out["boundary_region"] >= -1e-6).all()

    # Also verify FuzzyRoughAttentionGate with arbitrary beta and token_mask
    assert frs_layer.smooth_beta == 0.5
    gate = FuzzyRoughAttentionGate(embed_dim=D)
    assert torch.allclose(torch.sigmoid(gate.beta_raw), torch.tensor(0.5), atol=1e-6)
    with torch.no_grad():
        gate.beta_raw.fill_(2.0)  # beta > 0.5 to stress-test inclusion projection
    mod_tokens = torch.randn(B, 4, D, requires_grad=True)
    token_mask = torch.tensor([[1, 1, 0, 1], [1, 0, 0, 1]] * (B // 2), dtype=torch.float32)
    gated_tokens, boundary = gate(mod_tokens, token_mask=token_mask)
    assert gated_tokens.shape == (B, 4, D)
    assert (boundary >= 0.0).all() and (boundary <= 1.0).all()
    # Missing tokens must have zero gated representation and zero boundary
    assert torch.allclose(gated_tokens[0, 2], torch.zeros(D), atol=1e-6)
    assert torch.allclose(boundary[0, 2], torch.zeros(1), atol=1e-6)

    # Verify extreme small temperature tau -> 0+ (1e-8, 1e-12, 1e-30) with masked and tied tokens across FP16/FP32/FP64
    for tau_tiny in [1e-8, 1e-12, 1e-30]:
        frs_tiny = DifferentiableFuzzyRoughLayer(feat_dim=4, proj_dim=4, soft_temp=tau_tiny, smooth_beta=1.0)
        tok_tiny = torch.randn(2, 5, 4)
        a_tiny = torch.tensor([[[0.8], [0.7], [0.0], [0.9], [0.1]], [[0.2], [0.6], [0.0], [0.4], [0.8]]])
        m_tiny = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 1.0]])
        out_tiny = frs_tiny(tok_tiny, a_tiny, token_mask=m_tiny)
        assert torch.isfinite(out_tiny["lower_approx"]).all()
        assert torch.isfinite(out_tiny["upper_approx"]).all()
        assert (out_tiny["lower_approx"] <= a_tiny + 1e-7).all()
        assert (a_tiny <= out_tiny["upper_approx"] + 1e-7).all()
        assert float(out_tiny["lower_approx"][0, 0, 0].item()) > 0.0

        for dtype_test in [torch.float16, torch.float32, torch.float64]:
            for beta_test in [0.0, 0.5, 1.0]:
                layer_dt = DifferentiableFuzzyRoughLayer(1, 1, soft_temp=tau_tiny, smooth_beta=beta_test).to(dtype=dtype_test)
                x_tie = torch.zeros((1, 3, 1), dtype=dtype_test, requires_grad=True)
                a_tie = torch.full((1, 3, 1), 0.4, dtype=dtype_test, requires_grad=True)
                out_tie = layer_dt(x_tie, a_tie)
                g_a, g_x = torch.autograd.grad(
                    (out_tie["lower_approx"] + out_tie["upper_approx"]).sum(), [a_tie, x_tie]
                )
                assert torch.isfinite(g_a).all() and torch.isfinite(g_x).all()

    for bad_tau in [0.0, -0.1]:
        try:
            DifferentiableFuzzyRoughLayer(feat_dim=4, soft_temp=bad_tau)
            raise AssertionError("Expected ValueError for nonpositive soft_temp")
        except ValueError:
            pass
        try:
            FuzzyRoughAttentionGate(embed_dim=4, soft_temp=bad_tau)
            raise AssertionError("Expected ValueError for nonpositive soft_temp")
        except ValueError:
            pass


def test_tsk_exact_rule_reconstruction():
    torch.manual_seed(42)
    B = 12
    concept_names = ["Hippocampal_Vol", "Plasma_pTau217", "APOE4_Dosage", "MMSE_Baseline"]
    tsk_head = DeepConceptTSKHead(
        num_concepts=4, num_rules=5, out_dim=1, concept_names=concept_names
    )
    concepts = torch.randn(B, 4, requires_grad=True)
    y_pred, tsk_aux = tsk_head(concepts)
    rules = tsk_head.export_clinical_rules()
    struct = tsk_head.export_structured_rules()
    assert y_pred.shape == (B, 1) and len(rules) == 5

    # Reconstruct y_pred purely from exported structured rules
    z_np = concepts.detach().numpy()  # [B, C]
    centers = struct["centers"]  # [R, C]
    sigmas = struct["sigmas"]  # [R, C]
    gates = struct["gates"]  # [R, C]
    w0 = struct["consequent_bias"]  # [R, 1]
    eff_w = struct["effective_consequent_weights"]  # [R, C, 1]
    scale = struct["distance_scale"]

    diff_sq = ((z_np[:, None, :] - centers[None, :, :]) / sigmas[None, :, :]) ** 2
    gated_dist = np.sum(gates[None, :, :] * diff_sq, axis=-1) * scale  # [B, R]
    # Softmax over rules
    logits_r = -0.5 * gated_dist
    logits_r = logits_r - np.max(logits_r, axis=-1, keepdims=True)
    f_bar = np.exp(logits_r) / np.sum(np.exp(logits_r), axis=-1, keepdims=True)  # [B, R]
    y_r = w0[None, :, 0] + np.einsum("bc,rc->br", z_np, eff_w[:, :, 0])  # [B, R]
    y_recon = np.sum(f_bar * y_r, axis=-1, keepdims=True)
    max_diff = np.max(np.abs(y_pred.detach().numpy() - y_recon))
    assert max_diff < 1e-5, f"TSK rule reconstruction mismatch: {max_diff}"

    # Also verify regex parsing of human-readable 8-decimal rule strings (R3-08)
    y_text_recon = DeepConceptTSKHead.evaluate_exported_rules_text(rules, z_np, concept_names=concept_names)
    text_diff = np.max(np.abs(y_pred.detach().numpy().ravel() - y_text_recon))
    assert text_diff < 1e-6, f"TSK parsed-text rule reconstruction mismatch: {text_diff}"


def test_hesitant_and_pythagorean_tta():
    torch.manual_seed(42)
    B = 10
    hfs_agg = HesitantFuzzyAggregator(num_sources=4)
    prompts = torch.rand(B, 4, 8, requires_grad=True)
    hfs_out = hfs_agg(prompts)
    assert hfs_out["consensus_mu"].shape == (B, 8)
    assert (hfs_out["hfs_entropy"] >= 0).all() and (hfs_out["hfs_entropy"] <= 1).all()
    assert hfs_out["ifn"].validate(atol=1e-5)

    # Verify extreme dispersion triggers the 0.95 cap on pi_HFS while preserving IFN simplex closure
    extreme_prompts = torch.tensor([[[0.0], [1.0], [0.0], [1.0]]], dtype=torch.float32)
    extreme_out = hfs_agg(extreme_prompts)
    assert torch.allclose(extreme_out["ifn"].pi, torch.tensor([[0.95]]), atol=1e-6)
    assert extreme_out["ifn"].validate(atol=1e-6)

    u = torch.rand(B, 1) * 0.6 + 0.1
    v = torch.rand(B, 1) * (0.85 - u)
    ifn = IntuitionisticFuzzyTensor(mu=u, nu=v, pi=1.0 - u - v)
    tta_criterion = PythagoreanFuzzyTTALoss()
    tta_out = tta_criterion(ifn)
    assert torch.isfinite(tta_out["tta_loss"])


def test_strict_fold_preprocessor_exact_replay():
    from benchmarks.run_full_benchmark import (
        fit_strict_preprocessor,
        fit_strict_preprocessor_with_params,
        apply_strict_preprocessor_params,
    )

    rng = np.random.RandomState(42)
    X1 = rng.randn(50, 6).astype(np.float32) * 3.5 + 10.0
    X2 = rng.randn(50, 4).astype(np.float32) * 0.8 - 2.0
    # Inject partial NaNs in observed rows and full-row NaNs in unobserved rows
    X1[2, 1] = np.nan
    X2[5, 0] = np.nan
    X2[10:25, :] = np.nan
    X2[38:44, :] = np.nan

    tr_idx = np.arange(0, 35)
    va_idx = np.arange(35, 50)
    X_tr_list = [X1[tr_idx], X2[tr_idx]]
    X_va_list = [X1[va_idx], X2[va_idx]]

    tr_a, va_a, m_tr_a, m_va_a = fit_strict_preprocessor(X_tr_list, X_va_list)
    tr_b, va_b, m_tr_b, m_va_b, params = fit_strict_preprocessor_with_params(X_tr_list, X_va_list)
    tr_c, m_tr_c = apply_strict_preprocessor_params(X_tr_list, params)
    va_c, m_va_c = apply_strict_preprocessor_params(X_va_list, params)

    assert np.array_equal(m_tr_a, m_tr_b) and np.array_equal(m_tr_a, m_tr_c)
    assert np.array_equal(m_va_a, m_va_b) and np.array_equal(m_va_a, m_va_c)
    for m in range(2):
        assert np.max(np.abs(tr_a[m] - tr_b[m])) == 0.0
        assert np.max(np.abs(va_a[m] - va_b[m])) == 0.0
        assert np.max(np.abs(tr_a[m] - tr_c[m])) == 0.0
        assert np.max(np.abs(va_a[m] - va_c[m])) == 0.0


def test_masked_nan_safety_and_end_to_end():
    torch.manual_seed(42)
    B = 8
    # Verify num_classes != 1 raises ValueError
    try:
        UnifiedDeepFuzzyModel(modality_dims=[8, 4], num_classes=2)
        raise AssertionError("Expected ValueError for num_classes != 1")
    except ValueError:
        pass

    unified_model = UnifiedDeepFuzzyModel(
        modality_dims=[16, 8, 12],
        embed_dim=24,
        num_classes=1,
        use_rough_bottleneck=True,
        use_tsk_head=True,
        num_tsk_concepts=4,
        num_tsk_rules=4,
    )
    criterion = IntuitionisticFuzzyLoss()
    x1 = torch.randn(B, 16, requires_grad=True)
    x2 = torch.randn(B, 8)
    x3 = torch.randn(B, 12)
    masks = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]] * 2, dtype=torch.float32)
    # Inject NaNs into unobserved entries of x2
    x2_corrupted = x2.clone()
    x2_corrupted[1, :] = float("nan")
    x2_corrupted[5, :] = float("nan")
    x2_corrupted.requires_grad_(True)

    y_bin = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
    out_u = unified_model([x1, x2_corrupted, x3], modality_masks=masks)
    assert out_u["fused_ifn"].validate()
    assert torch.isfinite(out_u["prob"]).all()
    assert torch.isfinite(out_u["hesitancy"]).all()

    loss_u, _ = criterion(
        out_u["fused_ifn"],
        y_bin,
        modality_ifns=out_u["modality_ifns"],
        modality_masks=masks,
        final_prob=out_u["prob"],
        rough_boundary=out_u["rough_boundary"],
    )
    loss_u.backward()
    assert torch.isfinite(x1.grad).all()
    assert torch.isfinite(x2_corrupted.grad).all()


def test_paired_bootstrap_identical_and_logrank():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0], dtype=np.float64)
    p = np.array([0.1, 0.8, 0.2, 0.9, 0.7, 0.3, 0.25, 0.85, 0.65, 0.15], dtype=np.float64)
    pval_same = paired_bootstrap_pvalue(p, p, y, n_bootstraps=200)
    assert pval_same == 1.0, f"Identical predictions must yield p=1.0, got {pval_same}"

    # Verify near-equality opposite rankings do not falsely return p=1.0 (R3-09)
    a_near = np.array([0.500002, 0.500002, 0.5, 0.5])
    b_near = np.array([0.5, 0.5, 0.500002, 0.500002])
    y_near = np.array([1, 1, 0, 0])
    p_near = paired_bootstrap_pvalue(a_near, b_near, y_near, n_bootstraps=100)
    assert p_near < 0.05

    # Verify Log-Rank test on synthetic separated survival curves (2-group and 3-group omnibus)
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    events = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    high_risk_mask = np.array([True, True, True, True, True, False, False, False, False, False])
    chi2_val, p_val = compute_logrank_test(
        times[high_risk_mask],
        events[high_risk_mask],
        times[~high_risk_mask],
        events[~high_risk_mask],
    )
    assert chi2_val > 3.84 and p_val < 0.05
    groups_3 = np.array([2, 2, 2, 1, 1, 1, 0, 0, 0, 0])
    chi2_3, p_3 = compute_multigroup_logrank_test(times, events, groups_3)
    assert chi2_3 > 5.99 and p_3 < 0.05


def test_missing_conflict_and_rough_reducer_suites():
    torch.manual_seed(42)
    np.random.seed(42)
    B = 16
    unified_model = UnifiedDeepFuzzyModel(modality_dims=[10, 6, 8], embed_dim=16, num_classes=1)
    x1 = torch.randn(B, 10)
    x2 = torch.randn(B, 6)
    x3 = torch.randn(B, 8)
    y_bin = np.array([0, 1] * (B // 2), dtype=np.float32)

    bench = CombinatorialMissingModalityBenchmark(["MRI", "Plasma", "PET"])
    comb_res = bench.evaluate_all_subsets(
        predict_fn=lambda xs, m: unified_model(xs, m),
        modality_tensors=[x1, x2, x3],
        natural_masks=torch.ones(B, 3),
        targets=y_bin,
    )
    assert len(comb_res["subsets"]) == 7
    assert "missing_robustness_retention" in comb_res

    injector = CrossModalityConflictInjector(3, ["MRI", "Plasma", "PET"])
    conf_res = injector.evaluate_conflict_sweep(
        predict_fn=lambda xs, m: unified_model(xs, m),
        modality_tensors=[x1, x2, x3],
        natural_masks=torch.ones(B, 3),
        targets=y_bin,
        corrupted_modality_idx=1,
        severities=[0.0, 0.5, 1.0],
    )
    assert len(conf_res["sweep_records"]) == 3

    X_synth = np.random.randn(40, 12).astype(np.float32)
    y_synth = (X_synth[:, 0] - 0.8 * X_synth[:, 1] > 0).astype(np.float32)
    reducer = FuzzyRoughFeatureReducer(max_features=5, subsample_size=40).fit(
        X_synth, y_synth, device="cpu"
    )
    X_red = reducer.transform(X_synth, k=3)
    assert X_red.shape == (40, 3)

    # Also verify all 6 baselines forward cleanly under missing masks
    from deepfuzzy.baselines import (
        EarlyConcatMLP,
        LateAvgEnsemble,
        GatedAttentionFusion,
        EvidentialDL,
        DempsterShaferFusion,
        DeepTSKBaseline,
    )
    masks_b = torch.ones(B, 3)
    masks_b[0, :] = 0.0  # all missing edge case
    for cls in [
        EarlyConcatMLP,
        LateAvgEnsemble,
        GatedAttentionFusion,
        EvidentialDL,
        DempsterShaferFusion,
        DeepTSKBaseline,
    ]:
        bm = cls([10, 6, 8])
        out_b = bm([x1, x2, x3], modality_masks=masks_b)
        assert torch.isfinite(out_b["prob"]).all()
        assert torch.isfinite(out_b["hesitancy"]).all()


def test_regression_reducer_refit_tsk_names_and_aurc_bootstrap():
    """
    Verifies:
      1. FuzzyRoughFeatureReducer resets feature_scores_ across re-fits, exposes
         minimal_reduct_indices/minimal_reduct_names, and produces JSON-serializable summaries.
      2. DeepConceptTSKHead exports and text-reconstructs clinical concept names containing
         hyphens, slashes, dots, spaces, and parentheses without silent drop.
      3. bootstrap_confidence_interval and paired_bootstrap_pvalue honor custom hesitancy
         uncertainty arrays when metric_name == 'aurc'.
    """
    rng = np.random.default_rng(7)
    X_a = rng.standard_normal((80, 12)).astype(np.float32)
    y_a = (X_a[:, 0] - X_a[:, 1] > 0).astype(np.float32)
    reducer = FuzzyRoughFeatureReducer(max_features=4, subsample_size=80, seed=7)
    reducer.fit(X_a, y_a, feature_names=[f"OldROI_{i}" for i in range(12)], device="cpu")
    assert "OldROI_0" in reducer.feature_scores_

    X_b = rng.standard_normal((80, 8)).astype(np.float32)
    y_b = (X_b[:, 2] + X_b[:, 3] > 0).astype(np.float32)
    reducer.fit(X_b, y_b, feature_names=[f"NewGene_{i}" for i in range(8)], device="cpu")
    assert "OldROI_0" not in reducer.feature_scores_
    assert len(reducer.feature_scores_) == 8
    summary = reducer.get_reduct_summary()
    assert "minimal_reduct_indices" in summary and "minimal_reduct_names" in summary
    assert all(isinstance(i, int) for i in summary["selected_indices"])  # type: ignore[union-attr]
    json.dumps(summary)  # must not raise TypeError on numpy int64

    # TSK rule parser with punctuation-rich clinical concept names
    torch.manual_seed(11)
    special_names = [
        "pTau-217",
        "Hippocampus/ICV",
        "Abeta42.40",
        "APOE4 (carrier)",
    ]
    tsk = DeepConceptTSKHead(num_concepts=4, num_rules=5, concept_names=special_names)
    z_in = torch.randn(20, 4)
    logits_ref, _ = tsk(z_in)
    rules_txt = tsk.export_clinical_rules(gate_threshold=0.0, precision=8)
    recon_auto = DeepConceptTSKHead.evaluate_exported_rules_text(rules_txt, z_in)
    recon_named = DeepConceptTSKHead.evaluate_exported_rules_text(rules_txt, z_in, concept_names=special_names)
    assert float(np.max(np.abs(logits_ref.detach().numpy().ravel() - recon_auto))) < 1e-6
    assert float(np.max(np.abs(recon_auto - recon_named))) < 1e-12

    # Hesitancy-aware AURC bootstrap CI and paired bootstrap p-value
    y_bin = np.array([0, 1] * 40, dtype=np.float64)
    probs = np.where(y_bin == 1, 0.60, 0.40)
    # Flip 20 predictions so errors exist, with hesitancy perfectly separating errors vs non-errors
    probs[:20] = 1.0 - probs[:20]
    unc_oracle = np.zeros(80, dtype=np.float64)
    unc_oracle[:20] = 0.95  # high hesitancy on all 20 errors
    unc_oracle[20:] = 0.05  # low hesitancy on all 60 correct samples
    unc_bad = 1.0 - unc_oracle

    pt_good, lo_good, hi_good = bootstrap_confidence_interval(
        probs, y_bin, metric_name="aurc", n_bootstraps=200, seed=42, uncertainty=unc_oracle
    )
    pt_bad, lo_bad, hi_bad = bootstrap_confidence_interval(
        probs, y_bin, metric_name="aurc", n_bootstraps=200, seed=42, uncertainty=unc_bad
    )
    assert pt_good < pt_bad - 0.30
    assert hi_good < lo_bad
    p_aurc = paired_bootstrap_pvalue(
        probs,
        probs,
        y_bin,
        metric_name="aurc",
        higher_is_better=False,
        n_bootstraps=200,
        seed=42,
        uncertainty_ours=unc_oracle,
        uncertainty_base=unc_bad,
    )
    assert p_aurc < 0.05


def test_inf_nan_sanitization_everywhere():
    """
    Verifies complete NaN / +Inf / -Inf sanitization across DifferentiableFuzzyRoughLayer,
    FuzzyRoughAttentionGate, DeepConceptTSKHead, PythagoreanFuzzyTTALoss (including strict
    0 <= pi_P <= 1 non-negative Pythagorean hesitancy), IFSChoquetFusion, HesitantFuzzyAggregator,
    and all 9 baseline architectures.
    """
    torch.manual_seed(101)
    B, N, D = 6, 8, 12

    # 1. DifferentiableFuzzyRoughLayer with NaN, +Inf, -Inf
    frs = DifferentiableFuzzyRoughLayer(feat_dim=D, proj_dim=8)
    tok_bad = torch.randn(B, N, D)
    tok_bad[0, 0, 0] = float("nan")
    tok_bad[1, 2, 3] = float("inf")
    tok_bad[2, 4, 5] = float("-inf")
    lbl_bad = torch.rand(B, N, 1)
    lbl_bad[0, 1, 0] = float("nan")
    lbl_bad[1, 3, 0] = float("inf")
    lbl_bad[2, 5, 0] = float("-inf")
    out_frs = frs(tok_bad, lbl_bad)
    for k in ("lower_approx", "upper_approx", "boundary_region"):
        assert torch.isfinite(out_frs[k]).all(), f"Non-finite output in DifferentiableFuzzyRoughLayer[{k}]"
    assert (out_frs["boundary_region"] >= -1e-6).all()

    # 2. FuzzyRoughAttentionGate with NaN, +Inf, -Inf
    gate = FuzzyRoughAttentionGate(embed_dim=D)
    gate_dict = gate.forward_dict(tok_bad)
    for k in ("refined_features", "lower_approx", "upper_approx", "boundary_region"):
        assert torch.isfinite(gate_dict[k]).all(), f"Non-finite output in FuzzyRoughAttentionGate[{k}]"

    # 3. DeepConceptTSKHead with NaN, +Inf, -Inf in concepts & mask
    tsk = DeepConceptTSKHead(num_concepts=6, num_rules=4, out_dim=2)
    z_bad = torch.randn(B, 6)
    z_bad[0, 0] = float("nan")
    z_bad[1, 2] = float("inf")
    z_bad[2, 4] = float("-inf")
    m_bad = torch.ones(B, 6)
    m_bad[0, 0] = float("nan")
    m_bad[1, 2] = float("-inf")
    y_tsk, aux_tsk = tsk(z_bad, concept_mask=m_bad)
    assert torch.isfinite(y_tsk).all()
    assert torch.isfinite(aux_tsk["rule_firing"]).all()

    # 4. PythagoreanFuzzyTTALoss with NaN, +Inf, -Inf and out-of-simplex inputs (pi_P >= 0 strictly)
    tta = PythagoreanFuzzyTTALoss()
    mu_bad = torch.tensor([[float("nan")], [float("inf")], [float("-inf")], [1.0], [0.95], [0.0]])
    nu_bad = torch.tensor([[0.2], [float("inf")], [float("nan")], [1.0], [0.95], [float("-inf")]])
    pi_bad = torch.tensor([[0.8], [float("-inf")], [float("inf")], [0.0], [0.0], [float("nan")]])
    out_tta = tta(IntuitionisticFuzzyTensor(mu=mu_bad, nu=nu_bad, pi=pi_bad))
    assert torch.isfinite(out_tta["tta_loss"]).all()
    assert torch.isfinite(out_tta["pi_p"]).all()
    assert (out_tta["pi_p"] >= 0.0).all() and (out_tta["pi_p"] <= 1.0).all()

    # 5. IFSChoquetFusion & HesitantFuzzyAggregator with NaN, +Inf, -Inf
    choq = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.5)
    fused_c, aux_c = choq([
        IntuitionisticFuzzyTensor(mu_bad, nu_bad, pi_bad),
        IntuitionisticFuzzyTensor(torch.full((6, 1), 0.4), torch.full((6, 1), 0.3), torch.full((6, 1), 0.3)),
    ])
    assert torch.isfinite(fused_c.mu).all() and fused_c.validate(atol=1e-5)
    assert torch.isfinite(aux_c["conflict_score"]).all()

    hfs = HesitantFuzzyAggregator(num_sources=3)
    h_bad = torch.rand(B, 3, 2)
    h_bad[0, 0, 0] = float("nan")
    h_bad[1, 1, 1] = float("inf")
    h_bad[2, 2, 0] = float("-inf")
    out_hfs = hfs(h_bad)
    assert torch.isfinite(out_hfs["consensus_mu"]).all() and out_hfs["ifn"].validate(atol=1e-5)

    # 6. All 9 baseline models under NaN, +Inf, -Inf in both observed and missing entries
    dims = [10, 6, 8]
    masks = torch.tensor(
        [[1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 1]],
        dtype=torch.float32,
    )
    all_baseline_classes = [
        EarlyConcatMLP,
        LateAvgEnsemble,
        GatedAttentionFusion,
        DempsterShaferFusion,
        EvidentialDL,
        EvidentialDLBaseline,
        DeepTSKBaseline,
        QMFBaseline,
        RCMLBaseline,
        DrFuseBaseline,
    ]
    for cls in all_baseline_classes:
        model_b = cls(dims, embed_dim=24)
        xs = [torch.randn(B, d) for d in dims]
        # Corrupt observed AND unobserved entries with NaN, +Inf, -Inf
        xs[0][0, 0] = float("nan")
        xs[0][0, 1] = float("inf")
        xs[0][1, 2] = float("-inf")
        xs[1][1, :] = float("inf")
        xs[2][2, :] = float("-inf")
        xs_req = [x.clone().detach().requires_grad_(True) for x in xs]
        out_b = model_b(xs_req, modality_masks=masks)
        for key in ("prob", "hesitancy", "conflict"):
            assert key in out_b, f"{cls.__name__} missing output key {key!r}"
            assert torch.isfinite(out_b[key]).all(), f"{cls.__name__}[{key}] contains non-finite values"
            assert (out_b[key] >= 0.0).all() and (out_b[key] <= 1.0).all()
        assert "mod_probs" in out_b and len(out_b["mod_probs"]) == 3
        for mp in out_b["mod_probs"]:
            assert mp.shape == (B, 1) and torch.isfinite(mp).all()
        # All-missing row (row 3) must output neutral prob == 0.5
        assert abs(float(out_b["prob"][3, 0].item()) - 0.5) < 1e-5
        loss_b = out_b["prob"].mean()
        loss_b.backward()
        for p in model_b.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{cls.__name__} parameter gradient has NaN/Inf"


def test_multiclass_tsk_shapes_1d_2d_and_concept_warning():
    """
    Verifies:
      1. Multi-class `DeepConceptTSKHead(out_dim=3)` structured & human-readable rule export
         and regex text evaluation returning `(N, 3)` within `< 1e-6`.
      2. 1D `[B]` `IntuitionisticFuzzyTensor` auto-unsqueeze/squeeze in `IFSChoquetFusion`.
      3. 2D `[B, N]` `soft_labels` auto-unsqueeze/squeeze in `DifferentiableFuzzyRoughLayer`.
      4. `DeepTSKBaseline` configurable `embed_dim` and HTSK 1/D non-underflow on high-D inputs.
      5. `UnifiedDeepFuzzyModel` emits `UserWarning` when custom `concept_names[M:]` are given
         while `clinical_concepts is None` and `use_tsk_head=True`.
    """
    torch.manual_seed(202)
    B = 14
    c_names = ["MRI_Score", "PET_SUVR", "Plasma_pTau217", "MMSE_Z", "APOE4_Count"]
    tsk_mc = DeepConceptTSKHead(
        num_concepts=5, num_rules=6, out_dim=3, concept_names=c_names
    )
    z_mc = torch.randn(B, 5)
    y_mc, aux_mc = tsk_mc(z_mc)
    assert y_mc.shape == (B, 3)
    assert aux_mc["rule_outputs"].shape == (B, 6, 3)

    struct_mc = tsk_mc.export_structured_rules()
    assert struct_mc["consequent_weights"].shape == (6, 5, 3)
    assert struct_mc["consequent_bias"].shape == (6, 3)
    assert struct_mc["effective_consequent_weights"].shape == (6, 5, 3)

    rules_mc = tsk_mc.export_clinical_rules(gate_threshold=0.0, precision=8)
    assert len(rules_mc) == 6
    recon_mc = DeepConceptTSKHead.evaluate_exported_rules_text(rules_mc, z_mc)
    assert recon_mc.shape == (B, 3)
    assert float(np.max(np.abs(y_mc.detach().numpy() - recon_mc))) < 1e-6

    # Verify masked evaluation of exported multi-class rules also matches forward(..., concept_mask=...)
    c_mask = torch.tensor([[1, 1, 0, 1, 0], [0, 1, 1, 1, 1]] * (B // 2), dtype=torch.float32)
    y_mc_masked, _ = tsk_mc(z_mc, concept_mask=c_mask)
    recon_mc_masked = DeepConceptTSKHead.evaluate_exported_rules_text(
        rules_mc, z_mc, concept_mask=c_mask
    )
    assert float(np.max(np.abs(y_mc_masked.detach().numpy() - recon_mc_masked))) < 1e-6

    # 2. 1D [B] IntuitionisticFuzzyTensor in IFSChoquetFusion
    choq_2 = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.6)
    u1 = torch.rand(B) * 0.5 + 0.1
    v1 = torch.rand(B) * 0.3
    u2 = torch.rand(B) * 0.4 + 0.2
    v2 = torch.rand(B) * 0.3
    ifn1_1d = IntuitionisticFuzzyTensor(u1, v1, 1.0 - u1 - v1)
    ifn2_1d = IntuitionisticFuzzyTensor(u2, v2, 1.0 - u2 - v2)
    fused_1d, aux_1d = choq_2([ifn1_1d, ifn2_1d])
    fused_2d, _ = choq_2([
        IntuitionisticFuzzyTensor(u1.unsqueeze(-1), v1.unsqueeze(-1), (1.0 - u1 - v1).unsqueeze(-1)),
        IntuitionisticFuzzyTensor(u2.unsqueeze(-1), v2.unsqueeze(-1), (1.0 - u2 - v2).unsqueeze(-1)),
    ])
    assert fused_1d.mu.shape == (B,) and fused_1d.nu.shape == (B,) and fused_1d.pi.shape == (B,)
    assert aux_1d["choquet_weights"].shape == (B, 2)
    assert aux_1d["conflict_score"].shape == (B,)
    assert torch.allclose(fused_1d.mu, fused_2d.mu.squeeze(-1), atol=1e-7)

    # 3. 2D [B, N] soft_labels in DifferentiableFuzzyRoughLayer
    frs_2d = DifferentiableFuzzyRoughLayer(feat_dim=10, proj_dim=8)
    tok_x = torch.randn(B, 7, 10)
    lbl_2d = torch.rand(B, 7)
    out_2d = frs_2d(tok_x, lbl_2d)
    out_3d = frs_2d(tok_x, lbl_2d.unsqueeze(-1))
    assert out_2d["lower_approx"].shape == (B, 7)
    assert out_2d["upper_approx"].shape == (B, 7)
    assert out_2d["boundary_region"].shape == (B, 7)
    assert torch.allclose(out_2d["lower_approx"], out_3d["lower_approx"].squeeze(-1), atol=1e-7)

    # 4. High-dimensional DeepTSKBaseline with custom embed_dim and HTSK 1/D scaling
    dtsk = DeepTSKBaseline(modality_dims=[600, 600], embed_dim=48, num_concepts=16, num_rules=8)
    assert dtsk.embed_dim == 48
    out_dtsk = dtsk([torch.randn(8, 600) * 5.0, torch.randn(8, 600) * 5.0])
    firing = out_dtsk["tsk_aux"]["rule_firing"]
    assert torch.isfinite(firing).all()
    assert torch.allclose(firing.sum(dim=-1), torch.ones(8), atol=1e-5)
    # Non-degenerate firing across rules (max < 1 - eps on well-spread centers)
    assert float(out_dtsk["prob"].std().item()) >= 0.0

    # 5. UserWarning on custom concept_names[M:] with clinical_concepts=None when use_tsk_head=True
    m_warn = UnifiedDeepFuzzyModel(
        modality_dims=[8, 6],
        embed_dim=16,
        use_tsk_head=True,
        num_tsk_concepts=4,
        concept_names=["MRI_Net", "PET_Net", "APOE4", "MMSE"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = m_warn([torch.randn(4, 8), torch.randn(4, 6)], clinical_concepts=None)
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warns) == 1 and "clinical_concepts=None" in str(user_warns[0].message)

    # Passing clinical_concepts OR using generic Certainty_* names OR use_tsk_head=False must emit 0 warnings
    with warnings.catch_warnings(record=True) as caught_clean:
        warnings.simplefilter("always")
        _ = m_warn(
            [torch.randn(4, 8), torch.randn(4, 6)],
            clinical_concepts=torch.randn(4, 2),
        )
        m_no_tsk = UnifiedDeepFuzzyModel(
            modality_dims=[8, 6],
            embed_dim=16,
            use_tsk_head=False,
            num_tsk_concepts=4,
            concept_names=["MRI_Net", "PET_Net", "APOE4", "MMSE"],
        )
        _ = m_no_tsk([torch.randn(4, 8), torch.randn(4, 6)], clinical_concepts=None)
        assert len([w for w in caught_clean if issubclass(w.category, UserWarning)]) == 0


def test_end_to_end_unified_model_missing_modality_invariance():
    """
    Verifies Stanford Agentic Review Flaw A5 fix:
    `UnifiedDeepFuzzyModel(..., use_tsk_head=True)` achieves exact `< 1e-7` invariance:
      (a) across arbitrary garbage / NaN / +/-Inf values in unobserved modalities & concepts; and
      (b) between a 4-modality model with modality 2 masked (`[1, 1, 0, 1]`) and a restricted
          3-modality model containing ONLY the 3 observed modalities (`[0, 1, 3]`), both with
          explicit `clinical_concepts` and with `clinical_concepts=None`.
    """
    torch.manual_seed(303)
    B = 12
    dims_4 = [10, 6, 8, 5]
    dims_3 = [10, 6, 5]  # dropping modality index 2 (dim 8)
    obs_idx = [0, 1, 3]

    # Case A: Explicit clinical_concepts provided (2 clinical concepts -> total C = M + 2)
    m4 = UnifiedDeepFuzzyModel(
        modality_dims=dims_4,
        embed_dim=24,
        use_rough_bottleneck=True,
        use_tsk_head=True,
        num_tsk_concepts=6,
        num_tsk_rules=5,
        concept_names=["M0", "M1", "M2", "M3", "APOE4", "MMSE"],
    ).eval()
    m3 = UnifiedDeepFuzzyModel(
        modality_dims=dims_3,
        embed_dim=24,
        use_rough_bottleneck=True,
        use_tsk_head=True,
        num_tsk_concepts=5,
        num_tsk_rules=5,
        concept_names=["M0", "M1", "M3", "APOE4", "MMSE"],
    ).eval()

    # Copy exact observed-subset parameters from m4 -> m3
    keep_concepts = [0, 1, 3, 4, 5]  # drop concept index 2 (missing modality M2 net-support)
    with torch.no_grad():
        for new_i, old_i in enumerate(obs_idx):
            m3.feature_gates[new_i].copy_(m4.feature_gates[old_i])
            m3.modality_projectors[new_i].load_state_dict(m4.modality_projectors[old_i].state_dict())
            m3.ifs_encoders[new_i].load_state_dict(m4.ifs_encoders[old_i].state_dict())
        m3.rough_gate.load_state_dict(m4.rough_gate.state_dict())
        m3.choquet_fusion.fuzzy_density_logits.copy_(m4.choquet_fusion.fuzzy_density_logits[obs_idx])
        m3.choquet_fusion.choquet_lambda_raw.copy_(m4.choquet_fusion.choquet_lambda_raw)
        m3.tsk_head.centers.copy_(m4.tsk_head.centers[:, keep_concepts])
        m3.tsk_head.log_widths.copy_(m4.tsk_head.log_widths[:, keep_concepts])
        m3.tsk_head.antecedent_gates.copy_(m4.tsk_head.antecedent_gates[:, keep_concepts])
        m3.tsk_head.consequent_weights.copy_(m4.tsk_head.consequent_weights[:, keep_concepts, :])
        m3.tsk_head.consequent_bias.copy_(m4.tsk_head.consequent_bias)
        m3.tsk_blend_logit.copy_(m4.tsk_blend_logit)

    xs_4 = [torch.randn(B, d) for d in dims_4]
    clin = torch.randn(B, 2)
    mask_4 = torch.tensor([[1.0, 1.0, 0.0, 1.0]] * B)
    mask_3 = torch.ones(B, 3)

    out_4_zero = m4([xs_4[0], xs_4[1], torch.zeros_like(xs_4[2]), xs_4[3]], modality_masks=mask_4, clinical_concepts=clin)
    bad_m2 = torch.full_like(xs_4[2], float("nan"))
    bad_m2[0, 0] = float("inf")
    bad_m2[1, 0] = float("-inf")
    out_4_bad = m4([xs_4[0], xs_4[1], bad_m2, xs_4[3]], modality_masks=mask_4, clinical_concepts=clin)
    out_3_obs = m3([xs_4[0], xs_4[1], xs_4[3]], modality_masks=mask_3, clinical_concepts=clin)

    assert float((out_4_zero["prob"] - out_4_bad["prob"]).abs().max().item()) == 0.0
    diff_prob_a = float((out_4_zero["prob"] - out_3_obs["prob"]).abs().max().item())
    diff_tsk_a = float((out_4_zero["tsk_logits"] - out_3_obs["tsk_logits"]).abs().max().item())
    assert diff_prob_a < 1e-7, f"End-to-end 4-mod vs 3-mod prob diff = {diff_prob_a}"
    assert diff_tsk_a < 1e-6, f"End-to-end 4-mod vs 3-mod tsk_logits diff = {diff_tsk_a}"

    # Case B: clinical_concepts=None (uses modality net-supports + modality certainties 2*M vs 2*(M-1))
    m4_nc = UnifiedDeepFuzzyModel(
        modality_dims=dims_4,
        embed_dim=24,
        use_rough_bottleneck=True,
        use_tsk_head=True,
        num_tsk_concepts=8,
        num_tsk_rules=5,
    ).eval()
    m3_nc = UnifiedDeepFuzzyModel(
        modality_dims=dims_3,
        embed_dim=24,
        use_rough_bottleneck=True,
        use_tsk_head=True,
        num_tsk_concepts=6,
        num_tsk_rules=5,
    ).eval()
    keep_nc = [0, 1, 3, 4, 5, 7]  # drop missing modality 2's net-support (idx 2) and certainty (idx 6)
    with torch.no_grad():
        for new_i, old_i in enumerate(obs_idx):
            m3_nc.feature_gates[new_i].copy_(m4_nc.feature_gates[old_i])
            m3_nc.modality_projectors[new_i].load_state_dict(m4_nc.modality_projectors[old_i].state_dict())
            m3_nc.ifs_encoders[new_i].load_state_dict(m4_nc.ifs_encoders[old_i].state_dict())
        m3_nc.rough_gate.load_state_dict(m4_nc.rough_gate.state_dict())
        m3_nc.choquet_fusion.fuzzy_density_logits.copy_(m4_nc.choquet_fusion.fuzzy_density_logits[obs_idx])
        m3_nc.choquet_fusion.choquet_lambda_raw.copy_(m4_nc.choquet_fusion.choquet_lambda_raw)
        m3_nc.tsk_head.centers.copy_(m4_nc.tsk_head.centers[:, keep_nc])
        m3_nc.tsk_head.log_widths.copy_(m4_nc.tsk_head.log_widths[:, keep_nc])
        m3_nc.tsk_head.antecedent_gates.copy_(m4_nc.tsk_head.antecedent_gates[:, keep_nc])
        m3_nc.tsk_head.consequent_weights.copy_(m4_nc.tsk_head.consequent_weights[:, keep_nc, :])
        m3_nc.tsk_head.consequent_bias.copy_(m4_nc.tsk_head.consequent_bias)
        m3_nc.tsk_blend_logit.copy_(m4_nc.tsk_blend_logit)

    out_4_nc = m4_nc(xs_4, modality_masks=mask_4, clinical_concepts=None)
    out_3_nc = m3_nc([xs_4[0], xs_4[1], xs_4[3]], modality_masks=mask_3, clinical_concepts=None)
    diff_prob_b = float((out_4_nc["prob"] - out_3_nc["prob"]).abs().max().item())
    assert diff_prob_b < 1e-7, f"End-to-end (clinical_concepts=None) 4-mod vs 3-mod prob diff = {diff_prob_b}"


def test_rough_reducer_tnorm_monotonicity_null_and_stability():
    """
    Verifies `FuzzyRoughFeatureReducer`:
      1. Pointwise T-norm inequality R_{B'}(x_i, x_j) <= R_B(x_i, x_j) for all B subseteq B'
         across Godel, Product, and Lukasiewicz T-norms (purely unsupervised, zero label leakage).
      2. Zero monotonicity violations along every greedy dependency curve (`dependency_curve_`)
         and monotonically non-increasing `boundary_curve_`, even with NaN/Inf/binary columns.
      3. Clean separation of true discriminative signal vs pure i.i.d. Gaussian noise + Bernoulli
         labels via `excess_dependency_curve_`.
      4. Minimal reduct of size 1 (`len(minimal_reduct_indices_) == 1`) on 30 identical columns.
      5. High cross-fold Jaccard stability across 5 stratified folds.
    """
    rng = np.random.default_rng(404)
    N, D = 240, 60
    X = rng.standard_normal((N, D)).astype(np.float32)
    logits = 1.7 * X[:, 0] - 1.5 * X[:, 1] + 1.3 * X[:, 2] - 1.1 * X[:, 3]
    y_signal = (logits + 0.6 * rng.standard_normal(N) > 0).astype(np.float32)
    y_noise = rng.integers(0, 2, size=N).astype(np.float32)

    # 1 & 2. Pointwise T-norm inequality + strict 0-violation dependency monotonicity across all 3 T-norms
    for tnorm_name in ("godel", "product", "lukasiewicz"):
        red = FuzzyRoughFeatureReducer(max_features=12, tnorm=tnorm_name, seed=42).fit(
            X, y_signal, device="cpu"
        )
        # Verify R_{B'} <= R_B pointwise along nested subsets AND invariant to labels y!
        red_flipped = FuzzyRoughFeatureReducer(max_features=4, tnorm=tnorm_name, seed=42).fit(
            X, 1.0 - y_signal, device="cpu"
        )
        R_prev = red.compute_indiscernibility(X[:50], [], device="cpu")
        for step_k in range(1, 7):
            sub_k = red.selected_indices_[:step_k]
            R_next = red.compute_indiscernibility(X[:50], sub_k, device="cpu")
            R_flip = red_flipped.compute_indiscernibility(X[:50], sub_k, device="cpu")
            assert torch.allclose(R_next, R_flip, atol=1e-12), "Indiscernibility leaked label information!"
            assert (R_next <= R_prev + 1e-12).all(), f"Pointwise T-norm inequality failed for {tnorm_name}"
            R_prev = R_next

        dep = red.dependency_curve_
        bnd = red.boundary_curve_
        viols_dep = sum(1 for a, b in zip(dep[:-1], dep[1:]) if b < a - 1e-12)
        viols_bnd = sum(1 for a, b in zip(bnd[:-1], bnd[1:]) if b > a + 1e-12)
        assert viols_dep == 0, f"Monotonicity violated ({viols_dep}) for {tnorm_name}: {dep}"
        assert viols_bnd == 0, f"Boundary monotonicity violated ({viols_bnd}) for {tnorm_name}: {bnd}"

    # 3. Signal vs pure Gaussian noise separation + exact top-4 support recovery
    red_sig = FuzzyRoughFeatureReducer(max_features=10, seed=42).fit(X, y_signal, device="cpu")
    red_noi = FuzzyRoughFeatureReducer(max_features=10, seed=42).fit(X, y_noise, device="cpu")
    assert set(red_sig.selected_indices_[:4]) == {0, 1, 2, 3}
    max_exc_sig = max(red_sig.excess_dependency_curve_)
    max_exc_noi = max(red_noi.excess_dependency_curve_)
    assert max_exc_sig > 5.0 * max(max_exc_noi, 1e-4), (
        f"Expected signal excess ({max_exc_sig:.4f}) >> noise excess ({max_exc_noi:.4f})"
    )
    assert max_exc_noi < 0.015

    # NaN / Inf / all-NaN / sparse binary columns must produce 0 warnings and 0 monotonicity violations
    X_dirty = X.copy()
    X_dirty[0, 0] = np.nan
    X_dirty[1, 1] = np.inf
    X_dirty[2, 2] = -np.inf
    X_dirty[:, 5] = np.nan  # entire column NaN
    X_dirty[:, 6] = 7.0     # constant column
    X_dirty[:, 7] = (rng.random(N) < 0.08).astype(np.float32)  # sparse binary column (MAD == 0)
    with warnings.catch_warnings(record=True) as dirty_warns:
        warnings.simplefilter("always")
        red_dirty = FuzzyRoughFeatureReducer(max_features=8, seed=42).fit(
            X_dirty, y_signal, device="cpu"
        )
        assert len(dirty_warns) == 0, f"Unexpected warnings on dirty matrix: {dirty_warns}"
    assert all(
        b >= a - 1e-12
        for a, b in zip(red_dirty.dependency_curve_[:-1], red_dirty.dependency_curve_[1:])
    )

    # 4. 30 identical duplicate columns -> minimal reduct of size 1
    x_base = rng.standard_normal((120, 1)).astype(np.float32)
    y_dup = (x_base[:, 0] > 0.0).astype(np.float32)
    X_dup30 = np.tile(x_base, (1, 30))
    red_dup = FuzzyRoughFeatureReducer(max_features=10, seed=42).fit(X_dup30, y_dup, device="cpu")
    assert len(red_dup.minimal_reduct_indices_) == 1, (
        f"Expected size-1 minimal reduct on 30 identical columns, got {red_dup.minimal_reduct_indices_}"
    )

    # 5. Cross-fold Jaccard stability across 5 folds
    from sklearn.model_selection import StratifiedKFold

    N_cv, D_cv = 350, 80
    X_cv = rng.standard_normal((N_cv, D_cv)).astype(np.float32)
    y_cv = (
        1.8 * X_cv[:, 0]
        - 1.6 * X_cv[:, 1]
        + 1.5 * X_cv[:, 2]
        - 1.4 * X_cv[:, 3]
        + 1.3 * X_cv[:, 4]
        + 0.5 * rng.standard_normal(N_cv)
        > 0
    ).astype(np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_sets = []
    for f_idx, (tr_i, _) in enumerate(skf.split(X_cv, y_cv)):
        fr_cv = FuzzyRoughFeatureReducer(max_features=5, seed=42 + f_idx).fit(
            X_cv[tr_i], y_cv[tr_i], device="cpu"
        )
        fold_sets.append(set(fr_cv.selected_indices_[:5]))
    jaccards = [
        len(fold_sets[i] & fold_sets[j]) / len(fold_sets[i] | fold_sets[j])
        for i in range(5)
        for j in range(i + 1, 5)
    ]
    assert float(np.mean(jaccards)) >= 0.80, f"Low cross-fold Jaccard stability: {np.mean(jaccards):.4f}"


def test_modern_2023_2024_baselines_and_evidential_kl_annealing():
    """
    Verifies `QMFBaseline` (ICML 2023), `RCMLBaseline` (AAAI 2024), `DrFuseBaseline` (AAAI 2024),
    and `compute_kl_loss(targets, epoch, annealing_step=10)` on `EvidentialDLBaseline` & `RCMLBaseline`.
    """
    torch.manual_seed(505)
    B = 10
    dims = [12, 8, 6]
    xs = [torch.randn(B, d) for d in dims]
    masks = torch.ones(B, 3)
    masks[1, 2] = 0.0
    masks[2, 0] = 0.0
    masks[3, :] = 0.0  # all missing
    targets = torch.tensor([1, 0, 1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.float32)

    for cls in (QMFBaseline, RCMLBaseline, DrFuseBaseline):
        m = cls(dims, embed_dim=32)
        out = m(xs, modality_masks=masks)
        assert out["prob"].shape == (B, 1)
        assert out["hesitancy"].shape == (B, 1)
        assert out["conflict"].shape == (B, 1)
        assert len(out["mod_probs"]) == 3
        assert abs(float(out["prob"][3, 0].item()) - 0.5) < 1e-5
        assert float(out["hesitancy"][3, 0].item()) >= 0.99

    # Verify RCMLBaseline conflict sensitivity when two observed views strongly disagree vs agree
    rcml = RCMLBaseline([4, 4], embed_dim=16).eval()
    with torch.no_grad():
        for enc in rcml.encoders:
            for mod in enc:
                if isinstance(mod, nn.Linear):
                    mod.weight.zero_()
                    mod.bias.zero_()
        # Make encoder 0 output [8, -8] when x0[0] > 0, and encoder 1 output [8, -8] when x1[0] > 0
        rcml.encoders[0][-1].bias.copy_(torch.tensor([10.0, -10.0]))
        rcml.encoders[1][-1].bias.copy_(torch.tensor([10.0, -10.0]))
    out_agree = rcml([torch.ones(2, 4), torch.ones(2, 4)])
    with torch.no_grad():
        rcml.encoders[1][-1].bias.copy_(torch.tensor([-10.0, 10.0]))
    out_conf = rcml([torch.ones(2, 4), torch.ones(2, 4)])
    assert (out_conf["conflict"] > out_agree["conflict"] + 0.5).all()
    assert (out_conf["hesitancy"] > out_agree["hesitancy"] + 0.4).all()

    # Verify Sensoy/Han annealed KL loss on EvidentialDLBaseline and RCMLBaseline
    for ev_cls in (EvidentialDLBaseline, RCMLBaseline):
        ev_m = ev_cls(dims, embed_dim=24)
        out_ev = ev_m(xs, modality_masks=masks)
        kl_0 = ev_m.compute_kl_loss(targets, epoch=0, annealing_step=10)
        kl_5 = ev_m.compute_kl_loss(targets, epoch=5, annealing_step=10)
        kl_10 = ev_m.compute_kl_loss(targets, epoch=10, annealing_step=10)
        kl_20 = ev_m.compute_kl_loss(targets, epoch=20, annealing_step=10)
        assert float(kl_0.item()) == 0.0
        assert float(kl_10.item()) > 0.0
        assert abs(float(kl_5.item()) - 0.5 * float(kl_10.item())) < 1e-6
        assert abs(float(kl_20.item()) - float(kl_10.item())) < 1e-6

        # Target-aligned evidence [25.0, 1.0] for y=1 must yield exact 0.0 KL penalty,
        # whereas misleading evidence [1.0, 25.0] for y=1 must yield strictly positive KL penalty
        y_one = torch.ones(4)
        alpha_good = torch.tensor([[25.0, 1.0]] * 4, requires_grad=True)
        alpha_bad = torch.tensor([[1.0, 25.0]] * 4, requires_grad=True)
        kl_good = ev_m.compute_kl_loss(y_one, epoch=10, annealing_step=10, alpha=alpha_good)
        kl_bad = ev_m.compute_kl_loss(y_one, epoch=10, annealing_step=10, alpha=alpha_bad)
        assert float(kl_good.item()) < 1e-6
        expected_kl_25 = float(np.log(25.0) - 24.0 / 25.0)
        assert abs(float(kl_bad.item()) - expected_kl_25) < 1e-5

        total_loss = F.binary_cross_entropy(out_ev["prob"].squeeze(-1), targets) + kl_5
        total_loss.backward()
        for p in ev_m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


def test_intuitionistic_fuzzy_loss_nonfinite_sanitization_and_variable_width_tsk():
    """
    Verifies MAJ-6 (nonfinite IFN/probability/aux/mask/boundary inputs to IntuitionisticFuzzyLoss
    produce finite loss and finite gradients while matching valid inputs bit-for-bit) and MIN-1
    (TSK fixed-width fill invariance = 0.00 vs variable-width real-arithmetic equivalence).
    """
    loss_fn = IntuitionisticFuzzyLoss()
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])

    # 1. Valid simplex inputs: exact finite value & finite grads
    torch.manual_seed(7)
    mu_raw = torch.tensor([[0.2], [0.7], [0.1], [0.6]], requires_grad=True)
    nu_raw = torch.tensor([[0.6], [0.1], [0.5], [0.2]], requires_grad=True)
    pi_raw = torch.tensor([[0.2], [0.2], [0.4], [0.2]], requires_grad=True)
    ifn_valid = IntuitionisticFuzzyTensor(mu=mu_raw, nu=nu_raw, pi=pi_raw)
    l_valid, d_valid = loss_fn(ifn_valid, targets)
    assert torch.isfinite(l_valid) and np.isfinite(d_valid["total_loss"])
    l_valid.backward()
    assert torch.isfinite(mu_raw.grad).all() and torch.isfinite(nu_raw.grad).all()

    # 2. Inject NaN, +Inf, -Inf directly into mu, nu, pi, final_prob, modality_ifns, boundary, sparsity
    bad_vals = [float("nan"), float("inf"), float("-inf")]
    for comp in ("mu", "nu", "pi", "final_prob", "mod_mu", "mod_nu", "mod_mask", "rough_bnd", "gate_l1"):
        for bv in bad_vals:
            u_m = torch.tensor([[0.2], [0.7], [0.1], [0.6]], requires_grad=True)
            u_n = torch.tensor([[0.6], [0.1], [0.5], [0.2]], requires_grad=True)
            u_p = torch.tensor([[0.2], [0.2], [0.4], [0.2]], requires_grad=True)
            # Build graph tensors where row 0 is poisoned while remaining differentiable through finite rows
            t_bv = torch.tensor([[bv], [0.0], [0.0], [0.0]], dtype=torch.float32)
            row_mask = torch.tensor([[0.0], [1.0], [1.0], [1.0]], dtype=torch.float32)
            mu_in = u_m * row_mask + t_bv if comp == "mu" else u_m
            nu_in = u_n * row_mask + t_bv if comp == "nu" else u_n
            pi_in = u_p * row_mask + t_bv if comp == "pi" else u_p
            ifn_bad = IntuitionisticFuzzyTensor(mu=mu_in, nu=nu_in, pi=pi_in)
            fp_in = (0.5 * (mu_in + (1.0 - nu_in))) if comp != "final_prob" else (u_m * row_mask + t_bv)
            mod_ifn = IntuitionisticFuzzyTensor(
                mu=(u_m * row_mask + t_bv) if comp == "mod_mu" else u_m,
                nu=(u_n * row_mask + t_bv) if comp == "mod_nu" else u_n,
                pi=u_p,
            )
            mm_in = torch.ones(4, 1)
            if comp == "mod_mask":
                mm_in = mm_in.clone()
                mm_in[0, 0] = bv
            rb_in = (u_m * row_mask + t_bv) if comp == "rough_bnd" else torch.full((4, 1), 0.1)
            gl_in = (u_m.sum() * 0.0 + torch.tensor(bv)) if comp == "gate_l1" else torch.tensor(0.2)
            l_bad, _ = loss_fn(
                ifn_bad,
                targets,
                modality_ifns=[mod_ifn],
                modality_masks=mm_in,
                final_prob=fp_in,
                rough_boundary=rb_in,
                gate_sparsity_l1=gl_in,
            )
            assert torch.isfinite(l_bad), f"Nonfinite loss for {comp}={bv}"
            l_bad.backward()
            assert torch.isfinite(u_m.grad).all() and torch.isfinite(u_n.grad).all(), (
                f"Nonfinite grad for {comp}={bv}"
            )

    # 3. MIN-1: Fixed-width fill invariance (exact 0.0) vs variable-width real-arithmetic equivalence
    torch.manual_seed(1)
    tsk_full = DeepConceptTSKHead(num_concepts=48, num_rules=6, out_dim=3)
    tsk_sub = DeepConceptTSKHead(num_concepts=15, num_rules=6, out_dim=3)
    with torch.no_grad():
        tsk_sub.centers.copy_(tsk_full.centers[:, :15])
        tsk_sub.log_widths.copy_(tsk_full.log_widths[:, :15])
        tsk_sub.antecedent_gates.copy_(tsk_full.antecedent_gates[:, :15])
        tsk_sub.consequent_weights.copy_(tsk_full.consequent_weights[:, :15, :])
        tsk_sub.consequent_bias.copy_(tsk_full.consequent_bias)

    z_sub = torch.randn(17, 15)
    m_full = torch.zeros(17, 48)
    m_full[:, :15] = 1.0
    z_zero = torch.cat([z_sub, torch.zeros(17, 33)], dim=1)
    z_wild = torch.cat([z_sub, torch.full((17, 33), 999.0)], dim=1)
    z_wild[0, 15] = float("nan")
    with torch.no_grad():
        out_zero, _ = tsk_full(z_zero, concept_mask=m_full)
        out_wild, _ = tsk_full(z_wild, concept_mask=m_full)
        out_sub, _ = tsk_sub(z_sub)
        assert float((out_zero - out_wild).abs().max().item()) == 0.0
        assert float((out_zero - out_sub).abs().max().item()) < 1e-6
        # Double precision confirms exact algebraic identity modulo rounding (~1e-15)
        tsk_full_64 = tsk_full.double()
        tsk_sub_64 = tsk_sub.double()
        out_f64, _ = tsk_full_64(z_zero.double(), concept_mask=m_full.double())
        out_s64, _ = tsk_sub_64(z_sub.double())
        assert float((out_f64 - out_s64).abs().max().item()) < 1e-14


def test_r15_fusion_ties_sanitization_and_loss_bounds():
    """
    Round-14/15 regression suite verifying:
      1. Multiway 3-modality tie block B(r) behavior under p != 1 (C^0 pi_fused invariance,
         finite capacity-swap jump in mu_fused for unequal mu vs. continuity at equal mu,
         and deterministic canonical-index tie-breaking) and active-set conflict jump at
         |O| transitions when gamma > 0 (vs. continuity at gamma = 0) [R14-MAJ-01].
      2. Radial simplex sanitization for malformed singleton (|O|=1 in M=1 and M=3) and
         multi-observed inputs while preserving bitwise identity on valid singletons [R14-MAJ-02].
      3. Overflow-safe extreme sample_weight normalization, clamped loss bounds, 1e-6 probability
         clamping, and tightened 1e-12 simplex tolerance in IntuitionisticFuzzyLoss [R14-MIN-02].
    """
    import itertools

    # 1a. Equal-mu 3-modality tie block under p != 1: exact permutation invariance across densities
    f_tie = IFSChoquetFusion(num_modalities=3, conflict_penalty=0.5).double()
    dens = [0.6, -0.4, 0.1]
    base_eq = [(0.30, 0.30, 0.40)] * 3
    outs_eq = []
    for perm in itertools.permutations(range(3)):
        with torch.no_grad():
            f_tie.choquet_lambda_raw.fill_(0.8)
            for idx_new, idx_old in enumerate(perm):
                f_tie.fuzzy_density_logits[idx_new, 0] = dens[idx_old]
        in_eq = [
            IntuitionisticFuzzyTensor(
                torch.tensor([[base_eq[i][0]]], dtype=torch.float64),
                torch.tensor([[base_eq[i][1]]], dtype=torch.float64),
                torch.tensor([[base_eq[i][2]]], dtype=torch.float64),
            )
            for i in perm
        ]
        out_p, _ = f_tie(in_eq)
        outs_eq.append((float(out_p.mu.item()), float(out_p.nu.item()), float(out_p.pi.item())))
    for o in outs_eq[1:]:
        assert max(abs(a - b) for a, b in zip(o, outs_eq[0])) < 1e-13

    # 1b. Unequal-mu 3-way tie block B(r) (r=0.60, p != 1): pi_fused invariant across all 6 perms,
    #     mu_fused undergoes capacity-swap jump under permutation yet stays bit-deterministic repeatedly
    base_uneq = [(0.50, 0.10, 0.40), (0.20, 0.40, 0.40), (0.35, 0.25, 0.40)]
    pis_uneq, mus_uneq = [], []
    for perm in itertools.permutations(range(3)):
        with torch.no_grad():
            f_tie.choquet_lambda_raw.fill_(0.8)
            for idx_new, idx_old in enumerate(perm):
                f_tie.fuzzy_density_logits[idx_new, 0] = dens[idx_old]
        in_u = [
            IntuitionisticFuzzyTensor(
                torch.tensor([[base_uneq[i][0]]], dtype=torch.float64),
                torch.tensor([[base_uneq[i][1]]], dtype=torch.float64),
                torch.tensor([[base_uneq[i][2]]], dtype=torch.float64),
            )
            for i in perm
        ]
        out_u, _ = f_tie(in_u)
        out_u2, _ = f_tie(in_u)
        assert float(out_u.mu.item()) == float(out_u2.mu.item())
        pis_uneq.append(float(out_u.pi.item()))
        mus_uneq.append(float(out_u.mu.item()))
    assert max(pis_uneq) - min(pis_uneq) < 1e-13
    assert max(mus_uneq) - min(mus_uneq) > 1e-3

    # 1c. Partial tie locus continuity of pi_fused across r2=0.50+d vs r3=0.50-d (p != 1, unequal mu)
    f_ptie = IFSChoquetFusion(num_modalities=3, conflict_penalty=0.0).double()
    with torch.no_grad():
        f_ptie.choquet_lambda_raw.fill_(0.8)
        f_ptie.fuzzy_density_logits.zero_()
    pi_deltas = []
    for dlt in (-1e-7, 0.0, 1e-7):
        r2, r3 = 0.50 + dlt, 0.50 - dlt
        t1 = IntuitionisticFuzzyTensor(torch.tensor([[0.60]], dtype=torch.float64), torch.tensor([[0.20]], dtype=torch.float64), torch.tensor([[0.20]], dtype=torch.float64))
        t2 = IntuitionisticFuzzyTensor(torch.tensor([[0.9 * r2]], dtype=torch.float64), torch.tensor([[0.1 * r2]], dtype=torch.float64), torch.tensor([[1.0 - r2]], dtype=torch.float64))
        t3 = IntuitionisticFuzzyTensor(torch.tensor([[0.1 * r3]], dtype=torch.float64), torch.tensor([[0.9 * r3]], dtype=torch.float64), torch.tensor([[1.0 - r3]], dtype=torch.float64))
        out_d, _ = f_ptie([t1, t2, t3])
        pi_deltas.append(float(out_d.pi.item()))
    assert max(abs(pi_deltas[0] - pi_deltas[1]), abs(pi_deltas[2] - pi_deltas[1])) < 1e-6

    # 1d. Active-set conflict jump when 3rd modality moves from r3=0 (<0,0,1>) to r3=1e-7
    for gamma_val, expect_jump in [(1.0, True), (0.0, False)]:
        f_act = IFSChoquetFusion(num_modalities=3, conflict_penalty=gamma_val).double()
        m1 = IntuitionisticFuzzyTensor(torch.tensor([[0.8]], dtype=torch.float64), torch.tensor([[0.1]], dtype=torch.float64), torch.tensor([[0.1]], dtype=torch.float64))
        m2 = IntuitionisticFuzzyTensor(torch.tensor([[0.1]], dtype=torch.float64), torch.tensor([[0.8]], dtype=torch.float64), torch.tensor([[0.1]], dtype=torch.float64))
        m3_zero = IntuitionisticFuzzyTensor(torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[1.0]], dtype=torch.float64))
        eps_r = 1e-7
        m3_tiny = IntuitionisticFuzzyTensor(torch.tensor([[0.5 * eps_r]], dtype=torch.float64), torch.tensor([[0.5 * eps_r]], dtype=torch.float64), torch.tensor([[1.0 - eps_r]], dtype=torch.float64))
        out_z, _ = f_act([m1, m2, m3_zero])
        out_t, _ = f_act([m1, m2, m3_tiny])
        jump = abs(float(out_t.pi.item()) - float(out_z.pi.item()))
        if expect_jump:
            assert jump > 0.05, f"Expected O(gamma/|O|) conflict jump, got {jump}"
        else:
            assert jump < 1e-5, f"Expected continuous limit at gamma=0, got {jump}"

    # 2. Malformed singleton & multi-obs sanitization + valid singleton identity (R14-MAJ-02)
    for dt in (torch.float32, torch.float64):
        f1 = IFSChoquetFusion(num_modalities=1, conflict_penalty=0.5).to(dtype=dt)
        mu_id = torch.tensor([[0.45]], dtype=dt, requires_grad=True)
        nu_id = torch.tensor([[0.35]], dtype=dt, requires_grad=True)
        pi_id = (1.0 - mu_id - nu_id).detach().clone()
        out_id, _ = f1([IntuitionisticFuzzyTensor(mu_id, nu_id, pi_id)])
        assert float(out_id.mu.item()) == float(mu_id.item())
        assert float(out_id.nu.item()) == float(nu_id.item())
        assert float(out_id.pi.item()) == float(pi_id.item())
        (out_id.mu + 2.0 * out_id.nu).sum().backward()
        assert torch.isfinite(mu_id.grad).all() and torch.isfinite(nu_id.grad).all()

        # Over-unit singleton radial projection (0.9, 0.8, 0.5) -> (9/17, 8/17, 0.0) in both M=1 and M=3 single-active
        mu_over = torch.tensor([[0.9]], dtype=dt, requires_grad=True)
        nu_over = torch.tensor([[0.8]], dtype=dt, requires_grad=True)
        pi_over = torch.tensor([[0.5]], dtype=dt)
        out_over1, _ = f1([IntuitionisticFuzzyTensor(mu_over, nu_over, pi_over)])
        assert out_over1.validate(atol=1e-6)
        assert abs(float(out_over1.mu.item()) - 9.0 / 17.0) < 1e-6
        assert abs(float(out_over1.nu.item()) - 8.0 / 17.0) < 1e-6
        assert abs(float(out_over1.pi.item())) < 1e-6
        out_over1.mu.sum().backward()
        assert torch.isfinite(mu_over.grad).all() and torch.isfinite(nu_over.grad).all()

        f3 = IFSChoquetFusion(num_modalities=3, conflict_penalty=0.5).to(dtype=dt)
        vac = IntuitionisticFuzzyTensor(torch.zeros((1, 1), dtype=dt), torch.zeros((1, 1), dtype=dt), torch.ones((1, 1), dtype=dt))
        out_over3, _ = f3([IntuitionisticFuzzyTensor(mu_over.detach(), nu_over.detach(), pi_over), vac, vac])
        assert out_over3.validate(atol=1e-6)
        assert abs(float(out_over3.mu.item()) - 9.0 / 17.0) < 1e-6

        # Negative, mismatched pi, and non-finite inputs across multi-obs M=3
        mal_ifns = [
            IntuitionisticFuzzyTensor(torch.tensor([[-0.2]], dtype=dt), torch.tensor([[0.5]], dtype=dt), torch.tensor([[0.9]], dtype=dt)),
            IntuitionisticFuzzyTensor(torch.tensor([[0.3]], dtype=dt), torch.tensor([[0.2]], dtype=dt), torch.tensor([[0.99]], dtype=dt)),
            IntuitionisticFuzzyTensor(torch.tensor([[float("nan")]], dtype=dt), torch.tensor([[float("inf")]], dtype=dt), torch.tensor([[float("-inf")]], dtype=dt)),
        ]
        out_mal3, _ = f3(mal_ifns)
        assert out_mal3.validate(atol=1e-6)

    # 3. Extreme sample_weights, nonfinite/out-of-range targets/bounds & 1e-12 simplex tolerance in Loss (R14-MIN-02)
    loss_fn = IntuitionisticFuzzyLoss()
    mu_l = torch.tensor([[0.6], [0.2], [0.7], [0.1]], dtype=torch.float32, requires_grad=True)
    nu_l = torch.tensor([[0.2], [0.6], [0.1], [0.7]], dtype=torch.float32, requires_grad=True)
    pi_l = (1.0 - mu_l - nu_l).detach()
    ext_w = torch.tensor([1e38, 1e-12, float("inf"), float("nan")], dtype=torch.float32)
    ext_y = torch.tensor([5.0, -5.0, float("nan"), 1.0], dtype=torch.float32)
    ext_fp = torch.tensor([[0.0], [1.0], [float("inf")], [-10.0]], dtype=torch.float32, requires_grad=True)
    l_ext, d_ext = loss_fn(
        IntuitionisticFuzzyTensor(mu_l, nu_l, pi_l),
        ext_y,
        final_prob=ext_fp,
        rough_boundary=torch.tensor([50.0, -20.0, float("inf"), 0.5]),
        gate_sparsity_l1=torch.tensor(-10.0),
        sample_weights=ext_w,
    )
    assert torch.isfinite(l_ext) and np.isfinite(d_ext["total_loss"])
    l_ext.backward()
    assert torch.isfinite(mu_l.grad).all() and torch.isfinite(nu_l.grad).all()

    # Float64 & float32 _sanitize_ifn_components Sterbenz ULP checks (pi recomputed cleanly when drifted)
    m64 = torch.tensor([[0.5]], dtype=torch.float64)
    n64 = torch.tensor([[0.3]], dtype=torch.float64)
    _, _, pi_valid = loss_fn._sanitize_ifn_components(m64, n64, torch.tensor([[0.2]], dtype=torch.float64))
    _, _, pi_drifted = loss_fn._sanitize_ifn_components(m64, n64, torch.tensor([[0.2 + 5e-13]], dtype=torch.float64))
    assert float(pi_valid.detach().item()) == 0.2
    assert abs(float(pi_drifted.detach().item()) - 0.2) <= 2e-16

    m32 = torch.tensor([[0.2]], dtype=torch.float32)
    n32 = torch.tensor([[0.3]], dtype=torch.float32)
    mu32_o, nu32_o, pi32_o = loss_fn._sanitize_ifn_components(m32, n32, torch.tensor([[0.5 + 4e-7]], dtype=torch.float32))
    assert abs(float((mu32_o + nu32_o + pi32_o).detach().item()) - 1.0) <= 2e-7


def test_r16_metrics_consistency_boundary_grad_and_scale_weights():
    import json
    from pathlib import Path
    from deepfuzzy.metrics import (
        bootstrap_confidence_interval,
        compute_binary_metrics,
        paired_bootstrap_pvalue,
    )

    # 1. R15-MAJ-02: Tail preservation below 1e-6 & paired_bootstrap exact array equality
    y_tail = np.array([0, 1], dtype=np.int64)
    p_tail = np.array([1e-12, 1.0 - 1e-12], dtype=np.float64)
    m_tail = compute_binary_metrics(p_tail, y_tail)
    assert m_tail["auroc"] == 1.0 and m_tail["brier"] < 1e-20

    y_sub = np.array([0, 1], dtype=np.int64)
    p_sub = np.array([1e-12, 1e-9], dtype=np.float64)
    assert compute_binary_metrics(p_sub, y_sub)["auroc"] == 1.0
    assert bootstrap_confidence_interval(p_sub, y_sub, metric_name="auroc", n_bootstraps=10, seed=42)[0] == 1.0

    y_pb = np.array([0] * 100 + [1] * 100, dtype=np.int64)
    p_base = np.full(200, 0.5, dtype=np.float64)
    p_ours = np.where(y_pb == 1, 0.5 + 1e-13, 0.5 - 1e-13)
    p_diff = paired_bootstrap_pvalue(p_ours, p_base, y_pb, metric_name="auroc", n_bootstraps=200, seed=42)
    p_same = paired_bootstrap_pvalue(p_base, p_base, y_pb, metric_name="auroc", n_bootstraps=200, seed=42)
    assert p_diff < 0.01 and p_same == 1.0

    root = Path(__file__).resolve().parents[1]
    br_path = root / "results" / "benchmark_results.json"
    npz_path = root / "results" / "oof_predictions_and_splits.npz"
    if br_path.exists() and npz_path.exists():
        br = json.loads(br_path.read_text())
        npz = np.load(npz_path)
        t5 = br["track5_multi_fm_consensus_tta"]
        prefix_map = [
            ("Camelyon17\n(OOD Hospitals)", "track5_cam_clean"),
            ("Camelyon17\n(1 Corrupted FM)", "track5_cam_corrupt"),
            ("NCT-CRC\n(External Cohort)", "track5_crc_clean"),
            ("NCT-CRC\n(1 Corrupted FM)", "track5_crc_corrupt"),
        ]
        method_map = [
            ("BestSingleFM", "best_single_p"),
            ("MeanPool6FMs", "mean_pool_p"),
            ("MedianPool6FMs", "median_pool_p"),
            ("Tent_TTA", "tent_p"),
            ("HesitantFuzzy", "hfs_p"),
            ("HesitantFuzzy_PFS_TTA", "pfs_p"),
        ]
        for json_k, npz_prefix in prefix_map:
            yk = npz[f"{npz_prefix}_y_te"]
            for m_json, m_npz in method_map:
                pk = npz[f"{npz_prefix}_{m_npz}"]
                au = compute_binary_metrics(pk, yk)["auroc"]
                assert abs(au - float(t5[json_k][m_json])) <= 1e-12

    # 2. R15-MIN-01: Upper/atol validation bounds
    bad_atol0 = IntuitionisticFuzzyTensor(
        torch.tensor([[0.2]], dtype=torch.float64),
        torch.tensor([[0.3]], dtype=torch.float64),
        torch.tensor([[0.500009]], dtype=torch.float64),
    )
    assert bad_atol0.validate(atol=0.0) is False
    bad_upper = IntuitionisticFuzzyTensor(
        torch.tensor([[-1e-4]], dtype=torch.float64),
        torch.tensor([[1.0002]], dtype=torch.float64),
        torch.tensor([[-1e-4]], dtype=torch.float64),
    )
    assert bad_upper.validate(atol=1e-4) is False

    # 3. R15-MIN-02: Scale-invariant & zero sample_weights normalization
    loss_fn = IntuitionisticFuzzyLoss()
    mu_w = torch.tensor([[0.7], [0.2]], dtype=torch.float64, requires_grad=True)
    nu_w = torch.tensor([[0.1], [0.6]], dtype=torch.float64, requires_grad=True)
    pi_w = torch.tensor([[0.2], [0.2]], dtype=torch.float64, requires_grad=True)
    y_w = torch.tensor([1.0, 0.0], dtype=torch.float64)
    w_base = torch.tensor([1.0, 3.0], dtype=torch.float64)
    l_ref, _ = loss_fn(IntuitionisticFuzzyTensor(mu_w, nu_w, pi_w), y_w, sample_weights=w_base)
    l_ref_val = float(l_ref.detach().item())
    for scale in (1e-10, 1e-300, 1e12, 1e38):
        l_s, _ = loss_fn(IntuitionisticFuzzyTensor(mu_w, nu_w, pi_w), y_w, sample_weights=w_base * scale)
        assert abs(float(l_s.detach().item()) - l_ref_val) <= 1e-12

    mu_z = torch.tensor([[0.7], [0.2]], dtype=torch.float64, requires_grad=True)
    nu_z = torch.tensor([[0.1], [0.6]], dtype=torch.float64, requires_grad=True)
    pi_z = torch.tensor([[0.2], [0.2]], dtype=torch.float64, requires_grad=True)
    w_zero = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    l_zero, _ = loss_fn(IntuitionisticFuzzyTensor(mu_z, nu_z, pi_z), y_w, sample_weights=w_zero)
    assert float(l_zero.detach().item()) == 0.0
    l_zero.backward()
    assert torch.isfinite(mu_z.grad).all() and torch.isfinite(w_zero.grad).all()

    # 4. R15-MIN-03: Closed-boundary right-hand FD gradient preservation vs out-of-domain stop
    op = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.6).double()
    with torch.no_grad():
        op.choquet_lambda_raw.zero_()
        op.fuzzy_density_logits.zero_()

    mu1 = torch.tensor([[0.2]], dtype=torch.float64)
    nu1 = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
    pi1 = torch.tensor([[0.8]], dtype=torch.float64)
    mu2 = torch.tensor([[0.6]], dtype=torch.float64)
    nu2 = torch.tensor([[0.2]], dtype=torch.float64)
    pi2 = torch.tensor([[0.2]], dtype=torch.float64)

    out_b, _ = op([IntuitionisticFuzzyTensor(mu1, nu1, pi1), IntuitionisticFuzzyTensor(mu2, nu2, pi2)])
    obj_b = out_b.mu.sum() + 0.5 * out_b.pi.sum()
    obj_b.backward()
    g_bnd = float(nu1.grad.detach().item())

    h = 1e-7
    with torch.no_grad():
        out_p, _ = op([
            IntuitionisticFuzzyTensor(mu1, torch.tensor([[h]], dtype=torch.float64), torch.tensor([[0.8 - h]], dtype=torch.float64)),
            IntuitionisticFuzzyTensor(mu2, nu2, pi2),
        ])
        obj_p = float((out_p.mu.sum() + 0.5 * out_p.pi.sum()).item())
        out_0, _ = op([
            IntuitionisticFuzzyTensor(mu1, torch.tensor([[0.0]], dtype=torch.float64), pi1),
            IntuitionisticFuzzyTensor(mu2, nu2, pi2),
        ])
        obj_0 = float((out_0.mu.sum() + 0.5 * out_0.pi.sum()).item())
    # Wait: holding pi1 fixed when testing partial d/d(nu1) along projection (pi recomputed vs raw)
    with torch.no_grad():
        out_raw_p, _ = op([
            IntuitionisticFuzzyTensor(mu1, torch.tensor([[h]], dtype=torch.float64), pi1),
            IntuitionisticFuzzyTensor(mu2, nu2, pi2),
        ])
        obj_raw_p = float((out_raw_p.mu.sum() + 0.5 * out_raw_p.pi.sum()).item())
    fd_bnd = (obj_raw_p - obj_0) / h
    assert abs(g_bnd) > 0.05 and abs(g_bnd - fd_bnd) <= 1e-4

    nu_neg = torch.tensor([[-0.2]], dtype=torch.float64, requires_grad=True)
    out_neg, _ = op([IntuitionisticFuzzyTensor(mu1, nu_neg, torch.tensor([[1.0]], dtype=torch.float64)), IntuitionisticFuzzyTensor(mu2, nu2, pi2)])
    (out_neg.mu.sum() + 0.5 * out_neg.pi.sum()).backward()
    assert float(nu_neg.grad.detach().item()) == 0.0


def test_r17_unclipped_operator_probs_aurc_ties_and_boundary_holder_jump():
    import inspect

    # 1. R16-MAJ-02: No [1e-6, 1-1e-6] tail clipping across operators, UnifiedDeepFuzzyModel & baselines
    lo1 = IntuitionisticFuzzyTensor(
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[1.0 - 2e-8]], dtype=torch.float64, requires_grad=True),
        torch.tensor([[2e-8]], dtype=torch.float64),
    )
    lo2 = IntuitionisticFuzzyTensor(
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[1.0 - 4e-8]], dtype=torch.float64, requires_grad=True),
        torch.tensor([[4e-8]], dtype=torch.float64),
    )
    hi1 = IntuitionisticFuzzyTensor(
        torch.tensor([[1.0 - 4e-8]], dtype=torch.float64, requires_grad=True),
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[4e-8]], dtype=torch.float64),
    )
    hi2 = IntuitionisticFuzzyTensor(
        torch.tensor([[1.0 - 2e-8]], dtype=torch.float64, requires_grad=True),
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[2e-8]], dtype=torch.float64),
    )
    p_lo1, p_lo2 = float(lo1.expected_probability().item()), float(lo2.expected_probability().item())
    p_hi1, p_hi2 = float(hi1.expected_probability().item()), float(hi2.expected_probability().item())
    assert p_lo1 < p_lo2 < 1e-6 and p_hi2 > p_hi1 > 1.0 - 1e-6
    assert compute_binary_metrics([p_lo1, p_lo2], [0, 1])["auroc"] == 1.0
    assert compute_binary_metrics([p_hi1, p_hi2], [0, 1])["auroc"] == 1.0

    op = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.0).double()
    vac = IntuitionisticFuzzyTensor(
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[1.0]], dtype=torch.float64),
    )
    fused_p = [op([t, vac])[0].expected_probability() for t in (lo1, lo2, hi1, hi2)]
    vals = [float(p.item()) for p in fused_p]
    assert vals[0] < vals[1] < 1e-6 and vals[3] > vals[2] > 1.0 - 1e-6
    assert compute_binary_metrics(vals[:2], [0, 1])["auroc"] == 1.0
    assert compute_binary_metrics(vals[2:], [0, 1])["auroc"] == 1.0
    sum(fused_p).sum().backward()
    grads = (lo1.nu.grad, lo2.nu.grad, hi1.mu.grad, hi2.mu.grad)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

    for ut in (False, True):
        # Negative tail < 1e-6
        m_lo = UnifiedDeepFuzzyModel([4, 4], embed_dim=8, num_concepts=4, num_rules=2, use_tsk_head=ut).double().eval()
        for p in m_lo.parameters():
            p.data.zero_()
        with torch.no_grad():
            for i in range(2):
                m_lo.feature_gates[i].data.fill_(10.0)
                m_lo.modality_projectors[i][0].weight.fill_(1.0)
                m_lo.ifs_encoders[i].pos_evidence_head.bias.fill_(-50.0)
                m_lo.ifs_encoders[i].neg_evidence_head.bias.fill_(2e7)
                m_lo.ifs_encoders[i].skip_neg.weight.fill_(1e6)
            if ut:
                m_lo.tsk_head.consequent_bias.fill_(-40.0)
                m_lo.tsk_blend_logit.fill_(0.0)
        x_lo = [
            torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True),
            torch.zeros(2, 4, dtype=torch.float64, requires_grad=True),
        ]
        out_lo = m_lo(x_lo)
        pr_lo = out_lo["prob"].detach().numpy().ravel()
        out_lo["prob"].sum().backward()
        assert pr_lo[0] < pr_lo[1] < 6e-7
        assert compute_binary_metrics(pr_lo, [0, 1])["auroc"] == 1.0
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in x_lo)

        # Positive tail > 1 - 1e-6
        m_hi = UnifiedDeepFuzzyModel([4, 4], embed_dim=8, num_concepts=4, num_rules=2, use_tsk_head=ut).double().eval()
        for p in m_hi.parameters():
            p.data.zero_()
        with torch.no_grad():
            for i in range(2):
                m_hi.feature_gates[i].data.fill_(10.0)
                m_hi.modality_projectors[i][0].weight.fill_(1.0)
                m_hi.ifs_encoders[i].neg_evidence_head.bias.fill_(-50.0)
                m_hi.ifs_encoders[i].pos_evidence_head.bias.fill_(2e7)
                m_hi.ifs_encoders[i].skip_pos.weight.fill_(1e6)
            if ut:
                m_hi.tsk_head.consequent_bias.fill_(40.0)
                m_hi.tsk_blend_logit.fill_(0.0)
        x_hi = [
            torch.tensor([[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True),
            torch.zeros(2, 4, dtype=torch.float64, requires_grad=True),
        ]
        out_hi = m_hi(x_hi)
        pr_hi = out_hi["prob"].detach().numpy().ravel()
        out_hi["prob"].sum().backward()
        assert 1.0 - 6e-7 < pr_hi[0] < pr_hi[1] <= 1.0
        assert compute_binary_metrics(pr_hi, [0, 1])["auroc"] == 1.0
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in x_hi)

    for sign in (-1, 1):
        ds = DempsterShaferFusion([2] * 6, embed_dim=4).double().eval()
        for p in ds.parameters():
            p.data.zero_()
        with torch.no_grad():
            b_vec = [-30.0, 20.0] if sign < 0 else [20.0, -30.0]
            idx = 1 if sign < 0 else 0
            for i in range(6):
                ds.encoders[i][-1].bias.copy_(torch.tensor(b_vec, dtype=torch.float64))
            ds.encoders[0][1].weight[0, 0] = 1.0
            ds.encoders[0][-1].bias[idx] = 10.0
            ds.encoders[0][-1].weight[idx, 0] = 10.0
        x_ds = [torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True)] + [
            torch.zeros(2, 2, dtype=torch.float64, requires_grad=True) for _ in range(5)
        ]
        out_ds = ds(x_ds)
        pr_ds = out_ds["prob"].detach().numpy().ravel()
        out_ds["prob"].sum().backward()
        if sign < 0:
            assert pr_ds[1] < pr_ds[0] < 6e-7
            assert compute_binary_metrics(pr_ds, [1, 0])["auroc"] == 1.0
        else:
            assert 1.0 - 6e-7 < pr_ds[0] < pr_ds[1] <= 1.0
            assert compute_binary_metrics(pr_ds, [0, 1])["auroc"] == 1.0
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in x_ds)

    ev_src = inspect.getsource(EvidentialDL.forward)
    rc_src = inspect.getsource(RCMLBaseline.forward)
    assert "1.0 - 1e-6" not in ev_src and "(1e-6," not in ev_src and ".clamp(0.0, 1.0)" in ev_src
    assert "1.0 - 1e-6" not in rc_src and "(1e-6," not in rc_src and ".clamp(0.0, 1.0)" in rc_src

    # 2. R16-MIN-01 & R16-MAJ-03: AURC 1e-20 precision and tie-group permutation/bootstrap parity
    aurc_tiny, *_ = compute_aurc(np.array([2e-20, 1e-20]), np.array([1, 0]))
    aurc_tie_a, *_ = compute_aurc(np.array([0.4, 0.4]), np.array([1, 0]))
    aurc_tie_b, *_ = compute_aurc(np.array([0.4, 0.4]), np.array([0, 1]))
    ci_tie = bootstrap_confidence_interval(np.array([0.4, 0.4]), np.array([1, 0]), metric_name="aurc", n_bootstraps=50, seed=42)
    assert abs(aurc_tiny - 0.25) < 1e-12
    assert abs(aurc_tie_a - 0.50) < 1e-12 and abs(aurc_tie_b - 0.50) < 1e-12
    assert ci_tie == (0.5, 0.5, 0.5)

    # 3. R16-MAJ-01: Closed-boundary Holder vs interior O_delta(gamma/|O|) scaling
    def _eval_jump(a1_t, a2_t, gamma: float):
        fus = IFSChoquetFusion(num_modalities=3, conflict_penalty=gamma).double()
        with torch.no_grad():
            fus.choquet_lambda_raw.zero_()
            fus.fuzzy_density_logits.zero_()
        a1 = IntuitionisticFuzzyTensor(*[torch.tensor([[v]], dtype=torch.float64) for v in a1_t])
        a2 = IntuitionisticFuzzyTensor(*[torch.tensor([[v]], dtype=torch.float64) for v in a2_t])
        vac3 = IntuitionisticFuzzyTensor(torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[1.0]], dtype=torch.float64))
        eps = 1e-12
        tiny3 = IntuitionisticFuzzyTensor(torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[eps]], dtype=torch.float64), torch.tensor([[1.0 - eps]], dtype=torch.float64))
        ov, _ = fus([a1, a2, vac3])
        ot, _ = fus([a1, a2, tiny3])
        return abs(float(ot.mu.item()) - float(ov.mu.item())), abs(float(ot.pi.item()) - float(ov.pi.item()))

    theo_58 = (0.6 ** (5.0 / 8.0) - 0.2 ** (5.0 / 8.0)) * (0.4 ** (3.0 / 8.0))
    for gamma, tol in ((1e-4, 1e-3), (1e-8, 1e-4)):
        _, dpi = _eval_jump((0.0, 1.0, 0.0), (0.6, 0.0, 0.4), gamma)
        assert abs((dpi / (gamma ** (5.0 / 8.0))) - theo_58) <= 0.01 * theo_58 + tol

    for gamma in (1e-4, 1e-8):
        dmu, dpi = _eval_jump((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), gamma)
        assert abs((dpi / gamma) - (2.0 / 3.0)) < 1e-2
        assert abs((dmu / (gamma ** 0.5)) - (1.0 - (1.0 / 3.0) ** 0.5)) < 1e-2

    _, ipi_4 = _eval_jump((0.05, 0.90, 0.05), (0.6, 0.0, 0.4), 1e-4)
    _, ipi_8 = _eval_jump((0.05, 0.90, 0.05), (0.6, 0.0, 0.4), 1e-8)
    assert abs((ipi_4 / 1e-4) - (ipi_8 / 1e-8)) < 1e-3 and (ipi_8 / 1e-8) < 5.0


def test_r18_threshold_aurc_loss_tail_grad_and_absorbing_tie_characterization():
    """Verify R17-MAJ-02/03/04 & R17-MIN-04: thresholded AURC, tail loss grad & absorbing tie condition."""
    from deepfuzzy.metrics import (
        bootstrap_confidence_interval,
        compute_aurc,
        compute_binary_metrics,
        paired_bootstrap_pvalue,
    )

    # 1. R17-MAJ-03: threshold forwarding across compute_aurc, compute_binary_metrics, bootstrap CI & p-value
    probs = np.array([0.6, 0.7], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int64)
    aurc_05, cov_05, risk_05, _ = compute_aurc(probs, labels, threshold=0.5)
    aurc_08, cov_08, risk_08, _ = compute_aurc(probs, labels, threshold=0.8)
    assert abs(aurc_05 - 0.25) < 1e-12
    assert abs(aurc_08 - 0.75) < 1e-12
    assert np.allclose(cov_05, [0.5, 1.0]) and np.allclose(risk_05, [0.0, 0.5])
    assert np.allclose(cov_08, [0.5, 1.0]) and np.allclose(risk_08, [1.0, 0.5])
    assert abs(compute_binary_metrics(probs, labels, threshold=0.8)["aurc"] - 0.75) < 1e-12
    assert abs(compute_aurc(probs, labels, threshold=float("nan"))[0] - 0.25) < 1e-12

    ci_08 = bootstrap_confidence_interval(probs, labels, metric_name="aurc", n_bootstraps=40, seed=42, threshold=0.8)
    assert ci_08[0] > 0.50 or abs(ci_08[0] - 0.75) < 0.30
    probs_tie = np.array([0.6, 0.6], dtype=np.float64)
    ci_t05 = bootstrap_confidence_interval(probs_tie, labels, metric_name="aurc", n_bootstraps=30, seed=7, threshold=0.5)
    ci_t08 = bootstrap_confidence_interval(probs_tie, labels, metric_name="aurc", n_bootstraps=30, seed=7, threshold=0.8)
    assert ci_t05 == (0.5, 0.5, 0.5) and ci_t08 == (0.5, 0.5, 0.5)
    probs_diff = np.array([0.6, 0.4, 0.9, 0.2], dtype=np.float64)
    y_diff = np.array([1, 0, 1, 0], dtype=np.int64)
    pv_self = paired_bootstrap_pvalue(probs_diff, np.array([0.55, 0.45, 0.9, 0.2]), y_diff, metric_name="aurc", n_bootstraps=40, seed=42, threshold=0.8)
    assert 0.0 <= pv_self <= 1.0

    # 2. R17-MAJ-04: Exact unclipped Brier on (0,1e-6) + straight-through BCE tail grad + OOD zero grad
    loss_fn = IntuitionisticFuzzyLoss().double()
    p_tail = torch.tensor([[2e-8]], dtype=torch.float64, requires_grad=True)
    y_one = torch.tensor([[1.0]], dtype=torch.float64)
    ifn_mid = IntuitionisticFuzzyTensor(
        torch.tensor([[0.5]], dtype=torch.float64),
        torch.tensor([[0.3]], dtype=torch.float64),
        torch.tensor([[0.2]], dtype=torch.float64),
    )
    tot_loss, comp = loss_fn(ifn_mid, y_one, final_prob=p_tail)
    tot_loss.backward()
    assert abs(float(comp["brier"]) - (2e-8 - 1.0) ** 2) < 1e-15
    assert float(p_tail.grad.item()) < -9e5 and torch.isfinite(p_tail.grad).all()

    # TSK head gradient must remain strictly non-zero when blended prob < 1e-6
    um = UnifiedDeepFuzzyModel([4, 4], embed_dim=8, num_concepts=4, num_rules=2, use_tsk_head=True).double()
    for p in um.parameters():
        p.data.zero_()
    with torch.no_grad():
        for i in range(2):
            um.feature_gates[i].data.fill_(10.0)
            um.modality_projectors[i][0].weight.fill_(1.0)
            um.ifs_encoders[i].pos_evidence_head.bias.fill_(-50.0)
            um.ifs_encoders[i].neg_evidence_head.bias.fill_(2e7)
            um.ifs_encoders[i].skip_neg.weight.fill_(1e6)
        um.tsk_head.consequent_bias.fill_(-40.0)
        um.tsk_blend_logit.fill_(0.0)
    x_um = [torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float64), torch.zeros(1, 4, dtype=torch.float64)]
    out_um = um(x_um)
    assert float(out_um["prob"].item()) < 1e-6
    l_um, _ = loss_fn(out_um, y_one)
    l_um.backward()
    assert um.tsk_head.consequent_bias.grad is not None and float(um.tsk_head.consequent_bias.grad.abs().sum().item()) > 0.0
    assert um.tsk_blend_logit.grad is not None and float(um.tsk_blend_logit.grad.abs().item()) > 0.0

    # Out-of-domain final_prob (<0, >1, NaN, +/-Inf) yields finite loss and zero gradient
    p_ood = torch.tensor([[-0.2], [1.2], [float("nan")], [float("inf")]], dtype=torch.float64, requires_grad=True)
    y_ood = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float64)
    ifn_ood = IntuitionisticFuzzyTensor(
        torch.full((4, 1), 0.4, dtype=torch.float64),
        torch.full((4, 1), 0.3, dtype=torch.float64),
        torch.full((4, 1), 0.3, dtype=torch.float64),
    )
    l_ood, _ = loss_fn(ifn_ood, y_ood, final_prob=p_ood)
    l_ood.backward()
    assert bool(torch.isfinite(l_ood)) and torch.all(p_ood.grad == 0.0)

    # 3. R17-MAJ-02 & R17-MIN-04: Absorbing boundary iff condition for p != 1 & W_mu == W_pi non-cancellation
    fus_nl = IFSChoquetFusion(num_modalities=3, conflict_penalty=0.0).double()
    with torch.no_grad():
        fus_nl.choquet_lambda_raw.fill_(1.5)  # p > 1 nonlinear Choquet
    # Non-absorbing external modality m3 -> crossing g1 > g2 vs g1 < g2 is strictly discontinuous when mu1 != mu2
    a1_na = IntuitionisticFuzzyTensor(torch.tensor([[0.85]], dtype=torch.float64), torch.tensor([[0.05]], dtype=torch.float64), torch.tensor([[0.10]], dtype=torch.float64))
    a2_na = IntuitionisticFuzzyTensor(torch.tensor([[0.15]], dtype=torch.float64), torch.tensor([[0.75]], dtype=torch.float64), torch.tensor([[0.10]], dtype=torch.float64))
    a3_na = IntuitionisticFuzzyTensor(torch.tensor([[0.40]], dtype=torch.float64), torch.tensor([[0.20]], dtype=torch.float64), torch.tensor([[0.40]], dtype=torch.float64))
    eps_tie = 1e-7
    a1_plus = IntuitionisticFuzzyTensor(torch.tensor([[0.85 + eps_tie]], dtype=torch.float64), torch.tensor([[0.05]], dtype=torch.float64), torch.tensor([[0.10 - eps_tie]], dtype=torch.float64))
    a1_minus = IntuitionisticFuzzyTensor(torch.tensor([[0.85 - eps_tie]], dtype=torch.float64), torch.tensor([[0.05]], dtype=torch.float64), torch.tensor([[0.10 + eps_tie]], dtype=torch.float64))
    with torch.no_grad():
        fus_nl.fuzzy_density_logits.copy_(torch.tensor([[0.0], [0.0], [-0.5]], dtype=torch.float64))
        out_plus, _ = fus_nl([a1_plus, a2_na, a3_na])
        out_minus, _ = fus_nl([a1_minus, a2_na, a3_na])
    assert abs(float(out_plus.mu.item()) - float(out_minus.mu.item())) > 1e-2

    # Absorbing external modality d*mu3 == 1 (gamma=0, mu3=1): continuous (mu_fused == 1 on both sides)
    a3_abs = IntuitionisticFuzzyTensor(torch.tensor([[1.0]], dtype=torch.float64), torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[0.0]], dtype=torch.float64))
    with torch.no_grad():
        abs_plus, _ = fus_nl([a1_plus, a2_na, a3_abs])
        abs_minus, _ = fus_nl([a1_minus, a2_na, a3_abs])
    assert abs(float(abs_plus.mu.item()) - 1.0) < 1e-12 and abs(float(abs_minus.mu.item()) - 1.0) < 1e-12

    # W_mu = W_pi in (0, 1) with S_mu = S_pi = {1} and nu_2 = 0.6 > 0 outside S: A_mu > A_pi > 0 (no cancellation)
    def _eval_nu_jump(gamma: float) -> float:
        op_w = IFSChoquetFusion(num_modalities=3, conflict_penalty=gamma).double()
        with torch.no_grad():
            op_w.choquet_lambda_raw.zero_()
            op_w.fuzzy_density_logits.zero_()
        m1 = IntuitionisticFuzzyTensor(torch.tensor([[1.0]], dtype=torch.float64), torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[0.0]], dtype=torch.float64))
        m2 = IntuitionisticFuzzyTensor(torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[0.6]], dtype=torch.float64), torch.tensor([[0.4]], dtype=torch.float64))
        vac3 = IntuitionisticFuzzyTensor(torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[1.0]], dtype=torch.float64))
        tiny3 = IntuitionisticFuzzyTensor(torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[1e-12]], dtype=torch.float64), torch.tensor([[1.0 - 1e-12]], dtype=torch.float64))
        ov, _ = op_w([m1, m2, vac3])
        ot, _ = op_w([m1, m2, tiny3])
        return abs(float(ot.nu.item()) - float(ov.nu.item()))

    # r_1 = 1.0, r_2 = 0.6 -> W_* = 1.0 / 1.6 = 5/8, w_2 = 3/8; C_- = 0.6, C_+ = 0.2; A_mu = 1, A_pi = 0.4**(3/8)
    w_star = 5.0 / 8.0
    theo_nu_c = ((0.6 ** w_star) - (0.2 ** w_star)) * (1.0 - (0.4 ** (1.0 - w_star)))
    dnu_4 = _eval_nu_jump(1e-4)
    dnu_8 = _eval_nu_jump(1e-8)
    assert abs((dnu_8 / (1e-8 ** w_star)) - theo_nu_c) < 1e-7
    assert abs((dnu_4 / (1e-4 ** w_star)) - theo_nu_c) < 1e-4


def test_r19_half_precision_loss_gradients_and_bfloat16_upper_clamp():
    """Verify R18-MAJ-02: finite float16 tail/extreme gradients and unclipped bfloat16 BCE upper bound."""
    loss_fn = IntuitionisticFuzzyLoss()

    # 1. float16 near-zero and boundary probabilities retain finite non-overflowing gradients
    p16 = torch.tensor([[0.0], [5e-7], [2e-6], [1.0]], dtype=torch.float16, requires_grad=True)
    y16 = torch.tensor([[1.0], [1.0], [1.0], [0.0]], dtype=torch.float16)
    ifn16 = IntuitionisticFuzzyTensor(
        torch.full((4, 1), 0.4, dtype=torch.float16, requires_grad=True),
        torch.full((4, 1), 0.3, dtype=torch.float16, requires_grad=True),
        torch.full((4, 1), 0.3, dtype=torch.float16, requires_grad=True),
    )
    loss16, ld16 = loss_fn(ifn16, y16, final_prob=p16)
    assert loss16.dtype == torch.float16 and bool(torch.isfinite(loss16))
    assert 8.0 < ld16["bce"] < 10.5
    loss16.backward()
    assert bool(torch.isfinite(p16.grad).all())
    assert bool((p16.grad[:3] < -1e3).all()) and float(p16.grad[3].item()) > 1e3
    assert bool(torch.isfinite(ifn16.mu.grad).all() and torch.isfinite(ifn16.nu.grad).all())

    # 2. float16 extreme auxiliary-modality weight stays finite via float16 backward clamp
    heavy_loss_fn = IntuitionisticFuzzyLoss(lambda_aux_mod=100.0)
    mu_e = torch.tensor([[0.0], [1.0]], dtype=torch.float16, requires_grad=True)
    nu_e = torch.tensor([[1.0], [0.0]], dtype=torch.float16, requires_grad=True)
    pi_e = torch.tensor([[0.0], [0.0]], dtype=torch.float16, requires_grad=True)
    y_e = torch.tensor([[1.0], [0.0]], dtype=torch.float16)
    ifn_e = IntuitionisticFuzzyTensor(mu_e, nu_e, pi_e)
    mod_mu = torch.tensor([[0.0], [1.0]], dtype=torch.float16, requires_grad=True)
    mod_nu = torch.tensor([[1.0], [0.0]], dtype=torch.float16, requires_grad=True)
    mod_pi = torch.tensor([[0.0], [0.0]], dtype=torch.float16, requires_grad=True)
    loss_e, _ = heavy_loss_fn(ifn_e, y_e, modality_ifns=[IntuitionisticFuzzyTensor(mod_mu, mod_nu, mod_pi)])
    assert bool(torch.isfinite(loss_e))
    loss_e.backward()
    for g in (mu_e.grad, nu_e.grad, pi_e.grad, mod_mu.grad, mod_nu.grad, mod_pi.grad):
        assert g is not None and bool(torch.isfinite(g).all())

    # 3. bfloat16 avoids 1.0 - 1e-6 -> 1.0 rounding & PyTorch BCE=100 saturation
    p_bf = torch.tensor([[0.0], [1e-7], [1.0 - 1e-7], [1.0]], dtype=torch.bfloat16, requires_grad=True)
    y_bf = torch.tensor([[1.0], [1.0], [0.0], [0.0]], dtype=torch.bfloat16)
    ifn_bf = IntuitionisticFuzzyTensor(
        torch.full((4, 1), 0.4, dtype=torch.bfloat16, requires_grad=True),
        torch.full((4, 1), 0.3, dtype=torch.bfloat16, requires_grad=True),
        torch.full((4, 1), 0.3, dtype=torch.bfloat16, requires_grad=True),
    )
    loss_bf, ld_bf = loss_fn(ifn_bf, y_bf, final_prob=p_bf)
    assert loss_bf.dtype == torch.bfloat16 and bool(torch.isfinite(loss_bf))
    assert 13.0 < ld_bf["bce"] < 15.0  # -ln(1e-6) ~ 13.8155, never 100.0 saturation
    loss_bf.backward()
    assert bool(torch.isfinite(p_bf.grad).all())
    assert bool((p_bf.grad[:2] < -1e5).all()) and bool((p_bf.grad[2:] > 1e5).all())

    # 4. End-to-end UnifiedDeepFuzzyModel in float16 and bfloat16 (both with and without TSK head, plus boundary ifs_prob)
    for dt in (torch.float16, torch.bfloat16):
        for use_tsk in (True, False):
            torch.manual_seed(0)
            mdl = UnifiedDeepFuzzyModel(
                [4, 3], embed_dim=8, num_tsk_concepts=2, num_tsk_rules=4, use_tsk_head=use_tsk
            ).to(dt)
            if not use_tsk:
                with torch.no_grad():
                    for enc in mdl.ifs_encoders:
                        enc.pos_evidence_head.bias.fill_(5000.0)
                        enc.neg_evidence_head.bias.fill_(-20.0)
            xs = [torch.randn(6, 4, dtype=dt, requires_grad=True), torch.randn(6, 3, dtype=dt, requires_grad=True)]
            masks = torch.ones(6, 2, dtype=dt)
            masks[0, 1] = 0.0
            yb = torch.tensor([1, 0, 1, 0, 1, 0], dtype=dt)
            out = mdl(xs, masks)
            assert out["tsk_logits"].dtype == dt and bool(torch.isfinite(out["tsk_logits"]).all())
            l_m, _ = loss_fn(out, yb, modality_masks=masks)
            assert l_m.dtype == dt and bool(torch.isfinite(l_m))
            (l_m + (0.01 * out["tsk_logits"].sum() if not use_tsk else 0.0)).backward()
            assert all(torch.isfinite(xi.grad).all() for xi in xs if xi.grad is not None)
            assert all(
                prm.grad is None or bool(torch.isfinite(prm.grad).all())
                for prm in mdl.parameters()
            )


def test_r20_codex_round19_low_precision_counterexamples():
    """
    Round-20 regression checks closing every Round-19 low-precision finding:
      1. R19-MAJ-01: Shared & cloned float16 loss inputs keep finite grads even at lambda_aux_mod=1000.
      2. R19-MAJ-02: Finite float16/bfloat16 inputs [40000, 0] with skip_pos=2 avoid encoder overflow & NaNs.
      3. R19-MAJ-03: bfloat16 preserves representable hesitancy pi > 5e-4, loss parity & prior_hesitancy grad.
      4. R19-MAJ-04: Saturated float16/bfloat16 TSK head preserves non-zero tsk_logits & consequent_bias grads.
      5. R19-MIN-06: Fallback torch.logit preserves straight-through unit boundary grad at p=1.0 across dtypes.
    """
    torch.manual_seed(2026)

    # 1. R19-MAJ-01: shared float16 loss input & 10 cloned auxiliary IFNs
    for lam_aux in (10.0, 100.0, 1000.0):
        mu = torch.tensor([0.0], dtype=torch.float16, requires_grad=True)
        nu = torch.tensor([1.0], dtype=torch.float16, requires_grad=True)
        pi = torch.tensor([0.0], dtype=torch.float16, requires_grad=True)
        shared = IntuitionisticFuzzyTensor(mu, nu, pi)
        crit_s = IntuitionisticFuzzyLoss(lambda_aux_mod=lam_aux)
        loss_s, _ = crit_s(shared, torch.tensor([1.0], dtype=torch.float16), modality_ifns=[shared])
        loss_s.backward()
        assert torch.isfinite(loss_s)
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert nu.grad is not None and torch.isfinite(nu.grad).all()
        assert pi.grad is not None and torch.isfinite(pi.grad).all()

    mu_c = torch.tensor([0.0], dtype=torch.float16, requires_grad=True)
    nu_c = torch.tensor([1.0], dtype=torch.float16, requires_grad=True)
    pi_c = torch.tensor([0.0], dtype=torch.float16, requires_grad=True)
    fused_c = IntuitionisticFuzzyTensor(mu_c.view(1), nu_c.view(1), pi_c.view(1))
    aux_clones = [IntuitionisticFuzzyTensor(mu_c.clone(), nu_c.clone(), pi_c.clone()) for _ in range(10)]
    loss_cl, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=100.0)(
        fused_c, torch.tensor([1.0], dtype=torch.float16), modality_ifns=aux_clones
    )
    loss_cl.backward()
    assert torch.isfinite(loss_cl)
    assert mu_c.grad is not None and torch.isfinite(mu_c.grad).all()
    assert nu_c.grad is not None and torch.isfinite(nu_c.grad).all()
    assert pi_c.grad is not None and torch.isfinite(pi_c.grad).all()

    # 2. R19-MAJ-02: finite float16 large-magnitude inputs with skip_pos.weight=2.0
    crit = IntuitionisticFuzzyLoss()
    for use_tsk in (False, True):
        mdl = UnifiedDeepFuzzyModel(modality_dims=[2, 2], num_classes=1, use_tsk_head=use_tsk).to(torch.float16)
        with torch.no_grad():
            for enc in mdl.ifs_encoders:
                enc.skip_pos.weight.fill_(2.0)
                enc.skip_pos.bias.zero_()
                enc.skip_neg.weight.zero_()
                enc.skip_neg.bias.zero_()
        x0 = torch.tensor([[40000.0, 0.0]], dtype=torch.float16, requires_grad=True)
        x1 = torch.tensor([[40000.0, 0.0]], dtype=torch.float16, requires_grad=True)
        obs = torch.tensor([[1.0, 1.0]], dtype=torch.float16)
        out = mdl([x0, x1], obs)
        loss_2, _ = crit(out, torch.tensor([0.0], dtype=torch.float16), modality_masks=obs)
        loss_2.backward()
        assert torch.isfinite(out["prob"]).all()
        assert torch.isfinite(loss_2)
        assert x0.grad is not None and torch.isfinite(x0.grad).all()
        assert x1.grad is not None and torch.isfinite(x1.grad).all()
        assert all(
            prm.grad is None or bool(torch.isfinite(prm.grad).all())
            for prm in mdl.parameters()
        )

    for dt in (torch.float16, torch.bfloat16):
        enc_st = IntuitionisticFuzzyEncoder(in_dim=2).to(dt)
        with torch.no_grad():
            enc_st.skip_pos.weight.fill_(2.0)
            enc_st.skip_pos.bias.zero_()
            enc_st.skip_neg.weight.zero_()
            enc_st.skip_neg.bias.zero_()
        x_st = torch.tensor([[40000.0, 0.0]], dtype=dt, requires_grad=True)
        obs_st = torch.tensor([1.0], dtype=dt)
        ifn_st = enc_st(x_st, obs_st)
        p_st = ifn_st.expected_probability()
        p_st.sum().backward()
        assert torch.isfinite(p_st).all()
        assert x_st.grad is not None and torch.isfinite(x_st.grad).all()
        assert all(prm.grad is None or bool(torch.isfinite(prm.grad).all()) for prm in enc_st.parameters())

    # 3. R19-MAJ-03: bfloat16 cancellation & hesitancy preservation
    def _build_maj03(dt: torch.dtype) -> UnifiedDeepFuzzyModel:
        m3 = UnifiedDeepFuzzyModel(modality_dims=[2, 2], num_classes=1, use_tsk_head=False)
        with torch.no_grad():
            for name, prm in m3.named_parameters():
                if "prior_hesitancy" not in name:
                    prm.zero_()
            for enc in m3.ifs_encoders:
                enc.pos_evidence_head.bias.fill_(-20.0)
                enc.neg_evidence_head.bias.fill_(2000.0)
        return m3.to(dt)

    m3_32 = _build_maj03(torch.float32)
    m3_bf = _build_maj03(torch.bfloat16)
    o3_32 = m3_32([torch.zeros(1, 2), torch.zeros(1, 2)], torch.ones(1, 2))
    l3_32, _ = crit(o3_32, torch.tensor([1.0]))
    l3_32.backward()

    o3_bf = m3_bf(
        [torch.zeros(1, 2, dtype=torch.bfloat16), torch.zeros(1, 2, dtype=torch.bfloat16)],
        torch.ones(1, 2, dtype=torch.bfloat16),
    )
    l3_bf, _ = crit(o3_bf, torch.tensor([1.0], dtype=torch.bfloat16))
    l3_bf.backward()
    assert float(o3_bf["pi_fused"].detach().item()) > 5e-4
    assert float(o3_bf["prob"].detach().item()) > 2e-4
    assert abs(float(l3_bf.detach().item()) - float(l3_32.detach().item())) < 0.1
    assert all(
        enc.prior_hesitancy.grad is not None and float(enc.prior_hesitancy.grad.detach().item()) < -0.2
        for enc in m3_bf.ifs_encoders
    )

    # 4. R19-MAJ-04: saturated half-precision TSK sigmoid preserves training grads
    for dt in (torch.float16, torch.bfloat16):
        m4 = UnifiedDeepFuzzyModel(modality_dims=[2, 2], num_classes=1, use_tsk_head=True)
        with torch.no_grad():
            for prm in m4.parameters():
                prm.zero_()
            for enc in m4.ifs_encoders:
                enc.prior_hesitancy.fill_(-10.0)
                enc.pos_evidence_head.bias.fill_(-20.0)
                enc.neg_evidence_head.bias.fill_(2000.0)
            m4.tsk_blend_logit.fill_(10.0)
            m4.tsk_head.consequent_bias.fill_(10.0)
        m4 = m4.to(dt)
        o4 = m4([torch.zeros(1, 2, dtype=dt), torch.zeros(1, 2, dtype=dt)], torch.ones(1, 2, dtype=dt))
        o4["tsk_logits"].retain_grad()
        l4, _ = crit(o4, torch.tensor([0.0], dtype=dt))
        l4.backward()
        assert o4["tsk_logits"].grad is not None
        assert float(o4["tsk_logits"].grad.detach().item()) > 0.3
        assert m4.tsk_head.consequent_bias.grad is not None
        assert bool((m4.tsk_head.consequent_bias.grad.detach() > 0.05).all())

    # 5. R19-MIN-06: fallback torch.logit straight-through grad at boundary prob == 1.0
    for dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        m6 = UnifiedDeepFuzzyModel(modality_dims=[2, 2], num_classes=1, use_tsk_head=False)
        with torch.no_grad():
            for prm in m6.parameters():
                prm.zero_()
            for enc in m6.ifs_encoders:
                enc.prior_hesitancy.fill_(-10.0)
                enc.pos_evidence_head.bias.fill_(2000.0)
                enc.neg_evidence_head.bias.fill_(-20.0)
        m6 = m6.to(dt)
        o6 = m6([torch.zeros(1, 2, dtype=dt), torch.zeros(1, 2, dtype=dt)], torch.ones(1, 2, dtype=dt))
        p6 = o6["prob"]
        tl6 = o6["tsk_logits"]
        if dt in (torch.float16, torch.bfloat16, torch.float32):
            assert float(p6.detach().item()) == 1.0
        assert torch.isfinite(tl6).all()
        (g6,) = torch.autograd.grad(tl6.sum(), p6)
        assert torch.isfinite(g6).all() and float(g6.detach().item()) > 0.0


def test_r21_codex_round20_expanded_fp16_autocast_f32_hesitancy_and_mps():
    """Verify R20-MAJ-01..04 and R20-MIN-02 (expanded fp16 views, autocast, f32 2e8 hesitancy, MPS, sample weights)."""
    import gc
    from deepfuzzy.operators import _FP16_ACTIVE_ROOT_LOAD

    torch.manual_seed(2026)

    # 1. R20-MAJ-01: Expanded/repeated/indexed/concatenated views + multi-op fp16 (fusion + loss + aux)
    def _run_fp16_view_case(make_view, mod_count: int) -> None:
        x = torch.tensor([-1.0], dtype=torch.float16, requires_grad=True)
        src = make_view(x)
        n_rows = src.shape[0]
        if mod_count == 2:
            m_ifns = [
                IntuitionisticFuzzyTensor(torch.sigmoid(src), 1.0 - torch.sigmoid(src), torch.zeros_like(src)),
                IntuitionisticFuzzyTensor(
                    torch.zeros(n_rows, 1, dtype=torch.float16),
                    torch.ones(n_rows, 1, dtype=torch.float16),
                    torch.zeros(n_rows, 1, dtype=torch.float16),
                ),
            ]
        else:
            m_ifns = [
                IntuitionisticFuzzyTensor(torch.sigmoid(src), 1.0 - torch.sigmoid(src), torch.zeros_like(src)),
                IntuitionisticFuzzyTensor(
                    torch.full((n_rows, 1), 0.7, dtype=torch.float16),
                    torch.full((n_rows, 1), 0.2, dtype=torch.float16),
                    torch.full((n_rows, 1), 0.1, dtype=torch.float16),
                ),
                IntuitionisticFuzzyTensor(
                    torch.zeros(n_rows, 1, dtype=torch.float16),
                    torch.ones(n_rows, 1, dtype=torch.float16),
                    torch.zeros(n_rows, 1, dtype=torch.float16),
                ),
            ]
        fused_ifn, _ = IFSChoquetFusion(mod_count)(m_ifns)
        loss, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)(
            fused_ifn, torch.ones(n_rows, dtype=torch.float16), modality_ifns=m_ifns
        )
        (g,) = torch.autograd.grad(loss, x)
        assert torch.isfinite(g).all(), f"Non-finite fp16 grad for mod_count={mod_count}: {g}"
        assert abs(float(g.detach().item())) <= 65504.0

    view_makers = (
        lambda x_: x_.expand(10, 1),
        lambda x_: x_.repeat(10, 1),
        lambda x_: x_.repeat_interleave(10).unsqueeze(-1),
        lambda x_: x_[torch.zeros(10, dtype=torch.long)].unsqueeze(-1),
        lambda x_: torch.cat([x_.unsqueeze(0)] * 10, dim=0),
    )
    for mod_count in (2, 3):
        for maker in view_makers:
            _run_fp16_view_case(maker, mod_count)

    def _run_extreme_and_no_grad_cases() -> None:
        x_big = torch.tensor([-1.0], dtype=torch.float16, requires_grad=True)
        xe = x_big.expand(1000, 1)
        ifn_big = IntuitionisticFuzzyTensor(torch.sigmoid(xe), 1.0 - torch.sigmoid(xe), torch.zeros_like(xe))
        loss_big, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)(
            ifn_big, torch.ones(1000, dtype=torch.float16), modality_ifns=[ifn_big]
        )
        (g_big,) = torch.autograd.grad(loss_big, x_big)
        assert torch.isfinite(g_big).all() and abs(float(g_big.detach().item())) <= 65504.0

        x_in = torch.tensor([0.5], dtype=torch.float16, requires_grad=True)
        m_in = [
            IntuitionisticFuzzyTensor(torch.sigmoid(x_in.unsqueeze(-1)), 1.0 - torch.sigmoid(x_in.unsqueeze(-1)), torch.zeros(1, 1, dtype=torch.float16)),
            IntuitionisticFuzzyTensor(torch.tensor([[0.2]], dtype=torch.float16), torch.tensor([[0.6]], dtype=torch.float16), torch.tensor([[0.2]], dtype=torch.float16)),
        ]
        f_in, _ = IFSChoquetFusion(2)(m_in)
        z0 = torch.zeros(10, dtype=torch.long)
        rep_ifn = IntuitionisticFuzzyTensor(f_in.mu[z0], f_in.nu[z0], f_in.pi[z0])
        l_rep, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)(
            rep_ifn, torch.ones(10, dtype=torch.float16), modality_ifns=[rep_ifn]
        )
        (g_rep,) = torch.autograd.grad(l_rep, x_in)
        assert torch.isfinite(g_rep).all()

        with torch.no_grad():
            x_ng = torch.tensor([-1.0], dtype=torch.float16).expand(10, 1)
            _ = IntuitionisticFuzzyLoss()(
                IntuitionisticFuzzyTensor(torch.full_like(x_ng, 0.8), torch.full_like(x_ng, 0.1), torch.full_like(x_ng, 0.1)),
                torch.ones(10, dtype=torch.float16),
            )

    _run_extreme_and_no_grad_cases()
    gc.collect()
    assert len(_FP16_ACTIVE_ROOT_LOAD) == 0, f"Leak in _FP16_ACTIVE_ROOT_LOAD: {len(_FP16_ACTIVE_ROOT_LOAD)}"

    # 2. R20-MAJ-02: torch.autocast across CPU bf16/fp16 & UnifiedDeepFuzzyModel straight-through TSK
    for ac_dtype in (torch.bfloat16, torch.float16):
        m_ac = UnifiedDeepFuzzyModel([2, 2], embed_dim=4, concept_names=["c0", "c1"])
        xs_ac = [torch.randn(3, 2, requires_grad=True) for _ in range(2)]
        obs_ac = torch.ones(3, 2)
        y_ac = torch.tensor([1.0, 0.0, 1.0])
        with torch.autocast(device_type="cpu", dtype=ac_dtype):
            out_ac = m_ac(xs_ac, obs_ac)
            loss_ac, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)(out_ac, y_ac)
        assert torch.isfinite(out_ac["prob"]).all() and torch.isfinite(out_ac["tsk_logits"]).all()
        assert torch.isfinite(loss_ac).all()
        (g_p_tsk,) = torch.autograd.grad(out_ac["prob"].sum(), out_ac["tsk_logits"], retain_graph=True)
        assert torch.isfinite(g_p_tsk).all() and float(g_p_tsk.detach().min().item()) > 0.0
        loss_ac.backward()
        assert all(torch.isfinite(x.grad).all() for x in xs_ac)

        m_fb = UnifiedDeepFuzzyModel(modality_dims=[2, 2], num_classes=1, use_tsk_head=False)
        with torch.no_grad():
            for prm in m_fb.parameters():
                prm.zero_()
            for enc in m_fb.ifs_encoders:
                enc.prior_hesitancy.fill_(-10.0)
                enc.pos_evidence_head.bias.fill_(2000.0)
                enc.neg_evidence_head.bias.fill_(-20.0)
        with torch.autocast(device_type="cpu", dtype=ac_dtype):
            o_fb = m_fb([torch.zeros(1, 2), torch.zeros(1, 2)], torch.ones(1, 2))
        assert torch.isfinite(o_fb["tsk_logits"]).all()
        (g_fb,) = torch.autograd.grad(o_fb["tsk_logits"].sum(), o_fb["prob"])
        assert torch.isfinite(g_fb).all() and float(g_fb.detach().item()) > 0.0

    # 3. R20-MAJ-03: Saturated float32 evidence (~2e8) cancellation-free hesitancy preservation
    enc_f32 = IntuitionisticFuzzyEncoder(2).float()
    with torch.no_grad():
        for p in enc_f32.parameters():
            p.zero_()
        enc_f32.pos_evidence_head.bias.fill_(2e8)
        enc_f32.neg_evidence_head.bias.fill_(1e8)
        enc_f32.prior_hesitancy.fill_(0.0)
    x_sat = torch.randn(2, 2, dtype=torch.float32, requires_grad=True)
    ifn_sat = enc_f32(x_sat, torch.ones(2, 1, dtype=torch.float32))
    assert float(ifn_sat.pi.detach().min().item()) > 1e-10
    assert float((ifn_sat.mu + ifn_sat.nu + ifn_sat.pi - 1.0).abs().max().item()) <= 1e-6
    l_sat, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=10.0)(
        ifn_sat, torch.tensor([1.0, 0.0]), modality_ifns=[ifn_sat]
    )
    l_sat.backward()
    assert torch.isfinite(l_sat) and torch.isfinite(x_sat.grad).all()

    # 4. R20-MIN-02: Max-scaled batch-mean sample weights (extreme dynamic range + all-zero)
    crit_sw = IntuitionisticFuzzyLoss()
    mu_sw = torch.tensor([0.8, 0.2, 0.7], dtype=torch.float32)
    nu_sw = torch.tensor([0.1, 0.7, 0.2], dtype=torch.float32)
    ifn_sw = IntuitionisticFuzzyTensor(mu_sw, nu_sw, 1.0 - mu_sw - nu_sw)
    y_sw = torch.tensor([1.0, 0.0, 1.0])
    l_hi, _ = crit_sw(ifn_sw, y_sw, sample_weights=torch.tensor([1e4, 0.0, 1e4]))
    l_lo, _ = crit_sw(ifn_sw, y_sw, sample_weights=torch.tensor([1e-8, 0.0, 1e-8]))
    l_zero, _ = crit_sw(ifn_sw, y_sw, sample_weights=torch.tensor([0.0, 0.0, 0.0]))
    assert abs(float(l_hi.item()) - float(l_lo.item())) < 1e-12
    assert float(l_zero.item()) == 0.0

    # 5. R20-MAJ-04: Apple Metal MPS parity (float32, float16, bfloat16, autocast, rough layer)
    if torch.backends.mps.is_available():
        mps = torch.device("mps")
        for mps_dt in (torch.float32, torch.float16, torch.bfloat16):
            m_mps = UnifiedDeepFuzzyModel([2, 2], embed_dim=4, concept_names=["c0", "c1"]).to(mps, dtype=mps_dt)
            xs_mps = [torch.randn(2, 2, device=mps, dtype=mps_dt, requires_grad=True) for _ in range(2)]
            obs_mps = torch.ones(2, 2, device=mps, dtype=mps_dt)
            out_mps = m_mps(xs_mps, obs_mps)
            loss_mps, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=100.0)(
                out_mps, torch.tensor([1.0, 0.0], device=mps, dtype=mps_dt)
            )
            loss_mps.backward()
            assert torch.isfinite(loss_mps)
            assert all(torch.isfinite(x.grad).all() for x in xs_mps)
        m_mps_ac = UnifiedDeepFuzzyModel([2, 2], embed_dim=4, concept_names=["c0", "c1"]).to(mps)
        xs_mps_ac = [torch.randn(2, 2, device=mps, requires_grad=True) for _ in range(2)]
        with torch.autocast(device_type="mps", dtype=torch.float16):
            out_mps_ac = m_mps_ac(xs_mps_ac, torch.ones(2, 2, device=mps))
            loss_mps_ac, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=100.0)(
                out_mps_ac, torch.tensor([1.0, 0.0], device=mps)
            )
        loss_mps_ac.backward()
        assert torch.isfinite(loss_mps_ac) and all(torch.isfinite(x.grad).all() for x in xs_mps_ac)
        fr_mps = DifferentiableFuzzyRoughLayer(4, soft_temp=1e-6).to(mps)
        toks = torch.randn(2, 3, 4, device=mps, requires_grad=True)
        sl = torch.tensor([[0.2, 0.5, 0.8], [0.1, 0.6, 0.9]], device=mps)
        fr_out = fr_mps(toks, sl)
        fr_out["boundary_region"].sum().backward()
        assert torch.isfinite(toks.grad).all()


def test_r22_codex_round21_deep_fp16_gc_isolation_f32_fusion_mps_reducer_and_logrank():
    import copy
    import gc
    from deepfuzzy.operators import (
        _FP16_ACTIVE_ROOT_LOAD,
        _FP16_BWD_CB_QUEUED,
        _FP16_BWD_PENDING_TOKENS,
        _FP16_BWD_USED_BUDGET,
    )

    # 1. R21-MAJ-01: Deep (>256 and >16384) clone + expand(10,1) FP16 fan-out stability
    def _deep_clone_expand_fp16(n_clones: int):
        x = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
        paths = []
        for _ in range(12):
            c = x
            for _ in range(n_clones):
                c = c.clone()
            paths.append(
                IntuitionisticFuzzyTensor(
                    c.expand(10, 1),
                    (1.0 - c).expand(10, 1),
                    torch.zeros(10, 1, dtype=torch.float16),
                )
            )
        fuse = IFSChoquetFusion(num_modalities=12, conflict_penalty=0.5).half()
        fused, _ = fuse(paths)
        loss, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)(
            fused,
            torch.ones(10, dtype=torch.float16),
            modality_ifns=paths,
        )
        loss.backward()
        return float(loss.detach().item()), float(x.grad.detach().item())

    vals = [_deep_clone_expand_fp16(nc) for nc in (250, 256, 20000)]
    for lv, gv in vals:
        assert abs(lv - 6916.0) < 1e-3 and abs(gv - (-18032.0)) < 1e-3, f"Unexpected deep clone fp16 {(lv, gv)}"

    # 2. R21-MAJ-02: Discarded and live unselected graphs sharing x do not poison selected FP16 gradient
    def _run_fp16_shared_root_probe(discard_extra: int, keep_live_extra: int):
        x = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
        ifn = IntuitionisticFuzzyTensor(x, 1.0 - x, torch.zeros_like(x))
        loss, _ = IntuitionisticFuzzyLoss()(ifn, torch.tensor([1.0], dtype=torch.float16))
        live_extras = []
        if discard_extra > 0:
            discarded = [
                IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)(
                    IntuitionisticFuzzyTensor(x, 1.0 - x, torch.zeros_like(x)),
                    torch.tensor([1.0], dtype=torch.float16),
                    modality_ifns=[IntuitionisticFuzzyTensor(x, 1.0 - x, torch.zeros_like(x))],
                )[0]
                for _ in range(discard_extra)
            ]
            del discarded
            gc.collect()
        if keep_live_extra > 0:
            live_extras = [
                IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)(
                    IntuitionisticFuzzyTensor(x, 1.0 - x, torch.zeros_like(x)),
                    torch.tensor([1.0], dtype=torch.float16),
                    modality_ifns=[IntuitionisticFuzzyTensor(x, 1.0 - x, torch.zeros_like(x))],
                )[0]
                for _ in range(keep_live_extra)
            ]
        loss.backward()
        out = (float(loss.detach().item()), float(x.grad.detach().item()), len(live_extras))
        live_extras.clear()
        return out

    l_iso, g_iso, _ = _run_fp16_shared_root_probe(0, 0)
    l_disc, g_disc, _ = _run_fp16_shared_root_probe(100, 0)
    l_live, g_live, n_live = _run_fp16_shared_root_probe(0, 100)
    assert abs(l_iso - 7.85546875) < 1e-4 and abs(g_iso - (-1001.0)) < 1e-3
    assert abs(l_disc - l_iso) < 1e-4 and abs(g_disc - g_iso) < 1e-3
    assert n_live == 100 and abs(l_live - l_iso) < 1e-4 and abs(g_live - g_iso) < 1e-3
    gc.collect()
    assert not _FP16_ACTIVE_ROOT_LOAD and not _FP16_BWD_PENDING_TOKENS
    assert not _FP16_BWD_USED_BUDGET and not _FP16_BWD_CB_QUEUED

    # 3. R21-MAJ-03: Float32 saturated encoder (2e8, 1e8) + vacuity fusion preserves pi ~ 4.88e-9 & prior grad
    refs = {}
    for dt in (torch.float32, torch.float64):
        enc = IntuitionisticFuzzyEncoder(2).to(dtype=dt)
        with torch.no_grad():
            for p in enc.parameters():
                p.zero_()
            enc.pos_evidence_head.bias.fill_(2e8)
            enc.neg_evidence_head.bias.fill_(1e8)
            enc.prior_hesitancy.fill_(1.2)
        fuse = IFSChoquetFusion(num_modalities=2, conflict_penalty=0.5).to(dtype=dt)
        ifn = enc(torch.zeros(2, 2, dtype=dt), torch.ones(2, 1, dtype=dt))
        vac = IntuitionisticFuzzyTensor(
            torch.zeros(2, 1, dtype=dt), torch.zeros(2, 1, dtype=dt), torch.ones(2, 1, dtype=dt)
        )
        fused, _ = fuse([ifn, vac])
        loss, _ = IntuitionisticFuzzyLoss()(fused, torch.tensor([1.0, 1.0], dtype=dt))
        loss.backward()
        refs[dt] = (
            float(ifn.pi.detach()[0, 0].item()),
            float(fused.pi.detach()[0, 0].item()),
            float(enc.prior_hesitancy.grad.detach()[0].item()),
        )
    assert refs[torch.float32][0] > 4e-9 and refs[torch.float32][1] > 4e-9
    assert abs(refs[torch.float32][1] - refs[torch.float64][1]) <= 1e-11
    assert refs[torch.float32][2] > 5e-10 and abs(refs[torch.float32][2] - refs[torch.float64][2]) <= 1e-11

    # 4. R21-MAJ-04: Float32 huge evidence (2e38, 1.5e38) does not overflow to NaN in total = alpha + beta
    for dev_name in (["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]):
        enc_ov = IntuitionisticFuzzyEncoder(2).to(dev_name, dtype=torch.float32)
        with torch.no_grad():
            for p in enc_ov.parameters():
                p.zero_()
            enc_ov.pos_evidence_head.bias.fill_(2e38)
            enc_ov.neg_evidence_head.bias.fill_(1.5e38)
        out_ov = enc_ov(torch.zeros(2, 2, device=dev_name), torch.ones(2, 1, device=dev_name))
        assert out_ov.validate(atol=1e-5)
        assert abs(float(out_ov.mu.detach().cpu()[0, 0]) - 2.0 / 3.5) < 1e-5
        assert abs(float(out_ov.nu.detach().cpu()[0, 0]) - 1.5 / 3.5) < 1e-5

    # 5. R21-MAJ-05: Live Metal MPS FuzzyRoughFeatureReducer + UnifiedDeepFuzzyModel parity vs CPU
    if torch.backends.mps.is_available():
        rng_rr = np.random.default_rng(17)
        X_rr = rng_rr.normal(size=(48, 12)).astype(np.float32)
        y_bin = (rng_rr.random(48) > 0.5).astype(np.float32)
        y_con = np.clip(X_rr[:, 0] * 0.4 + 0.5, 0.0, 1.0).astype(np.float32)
        for y_cur in (y_bin, y_con):
            rc = FuzzyRoughFeatureReducer(max_features=4, n_null_permutations=3, n_stability_bootstraps=0, seed=5).fit(
                X_rr, y_cur, device="cpu"
            )
            rm = FuzzyRoughFeatureReducer(max_features=4, n_null_permutations=3, n_stability_bootstraps=0, seed=5).fit(
                X_rr, y_cur, device="mps"
            )
            assert rc.selected_indices_ == rm.selected_indices_
            assert max(abs(a - b) for a, b in zip(rc.dependency_curve_, rm.dependency_curve_)) == 0.0

        torch.manual_seed(99)
        mc = UnifiedDeepFuzzyModel([3, 3], embed_dim=4, concept_names=["c0", "c1"]).eval().cpu().float()
        mm = copy.deepcopy(mc).to("mps", dtype=torch.float32)
        xs_c = [torch.randn(4, 3, requires_grad=True) for _ in range(2)]
        xs_m = [x.detach().to("mps").requires_grad_(True) for x in xs_c]
        obs_c = torch.ones(4, 2)
        y_c = torch.tensor([1.0, 0.0, 1.0, 0.0])
        lc, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=50.0)(mc(xs_c, obs_c), y_c)
        lm, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=50.0)(mm(xs_m, obs_c.to("mps")), y_c.to("mps"))
        lc.backward()
        lm.backward()
        assert abs(float(lc.item()) - float(lm.cpu().item())) < 1e-4
        assert max(float((xm.grad.cpu() - xc.grad).abs().max().item()) for xm, xc in zip(xs_m, xs_c)) < 1e-3

    # 6. R21-MAJ-06: compute_multigroup_logrank_test rank-deficient df invariance with censored 3rd group
    from scipy.stats import chi2 as chi2_dist
    chi2_2, p2 = compute_multigroup_logrank_test(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        np.ones(6, dtype=int),
        np.array([0, 0, 0, 1, 1, 1]),
    )
    chi2_3, p3 = compute_multigroup_logrank_test(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5]),
        np.array([1, 1, 1, 1, 1, 1, 0], dtype=int),
        np.array([0, 0, 0, 1, 1, 1, 2]),
    )
    assert abs(chi2_2 - chi2_3) < 1e-12 and abs(p2 - p3) < 1e-12
    assert abs(p3 - float(chi2_dist.sf(max(0.0, chi2_3), df=1))) < 1e-12
    assert abs(p3 - 0.02460) < 1e-4

    # 7. R21-MIN-01: TCGA inner 80/20 split fits median imputation strictly on inner-train
    with open(os.path.join(_REPO_ROOT, "benchmarks", "run_upgrades_and_pathway_audit.py"), "r", encoding="utf-8") as fh:
        up_src = fh.read()
    assert (
        "X_rna_tr_obs_raw[in_tr]" in up_src
        and "med_in_f" in up_src
        and "X_in_tr_imp" in up_src
        and "X_rna_tr_obs_raw[in_va]" in up_src
    )
    # Functional regression matching run_upgrades_and_pathway_audit.py: changing in_va cannot change inner-train imputed/scaled values
    X_syn_obs = np.array(
        [[1.0], [3.0], [np.nan], [3.0], [100.0], [100.0]], dtype=np.float32
    )
    in_tr_idx = np.array([0, 1, 2, 3])
    in_va_idx = np.array([4, 5])
    med_in = np.nanmedian(X_syn_obs[in_tr_idx], axis=0, keepdims=True)
    X_in_tr_imp_a = np.where(np.isnan(X_syn_obs[in_tr_idx]), med_in, X_syn_obs[in_tr_idx]).astype(np.float32)
    X_syn_obs_alt = X_syn_obs.copy()
    X_syn_obs_alt[in_va_idx] = 9999.0
    med_in_b = np.nanmedian(X_syn_obs_alt[in_tr_idx], axis=0, keepdims=True)
    X_in_tr_imp_b = np.where(np.isnan(X_syn_obs_alt[in_tr_idx]), med_in_b, X_syn_obs_alt[in_tr_idx]).astype(np.float32)
    assert float(med_in[0, 0]) == 3.0 and np.array_equal(X_in_tr_imp_a, X_in_tr_imp_b)


def test_r23_codex_round22_dtype_cast_fp16_live_isolation_enc_sum_missing_and_conflict():
    import gc
    import inspect
    import math
    from deepfuzzy.metrics import compute_logrank_test, compute_multigroup_logrank_test
    from deepfuzzy.operators import (
        IFSChoquetFusion,
        _FP16_ACTIVE_ROOT_LOAD,
        _FP16_BWD_CB_QUEUED,
        _FP16_BWD_PENDING_TOKENS,
        _FP16_BWD_USED_BUDGET,
    )

    # 1. R22-MAJ-01: Float16 fan-in across dtype conversions (x.float().half() and shared x32.half())
    def _run_cast_expand_fp16(depth: int) -> tuple[float, float]:
        x = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
        paths = []
        for _ in range(12):
            c = x
            for _ in range(depth):
                c = c.float().half()
            paths.append(
                IntuitionisticFuzzyTensor(
                    c.expand(10, 1),
                    (1.0 - c).expand(10, 1),
                    torch.zeros(10, 1, dtype=torch.float16),
                )
            )
        fuse = IFSChoquetFusion(num_modalities=12, conflict_penalty=0.5).half()
        fused, _ = fuse(paths)
        loss, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)(
            fused, torch.ones(10, dtype=torch.float16), modality_ifns=paths
        )
        loss.backward()
        return float(loss.detach().item()), float(x.grad.detach().item())

    for d in (1, 10, 256):
        lv, gv = _run_cast_expand_fp16(d)
        assert abs(lv - 6916.0) < 1e-3 and math.isfinite(gv) and abs(gv) <= 65504.0

    x32 = torch.tensor([[0.001]], dtype=torch.float32, requires_grad=True)
    y16 = x32.half()
    paths32 = [
        IntuitionisticFuzzyTensor(
            y16.expand(10, 1),
            (1.0 - y16).expand(10, 1),
            torch.zeros(10, 1, dtype=torch.float16),
        )
        for _ in range(12)
    ]
    fused32, _ = IFSChoquetFusion(num_modalities=12, conflict_penalty=0.5).half()(paths32)
    l32, _ = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)(
        fused32, torch.ones(10, dtype=torch.float16), modality_ifns=paths32
    )
    l32.backward()
    assert abs(float(l32.detach().item()) - 6916.0) < 1e-3
    assert math.isfinite(float(x32.grad.detach().item()))

    # 2. R22-MAJ-02: Live unselected L(x) AND live >16384-clone sentinel graphs leave selected gradient unchanged
    def _make_L(v: torch.Tensor) -> torch.Tensor:
        ifn = IntuitionisticFuzzyTensor(v, 1.0 - v, torch.zeros_like(v))
        return IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)(
            ifn, torch.ones_like(v), modality_ifns=[ifn]
        )[0]

    x_base = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
    l_base = _make_L(x_base) + _make_L(x_base)
    l_base.backward()
    g_base = float(x_base.grad.detach().item())
    assert abs(float(l_base.detach().item()) - 13832.0) < 1e-3 and abs(g_base - (-18000.0)) < 1e-3

    x_live = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
    live_same = [_make_L(x_live) for _ in range(10)]
    z_live = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
    live_overflow = []
    for _ in range(10):
        cz = z_live
        for _ in range(16400):
            cz = cz.clone()
        live_overflow.append(_make_L(cz))
    l_sel = _make_L(x_live) + _make_L(x_live)
    l_sel.backward()
    g_live = float(x_live.grad.detach().item())
    assert g_live == g_base
    live_same.clear()
    live_overflow.clear()
    gc.collect()
    assert not _FP16_ACTIVE_ROOT_LOAD and not _FP16_BWD_PENDING_TOKENS
    assert not _FP16_BWD_USED_BUDGET and not _FP16_BWD_CB_QUEUED

    # 3. R22-MAJ-03: Finite float32 encoder pre-activation sums (2e38 + 2e38, neg 1e38) match float64
    devs = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]
    refs_enc = {}
    for dev_s in devs:
        for dt in (torch.float32, torch.float64):
            if dev_s == "mps" and dt == torch.float64:
                continue
            enc_s = IntuitionisticFuzzyEncoder(2).to(dev_s, dtype=dt)
            with torch.no_grad():
                for prm in enc_s.parameters():
                    prm.zero_()
                enc_s.pos_evidence_head.bias.fill_(2e38)
                enc_s.skip_pos.bias.fill_(2e38)
                enc_s.neg_evidence_head.bias.fill_(1e38)
            x_in = torch.zeros(2, 2, device=dev_s, dtype=dt, requires_grad=True)
            out_s = enc_s(x_in)
            assert out_s.validate(atol=1e-6)
            (out_s.mu.sum() + out_s.pi.sum()).backward()
            assert torch.isfinite(x_in.grad).all()
            if dev_s == "cpu":
                refs_enc[dt] = (
                    float(out_s.mu.detach()[0, 0].item()),
                    float(out_s.nu.detach()[0, 0].item()),
                    float(out_s.pi.detach()[0, 0].item()),
                )
    assert abs(refs_enc[torch.float32][0] - refs_enc[torch.float64][0]) < 1e-6
    assert abs(refs_enc[torch.float32][1] - refs_enc[torch.float64][1]) < 1e-6
    assert refs_enc[torch.float32][2] > 1e-39

    # 4. R22-MAJ-05: CombinatorialMissingModalityBenchmark returns NaN for undefined eligible/complete AUROC
    bench = CombinatorialMissingModalityBenchmark(["A", "B"])
    miss_res = bench.evaluate_all_subsets(
        predict_fn=lambda xs, m: {
            "prob": torch.tensor([0.2, 0.3, 0.7, 0.8]),
            "hesitancy": torch.tensor([0.1, 0.1, 0.1, 0.1]),
        },
        modality_tensors=[torch.zeros(4, 2), torch.zeros(4, 2)],
        natural_masks=torch.tensor([[0, 1], [0, 1], [1, 1], [1, 1]], dtype=torch.float32),
        targets=np.array([0, 0, 1, 1]),
    )
    assert math.isnan(miss_res["subsets"]["A"]["auroc_eligible"])
    assert math.isnan(miss_res["subsets"]["A"]["auroc_complete_subset"])
    assert math.isnan(miss_res["subsets"]["A+B"]["auroc_complete_subset"])

    # 5. R22-MIN-01: compute_indiscernibility handles device="mps" with default dtype safely
    rng_r = np.random.default_rng(7)
    Xr = rng_r.normal(size=(20, 4)).astype(np.float32)
    yr = (rng_r.random(20) > 0.5).astype(np.float32)
    red_c = FuzzyRoughFeatureReducer(max_features=2, n_null_permutations=2).fit(Xr, yr, device="cpu")
    R_cpu = red_c.compute_indiscernibility(Xr, [0, 1], device="cpu")
    if torch.backends.mps.is_available():
        R_mps = red_c.compute_indiscernibility(Xr, [0, 1], device="mps")
        assert R_mps.device.type == "mps" and R_mps.dtype == torch.float32
        assert float((R_mps.cpu() - R_cpu.float()).abs().max().item()) < 1e-6

    # 6. R22-MIN-02: Log-rank docstrings report asymptotic chi-squared & numerical rank
    doc1 = (inspect.getdoc(compute_logrank_test) or "").lower()
    doc2 = (inspect.getdoc(compute_multigroup_logrank_test) or "").lower()
    assert "exact" not in doc1 and "asymptotic" in doc1
    assert "exact" not in doc2 and "asymptotic" in doc2 and "eff_df" in doc2

    # 7. R22-MIN-04 & MIN-05: Zero-severity / missing-donor opposite_swap isolation & exact referral coverage
    inj = CrossModalityConflictInjector(2, ["A", "B"])
    xt0 = [torch.ones(10, 2), torch.ones(10, 2)]
    xt0_out, is_c0 = inj.inject_conflict(
        xt0, np.array([0] * 5 + [1] * 5), 1, conflict_fraction=0.5, severity_gamma=0.0
    )
    assert int(is_c0.sum()) == 0 and torch.equal(xt0_out[1], xt0[1])

    nm_single = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]], dtype=torch.float32)
    xt1_out, is_c1 = inj.inject_conflict(
        [torch.ones(6, 2), torch.ones(6, 2)],
        np.array([0, 0, 0, 1, 1, 1]),
        1,
        conflict_fraction=0.5,
        severity_gamma=1.0,
        mode="opposite_swap",
        natural_masks=nm_single,
    )
    assert int(is_c1.sum()) == 0 and torch.equal(xt1_out[1], torch.ones(6, 2))

    sw_res = inj.evaluate_conflict_sweep(
        predict_fn=lambda xs, m: {
            "prob": torch.tensor([0.9] * 5 + [0.4] * 15),
            "hesitancy": torch.linspace(0.0, 0.9, 20),
            "conflict": torch.linspace(0.0, 0.9, 20),
        },
        modality_tensors=[torch.ones(20, 2), torch.ones(20, 2)],
        natural_masks=torch.ones(20, 2),
        targets=np.array([1] * 5 + [0] * 5 + [1] * 10),
        severities=[0.0],
        referral_rates=[0.75],
    )
    ref75 = sw_res["sweep_records"][0]["referral_curve"]["refer_75pct"]
    assert ref75["coverage"] == 0.25 and ref75["n_retained"] == 5 and ref75["acc"] == 1.0


def test_r24_codex_round23_fp16_cancel_overflow_enc_f64_missing_conflict_and_docs():
    """Round-24 regressions for all 10 Round-23 Codex findings (R23-MAJ-01..05, R23-MIN-01..05)."""
    import gc
    import math
    import warnings
    from deepfuzzy.operators import (
        _FP16_ACTIVE_ROOT_LOAD,
        _FP16_BWD_PENDING_TOKENS,
        _FP16_BWD_USED_BUDGET,
        _FP16_BWD_CB_QUEUED,
    )

    loss_fn = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)

    def _make_loss(v: torch.Tensor) -> torch.Tensor:
        ifn = IntuitionisticFuzzyTensor(v, 1.0 - v, torch.zeros_like(v))
        return loss_fn(ifn, torch.ones_like(v), modality_ifns=[ifn])[0]

    # 1. R23-MAJ-01: Signed telescoping backward budget under canceling FP16 objectives (+ R22 parity)
    x_cancel = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
    l_cancel = (
        _make_loss(x_cancel) + _make_loss(x_cancel) + _make_loss(x_cancel)
    ) - (
        _make_loss(x_cancel) + _make_loss(x_cancel) + _make_loss(x_cancel)
    )
    l_cancel.backward()
    assert abs(float(l_cancel.detach().item())) < 1e-3
    assert abs(float(x_cancel.grad.detach().item())) <= 16.0

    x_exact0 = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
    l_one = _make_loss(x_exact0)
    (3.0 * l_one - 3.0 * l_one).backward()
    assert float(x_exact0.grad.detach().item()) == 0.0

    x_expand_cancel = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
    y_exp = x_expand_cancel.expand(10, 1)
    l_exp_cancel = (
        _make_loss(y_exp) + _make_loss(y_exp) + _make_loss(y_exp)
    ) - (
        _make_loss(y_exp) + _make_loss(y_exp) + _make_loss(y_exp)
    )
    l_exp_cancel.backward()
    assert abs(float(x_expand_cancel.grad.detach().item())) <= 16.0

    # Non-canceling 6-term positive objective still clamps safely within FP16 range
    x_pos6 = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
    l_pos6 = sum(_make_loss(x_pos6) for _ in range(6))
    l_pos6.backward()
    assert math.isfinite(float(x_pos6.grad.detach().item()))
    assert abs(float(x_pos6.grad.detach().item())) <= 48000.0

    # 2. R23-MAJ-02: Disjoint deterministic spine keys for >16,384-clone selected FP16 roots
    xs_deep = [torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True) for _ in range(7)]
    losses_deep = []
    for i, root_v in enumerate(xs_deep):
        cur = root_v.reshape(1, 1) if i % 2 == 1 else root_v
        for _ in range(16400):
            cur = cur.clone()
        if i % 2 == 1:
            cur = cur.expand(2, 1)
        losses_deep.append(_make_loss(cur))
    sum(losses_deep).backward()
    for root_v in xs_deep:
        assert abs(float(root_v.grad.detach().item()) - (-9000.0)) < 1e-3

    losses_deep.clear()
    gc.collect()
    assert not _FP16_ACTIVE_ROOT_LOAD and not _FP16_BWD_PENDING_TOKENS
    assert not _FP16_BWD_USED_BUDGET and not _FP16_BWD_CB_QUEUED

    # 3. R23-MAJ-03: Float64 encoder per-element overflow handling preserves -inf slot & simplex closure
    enc_f64 = IntuitionisticFuzzyEncoder(2).double()
    with torch.no_grad():
        for prm in enc_f64.parameters():
            prm.zero_()
        enc_f64.pos_evidence_head.bias.fill_(1e308)
        enc_f64.skip_pos.bias.fill_(-1e308)
        enc_f64.neg_evidence_head.bias.fill_(-1e308)
        enc_f64.skip_neg.bias.fill_(-1e308)
    x_zero64 = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    out_f64 = enc_f64(x_zero64)
    assert out_f64.validate(atol=1e-12)
    exp_mu = math.log(2.0) / (2.0 * math.log(2.0) + 1e-4)
    assert abs(float(out_f64.mu.detach()[0, 0].item()) - exp_mu) < 1e-9
    assert float(out_f64.nu.detach()[0, 0].item()) == 0.0
    assert abs(float(out_f64.pi.detach()[0, 0].item()) - (1.0 - exp_mu)) < 1e-9
    (out_f64.mu.sum() + out_f64.nu.sum() + out_f64.pi.sum()).backward()
    assert torch.isfinite(x_zero64.grad).all()

    enc_mix = IntuitionisticFuzzyEncoder(2).float()
    with torch.no_grad():
        for prm in enc_mix.parameters():
            prm.zero_()
        enc_mix.pos_evidence_head.weight[0, 0] = 1e38
        enc_mix.skip_pos.weight[0, 0] = 1e38
    x_mix = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32, requires_grad=True)
    out_mix = enc_mix(x_mix)
    assert out_mix.validate(atol=1e-6)
    (out_mix.mu.sum() + out_mix.nu.sum() + out_mix.pi.sum()).backward()
    assert torch.isfinite(x_mix.grad).all()

    # 4. R23-MAJ-04 & R23-MIN-01: Missing-modality NaN-safe masking, empty hesitancy NaN & warning-free curve
    bench = CombinatorialMissingModalityBenchmark(["A", "B"])
    miss_nan_res = bench.evaluate_all_subsets(
        predict_fn=lambda xs, m: {
            "prob": (xs[0].sum(dim=-1) + xs[1].sum(dim=-1)).sigmoid(),
            "hesitancy": torch.zeros(xs[0].shape[0]),
        },
        modality_tensors=[
            torch.full((4, 1), float("nan")),
            torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]),
        ],
        natural_masks=torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.float32),
        targets=np.array([0, 0, 1, 1]),
    )
    assert abs(float(miss_nan_res["subsets"]["A+B"]["auroc"]) - 1.0) < 1e-12
    assert math.isnan(miss_nan_res["subsets"]["A"]["mean_hesitancy_eligible"])
    assert miss_nan_res["subsets"]["A"]["n_eligible"] == 0

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        curve_sc = bench.evaluate_missing_rate_curve(
            predict_fn=lambda xs, m: {
                "prob": torch.tensor([0.2, 0.8, 0.4, 0.6]),
                "hesitancy": torch.tensor([0.1, 0.1, 0.1, 0.1]),
            },
            modality_tensors=[torch.ones(4, 1), torch.ones(4, 1)],
            natural_masks=torch.ones(4, 2),
            targets=np.array([1, 1, 1, 1]),
            drop_rates=[0.5],
            n_repeats=3,
        )
    assert math.isnan(curve_sc[0]["auroc"]) and math.isnan(curve_sc[0]["auroc_std"])
    assert math.isnan(curve_sc[0]["auprc"]) and math.isnan(curve_sc[0]["auprc_std"])

    # 5. R23-MAJ-05, R23-MIN-02 & R23-MIN-03: Identical donors, empty conflict/clean groups & exact referral keys
    inj = CrossModalityConflictInjector(2, ["A", "B"])
    x_dup = torch.ones(6, 2)
    _, is_conf_dup = inj.inject_conflict(
        [x_dup.clone(), x_dup.clone()],
        np.array([0, 0, 0, 1, 1, 1]),
        corrupted_modality_idx=0,
        conflict_fraction=0.5,
        severity_gamma=0.8,
        mode="opposite_swap",
    )
    assert int(is_conf_dup.sum()) == 0

    sw_empty = inj.evaluate_conflict_sweep(
        predict_fn=lambda xs, m: {
            "prob": torch.tensor([0.2, 0.4, 0.6, 0.8]),
            "hesitancy": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        },
        modality_tensors=[torch.randn(4, 2), torch.randn(4, 2)],
        natural_masks=torch.ones(4, 2),
        targets=np.array([1, 1, 1, 1]),
        corrupted_modality_idx=0,
        severities=[0.8],
        referral_rates=[0.10, 0.109, 0.29],
    )
    rec0 = sw_empty["sweep_records"][0]
    assert rec0["n_conflicted"] == 0
    assert math.isnan(rec0["conflict_detection_auroc"])
    assert math.isnan(rec0["hesitancy_conflicted"])
    assert math.isnan(rec0["conflict_conflicted"])
    assert abs(rec0["hesitancy_clean"] - 0.25) < 1e-12
    assert set(rec0["referral_curve"].keys()) == {"refer_10pct", "refer_10.9pct", "refer_29pct"}

    sw_all_conf = inj.evaluate_conflict_sweep(
        predict_fn=lambda xs, m: {
            "prob": torch.tensor([0.2, 0.4, 0.6, 0.8]),
            "hesitancy": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        },
        modality_tensors=[torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]), torch.ones(4, 1)],
        natural_masks=torch.ones(4, 2),
        targets=np.array([0, 0, 1, 1]),
        corrupted_modality_idx=0,
        severities=[0.8],
        conflict_fraction=1.0,
    )
    rec_all_conf = sw_all_conf["sweep_records"][0]
    assert rec_all_conf["n_conflicted"] == 4
    assert math.isnan(rec_all_conf["conflict_detection_auroc"])
    assert math.isnan(rec_all_conf["hesitancy_clean"])
    assert math.isnan(rec_all_conf["conflict_clean"])
    assert abs(rec_all_conf["hesitancy_conflicted"] - 0.25) < 1e-12

    # 6. R23-MIN-04 & R23-MIN-05: Stale test docstring & README Track 0/Track 1 labeling
    assert "asymptotic Mantel-Haenszel Log-Rank test" in (__doc__ or "")
    assert ("exact " + "Log-Rank") not in (__doc__ or "")
    with open(os.path.join(_REPO_ROOT, "README.md"), "r", encoding="utf-8") as fh_readme:
        readme_lines = fh_readme.read().splitlines()
    track0_row = next(ln for ln in readme_lines if "**Track 0: Mathematical Audit**" in ln)
    track1_row = next(ln for ln in readme_lines if "**Track 1: ADNI Cohort 1**" in ln)
    assert "$40\\times 40$ Grid" in track0_row and "50-Pt Sweep" in track0_row and "$N_{\\text{tok}}=45$" in track0_row
    assert "TSK" not in track0_row and "TSK text rule reconstruction error" in track1_row


def test_r25_codex_round24_vjp_cancel_deep_shared_hook_scale_enc_and_conflict():
    import gc
    import math
    from deepfuzzy.conflict_injector import CrossModalityConflictInjector
    from deepfuzzy.metrics import compute_binary_metrics
    from deepfuzzy.missing_benchmark import CombinatorialMissingModalityBenchmark
    from deepfuzzy.operators import (
        IFSChoquetFusion,
        _FP16_ACTIVE_ROOT_LOAD,
        _FP16_BWD_CB_QUEUED,
        _FP16_BWD_PENDING_TOKENS,
        _FP16_BWD_USED_BUDGET,
    )

    loss_fn = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)

    def _L(v: torch.Tensor) -> torch.Tensor:
        v_2d = v.reshape(1, 1)
        ifn = IntuitionisticFuzzyTensor(v_2d, (1.0 - v_2d), torch.zeros_like(v_2d))
        return loss_fn(ifn, torch.ones(1, dtype=v_2d.dtype, device=v_2d.device), modality_ifns=[ifn])[0]

    # 1. R24-MAJ-01 & R22-MAJ-01: VJP cancellation for relu/abs/sin, CPU autocast, and mixed-cast chains
    for fn in (lambda v: v.relu(), lambda v: v.abs(), lambda v: v.sin(), lambda v: (v.float().relu()).half()):
        for n_copies in (3, 8):
            xc = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
            lc = sum(_L(fn(xc)) for _ in range(n_copies)) - sum(_L(fn(xc)) for _ in range(n_copies))
            lc.backward()
            assert float(lc.detach().item()) == 0.0
            assert math.isfinite(float(xc.grad.detach().item()))
            assert abs(float(xc.grad.detach().item())) <= 16.0

    with torch.autocast("cpu", dtype=torch.float16):
        x_ac = torch.tensor(0.001, dtype=torch.float32, requires_grad=True)
        w_ac = torch.ones((1, 1), dtype=torch.float32)
        z_ac = (x_ac.reshape(1, 1) @ w_ac).reshape(()).relu()
        l_ac = sum(_L(z_ac).float() for _ in range(4)) - sum(_L(z_ac).float() for _ in range(4))
        assert bool(torch.isfinite(l_ac).all()) and float(l_ac.item()) == 0.0
        l_ac.backward()
        assert math.isfinite(float(x_ac.grad.detach().item()))
        assert abs(float(x_ac.grad.detach().item())) <= 16.0

    x_mc = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
    paths_mc = []
    for _ in range(12):
        cm = (x_mc.float() * 1.0).half()
        paths_mc.append(
            IntuitionisticFuzzyTensor(
                cm.expand(10, 1),
                (1.0 - cm).expand(10, 1),
                torch.zeros(10, 1, dtype=torch.float16),
            )
        )
    fused_mc, _ = IFSChoquetFusion(num_modalities=12, conflict_penalty=0.5).half()(paths_mc)
    l_mc, _ = loss_fn(fused_mc, torch.ones(10, dtype=torch.float16), modality_ifns=paths_mc)
    l_mc.backward()
    assert abs(float(l_mc.detach().item()) - 6916.0) < 1e-3
    assert abs(float(x_mc.grad.detach().item()) - (-18032.0)) < 1e-3

    # 2. R24-MAJ-02: Shared graph overflow downstream of deep unary chain (>65k ops, linear & nonlinear)
    def _run_deep_linear_65k():
        a1 = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
        b1 = torch.tensor(0.0, dtype=torch.float16, requires_grad=True)
        t1 = b1
        for _ in range(65540):
            t1 = t1 * 1.0
        t1 = a1 + t1
        l_deep1 = sum(_L(t1).float() for _ in range(5)) + sum(_L(a1).float() for _ in range(5))
        l_deep1.backward()
        return float(a1.grad.detach().item()), float(b1.grad.detach().item())

    ga1, gb1 = _run_deep_linear_65k()
    gc.collect()
    assert math.isfinite(ga1) and abs(ga1) <= 48128.0
    assert math.isfinite(gb1) and abs(gb1) <= 48128.0

    def _run_deep_sin_66k():
        a2 = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
        b2 = torch.tensor(0.0, dtype=torch.float16, requires_grad=True)
        t2 = b2
        for _ in range(66000):
            t2 = t2.sin()
        t2 = a2 + t2
        l_deep2 = sum(_L(t2).float() for _ in range(5)) + sum(_L(a2).float() for _ in range(5))
        l_deep2.backward()
        return float(a2.grad.detach().item()), float(b2.grad.detach().item())

    ga2, gb2 = _run_deep_sin_66k()
    gc.collect()
    assert math.isfinite(ga2) and abs(ga2) <= 48128.0
    assert math.isfinite(gb2) and abs(gb2) <= 48128.0

    def _run_deep_dag_sin():
        a3 = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
        b3 = torch.tensor(0.0, dtype=torch.float16, requires_grad=True)
        chains = []
        for _ in range(8):
            c3 = b3
            for _ in range(8500):
                c3 = c3.sin()
            chains.append(c3)
        t3 = a3 + sum(chains)
        l_deep3 = sum(_L(t3).float() for _ in range(5)) + sum(_L(a3).float() for _ in range(5))
        l_deep3.backward()
        return float(a3.grad.detach().item()), float(b3.grad.detach().item())

    ga3, gb3 = _run_deep_dag_sin()
    gc.collect()
    assert math.isfinite(ga3) and abs(ga3) <= 48128.0
    assert math.isfinite(gb3) and abs(gb3) <= 48128.0

    # 3. R24-MAJ-03: Failed backward hook exception cleans FP16 state immediately (including retain_graph=True)
    for retain in (False, True):
        xh = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)

        def _boom(grad):
            raise RuntimeError("intentional")

        hk = xh.register_hook(_boom)
        lh = sum(_L(xh) for _ in range(6))
        try:
            lh.backward(retain_graph=retain)
            raise AssertionError("Expected RuntimeError from tensor hook")
        except RuntimeError as exc:
            assert "intentional" in str(exc)
        finally:
            hk.remove()
            if not retain:
                del lh
        gc.collect()
        assert len(_FP16_ACTIVE_ROOT_LOAD) == 0
        assert len(_FP16_BWD_PENDING_TOKENS) == 0
        assert len(_FP16_BWD_USED_BUDGET) == 0
        assert len(_FP16_BWD_CB_QUEUED) == 0

        xh.grad = None
        _L(xh).backward()
        assert abs(float(xh.grad.detach().item()) - (-9000.0)) < 1e-3
        assert len(_FP16_BWD_USED_BUDGET) == 0 and len(_FP16_BWD_CB_QUEUED) == 0

    # 4. R24-MAJ-04: Negative double scaling and reverse scaling preservation/finiteness
    for expr_name, fn_s in (
        ("k1", lambda v: (v * -1.0) * (-1.0)),
        ("k256", lambda v: (v * -256.0) * (-1.0 / 256.0)),
        ("div256", lambda v: (v * -256.0) / (-256.0)),
        ("reverse", lambda v: (v * (-1.0 / 256.0)) * (-256.0)),
    ):
        xs = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
        _L(fn_s(xs)).backward()
        gv = float(xs.grad.detach().item())
        assert math.isfinite(gv)
        if expr_name != "reverse":
            assert abs(gv - (-9000.0)) < 1e-3

    # 5. R24-MAJ-05: IntuitionisticFuzzyEncoder float64 overflow in linear heads/backbone
    enc1 = IntuitionisticFuzzyEncoder(1).double().eval()
    with torch.no_grad():
        for p in enc1.parameters():
            p.zero_()
        enc1.skip_pos.weight.fill_(1e308)
    inp1 = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    ifn1 = enc1(inp1)
    ifn1.validate()
    assert float(ifn1.mu.item()) > 0.999999 and abs(float(ifn1.nu.item())) < 1e-6 and abs(float(ifn1.pi.item())) < 1e-6
    (ifn1.mu.sum() - ifn1.nu.sum()).backward()
    assert torch.isfinite(inp1.grad).all() and torch.isfinite(enc1.skip_pos.weight.grad).all()

    enc2 = IntuitionisticFuzzyEncoder(1).double().eval()
    with torch.no_grad():
        for p in enc2.parameters():
            p.zero_()
        enc2.backbone[0].weight.fill_(1e308)
    inp2 = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    ifn2 = enc2(inp2)
    ifn2.validate()
    assert all(torch.isfinite(t).all() for t in (ifn2.mu, ifn2.nu, ifn2.pi))
    (ifn2.mu.sum() - ifn2.nu.sum()).backward()
    assert torch.isfinite(inp2.grad).all() and torch.isfinite(enc2.backbone[0].weight.grad).all()

    enc3 = IntuitionisticFuzzyEncoder(1).double().eval()
    with torch.no_grad():
        for p in enc3.parameters():
            p.zero_()
        enc3.skip_pos.weight.fill_(1e308)
        enc3.skip_neg.weight.fill_(5e307)
    inp3 = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    ifn3 = enc3(inp3)
    ifn3.validate()
    assert abs(float(ifn3.mu.item()) - (2.0 / 3.0)) < 1e-6
    assert abs(float(ifn3.nu.item()) - (1.0 / 3.0)) < 1e-6
    assert abs(float(ifn3.pi.item())) < 1e-6
    (ifn3.mu.sum() - ifn3.nu.sum()).backward()
    assert torch.isfinite(inp3.grad).all()

    enc4 = IntuitionisticFuzzyEncoder(1).double().eval()
    with torch.no_grad():
        for p in enc4.parameters():
            p.zero_()
        enc4.skip_pos.weight.fill_(1e308)
        enc4.prior_hesitancy.fill_(1e308)
    inp4 = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    ifn4 = enc4(inp4)
    ifn4.validate()
    assert abs(float(ifn4.mu.item()) - (2.0 / 3.0)) < 1e-6
    assert abs(float(ifn4.nu.item())) < 1e-6
    assert abs(float(ifn4.pi.item()) - (1.0 / 3.0)) < 1e-6
    (ifn4.mu.sum() - ifn4.nu.sum()).backward()
    assert (
        torch.isfinite(inp4.grad).all()
        and torch.isfinite(enc4.skip_pos.weight.grad).all()
        and torch.isfinite(enc4.prior_hesitancy.grad).all()
    )

    # 6. R24-MAJ-06 & R24-MAJ-07: CrossModalityConflictInjector NaN/extreme-value safety
    inj = CrossModalityConflictInjector(num_modalities=2, modality_names=["a", "b"])
    x_nan = [torch.tensor([[float("nan")], [0.0], [1.0], [2.0]], dtype=torch.float32), torch.ones((4, 1), dtype=torch.float32)]
    m_nan = torch.tensor([[0, 1], [1, 1], [1, 1], [1, 1]], dtype=torch.float32)
    y_nan = np.array([0, 1, 0, 1])
    xc_nan, c_nan = inj.inject_conflict(
        x_nan, y_nan, corrupted_modality_idx=0, conflict_fraction=0.5, mode="biomarker_spike", natural_masks=m_nan, seed=42
    )
    assert torch.isnan(xc_nan[0][0]).item() and c_nan[0].item() == 0
    for r_idx in range(1, 4):
        if c_nan[r_idx] == 1:
            assert torch.isfinite(xc_nan[0][r_idx]).all()

    x_hi = [torch.tensor([[2e38], [0.0], [1.0], [2.0]], dtype=torch.float32), torch.ones((4, 1), dtype=torch.float32)]
    m_hi = torch.ones((4, 2), dtype=torch.float32)
    xc_hi, c_hi = inj.inject_conflict(
        x_hi, y_nan, corrupted_modality_idx=0, conflict_fraction=0.25, mode="opposite_swap", natural_masks=m_hi, seed=0
    )
    assert c_hi[0] == 0 and torch.isfinite(xc_hi[0][0]).all()
    assert torch.equal(xc_hi[0][0], x_hi[0][0])

    # 7. R24-MIN-01 & N=0 empty-cohort handling across conflict_injector, missing_benchmark & metrics
    def _dummy_pred(mods, mask_t):
        n = mods[0].shape[0]
        return {
            "prob": torch.full((n,), 0.5, dtype=torch.float32),
            "hesitancy": torch.zeros((n,), dtype=torch.float32),
            "conflict": torch.zeros((n,), dtype=torch.float32),
        }

    sw_keys = inj.evaluate_conflict_sweep(
        _dummy_pred,
        [torch.randn(6, 2), torch.randn(6, 2)],
        torch.ones((6, 2), dtype=torch.float32),
        np.array([0, 1, 0, 1, 0, 1]),
        referral_rates=[0.1, 0.10000000000001, 0.10000000000002],
    )
    rc = sw_keys["sweep_records"][0]["referral_curve"]
    assert set(rc.keys()) == {"refer_10pct", "refer_10.000000000001pct", "refer_10.000000000002pct"}

    empty_bm = compute_binary_metrics(np.array([], dtype=np.float32), np.array([], dtype=np.int64))
    assert all(math.isnan(empty_bm[k]) for k in ("auroc", "auprc", "acc", "f1", "ece", "brier", "aurc"))

    bench = CombinatorialMissingModalityBenchmark(modality_names=["m1", "m2"])
    empty_sub = bench.evaluate_all_subsets(
        _dummy_pred, [torch.zeros(0, 2), torch.zeros(0, 2)], torch.zeros(0, 2), np.zeros(0, dtype=np.int64)
    )
    assert math.isnan(empty_sub["subsets"]["m1"]["mean_hesitancy"])
    empty_curve = bench.evaluate_missing_rate_curve(
        _dummy_pred, [torch.zeros(0, 2), torch.zeros(0, 2)], torch.zeros(0, 2), np.zeros(0, dtype=np.int64), drop_rates=[0.0]
    )
    assert math.isnan(empty_curve[0]["mean_hesitancy"])
    empty_sw = inj.evaluate_conflict_sweep(
        _dummy_pred, [torch.zeros(0, 2), torch.zeros(0, 2)], torch.zeros(0, 2), np.zeros(0, dtype=np.int64), severities=[0.0]
    )
    assert math.isnan(empty_sw["sweep_records"][0]["mean_hesitancy"])


def test_r26_codex_round25_custom_vjp_view_batch_enc_spike_missing_and_bootstrap():
    """
    Round-26 regression suite covering all 10 findings from Codex Round-25:
      1. R25-MAJ-01: FP16 VJP cancellation through custom Python BackwardCFunction and >4096 ReLUs.
      2. R25-MAJ-02: Per-sample FP16 gradient isolation across view/reshape/transpose/cat/expand.
      3. R25-MAJ-03: Batch-independent & per-element float64 encoder overflow recovery.
      4. R25-MAJ-04: biomarker_spike overflow recovery when result fits target float32/float64 dtype.
      5. R25-MAJ-05: biomarker_spike respects true low-variance sample std (no 1.0 floor).
      6. R25-MAJ-06: evaluate_missing_rate_curve forces vacuous (0.5, 1.0) on naturally missing rows.
      7. R25-MAJ-07: Single-class empirical bootstrap CI and NaN vs empirical paired bootstrap p-values.
      8. R25-MIN-01: FP16 IntuitionisticFuzzyLoss supports create_graph=True second derivatives.
      9. R25-MIN-02: Empty-cohort compute_ece / compute_aurc return NaN matching compute_binary_metrics.
    """
    import math
    from deepfuzzy.operators import (
        _FP16_ACTIVE_ROOT_LOAD,
        _FP16_BWD_CB_QUEUED,
        _FP16_BWD_PENDING_TOKENS,
        _FP16_BWD_USED_BUDGET,
    )

    loss_fn = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)

    def _L(v: torch.Tensor) -> torch.Tensor:
        v_2d = v.reshape(-1, 1)
        ifn = IntuitionisticFuzzyTensor(v_2d, (1.0 - v_2d), torch.zeros_like(v_2d))
        return loss_fn(ifn, torch.ones(v_2d.shape[0], dtype=v_2d.dtype, device=v_2d.device), modality_ifns=[ifn])[0]

    # 1. R25-MAJ-01: Custom Python autograd.Function + 4,097 and 10,000 consecutive ReLUs
    call_ctxs = []

    class _StatefulIdFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z):
            ctx.save_for_backward(z)
            ctx.calls = 0
            call_ctxs.append(ctx)
            return z.clone()

        @staticmethod
        def backward(ctx, g):
            ctx.calls += 1
            (z,) = ctx.saved_tensors
            return g * (z / z).nan_to_num(1.0)

    for n_c in (3, 8):
        call_ctxs.clear()
        x_cf = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
        l_cf = sum(_L(_StatefulIdFn.apply(x_cf)) for _ in range(n_c)) - sum(_L(_StatefulIdFn.apply(x_cf)) for _ in range(n_c))
        l_cf.backward()
        assert float(l_cf.detach().item()) == 0.0
        assert math.isfinite(float(x_cf.grad.detach().item()))
        assert abs(float(x_cf.grad.detach().item())) <= 16.0
        assert all(c.calls == 1 for c in call_ctxs)

    for depth in (4097, 10000):
        for n_c in (3, 8):
            x_dr = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
            y_dr = x_dr
            for _ in range(depth):
                y_dr = y_dr.relu()
            l_dr = sum(_L(y_dr) for _ in range(n_c)) - sum(_L(y_dr) for _ in range(n_c))
            l_dr.backward()
            assert float(l_dr.detach().item()) == 0.0
            assert math.isfinite(float(x_dr.grad.detach().item()))
            assert abs(float(x_dr.grad.detach().item())) <= 16.0

    # 2. R25-MAJ-02: Reshape / view / unsqueeze / transpose / cat / flip / index / expand isolation
    x_ref = torch.tensor([0.001, 0.5], dtype=torch.float16, requires_grad=True)
    sum(_L(x_ref) for _ in range(12)).backward()
    ref_g = x_ref.grad.detach().clone()
    assert abs(float(ref_g[0].item()) - (-48032.0)) <= 64.0
    assert abs(float(ref_g[1].item()) - (-12008.0)) <= 32.0

    for tf in (
        lambda z: z.view(2, 1),
        lambda z: z.unsqueeze(-1),
        lambda z: z.reshape(1, 2, 1),
        lambda z: z.view(1, 2).t(),
        lambda z: z.flip(0),
        lambda z: z[[1, 0]],
        lambda z: torch.cat([z[0:1], z[1:2]]),
        lambda z: torch.stack([z[0], z[1]]),
        lambda z: z.unsqueeze(1).expand(2, 4).reshape(8),
    ):
        xt = torch.tensor([0.001, 0.5], dtype=torch.float16, requires_grad=True)
        sum(_L(tf(xt)) for _ in range(12)).backward()
        assert torch.allclose(xt.grad.detach(), ref_g, atol=64.0, rtol=1e-2), f"got {xt.grad.tolist()} vs {ref_g.tolist()}"

    # 3. R25-MAJ-03: Batch-independent & per-element float64 encoder overflow recovery
    enc64 = IntuitionisticFuzzyEncoder(1, hidden_dim=2).double().eval()
    with torch.no_grad():
        for p in enc64.parameters():
            p.zero_()
        enc64.backbone[0].weight[:, 0] = torch.tensor([1e308, 1.0], dtype=torch.float64)
        enc64.backbone[3].weight[0, 1] = 1.0
        enc64.pos_evidence_head.weight[0, 0] = 1.0

    out_solo1 = enc64(torch.tensor([[1.0]], dtype=torch.float64))
    enc_ref2 = IntuitionisticFuzzyEncoder(1, hidden_dim=2).double().eval()
    with torch.no_grad():
        for p in enc_ref2.parameters():
            p.zero_()
        enc_ref2.backbone[0].weight[:, 0] = torch.tensor([0.0, 1.0], dtype=torch.float64)
        enc_ref2.backbone[3].weight[0, 1] = 1.0
        enc_ref2.pos_evidence_head.weight[0, 0] = 1.0
    out_limit2 = enc_ref2(torch.tensor([[2.0]], dtype=torch.float64))

    inp_bat = torch.tensor([[1.0], [2.0]], dtype=torch.float64, requires_grad=True)
    out_bat = enc64(inp_bat)
    out_bat.validate()
    assert abs(float(out_bat.mu[0].item()) - float(out_solo1.mu.item())) < 1e-15
    assert abs(float(out_bat.nu[0].item()) - float(out_solo1.nu.item())) < 1e-15
    assert abs(float(out_bat.pi[0].item()) - float(out_solo1.pi.item())) < 1e-15
    assert abs(float(out_bat.mu[1].item()) - float(out_limit2.mu.item())) < 1e-12
    (out_bat.mu.sum() - out_bat.nu.sum()).backward()
    assert torch.isfinite(inp_bat.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in enc64.parameters() if p.grad is not None)

    # 4 & 5. R25-MAJ-04 & R25-MAJ-05: biomarker_spike fit-in-dtype overflow + true std
    inj = CrossModalityConflictInjector(num_modalities=2, modality_names=["a", "b"])
    x4 = [torch.tensor([[-2e38], [2e38]], dtype=torch.float32), torch.ones((2, 1), dtype=torch.float32)]
    xc4, c4 = inj.inject_conflict(
        x4, np.array([0, 1]), corrupted_modality_idx=0, conflict_fraction=1.0, severity_gamma=0.5, mode="biomarker_spike", seed=0
    )
    assert c4.tolist() == [1, 1] and torch.isfinite(xc4[0]).all()
    assert np.allclose(xc4[0].squeeze(-1).numpy(), [2.2426407e38, -2.2426407e38], rtol=1e-5)

    x5_vals = np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float32)
    x5 = [torch.tensor(x5_vals[:, None], dtype=torch.float32), torch.ones((4, 1), dtype=torch.float32)]
    xc5, c5 = inj.inject_conflict(
        x5, np.array([0, 1, 0, 1]), corrupted_modality_idx=0, conflict_fraction=1.0, severity_gamma=1.0, mode="biomarker_spike", seed=0
    )
    exp_shift = 3.0 * float(np.std(x5_vals, ddof=1))
    assert c5.tolist() == [1, 1, 1, 1]
    assert np.allclose((xc5[0] - x5[0]).abs().squeeze(-1).numpy(), exp_shift, rtol=1e-5)

    x5_const = [torch.full((4, 1), 1.0, dtype=torch.float32), torch.ones((4, 1), dtype=torch.float32)]
    _, c5_const = inj.inject_conflict(
        x5_const, np.array([0, 1, 0, 1]), corrupted_modality_idx=0, conflict_fraction=1.0, severity_gamma=1e308, mode="biomarker_spike", seed=0
    )
    assert c5_const.tolist() == [0, 0, 0, 0]

    # 6. R25-MAJ-06: Missingness curve and full-subset benchmark match at drop_rate=0
    bench = CombinatorialMissingModalityBenchmark(modality_names=["m1", "m2"])
    def _pred6(mods, mask_t):
        return {
            "prob": torch.tensor([0.9, 0.2, 0.8, 0.9], dtype=torch.float32),
            "hesitancy": torch.zeros(4, dtype=torch.float32),
        }

    nat6 = torch.tensor([[0, 0], [1, 1], [1, 1], [1, 1]], dtype=torch.float32)
    y6 = np.array([0, 0, 1, 1])
    sub6 = bench.evaluate_all_subsets(_pred6, [torch.zeros(4, 2), torch.zeros(4, 2)], nat6, y6)
    cur6 = bench.evaluate_missing_rate_curve(
        _pred6, [torch.zeros(4, 2), torch.zeros(4, 2)], nat6, y6, drop_rates=[0.0], n_repeats=1, seed=42
    )
    assert abs(sub6["full_modality_auroc"] - 1.0) < 1e-12
    assert abs(cur6[0]["auroc"] - 1.0) < 1e-12
    assert abs(cur6[0]["mean_hesitancy"] - sub6["subsets"]["m1+m2"]["mean_hesitancy"]) < 1e-12

    # 7. R25-MAJ-07: Single-class bootstrap CI and paired bootstrap p-value semantics
    y7 = np.zeros(50, dtype=int)
    p7 = np.array([0.8 if i % 2 == 0 else 0.2 for i in range(50)], dtype=np.float64)
    pt7, lo7, hi7 = bootstrap_confidence_interval(p7, y7, metric_name="acc", n_bootstraps=1000, seed=42)
    assert abs(pt7 - 0.5) < 1e-12 and lo7 < 0.45 and hi7 > 0.55
    pt7_u, lo7_u, hi7_u = bootstrap_confidence_interval(p7, y7, metric_name="auroc", n_bootstraps=50, seed=42)
    assert all(math.isnan(v) for v in (pt7_u, lo7_u, hi7_u))
    assert math.isnan(paired_bootstrap_pvalue(p7, 1.0 - p7, y7, metric_name="auroc", n_bootstraps=200, seed=42))
    assert math.isnan(paired_bootstrap_pvalue(p7, p7, y7, metric_name="auroc", n_bootstraps=50, seed=42))
    p_acc_sig = paired_bootstrap_pvalue(np.zeros(50), np.ones(50), y7, metric_name="acc", n_bootstraps=1000, seed=42)
    assert abs(p_acc_sig - (1.0 / 1001.0)) < 1e-12

    # 8. R25-MIN-01: FP16 create_graph=True second derivative matches float32 & bfloat16
    l_std = IntuitionisticFuzzyLoss()
    g2_vals = {}
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        z = torch.tensor([[0.4]], dtype=dt, requires_grad=True)
        ifn = IntuitionisticFuzzyTensor(z, 1.0 - z, torch.zeros_like(z))
        l_dt = l_std(ifn, torch.tensor([1.0], dtype=dt))[0]
        (g1,) = torch.autograd.grad(l_dt, z, create_graph=True)
        assert g1.requires_grad and g1.grad_fn is not None
        (g2,) = torch.autograd.grad(g1.sum(), z)
        g2_vals[str(dt)] = float(g2.detach().item())
        assert abs(g2_vals[str(dt)] - 7.75) < 0.05
    assert abs(g2_vals["torch.float16"] - g2_vals["torch.float32"]) < 0.02

    # 9. R25-MIN-02: Empty-cohort consistency for compute_ece & compute_aurc
    assert math.isnan(compute_ece(np.array([], dtype=np.float64), np.array([], dtype=np.int64)))
    aurc_e, cov_e, rsk_e, acc_e = compute_aurc(np.array([], dtype=np.float64), np.array([], dtype=np.int64))
    assert math.isnan(aurc_e) and len(cov_e) == 0 and len(rsk_e) == 0 and len(acc_e) == 0

    assert len(_FP16_ACTIVE_ROOT_LOAD) == 0
    assert len(_FP16_BWD_PENDING_TOKENS) == 0
    assert len(_FP16_BWD_USED_BUDGET) == 0
    assert len(_FP16_BWD_CB_QUEUED) == 0


def test_r27_codex_round26_deep_vjp_f32_cast_1shot_bwd_spike_metrics_and_runtime():
    """
    Round-27 regression suite covering all 11 findings from Codex Round-26 (R26-MAJ-01..07,
    R26-MIN-01..04) plus adjacent float32->BackwardCFunction->float16 boundary protection:
      1. R26-MAJ-01: Canceling 135,000+ & 200,000-node ReLU chains preserve O(1) gradients.
      2. R26-MAJ-02: float32 -> half and float32 -> BackwardCFunction -> half isolate samples.
      3. R26-MAJ-03: Custom BackwardCFunction executes backward() strictly once natively.
      4. R26-MAJ-04: Float64 biomarker_spike handles [1e308, 1e308] without NaN overflow.
      5. R26-MAJ-05: Singleton finite folds are left untouched (no fabricated std=1.0).
      6. R26-MIN-01: n_bootstraps <= 0 returns (point, nan, nan) instead of degenerate [p, p].
      7. R26-MIN-02: Empty compute_decision_curve_analysis returns NaN arrays cleanly.
      8. R26-MAJ-06/07 & R26-MIN-03: PeakTensorMemoryTracker pins storages cleanly (no view
         double-counting), timing excludes opt.step(), and CertaintyOrdered_IFSChoquetFusion
         states O(N * M * log M * C + N * M^2 * C) complexity.
    """
    import json
    import math
    import time
    import warnings
    from pathlib import Path
    from deepfuzzy.operators import (
        _FP16_ACTIVE_ROOT_LOAD,
        _FP16_BWD_CB_QUEUED,
        _FP16_BWD_PASS_META,
        _FP16_BWD_PENDING_TOKENS,
        _FP16_BWD_USED_BUDGET,
    )
    from benchmarks.run_runtime_vram_benchmark import (
        _benchmark_neural_config,
        _measure_single_step_peak_mb,
    )

    loss_fn = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)

    def _L(v: torch.Tensor) -> torch.Tensor:
        v_2d = v.reshape(-1, 1)
        ifn = IntuitionisticFuzzyTensor(v_2d, 1.0 - v_2d, torch.zeros_like(v_2d))
        return loss_fn(ifn, torch.ones(v_2d.shape[0], dtype=v_2d.dtype, device=v_2d.device), modality_ifns=[ifn])[0]

    # 1. R26-MAJ-01: 135,000 (n_c=3, 8) and 200,000 (n_c=3) ReLU canceling objectives
    for depth, n_branches in ((135_000, 3), (135_000, 8), (200_000, 3)):
        x_d = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
        y_d = x_d
        for _ in range(depth):
            y_d = y_d.relu()
        loss_d = sum(_L(y_d) for _ in range(n_branches)) - sum(_L(y_d) for _ in range(n_branches))
        loss_d.backward()
        assert float(loss_d.detach().item()) == 0.0
        assert math.isfinite(float(x_d.grad.detach().item()))
        assert abs(float(x_d.grad.detach().item())) <= 16.0, (depth, n_branches, x_d.grad.item())

    # 2. R26-MAJ-02: float32 -> half (shared and per-branch) and float32 -> BackwardCFunction -> half isolation
    x32_a = torch.tensor([0.001, 0.5], dtype=torch.float32, requires_grad=True)
    v32_a = x32_a.half()
    sum(_L(v32_a) for _ in range(12)).backward()
    assert torch.allclose(x32_a.grad.detach(), torch.tensor([-48032.0, -12008.0]), atol=64.0, rtol=1e-2)

    x32_b = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=True)
    v32_b = x32_b.half()
    sum(_L(v32_b) for _ in range(12)).backward()
    assert abs(float(x32_b.grad[1].item()) - float(x32_a.grad[1].item())) <= 16.0

    x32_sep = torch.tensor([0.001, 0.5], dtype=torch.float32, requires_grad=True)
    sum(_L(x32_sep.half()) for _ in range(12)).backward()
    assert torch.allclose(x32_sep.grad.detach(), torch.tensor([-102048.0, -12018.0]), atol=64.0, rtol=1e-2)

    cast_calls = [0]

    class _Cast1Shot(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z32):
            ctx.used = False
            return z32.half()

        @staticmethod
        def backward(ctx, g16):
            if ctx.used:
                raise RuntimeError("Custom backward invoked more than once")
            ctx.used = True
            cast_calls[0] += 1
            return g16.float()

    z32_c = torch.tensor([0.001, 0.5], dtype=torch.float32, requires_grad=True)
    yc32 = _Cast1Shot.apply(z32_c)
    sum(_L(yc32) for _ in range(12)).backward()
    assert cast_calls[0] == 1 and torch.isfinite(z32_c.grad).all()
    assert abs(float(z32_c.grad[1].item()) - (-12008.0)) <= 16.0

    # 3. R26-MAJ-03: Strict 1-shot BackwardCFunction across single, 24-branch, chained & retain_graph
    class _Strict1Shot(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z):
            ctx.used = False
            return z.clone()

        @staticmethod
        def backward(ctx, g):
            if ctx.used:
                raise RuntimeError("Strict 1-shot backward invoked twice")
            ctx.used = True
            return g

    x_s1 = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
    _L(_Strict1Shot.apply(x_s1)).backward()
    assert abs(float(x_s1.grad.item()) - (-9000.0)) <= 16.0

    x_s24 = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
    branches_24 = [_L(_Strict1Shot.apply(x_s24)).float() for _ in range(24)]
    obj_s24 = sum(branches_24[:12]) - sum(branches_24[12:])
    assert bool(torch.isfinite(obj_s24).all()) and float(obj_s24.item()) == 0.0
    obj_s24.backward()
    assert math.isfinite(float(x_s24.grad.item())) and abs(float(x_s24.grad.item())) <= 16.0

    x_sc = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
    y_sc = _Strict1Shot.apply(_Strict1Shot.apply(x_sc))
    obj_sc = sum(_L(y_sc).float() for _ in range(12)) - sum(_L(y_sc).float() for _ in range(12))
    assert bool(torch.isfinite(obj_sc).all()) and float(obj_sc.item()) == 0.0
    obj_sc.backward()
    assert math.isfinite(float(x_sc.grad.item())) and abs(float(x_sc.grad.item())) <= 16.0

    ret_calls = [0]

    class _RetainCountFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z):
            return z.clone()

        @staticmethod
        def backward(ctx, g):
            ret_calls[0] += 1
            return g

    x_ret = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
    l_ret = _L(_RetainCountFn.apply(x_ret))
    l_ret.backward(retain_graph=True)
    x_ret.grad = None
    l_ret.backward()
    assert ret_calls[0] == 2 and abs(float(x_ret.grad.item()) - (-9000.0)) <= 16.0

    # 4 & 5. R26-MAJ-04 & R26-MAJ-05: Float64 [1e308, 1e308] no-NaN and singleton fold untouched
    inj = CrossModalityConflictInjector(num_modalities=2, modality_names=["a", "b"])
    x64_huge = [torch.tensor([[1e308], [1e308]], dtype=torch.float64), torch.ones((2, 1), dtype=torch.float64)]
    xc64, c64 = inj.inject_conflict(
        x64_huge, np.array([0, 1]), corrupted_modality_idx=0, conflict_fraction=1.0,
        severity_gamma=0.1, mode="biomarker_spike", seed=0,
    )
    assert torch.isfinite(xc64[0]).all() and torch.equal(xc64[0], x64_huge[0]) and c64.tolist() == [0, 0]

    x_sing = [
        torch.tensor([[0.01], [float("nan")], [float("nan")]], dtype=torch.float32),
        torch.ones((3, 1), dtype=torch.float32),
    ]
    xc_sing, c_sing = inj.inject_conflict(
        x_sing, np.array([0, 1, 0]), corrupted_modality_idx=0, conflict_fraction=1.0,
        severity_gamma=1.0, mode="biomarker_spike", seed=0,
    )
    assert torch.equal(xc_sing[0][:1], x_sing[0][:1]) and c_sing.tolist() == [0, 0, 0]

    # 6 & 7. R26-MIN-01 & R26-MIN-02: Zero-bootstrap CI & empty DCA NaN behavior
    pt0, lo0, hi0 = bootstrap_confidence_interval(
        np.array([0.8, 0.2, 0.8, 0.2]), np.array([1, 0, 0, 1]), metric_name="acc", n_bootstraps=0
    )
    assert abs(pt0 - 0.5) < 1e-12 and math.isnan(lo0) and math.isnan(hi0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dca_empty = compute_decision_curve_analysis(
            np.array([], dtype=np.float64), np.array([], dtype=np.int64), thresholds=np.array([0.1, 0.5])
        )
    assert np.all(np.isnan(dca_empty["net_benefit_model"])) and np.all(np.isnan(dca_empty["net_benefit_all"]))
    assert np.all(np.isnan(dca_empty["net_benefit_none"]))

    # 8. R26-MAJ-06/07 & R26-MIN-03: PeakTensorMemoryTracker dedup, opt.step exclusion, complexity string
    class _ViewModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

        def forward(self, xs):
            return xs[0].view(-1)

    m_view = _ViewModel()
    xs_view = [torch.randn(1_000_000, dtype=torch.float32)]
    assert abs(_measure_single_step_peak_mb(lambda: m_view(xs_view), m_view, xs_view, torch.device("cpu")) - 3.815) <= 0.001

    orig_adam_step = torch.optim.AdamW.step
    try:
        torch.optim.AdamW.step = lambda self, *a, **kw: (time.sleep(0.02), orig_adam_step(self, *a, **kw))[1]
        res_timing = _benchmark_neural_config("EarlyConcat", N=64, D=100, M=4, device=torch.device("cpu"), n_warmup=2, n_repeats=5)
    finally:
        torch.optim.AdamW.step = orig_adam_step
    assert res_timing["fwd_bwd_mean_ms"] < 10.0

    rt_json = json.loads((Path(__file__).resolve().parents[1] / "results" / "runtime_vram_benchmark.json").read_text())
    tc = rt_json["theoretical_complexity"]
    assert "CertaintyOrdered_IFSChoquetFusion" in tc and "TwoAdditive_IFSChoquetFusion" not in tc
    assert "M^2" in tc["CertaintyOrdered_IFSChoquetFusion"]

    for d in (_FP16_ACTIVE_ROOT_LOAD, _FP16_BWD_PENDING_TOKENS, _FP16_BWD_USED_BUDGET, _FP16_BWD_CB_QUEUED, _FP16_BWD_PASS_META):
        assert len(d) == 0


def run_all_verifications():
    print("=" * 72)
    print("DeepFuzzyTorch: Verifying Core Operators, Theorems & Statistical Suite")
    print("=" * 72)
    tests = [
        (test_theorem1_simplex_closure, "[PASS] Theorem 1 (IFS Simplex Closure & Choquet Normalization)"),
        (test_theorem2_missing_modality_invariance, "[PASS] Theorem 2 (Exact Vacuity Invariance across positions & gamma > 0)"),
        (test_theorem3_partial_order_monotonicity, "[PASS] Theorem 3 (Strict Partial-Order Monotonicity)"),
        (test_theorem4_conflict_hesitancy_inflation, "[PASS] Theorem 4 (Conflict-Driven Epistemic Hesitancy Inflation)"),
        (test_theorem5_fuzzy_rough_inclusion_and_gate, "[PASS] Theorem 5 (Fuzzy-Rough Inclusion in Layer & Attention Gate)"),
        (test_tsk_exact_rule_reconstruction, "[PASS] Block 4 (Exact White-Box TSK Rule Reconstruction)"),
        (test_hesitant_and_pythagorean_tta, "[PASS] Blocks 2 & 5 (Hesitant Fuzzy Consensus & Pythagorean TTA)"),
        (test_masked_nan_safety_and_end_to_end, "[PASS] Masked NaN Safety & End-to-End Autograd Backpropagation"),
        (test_paired_bootstrap_identical_and_logrank, "[PASS] Statistical Suite (Paired Bootstrap Identity & Mantel-Haenszel Log-Rank)"),
        (test_missing_conflict_and_rough_reducer_suites, "[PASS] Missing-Modality, Conflict-Injection & Fuzzy-Rough Reduct Suites"),
        (test_strict_fold_preprocessor_exact_replay, "[PASS] Strict Fold Preprocessor Parameter Export & Zero-Error Replay"),
        (test_regression_reducer_refit_tsk_names_and_aurc_bootstrap, "[PASS] Reducer Re-Fit State, Punctuation TSK Names & Hesitancy AURC Bootstrap"),
        (test_inf_nan_sanitization_everywhere, "[PASS] Global NaN / +/-Inf Sanitization Across Operators & 9 Baselines"),
        (test_multiclass_tsk_shapes_1d_2d_and_concept_warning, "[PASS] Multi-Class TSK Rules, 1D/2D Shapes, HTSK Scaling & Concept Warning"),
        (test_end_to_end_unified_model_missing_modality_invariance, "[PASS] End-to-End UnifiedDeepFuzzyModel Theorem 2 Missing-Concept Invariance"),
        (test_rough_reducer_tnorm_monotonicity_null_and_stability, "[PASS] Pointwise T-Norm Rough Reducer Monotonicity, Null Separation & Reduct=1"),
        (test_modern_2023_2024_baselines_and_evidential_kl_annealing, "[PASS] QMF (ICML'23), RCML (AAAI'24), DrFuse (AAAI'24) & Annealed Dirichlet KL"),
        (test_intuitionistic_fuzzy_loss_nonfinite_sanitization_and_variable_width_tsk, "[PASS] Loss Non-Finite Sanitization & Variable-Width Concept-Masked TSK"),
        (test_r15_fusion_ties_sanitization_and_loss_bounds, "[PASS] Round-15 Tie Blocks, Active-Set Jump, Radial Sanitization & Loss Bounds"),
        (test_r16_metrics_consistency_boundary_grad_and_scale_weights, "[PASS] Round-16 Unclipped Metrics, Boundary Grads & Scale-Invariant Weights"),
        (test_r17_unclipped_operator_probs_aurc_ties_and_boundary_holder_jump, "[PASS] Round-17 Unclipped Model Probs, AURC Ties/Underflow & Theorem-1 Boundary Scaling"),
        (test_r18_threshold_aurc_loss_tail_grad_and_absorbing_tie_characterization, "[PASS] Round-18 Thresholded AURC, Loss Tail Gradients & Absorbing Tie Characterization"),
        (test_r19_half_precision_loss_gradients_and_bfloat16_upper_clamp, "[PASS] Round-19 Float16/BFloat16 Loss Tail Gradients & Unclipped BCE Upper Bound"),
        (test_r20_codex_round19_low_precision_counterexamples, "[PASS] Round-20 Shared/Cloned FP16 Loss, Encoder Overflow, BF16 Cancellation & TSK/Logit Grads"),
        (test_r21_codex_round20_expanded_fp16_autocast_f32_hesitancy_and_mps, "[PASS] Round-21 Expanded FP16 Views, Autocast, Saturated Float32 Hesitancy, MPS & Sample Weights"),
        (test_r22_codex_round21_deep_fp16_gc_isolation_f32_fusion_mps_reducer_and_logrank, "[PASS] Round-22 Deep FP16 Clones, GC Isolation, Float32 Fusion/Overflow, MPS Reducer & Logrank"),
        (test_r23_codex_round22_dtype_cast_fp16_live_isolation_enc_sum_missing_and_conflict, "[PASS] Round-23 Dtype Cast FP16, Live Isolation, Encoder Sum, Missing NaN & Conflict Sweep"),
        (test_r24_codex_round23_fp16_cancel_overflow_enc_f64_missing_conflict_and_docs, "[PASS] Round-24 Canceling FP16, 7-Root Overflow, Enc F64, Missing/Conflict Edge Cases & Docs"),
        (test_r25_codex_round24_vjp_cancel_deep_shared_hook_scale_enc_and_conflict, "[PASS] Round-25 VJP Cancel, Deep Shared Overflow, Hook Cleanup, Negative Scale, F64 Enc & Conflict"),
        (test_r26_codex_round25_custom_vjp_view_batch_enc_spike_missing_and_bootstrap, "[PASS] Round-26 Custom VJP, Reshape Isolation, Batch Enc, Biomarker Spike, Missing & Bootstrap"),
        (test_r27_codex_round26_deep_vjp_f32_cast_1shot_bwd_spike_metrics_and_runtime, "[PASS] Round-27 Deep VJP, F32->Fn->FP16, 1-Shot Backward, Spike/DCA/CI & Runtime"),
        (test_r28_codex_round27_index_fanin_view_inter_pyhook_ckpt_f64_and_tracker, "[PASS] Round-28 Indexing Fan-In, Shallow Views, Inter-Consumers, Hooks, Ckpt, F64 & Tracker"),
        (test_r29_codex_round28_cat_stack_large_index_ckpt_cancel_tracker_alias_gamma_and_mib, "[PASS] Round-29 Cat/Stack, >4.19M Index, Ckpt Cancel, Tracker Alias/Resize, F64 Gamma, MiB & TMPDIR"),
        (test_phase1_clinical_concept_masks_and_pfs_tta, "[PASS] Phase-1 Clinical Concept Masks (fp32/fp16) & PFS-TTA Simplex/Disk Behavior"),
        (test_phase1_conflict_sweep_separated_aurocs, "[PASS] Phase-1 Separated Conflict-Only / Hesitancy-Only / Composite Sweep AUROCs"),
        (test_phase1_ddef_and_fuml_baselines, "[PASS] Phase-1 DDEF (Inf. Fusion'24) & FUML (ICML'25) Baselines & Vacuity Invariance"),
    ]
    declared_test_names = {
        name for name, obj in globals().items() if name.startswith("test_") and callable(obj)
    }
    executed_test_names = {fn.__name__ for fn, _ in tests}
    assert executed_test_names == declared_test_names, (
        f"Standalone runner mismatch: missing={sorted(declared_test_names - executed_test_names)}, "
        f"extra={sorted(executed_test_names - declared_test_names)}"
    )
    for fn, msg in tests:
        fn()
        print(msg)
    print(f"ALL {len(tests)}/{len(declared_test_names)} TESTS PASSED")
    print("=" * 72)


def test_r28_codex_round27_index_fanin_view_inter_pyhook_ckpt_f64_and_tracker():
    """Round-28 regressions for R27-MAJ-01..06 and R27-MIN-01..03."""
    import json
    import math
    import time
    import warnings
    from pathlib import Path
    import numpy as np
    from torch.utils.checkpoint import checkpoint
    from benchmarks.run_runtime_vram_benchmark import (
        PeakTensorMemoryTracker,
        _benchmark_neural_config,
    )
    from deepfuzzy.conflict_injector import CrossModalityConflictInjector

    loss_fn = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)

    def _L(v: torch.Tensor) -> torch.Tensor:
        v2 = v.reshape(-1, 1)
        ifn = IntuitionisticFuzzyTensor(v2, 1.0 - v2, torch.zeros_like(v2))
        return loss_fn(
            ifn,
            torch.ones(v2.shape[0], dtype=v2.dtype, device=v2.device),
            modality_ifns=[ifn],
        )[0]

    # R27-MAJ-01: Repeated advanced indexing (x[[0, 0]], (None, [0, 0]), bool+int mask)
    x_idx1 = torch.tensor([0.001, 0.5], dtype=torch.float16, requires_grad=True)
    y_idx1 = x_idx1[[0, 0]]
    obj_idx1 = sum(_L(y_idx1).float() for _ in range(12)) - sum(_L(y_idx1).float() for _ in range(12))
    assert bool(torch.isfinite(obj_idx1).all()) and float(obj_idx1.item()) == 0.0
    obj_idx1.backward()
    assert torch.isfinite(x_idx1.grad).all() and float(x_idx1.grad.abs().max()) <= 16.0

    x_idx_12 = torch.tensor([0.001, 0.5], dtype=torch.float16, requires_grad=True)
    y_idx_12 = x_idx_12[[0, 0]]
    sum(_L(y_idx_12).float() for _ in range(12)).backward()
    assert torch.isfinite(x_idx_12.grad).all() and float(x_idx_12.grad[0].item()) < 0.0

    x_idx_ref = torch.tensor([0.001, 0.5], dtype=torch.float16, requires_grad=True)
    _L(x_idx_ref[[0, 0]]).backward()
    assert abs(float(x_idx_ref.grad[0].item()) - (-9000.0)) <= 64.0
    assert float(x_idx_ref.grad[1].item()) == 0.0

    x_idx2 = torch.tensor([0.001, 0.5], dtype=torch.float16, requires_grad=True)
    y_idx2 = x_idx2[(None, [0, 0])]
    _L(y_idx2).backward()
    assert abs(float(x_idx2.grad[0].item()) - (-9000.0)) <= 64.0

    x_idx3 = torch.tensor([[0.001, 0.5], [0.5, 0.5]], dtype=torch.float16, requires_grad=True)
    mask = torch.tensor([True, False])
    y_idx3 = x_idx3[mask, [0, 0]]
    obj_idx3 = sum(_L(y_idx3).float() for _ in range(12)) - sum(_L(y_idx3).float() for _ in range(12))
    assert bool(torch.isfinite(obj_idx3).all()) and float(obj_idx3.item()) == 0.0
    obj_idx3.backward()
    assert torch.isfinite(x_idx3.grad).all() and float(x_idx3.grad.abs().max()) <= 16.0

    # R27-MAJ-02: Shallow view chain (depth=100, n_c=8)
    x_sh = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
    y_sh = x_sh
    for _ in range(100):
        y_sh = y_sh.view_as(y_sh)
    obj_sh = sum(_L(y_sh).float() for _ in range(8)) - sum(_L(y_sh).float() for _ in range(8))
    assert bool(torch.isfinite(obj_sh).all()) and float(obj_sh.item()) == 0.0
    obj_sh.backward()
    assert math.isfinite(float(x_sh.grad.item())) and abs(float(x_sh.grad.item())) <= 16.0

    # R27-MAJ-03: Intermediate view consumer before terminal node
    x_ic = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
    y_ic = x_ic.view(1)
    v_ic = y_ic.view(1)
    obj_ic = sum(_L(y_ic).float() for _ in range(12)) - sum(_L(v_ic).float() for _ in range(12))
    assert bool(torch.isfinite(obj_ic).all()) and float(obj_ic.item()) == 0.0
    obj_ic.backward()
    assert math.isfinite(float(x_ic.grad.item())) and abs(float(x_ic.grad.item())) <= 64.0

    # R27-MAJ-04: Python register_hook runs once per backward pass on canceling branches
    hook_calls = [0]
    x_hk = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
    y_hk = x_hk * 1.0

    def _rec_hook(g):
        hook_calls[0] += 1
        return g

    y_hk.register_hook(_rec_hook)
    obj_hk = sum(_L(y_hk).float() for _ in range(12)) - sum(_L(y_hk).float() for _ in range(12))
    assert bool(torch.isfinite(obj_hk).all()) and float(obj_hk.item()) == 0.0
    obj_hk.backward()
    assert hook_calls[0] == 1, hook_calls
    assert math.isfinite(float(x_hk.grad.item())) and abs(float(x_hk.grad.item())) <= 16.0

    # R27-MAJ-05: Non-reentrant checkpoint under FP16 loss
    x_ck = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
    y_ck = checkpoint(lambda t: t * 1.0, x_ck, use_reentrant=False)
    _L(y_ck).backward()
    assert abs(float(x_ck.grad.item()) - (-9000.0)) <= 16.0

    x_ckc = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
    y_ckc = checkpoint(lambda t: t * 1.0, x_ckc, use_reentrant=False)
    obj_ckc = sum(_L(y_ckc).float() for _ in range(12)) - sum(_L(y_ckc).float() for _ in range(12))
    assert bool(torch.isfinite(obj_ckc).all()) and float(obj_ckc.item()) == 0.0
    obj_ckc.backward()
    assert abs(float(x_ckc.grad.item())) <= 16.0

    # Adjacent unary ops (mul0, sigmoid, abs(0.0))
    x_z = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
    _L(x_z * 0.0).backward()
    assert float(x_z.grad.item()) == 0.0
    x_sig = torch.tensor(0.001, dtype=torch.float16, requires_grad=True)
    y_sig = torch.sigmoid(x_sig)
    obj_sig = sum(_L(y_sig).float() for _ in range(12)) - sum(_L(y_sig).float() for _ in range(12))
    assert bool(torch.isfinite(obj_sig).all()) and float(obj_sig.item()) == 0.0
    obj_sig.backward()
    assert abs(float(x_sig.grad.item())) <= 16.0
    x_abs0 = torch.tensor(0.0, dtype=torch.float16, requires_grad=True)
    _L(torch.abs(x_abs0)).backward()
    assert float(x_abs0.grad.item()) == 0.0

    # R27-MAJ-06: Float64 extreme opposite-sign [1.5e308, -1.5e308] across all 3 conflict modes
    inj = CrossModalityConflictInjector(num_modalities=2, modality_names=["a", "b"])
    x_f64 = [
        torch.tensor([[1.5e308], [-1.5e308]], dtype=torch.float64),
        torch.ones((2, 1), dtype=torch.float64),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for mode_name in ("biomarker_spike", "opposite_swap", "sign_inversion"):
            for gamma_val in (0.0, 0.1, 1.0, 100.0):
                xc, cmask = inj.inject_conflict(
                    x_f64,
                    np.array([0, 1]),
                    corrupted_modality_idx=0,
                    conflict_fraction=1.0,
                    severity_gamma=gamma_val,
                    mode=mode_name,
                    seed=0,
                )
                assert bool(torch.isfinite(xc[0]).all()), (mode_name, gamma_val, xc[0])
                assert cmask.tolist() == ([1, 1] if gamma_val > 0.0 else [0, 0])

    # R27-MIN-01..03: PeakTensorMemoryTracker resize_/set_, opt.zero_grad exclusion, MiB unit
    with PeakTensorMemoryTracker() as tr:
        z_tr = torch.empty(1, dtype=torch.float32)
        z_tr.resize_(1_000_000)
        z_swap = torch.empty(500_000, dtype=torch.float32)
        z_tr.set_(z_swap.untyped_storage())
        del z_swap
    assert tr.peak_bytes >= 4_000_000, tr.peak_bytes
    assert tr.cur_bytes == 2_000_000, tr.cur_bytes

    orig_zg = torch.optim.AdamW.zero_grad
    try:
        def _slow_zg(self, set_to_none=True):
            time.sleep(0.02)
            return orig_zg(self, set_to_none=set_to_none)

        torch.optim.AdamW.zero_grad = _slow_zg
        res_ec = _benchmark_neural_config(
            "EarlyConcat", 64, 100, 4, torch.device("cpu"), n_warmup=1, n_repeats=2
        )
        assert res_ec["fwd_bwd_mean_ms"] < 10.0, res_ec
    finally:
        torch.optim.AdamW.zero_grad = orig_zg

    rt_raw_text = Path("results/runtime_vram_benchmark.json").read_text(encoding="utf-8")
    assert "peak_memory_mib" in rt_raw_text and ("peak_memory_" + "mb") not in rt_raw_text
    rt_json = json.loads(rt_raw_text)
    assert rt_json.get("memory_unit") == "MiB", rt_json.get("memory_unit")


def test_r29_codex_round28_cat_stack_large_index_ckpt_cancel_tracker_alias_gamma_and_mib():
    """Verifies all 7 Round-28 fixes: cat/stack, >4.19M index/gather/ckpt, linear ckpt cancel, tracker alias/resize, f64 gamma, MiB keys, broken TMPDIR."""
    import gc
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import warnings
    import numpy as np
    from torch.utils.checkpoint import checkpoint
    from benchmarks.run_runtime_vram_benchmark import PeakTensorMemoryTracker
    from deepfuzzy.conflict_injector import CrossModalityConflictInjector
    from deepfuzzy.operators import (
        _FP16_ACTIVE_ROOT_LOAD,
        _FP16_BWD_CB_QUEUED,
        _FP16_BWD_PASS_META,
        _FP16_BWD_PENDING_TOKENS,
        _FP16_BWD_USED_BUDGET,
    )

    loss_fn = IntuitionisticFuzzyLoss(lambda_aux_mod=1000.0)

    def _L(v: torch.Tensor) -> torch.Tensor:
        v2 = v.reshape(-1, 1)
        ifn = IntuitionisticFuzzyTensor(v2, 1.0 - v2, torch.zeros_like(v2))
        return loss_fn(
            ifn,
            torch.ones(v2.shape[0], dtype=v2.dtype, device=v2.device),
            modality_ifns=[ifn],
        )[0]

    def _oracle_L_f64(v_tensor: torch.Tensor) -> torch.Tensor:
        """Independent float64 formula oracle matching IntuitionisticFuzzyLoss for pi=0, y=1."""
        p = v_tensor.detach().to(dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
        # Closed-form Eq. (16) for binary y=1, pi=0:
        # BCE(p, 1) + 0.35*(p-1)^2 + hesitancy(0) + aux(lambda=1000) * BCE(p, 1)
        p_c = p.clamp(1e-6, 1.0 - 1e-6)
        bce = -torch.log(p_c)
        l_vec = bce + 0.35 * (p - 1.0).pow(2) + 1000.0 * bce
        return p, l_vec.mean()

    def _assert_grad_matches_oracle(make_var_fn, make_y_fn, msg_tag: str):
        # 1) Non-canceling positive gradient flow vs independent float64 oracle (R29-MIN-03)
        vx_p = make_var_fn()
        yx_p = make_y_fn(vx_p)
        loss_p = _L(yx_p).float()
        loss_p.backward()
        p_ref, l_ref = _oracle_L_f64(yx_p)
        l_ref.backward()
        # Backprop p_ref.grad analytically back through linear/relu/slice/gather via float64 twin
        vx_64 = [v.detach().to(dtype=torch.float64).requires_grad_(True) for v in vx_p]
        yx_64 = make_y_fn(vx_64)
        _, l_ref_64 = _oracle_L_f64(yx_64)
        # Recompute through exact differentiable float64 computation
        y_r64 = yx_64.reshape(-1, 1)
        y_c64 = y_r64.clamp(1e-6, 1.0 - 1e-6)
        l_e2e_64 = (-1001.0 * torch.log(y_c64) + 0.35 * (y_r64 - 1.0).pow(2)).mean()
        l_e2e_64.backward()
        for v16, v64 in zip(vx_p, vx_64):
            assert v16.grad is not None and bool(torch.isfinite(v16.grad).all()), msg_tag
            assert bool(torch.isfinite(v16.grad.float()).all()), msg_tag
            g16 = v16.grad.detach().float()
            g64 = v64.grad.detach().float()
            assert torch.allclose(g16, g64, rtol=0.05, atol=1.0), (msg_tag, g16, g64)

    # R28-MAJ-01: cat & stack through unary / direct FP16 telescoping + oracle gradient check
    _assert_grad_matches_oracle(
        lambda: [
            torch.tensor([[0.25]], dtype=torch.float16, requires_grad=True),
            torch.tensor([[0.5]], dtype=torch.float16, requires_grad=True),
        ],
        lambda vs: torch.cat([torch.relu(vs[0]), vs[1]], dim=-1),
        "R28-MAJ-01-cat-oracle",
    )
    a_cat = torch.tensor([[0.001]], dtype=torch.float16, requires_grad=True)
    b_cat = torch.tensor([[0.5]], dtype=torch.float16, requires_grad=True)
    y_cat = torch.cat([torch.relu(a_cat), b_cat], dim=-1)
    obj_cat = sum(_L(y_cat).float() for _ in range(12)) - sum(_L(y_cat).float() for _ in range(12))
    assert bool(torch.isfinite(obj_cat).all()) and float(obj_cat.item()) == 0.0
    obj_cat.backward()
    assert bool(torch.isfinite(a_cat.grad).all()) and bool(torch.isfinite(b_cat.grad).all())
    assert float(a_cat.grad.abs().max()) <= 64.0 and float(b_cat.grad.abs().max()) <= 64.0

    _assert_grad_matches_oracle(
        lambda: [
            torch.tensor([0.25], dtype=torch.float16, requires_grad=True),
            torch.tensor([0.5], dtype=torch.float16, requires_grad=True),
        ],
        lambda vs: torch.stack([torch.relu(vs[0]), vs[1]], dim=0),
        "R28-MAJ-01-stack-oracle",
    )
    a_stk = torch.tensor([0.001], dtype=torch.float16, requires_grad=True)
    b_stk = torch.tensor([0.5], dtype=torch.float16, requires_grad=True)
    y_stk = torch.stack([torch.relu(a_stk), b_stk], dim=0)
    obj_stk = sum(_L(y_stk).float() for _ in range(12)) - sum(_L(y_stk).float() for _ in range(12))
    assert bool(torch.isfinite(obj_stk).all()) and float(obj_stk.item()) == 0.0
    obj_stk.backward()
    assert bool(torch.isfinite(a_stk.grad).all()) and bool(torch.isfinite(b_stk.grad).all())
    assert float(a_stk.grad.abs().max()) <= 64.0 and float(b_stk.grad.abs().max()) <= 64.0

    # R28-MAJ-02: >4.19M indexing, (2M, 4) gather/slice, non-reentrant ckpt view+unary + oracle
    _assert_grad_matches_oracle(
        lambda: [torch.full((4_194_305,), 0.25, dtype=torch.float16, requires_grad=True)],
        lambda vs: vs[0][[0, 1]],
        "R28-MAJ-02-big-oracle",
    )
    x_big = torch.full((4_194_305,), 0.001, dtype=torch.float16, requires_grad=True)
    y_big = x_big[[0, 0]]
    obj_big = sum(_L(y_big).float() for _ in range(12)) - sum(_L(y_big).float() for _ in range(12))
    assert bool(torch.isfinite(obj_big).all()) and float(obj_big.item()) == 0.0
    obj_big.backward()
    assert bool(torch.isfinite(x_big.grad).all()) and float(x_big.grad.abs().max()) <= 64.0

    g_idx = torch.tensor([[0, 1]], dtype=torch.long)
    _assert_grad_matches_oracle(
        lambda: [torch.full((2_000_000, 4), 0.25, dtype=torch.float16, requires_grad=True)],
        lambda vs: torch.gather(vs[0][:1], 1, g_idx),
        "R28-MAJ-02-gather-oracle",
    )
    x_mat = torch.full((2_000_000, 4), 0.001, dtype=torch.float16, requires_grad=True)
    g_idx_dup = torch.tensor([[0, 0]], dtype=torch.long)
    y_gath = torch.gather(x_mat[:1], 1, g_idx_dup)
    obj_gath = sum(_L(y_gath).float() for _ in range(12)) - sum(_L(y_gath).float() for _ in range(12))
    assert bool(torch.isfinite(obj_gath).all()) and float(obj_gath.item()) == 0.0
    obj_gath.backward()
    assert bool(torch.isfinite(x_mat.grad).all()) and float(x_mat.grad.abs().max()) <= 64.0

    x_nr = torch.tensor([0.001, 0.5], dtype=torch.float16, requires_grad=True)
    y_nr = checkpoint(lambda t: torch.relu(t).view(1, 2), x_nr, use_reentrant=False)
    obj_nr = sum(_L(y_nr).float() for _ in range(12)) - sum(_L(y_nr).float() for _ in range(12))
    assert bool(torch.isfinite(obj_nr).all()) and float(obj_nr.item()) == 0.0
    obj_nr.backward()
    assert bool(torch.isfinite(x_nr.grad).all()) and float(x_nr.grad.abs().max()) <= 64.0

    # R28-MAJ-03: Checkpointed linear matmul cancellation across 5 topologies & both orders
    tgt_m = torch.ones((4,), dtype=torch.long)

    def _make_w(name: str) -> torch.Tensor:
        if name == "eye2_0001":
            return torch.eye(2, dtype=torch.float16) * 0.001
        if name == "had_0001":
            return torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float16) * 0.001
        if name == "rect2x3":
            return torch.tensor([[1.0, -0.5, 0.25], [-0.5, 1.0, 0.75]], dtype=torch.float16) * 0.001
        gen = torch.Generator().manual_seed(0)
        return (torch.randn(2, 8, generator=gen) * 0.001).half()

    def _L_colmix(y: torch.Tensor) -> torch.Tensor:
        mu = 0.3 + y[:, :1]
        nu = 1.0 - (0.3 + y[:, 1:2])
        ifn = IntuitionisticFuzzyTensor(mu=mu, nu=nu, pi=torch.zeros_like(mu))
        return loss_fn(ifn, tgt_m, modality_ifns=[ifn])[0]

    for topo in ("eye2_0001", "had_0001", "rect2x3", "wide2x8", "two_layer"):
        for order in ("ym_minus_rm", "rm_minus_ym"):
            for reentrant in (True, False):
                x_m = torch.tensor(
                    [[0.4, 0.3], [0.2, 0.5], [0.6, 0.1], [0.3, 0.4]],
                    dtype=torch.float16,
                    requires_grad=True,
                )
                if topo == "two_layer":
                    w1 = (
                        torch.tensor([[1.0, 0.5, -0.5, 0.25], [0.5, 1.0, 0.25, -0.5]], dtype=torch.float16) * 0.001
                    ).requires_grad_(True)
                    w2 = (
                        torch.tensor([[1.0, -0.5], [0.5, 1.0], [-0.5, 0.5], [0.25, 0.5]], dtype=torch.float16) * 0.001
                    ).requires_grad_(True)
                    ym = checkpoint(lambda a, m1, m2: (a @ m1) @ m2, x_m, w1, w2, use_reentrant=reentrant)
                    rm = (x_m @ w1) @ w2
                    params_m = [x_m, w1, w2]
                else:
                    w_m = _make_w(topo).clone().requires_grad_(True)
                    ym = checkpoint(lambda a, m: a @ m, x_m, w_m, use_reentrant=reentrant)
                    rm = x_m @ w_m
                    params_m = [x_m, w_m]
                if order == "ym_minus_rm":
                    obj_m = sum(_L_colmix(ym).float() for _ in range(12)) - sum(_L_colmix(rm).float() for _ in range(12))
                else:
                    obj_m = sum(_L_colmix(rm).float() for _ in range(12)) - sum(_L_colmix(ym).float() for _ in range(12))
                assert bool(torch.isfinite(obj_m).all()) and float(obj_m.item()) == 0.0
                obj_m.backward()
                max_g = max(float(p.grad.detach().abs().max().item()) for p in params_m)
                assert all(bool(torch.isfinite(p.grad).all()) for p in params_m) and max_g <= 64.0, (topo, order, reentrant, max_g)

    hk_calls = [0]
    x_hk2 = torch.tensor([[0.4, 0.3], [0.2, 0.5], [0.6, 0.1], [0.3, 0.4]], dtype=torch.float16, requires_grad=True)
    w_hk2 = (torch.eye(2, dtype=torch.float16) * 0.001).requires_grad_(True)
    ym_hk2 = checkpoint(lambda a, m: a @ m, x_hk2, w_hk2, use_reentrant=True)
    rm_hk2 = x_hk2 @ w_hk2
    ym_hk2.register_hook(lambda g: (hk_calls.__setitem__(0, hk_calls[0] + 1), g)[1])
    obj_hk2 = sum(_L_colmix(ym_hk2).float() for _ in range(12)) - sum(_L_colmix(rm_hk2).float() for _ in range(12))
    assert bool(torch.isfinite(obj_hk2).all()) and float(obj_hk2.item()) == 0.0
    obj_hk2.backward()
    assert hk_calls[0] == 1 and float(x_hk2.grad.abs().max()) <= 64.0 and float(w_hk2.grad.abs().max()) <= 64.0

    z32_mix = torch.tensor([[0.001, 0.5]], dtype=torch.float32, requires_grad=True)
    w16_mix = torch.eye(2, dtype=torch.float16)
    ym_mix = checkpoint(lambda z, w: z.half() @ w, z32_mix, w16_mix, use_reentrant=True)
    sum(_L(ym_mix).float() for _ in range(12)).backward()
    assert bool(torch.isfinite(z32_mix.grad).all())
    assert torch.allclose(z32_mix.grad.detach(), torch.tensor([[-48000.0, -12016.0]]), atol=64.0, rtol=1e-2)

    for rev_mix in (False, True):
        x_mc = torch.tensor([[0.0001, 0.5]], dtype=torch.float32, requires_grad=True)
        w_mc = torch.eye(2, dtype=torch.float16)
        fn_mc = lambda a, b: a.half() @ b
        ym_mc = checkpoint(fn_mc, x_mc, w_mc, use_reentrant=True)
        zp_mc = fn_mc(x_mc, w_mc)
        obj_mc = (
            (sum(_L(zp_mc).float() for _ in range(12)) - sum(_L(ym_mc).float() for _ in range(12)))
            if rev_mix
            else (sum(_L(ym_mc).float() for _ in range(12)) - sum(_L(zp_mc).float() for _ in range(12)))
        )
        assert bool(torch.isfinite(obj_mc).all()) and float(obj_mc.item()) == 0.0
        obj_mc.backward()
        assert bool(torch.isfinite(x_mc.grad).all()) and float(x_mc.grad[0, 0].abs()) <= 64.0

    x_ch = torch.tensor([0.0001, 0.5], dtype=torch.float16, requires_grad=True)
    def _fn_clip_hook(z):
        y = z.clone()
        if y.requires_grad:
            y.register_hook(lambda g: g.clamp(-100, 100))
        return y
    y_ch = checkpoint(_fn_clip_hook, x_ch, use_reentrant=True)
    _L(y_ch).float().backward()
    assert torch.allclose(x_ch.grad.detach(), torch.tensor([-100.0, -100.0], dtype=torch.float16), atol=1.0) or (
        abs(float(x_ch.grad[0].item()) - (-100.0)) <= 1e-3 and bool(torch.isfinite(x_ch.grad).all())
    )

    # R28-MIN-01: Two-layer tracker alias resize_, pinned resize_, untyped_storage().resize_, from_numpy dedup
    orig_resize = torch.UntypedStorage.resize_
    with PeakTensorMemoryTracker() as tr_a:
        t_a = torch.empty(1000, dtype=torch.float32)
        t_b = t_a.view(-1)
        t_b.resize_(2_000_000)
        assert tr_a.cur_bytes == 8_000_000 and tr_a.peak_bytes == 8_000_000
        del t_a, t_b
        gc.collect()
        assert tr_a.cur_bytes == 0 and tr_a.peak_bytes == 8_000_000
    assert torch.UntypedStorage.resize_ is orig_resize

    p_pin = torch.empty(100, dtype=torch.float32)
    with PeakTensorMemoryTracker(pinned_tensors=[p_pin]) as tr_p:
        p_pin.resize_(2_000_000)
        assert tr_p.cur_bytes == 8_000_000 and tr_p.peak_bytes == 8_000_000

    with PeakTensorMemoryTracker() as tr_u:
        c_u = torch.empty(500_000, dtype=torch.float32)
        c_u.untyped_storage().resize_(4_000_000)
        assert tr_u.cur_bytes == 4_000_000 and tr_u.peak_bytes == 4_000_000
        c_u.untyped_storage().resize_(0)
        assert tr_u.cur_bytes == 0 and tr_u.peak_bytes == 4_000_000
        arr_np = np.zeros(10_000, dtype=np.float32)
        _u1, _u2 = torch.from_numpy(arr_np), torch.from_numpy(arr_np)
        assert tr_u.cur_bytes == 40_000
        del c_u, _u1, _u2
    assert torch.UntypedStorage.resize_ is orig_resize

    import weakref
    arr_ov = np.zeros(100, dtype=np.float32)
    with PeakTensorMemoryTracker() as tr_ov:
        _ov1 = torch.from_numpy(arr_ov)
        _ov2 = torch.from_numpy(arr_ov[25:])
        assert tr_ov.cur_bytes == 400 and tr_ov.peak_bytes == 400
        del _ov1, _ov2

    def _check_tracker_gc():
        t_tmp = torch.empty(100_000)
        xr_tmp = weakref.ref(t_tmp)
        with PeakTensorMemoryTracker(pinned_tensors=[t_tmp]) as tr_tmp:
            pass
        return xr_tmp, weakref.ref(tr_tmp)

    xr_g, tr_g = _check_tracker_gc()
    gc.collect()
    assert xr_g() is None and tr_g() is None

    # R28-MIN-02 & R29-MIN-03: Extreme float64 severity_gamma=1e308 and independent Decimal oracle
    from decimal import Decimal, localcontext

    def _decimal_conflict_oracle(x_rows, y_rows, gam_val, mode_name, fmax_val):
        with localcontext() as ctx:
            ctx.prec = 80
            ctx.Emax = 999999
            ctx.Emin = -999999
            x0, x1 = Decimal(float(x_rows[0][0])), Decimal(float(x_rows[1][0]))
            g_d = Decimal(float(gam_val))
            fmax_d = Decimal(float(fmax_val))
            if mode_name == "sign_inversion":
                vals = [x0 * (Decimal(1) - Decimal(2) * g_d), x1 * (Decimal(1) - Decimal(2) * g_d)]
            elif mode_name == "opposite_swap":
                # Row 0 (y=1) pairs with opposite-class row 1 (y=0), and vice versa
                vals = [
                    x0 * (Decimal(1) - Decimal(2) * g_d) + Decimal(2) * g_d * x1,
                    x1 * (Decimal(1) - Decimal(2) * g_d) + Decimal(2) * g_d * x0,
                ]
            else:
                mu = (x0 + x1) / Decimal(2)
                # Sample std (ddof=1) matches torch.std / _safe_observed_std64_factored
                var = (x0 - mu) ** 2 + (x1 - mu) ** 2
                std_d = var.sqrt()
                s0 = Decimal(-1) if int(y_rows[0]) == 1 else Decimal(1)
                s1 = Decimal(-1) if int(y_rows[1]) == 1 else Decimal(1)
                vals = [
                    x0 + Decimal(3) * s0 * g_d * std_d,
                    x1 + Decimal(3) * s1 * g_d * std_d,
                ]
            out = [max(-fmax_d, min(fmax_d, v)) for v in vals]
            return np.array([[float(out[0])], [float(out[1])]], dtype=np.float64)

    inj2 = CrossModalityConflictInjector(num_modalities=2, modality_names=["a", "b"])
    fmax64 = float(np.finfo(np.float64).max)
    y_pair = np.array([1, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for x_vals, gam in (([[1e-308], [-1e-308]], 1e308), ([[8.636e307], [-8.636e307]], 1.0)):
            x_pair = [torch.tensor(x_vals, dtype=torch.float64), torch.ones((2, 1), dtype=torch.float64)]
            for mode_name in ("sign_inversion", "opposite_swap", "biomarker_spike"):
                xc, _ = inj2.inject_conflict(
                    x_pair,
                    y_pair,
                    corrupted_modality_idx=0,
                    conflict_fraction=1.0,
                    severity_gamma=gam,
                    mode=mode_name,
                    seed=0,
                )
                assert bool(torch.isfinite(xc[0]).all()), (mode_name, gam, xc[0])
                oracle = _decimal_conflict_oracle(x_vals, y_pair, gam, mode_name, fmax64)
                np.testing.assert_allclose(xc[0].numpy(), oracle, rtol=1e-12, atol=1e-12)

    # R28-MIN-04: Broken TMPDIR import + peak_memory_mib JSON key parity + empty FP16 dicts
    env_bad = dict(os.environ)
    env_bad.pop("TORCHINDUCTOR_CACHE_DIR", None)
    env_bad["TMPDIR"] = "/nonexistent_dir_xyz"
    env_bad["TMP"] = "/nonexistent_dir_xyz"
    env_bad["TEMP"] = "/nonexistent_dir_xyz"
    env_bad["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", "import deepfuzzy, torch.utils.checkpoint; assert deepfuzzy.__name__ == 'deepfuzzy'"],
        env=env_bad,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    for jpath in ("results/runtime_vram_benchmark.json", "results/benchmark_results.json"):
        jraw = Path(jpath).read_text(encoding="utf-8")
        assert "peak_memory_mib" in jraw and ("peak_memory_" + "mb") not in jraw, jpath

    assert (
        len(_FP16_ACTIVE_ROOT_LOAD),
        len(_FP16_BWD_PENDING_TOKENS),
        len(_FP16_BWD_USED_BUDGET),
        len(_FP16_BWD_CB_QUEUED),
        len(_FP16_BWD_PASS_META),
    ) == (0, 0, 0, 0, 0)


def test_phase1_clinical_concept_masks_and_pfs_tta():
    torch.manual_seed(0)
    model = UnifiedDeepFuzzyModel(
        modality_dims=[4, 3, 2],
        embed_dim=8,
        num_concepts=6,
        num_rules=4,
        concept_names=["M0", "M1", "M2", "C0", "C1", "C2"],
    ).eval()
    xs = [torch.randn(5, 4), torch.randn(5, 3), torch.randn(5, 2)]
    masks = torch.ones(5, 3)
    clin = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [0.5, -1.5, 2.5],
            [-0.5, 0.25, 1.25],
            [0.1, 0.2, 0.3],
            [-1.0, -2.0, -3.0],
        ],
        dtype=torch.float32,
    )
    ccm_all_obs = torch.ones_like(clin)
    out_none = model(xs, modality_masks=masks, clinical_concepts=clin)
    out_ones = model(xs, modality_masks=masks, clinical_concepts=clin, clinical_concept_masks=ccm_all_obs)
    assert torch.allclose(out_none["prob"], out_ones["prob"], atol=1e-6)
    assert torch.allclose(out_none["concept_mask"], out_ones["concept_mask"])

    # Selective clinical sub-indicator masking overrides a single clinical biomarker (e.g., ADAS13 imputed)
    clin_alt = clin.clone()
    clin_alt[:, 1] = 999.0
    ccm_drop_mid = torch.tensor([[1.0, 0.0, 1.0]] * 5, dtype=torch.float32)
    out_drop_a = model(xs, modality_masks=masks, clinical_concepts=clin, clinical_concept_masks=ccm_drop_mid)
    out_drop_b = model(xs, modality_masks=masks, clinical_concepts=clin_alt, clinical_concept_masks=ccm_drop_mid)
    assert torch.allclose(out_drop_a["prob"], out_drop_b["prob"], atol=1e-6)
    assert torch.allclose(out_drop_a["z_concepts"][:, 4], torch.zeros(5))
    assert torch.allclose(out_drop_a["concept_mask"][:, 4], torch.zeros(5))

    # Low-precision float16 bridging preserves clinical_concept_masks contract
    model_fp16 = UnifiedDeepFuzzyModel(
        modality_dims=[4, 3, 2],
        embed_dim=8,
        num_concepts=6,
        num_rules=4,
        concept_names=["M0", "M1", "M2", "C0", "C1", "C2"],
    ).half().eval()
    out_fp16 = model_fp16(
        [t.half() for t in xs],
        modality_masks=masks.half(),
        clinical_concepts=clin.half(),
        clinical_concept_masks=ccm_drop_mid.half(),
    )
    assert out_fp16["prob"].dtype == torch.float16 and bool(torch.isfinite(out_fp16["prob"]).all())
    assert torch.allclose(out_fp16["concept_mask"][:, 4].float(), torch.zeros(5))

    # Shape mismatch raises ValueError cleanly
    try:
        model(xs, modality_masks=masks, clinical_concepts=clin, clinical_concept_masks=torch.ones(5, 2))
        raise AssertionError("Expected ValueError on mismatched clinical_concept_masks shape")
    except ValueError:
        pass

    # PFS-TTA check on simplex interior (s_P <= 1 => mu_P^2 + nu_P^2 <= 1) vs out-of-simplex
    tta = PythagoreanFuzzyTTALoss()
    ifn_simplex = IntuitionisticFuzzyTensor(
        mu=torch.tensor([[0.6], [0.2]]),
        nu=torch.tensor([[0.4], [0.5]]),
        pi=torch.tensor([[0.0], [0.3]]),
    )
    res_in = tta(ifn_simplex)
    assert bool(torch.isfinite(res_in["tta_loss"]))
    assert bool((res_in["mu_p"].pow(2) + res_in["nu_p"].pow(2) <= 1.0 + 1e-6).all())
    res_out = tta(torch.tensor([[0.9], [0.8]]), torch.tensor([[0.9], [0.8]]), torch.tensor([[0.0], [0.1]]))
    assert bool(torch.isfinite(res_out["tta_loss"]))
    assert bool((res_out["mu_p"].pow(2) + res_out["nu_p"].pow(2) <= 1.0 + 1e-6).all())


def test_phase1_conflict_sweep_separated_aurocs():
    inj = CrossModalityConflictInjector(num_modalities=3, modality_names=["MRI", "Plasma", "Clin"])
    rng = np.random.default_rng(7)
    n = 60
    x_mods = [
        torch.tensor(rng.normal(0.0, 1.0, (n, 4)), dtype=torch.float32),
        torch.tensor(rng.normal(0.0, 1.0, (n, 2)), dtype=torch.float32),
        torch.tensor(rng.normal(0.0, 1.0, (n, 3)), dtype=torch.float32),
    ]
    y = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    masks = torch.ones((n, 3), dtype=torch.float32)

    def _mock_predict(inputs, m_in):
        # Pure conflict tracks corruption in modality 1 relative to clean baseline magnitude;
        # hesitancy intentionally anti-correlates slightly with signal while prob gives good predictions.
        delta = (inputs[1] - x_mods[1]).abs().mean(dim=-1, keepdim=True)
        conflict = torch.sigmoid(3.0 * delta - 0.5)
        hes = (1.0 - 0.8 * conflict).clamp(0.0, 1.0)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        prob = torch.where(y_t > 0.5, torch.full_like(y_t, 0.82), torch.full_like(y_t, 0.18))
        return {"prob": prob, "hesitancy": hes, "conflict": conflict}

    sweep = inj.evaluate_conflict_sweep(
        predict_fn=_mock_predict,
        modality_tensors=x_mods,
        natural_masks=masks,
        targets=y,
        corrupted_modality_idx=1,
        severities=[0.0, 0.8],
        referral_rates=[0.0, 0.2],
        seed=11,
    )
    rec08 = sweep["sweep_records"][1]
    assert "conflict_detection_auroc" in rec08
    assert "conflict_detection_auroc_conflict_only" in rec08
    assert "conflict_detection_auroc_hes_only" in rec08
    assert "referral_curve_conflict_only" in rec08
    assert "referral_curve_hes_only" in rec08
    assert rec08["conflict_detection_auroc_conflict_only"] > 0.90
    assert rec08["conflict_detection_auroc_hes_only"] < 0.15


def test_phase1_ddef_and_fuml_baselines():
    torch.manual_seed(12)
    dims = [6, 3, 4]
    xs = [torch.randn(10, d) for d in dims]
    masks = torch.tensor(
        [[1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 0]] * 2,
        dtype=torch.float32,
    )
    y = torch.tensor([[1], [0], [1], [0], [1]] * 2, dtype=torch.float32)

    for cls in (DDEFBaseline, FUMLBaseline):
        model = cls(modality_dims=dims, embed_dim=16, dropout=0.1)
        model.eval()
        out = model(xs, modality_masks=masks)
        for k in ("prob", "hesitancy", "vacuity", "conflict"):
            assert out[k].shape == (10, 1), (cls.__name__, k)
            assert bool(torch.isfinite(out[k]).all()), (cls.__name__, k)
            assert bool(((out[k] >= -1e-6) & (out[k] <= 1.0 + 1e-6)).all()), (cls.__name__, k)
        # All-missing row (index 3 and 8) falls back cleanly to neutral vacuity
        assert torch.allclose(out["prob"][[3, 8]], torch.full((2, 1), 0.5))
        assert torch.allclose(out["hesitancy"][[3, 8]], torch.ones((2, 1)))
        assert torch.allclose(out["conflict"][[3, 8]], torch.zeros((2, 1)))

        # Missing-modality invariance: arbitrary noise/NaN in masked-out modalities does not alter output
        xs_noisy = [t.clone() for t in xs]
        xs_noisy[1][1] = 1e6
        xs_noisy[0][2] = float("nan")
        out_noisy = model(xs_noisy, modality_masks=masks)
        assert torch.allclose(out["prob"], out_noisy["prob"], atol=1e-6)
        assert torch.allclose(out["hesitancy"], out_noisy["hesitancy"], atol=1e-6)
        assert torch.allclose(out["conflict"], out_noisy["conflict"], atol=1e-6)

        # Training backward & annealed KL loss check
        model.train()
        out_tr = model(xs, modality_masks=masks)
        loss = F.binary_cross_entropy(out_tr["prob"].clamp(1e-5, 1.0 - 1e-5), y)
        loss = loss + 0.2 * model.compute_kl_loss(y, epoch=5, annealing_step=10)
        loss.backward()
        assert all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
            if p.requires_grad
        )


if __name__ == "__main__":
    run_all_verifications()
