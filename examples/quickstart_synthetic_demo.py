"""
Runnable synthetic quickstart demonstrating all five DeepFuzzyTorch operator blocks:
  1. Missing-modality vacuity masking and concept-masked TSK invariance (Theorem 2 & Corollary 2.1)
  2. Cross-modal conflict discounting and hesitancy inflation (Theorems 3 & 4)
  3. Differentiable Hesitant Fuzzy multi-expert aggregation + Pythagorean TTA (Block 2 & Block 5)
  4. Pointwise T-norm fuzzy-rough feature reduction (Theorem 5 Part II)
  5. Auditable 8-decimal white-box TSK rule export and text reconstruction
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from deepfuzzy import (
    DDEFBaseline,
    FUMLBaseline,
    FuzzyRoughFeatureReducer,
    HesitantFuzzyAggregator,
    PythagoreanFuzzyTTALoss,
    UnifiedDeepFuzzyModel,
)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    concept_names = [
        "MRI_NetSupport",
        "Plasma_NetSupport",
        "FDG_NetSupport",
        "Clin_NetSupport",
        "APOE4",
        "MMSE",
        "ADAS13",
        "CDRSB",
    ]
    model = UnifiedDeepFuzzyModel(
        modality_dims=[24, 4, 1, 17],
        embed_dim=40,
        num_classes=1,
        conflict_penalty=0.65,
        use_rough_bottleneck=True,
        use_tsk_head=True,
        num_tsk_concepts=8,
        num_tsk_rules=6,
        concept_names=concept_names,
    )
    model.eval()

    bsz = 16
    x_mri = torch.randn(bsz, 24)
    x_plasma = torch.randn(bsz, 4)
    x_fdg = torch.randn(bsz, 1)
    x_clin = torch.randn(bsz, 17)
    modality_masks = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]] * (bsz // 2), dtype=torch.float32)
    clin_masks = torch.ones(bsz, 4, dtype=torch.float32)
    clin_masks[0, 2] = 0.0  # Simulate an unobserved ADAS13 clinical concept on sample 0

    with torch.no_grad():
        out_a = model(
            [x_mri, x_plasma, x_fdg, x_clin],
            modality_masks=modality_masks,
            clinical_concepts=x_clin[:, :4],
            clinical_concept_masks=clin_masks,
        )
        # Replace unacquired modality inputs and masked clinical_concepts slot with NaNs/sentinels
        x_fdg_alt = x_fdg.clone()
        x_fdg_alt[modality_masks[:, 2] < 0.5] = float("nan")
        x_plasma_alt = x_plasma.clone()
        x_plasma_alt[modality_masks[:, 1] < 0.5] = 1e6
        clin_concepts_alt = x_clin[:, :4].clone()
        clin_concepts_alt[0, 2] = float("nan")
        out_b = model(
            [x_mri, x_plasma_alt, x_fdg_alt, x_clin],
            modality_masks=modality_masks,
            clinical_concepts=clin_concepts_alt,
            clinical_concept_masks=clin_masks,
        )

    pad_diff = float((out_a["prob"] - out_b["prob"]).abs().max().item())
    assert out_a["fused_ifn"].validate(), "Simplex closure check failed"
    assert pad_diff == 0.0, f"Expected exact 0.00 padding invariance, got {pad_diff}"

    # Rule export & text reconstruction
    rules = model.tsk_head.export_clinical_rules(gate_threshold=0.0, precision=8)
    recon = model.tsk_head.evaluate_exported_rules_text(
        rules,
        out_a["z_concepts"],
        concept_names=concept_names,
        concept_mask=out_a["concept_mask"],
    )
    rule_err = float(np.max(np.abs(out_a["tsk_logits"].detach().cpu().numpy().ravel() - recon)))

    # Fuzzy-rough feature selection + HFS/PFS + DDEF/FUML smoke check
    X_syn = np.random.randn(80, 12).astype(np.float32)
    y_syn = (X_syn[:, 0] + 0.6 * X_syn[:, 1] > 0).astype(np.float32)
    reducer = FuzzyRoughFeatureReducer(max_features=4, n_null_permutations=4, n_stability_bootstraps=2, seed=42)
    reducer.fit(X_syn, y_syn, device="cpu")
    assert all(
        reducer.dependency_curve_[i] <= reducer.dependency_curve_[i + 1] + 1e-6
        for i in range(len(reducer.dependency_curve_) - 1)
    )

    hfs = HesitantFuzzyAggregator(num_sources=4)
    pfs_tta = PythagoreanFuzzyTTALoss()
    hfs_out = hfs(torch.rand(bsz, 4))
    tta_out = pfs_tta(hfs_out["ifn"])

    ddef = DDEFBaseline([24, 4, 1, 17], embed_dim=32)
    fuml = FUMLBaseline([24, 4, 1, 17], embed_dim=32)
    assert ddef([x_mri, x_plasma, x_fdg, x_clin], modality_masks)["prob"].shape == (bsz, 1)
    assert fuml([x_mri, x_plasma, x_fdg, x_clin], modality_masks)["prob"].shape == (bsz, 1)

    print(
        f"[Quickstart OK] SimplexValid=True | MaxPaddingDiff={pad_diff:.2e} | "
        f"Rules={len(rules)} | TextLogitErr={rule_err:.2e} | "
        f"TopReduct={reducer.selected_indices_} | PFS_TTA={float(tta_out['tta_loss'].detach().item()):.4f}"
    )


if __name__ == "__main__":
    main()
