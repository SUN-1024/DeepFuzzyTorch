"""
DeepFuzzyTorch: A Unified Differentiable High-Order Fuzzy & Granular Computing Library for PyTorch.

Designed as a foundational operator engine and open-source benchmark suite for high-impact
multimodal medical AI, computational pathology, and multi-omics integration:
  1. IntuitionisticFuzzyEncoder & IFSChoquetFusion (Atanassov IFS + Choquet Integral)
  2. HesitantFuzzyAggregator (Torra HFS + Xu & Xia, 2012 / Liao & Xu, 2013 Hesitant Fuzzy Dispersion)
  3. DifferentiableFuzzyRoughLayer & FuzzyRoughFeatureReducer (Fuzzy-Rough Approximations & Reduct)
  4. DeepConceptTSKHead (White-Box Takagi-Sugeno-Kang Fuzzy Rule Bottleneck)
  5. PythagoreanFuzzyTTALoss (Pythagorean Fuzzy Divergence for Test-Time Adaptation)
  6. UnifiedDeepFuzzyModel & IntuitionisticFuzzyLoss (End-to-End Multimodal Architecture)
  7. CombinatorialMissingModalityBenchmark (2^M - 1 Exhaustive Missing-Modality Evaluator)
  8. CrossModalityConflictInjector (Conflict Injection & Selective Referral Safety Suite)
  9. SOTA Multimodal & Uncertainty Baselines (EarlyConcatMLP, LateAvgEnsemble,
     GatedAttentionFusion, DempsterShaferFusion, EvidentialDL/EvidentialDLBaseline,
     DeepTSKBaseline, QMFBaseline (adapted), RCMLBaseline, DrFuseBaseline)
"""

from __future__ import annotations

import os
import tempfile

if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
    try:
        tempfile.gettempdir()
    except FileNotFoundError:
        for _cand_dir in (
            "/tmp",
            "/private/tmp",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ):
            if os.path.isdir(_cand_dir) and os.access(_cand_dir, os.W_OK):
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cand_dir
                break

from .operators import (
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
)
from .baselines import (
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
)
from .metrics import (
    compute_ece,
    compute_aurc,
    compute_binary_metrics,
    bootstrap_confidence_interval,
    paired_bootstrap_pvalue,
    compute_decision_curve_analysis,
    compute_logrank_test,
    compute_multigroup_logrank_test,
)
from .missing_benchmark import CombinatorialMissingModalityBenchmark
from .conflict_injector import CrossModalityConflictInjector
from .rough_reducer import FuzzyRoughFeatureReducer

__version__ = "0.2.4"
__all__ = [
    "IntuitionisticFuzzyTensor",
    "IntuitionisticFuzzyEncoder",
    "IFSChoquetFusion",
    "HesitantFuzzyAggregator",
    "DifferentiableFuzzyRoughLayer",
    "FuzzyRoughAttentionGate",
    "DeepConceptTSKHead",
    "PythagoreanFuzzyTTALoss",
    "IntuitionisticFuzzyLoss",
    "UnifiedDeepFuzzyModel",
    "EarlyConcatMLP",
    "LateAvgEnsemble",
    "GatedAttentionFusion",
    "DempsterShaferFusion",
    "EvidentialDL",
    "EvidentialDLBaseline",
    "DeepTSKBaseline",
    "QMFBaseline",
    "RCMLBaseline",
    "DrFuseBaseline",
    "DDEFBaseline",
    "FUMLBaseline",
    "compute_ece",
    "compute_aurc",
    "compute_binary_metrics",
    "bootstrap_confidence_interval",
    "paired_bootstrap_pvalue",
    "compute_decision_curve_analysis",
    "compute_logrank_test",
    "compute_multigroup_logrank_test",
    "CombinatorialMissingModalityBenchmark",
    "CrossModalityConflictInjector",
    "FuzzyRoughFeatureReducer",
]
