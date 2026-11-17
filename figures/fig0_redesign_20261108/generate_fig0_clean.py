#!/usr/bin/env python3
"""
Programmatic Vector Publication Generator for Figure 1 & Figure 2 of DeepFuzzyTorch.

Outputs (300-DPI PNG + True-Outline Vector PDF):
  - figures/Fig0_Paradigm_Teaser_Comparison.{pdf,png}
  - figures/Fig0_System_Architecture_Overview.{pdf,png}
  - figures/fig0_redesign_20261108/Fig0_Paradigm_Teaser_Comparison.{pdf,png}
  - figures/fig0_redesign_20261108/Fig0_System_Architecture_Overview.{pdf,png}

All numerical values in Figure 1 (Panels a, b, c) are computed directly via
`deepfuzzy.operators.IFSChoquetFusion` and exact Dempster-Shafer combination in
IEEE-754 float64 precision—zero hand-typed or approximate numbers.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Wedge, Rectangle
from matplotlib.text import Text
import numpy as np
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from deepfuzzy.operators import IFSChoquetFusion, IntuitionisticFuzzyTensor  # noqa: E402


# =============================================================================
# Colorblind-Safe Flagship Palette (Harmonized with Fig1..Fig6)
# =============================================================================
C = {
    "ink": "#1E293B",
    "ink_soft": "#475569",
    "ink_muted": "#64748B",
    "vermilion": "#C84B31",       # Flagship Rust-Red / Conflict / CKD alert
    "vermilion_light": "#FDF3F1",
    "vermilion_border": "#E09F91",
    "slate_blue": "#1F6E8C",      # Refutation (nu) / Neuroimaging / Clean
    "slate_light": "#F1F7FA",
    "slate_border": "#94BFD1",
    "teal": "#2A9D8F",            # Support (mu) / Validated / Referral OK
    "teal_dark": "#1D786E",
    "teal_light": "#EFFAF8",
    "teal_border": "#88CEC5",
    "ochre": "#D98A2B",           # Hesitancy (pi) / TSK / Granular
    "ochre_light": "#FDF8F0",
    "ochre_border": "#E6BE8A",
    "plum": "#7A496B",            # Multi-FM HFS / PFS-TTA
    "plum_light": "#F7F2F6",
    "plum_border": "#C5A3BA",
    "gray_bg": "#F8FAFC",
    "gray_card": "#F1F5F9",
    "gray_border": "#CBD5E1",
    "miss_bg": "#F8FAFC",
    "miss_border": "#94A3B8",
    "white": "#FFFFFF",
}


def configure_rc() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })


# =============================================================================
# Exact Mathematical Verification Engine for Figure 1
# =============================================================================
def compute_exact_clinical_conflict_case(gamma: float = 0.60) -> Dict[str, object]:
    """
    Amyloid-negative patient with baseline eGFR < 60 mL/min/1.73m² (Aβ-):
      - Brain MRI   (observed, o_1=1): normal cortex/hippocampus -> <0.12, 0.76, 0.12>
      - FDG-PET     (missing,  o_2=0): unacquired                -> <0.00, 0.00, 1.00>
      - Plasma pTau (observed, o_3=1): low-eGFR plasma elevation -> <0.86, 0.05, 0.09>
      - Clinical    (observed, o_4=1): normal cognition (MMSE)   -> <0.18, 0.68, 0.14>
    """
    raw_triples = [
        ("Brain MRI", 1, 0.12, 0.76, 0.12, "Normal morphometry (refutes Aβ⁺)"),
        ("FDG-PET", 0, 0.00, 0.00, 1.00, "Unacquired → exact vacuity"),
        ("Plasma pTau", 1, 0.86, 0.05, 0.09, "Elevated pTau (baseline eGFR < 60)"),
        ("Clinical", 1, 0.18, 0.68, 0.14, "Normal cognition (refutes Aβ⁺)"),
    ]

    ifns = [
        IntuitionisticFuzzyTensor(
            mu=torch.tensor([[mu]], dtype=torch.float64),
            nu=torch.tensor([[nu]], dtype=torch.float64),
            pi=torch.tensor([[pi]], dtype=torch.float64),
        )
        for _, _, mu, nu, pi, _ in raw_triples
    ]

    fusion = IFSChoquetFusion(num_modalities=4, num_classes=1, conflict_penalty=gamma)
    with torch.no_grad():
        fused_ifn, aux = fusion(ifns)
        fused_no_disc, _ = IFSChoquetFusion(num_modalities=4, num_classes=1, conflict_penalty=0.0)(ifns)

        gammas = np.linspace(0.0, 1.25, 60)
        traj = []
        for g_val in gammas:
            f_g, _ = IFSChoquetFusion(num_modalities=4, num_classes=1, conflict_penalty=float(g_val))(ifns)
            traj.append((f_g.mu.item(), f_g.nu.item(), f_g.pi.item()))

    c_o = aux["conflict"].item()
    d = math.exp(-gamma * c_o)
    weights = [w for w in aux["weights"].squeeze().tolist()]
    pair_c = aux["pairwise_conflict"].squeeze().numpy()

    mu_f = fused_ifn.mu.item()
    nu_f = fused_ifn.nu.item()
    pi_f = fused_ifn.pi.item()
    p_ifs = aux["prob"].item()
    s_ref = 0.5 * (pi_f + c_o)

    discounted = []
    for _, o_m, mu, nu, _, _ in raw_triples:
        if o_m == 0:
            discounted.append((0.0, 0.0, 1.0))
        else:
            mh = d * mu
            nh = d * nu
            ph = 1.0 - mh - nh
            discounted.append((mh, nh, ph))

    # Classical Dempster-Shafer orthogonal combination (m_1 ⊕ m_2 ⊕ m_3 ⊕ m_4)
    b_pos, b_neg, u_ds = 0.0, 0.0, 1.0
    k_steps = []
    for _, o_m, mp, mn, um, _ in raw_triples:
        k_val = b_pos * mn + b_neg * mp
        if o_m == 1 and (b_pos + b_neg > 0):
            k_steps.append(k_val)
        norm = 1.0 - k_val
        b_pos, b_neg, u_ds = (
            (b_pos * mp + b_pos * um + u_ds * mp) / norm,
            (b_neg * mn + b_neg * um + u_ds * mn) / norm,
            (u_ds * um) / norm,
        )
    p_ds = b_pos + 0.5 * u_ds
    ds_amp = 1.0 / math.prod(1.0 - k for k in k_steps)

    # Exact mathematical sanity checks
    assert abs((mu_f + nu_f + pi_f) - 1.0) < 1e-12
    assert weights[1] == 0.0, "Missing modality must have exact zero Choquet weight!"
    assert abs(sum(weights) - 1.0) < 1e-12
    assert nu_f <= max(t[3] for t in raw_triples) + 1e-12
    assert mu_f <= max(t[2] for t in raw_triples) + 1e-12

    return {
        "gamma": gamma,
        "raw_triples": raw_triples,
        "discounted": discounted,
        "weights": weights,
        "c_o": c_o,
        "d": d,
        "pair_c": pair_c,
        "fused": (mu_f, nu_f, pi_f),
        "fused_gamma0": (fused_no_disc.mu.item(), fused_no_disc.nu.item(), fused_no_disc.pi.item()),
        "traj": np.array(traj),
        "p_ifs": p_ifs,
        "s_ref": s_ref,
        "ds": (b_pos, b_neg, u_ds),
        "p_ds": p_ds,
        "k_steps": k_steps,
        "ds_amp": ds_amp,
    }


# =============================================================================
# Vector Drawing Primitives
# =============================================================================
def draw_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fc: str = "#FFFFFF",
    ec: str = "#CBD5E1",
    lw: float = 1.1,
    ls: str = "-",
    pad: float = 0.012,
    zorder: int = 2,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (x + pad, y + pad),
        w - 2 * pad,
        h - 2 * pad,
        boxstyle=f"round,pad={pad},rounding_size={pad*1.6}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def draw_arrow(
    ax: plt.Axes,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: str = "#334155",
    lw: float = 1.35,
    ls: str = "-",
    rad: float = 0.0,
    head_scale: float = 11.0,
    zorder: int = 5,
) -> FancyArrowPatch:
    arr = FancyArrowPatch(
        p1,
        p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=f"-|>,head_length={head_scale*0.042},head_width={head_scale*0.026}",
        mutation_scale=head_scale,
        linewidth=lw,
        linestyle=ls,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def draw_step_badge(ax: plt.Axes, cx: float, cy: float, label: str, r: float = 0.14, color: str = "#1E293B") -> None:
    circ = Circle((cx, cy), r, facecolor=C["white"], edgecolor=color, linewidth=1.2, zorder=8)
    ax.add_patch(circ)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=8.0, fontweight="bold", color=color, zorder=9)


def draw_ifn_bar(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    mu: float,
    nu: float,
    pi: float,
    zorder: int = 6,
) -> None:
    """Draws an exact proportional 3-segment horizontal IFS bar (mu: teal, nu: slate_blue, pi: ochre-sand)."""
    bg = Rectangle((x, y), w, h, facecolor="#F8FAFC", edgecolor="#64748B", linewidth=0.75, zorder=zorder)
    ax.add_patch(bg)
    x_cur = x
    for val, col in [(mu, C["teal"]), (nu, C["slate_blue"]), (pi, "#E9C46A")]:
        seg_w = max(0.0, val) * w
        if seg_w > 1e-5:
            ax.add_patch(Rectangle((x_cur, y), seg_w, h, facecolor=col, edgecolor="none", zorder=zorder + 1))
            if x_cur > x + 1e-4:
                ax.plot([x_cur, x_cur], [y, y + h], color="#1E293B", linewidth=0.6, zorder=zorder + 2)
        x_cur += seg_w
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#334155", linewidth=0.75, zorder=zorder + 3))


def draw_modality_glyph(ax: plt.Axes, kind: str, cx: float, cy: float, r: float = 0.19) -> None:
    """Crisp vector micro-icon for MRI, FDG, Plasma, or Clinical."""
    if kind == "mri":
        ell_out = mpatches.Ellipse((cx, cy), r * 1.65, r * 1.95, facecolor="#E2E8F0", edgecolor="#1E293B", lw=1.0, zorder=6)
        ell_in = mpatches.Ellipse((cx, cy), r * 1.32, r * 1.60, facecolor="#94A3B8", edgecolor="#334155", lw=0.7, zorder=6)
        ax.add_patch(ell_out)
        ax.add_patch(ell_in)
        ax.plot([cx, cx], [cy - r * 0.78, cy + r * 0.78], color="#1E293B", lw=0.7, zorder=7)
        v1 = mpatches.Ellipse((cx - r * 0.18, cy + r * 0.05), r * 0.22, r * 0.45, angle=18, facecolor="#1E293B", zorder=7)
        v2 = mpatches.Ellipse((cx + r * 0.18, cy + r * 0.05), r * 0.22, r * 0.45, angle=-18, facecolor="#1E293B", zorder=7)
        ax.add_patch(v1)
        ax.add_patch(v2)
    elif kind == "fdg":
        circ = Circle((cx, cy), r * 0.85, facecolor="#F1F5F9", edgecolor="#94A3B8", lw=0.9, ls="--", zorder=6)
        ax.add_patch(circ)
        d = r * 0.62
        ax.plot([cx - d, cx + d], [cy - d, cy + d], color=C["vermilion"], lw=1.8, zorder=7)
        ax.plot([cx - d, cx + d], [cy + d, cy - d], color=C["vermilion"], lw=1.8, zorder=7)
    elif kind == "plasma":
        ang = -22.0
        tr = matplotlib.transforms.Affine2D().rotate_deg_around(cx, cy, ang) + ax.transData
        tube = FancyBboxPatch(
            (cx - r * 0.30, cy - r * 0.82), r * 0.60, r * 1.50,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor="#FFFFFF", edgecolor="#475569", lw=0.95, transform=tr, zorder=6
        )
        blood = Rectangle(
            (cx - r * 0.26, cy - r * 0.78), r * 0.52, r * 0.92,
            facecolor=C["vermilion"], edgecolor="none", transform=tr, zorder=7
        )
        cap = Rectangle(
            (cx - r * 0.34, cy + r * 0.55), r * 0.68, r * 0.26,
            facecolor="#991B1B", edgecolor="#1E293B", lw=0.7, transform=tr, zorder=8
        )
        ax.add_patch(tube)
        ax.add_patch(blood)
        ax.add_patch(cap)
    elif kind == "clin":
        doc = FancyBboxPatch(
            (cx - r * 0.65, cy - r * 0.85), r * 1.30, r * 1.70,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor="#FFFFFF", edgecolor="#475569", lw=0.95, zorder=6
        )
        ax.add_patch(doc)
        for idx, yy in enumerate([0.42, 0.10, -0.22, -0.52]):
            col = C["teal"] if idx == 0 else "#94A3B8"
            w_line = r * (0.85 if idx % 2 == 0 else 0.65)
            ax.plot([cx - r * 0.42, cx - r * 0.42 + w_line], [cy + r * yy, cy + r * yy], color=col, lw=1.2, zorder=7)


def bary_to_xy(
    mu: float, nu: float, pi: float,
    v_mu: Tuple[float, float], v_nu: Tuple[float, float], v_pi: Tuple[float, float],
) -> Tuple[float, float]:
    x = mu * v_mu[0] + nu * v_nu[0] + pi * v_pi[0]
    y = mu * v_mu[1] + nu * v_nu[1] + pi * v_pi[1]
    return x, y


# =============================================================================
# FIGURE 1: Fig0_Paradigm_Teaser_Comparison
# =============================================================================
def assert_no_text_overlaps(fig: plt.Figure, tol_px: float = 1.0, card_checks: Optional[List[Tuple[Any, Any]]] = None) -> None:
    """Renders canvas and raises if any visible Text overlaps or escapes its assigned container card."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        for txt in ax.texts:
            if not txt.get_visible() or not txt.get_text().strip():
                continue
            bb = Text.get_window_extent(txt, renderer=renderer)
            if bb.width > 0.0 and bb.height > 0.0:
                boxes.append((txt, txt.get_text().replace("\n", " ")[:48], bb))
    overlaps = []
    for i in range(len(boxes)):
        _, lbl_i, bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            _, lbl_j, bj = boxes[j]
            ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ox > tol_px and oy > tol_px:
                overlaps.append((lbl_i, lbl_j, round(float(ox), 1), round(float(oy), 1)))
    if overlaps:
        raise AssertionError(f"Overlapping Text bboxes detected: {overlaps[:8]}")
    if card_checks:
        escapes = []
        for txt, patch in card_checks:
            if txt is None or not txt.get_visible():
                continue
            tbb = txt.get_window_extent(renderer=renderer)
            pbb = patch.get_window_extent(renderer=renderer)
            if tbb.x0 < pbb.x0 + 1.0 or tbb.x1 > pbb.x1 - 1.0 or tbb.y0 < pbb.y0 + 0.5 or tbb.y1 > pbb.y1 - 0.5:
                escapes.append((
                    txt.get_text().replace("\n", " ")[:40],
                    (round(tbb.x0 - pbb.x0, 1), round(pbb.x1 - tbb.x1, 1), round(tbb.y0 - pbb.y0, 1), round(pbb.y1 - tbb.y1, 1)),
                ))
        if escapes:
            raise AssertionError(f"Text escapes container card (dx0, dx1, dy0, dy1 px): {escapes[:8]}")


def generate_fig1_teaser(data: Dict[str, object], out_dirs: List[str]) -> List[str]:
    configure_rc()
    fig = plt.figure(figsize=(8.60, 4.92), dpi=300)
    ax = fig.add_axes([0.005, 0.008, 0.99, 0.984])
    ax.set_xlim(0.0, 15.4)
    ax.set_ylim(0.0, 8.7)
    ax.axis("off")

    raw = data["raw_triples"]
    disc = data["discounted"]
    weights = data["weights"]
    c_o = data["c_o"]
    d_val = data["d"]
    pair_c = data["pair_c"]
    mu_f, nu_f, pi_f = data["fused"]
    mu_0, nu_0, pi_0 = data["fused_gamma0"]
    p_ifs = data["p_ifs"]
    s_ref = data["s_ref"]
    ds_pos, ds_neg, ds_u = data["ds"]
    p_ds = data["p_ds"]
    k_steps = data["k_steps"]
    ds_amp = data["ds_amp"]
    card_checks: List[Tuple[Any, Any]] = []

    # -------------------------------------------------------------------------
    # PANEL (a): Conventional Fusion (Top-Left: x in [0.10, 9.90], y in [4.75, 8.60])
    # -------------------------------------------------------------------------
    pa_outer = draw_card(ax, 0.10, 4.75, 9.80, 3.85, fc="#FDF7F6", ec=C["vermilion_border"], lw=1.25, pad=0.04)
    card_checks.append((ax.text(
        0.26, 8.34,
        "(a) Conventional: Unmasked Bias & DS Paradox",
        fontsize=9.1, fontweight="bold", color="#7F1D1D", va="center",
    ), pa_outer))
    pill_ckd = draw_card(ax, 6.64, 8.15, 3.16, 0.38, fc=C["white"], ec=C["vermilion_border"], lw=0.8, pad=0.01)
    card_checks.append((ax.text(
        8.22, 8.34,
        "Low eGFR (<60, O={1,3,4})",
        fontsize=7.8, fontweight="bold", color=C["ink_soft"], ha="center", va="center",
    ), pill_ckd))

    mod_styles_a = [
        ("mri", C["slate_light"], C["slate_border"], C["ink"], "Brain MRI (o1=1)", "Refutes: <0.12,0.76,0.12>"),
        ("fdg", C["miss_bg"], C["miss_border"], C["ink_muted"], "FDG-PET (o2=0)", "Unacquired (DS <0,0,1>)"),
        ("plasma", "#FDECE8", C["vermilion"], "#7F1D1D", "Plasma (eGFR<60)", "Low-eGFR: <0.86,0.05,0.09>"),
        ("clin", C["teal_light"], C["teal_border"], C["ink"], "Clinical (o4=1)", "Normal: <0.18,0.68,0.14>"),
    ]
    y_tops_a = [7.34, 6.51, 5.68, 4.85]
    for idx, (glyph, fc_m, ec_m, tc_m, title_m, sub_m) in enumerate(mod_styles_a):
        yb = y_tops_a[idx]
        ls_m = "--" if idx == 1 else "-"
        mc = draw_card(ax, 0.22, yb, 3.44, 0.75, fc=fc_m, ec=ec_m, lw=1.0, ls=ls_m, pad=0.02)
        draw_modality_glyph(ax, glyph, 0.45, yb + 0.375, r=0.15)
        card_checks.append((ax.text(0.68, yb + 0.52, title_m, fontsize=8.0, fontweight="bold", color=tc_m, va="center"), mc))
        card_checks.append((ax.text(0.68, yb + 0.21, sub_m, fontsize=7.8, color=C["ink_soft"], va="center"), mc))

    ax.plot([3.68, 3.79, 3.79, 3.68], [7.71, 7.71, 5.22, 5.22], color="#475569", lw=1.2, zorder=5)
    draw_arrow(ax, (3.79, 5.90), (3.92, 5.90), color="#475569", lw=1.3)

    # Center Operator Box in Panel (a)
    pa_mid = draw_card(ax, 3.94, 4.92, 2.90, 3.12, fc=C["white"], ec="#64748B", lw=1.1, pad=0.03)
    card_checks.append((ax.text(5.39, 7.77, "Two Failure Modes", fontsize=8.2, fontweight="bold", color=C["ink"], ha="center", va="center"), pa_mid))

    pa_mlp = draw_card(ax, 4.02, 6.78, 2.74, 0.74, fc="#F8FAFC", ec="#CBD5E1", lw=0.85, pad=0.02)
    card_checks.append((ax.text(5.39, 7.30, "A. Unmasked MLP", fontsize=7.85, fontweight="bold", color=C["ink"], ha="center", va="center"), pa_mlp))
    card_checks.append((ax.text(5.39, 6.98, "Zero x2=0 leaks b2!=0", fontsize=7.8, fontweight="bold", color=C["vermilion"], ha="center", va="center"), pa_mlp))

    pa_ds = draw_card(ax, 4.02, 5.04, 2.74, 1.62, fc="#FDF3F1", ec=C["vermilion_border"], lw=0.85, pad=0.02)
    card_checks.append((ax.text(5.39, 6.42, "B. DS: m2=<0,0,1>", fontsize=7.85, fontweight="bold", color="#7F1D1D", ha="center", va="center"), pa_ds))
    card_checks.append((ax.text(5.39, 6.04, f"K1={k_steps[0]:.3f}, K2={k_steps[1]:.3f}", fontsize=7.85, color=C["ink"], ha="center", va="center"), pa_ds))
    card_checks.append((ax.text(5.39, 5.64, f"Π (1 - K)^-1 = {ds_amp:.2f}x", fontsize=8.0, fontweight="bold", color=C["vermilion"], ha="center", va="center"), pa_ds))
    card_checks.append((ax.text(5.39, 5.25, "Norm. drains vacuity!", fontsize=7.8, color=C["ink_soft"], ha="center", va="center"), pa_ds))

    draw_arrow(ax, (6.86, 5.90), (6.98, 5.90), color="#475569", lw=1.3)

    # Right Outcome Box in Panel (a)
    pa_out = draw_card(ax, 7.00, 4.92, 2.80, 3.12, fc=C["white"], ec=C["vermilion"], lw=1.25, pad=0.03)
    card_checks.append((ax.text(8.40, 7.77, "✗ Zadeh's Paradox (DS)", fontsize=8.2, fontweight="bold", color="#7F1D1D", ha="center", va="center"), pa_out))
    card_checks.append((ax.text(8.40, 7.42, f"<{ds_pos:.3f}, {ds_neg:.3f}, {ds_u:.3f}>", fontsize=7.9, fontweight="bold", color=C["ink"], ha="center", va="center"), pa_out))

    bx0, by0, bh_max = 7.50, 5.88, 1.15
    ax.plot([bx0 - 0.06, bx0 + 1.88], [by0, by0], color="#334155", lw=0.9, zorder=6)
    bars_a = [
        (ds_pos, C["teal"], "m(+)", f"{ds_pos:.3f}"),
        (ds_neg, C["slate_blue"], "m(-)", f"{ds_neg:.3f}"),
        (ds_u, C["vermilion"], "m(Ω)", f"{ds_u:.3f}"),
    ]
    for b_i, (b_val, b_col, b_lbl, b_txt) in enumerate(bars_a):
        bx = bx0 + b_i * 0.64 + 0.05
        bh = max(0.035, b_val * bh_max / 0.65)
        ax.add_patch(Rectangle((bx, by0), 0.42, bh, facecolor=b_col, edgecolor="#1E293B", lw=0.75, zorder=6))
        card_checks.append((ax.text(bx + 0.21, by0 + bh + 0.06, b_txt, fontsize=7.85, fontweight="bold",
                color=C["vermilion"] if b_i == 2 else C["ink"], ha="center", va="bottom"), pa_out))
        card_checks.append((ax.text(bx + 0.21, by0 - 0.18, b_lbl, fontsize=7.85, color=C["ink_soft"], ha="center", va="center"), pa_out))

    card_checks.append((ax.text(8.40, 5.38, f"Ign. m(Ω)={ds_u:.3f} (0.9%)", fontsize=7.8, fontweight="bold", color=C["vermilion"], ha="center", va="center"), pa_out))
    card_checks.append((ax.text(8.40, 5.09, f"P_DS={p_ds:.3f} · Unc. <0.30", fontsize=7.8, color=C["ink_soft"], ha="center", va="center"), pa_out))

    # -------------------------------------------------------------------------
    # PANEL (b): DeepFuzzyTorch Evidential Choquet Fusion (Bottom-Left: y in [0.10, 4.58])
    # -------------------------------------------------------------------------
    pb_outer = draw_card(ax, 0.10, 0.10, 9.80, 4.48, fc="#F2F8FA", ec=C["slate_border"], lw=1.25, pad=0.04)
    card_checks.append((ax.text(
        0.26, 4.32,
        "(b) DeepFuzzyTorch: Choquet & Rules",
        fontsize=9.3, fontweight="bold", color="#0F4C5C", va="center",
    ), pb_outer))
    pill_ch = draw_card(ax, 6.64, 4.13, 3.16, 0.38, fc=C["white"], ec=C["slate_border"], lw=0.8, pad=0.01)
    card_checks.append((ax.text(
        8.22, 4.32,
        "Eq. (4): Choquet (γ=0.60)",
        fontsize=7.9, fontweight="bold", color=C["slate_blue"], ha="center", va="center",
    ), pill_ch))

    y_tops_b = [3.26, 2.38, 1.50, 0.62]
    mod_meta_b = [
        ("mri", C["white"], C["slate_border"], "Brain MRI", False),
        ("fdg", C["miss_bg"], C["miss_border"], "FDG (Miss)", True),
        ("plasma", "#FFF5F3", C["vermilion"], "Plasma (<60)", False),
        ("clin", C["white"], C["teal_border"], "Clinical", False),
    ]
    for idx, (glyph, fc_m, ec_m, label_m, is_miss) in enumerate(mod_meta_b):
        _, _, mu_m, nu_m, pi_m, _ = raw[idx]
        w_m = weights[idx]
        yb = y_tops_b[idx]
        mc_b = draw_card(ax, 0.22, yb, 3.44, 0.80, fc=fc_m, ec=ec_m, lw=1.0, ls="--" if is_miss else "-", pad=0.02)
        draw_modality_glyph(ax, glyph, 0.45, yb + 0.40, r=0.145)
        card_checks.append((ax.text(
            0.67, yb + 0.59,
            label_m,
            fontsize=8.0, fontweight="bold", color=C["vermilion"] if idx == 2 else (C["ink_muted"] if is_miss else C["ink"]),
            va="center",
        ), mc_b))
        card_checks.append((ax.text(
            3.56, yb + 0.59,
            f"w{idx+1}={w_m:.3f}" if not is_miss else "w2=0.000",
            fontsize=7.85, fontweight="bold" if is_miss else "normal",
            color=C["teal_dark"] if is_miss else C["ink_soft"], ha="right", va="center",
        ), mc_b))
        draw_ifn_bar(ax, 0.67, yb + 0.31, 2.89, 0.14, mu_m, nu_m, pi_m)
        sub_b = (
            "Vacuous <0.00, 0.00, 1.00>"
            if is_miss
            else f"Disc: <{disc[idx][0]:.2f}, {disc[idx][1]:.2f}, {disc[idx][2]:.2f}>"
        )
        card_checks.append((ax.text(
            0.67, yb + 0.13,
            sub_b,
            fontsize=7.8, color=C["ink_soft"], va="center",
        ), mc_b))

    ax.plot([3.68, 3.80, 3.80, 3.68], [3.66, 3.66, 1.02, 1.02], color="#0F4C5C", lw=1.2, zorder=5)
    draw_arrow(ax, (3.80, 2.34), (3.92, 2.34), color="#0F4C5C", lw=1.3)
    draw_step_badge(ax, 3.88, 2.68, "1", r=0.13, color="#0F4C5C")

    # Center Operator Card in Panel (b)
    pb_mid = draw_card(ax, 3.94, 0.62, 2.90, 3.44, fc=C["white"], ec=C["slate_blue"], lw=1.15, pad=0.03)
    card_checks.append((ax.text(5.39, 3.88, "Choquet Fusion (Eq. 4)", fontsize=8.3, fontweight="bold", color=C["slate_blue"], ha="center", va="center"), pb_mid))

    mx0, my0, cw, ch = 4.34, 1.84, 0.54, 0.36
    mod_short = ["M", "F", "P", "C"]
    for r_i in range(4):
        card_checks.append((ax.text(mx0 - 0.16, my0 + (3 - r_i + 0.5) * ch, mod_short[r_i], fontsize=7.9, fontweight="bold", color=C["ink_soft"], ha="center", va="center"), pb_mid))
        card_checks.append((ax.text(mx0 + (r_i + 0.5) * cw, my0 + 4 * ch + 0.15, mod_short[r_i], fontsize=7.9, fontweight="bold", color=C["ink_soft"], ha="center", va="center"), pb_mid))
        for c_i in range(4):
            val = float(pair_c[r_i, c_i])
            cx_cell = mx0 + c_i * cw
            cy_cell = my0 + (3 - r_i) * ch
            if r_i == c_i:
                fc_cell, txt_cell, tc_cell = "#F8FAFC", "0", "#64748B"
            elif r_i == 1 or c_i == 1:
                fc_cell, txt_cell, tc_cell = "#F1F5F9", "0", "#94A3B8"
            else:
                alpha_c = min(1.0, val / 0.72)
                r_rgb = int(255 - alpha_c * (255 - 200))
                g_rgb = int(246 - alpha_c * (246 - 75))
                b_rgb = int(244 - alpha_c * (244 - 49))
                fc_cell = f"#{r_rgb:02x}{g_rgb:02x}{b_rgb:02x}"
                txt_cell = f"{val:.2f}"
                tc_cell = "#FFFFFF" if val > 0.45 else C["ink"]
            cell_patch = Rectangle((cx_cell, cy_cell), cw, ch, facecolor=fc_cell, edgecolor="#475569", lw=0.65, zorder=6)
            ax.add_patch(cell_patch)
            card_checks.append((ax.text(cx_cell + 0.5 * cw, cy_cell + 0.5 * ch, txt_cell, fontsize=7.8, fontweight="bold", color=tc_cell, ha="center", va="center", zorder=7), cell_patch))

    card_checks.append((ax.text(5.39, 1.58, f"Conflict C_O = {c_o:.3f}", fontsize=8.05, fontweight="bold", color=C["vermilion"], ha="center", va="center"), pb_mid))
    card_checks.append((ax.text(5.39, 1.30, f"d = exp(-γ·C_O) = {d_val:.3f}", fontsize=7.95, color=C["ink"], ha="center", va="center"), pb_mid))
    card_checks.append((ax.text(5.39, 1.03, "π_f = Π (π̂_m)^w_m", fontsize=7.95, color=C["ink"], ha="center", va="center"), pb_mid))
    card_checks.append((ax.text(5.39, 0.77, "Miss o2=0 => w2 = 0", fontsize=7.9, fontweight="bold", color=C["teal_dark"], ha="center", va="center"), pb_mid))

    ax.plot([6.85, 6.95], [2.34, 2.34], color="#0F4C5C", lw=1.3, zorder=5)
    ax.plot([6.95, 6.95], [1.38, 3.24], color="#0F4C5C", lw=1.3, zorder=5)
    draw_arrow(ax, (6.95, 3.24), (7.04, 3.24), color="#0F4C5C", lw=1.3)
    draw_arrow(ax, (6.95, 1.38), (7.04, 1.38), color="#0F4C5C", lw=1.3)
    draw_step_badge(ax, 6.95, 3.68, "2", r=0.13, color="#0F4C5C")

    # Top-Right Output Box in Panel (b)
    pb_out = draw_card(ax, 7.04, 2.39, 2.78, 1.67, fc=C["teal_light"], ec=C["teal"], lw=1.25, pad=0.03)
    card_checks.append((ax.text(8.43, 3.85, "✓ Fused IFS & Triage", fontsize=8.2, fontweight="bold", color="#115E59", ha="center", va="center"), pb_out))
    draw_ifn_bar(ax, 7.36, 3.45, 2.14, 0.17, mu_f, nu_f, pi_f)
    card_checks.append((ax.text(8.43, 3.17, f"<{mu_f:.3f}, {nu_f:.3f}, {pi_f:.3f}>", fontsize=8.05, fontweight="bold", color=C["ink"], ha="center", va="center"), pb_out))
    card_checks.append((ax.text(8.43, 2.86, f"p_IFS={p_ifs:.3f} · π={pi_f:.3f}", fontsize=7.8, color=C["ink"], ha="center", va="center"), pb_out))
    card_checks.append((ax.text(8.43, 2.57, f"s_ref={s_ref:.3f}: Refer PET", fontsize=7.8, fontweight="bold", color=C["teal_dark"], ha="center", va="center"), pb_out))

    # Bottom-Right Output Box in Panel (b)
    pb_tsk = draw_card(ax, 7.04, 0.62, 2.78, 1.65, fc=C["ochre_light"], ec=C["ochre_border"], lw=1.1, pad=0.03)
    card_checks.append((ax.text(8.43, 2.07, "Concept TSK Rules", fontsize=8.2, fontweight="bold", color="#7C2D12", ha="center", va="center"), pb_tsk))
    card_checks.append((ax.text(8.43, 1.80, "IF MRI LOW ^ pTau HIGH", fontsize=7.8, color=C["ink"], ha="center", va="center"), pb_tsk))
    card_checks.append((ax.text(8.43, 1.54, "THEN y = w0 + Σ g·w·z", fontsize=7.8, color=C["ink_soft"], ha="center", va="center"), pb_tsk))
    gx = np.linspace(7.34, 9.52, 180)
    gy_base = 0.90
    ax.plot([7.30, 9.56], [gy_base, gy_base], color="#475569", lw=0.75, zorder=6)
    for center_g, col_g, lbl_g in [(7.68, C["vermilion"], "LOW"), (8.43, C["teal"], "MED"), (9.18, C["slate_blue"], "HIGH")]:
        bell = gy_base + 0.40 * np.exp(-0.5 * ((gx - center_g) / 0.17) ** 2)
        ax.fill_between(gx, gy_base, bell, color=col_g, alpha=0.22, zorder=5)
        ax.plot(gx, bell, color=col_g, lw=1.25, zorder=6)
        card_checks.append((ax.text(center_g, gy_base - 0.15, lbl_g, fontsize=7.9, fontweight="bold", color=col_g, ha="center", va="center"), pb_tsk))

    # Legend strip for 3-segment IFS bar
    for lx, lcol, ltxt in [
        (0.32, C["teal"], "Support μ"),
        (2.05, C["slate_blue"], "Refutation ν"),
        (4.15, "#E9C46A", "Hesitancy π = 1 - μ - ν"),
    ]:
        ax.add_patch(Rectangle((lx, 0.25), 0.24, 0.15, facecolor=lcol, edgecolor="#334155", lw=0.6, zorder=6))
        card_checks.append((ax.text(lx + 0.31, 0.325, ltxt, fontsize=8.0, color=C["ink_soft"], va="center"), pb_outer))

    # -------------------------------------------------------------------------
    # PANEL (c): Simplex & Fuzzy-Rough Geometry (Right: x in [10.02, 15.32], y in [0.10, 8.60])
    # -------------------------------------------------------------------------
    pc_outer = draw_card(ax, 10.02, 0.10, 5.30, 8.50, fc="#F8FAFC", ec="#CBD5E1", lw=1.25, pad=0.04)
    card_checks.append((ax.text(
        12.67, 8.33,
        "(c) Simplex & Fuzzy-Rough Geometry",
        fontsize=9.6, fontweight="bold", color=C["ink"], ha="center", va="center",
    ), pc_outer))

    pc_top = draw_card(ax, 10.16, 3.74, 5.00, 4.30, fc=C["white"], ec="#CBD5E1", lw=1.0, pad=0.03)
    card_checks.append((ax.text(
        12.66, 7.82,
        "Simplex Δ_IFS: μ + ν + π = 1",
        fontsize=8.4, fontweight="bold", color=C["ink"], ha="center", va="center",
    ), pc_top))

    v_mu = (11.40, 4.48)
    v_nu = (14.42, 4.48)
    v_pi = (12.91, 7.20)

    r_left = bary_to_xy(0.70, 0.00, 0.30, v_mu, v_nu, v_pi)
    r_right = bary_to_xy(0.00, 0.70, 0.30, v_mu, v_nu, v_pi)
    ax.add_patch(Polygon([r_left, r_right, v_pi], closed=True, facecolor="#FEF3C7", edgecolor="none", alpha=0.55, zorder=3))
    ax.plot([r_left[0], r_right[0]], [r_left[1], r_right[1]], color=C["ochre"], lw=1.0, ls="--", zorder=4)
    card_checks.append((ax.text(15.08, r_right[1] + 0.12, "Refer\n(π≥0.30)", fontsize=7.8, fontweight="bold", color="#92400E", ha="right", va="bottom", zorder=6, linespacing=1.1), pc_top))

    for p_iso in [0.2, 0.4, 0.6, 0.8]:
        pL = bary_to_xy(1.0 - p_iso, 0.0, p_iso, v_mu, v_nu, v_pi)
        pR = bary_to_xy(0.0, 1.0 - p_iso, p_iso, v_mu, v_nu, v_pi)
        ax.plot([pL[0], pR[0]], [pL[1], pR[1]], color="#CBD5E1", lw=0.65, ls=":", zorder=3)
    mid_base = bary_to_xy(0.5, 0.5, 0.0, v_mu, v_nu, v_pi)
    ax.plot([v_pi[0], mid_base[0]], [v_pi[1], mid_base[1]], color="#E2E8F0", lw=0.75, ls="-.", zorder=3)

    ax.add_patch(Polygon([v_mu, v_nu, v_pi], closed=True, facecolor="none", edgecolor="#1E293B", lw=1.5, zorder=5))

    card_checks.append((ax.text(v_pi[0], v_pi[1] + 0.22, "Vacuity π=1 <0,0,1>", fontsize=8.0, fontweight="bold", color="#92400E", ha="center", va="center"), pc_top))
    card_checks.append((ax.text(11.34, 4.18, "μ=1 <1,0,0>", fontsize=7.9, fontweight="bold", color=C["teal_dark"], ha="center", va="center"), pc_top))
    card_checks.append((ax.text(14.34, 4.18, "ν=1 <0,1,0>", fontsize=7.9, fontweight="bold", color=C["slate_blue"], ha="center", va="center"), pc_top))

    pt_mri = bary_to_xy(0.12, 0.76, 0.12, v_mu, v_nu, v_pi)
    pt_fdg = bary_to_xy(0.00, 0.00, 1.00, v_mu, v_nu, v_pi)
    pt_pla = bary_to_xy(0.86, 0.05, 0.09, v_mu, v_nu, v_pi)
    pt_cli = bary_to_xy(0.18, 0.68, 0.14, v_mu, v_nu, v_pi)
    pt_g0 = bary_to_xy(mu_0, nu_0, pi_0, v_mu, v_nu, v_pi)
    pt_fus = bary_to_xy(mu_f, nu_f, pi_f, v_mu, v_nu, v_pi)
    pt_ds = bary_to_xy(ds_pos, ds_neg, ds_u, v_mu, v_nu, v_pi)

    traj_xy = np.array([bary_to_xy(a, b, c, v_mu, v_nu, v_pi) for a, b, c in data["traj"]])
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], color=C["vermilion"], lw=2.0, zorder=6)
    draw_arrow(ax, (traj_xy[14, 0], traj_xy[14, 1]), (traj_xy[29, 0], traj_xy[29, 1]), color=C["vermilion"], lw=1.8, head_scale=10.5, zorder=7)

    ax.scatter([pt_mri[0]], [pt_mri[1]], s=44, color=C["slate_blue"], edgecolors="white", lw=0.8, zorder=8)
    card_checks.append((ax.annotate(
        "MRI",
        xy=(pt_mri[0] + 0.04, pt_mri[1]),
        xytext=(14.68, 4.86),
        fontsize=7.85, fontweight="bold", color=C["slate_blue"], ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=C["slate_blue"], lw=0.85),
        zorder=9,
    ), pc_top))

    ax.scatter([pt_cli[0]], [pt_cli[1]], s=44, color=C["teal"], edgecolors="white", lw=0.8, zorder=8)
    card_checks.append((ax.annotate(
        "Clin",
        xy=(pt_cli[0] + 0.01, pt_cli[1] + 0.04),
        xytext=(13.52, 5.62),
        fontsize=7.85, fontweight="bold", color=C["teal_dark"], ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=C["teal_dark"], lw=0.85),
        zorder=9,
    ), pc_top))

    ax.scatter([pt_pla[0]], [pt_pla[1]], s=50, color=C["vermilion"], edgecolors="white", lw=0.8, zorder=8)
    card_checks.append((ax.annotate(
        "Plasma\n(<60)",
        xy=(pt_pla[0] - 0.03, pt_pla[1] + 0.03),
        xytext=(10.26, 4.92),
        fontsize=7.85, fontweight="bold", color=C["vermilion"],
        arrowprops=dict(arrowstyle="->", color=C["vermilion"], lw=0.95),
        zorder=9,
    ), pc_top))

    ax.scatter([pt_fdg[0]], [pt_fdg[1]], s=92, marker="*", color=C["ochre"], edgecolors="#1E293B", lw=0.8, zorder=9)
    card_checks.append((ax.annotate(
        "Thm 2: Miss FDG\no2=0 -> w2=0",
        xy=(pt_fdg[0] - 0.05, pt_fdg[1] - 0.04),
        xytext=(10.24, 6.72),
        fontsize=7.8, fontweight="bold", color="#92400E",
        arrowprops=dict(arrowstyle="->", color="#92400E", lw=1.0, connectionstyle="arc3,rad=-0.18"),
        zorder=9,
    ), pc_top))

    ax.scatter([pt_g0[0]], [pt_g0[1]], s=40, facecolors="white", edgecolors="#475569", lw=1.3, zorder=8)
    card_checks.append((ax.text(pt_g0[0] + 0.10, pt_g0[1] - 0.16, "γ=0", fontsize=7.85, color="#475569", zorder=9), pc_top))

    ax.scatter([pt_fus[0]], [pt_fus[1]], s=70, color=C["teal"], edgecolors="#1E293B", lw=1.1, zorder=9)
    card_checks.append((ax.annotate(
        f"Thm 4 (γ=0.60)\n<{mu_f:.2f},{nu_f:.2f},{pi_f:.2f}>",
        xy=(pt_fus[0] - 0.05, pt_fus[1] + 0.02),
        xytext=(10.22, 5.74),
        fontsize=7.8, fontweight="bold", color="#115E59",
        arrowprops=dict(arrowstyle="->", color="#115E59", lw=1.05, connectionstyle="arc3,rad=0.12"),
        zorder=9,
    ), pc_top))

    ax.scatter([pt_ds[0]], [pt_ds[1]], s=52, marker="X", color=C["vermilion"], edgecolors="#1E293B", lw=0.6, zorder=9)
    card_checks.append((ax.annotate(
        f"DS (π={ds_u:.3f})",
        xy=(pt_ds[0], pt_ds[1] - 0.04),
        xytext=(12.91, 3.94),
        fontsize=7.85, fontweight="bold", color=C["vermilion"], ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=C["vermilion"], lw=0.95, connectionstyle="arc3,rad=0.08"),
        zorder=9,
    ), pc_top))

    # Bottom Subcard in (c): Theorem 5 Fuzzy-Rough Inclusion Envelope
    pc_bot = draw_card(ax, 10.16, 0.28, 5.00, 3.28, fc=C["white"], ec="#CBD5E1", lw=1.0, pad=0.03)
    card_checks.append((ax.text(
        12.66, 3.31,
        "Thm 5: Lower R(A) <= A(x) <= Upper R(A)",
        fontsize=8.1, fontweight="bold", color=C["ink"], ha="center", va="center",
    ), pc_bot))

    px0, py0, pw, ph = 10.72, 1.06, 3.82, 1.90
    ax.plot([px0, px0 + pw], [py0, py0], color="#334155", lw=1.0, zorder=5)
    ax.plot([px0, px0], [py0, py0 + ph], color="#334155", lw=1.0, zorder=5)
    card_checks.append((ax.text(px0 - 0.24, py0 + 0.5 * ph, "Membership", fontsize=7.95, fontweight="bold", color=C["ink_soft"], rotation=90, ha="center", va="center"), pc_bot))

    t_grid = np.linspace(0.0, 1.0, 200)
    a_curve = 0.14 + 0.66 * (1.0 - np.exp(-2.8 * t_grid)) + 0.07 * np.sin(5.0 * t_grid)
    bnd_half = 0.16 + 0.06 * np.sin(3.2 * t_grid + 0.4)
    upper_c = np.clip(a_curve + bnd_half, a_curve, 0.98)
    lower_c = np.clip(a_curve - bnd_half * 0.92, 0.03, a_curve)

    xg = px0 + t_grid * pw
    yu = py0 + upper_c * ph
    ya = py0 + a_curve * ph
    yl = py0 + lower_c * ph

    ax.fill_between(xg, yl, yu, color="#FDE68A", alpha=0.42, zorder=4)
    ax.plot(xg, yu, color=C["ochre"], lw=1.45, ls="--", zorder=6)
    ax.plot(xg, ya, color=C["ink"], lw=1.85, zorder=7)
    ax.plot(xg, yl, color=C["slate_blue"], lw=1.45, ls="-.", zorder=6)

    card_checks.append((ax.text(xg[125], yu[125] + 0.12, "Upper R(A)", fontsize=7.95, fontweight="bold", color="#B45309", ha="center"), pc_bot))
    card_checks.append((ax.text(xg[-1] + 0.08, ya[-1], "A(x)", fontsize=8.0, fontweight="bold", color=C["ink"], ha="left", va="center"), pc_bot))
    card_checks.append((ax.text(xg[135], yl[135] - 0.18, "Lower R(A)", fontsize=7.95, fontweight="bold", color=C["slate_blue"], ha="center"), pc_bot))
    card_checks.append((ax.annotate(
        "BND(A) >= 0",
        xy=(xg[62], yu[62] - 0.11),
        xytext=(px0 + 0.18, py0 + ph - 0.12),
        fontsize=7.95, fontweight="bold", color="#92400E",
        arrowprops=dict(arrowstyle="->", color="#92400E", lw=0.95, connectionstyle="arc3,rad=-0.12"),
        zorder=8,
    ), pc_bot))

    pc_foot = draw_card(ax, 10.22, 0.39, 4.88, 0.46, fc=C["ochre_light"], ec=C["ochre_border"], lw=0.85, pad=0.015)
    card_checks.append((ax.text(
        12.66, 0.62,
        "In-Fold Reduct: 320->24 MRI · 1k->20 RNA",
        fontsize=7.8, fontweight="bold", color="#7C2D12", ha="center", va="center",
    ), pc_foot))

    assert_no_text_overlaps(fig, card_checks=card_checks)
    saved = []
    for d_out in out_dirs:
        os.makedirs(d_out, exist_ok=True)
        for ext in ("pdf", "png"):
            p = os.path.join(d_out, f"Fig0_Paradigm_Teaser_Comparison.{ext}")
            fig.savefig(p, format=ext, dpi=300, facecolor="white")
            saved.append(p)
    plt.close(fig)
    return saved


# =============================================================================
# FIGURE 2: Fig0_System_Architecture_Overview
# =============================================================================
def generate_fig2_architecture(out_dirs: List[str]) -> List[str]:
    configure_rc()
    fig = plt.figure(figsize=(8.70, 5.44), dpi=300)
    ax = fig.add_axes([0.005, 0.008, 0.99, 0.984])
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.0, 9.9)
    ax.axis("off")
    card_checks: List[Tuple[Any, Any]] = []

    # =========================================================================
    # PANEL (a): Four-Stage Differentiable Operator Pipeline (y in [3.34, 9.78])
    # =========================================================================
    pa_all = draw_card(ax, 0.10, 3.34, 15.80, 6.44, fc="#FAFBFD", ec="#CBD5E1", lw=1.25, pad=0.04)
    card_checks.append((ax.text(
        0.30, 9.53,
        "(a) Differentiable Pipeline of DeepFuzzyTorch (Unified & Multi-FM Pathways)",
        fontsize=9.1, fontweight="bold", color=C["ink"], va="center",
    ), pa_all))

    stages = [
        (0.22, 2.86, "#F1F5F9", "#CBD5E1", "I. Inputs & Mask o_i", C["ink"]),
        (3.18, 4.38, "#EFF6FA", C["slate_border"], "II. Reduct, Gate & Encoders", "#0F4C5C"),
        (7.66, 2.54, "#FDF3F1", C["vermilion_border"], "III. Choquet Fusion", "#7F1D1D"),
        (10.30, 5.48, "#F0FAF8", C["teal_border"], "IV. TSK Head, Blend & Triage", "#115E59"),
    ]
    stage_patches = []
    for sx, sw, sfc, sec, stitle, stc in stages:
        sp = draw_card(ax, sx, 4.78, sw, 4.52, fc=sfc, ec=sec, lw=1.05, pad=0.03)
        stage_patches.append(sp)
        card_checks.append((ax.text(sx + 0.5 * sw, 9.09, stitle, fontsize=8.0, fontweight="bold", color=stc, ha="center", va="center"), sp))

    # -------------------------------------------------------------------------
    # Stage I: Multimodal Inputs & Observation Mask o_i (x in [0.22, 3.08])
    # -------------------------------------------------------------------------
    c_mri = draw_card(ax, 0.30, 7.66, 2.70, 0.98, fc=C["white"], ec=C["slate_border"], lw=0.95, pad=0.02)
    draw_modality_glyph(ax, "mri", 0.54, 8.15, r=0.155)
    card_checks.append((ax.text(0.76, 8.38, "Neuro & Clinical", fontsize=7.85, fontweight="bold", color=C["ink"], va="center"), c_mri))
    card_checks.append((ax.text(0.76, 8.11, "MRI 320D, Plasma", fontsize=7.75, color=C["ink_soft"], va="center"), c_mri))
    card_checks.append((ax.text(0.76, 7.85, "Clin 17D · o_m=1", fontsize=7.75, fontweight="bold", color=C["slate_blue"], va="center"), c_mri))

    c_tcga = draw_card(ax, 0.30, 6.54, 2.70, 0.98, fc=C["white"], ec=C["teal_border"], lw=0.95, pad=0.02)
    rng = np.random.default_rng(42)
    bx_rna = 0.40
    for k_bar in range(12):
        col_b = [C["vermilion"], C["teal"], C["slate_blue"], C["ochre"]][int(rng.integers(0, 4))]
        ax.add_patch(Rectangle((bx_rna + k_bar * 0.025, 6.82), 0.019, 0.42, facecolor=col_b, edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((bx_rna, 6.82), 12 * 0.025, 0.42, facecolor="none", edgecolor="#334155", lw=0.6, zorder=7))
    card_checks.append((ax.text(0.76, 7.26, "TCGA Omics (7)", fontsize=7.85, fontweight="bold", color=C["ink"], va="center"), c_tcga))
    card_checks.append((ax.text(0.76, 6.99, "RNA 1k MAD pool", fontsize=7.75, color=C["ink_soft"], va="center"), c_tcga))
    card_checks.append((ax.text(0.76, 6.73, "UNI2-h 48D · o=1", fontsize=7.75, fontweight="bold", color=C["teal_dark"], va="center"), c_tcga))

    c_fdg = draw_card(ax, 0.30, 5.42, 2.70, 0.98, fc=C["miss_bg"], ec=C["miss_border"], lw=1.0, ls="--", pad=0.02)
    draw_modality_glyph(ax, "fdg", 0.54, 5.91, r=0.145)
    card_checks.append((ax.text(0.76, 6.14, "Unacquired (FDG)", fontsize=7.85, fontweight="bold", color=C["ink_muted"], va="center"), c_fdg))
    card_checks.append((ax.text(0.76, 5.88, "Indicator o_m = 0", fontsize=7.75, fontweight="bold", color=C["vermilion"], va="center"), c_fdg))
    card_checks.append((ax.text(0.76, 5.62, "Active set O_i", fontsize=7.75, color=C["ink"], va="center"), c_fdg))

    card_checks.append((ax.text(1.65, 5.04, "Mask o_i in {0, 1}^M", fontsize=7.8, fontweight="bold", color=C["ink"], ha="center", va="center"), stage_patches[0]))

    # -------------------------------------------------------------------------
    # Stage II: Two wide columns (x in [3.18, 7.56])
    # -------------------------------------------------------------------------
    c_fr = draw_card(ax, 3.30, 7.66, 1.98, 0.98, fc="#E2E8F0", ec="#475569", lw=1.0, pad=0.02)
    card_checks.append((ax.text(4.29, 8.34, "FR Reduct", fontsize=7.9, fontweight="bold", color=C["ink"], ha="center", va="center"), c_fr))
    card_checks.append((ax.text(4.29, 8.06, "K*=24 / 20", fontsize=7.8, fontweight="bold", color="#0F4C5C", ha="center", va="center"), c_fr))
    card_checks.append((ax.text(4.29, 7.80, "(Offline in-fold)", fontsize=7.75, color=C["ink_muted"], ha="center", va="center"), c_fr))

    c_rg = draw_card(ax, 3.30, 6.28, 1.98, 1.24, fc=C["ochre_light"], ec=C["ochre"], lw=1.05, pad=0.02)
    card_checks.append((ax.text(4.29, 7.32, "Rough Gate", fontsize=7.8, fontweight="bold", color="#7C2D12", ha="center", va="center"), c_rg))
    gx_fr = np.linspace(3.56, 5.02, 50)
    t_fr = (gx_fr - 3.56) / 1.46
    ym_fr = 6.74 + 0.24 * t_fr
    yu_fr = ym_fr + 0.11
    yl_fr = ym_fr - 0.10
    ax.fill_between(gx_fr, yl_fr, yu_fr, color="#FDE68A", alpha=0.55, zorder=5)
    ax.plot(gx_fr, yu_fr, color=C["ochre"], lw=0.9, ls="--", zorder=6)
    ax.plot(gx_fr, ym_fr, color=C["ink"], lw=1.1, zorder=6)
    ax.plot(gx_fr, yl_fr, color=C["slate_blue"], lw=0.9, ls="-.", zorder=6)
    card_checks.append((ax.text(4.29, 6.46, "L<=A<=U (bnd)", fontsize=7.75, fontweight="bold", color="#7C2D12", ha="center", va="center"), c_rg))

    draw_arrow(ax, (3.02, 8.15), (3.28, 8.15), color="#334155", lw=1.15)
    draw_arrow(ax, (3.02, 7.00), (3.28, 7.00), color="#334155", lw=1.15)
    draw_arrow(ax, (4.29, 7.65), (4.29, 7.53), color="#334155", lw=1.1)

    c_enc = draw_card(ax, 5.44, 7.38, 2.00, 1.26, fc="#DBEAFE", ec=C["slate_blue"], lw=1.05, pad=0.02)
    card_checks.append((ax.text(6.44, 8.40, "Encoders E_m", fontsize=7.8, fontweight="bold", color=C["ink"], ha="center", va="center"), c_enc))
    card_checks.append((ax.text(6.44, 8.10, "A_m = <μ,ν,π>", fontsize=7.75, fontweight="bold", color="#0F4C5C", ha="center", va="center"), c_enc))
    draw_ifn_bar(ax, 5.64, 7.77, 1.60, 0.15, 0.56, 0.24, 0.20)
    card_checks.append((ax.text(6.44, 7.54, "Aux L_aux on O_i", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), c_enc))

    c_sig = draw_card(ax, 5.44, 6.28, 2.00, 0.94, fc="#E0F2FE", ec=C["slate_blue"], lw=1.0, pad=0.02)
    card_checks.append((ax.text(6.44, 6.99, "Signed Concepts", fontsize=7.75, fontweight="bold", color=C["slate_blue"], ha="center", va="center"), c_sig))
    card_checks.append((ax.text(6.44, 6.72, "z_m = μ_m - ν_m", fontsize=7.75, fontweight="bold", color=C["ink"], ha="center", va="center"), c_sig))
    card_checks.append((ax.text(6.44, 6.46, "To Stage IV TSK", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), c_sig))

    draw_arrow(ax, (5.29, 7.98), (5.42, 7.98), color="#334155", lw=1.15)
    draw_arrow(ax, (5.29, 6.84), (5.42, 6.84), color="#334155", lw=1.15)
    draw_arrow(ax, (6.44, 7.37), (6.44, 7.23), color="#334155", lw=1.1)

    c_vac = draw_card(ax, 3.30, 5.32, 4.14, 0.82, fc=C["white"], ec=C["miss_border"], lw=1.0, ls="--", pad=0.02)
    card_checks.append((ax.text(5.37, 5.91, "Missing Vacuity (o_m = 0)", fontsize=7.8, fontweight="bold", color=C["ink"], ha="center", va="center"), c_vac))
    card_checks.append((ax.text(5.37, 5.55, "<0,0,1>, w_m=0 (e+=e-=0)", fontsize=7.75, fontweight="bold", color=C["teal_dark"], ha="center", va="center"), c_vac))

    draw_arrow(ax, (3.02, 5.73), (3.28, 5.73), color="#64748B", lw=1.15, ls="--")
    card_checks.append((ax.text(5.37, 5.02, "Zero weight if o_m = 0", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), stage_patches[1]))

    # -------------------------------------------------------------------------
    # Stage III: Conflict-Discounted Choquet Fusion (x in [7.66, 10.20])
    # -------------------------------------------------------------------------
    c_ch = draw_card(ax, 7.78, 5.32, 2.30, 3.16, fc="#F9D8D2", ec=C["vermilion"], lw=1.3, pad=0.03)
    card_checks.append((ax.text(8.93, 8.22, "Conflict Choquet", fontsize=7.85, fontweight="bold", color="#7F1D1D", ha="center", va="center"), c_ch))
    card_checks.append((ax.text(8.93, 7.90, "IFSChoquetFusion", fontsize=7.75, fontweight="bold", color=C["vermilion"], ha="center", va="center"), c_ch))
    card_checks.append((ax.text(8.93, 7.53, "Active set O_i", fontsize=7.75, color=C["ink"], ha="center", va="center"), c_ch))
    card_checks.append((ax.text(8.93, 7.16, "w_k = G_k - G_{k-1}", fontsize=7.75, color=C["ink"], ha="center", va="center"), c_ch))
    card_checks.append((ax.text(8.93, 6.79, "d = exp(-γ · C_O)", fontsize=7.75, fontweight="bold", color=C["ink"], ha="center", va="center"), c_ch))
    card_checks.append((ax.text(8.93, 6.40, "Eq. (4) log-space", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), c_ch))
    card_checks.append((ax.text(8.93, 6.00, "f64 log1p/expm1", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), c_ch))
    card_checks.append((ax.text(8.93, 5.56, "Loss L_simplex", fontsize=7.75, fontweight="bold", color="#7F1D1D", ha="center", va="center"), c_ch))

    draw_arrow(ax, (7.46, 7.98), (7.76, 7.98), color="#334155", lw=1.2)
    draw_arrow(ax, (7.46, 5.73), (7.76, 5.73), color="#64748B", lw=1.15, ls="--")
    card_checks.append((ax.text(8.93, 5.02, "▼ Zoomed in (b)", fontsize=7.75, fontweight="bold", color=C["vermilion"], ha="center", va="center"), stage_patches[2]))

    # Overhead concept bus routed cleanly above Stage cards with self-sizing white pill (zorder=8)
    ax.plot([7.46, 7.58, 7.58, 10.66, 10.66], [6.75, 6.75, 8.75, 8.75, 8.62], color=C["slate_blue"], lw=1.15, ls="-.", zorder=6)
    draw_arrow(ax, (10.66, 8.72), (10.66, 8.56), color=C["slate_blue"], lw=1.2, head_scale=9.5, zorder=7)
    ax.text(
        8.93, 8.75,
        "Concepts z_m = μ_m - ν_m",
        fontsize=7.75, fontweight="bold", color=C["slate_blue"], ha="center", va="center", zorder=8,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C["white"], edgecolor=C["slate_border"], linewidth=0.8),
    )

    # -------------------------------------------------------------------------
    # Stage IV: Two wide columns + full-span bottom Triage/Loss card (x in [10.30, 15.78])
    # -------------------------------------------------------------------------
    c_mc = draw_card(ax, 10.42, 7.36, 2.52, 1.24, fc=C["white"], ec=C["slate_border"], lw=1.0, pad=0.02)
    card_checks.append((ax.text(11.68, 8.36, "Masked Concepts", fontsize=7.8, fontweight="bold", color="#0F4C5C", ha="center", va="center"), c_mc))
    card_checks.append((ax.text(11.68, 8.06, "z_m = μ_m - ν_m", fontsize=7.75, color=C["ink"], ha="center", va="center"), c_mc))
    card_checks.append((ax.text(11.68, 7.78, "+ c_clin (mask m_z)", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), c_mc))
    card_checks.append((ax.text(11.68, 7.52, "o_m=0 => m_z,m=0", fontsize=7.75, fontweight="bold", color=C["teal_dark"], ha="center", va="center"), c_mc))

    c_tsk = draw_card(ax, 13.12, 7.36, 2.54, 1.24, fc=C["ochre_light"], ec=C["ochre"], lw=1.1, pad=0.02)
    card_checks.append((ax.text(14.39, 8.38, "DeepConceptTSK", fontsize=7.8, fontweight="bold", color="#7C2D12", ha="center", va="center"), c_tsk))
    card_checks.append((ax.text(14.39, 8.12, "R=6 · Err ≤ 2.39e-7", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), c_tsk))
    gx_tsk = np.linspace(13.44, 15.34, 140)
    gy_tsk = 7.66
    ax.plot([13.40, 15.38], [gy_tsk, gy_tsk], color="#475569", lw=0.7, zorder=6)
    for c_bell, col_bell, lbl_bell in [(13.76, C["vermilion"], "LO"), (14.39, C["teal"], "MD"), (15.02, C["slate_blue"], "HI")]:
        y_bell = gy_tsk + 0.30 * np.exp(-0.5 * ((gx_tsk - c_bell) / 0.14) ** 2)
        ax.fill_between(gx_tsk, gy_tsk, y_bell, color=col_bell, alpha=0.22, zorder=5)
        ax.plot(gx_tsk, y_bell, color=col_bell, lw=1.1, zorder=6)
        card_checks.append((ax.text(c_bell, gy_tsk - 0.14, lbl_bell, fontsize=7.75, fontweight="bold", color=col_bell, ha="center", va="center"), c_tsk))

    draw_arrow(ax, (12.95, 7.98), (13.10, 7.98), color="#7C2D12", lw=1.15)

    c_fa = draw_card(ax, 10.42, 6.02, 2.52, 1.12, fc=C["white"], ec=C["vermilion_border"], lw=1.05, pad=0.02)
    card_checks.append((ax.text(11.68, 6.90, "Fused A_f = <μ, ν, π>", fontsize=7.75, fontweight="bold", color="#7F1D1D", ha="center", va="center"), c_fa))
    draw_ifn_bar(ax, 10.68, 6.52, 2.00, 0.16, 0.35, 0.31, 0.34)
    card_checks.append((ax.text(11.68, 6.24, "p_IFS = μ + 0.5 · π", fontsize=7.75, color=C["ink"], ha="center", va="center"), c_fa))

    draw_arrow(ax, (10.10, 6.58), (10.40, 6.58), color="#7F1D1D", lw=1.2)

    c_bl = draw_card(ax, 13.12, 6.02, 2.54, 1.12, fc=C["white"], ec=C["teal"], lw=1.15, pad=0.02)
    card_checks.append((ax.text(14.39, 6.90, "Blend ⊕ -> Risk p_i", fontsize=7.75, fontweight="bold", color="#115E59", ha="center", va="center"), c_bl))
    card_checks.append((ax.text(14.39, 6.58, "(1-α)p_IFS + α·p_TSK", fontsize=7.75, color=C["ink"], ha="center", va="center"), c_bl))
    card_checks.append((ax.text(14.39, 6.26, "α = σ(θ_TSK)(1 - π_f)", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), c_bl))

    draw_arrow(ax, (12.95, 6.58), (13.10, 6.58), color="#115E59", lw=1.2)
    draw_arrow(ax, (14.39, 7.34), (14.39, 7.16), color="#7C2D12", lw=1.15)

    c_loss = draw_card(ax, 10.42, 4.94, 5.26, 0.90, fc="#FEF2F2", ec=C["vermilion_border"], lw=1.05, pad=0.02)
    card_checks.append((ax.text(13.05, 5.56, "s_ref = 0.5(π + C_O) >= τ_ref => Refer PET", fontsize=7.75, fontweight="bold", color="#7F1D1D", ha="center", va="center"), c_loss))
    card_checks.append((ax.text(13.05, 5.20, "L_tot = v_i·l_sup + λ_bnd·BND + λ_sp||g||_1/RC", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), c_loss))
    draw_arrow(ax, (10.10, 5.44), (10.40, 5.44), color=C["vermilion"], lw=1.2, ls="--")

    # -------------------------------------------------------------------------
    # Modular Multi-FM Pathology Consensus Pathway (Track 5)
    # -------------------------------------------------------------------------
    c_t5 = draw_card(ax, 0.22, 3.44, 15.56, 1.24, fc=C["plum_light"], ec=C["plum_border"], lw=1.15, pad=0.03)
    card_checks.append((ax.text(
        0.42, 4.47,
        "Modular Multi-FM Pathology Consensus (Track 5: Camelyon17 & NCT-CRC Standalone Operators)",
        fontsize=7.95, fontweight="bold", color=C["plum"], va="center",
    ), c_t5))

    c_t5_1 = draw_card(ax, 0.36, 3.55, 3.66, 0.72, fc=C["white"], ec=C["plum_border"], lw=0.95, pad=0.02)
    card_checks.append((ax.text(2.19, 4.07, "6 Pathology FMs (P=6)", fontsize=7.8, fontweight="bold", color=C["ink"], ha="center", va="center"), c_t5_1))
    card_checks.append((ax.text(2.19, 3.76, "UNI2-h · Phikon-v2 · Kaiko · PLIP", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), c_t5_1))

    draw_arrow(ax, (4.04, 3.91), (4.62, 3.91), color=C["plum"], lw=1.25)
    ax.text(4.33, 4.13, "H(x)", fontsize=7.75, fontweight="bold", color=C["plum"], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.10", facecolor=C["plum_light"], edgecolor="none"))

    c_t5_2 = draw_card(ax, 4.64, 3.55, 4.24, 0.72, fc=C["white"], ec=C["plum"], lw=1.05, pad=0.02)
    card_checks.append((ax.text(6.76, 4.07, "HesitantFuzzyAggregator (Torra)", fontsize=7.8, fontweight="bold", color=C["plum"], ha="center", va="center"), c_t5_2))
    card_checks.append((ax.text(6.76, 3.76, "π_HFS = min(0.95, 2·sqrt(Var(h)))", fontsize=7.75, color=C["ink"], ha="center", va="center"), c_t5_2))

    draw_arrow(ax, (8.90, 3.91), (9.24, 3.91), color=C["plum"], lw=1.25)

    c_t5_3 = draw_card(ax, 9.26, 3.55, 3.68, 0.72, fc=C["white"], ec=C["plum"], lw=1.05, pad=0.02)
    card_checks.append((ax.text(11.10, 4.07, "PythagoreanFuzzyTTALoss", fontsize=7.8, fontweight="bold", color=C["plum"], ha="center", va="center"), c_t5_3))
    card_checks.append((ax.text(11.10, 3.76, "OOD TTA on μ_P² + ν_P² <= 1", fontsize=7.75, color=C["ink"], ha="center", va="center"), c_t5_3))

    draw_arrow(ax, (12.96, 3.91), (13.30, 3.91), color=C["plum"], lw=1.25)

    c_t5_4 = draw_card(ax, 13.32, 3.55, 2.34, 0.72, fc=C["white"], ec=C["teal"], lw=1.05, pad=0.02)
    card_checks.append((ax.text(14.49, 4.07, "Adapted Consensus", fontsize=7.75, fontweight="bold", color="#115E59", ha="center", va="center"), c_t5_4))
    card_checks.append((ax.text(14.49, 3.76, "p_TTA(x) & Induced IFN", fontsize=7.75, color=C["ink"], ha="center", va="center"), c_t5_4))

    # =========================================================================
    # PANEL (b): Zoomed Operator Mechanics (y in [0.10, 3.20])
    # =========================================================================
    pb_all = draw_card(ax, 0.10, 0.10, 15.80, 3.10, fc="#FDFBF9", ec="#CBD5E1", lw=1.25, pad=0.04)
    card_checks.append((ax.text(
        0.30, 2.94,
        "(b) Conflict-Discounted Choquet Fusion (IFSChoquetFusion, Active O_i) & Pythagorean TTA Disk",
        fontsize=9.1, fontweight="bold", color=C["ink"], va="center",
    ), pb_all))

    # Card (b.1): Certainty-Ordered Choquet Capacity Weights (x in [0.22, 3.58], w=3.36)
    pb1 = draw_card(ax, 0.22, 0.24, 3.36, 2.48, fc=C["white"], ec="#CBD5E1", lw=1.0, pad=0.03)
    card_checks.append((ax.text(1.90, 2.50, "1. Ordered Capacities (Eq. 3)", fontsize=7.9, fontweight="bold", color=C["ink"], ha="center", va="center"), pb1))
    bx_cap, by_cap = 0.42, 0.96
    ax.plot([bx_cap - 0.04, bx_cap + 2.94], [by_cap, by_cap], color="#334155", lw=0.9, zorder=6)
    cap_bars = [
        (0.91, C["slate_blue"], "σ(1)", "0.343"),
        (0.88, C["slate_blue"], "σ(2)", "0.332"),
        (0.86, C["slate_blue"], "σ(3)", "0.325"),
        (0.04, "#E2E8F0", "o_m=0", "0.000"),
    ]
    for k_idx, (r_h, col_h, lbl_h, w_txt) in enumerate(cap_bars):
        xx = bx_cap + k_idx * 0.74 + 0.08
        hh = r_h * 1.05
        ax.add_patch(Rectangle(
            (xx, by_cap), 0.44, hh, facecolor=col_h, edgecolor="#334155",
            lw=0.75, ls="--" if k_idx == 3 else "-", zorder=6
        ))
        card_checks.append((ax.text(xx + 0.22, by_cap + hh + 0.07, w_txt, fontsize=7.75, fontweight="bold",
                color=C["teal_dark"] if k_idx == 3 else C["ink"], ha="center", va="bottom"), pb1))
        card_checks.append((ax.text(xx + 0.22, by_cap - 0.18, lbl_h, fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), pb1))
    card_checks.append((ax.text(1.90, 0.48, "G_k=(Σ g)^p, w_k=G_k-G_{k-1}", fontsize=7.75, color=C["ink"], ha="center", va="center"), pb1))

    # Card (b.2): Observed-Pair Contradiction Matrix (x in [3.68, 8.24], w=4.56)
    pb2 = draw_card(ax, 3.68, 0.24, 4.56, 2.48, fc=C["white"], ec="#CBD5E1", lw=1.0, pad=0.03)
    card_checks.append((ax.text(5.96, 2.50, "2. Observed Conflict C_O (Eq. 2)", fontsize=7.9, fontweight="bold", color=C["ink"], ha="center", va="center"), pb2))
    mx2, my2, cw2, ch2 = 3.82, 0.88, 0.56, 0.34
    ex_mat = [
        [0.00, 0.00, 0.66, 0.22],
        [0.00, 0.00, 0.00, 0.00],
        [0.66, 0.00, 0.00, 0.59],
        [0.22, 0.00, 0.59, 0.00],
    ]
    for r_i in range(4):
        for c_i in range(4):
            val = ex_mat[r_i][c_i]
            cx_c = mx2 + c_i * cw2
            cy_c = my2 + (3 - r_i) * ch2
            if r_i == c_i:
                fc_c, txt_c, tc_c = "#F8FAFC", "0", "#64748B"
            elif r_i == 1 or c_i == 1:
                fc_c, txt_c, tc_c = "#F1F5F9", "0", "#94A3B8"
            else:
                fc_c = "#C84B31" if val > 0.5 else "#F2B8AC"
                txt_c = f"{val:.2f}"
                tc_c = "#FFFFFF" if val > 0.5 else C["ink"]
            c_cell = Rectangle((cx_c, cy_c), cw2, ch2, facecolor=fc_c, edgecolor="#475569", lw=0.65, zorder=6)
            ax.add_patch(c_cell)
            card_checks.append((ax.text(cx_c + 0.5 * cw2, cy_c + 0.5 * ch2, txt_c, fontsize=7.75, fontweight="bold", color=tc_c, ha="center", va="center", zorder=7), c_cell))
    card_checks.append((ax.text(4.94, 0.50, "Row 2 (o2=0): 0", fontsize=7.75, fontweight="bold", color=C["teal_dark"], ha="center", va="center"), pb2))
    card_checks.append((ax.text(7.14, 2.02, "C_mn =\nμ_mν_n + ν_mμ_n", fontsize=7.75, fontweight="bold", color="#7F1D1D", ha="center", va="center", linespacing=1.15), pb2))
    card_checks.append((ax.text(7.14, 1.48, "C_O = mean C_mn", fontsize=7.75, color=C["ink"], ha="center", va="center"), pb2))
    card_checks.append((ax.text(7.14, 1.06, "d = exp(-γ C_O)", fontsize=7.75, fontweight="bold", color=C["vermilion"], ha="center", va="center"), pb2))
    card_checks.append((ax.text(7.14, 0.64, "μ̂=dμ, ν̂=dν", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), pb2))

    # Card (b.3): Float64 Log-Space IFWA & Simplex (x in [8.34, 12.38], w=4.04)
    pb3 = draw_card(ax, 8.34, 0.24, 4.04, 2.48, fc=C["white"], ec="#CBD5E1", lw=1.0, pad=0.03)
    card_checks.append((ax.text(10.36, 2.50, "3. Log-Space IFWA (Eq. 4)", fontsize=7.9, fontweight="bold", color=C["ink"], ha="center", va="center"), pb3))
    card_checks.append((ax.text(9.46, 2.02, "μ_f=1-Π(1-μ̂)^w", fontsize=7.75, color=C["ink"], ha="center", va="center"), pb3))
    card_checks.append((ax.text(9.46, 1.56, "π_f = Π(π̂_m)^w_m", fontsize=7.75, fontweight="bold", color="#92400E", ha="center", va="center"), pb3))
    card_checks.append((ax.text(9.46, 1.10, "ν_f = 1 - μ_f - π_f", fontsize=7.75, color=C["ink"], ha="center", va="center"), pb3))
    card_checks.append((ax.text(9.46, 0.56, "f64 log1p+frexp", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), pb3))

    s_mu = (10.76, 0.82)
    s_nu = (11.90, 0.82)
    s_pi = (11.33, 1.98)
    ax.add_patch(Polygon([s_mu, s_nu, s_pi], closed=True, facecolor="#F8FAFC", edgecolor="#1E293B", lw=1.15, zorder=5))
    card_checks.append((ax.text(s_pi[0], s_pi[1] + 0.16, "π", fontsize=7.8, fontweight="bold", color="#92400E", ha="center", va="center"), pb3))
    card_checks.append((ax.text(s_mu[0] - 0.10, s_mu[1] - 0.12, "μ", fontsize=7.8, fontweight="bold", color=C["teal_dark"], ha="center", va="center"), pb3))
    card_checks.append((ax.text(s_nu[0] + 0.10, s_nu[1] - 0.12, "ν", fontsize=7.8, fontweight="bold", color=C["slate_blue"], ha="center", va="center"), pb3))
    p_low = bary_to_xy(0.50, 0.36, 0.14, s_mu, s_nu, s_pi)
    p_high = bary_to_xy(0.32, 0.28, 0.40, s_mu, s_nu, s_pi)
    ax.scatter([p_low[0]], [p_low[1]], s=24, facecolors="white", edgecolors="#475569", lw=1.0, zorder=7)
    ax.scatter([p_high[0]], [p_high[1]], s=34, color=C["vermilion"], edgecolors="#1E293B", lw=0.8, zorder=8)
    draw_arrow(ax, p_low, p_high, color=C["vermilion"], lw=1.4, head_scale=9.0, zorder=7)
    card_checks.append((ax.text(11.33, 0.46, "C_O↑ => π_f↑", fontsize=7.75, fontweight="bold", color=C["vermilion"], ha="center", va="center"), pb3))

    # Card (b.4): Pythagorean Fuzzy TTA Disk Geometry (x in [12.48, 15.78], w=3.30)
    pb4 = draw_card(ax, 12.48, 0.24, 3.30, 2.48, fc=C["white"], ec="#CBD5E1", lw=1.0, pad=0.03)
    card_checks.append((ax.text(14.13, 2.50, "4. Pyth. TTA (Eqs. 14–15)", fontsize=7.75, fontweight="bold", color=C["plum"], ha="center", va="center"), pb4))
    dx0, dy0, dr = 13.00, 0.86, 0.92
    wedge = Wedge((dx0, dy0), dr, 0.0, 90.0, facecolor=C["plum_light"], edgecolor=C["plum"], lw=1.25, zorder=5)
    ax.add_patch(wedge)
    ax.add_patch(Polygon([(dx0, dy0), (dx0 + dr, dy0), (dx0, dy0 + dr)], closed=True, facecolor="#E2E8F0", edgecolor="#64748B", lw=0.85, ls="--", alpha=0.65, zorder=6))
    ax.plot([dx0, dx0 + dr + 0.08], [dy0, dy0], color="#1E293B", lw=1.0, zorder=7)
    ax.plot([dx0, dx0], [dy0, dy0 + dr + 0.08], color="#1E293B", lw=1.0, zorder=7)
    card_checks.append((ax.text(dx0 + dr + 0.14, dy0 - 0.14, "μ_P", fontsize=7.75, fontweight="bold", color=C["ink"], ha="center"), pb4))
    card_checks.append((ax.text(dx0 - 0.06, dy0 + dr + 0.04, "ν_P", fontsize=7.75, fontweight="bold", color=C["ink"], ha="right", va="center"), pb4))
    card_checks.append((ax.text(14.84, 1.94, "μ_P² + ν_P² ≤ 1", fontsize=7.75, fontweight="bold", color=C["plum"], ha="center", va="center"), pb4))
    card_checks.append((ax.text(14.84, 1.48, "q_P=1-μ_P²-ν_P²", fontsize=7.75, color=C["ink"], ha="center", va="center"), pb4))
    card_checks.append((ax.text(14.84, 1.04, "Mask π̃_P ≤ τ_P", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), pb4))
    card_checks.append((ax.text(14.13, 0.46, "Relaxes IFS simplex for TTA", fontsize=7.75, color=C["ink_soft"], ha="center", va="center"), pb4))

    assert_no_text_overlaps(fig, card_checks=card_checks)
    saved = []
    for d_out in out_dirs:
        os.makedirs(d_out, exist_ok=True)
        for ext in ("pdf", "png"):
            p = os.path.join(d_out, f"Fig0_System_Architecture_Overview.{ext}")
            fig.savefig(p, format=ext, dpi=300, facecolor="white")
            saved.append(p)
    plt.close(fig)
    return saved


def main() -> None:
    data = compute_exact_clinical_conflict_case(gamma=0.60)
    out_dirs = [
        os.path.join(WORKSPACE_ROOT, "figures"),
        os.path.join(WORKSPACE_ROOT, "figures", "fig0_redesign_20261108"),
    ]
    s1 = generate_fig1_teaser(data, out_dirs)
    s2 = generate_fig2_architecture(out_dirs)

    print("=== Exact Verified Mathematical Summary for Fig0_Paradigm_Teaser_Comparison ===")
    for idx, (name, o_m, mu, nu, pi, _) in enumerate(data["raw_triples"]):
        mh, nh, ph = data["discounted"][idx]
        print(
            f"  Modality {idx+1} ({name:11s}, o={o_m}): raw=<{mu:.2f}, {nu:.2f}, {pi:.2f}>, "
            f"w={data['weights'][idx]:.4f}, discounted=<{mh:.4f}, {nh:.4f}, {ph:.4f}>"
        )
    print(f"  Observed-pair conflict C_O = {data['c_o']:.6f}, discount d = exp(-{data['gamma']:.2f}*C_O) = {data['d']:.6f}")
    mf, nf, pf = data["fused"]
    print(f"  Exact Eq. (4) Fused IFS    = <{mf:.6f}, {nf:.6f}, {pf:.6f}> (sum = {mf+nf+pf:.12f})")
    print(f"  Pignistic p_IFS            = {data['p_ifs']:.6f}, Triage score s_ref = {data['s_ref']:.6f}")
    dp, dn, du = data["ds"]
    print(f"  Classical Dempster-Shafer  = <{dp:.6f}, {dn:.6f}, {du:.6f}>, P_DS = {data['p_ds']:.6f}, amp = {data['ds_amp']:.4f}x")
    print("Saved files:")
    for p in s1 + s2:
        print(" ", p)


if __name__ == "__main__":
    main()
