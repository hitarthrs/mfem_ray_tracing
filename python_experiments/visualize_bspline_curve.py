"""Visualize a B-spline / NURBS curve (multi-span, needs a knot vector).

Unlike a single Bézier segment (visualize_bezier_curve.py), a curve like a
piecewise B-spline horseshoe needs:
  - control points  Q_0 .. Q_{n-1}
  - degree p
  - knot vector t   (length n + p + 1)
  - optional weights w_i for rational NURBS

Setup:
    cd python_faffing && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

Run demo (9 CPs, open uniform knots):
    python_faffing/.venv/bin/python python_faffing/visualize_bspline_curve.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline


def open_uniform_knots(num_control_points: int, degree: int) -> np.ndarray:
    """
    Open uniform knot vector on [0, 1] with n - p active spans.

    Example: n=9, p=2 -> 7 spans; n=9, p=5 -> 4 spans (closer to 4-span figures).
    """
    n = num_control_points
    p = degree
    num_spans = n - p
    if num_spans < 1:
        raise ValueError(f"Need n > p (got n={n}, p={p})")

    internal = np.linspace(0.0, 1.0, num_spans + 1)[1:-1]
    return np.concatenate([np.zeros(p + 1), internal, np.ones(p + 1)])


def evaluate_bspline(
    control_points: np.ndarray,
    knots: np.ndarray,
    degree: int,
    u: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    C(u) = sum_i N_{i,p}(u) Q_i           (B-spline)
    C(u) = sum_i N_{i,p}(u) w_i Q_i / sum_i N_{i,p}(u) w_i   (NURBS)
    """
    q = np.asarray(control_points, dtype=float)
    t = np.asarray(knots, dtype=float)
    p = int(degree)

    if q.ndim != 2 or q.shape[0] < p + 1:
        raise ValueError("control_points must be (n, dim) with n >= p+1")
    if t.size != q.shape[0] + p + 1:
        raise ValueError(f"knot vector length must be n+p+1={q.shape[0] + p + 1}, got {t.size}")

    u_min, u_max = t[p], t[-p - 1]
    if u is None:
        u = np.linspace(u_min, u_max, 400)
    u = np.asarray(u, dtype=float)

    dim = q.shape[1]
    if weights is None:
        curve = np.column_stack([BSpline(t, q[:, d], p)(u) for d in range(dim)])
        return u, curve

    w = np.asarray(weights, dtype=float)
    if w.shape != (q.shape[0],):
        raise ValueError("weights must have shape (n,)")

    basis = np.zeros((u.size, q.shape[0]))
    for i in range(q.shape[0]):
        coeff = np.zeros(q.shape[0])
        coeff[i] = 1.0
        basis[:, i] = BSpline(t, coeff, p)(u)

    wu = basis @ w
    num = basis * w
    curve = (num @ q) / wu[:, np.newaxis]
    return u, curve


def unique_knot_spans(knots: np.ndarray, degree: int) -> np.ndarray:
    """Distinct knot values that bound active spans in [t_p, t_{n}]."""
    t = np.asarray(knots, dtype=float)
    p = degree
    active = t[p : -p]
    return np.unique(active)


def horseshoe_control_points() -> np.ndarray:
    """Ten control points forming an open horseshoe (similar to IGA figures)."""
    return np.array(
        [
            [-2.0, 0.0],  # 1. Start on the left axis
            [-1.6, 1.4],  # 2. Top-left curve
            [-0.6, 2.0],  # 3. Top-left peak
            [0.6, 2.0],  # 4. Top-right peak
            [1.6, 1.4],  # 5. Top-right curve
            [2.0, 0.0],  # 6. Rightmost apex
            [1.6, -1.4],  # 7. Bottom-right curve
            [0.6, -2.0],  # 8. Bottom peak
            [-0.6, -1.4],  # 9. Bottom-left curve
            [-1.6, -0.7],  # 10. Added to complete the horseshoe flow
        ]
    )

def s_shaped_control_points() -> np.ndarray:
    return s_shaped_p6_control_points()


def s_shaped_p6_control_points() -> np.ndarray:
    """Nine control points for the S-shaped demo (used with degree-6 knots)."""
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 3.0],
            [6.0, 0.0],
            [7.0, -3.0],
            [9.0, -4.0],
            [11.0, -2.0],
            [12.0, 0.0],
        ],
        dtype=float,
    )


def s_shaped_knots_p6() -> np.ndarray:
    """Open knot vector for degree p=6: two spans [0, 0.5] and [0.5, 1] (n=9 CPs)."""
    return np.array(
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 0.5,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        dtype=float,
    )


def s_shaped_20_control_points() -> np.ndarray:
    return np.array([
    [0.0, 0.0],
    [1.0, 2.0],
    [3.0, 4.0],
    [4.3333333333, 3.3333333333],
    [5.2222222222, 1.7777777778],
    [5.7666666667, 0.5333333333],
    [6.3111111111, -0.7111111111],
    [6.7088888889, -1.5822222222],
    [7.1511111111, -2.3644444444],
    [7.7377777778, -3.1911111111],
    [8.4844444444, -3.6977777778],
    [9.6133333333, -3.2266666667],
    [11.04, -1.6],
    [11.6, -0.8],
    [12.0, 0.0],
])
    
def s_shaped_ultra_refined_control_points() -> np.ndarray:
    return np.array([[0.0000000000, 0.0000000000],
  [0.6000000000, 1.2000000000],
    [1.5600000000, 2.4000000000],
    [2.6480000000, 3.3280000000],
    [3.4032000000, 3.4880000000],
    [3.7360000000, 3.4773333333],
    [3.9955555556, 3.3244444444],
    [4.6888888889, 2.7111111111],
    [5.2222222222, 1.7777777778],
    [5.7666666667, 0.5333333333],
    [6.3111111111, -0.7111111111],
    [6.7088888889, -1.5822222222],
    [7.1511111111, -2.3644444444],
    [7.7377777778, -3.1911111111],
    [8.4844444444, -3.6977777778],
    [9.6133333333, -3.2266666667],
    [11.0400000000, -1.6000000000],
    [11.6000000000, -0.8000000000],
    [12.0000000000, 0.0000000000]])
    
# def multiple_peak_control_points() -> np.ndarray:
#     return np.array([
#         [0.0, 5.0],
#         [2.0, 8.0],
#         [5.0, 2.0],
#         [7.0, 19.0],
#         [8.0, -12.0],
#         [9.0, 19.0],
#         [11.0, 4.0],
#         [14.0, 4.0],
#         [17.0, 8.5],
#         [20.0, 6.0],
#         [22.0, 2.5],
#         [24.0, 5.0],
#     ])
    
def multiple_peak_control_points() -> np.ndarray:
    return np.array([
        [0.0, 5.0],
        [2.0, 8.0],
        [5.0, 2.0],
        # --- OSCILLATION ZONE START ---
        [6.2, 3.5],    # NEW POINT 1: Stabilizes the inflection into the first peak
        [7.0, 19.0],   # First High Peak
        [7.5, 4.0],    # NEW POINT 2: Sharpens the drop halfway down the first peak
        [8.0, -12.0],  # Deep Sharp Trough
        [8.5, 4.0],    # NEW POINT 3: Sharpens the ascent into the second peak
        [9.0, 19.0],   # Second High Peak
        [10.0, 3.0],   # NEW POINT 4: Captures the tight exit pullout before the plateau
        # --- OSCILLATION ZONE END ---
        [11.0, 4.0],
        [14.0, 4.0],
        [17.0, 8.5],
        [20.0, 6.0],
        [22.0, 2.5],
        [24.0, 5.0],
    ])

def multiple_peak_knots() -> np.ndarray:
    return np.array([0,0,0,0,0,0,0,0.25, 0.25, 0.25, 0.5, 0.5, 0.5,0.75, 0.75, 0.75,1,1,1,1,1,1,1])

def visualize_bspline_curve(
    control_points: np.ndarray,
    knots: np.ndarray,
    degree: int,
    weights: np.ndarray | None = None,
    *,
    title: str | None = None,
    show_knot_axis: bool = True,
    save_path: Path | None = None,
) -> None:
    q = np.asarray(control_points, dtype=float)
    t = np.asarray(knots, dtype=float)
    p = int(degree)

    if q.shape[1] != 2:
        raise ValueError("Only 2D plotting is implemented")

    u, curve = evaluate_bspline(q, t, p, weights=weights)
    kind = "NURBS" if weights is not None else "B-spline"
    num_spans = q.shape[0] - p

    if show_knot_axis:
        fig, (ax_curve, ax_u) = plt.subplots(
            2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax_curve = plt.subplots(figsize=(8, 5))
        ax_u = None

    ax_curve.plot(curve[:, 0], curve[:, 1], "b-", linewidth=2.5, label=f"{kind} curve")
    ax_curve.plot(q[:, 0], q[:, 1], "ko--", markersize=5, linewidth=1, label="Control polygon")
    for i, pt in enumerate(q):
        ax_curve.annotate(f"Q{i}", pt, textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax_curve.set_aspect("equal", adjustable="datalim")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="best")
    ax_curve.set_title(title or f"{kind} curve: n={q.shape[0]} CPs, p={p}, {num_spans} spans")

    if ax_u is not None:
        span_vals = unique_knot_spans(t, p)
        ax_u.set_xlim(span_vals[0], span_vals[-1])
        ax_u.set_ylim(0.0, 1.0)
        ax_u.set_xlabel("u (parameter)")
        ax_u.set_ylabel("E")
        ax_u.set_yticks([])
        for kv in span_vals:
            ax_u.axvline(kv, color="0.3", linewidth=1)
        for i in range(len(span_vals) - 1):
            mid = 0.5 * (span_vals[i] + span_vals[i + 1])
            ax_u.text(mid, 0.5, f"span {i}", ha="center", va="center", fontsize=9)
        ax_u.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a B-spline / NURBS curve.")
    parser.add_argument("--degree", type=int, default=2, help="Spline degree p (default: 2)")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional NURBS weights (one per control point)",
    )
    parser.add_argument(
        "--knots",
        nargs="+",
        type=float,
        default=None,
        help="Full knot vector (length n+p+1). Default: open uniform.",
    )
    parser.add_argument("--save", type=Path, default=None, help="Optional PNG output path")
    args = parser.parse_args()

    control_points = horseshoe_control_points()
    n = control_points.shape[0]
    p = args.degree

    if args.knots is not None:
        knots = np.array(args.knots, dtype=float)
    else:
        knots = open_uniform_knots(n, p)

    visualize_bspline_curve(
        control_points,
        knots,
        p,
        weights=args.weights,
        save_path=args.save,
    )
