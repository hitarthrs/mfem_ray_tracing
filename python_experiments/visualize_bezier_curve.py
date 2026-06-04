"""Visualize a polynomial or rational Bézier curve on [0, 1].

Setup (once):
    cd python_faffing && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

Run:
    python_faffing/.venv/bin/python python_faffing/visualize_bezier_curve.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb


def bernstein_matrix(p: int, u: np.ndarray) -> np.ndarray:
    """Return B[i, j] = B_{i,p}(u[j]) with shape (p+1, len(u))."""
    u = np.asarray(u, dtype=float)
    i = np.arange(p + 1)[:, np.newaxis]
    return comb(p, i, exact=False) * (u ** i) * ((1.0 - u) ** (p - i))


def evaluate_bezier(
    control_points: np.ndarray,
    u: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Polynomial:  C(u) = sum_i B_{i,p}(u) Q_i
    Rational:    C(u) = sum_i B_{i,p}(u) w_i Q_i / sum_i B_{i,p}(u) w_i

    control_points: (p+1, dim). No knot vector — Bézier lives on [0, 1].
    """
    q = np.asarray(control_points, dtype=float)
    if q.ndim != 2 or q.shape[0] < 2:
        raise ValueError("control_points must have shape (p+1, dim) with p >= 1")

    p = q.shape[0] - 1
    if u is None:
        u = np.linspace(0.0, 1.0, 200)
    u = np.asarray(u, dtype=float)

    b = bernstein_matrix(p, u)  # (p+1, n)
    if weights is None:
        curve = (b.T @ q)
        return u, curve

    w = np.asarray(weights, dtype=float)
    if w.shape != (p + 1,):
        raise ValueError("weights must have shape (p+1,)")

    bw = b * w[:, np.newaxis]
    num = (bw.T @ q)
    den = bw.sum(axis=0)
    return u, num / den[:, np.newaxis]


def visualize_bezier_curve(
    control_points: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    title: str | None = None,
    show_polygon: bool = True,
    save_path: Path | None = None,
) -> None:
    q = np.asarray(control_points, dtype=float)
    if q.shape[1] not in (2, 3):
        raise ValueError("Only 2D or 3D control points are supported")

    _, curve = evaluate_bezier(q, weights=weights)
    p = q.shape[0] - 1
    kind = "Rational" if weights is not None else "Polynomial"

    fig = plt.figure(figsize=(8, 5))
    if q.shape[1] == 2:
        ax = fig.add_subplot(111)
        ax.plot(curve[:, 0], curve[:, 1], "b-", linewidth=2, label="Bézier curve")
        if show_polygon:
            ax.plot(q[:, 0], q[:, 1], "ro--", markersize=6, label="Control polygon")
            for i, pt in enumerate(q):
                ax.annotate(f"Q{i}", pt, textcoords="offset points", xytext=(4, 4))
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], "b-", linewidth=2, label="Bézier curve")
        if show_polygon:
            ax.plot(q[:, 0], q[:, 1], q[:, 2], "ro--", markersize=6, label="Control polygon")
        ax.legend()

    ax.set_title(title or f"{kind} Bézier curve, degree p={p}")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a Bézier curve on [0, 1].")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional rational weights (one per control point)",
    )
    parser.add_argument("--save", type=Path, default=None, help="Optional PNG output path")
    args = parser.parse_args()

    # Cubic polynomial example — 4 control points, no knot vector.
    control_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, -1.0],
            [3.0, 1.0],
        ]
    )

    visualize_bezier_curve(control_points, weights=args.weights, save_path=args.save)
