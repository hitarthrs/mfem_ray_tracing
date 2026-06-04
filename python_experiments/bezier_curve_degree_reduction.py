"""Polynomial Bézier degree reduction (Piegl & Tiller, unit weights).

Mirrors ``BezierDegreeReduce`` in ``src/bezier_degree_reduction.cpp``.

Input control points have shape (degree + 1, dim) with degree >= 2.
Output has shape (degree, dim). ``max_err`` is the bound from eq. (5.43)
(even input degree) or (5.44) (odd input degree).

Run:
    python_faffing/.venv/bin/python python_faffing/bezier_curve_degree_reduction.py
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
from scipy.special import comb

from visualize_bezier_curve import evaluate_bezier, visualize_bezier_curve


def _alpha(i: int, degree: int) -> float:
    """alpha_i = i / n for input Bézier degree n."""
    return i / degree


def _bernstein(i: int, degree: int, u: float) -> float:
    return float(comb(degree, i, exact=False) * (u**i) * ((1.0 - u) ** (degree - i)))


def _max_bernstein(i: int, degree: int, samples: int = 512) -> float:
    peak = 0.0
    for s in range(samples + 1):
        u = s / samples
        peak = max(peak, _bernstein(i, degree, u))
    return peak


def _max_abs_bernstein_difference(i: int, j: int, degree: int, samples: int = 512) -> float:
    peak = 0.0
    for s in range(samples + 1):
        u = s / samples
        peak = max(peak, abs(_bernstein(i, degree, u) - _bernstein(j, degree, u)))
    return peak


def _forward_control_point(q_i: np.ndarray, p_prev: np.ndarray, i: int, degree: int) -> np.ndarray:
    """P_i = (Q_i - alpha_i P_{i-1}) / (1 - alpha_i)."""
    a = _alpha(i, degree)
    return (q_i - a * p_prev) / (1.0 - a)


def _backward_control_point(q_ip1: np.ndarray, p_next: np.ndarray, i: int, degree: int) -> np.ndarray:
    """P_i = (Q_{i+1} - (1 - alpha_{i+1}) P_{i+1}) / alpha_{i+1}."""
    a = _alpha(i + 1, degree)
    return (q_ip1 - (1.0 - a) * p_next) / a


def _validate_control_points(bpts: np.ndarray) -> None:
    bpts = np.asarray(bpts, dtype=float)
    if bpts.ndim != 2 or bpts.shape[0] < 3:
        raise ValueError("need at least 3 control points (input degree >= 2)")
    if not np.all(np.isfinite(bpts)):
        raise ValueError("control points must be finite")


def _max_error_even(
    bpts: np.ndarray,
    reduced: np.ndarray,
    degree: int,
    r: int,
) -> float:
    """Eq. (5.43): even input degree."""
    mid = 0.5 * (reduced[r] + reduced[r + 1])
    err_norm = float(np.linalg.norm(bpts[r + 1] - mid))
    return err_norm * _max_bernstein(r + 1, degree)


def _max_error_odd(
    p_left: np.ndarray,
    p_right: np.ndarray,
    r: int,
    degree: int,
) -> float:
    """Eq. (5.44): odd input degree."""
    diff_norm = float(np.linalg.norm(p_left - p_right))
    scale = 0.5 * (1.0 - _alpha(r, degree))
    return scale * diff_norm * _max_abs_bernstein_difference(r, r + 1, degree)


def bezier_degree_reduce(bpts: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Reduce polynomial Bézier degree by one (Piegl & Tiller).

    Parameters
    ----------
    bpts
        Control points with shape (n + 1, dim); input degree is n >= 2.

    Returns
    -------
    reduced
        Control points with shape (n, dim); output degree is n - 1.
    max_err
        Upper bound on max curve deviation over u in [0, 1].
    """
    bpts = np.asarray(bpts, dtype=float)
    _validate_control_points(bpts)

    # n = input degree. Odd/even branching uses n, not the CP count n + 1.
    degree = bpts.shape[0] - 1
    dim = bpts.shape[1]
    r = (degree - 1) // 2

    reduced = np.zeros((degree, dim), dtype=float)
    reduced[0] = bpts[0]

    for i in range(1, r + 1):
        reduced[i] = _forward_control_point(bpts[i], reduced[i - 1], i, degree)

    reduced[degree - 1] = bpts[degree]
    for i in range(degree - 2, r, -1):
        reduced[i] = _backward_control_point(bpts[i + 1], reduced[i + 1], i, degree)

    if degree % 2 == 1:
        p_left = _forward_control_point(bpts[r], reduced[r - 1], r, degree)
        p_right = _backward_control_point(bpts[r + 1], reduced[r + 1], r, degree)
        reduced[r] = 0.5 * (p_left + p_right)
        max_err = _max_error_odd(p_left, p_right, r, degree)
    else:
        max_err = _max_error_even(bpts, reduced, degree, r)

    return reduced, max_err


def max_curve_deviation(
    high: np.ndarray,
    low: np.ndarray,
    samples: int = 101,
) -> float:
    """Sample max Euclidean distance between two Bézier curves on [0, 1]."""
    u = np.linspace(0.0, 1.0, samples)
    _, curve_high = evaluate_bezier(high, u)
    _, curve_low = evaluate_bezier(low, u)
    return float(np.max(np.linalg.norm(curve_high - curve_low, axis=1)))


def _print_reduction(name: str, bpts: np.ndarray) -> None:
    reduced, max_err = bezier_degree_reduce(bpts)
    dev = max_curve_deviation(bpts, reduced)
    in_deg = bpts.shape[0] - 1
    out_deg = reduced.shape[0] - 1
    print(f"{name}: degree {in_deg} -> {out_deg}")
    print(f"  input CPs:\n{bpts}")
    print(f"  reduced CPs:\n{reduced}")
    print(f"  max_err bound = {max_err:.6g}")
    print(f"  sampled curve deviation = {dev:.6g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bézier degree reduction demo.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the symmetric cubic example before and after reduction",
    )
    parser.add_argument("--save", type=str, default=None, help="Optional PNG path")
    args = parser.parse_args()

    cubic = np.array(
        [
            [0.0, 0.0],
            [3.0, 6.0],
            [6.0, 6.0],
            [9.0, 0.0],
        ]
    )
    quintic = np.array(
        [
            [0.0, 0.0],
            [3.2, 12.8],
            [6.4, 19.2],
            [9.6, 19.2],
            [12.8, 12.8],
            [16.0, 0.0],
        ]
    )

    _print_reduction("Symmetric cubic", cubic)
    print()
    _print_reduction("Symmetric quintic", quintic)

    if args.plot:
        reduced, _ = bezier_degree_reduce(cubic)
        visualize_bezier_curve(
            cubic,
            title=f"Original cubic (p={cubic.shape[0] - 1})",
        )
        visualize_bezier_curve(
            reduced,
            title=f"Reduced quadratic (p={reduced.shape[0] - 1})",
            save_path=args.save,
        )
