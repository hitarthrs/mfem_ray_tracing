"""
GCD-based brute-force knot refinement.

From the active knot vector, compute span lengths between unique knot values,
take their GCD as base step ``d``, then build a uniform grid with step ``h = d / r``.
Original knots keep their multiplicities; grid points not already present are
candidates for insertion.

The grid step is clamped below by ``min_step`` (default 0.05) so ``-r`` cannot
subdivide finer than that gap.
"""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
from math import gcd, lcm

import numpy as np

from .knot_refinement_common import (
    as_fraction,
    new_knots_from_grid,
    refinement_grid,
    unique_active_knots,
)

DEFAULT_MIN_STEP = 0.01


def span_lengths(unique_knots: np.ndarray) -> list[Fraction]:
    """Consecutive span lengths between sorted unique knots."""
    unique = [as_fraction(float(v)) for v in unique_knots]
    return [unique[i + 1] - unique[i] for i in range(len(unique) - 1)]


def spans_gcd(spans: list[Fraction]) -> Fraction:
    """GCD of span lengths (exact rational arithmetic)."""
    if not spans:
        raise ValueError("need at least one span to compute GCD")
    if len(spans) == 1:
        return spans[0]

    common_den = reduce(lcm, (s.denominator for s in spans))
    scaled = [s.numerator * (common_den // s.denominator) for s in spans]
    return Fraction(reduce(gcd, scaled), common_den)


def refinement_step(base_step: Fraction, resolution: int) -> Fraction:
    """Requested grid step ``d / r`` with integer resolution ``r >= 1``."""
    if resolution < 1:
        raise ValueError("resolution must be >= 1")
    return base_step / resolution


def clamp_refinement_step(step: Fraction, min_step: float) -> Fraction:
    """Raise ``step`` to at least ``min_step`` (prevents over-refinement)."""
    if min_step <= 0:
        raise ValueError("min_step must be positive")
    floor = as_fraction(float(min_step))
    return step if step >= floor else floor


def gcd_new_knots(
    knotvector: np.ndarray,
    degree: int,
    *,
    resolution: int = 1,
    min_step: float = DEFAULT_MIN_STEP,
) -> tuple[list[float], Fraction, Fraction, Fraction, list[Fraction]]:
    """
    Plan GCD-based knot insertion (existing knots excluded).

    Returns
    -------
    new_knots : list[float]
        Sorted parameter values not already in ``U``.
    d : Fraction
        GCD of unique span lengths.
    h_req : Fraction
        Requested step ``d / resolution``.
    h : Fraction
        Actual step ``max(h_req, min_step)``.
    grid : list[Fraction]
        Full refinement grid on ``[U_p, U_n]`` (for logging).
    """
    unique = unique_active_knots(knotvector, degree)
    spans = span_lengths(unique)
    d = spans_gcd(spans)
    h_req = refinement_step(d, resolution)
    h = clamp_refinement_step(h_req, min_step)

    u_start = as_fraction(float(unique[0]))
    u_end = as_fraction(float(unique[-1]))
    grid = refinement_grid(u_start, u_end, h)
    new_knots = new_knots_from_grid(knotvector, grid)

    return new_knots, d, h_req, h, grid
