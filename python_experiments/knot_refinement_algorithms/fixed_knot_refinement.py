"""
Fixed-span uniform knot refinement.

Build a uniform grid on ``[U_p, U_n]`` with user-specified step ``d``. Knot values
already present in ``U`` keep their multiplicities; only missing grid points are
inserted (with the requested multiplicity).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .knot_refinement_common import (
    as_fraction,
    new_knots_from_grid,
    refinement_grid,
    unique_active_knots,
)

DEFAULT_STEP_D = 0.05


def fixed_new_knots(
    knotvector: np.ndarray,
    degree: int,
    *,
    step_d: float,
) -> tuple[list[float], float, list[Fraction]]:
    """
    Plan fixed-span knot insertion.

    Returns
    -------
    new_knots : list[float]
        Parameter values to insert (not already in ``U``).
    step_d : float
        User-specified uniform grid step ``d``.
    grid : list[Fraction]
        Full refinement grid on ``[U_p, U_n]`` (for logging).
    """
    if step_d <= 0:
        raise ValueError("step_d must be positive")

    unique = unique_active_knots(knotvector, degree)
    u_start = as_fraction(float(unique[0]))
    u_end = as_fraction(float(unique[-1]))
    step = as_fraction(float(step_d))
    grid = refinement_grid(u_start, u_end, step)
    new_knots = new_knots_from_grid(knotvector, grid)
    return new_knots, float(step_d), grid
