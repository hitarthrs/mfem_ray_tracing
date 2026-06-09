"""
Shared knot-refinement utilities and plug-and-play application hook.

Algorithms implement a ``*_new_knots(knotvector, degree, **opts)`` planner that
returns ``(new_knots, grid)``; existing knot values in ``U`` are left untouched.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from geomdl import BSpline, operations

DEFAULT_INSERT_MULTIPLICITY = 2


def as_fraction(value: float) -> Fraction:
    return Fraction(value).limit_denominator(10_000)


def unique_active_knots(knotvector: np.ndarray, degree: int) -> np.ndarray:
    """Sorted unique knot values in the active interval ``[U_p, U_n]``."""
    u = np.asarray(knotvector, dtype=float)
    p = int(degree)
    n = u.size - p - 1
    return np.unique(u[p : n + 1])


def refinement_grid(
    u_start: Fraction,
    u_end: Fraction,
    step: Fraction,
) -> list[Fraction]:
    """Uniform grid from ``u_start`` to ``u_end`` inclusive at spacing ``step``."""
    if step <= 0:
        raise ValueError("step must be positive")

    n_steps = int((u_end - u_start) / step)
    return [u_start + k * step for k in range(n_steps + 1)]


def knot_already_present(knotvector: np.ndarray, u: float, *, atol: float = 1e-9) -> bool:
    kv = np.asarray(knotvector, dtype=float)
    return bool(np.any(np.isclose(kv, float(u), atol=atol)))


def new_knots_from_grid(
    knotvector: np.ndarray,
    grid: list[Fraction],
) -> list[float]:
    """Grid points not already in ``U`` (existing multiplicities unchanged)."""
    kv = np.asarray(knotvector, dtype=float)
    new_knots: list[float] = []
    for t in grid:
        u_val = float(t)
        if not knot_already_present(kv, u_val):
            new_knots.append(u_val)
    return new_knots


def apply_new_knots(
    curve: BSpline.Curve,
    new_knots: list[float],
    insert_multiplicity: int,
    *,
    algorithm: str,
) -> list[tuple[float, int]]:
    """Insert planned knots via geomdl; returns ``(u, n_inserted)`` log."""
    if insert_multiplicity < 1:
        raise ValueError("insert_multiplicity must be >= 1")

    print(f"new knots to insert ({algorithm}, m={insert_multiplicity}): {new_knots}")
    log: list[tuple[float, int]] = []
    for u_val in new_knots:
        operations.insert_knot(curve, [float(u_val)], [int(insert_multiplicity)])
        log.append((float(u_val), insert_multiplicity))
        print(f"  {u_val:g}: inserted ×{insert_multiplicity}")

    if not new_knots:
        print("  (no new grid knots — refinement is a no-op)")

    return log
