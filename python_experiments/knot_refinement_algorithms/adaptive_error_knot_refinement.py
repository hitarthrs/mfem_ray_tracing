"""
Adaptive knot refinement driven by A5.11 degree-reduction error.

Each iteration:
  1. Degree-reduce the current curve (planning only — geometry input unchanged).
  2. Find knot index ``m`` with maximum ``error_array[m]``.
  3. Insert a knot at the midpoint of ``U[m]`` and the next unique knot value
     (multiplicity ``M`` for new values; existing values left untouched).

Repeat up to ``max_iterations`` (default 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from geomdl import BSpline, operations

from b_spline_curve_reduction import DegreeReduceCurve
from .knot_refinement_common import knot_already_present, unique_active_knots

DEFAULT_MAX_ITERATIONS = 5


@dataclass
class AdaptiveIterationRecord:
    iteration: int
    max_index: int
    max_error: float
    u_at: float
    u_next: float | None
    u_insert: float | None
    inserted: bool
    note: str


def next_unique_knot_after(
    knotvector: np.ndarray,
    degree: int,
    u_at: float,
    *,
    atol: float = 1e-9,
) -> float | None:
    """Smallest unique active knot strictly greater than ``u_at``."""
    unique = unique_active_knots(knotvector, degree)
    for v in unique:
        if float(v) > float(u_at) + atol:
            return float(v)
    return None


def midpoint_knot_for_max_error(
    knotvector: np.ndarray,
    degree: int,
    error_array: np.ndarray,
) -> tuple[int, float, float, float] | None:
    """
    Midpoint insertion site from the max-error knot index.

    Returns ``(m, u_at, u_next, u_mid)`` or ``None`` if no valid site exists.
    """
    err = np.asarray(error_array, dtype=float)
    if err.size == 0 or not np.any(err > 0):
        return None

    m = int(np.argmax(err))
    u = np.asarray(knotvector, dtype=float)
    if m < 0 or m >= u.size:
        return None

    u_at = float(u[m])
    u_next = next_unique_knot_after(u, degree, u_at)
    if u_next is None:
        return None

    u_mid = 0.5 * (u_at + u_next)
    return m, u_at, u_next, u_mid


def insert_single_knot_if_new(
    curve: BSpline.Curve,
    u_insert: float,
    insert_multiplicity: int,
) -> bool:
    """Insert at ``u_insert`` when not already in the knot vector."""
    kv = np.asarray(curve.knotvector, dtype=float)
    if knot_already_present(kv, u_insert):
        return False
    operations.insert_knot(curve, [float(u_insert)], [int(insert_multiplicity)])
    return True


def adaptive_error_refine(
    curve: BSpline.Curve,
    qw: np.ndarray,
    knotvector: np.ndarray,
    degree: int,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    insert_multiplicity: int = 2,
) -> list[AdaptiveIterationRecord]:
    """
    Iteratively refine ``curve`` in place using degree-reduction error feedback.

    ``qw`` / ``knotvector`` are refreshed from ``curve`` after each insertion.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if insert_multiplicity < 1:
        raise ValueError("insert_multiplicity must be >= 1")

    log: list[AdaptiveIterationRecord] = []
    qw_work = np.asarray(qw, dtype=float)
    u_work = np.asarray(knotvector, dtype=float)
    p = int(degree)

    for it in range(max_iterations):
        n = qw_work.shape[0]
        out = DegreeReduceCurve(n, p, u_work, qw_work)
        if out == 1:
            log.append(
                AdaptiveIterationRecord(
                    iteration=it,
                    max_index=-1,
                    max_error=float("inf"),
                    u_at=float("nan"),
                    u_next=None,
                    u_insert=None,
                    inserted=False,
                    note="degree reduction tolerance exceeded",
                )
            )
            print(f"  iter {it}: stop — tolerance exceeded")
            break

        _, _, err = out
        site = midpoint_knot_for_max_error(u_work, p, err)
        if site is None:
            log.append(
                AdaptiveIterationRecord(
                    iteration=it,
                    max_index=int(np.argmax(err)),
                    max_error=float(np.max(err)),
                    u_at=float(u_work[int(np.argmax(err))]),
                    u_next=None,
                    u_insert=None,
                    inserted=False,
                    note="no valid midpoint site",
                )
            )
            print(f"  iter {it}: stop — no valid midpoint site")
            break

        m, u_at, u_next, u_mid = site
        max_err = float(err[m])
        print(
            f"  iter {it}: max error {max_err:.6g} at index {m} "
            f"(U[{m}]={u_at:g}, next={u_next:g}) → insert at {u_mid:g}"
        )

        if knot_already_present(u_work, u_mid):
            log.append(
                AdaptiveIterationRecord(
                    iteration=it,
                    max_index=m,
                    max_error=max_err,
                    u_at=u_at,
                    u_next=u_next,
                    u_insert=u_mid,
                    inserted=False,
                    note="midpoint already in knot vector",
                )
            )
            print(f"    skip — {u_mid:g} already present")
            continue

        insert_single_knot_if_new(curve, u_mid, insert_multiplicity)
        qw_work, u_work, p = _curve_arrays(curve)
        log.append(
            AdaptiveIterationRecord(
                iteration=it,
                max_index=m,
                max_error=max_err,
                u_at=u_at,
                u_next=u_next,
                u_insert=u_mid,
                inserted=True,
                note=f"inserted ×{insert_multiplicity}",
            )
        )
        print(f"    inserted ×{insert_multiplicity}")

    return log


def _curve_arrays(curve: BSpline.Curve) -> tuple[np.ndarray, np.ndarray, int]:
    qw = np.asarray(curve.ctrlpts, dtype=float)[:, :2]
    u = np.asarray(curve.knotvector, dtype=float)
    return qw, u, int(curve.degree)
