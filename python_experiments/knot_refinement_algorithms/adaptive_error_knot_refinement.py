"""
Adaptive knot refinement driven by A5.11 degree-reduction error.

**Serial mode** (``batch_size=1``, default): each outer iteration inserts one knot
at the midpoint of the max-error knot span.

**Batched mode** (``batch_size>1``): each outer iteration runs one
``DegreeReduceCurve``, ranks knot indices by ``error_array``, and inserts up to
``batch_size`` distinct midpoint sites before the next reduction pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from geomdl import BSpline, NURBS, operations

from b_spline_curve_reduction import DegreeReduceCurve
from nurbs_degree_reduction import degree_reduce_unified
from .knot_refinement_common import knot_already_present, unique_active_knots

DEFAULT_MAX_ITERATIONS = 5
DEFAULT_BATCH_SIZE = 1


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
    batch_inserts: tuple[float, ...] = field(default_factory=tuple)


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


def midpoint_knot_for_index(
    knotvector: np.ndarray,
    degree: int,
    index: int,
) -> tuple[float, float, float] | None:
    """
    Midpoint insertion site for knot index ``index``.

    Returns ``(u_at, u_next, u_mid)`` or ``None``.
    """
    u = np.asarray(knotvector, dtype=float)
    if index < 0 or index >= u.size:
        return None

    u_at = float(u[index])
    u_next = next_unique_knot_after(u, degree, u_at)
    if u_next is None:
        return None

    u_mid = 0.5 * (u_at + u_next)
    return u_at, u_next, u_mid


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
    site = midpoint_knot_for_index(knotvector, degree, m)
    if site is None:
        return None

    u_at, u_next, u_mid = site
    return m, u_at, u_next, u_mid


def batch_midpoint_knots_for_errors(
    knotvector: np.ndarray,
    degree: int,
    error_array: np.ndarray,
    *,
    batch_size: int,
    atol: float = 1e-9,
) -> list[tuple[int, float, float, float, float]]:
    """
    Rank knot indices by error (descending) and plan up to ``batch_size`` midpoints.

    Returns a list of ``(m, err_m, u_at, u_next, u_mid)`` with distinct ``u_mid``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    err = np.asarray(error_array, dtype=float)
    u = np.asarray(knotvector, dtype=float)
    if err.size == 0 or not np.any(err > 0):
        return []

    order = np.argsort(err)[::-1]
    planned: list[tuple[int, float, float, float, float]] = []
    seen_mids: list[float] = []

    for m in order:
        if len(planned) >= batch_size:
            break
        if err[m] <= 0:
            break

        site = midpoint_knot_for_index(u, degree, int(m))
        if site is None:
            continue

        u_at, u_next, u_mid = site
        if any(abs(u_mid - s) <= atol for s in seen_mids):
            continue
        if knot_already_present(u, u_mid, atol=atol):
            continue

        planned.append((int(m), float(err[m]), u_at, u_next, u_mid))
        seen_mids.append(u_mid)

    return planned


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


def insert_batch_knots_if_new(
    curve: BSpline.Curve,
    u_values: list[float],
    insert_multiplicity: int,
) -> list[float]:
    """
    Insert multiple midpoints; returns the subset actually inserted.

    Inserts from largest ``u`` to smallest so geomdl bookkeeping stays stable.
    """
    unique_sorted = sorted({float(v) for v in u_values}, reverse=True)
    inserted: list[float] = []
    for u_val in unique_sorted:
        if insert_single_knot_if_new(curve, u_val, insert_multiplicity):
            inserted.append(u_val)
    return inserted


def adaptive_error_refine(
    curve: BSpline.Curve,
    qw: np.ndarray,
    knotvector: np.ndarray,
    degree: int,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    insert_multiplicity: int = 2,
    batch_size: int = DEFAULT_BATCH_SIZE,
    weights: np.ndarray | None = None,
) -> list[AdaptiveIterationRecord]:
    """
    Iteratively refine ``curve`` in place using degree-reduction error feedback.

    ``batch_size=1`` recovers the original one-knot-per-outer-iteration behaviour.
    ``batch_size>1`` inserts up to that many distinct midpoint knots per outer
    iteration (one ``DegreeReduceCurve`` call per outer iteration).

    ``qw`` / ``knotvector`` are refreshed from ``curve`` after each batch.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if insert_multiplicity < 1:
        raise ValueError("insert_multiplicity must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    log: list[AdaptiveIterationRecord] = []
    qw_work = np.asarray(qw, dtype=float)
    u_work = np.asarray(knotvector, dtype=float)
    w_work = None if weights is None else np.asarray(weights, dtype=float)
    p = int(degree)

    mode = "batched" if batch_size > 1 else "serial"
    if batch_size > 1:
        print(f"adaptive refinement ({mode}, max {max_iterations} outer iters, batch={batch_size}):")
    else:
        print(f"adaptive refinement (max {max_iterations} iterations):")

    for it in range(max_iterations):
        n = qw_work.shape[0]
        out = degree_reduce_unified(n, p, u_work, qw_work, weights=w_work)
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
        max_err = float(np.max(err))
        max_m = int(np.argmax(err))

        if batch_size == 1:
            site = midpoint_knot_for_max_error(u_work, p, err)
            if site is None:
                log.append(
                    AdaptiveIterationRecord(
                        iteration=it,
                        max_index=max_m,
                        max_error=max_err,
                        u_at=float(u_work[max_m]),
                        u_next=None,
                        u_insert=None,
                        inserted=False,
                        note="no valid midpoint site",
                    )
                )
                print(f"  iter {it}: stop — no valid midpoint site")
                break

            m, u_at, u_next, u_mid = site
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
            qw_work, u_work, p, w_work = _curve_arrays(curve)
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
                    batch_inserts=(u_mid,),
                )
            )
            print(f"    inserted ×{insert_multiplicity}")
            continue

        planned = batch_midpoint_knots_for_errors(
            u_work, p, err, batch_size=batch_size
        )
        if not planned:
            log.append(
                AdaptiveIterationRecord(
                    iteration=it,
                    max_index=max_m,
                    max_error=max_err,
                    u_at=float(u_work[max_m]),
                    u_next=None,
                    u_insert=None,
                    inserted=False,
                    note="no valid batch midpoint sites",
                )
            )
            print(f"  iter {it}: stop — no valid batch midpoint sites")
            break

        m0, _, u_at0, u_next0, u_mid0 = planned[0]
        print(
            f"  iter {it}: max error {max_err:.6g} at index {max_m}; "
            f"batch {len(planned)} site(s):"
        )
        for m, err_m, u_at, u_next, u_mid in planned:
            print(
                f"    index {m}: err={err_m:.6g} U[{m}]={u_at:g} next={u_next:g} → {u_mid:g}"
            )

        inserted = insert_batch_knots_if_new(
            curve, [u_mid for *_, u_mid in planned], insert_multiplicity
        )
        qw_work, u_work, p, w_work = _curve_arrays(curve)

        if not inserted:
            log.append(
                AdaptiveIterationRecord(
                    iteration=it,
                    max_index=m0,
                    max_error=max_err,
                    u_at=u_at0,
                    u_next=u_next0,
                    u_insert=u_mid0,
                    inserted=False,
                    note="batch sites already present",
                )
            )
            print("    skip — all batch midpoints already present")
            continue

        log.append(
            AdaptiveIterationRecord(
                iteration=it,
                max_index=m0,
                max_error=max_err,
                u_at=u_at0,
                u_next=u_next0,
                u_insert=u_mid0,
                inserted=True,
                note=f"inserted {len(inserted)} knot(s) ×{insert_multiplicity}",
                batch_inserts=tuple(inserted),
            )
        )
        print(f"    inserted {len(inserted)} knot(s) ×{insert_multiplicity}: {inserted}")

    return log


def _curve_arrays(
    curve: BSpline.Curve,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray | None]:
    if isinstance(curve, NURBS.Curve):
        ctrl = np.asarray(curve.ctrlpts, dtype=float)
        qw = ctrl[:, :2] if ctrl.shape[1] >= 2 else ctrl
        u = np.asarray(curve.knotvector, dtype=float)
        weights = np.asarray(curve.weights, dtype=float)
        return qw, u, int(curve.degree), weights
    qw = np.asarray(curve.ctrlpts, dtype=float)[:, :2]
    u = np.asarray(curve.knotvector, dtype=float)
    return qw, u, int(curve.degree), None
