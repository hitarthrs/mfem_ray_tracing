"""
Curvature-guided adaptive knot refinement.

Curvature is sampled each outer iteration but **never inserts knots**. It guides
which spans to refine and (optionally) where inside a span to insert.

Guide modes
-----------
multiplicative (default)
    ``score = err * (1 + α (κ/κ_max)^β)``
additive
    ``score = err + α * max(err) * (κ/κ_max)^β``
tiebreaker
    Rank by error buckets of width ``ε * max(err)``; within a bucket prefer
    higher ``κ``.
biased_site
    Rank by plain ``err``; insert at ``κ`` peak inside the span (not midpoint).
regional_budget
    Partition ``[u_min, u_max]`` into curvature-based zones; allocate batch
    slots per zone by mean ``κ``; pick top-error spans in each zone.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from geomdl import BSpline, NURBS

from nurbs_degree_reduction import degree_reduce_unified
from .adaptive_error_knot_refinement import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_ITERATIONS,
    AdaptiveIterationRecord,
    batch_midpoint_knots_for_errors,
    insert_batch_knots_if_new,
    insert_single_knot_if_new,
    midpoint_knot_for_index,
    midpoint_knot_for_max_error,
)
from .curvature_knot_refinement import (
    DEFAULT_H_MAX,
    DEFAULT_H_MIN,
    DEFAULT_N_SAMPLES,
    adaptive_grid_parameter,
    curve_sample_profile,
    spacing_from_kappa,
)
from .knot_refinement_common import knot_already_present, unique_active_knots

GuideMode = Literal[
    "multiplicative",
    "additive",
    "tiebreaker",
    "biased_site",
    "regional_budget",
]

DEFAULT_CURVATURE_ALPHA = 1.0
DEFAULT_CURVATURE_BETA = 1.0
DEFAULT_GUIDE_MODE: GuideMode = "multiplicative"
DEFAULT_TIE_EPSILON = 0.05


def span_max_curvature(
    knotvector: np.ndarray,
    degree: int,
    u_samples: np.ndarray,
    kappa: np.ndarray,
    *,
    atol: float = 1e-9,
) -> np.ndarray:
    """Maximum κ on each knot-index span ``[U[m], next_unique(U[m])]``."""
    u = np.asarray(knotvector, dtype=float)
    u_s = np.asarray(u_samples, dtype=float)
    k = np.asarray(kappa, dtype=float)
    out = np.zeros(u.size, dtype=float)

    for m in range(u.size):
        site = midpoint_knot_for_index(u, degree, m)
        if site is None:
            continue
        u_at, u_next, _ = site
        mask = (u_s >= float(u_at) - atol) & (u_s <= float(u_next) + atol)
        if np.any(mask):
            out[m] = float(np.max(k[mask]))

    return out


def normalized_span_kappa(
    span_kappa: np.ndarray,
    *,
    curvature_beta: float,
) -> np.ndarray:
    kappa = np.asarray(span_kappa, dtype=float)
    k_max = float(np.max(kappa))
    if k_max <= 0:
        return np.zeros_like(kappa)
    return np.power(kappa / k_max, float(curvature_beta))


def curvature_guided_scores(
    error_array: np.ndarray,
    span_kappa: np.ndarray,
    *,
    guide_mode: GuideMode,
    curvature_alpha: float,
    curvature_beta: float,
    tie_epsilon: float,
) -> np.ndarray:
    """Span scores for ranking (``biased_site`` / ``regional_budget`` → plain error)."""
    err = np.asarray(error_array, dtype=float)
    if guide_mode in ("biased_site", "regional_budget"):
        return err.copy()

    k_norm = normalized_span_kappa(span_kappa, curvature_beta=curvature_beta)
    max_err = float(np.max(err))

    if guide_mode == "multiplicative":
        if curvature_alpha <= 0:
            return err.copy()
        return err * (1.0 + float(curvature_alpha) * k_norm)

    if guide_mode == "additive":
        if curvature_alpha <= 0 or max_err <= 0:
            return err.copy()
        return err + float(curvature_alpha) * max_err * k_norm

    if guide_mode == "tiebreaker":
        if tie_epsilon <= 0 or max_err <= 0:
            return err.copy()
        band = float(tie_epsilon) * max_err
        if band <= 0:
            return err.copy()
        bucket = np.floor(err / band)
        # κ tie-breaks within a bucket; buckets dominate across bands.
        return bucket * max_err + k_norm

    raise ValueError(f"unknown guide_mode: {guide_mode!r}")


def insertion_site_for_index(
    knotvector: np.ndarray,
    degree: int,
    index: int,
    u_samples: np.ndarray,
    kappa: np.ndarray,
    *,
    guide_mode: GuideMode,
    atol: float = 1e-9,
) -> tuple[float, float, float] | None:
    """
    Insertion site for knot index ``index``.

    ``biased_site`` uses the ``κ`` peak in the span; other modes use the midpoint.
    """
    site = midpoint_knot_for_index(knotvector, degree, index)
    if site is None:
        return None

    u_at, u_next, u_mid = site
    if guide_mode != "biased_site":
        return u_at, u_next, u_mid

    u_s = np.asarray(u_samples, dtype=float)
    k = np.asarray(kappa, dtype=float)
    mask = (u_s >= float(u_at) - atol) & (u_s <= float(u_next) + atol)
    if not np.any(mask):
        return u_at, u_next, u_mid

    local_u = u_s[mask]
    local_k = k[mask]
    u_insert = float(local_u[int(np.argmax(local_k))])
    return u_at, u_next, u_insert


def batch_sites_for_scores(
    knotvector: np.ndarray,
    degree: int,
    error_array: np.ndarray,
    score_array: np.ndarray,
    u_samples: np.ndarray,
    kappa: np.ndarray,
    *,
    guide_mode: GuideMode,
    batch_size: int,
    atol: float = 1e-9,
) -> list[tuple[int, float, float, float, float, float]]:
    """
    Plan batched insertion sites ranked by ``score_array``.

    Returns ``(m, err_m, score_m, u_at, u_next, u_insert)``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    err = np.asarray(error_array, dtype=float)
    scores = np.asarray(score_array, dtype=float)
    u = np.asarray(knotvector, dtype=float)
    if err.size == 0 or not np.any(err > 0):
        return []

    order = np.argsort(scores)[::-1]
    planned: list[tuple[int, float, float, float, float, float]] = []
    seen_inserts: list[float] = []

    for m in order:
        if len(planned) >= batch_size:
            break
        if err[m] <= 0:
            break

        site = insertion_site_for_index(
            u, degree, int(m), u_samples, kappa, guide_mode=guide_mode, atol=atol
        )
        if site is None:
            continue

        u_at, u_next, u_insert = site
        if any(abs(u_insert - s) <= atol for s in seen_inserts):
            continue
        if knot_already_present(u, u_insert, atol=atol):
            continue

        planned.append(
            (int(m), float(err[m]), float(scores[m]), u_at, u_next, u_insert)
        )
        seen_inserts.append(u_insert)

    return planned


def top_score_site(
    knotvector: np.ndarray,
    degree: int,
    score_array: np.ndarray,
    u_samples: np.ndarray,
    kappa: np.ndarray,
    *,
    guide_mode: GuideMode,
) -> tuple[int, float, float, float] | None:
    """Insertion site for the highest-scoring knot index."""
    scores = np.asarray(score_array, dtype=float)
    if scores.size == 0 or not np.any(scores > 0):
        return None

    m = int(np.argmax(scores))
    site = insertion_site_for_index(
        knotvector, degree, m, u_samples, kappa, guide_mode=guide_mode
    )
    if site is None:
        return None

    u_at, u_next, u_insert = site
    return m, u_at, u_next, u_insert


def curvature_zone_boundaries(
    u_samples: np.ndarray,
    kappa: np.ndarray,
    u_end: float,
    *,
    h_min: float,
    h_max: float,
    spacing_alpha: float,
    spacing_beta: float,
) -> list[tuple[float, float]]:
    """Zone intervals from a virtual curvature spacing grid (no insertion)."""
    spacing = spacing_from_kappa(
        kappa,
        h_min=h_min,
        h_max=h_max,
        alpha=spacing_alpha,
        beta=spacing_beta,
    )
    grid = adaptive_grid_parameter(u_samples, spacing, u_end)
    if len(grid) < 2:
        return [(float(grid[0]), float(u_end))]

    zones: list[tuple[float, float]] = []
    for i in range(len(grid) - 1):
        z_lo, z_hi = float(grid[i]), float(grid[i + 1])
        if z_hi > z_lo:
            zones.append((z_lo, z_hi))
    return zones


def zone_mean_kappa(
    u_samples: np.ndarray,
    kappa: np.ndarray,
    z_lo: float,
    z_hi: float,
    *,
    atol: float = 1e-9,
) -> float:
    mask = (u_samples >= z_lo - atol) & (u_samples <= z_hi + atol)
    if not np.any(mask):
        return 0.0
    return float(np.mean(kappa[mask]))


def span_midpoint_u(
    knotvector: np.ndarray,
    degree: int,
    index: int,
) -> float | None:
    site = midpoint_knot_for_index(knotvector, degree, index)
    if site is None:
        return None
    return float(site[2])


def allocate_zone_quotas(
    zone_weights: list[float],
    batch_size: int,
) -> list[int]:
    """Distribute ``batch_size`` slots across zones proportional to weight."""
    if not zone_weights:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    total = float(sum(zone_weights))
    if total <= 0:
        base = batch_size // len(zone_weights)
        rem = batch_size % len(zone_weights)
        return [base + (1 if i < rem else 0) for i in range(len(zone_weights))]

    raw = [batch_size * w / total for w in zone_weights]
    quotas = [int(np.floor(r)) for r in raw]
    remaining = batch_size - sum(quotas)
    order = np.argsort([r - q for r, q in zip(raw, quotas)])[::-1]
    for i in order[:remaining]:
        quotas[int(i)] += 1
    return quotas


def batch_sites_regional_budget(
    knotvector: np.ndarray,
    degree: int,
    error_array: np.ndarray,
    u_samples: np.ndarray,
    kappa: np.ndarray,
    zones: list[tuple[float, float]],
    *,
    guide_mode: GuideMode,
    batch_size: int,
    atol: float = 1e-9,
) -> list[tuple[int, float, float, float, float, float]]:
    """
    Plan batched sites with per-zone error-ranked quotas.

    Spans are assigned to the zone containing their insertion parameter.
    """
    err = np.asarray(error_array, dtype=float)
    u = np.asarray(knotvector, dtype=float)
    if err.size == 0 or not np.any(err > 0) or not zones:
        return []

    zone_weights = [
        zone_mean_kappa(u_samples, kappa, z_lo, z_hi, atol=atol)
        for z_lo, z_hi in zones
    ]
    quotas = allocate_zone_quotas(zone_weights, batch_size)

    zone_candidates: list[list[int]] = [[] for _ in zones]
    for m in range(err.size):
        if err[m] <= 0:
            continue
        site = insertion_site_for_index(
            u, degree, m, u_samples, kappa, guide_mode=guide_mode, atol=atol
        )
        if site is None:
            continue
        u_insert = site[2]
        for zi, (z_lo, z_hi) in enumerate(zones):
            if z_lo - atol <= u_insert <= z_hi + atol:
                zone_candidates[zi].append(m)
                break

    planned: list[tuple[int, float, float, float, float, float]] = []
    seen_inserts: list[float] = []

    zone_order = np.argsort(zone_weights)[::-1]
    for zi in zone_order:
        quota = quotas[int(zi)]
        if quota <= 0:
            continue
        candidates = sorted(
            zone_candidates[int(zi)],
            key=lambda m: float(err[m]),
            reverse=True,
        )
        for m in candidates:
            if len(planned) >= batch_size or quota <= 0:
                break
            site = insertion_site_for_index(
                u, degree, m, u_samples, kappa, guide_mode=guide_mode, atol=atol
            )
            if site is None:
                continue
            u_at, u_next, u_insert = site
            if any(abs(u_insert - s) <= atol for s in seen_inserts):
                continue
            if knot_already_present(u, u_insert, atol=atol):
                continue
            planned.append(
                (int(m), float(err[m]), float(err[m]), u_at, u_next, u_insert)
            )
            seen_inserts.append(u_insert)
            quota -= 1

    return planned


def serial_site_regional_budget(
    knotvector: np.ndarray,
    degree: int,
    error_array: np.ndarray,
    u_samples: np.ndarray,
    kappa: np.ndarray,
    zones: list[tuple[float, float]],
    *,
    guide_mode: GuideMode,
    atol: float = 1e-9,
) -> tuple[int, float, float, float] | None:
    """Best error span in the highest-κ zone that still has a valid site."""
    err = np.asarray(error_array, dtype=float)
    u = np.asarray(knotvector, dtype=float)
    if err.size == 0 or not np.any(err > 0) or not zones:
        return None

    zone_weights = [
        zone_mean_kappa(u_samples, kappa, z_lo, z_hi, atol=atol)
        for z_lo, z_hi in zones
    ]
    zone_order = np.argsort(zone_weights)[::-1]

    for zi in zone_order:
        z_lo, z_hi = zones[int(zi)]
        candidates: list[int] = []
        for m in range(err.size):
            if err[m] <= 0:
                continue
            site = insertion_site_for_index(
                u, degree, m, u_samples, kappa, guide_mode=guide_mode, atol=atol
            )
            if site is None:
                continue
            u_insert = site[2]
            if z_lo - atol <= u_insert <= z_hi + atol:
                candidates.append(m)
        if not candidates:
            continue
        m = max(candidates, key=lambda idx: float(err[idx]))
        site = insertion_site_for_index(
            u, degree, m, u_samples, kappa, guide_mode=guide_mode, atol=atol
        )
        if site is None:
            continue
        u_at, u_next, u_insert = site
        if knot_already_present(u, u_insert, atol=atol):
            continue
        return int(m), u_at, u_next, u_insert

    return None


def _guide_mode_label(
    guide_mode: GuideMode,
    *,
    curvature_alpha: float,
    curvature_beta: float,
    tie_epsilon: float,
    h_min: float,
    h_max: float,
) -> str:
    if guide_mode == "multiplicative":
        return f"multiplicative α={curvature_alpha:g} β={curvature_beta:g}"
    if guide_mode == "additive":
        return f"additive α={curvature_alpha:g} β={curvature_beta:g}"
    if guide_mode == "tiebreaker":
        return f"tiebreaker ε={tie_epsilon:g} β={curvature_beta:g}"
    if guide_mode == "biased_site":
        return "biased_site (error rank, κ-peak insert)"
    if guide_mode == "regional_budget":
        return (
            f"regional_budget h=[{h_min:g},{h_max:g}] "
            f"spacing_α={curvature_alpha:g} β={curvature_beta:g}"
        )
    return guide_mode


def curvature_guided_adaptive_refine(
    curve: BSpline.Curve,
    qw: np.ndarray,
    knotvector: np.ndarray,
    degree: int,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    insert_multiplicity: int = 2,
    batch_size: int = DEFAULT_BATCH_SIZE,
    guide_mode: GuideMode = DEFAULT_GUIDE_MODE,
    curvature_alpha: float = DEFAULT_CURVATURE_ALPHA,
    curvature_beta: float = DEFAULT_CURVATURE_BETA,
    tie_epsilon: float = DEFAULT_TIE_EPSILON,
    n_samples: int = DEFAULT_N_SAMPLES,
    h_min: float = DEFAULT_H_MIN,
    h_max: float = DEFAULT_H_MAX,
    weights: np.ndarray | None = None,
) -> list[AdaptiveIterationRecord]:
    """
    Adaptive refinement with curvature guidance (no curvature knot insertion).

    See module docstring for ``guide_mode`` options.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if insert_multiplicity < 1:
        raise ValueError("insert_multiplicity must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if n_samples < 3:
        raise ValueError("n_samples must be >= 3")
    if curvature_beta < 0:
        raise ValueError("curvature_beta must be non-negative")
    if tie_epsilon < 0:
        raise ValueError("tie_epsilon must be non-negative")
    if h_min <= 0 or h_max <= 0 or h_min > h_max:
        raise ValueError("require 0 < h_min <= h_max")

    log: list[AdaptiveIterationRecord] = []
    qw_work = np.asarray(qw, dtype=float)
    u_work = np.asarray(knotvector, dtype=float)
    w_work = None if weights is None else np.asarray(weights, dtype=float)
    p = int(degree)

    use_error_only_ranking = (
        guide_mode in ("biased_site", "regional_budget")
        or (
            guide_mode in ("multiplicative", "additive")
            and curvature_alpha <= 0
        )
        or (guide_mode == "tiebreaker" and tie_epsilon <= 0)
    )

    mode = f"curvature-guided/{guide_mode}"
    if batch_size > 1:
        mode += f" batch={batch_size}"
    print(
        f"{mode} refinement (max {max_iterations} outer iters, "
        f"{_guide_mode_label(guide_mode, curvature_alpha=curvature_alpha, curvature_beta=curvature_beta, tie_epsilon=tie_epsilon, h_min=h_min, h_max=h_max)}, "
        f"n_samples={n_samples}):"
    )

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

        unique = unique_active_knots(u_work, p)
        u_min = float(unique[0])
        u_max = float(unique[-1])
        u_samples, _s_samples, kappa = curve_sample_profile(
            curve, u_min, u_max, n_samples=n_samples
        )
        span_kappa = span_max_curvature(u_work, p, u_samples, kappa)
        scores = curvature_guided_scores(
            err,
            span_kappa,
            guide_mode=guide_mode,
            curvature_alpha=curvature_alpha,
            curvature_beta=curvature_beta,
            tie_epsilon=tie_epsilon,
        )
        guided_m = int(np.argmax(scores))
        guided_score = float(scores[guided_m])

        zones: list[tuple[float, float]] = []
        if guide_mode == "regional_budget":
            zones = curvature_zone_boundaries(
                u_samples,
                kappa,
                u_max,
                h_min=h_min,
                h_max=h_max,
                spacing_alpha=curvature_alpha,
                spacing_beta=curvature_beta,
            )

        if batch_size == 1:
            if guide_mode == "regional_budget":
                site = serial_site_regional_budget(
                    u_work,
                    p,
                    err,
                    u_samples,
                    kappa,
                    zones,
                    guide_mode="biased_site",
                )
            elif use_error_only_ranking and guide_mode != "biased_site":
                site = midpoint_knot_for_max_error(u_work, p, err)
            else:
                site = top_score_site(
                    u_work,
                    p,
                    scores,
                    u_samples,
                    kappa,
                    guide_mode=guide_mode,
                )

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
                        note="no valid insertion site",
                    )
                )
                print(f"  iter {it}: stop — no valid insertion site")
                break

            m, u_at, u_next, u_insert = site
            site_tag = "midpoint" if guide_mode != "biased_site" else "κ-peak"
            if use_error_only_ranking and guide_mode not in ("biased_site", "regional_budget"):
                print(
                    f"  iter {it}: max error {max_err:.6g} at index {m} "
                    f"(U[{m}]={u_at:g}, next={u_next:g}) → insert at {u_insert:g}"
                )
            elif guide_mode == "regional_budget":
                print(
                    f"  iter {it}: max error {max_err:.6g} at index {max_m}; "
                    f"regional pick index {m} err={float(err[m]):.6g} "
                    f"κ_span={span_kappa[m]:.6g} → {u_insert:g} ({len(zones)} zones)"
                )
            else:
                print(
                    f"  iter {it}: max error {max_err:.6g} at index {max_m}; "
                    f"guided pick index {m} score={guided_score:.6g} "
                    f"(err={float(err[m]):.6g}, κ_span={span_kappa[m]:.6g}) "
                    f"→ {u_insert:g} ({site_tag})"
                )

            if knot_already_present(u_work, u_insert):
                log.append(
                    AdaptiveIterationRecord(
                        iteration=it,
                        max_index=m,
                        max_error=max_err,
                        u_at=u_at,
                        u_next=u_next,
                        u_insert=u_insert,
                        inserted=False,
                        note="insert site already in knot vector",
                    )
                )
                print(f"    skip — {u_insert:g} already present")
                continue

            insert_single_knot_if_new(curve, u_insert, insert_multiplicity)
            qw_work, u_work, p, w_work = _curve_arrays(curve)
            log.append(
                AdaptiveIterationRecord(
                    iteration=it,
                    max_index=m,
                    max_error=max_err,
                    u_at=u_at,
                    u_next=u_next,
                    u_insert=u_insert,
                    inserted=True,
                    note=f"inserted ×{insert_multiplicity}",
                    batch_inserts=(u_insert,),
                )
            )
            print(f"    inserted ×{insert_multiplicity}")
            continue

        if guide_mode == "regional_budget":
            planned = batch_sites_regional_budget(
                u_work,
                p,
                err,
                u_samples,
                kappa,
                zones,
                guide_mode="biased_site",
                batch_size=batch_size,
            )
        elif use_error_only_ranking and guide_mode != "biased_site":
            planned_simple = batch_midpoint_knots_for_errors(
                u_work, p, err, batch_size=batch_size
            )
            planned = [
                (m, err_m, err_m, u_at, u_next, u_mid)
                for m, err_m, u_at, u_next, u_mid in planned_simple
            ]
        else:
            planned = batch_sites_for_scores(
                u_work,
                p,
                err,
                scores,
                u_samples,
                kappa,
                guide_mode=guide_mode,
                batch_size=batch_size,
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
                    note="no valid batch insertion sites",
                )
            )
            print(f"  iter {it}: stop — no valid batch insertion sites")
            break

        m0, _, _, u_at0, u_next0, u_insert0 = planned[0]
        header = (
            f"  iter {it}: max error {max_err:.6g} at index {max_m}; "
            f"batch {len(planned)} site(s):"
        )
        if guide_mode == "regional_budget":
            header = (
                f"  iter {it}: max error {max_err:.6g} at index {max_m}; "
                f"regional batch {len(planned)} site(s) ({len(zones)} zones):"
            )
        elif not use_error_only_ranking or guide_mode == "biased_site":
            header = (
                f"  iter {it}: max error {max_err:.6g} at index {max_m} "
                f"(top score index {guided_m}, score={guided_score:.6g}); "
                f"batch {len(planned)} site(s):"
            )
        print(header)
        for m, err_m, score_m, u_at, u_next, u_insert in planned:
            extra = ""
            if guide_mode == "biased_site":
                extra = " κ-peak"
            print(
                f"    index {m}: score={score_m:.6g} err={err_m:.6g} "
                f"κ_span={span_kappa[m]:.6g} U[{m}]={u_at:g} next={u_next:g} "
                f"→ {u_insert:g}{extra}"
            )

        inserted = insert_batch_knots_if_new(
            curve, [u_ins for *_, u_ins in planned], insert_multiplicity
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
                    u_insert=u_insert0,
                    inserted=False,
                    note="batch sites already present",
                )
            )
            print("    skip — all batch sites already present")
            continue

        log.append(
            AdaptiveIterationRecord(
                iteration=it,
                max_index=m0,
                max_error=max_err,
                u_at=u_at0,
                u_next=u_next0,
                u_insert=u_insert0,
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
