"""
Curvature-adaptive knot refinement.

Sample curvature κ(u) and cumulative arc length s(u) via geomdl
``Curve.derivatives``. Map κ to local spacing

    ``h = clamp(h_max / (1 + α κ^β), h_min, h_max)``

then build a candidate knot grid by marching either:

- **arc** (default): step in integrated arc length ``s``, map ``s → u``
- **u**: step directly in parameter ``u``

Existing knots are left untouched.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from geomdl import BSpline

from .knot_refinement_common import new_knots_from_values, unique_active_knots

DEFAULT_H_MIN = 0.05
DEFAULT_H_MAX = 0.1
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0
DEFAULT_N_SAMPLES = 800
DEFAULT_MARCH_MODE = "arc"

MarchMode = Literal["arc", "u"]


def curve_sample_profile(
    curve: BSpline.Curve,
    u_min: float,
    u_max: float,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample ``(u, s(u), κ(u))`` on ``[u_min, u_max]`` using geomdl derivatives.

    Arc length is integrated along the polyline through evaluated points:

    ``s(u_i) = s(u_{i-1}) + ||C(u_i) - C(u_{i-1})||``

    Curvature uses the standard plane-curve formula from first/second derivatives.
    """
    if n_samples < 3:
        raise ValueError("n_samples must be >= 3")

    u = np.linspace(float(u_min), float(u_max), int(n_samples))
    s = np.zeros(u.size, dtype=float)
    kappa = np.zeros(u.size, dtype=float)
    speed_floor = 1e-12

    x_prev = y_prev = None
    for i, ui in enumerate(u):
        ders = curve.derivatives(float(ui), order=2)
        x, y, _ = ders[0]
        xp, yp, _ = ders[1]
        xpp, ypp, _ = ders[2]

        if x_prev is not None:
            s[i] = s[i - 1] + float(np.hypot(x - x_prev, y - y_prev))

        speed_sq = xp * xp + yp * yp
        if speed_sq >= speed_floor:
            kappa[i] = abs(xp * ypp - yp * xpp) / (speed_sq ** 1.5)

        x_prev, y_prev = x, y

    return u, s, kappa


def spacing_from_kappa(
    kappa: np.ndarray,
    *,
    h_min: float,
    h_max: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Map curvature samples to local spacing (``Δu`` or ``ds``, same units as
    ``h_min`` / ``h_max``).
    """
    if h_min <= 0 or h_max <= 0:
        raise ValueError("h_min and h_max must be positive")
    if h_min > h_max:
        raise ValueError("h_min must be <= h_max")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    kappa_max = float(np.max(kappa))
    if kappa_max <= 0:
        return np.full(kappa.shape, float(h_max))

    k_norm = kappa / kappa_max
    ds = h_max / (1.0 + alpha * np.power(k_norm, beta))
    return np.clip(ds, float(h_min), float(h_max))


def adaptive_grid_arc_length(
    u_samples: np.ndarray,
    s_samples: np.ndarray,
    ds_samples: np.ndarray,
    u_end: float,
    *,
    tol: float = 1e-9,
) -> list[float]:
    """
    Build a knot grid by marching in arc length, then map ``s → u``.

    ``ds_samples`` is the desired local arc-length step (interpolated in ``s``).
    """
    u_start = float(u_samples[0])
    u_stop = float(u_end)
    s_stop = float(s_samples[-1])

    if s_stop <= tol:
        return [u_start, u_stop]

    grid_u: list[float] = [u_start]
    s = 0.0

    while s < s_stop - tol:
        ds = float(np.interp(s, s_samples, ds_samples))
        s_next = min(s + ds, s_stop)
        if s_next <= s + tol:
            break
        u_next = float(np.interp(s_next, s_samples, u_samples))
        u_next = min(max(u_next, u_start), u_stop)
        grid_u.append(u_next)
        s = s_next

    if abs(grid_u[-1] - u_stop) > tol:
        grid_u.append(u_stop)

    return grid_u


def adaptive_grid_parameter(
    u_samples: np.ndarray,
    du_samples: np.ndarray,
    u_end: float,
    *,
    tol: float = 1e-9,
) -> list[float]:
    """
    Build a knot grid by marching in parameter space.

    ``du_samples`` is the desired local ``Δu`` step (interpolated in ``u``).
    """
    u_start = float(u_samples[0])
    u_stop = float(u_end)

    grid_u: list[float] = [u_start]
    u = u_start

    while u < u_stop - tol:
        du = float(np.interp(u, u_samples, du_samples))
        u_next = min(u + du, u_stop)
        if u_next <= u + tol:
            break
        grid_u.append(u_next)
        u = u_next

    if abs(grid_u[-1] - u_stop) > tol:
        grid_u.append(u_stop)

    return grid_u


def curvature_new_knots(
    curve: BSpline.Curve,
    knotvector: np.ndarray,
    degree: int,
    *,
    h_min: float = DEFAULT_H_MIN,
    h_max: float = DEFAULT_H_MAX,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    n_samples: int = DEFAULT_N_SAMPLES,
    march_mode: MarchMode = DEFAULT_MARCH_MODE,
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Plan curvature-adaptive knot insertion.

    Returns
    -------
    new_knots : list[float]
    u_samples, s_samples, kappa, spacing_samples : np.ndarray
        ``spacing_samples`` is ``ds`` (arc mode) or ``Δu`` (u mode).
    grid : list[float]
        Full non-uniform candidate grid in parameter space.
    """
    if march_mode not in ("arc", "u"):
        raise ValueError(f"march_mode must be 'arc' or 'u', got {march_mode!r}")

    unique = unique_active_knots(knotvector, degree)
    u_min = float(unique[0])
    u_max = float(unique[-1])

    u_samples, s_samples, kappa = curve_sample_profile(
        curve, u_min, u_max, n_samples=n_samples
    )

    if march_mode == "u":
        spacing_samples = spacing_from_kappa(
            kappa, h_min=h_min, h_max=h_max, alpha=alpha, beta=beta
        )
        grid = adaptive_grid_parameter(u_samples, spacing_samples, u_max)
    else:
        # ``h_min`` / ``h_max`` are spacings on [u_min, u_max]; convert to arc
        # length via mean speed ds/du.
        u_span = u_max - u_min
        s_total = float(s_samples[-1])
        speed_scale = s_total / u_span if u_span > 1e-12 else 1.0
        ds_min = float(h_min) * speed_scale
        ds_max = float(h_max) * speed_scale

        spacing_samples = spacing_from_kappa(
            kappa, h_min=ds_min, h_max=ds_max, alpha=alpha, beta=beta
        )
        grid = adaptive_grid_arc_length(u_samples, s_samples, spacing_samples, u_max)

    new_knots = new_knots_from_values(knotvector, grid)

    return new_knots, u_samples, s_samples, kappa, spacing_samples, grid
