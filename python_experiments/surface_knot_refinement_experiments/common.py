"""Shared surface degree-reduction experiment utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from geomdl import BSpline, NURBS, operations

from bspline_surface_visualizer import build_bspline_surface
from knot_refinement_experiments.common import (
    insertions_for_knot,
    knot_multiplicity_at,
    knot_tag,
    plot_error_array_on_ax,
)
from nurbs_surface_degree_reduction import degree_reduce_surface_unified
from surface_error_visualization import (
    plot_surface_error_heatmap_on_ax,
    plot_surface_wireframe_on_ax,
)

DELTA = 0.04
KNOT_INSERT_MULTIPLICITY = 2


def build_geomdl_from_surface_geometry(
    control_points: np.ndarray,
    knotvector_u: np.ndarray,
    knotvector_v: np.ndarray,
    degree_u: int,
    degree_v: int,
    *,
    weights: np.ndarray | None = None,
    delta: float = DELTA,
    name: str = "",
) -> BSpline.Surface | NURBS.Surface:
    """Build a geomdl B-spline or NURBS surface from numpy geometry."""
    net = np.asarray(control_points, dtype=float)
    ku = np.asarray(knotvector_u, dtype=float)
    kv = np.asarray(knotvector_v, dtype=float)
    n_u, n_v = net.shape[0], net.shape[1]

    if weights is None:
        surf = build_bspline_surface(net, ku, kv, degree_u, degree_v, delta=delta)
        if name:
            surf.name = name
        return surf

    w = np.asarray(weights, dtype=float)
    if w.shape != (n_u, n_v):
        raise ValueError(f"weights must have shape ({n_u}, {n_v})")

    ctrlptsw: list[list[float]] = []
    for i in range(n_u):
        for j in range(n_v):
            wt = float(w[i, j])
            x, y, z = net[i, j]
            ctrlptsw.append([x * wt, y * wt, z * wt, wt])

    surf = NURBS.Surface()
    surf.degree_u = int(degree_u)
    surf.degree_v = int(degree_v)
    surf.set_ctrlpts(ctrlptsw, n_u, n_v)
    surf.knotvector_u = ku.tolist()
    surf.knotvector_v = kv.tolist()
    surf.delta = float(delta)
    if name:
        surf.name = name
    return surf


def geometry_from_geomdl_surface(
    surface: BSpline.Surface | NURBS.Surface,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray | None]:
    """``(Qw, U, V, p_u, p_v, weights)``."""
    ku = np.asarray(surface.knotvector_u, dtype=float)
    kv = np.asarray(surface.knotvector_v, dtype=float)
    p_u = int(surface.degree_u)
    p_v = int(surface.degree_v)
    n_u = int(surface.ctrlpts_size_u)
    n_v = int(surface.ctrlpts_size_v)
    raw = np.array(surface.ctrlpts2d, dtype=float)

    if isinstance(surface, NURBS.Surface):
        if raw.shape[-1] == 4:
            w = raw[:, :, 3]
            net = raw[:, :, :3] / w[:, :, np.newaxis]
            weights = w
        else:
            net = raw
            w_flat = np.asarray(surface.weights, dtype=float)
            weights = w_flat.reshape(n_u, n_v)
        return net, ku, kv, p_u, p_v, weights

    if raw.shape[-1] > 3:
        net = raw[:, :, :3]
    else:
        net = raw
    return net, ku, kv, p_u, p_v, None


def insert_knot_u(surface: BSpline.Surface | NURBS.Surface, u: float, mult: int) -> None:
    operations.insert_knot(surface, [float(u), None], [int(mult), 0])


def insert_knot_v(surface: BSpline.Surface | NURBS.Surface, v: float, mult: int) -> None:
    operations.insert_knot(surface, [None, float(v)], [0, int(mult)])


def refine_surface_knots_u(
    surface: BSpline.Surface | NURBS.Surface,
    knots: tuple[float, ...],
    *,
    default_multiplicity: int = KNOT_INSERT_MULTIPLICITY,
) -> list[tuple[float, int, int]]:
    """Insert knots along the u direction; returns ``(u, existing_mult, n_inserted)`` log."""
    log: list[tuple[float, int, int]] = []
    for u in knots:
        u_val = float(u)
        kv = np.asarray(surface.knotvector_u, dtype=float)
        existing = knot_multiplicity_at(kv, u_val)
        n_insert = insertions_for_knot(kv, u_val, default_multiplicity)
        insert_knot_u(surface, u_val, n_insert)
        log.append((u_val, existing, n_insert))
    return log


def refine_surface_knots_v(
    surface: BSpline.Surface | NURBS.Surface,
    knots: tuple[float, ...],
    *,
    default_multiplicity: int = KNOT_INSERT_MULTIPLICITY,
) -> list[tuple[float, int, int]]:
    """Insert knots along the v direction."""
    log: list[tuple[float, int, int]] = []
    for v in knots:
        v_val = float(v)
        kv = np.asarray(surface.knotvector_v, dtype=float)
        existing = knot_multiplicity_at(kv, v_val)
        n_insert = insertions_for_knot(kv, v_val, default_multiplicity)
        insert_knot_v(surface, v_val, n_insert)
        log.append((v_val, existing, n_insert))
    return log


def degree_reduce_surface_or_exit(
    n_u: int,
    n_v: int,
    p_u: int,
    p_v: int,
    u: np.ndarray,
    v: np.ndarray,
    qw: np.ndarray,
    *,
    label: str,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """One-step surface reduction ``(p_u, p_v) -> (p_u-1, p_v-1)``."""
    out = degree_reduce_surface_unified(
        n_u, n_v, p_u, p_v, u, v, qw, weights=weights
    )
    if out == 1:
        raise SystemExit(f"tolerance exceeded ({label})")

    r, uh, vh, err_u, err_v, w_out = out
    print(f"--- {label} ---")
    print(f"R shape: {r.shape}")
    print("Uh:", uh)
    print("Vh:", vh)
    print("error_u:", err_u)
    print("error_v:", err_v)
    if np.any(err_u > 0):
        print("error_u (nonzero):", err_u[err_u > 0])
    if np.any(err_v > 0):
        print("error_v (nonzero):", err_v[err_v > 0])
    return r, uh, vh, err_u, err_v, w_out


def _make_plot_surface(
    qw: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p_u: int,
    p_v: int,
    weights: np.ndarray | None,
) -> BSpline.Surface | NURBS.Surface:
    return build_geomdl_from_surface_geometry(
        qw, u, v, p_u, p_v, weights=weights, delta=DELTA
    )


def render_surface_comparison_figure(
    *,
    qw_orig: np.ndarray,
    u_orig: np.ndarray,
    v_orig: np.ndarray,
    p_u: int,
    p_v: int,
    err_u_orig: np.ndarray,
    err_v_orig: np.ndarray,
    qw_ref: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
    p_u_ref: int,
    p_v_ref: int,
    err_u_ref: np.ndarray,
    err_v_ref: np.ndarray,
    save_path: Path,
    show: bool,
    weights_orig: np.ndarray | None = None,
    weights_ref: np.ndarray | None = None,
    suptitle: str | None = None,
) -> None:
    """
    3-row comparison: wireframe | error heatmap | knot-index error plots.

    Columns: original geometry | refined geometry.
    """
    surf_orig = _make_plot_surface(qw_orig, u_orig, v_orig, p_u, p_v, weights_orig)
    surf_ref = _make_plot_surface(qw_ref, u_ref, v_ref, p_u_ref, p_v_ref, weights_ref)

    fig = plt.figure(figsize=(16.0, 14.0), dpi=150)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.4, 0.9], hspace=0.35, wspace=0.15)

    ax_w0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax_w1 = fig.add_subplot(gs[0, 1], projection="3d")
    ax_h0 = fig.add_subplot(gs[1, 0], projection="3d")
    ax_h1 = fig.add_subplot(gs[1, 1], projection="3d")

    gs_err = gs[2, :].subgridspec(1, 4, wspace=0.25)
    ax_eu0 = fig.add_subplot(gs_err[0, 0])
    ax_ev0 = fig.add_subplot(gs_err[0, 1])
    ax_eu1 = fig.add_subplot(gs_err[0, 2])
    ax_ev1 = fig.add_subplot(gs_err[0, 3])

    plot_surface_wireframe_on_ax(
        ax_w0, surf_orig, title=f"original ({qw_orig.shape[0]}×{qw_orig.shape[1]} CPs)"
    )
    plot_surface_wireframe_on_ax(
        ax_w1, surf_ref, title=f"refined ({qw_ref.shape[0]}×{qw_ref.shape[1]} CPs)"
    )

    m0 = plot_surface_error_heatmap_on_ax(
        ax_h0,
        surf_orig,
        u_orig,
        v_orig,
        p_u,
        p_v,
        err_u_orig,
        err_v_orig,
        title="A5.11 error heatmap (original)",
    )
    m1 = plot_surface_error_heatmap_on_ax(
        ax_h1,
        surf_ref,
        u_ref,
        v_ref,
        p_u_ref,
        p_v_ref,
        err_u_ref,
        err_v_ref,
        title="A5.11 error heatmap (refined)",
    )

    plot_error_array_on_ax(ax_eu0, err_u_orig, u_orig, title="error_u (original)")
    plot_error_array_on_ax(ax_ev0, err_v_orig, v_orig, title="error_v (original)")
    plot_error_array_on_ax(ax_eu1, err_u_ref, u_ref, title="error_u (refined)")
    plot_error_array_on_ax(ax_ev1, err_v_ref, v_ref, title="error_v (refined)")

    fig.colorbar(m0, ax=[ax_h0], fraction=0.035, pad=0.02, label="A5.11 bound")
    fig.colorbar(m1, ax=[ax_h1], fraction=0.035, pad=0.02, label="A5.11 bound")

    title = suptitle or f"p{p_u}p{p_v} → p{p_u - 1}p{p_v - 1} surface reduction"
    fig.suptitle(title, fontsize=13, y=0.98)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


__all__ = [
    "DELTA",
    "KNOT_INSERT_MULTIPLICITY",
    "build_geomdl_from_surface_geometry",
    "degree_reduce_surface_or_exit",
    "geometry_from_geomdl_surface",
    "insert_knot_u",
    "insert_knot_v",
    "knot_tag",
    "refine_surface_knots_u",
    "refine_surface_knots_v",
    "render_surface_comparison_figure",
]
