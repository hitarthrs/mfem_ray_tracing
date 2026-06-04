"""Visualize B-spline / NURBS surfaces with geomdl.

Interactive 3D: geomdl ``VisMPL`` (rotate/zoom in a matplotlib window).
Static breakdown: 2D projection, iso-u / iso-v curves, knot spans → ``outputs/*``.

Setup:
    cd python_faffing && .venv/bin/pip install -r requirements.txt

Demo:
    .venv/bin/python bspline_surface_visualizer.py
    .venv/bin/python bspline_surface_visualizer.py --demo s-ribbon --vismpl-style wireframe
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from geomdl import BSpline
from geomdl.visualization import VisMPL
from matplotlib import cm

from visualize_bspline_curve import (
    open_uniform_knots,
    s_shaped_control_points,
    unique_knot_spans,
)


def build_bspline_surface(
    control_points: np.ndarray,
    knotvector_u: np.ndarray,
    knotvector_v: np.ndarray,
    degree_u: int,
    degree_v: int,
    *,
    delta: float = 0.05,
) -> BSpline.Surface:
    """
    Build a ``geomdl.BSpline.Surface`` from a 2D control net ``control_points[u, v, :]``.

    Each control point is ``(x, y)`` or ``(x, y, z)``; 2D inputs get ``z = 0``.
    """
    net = np.asarray(control_points, dtype=float)
    if net.ndim != 3:
        raise ValueError("control_points must have shape (n_u, n_v, dim) with dim 2 or 3")

    n_u, n_v, dim = net.shape
    if dim == 2:
        net3 = np.zeros((n_u, n_v, 3), dtype=float)
        net3[:, :, :2] = net
    elif dim == 3:
        net3 = net
    else:
        raise ValueError("Last dimension must be 2 or 3")

    ctrlpts: list[list[float]] = []
    for i in range(n_u):
        for j in range(n_v):
            ctrlpts.append(net3[i, j].tolist())

    surf = BSpline.Surface()
    surf.degree_u = int(degree_u)
    surf.degree_v = int(degree_v)
    surf.set_ctrlpts(ctrlpts, n_u, n_v)
    surf.knotvector_u = np.asarray(knotvector_u, dtype=float).tolist()
    surf.knotvector_v = np.asarray(knotvector_v, dtype=float).tolist()
    surf.delta = delta
    return surf


def evalpts_grid(surface: BSpline.Surface) -> np.ndarray:
    """Evaluation points as ``(n_v, n_u, 3)`` (v varies fastest in geomdl)."""
    _ = surface.evalpts
    pts = np.asarray(surface.evalpts, dtype=float)
    return pts.reshape(surface.sample_size_v, surface.sample_size_u, 3)


def iso_u_curve(surface: BSpline.Surface, u: float, n: int = 120) -> np.ndarray:
    """Sample ``C(u, v)`` for fixed ``u``."""
    v0, v1 = surface.domain[1]
    vs = np.linspace(v0, v1, n)
    return np.array([surface.evaluate_single([float(u), float(v)]) for v in vs])


def iso_v_curve(surface: BSpline.Surface, v: float, n: int = 120) -> np.ndarray:
    """Sample ``C(u, v)`` for fixed ``v``."""
    u0, u1 = surface.domain[0]
    us = np.linspace(u0, u1, n)
    return np.array([surface.evaluate_single([float(u), float(v)]) for u in us])


def _plot_knot_axis(ax: plt.Axes, knots: np.ndarray, degree: int, param_name: str) -> None:
    spans = unique_knot_spans(knots, degree)
    ax.set_xlim(spans[0], spans[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(param_name)
    ax.set_yticks([])
    for kv in spans:
        ax.axvline(kv, color="0.35", linewidth=1)
    for i in range(len(spans) - 1):
        mid = 0.5 * (spans[i] + spans[i + 1])
        ax.text(mid, 0.5, f"{i}", ha="center", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)


def _plot_surface_2d(
    ax: plt.Axes,
    grid: np.ndarray,
    ctrl2d: list[list[list[float]]],
    *,
    title: str,
) -> None:
    """Physical ``(x, y)`` view: wireframe + control net projection."""
    xg = grid[:, :, 0]
    yg = grid[:, :, 1]
    zg = grid[:, :, 2]

    ax.plot_surface(xg, yg, zg, cmap="viridis", alpha=0.55, linewidth=0, antialiased=True)
    ax.plot_wireframe(xg, yg, zg, color="0.15", linewidth=0.4, alpha=0.5)

    for i, row in enumerate(ctrl2d):
        xs = [p[0] for p in row]
        ys = [p[1] for p in row]
        ax.plot(xs, ys, "k--", linewidth=0.8, alpha=0.5)
        if i == 0:
            ax.plot(xs, ys, "ko", markersize=3, label="Control net (u-rows)")
    for j in range(len(ctrl2d[0])):
        xs = [ctrl2d[i][j][0] for i in range(len(ctrl2d))]
        ys = [ctrl2d[i][j][1] for i in range(len(ctrl2d))]
        ax.plot(xs, ys, "r--", linewidth=0.8, alpha=0.45)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def _plot_surface_xy_flat(
    ax: plt.Axes,
    grid: np.ndarray,
    ctrl2d: list[list[list[float]]],
    *,
    title: str,
) -> None:
    """Flat ``(x, y)`` projection (primary 2D view)."""
    xg = grid[:, :, 0]
    yg = grid[:, :, 1]
    zg = grid[:, :, 2]

    ax.plot(xg.T, yg.T, "b-", linewidth=0.9, alpha=0.7)
    ax.plot(xg, yg, "b-", linewidth=0.9, alpha=0.7)
    cf = ax.contourf(xg, yg, zg, levels=20, cmap="viridis", alpha=0.85)
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="z")

    for row in ctrl2d:
        xs = [p[0] for p in row]
        ys = [p[1] for p in row]
        ax.plot(xs, ys, "k--", linewidth=1, alpha=0.55)
    for j in range(len(ctrl2d[0])):
        xs = [ctrl2d[i][j][0] for i in range(len(ctrl2d))]
        ys = [ctrl2d[i][j][1] for i in range(len(ctrl2d))]
        ax.plot(xs, ys, "r--", linewidth=1, alpha=0.5)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def demo_saddle_patch(*, delta: float = 0.04) -> BSpline.Surface:
    """5×5 tensor-product patch with mild height variation (demo geometry)."""
    u_vals = np.linspace(0.0, 4.0, 5)
    v_vals = np.linspace(0.0, 3.0, 5)
    net = np.zeros((5, 5, 3), dtype=float)
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            net[i, j] = [
                u,
                v,
                0.35 * np.sin(u) * np.cos(v),
            ]
    ku = open_uniform_knots(5, 3)
    kv = open_uniform_knots(5, 3)
    return build_bspline_surface(net, ku, kv, 3, 3, delta=delta)


def attach_vismpl(
    surface: BSpline.Surface,
    style: Literal["surface", "wireframe"] = "surface",
    *,
    figure_size: tuple[float, float] = (12.0, 9.0),
) -> BSpline.Surface:
    """Attach a geomdl VisMPL component for ``surface.render()``."""
    config = VisMPL.VisConfig(
        figure_size=list(figure_size),
        figure_dpi=120,
        axes_equal=True,
        alpha=0.9,
    )
    if style == "wireframe":
        surface.vis = VisMPL.VisSurfWireframe(config)
    else:
        surface.vis = VisMPL.VisSurface(config)
    return surface


def render_surface_vismpl(
    surface: BSpline.Surface,
    *,
    style: Literal["surface", "wireframe"] = "surface",
    show: bool = True,
    save_path: Path | None = None,
    colormap=cm.viridis,
) -> None:
    """
    Open an interactive VisMPL 3D window (control net + surface).

    Drag to rotate, scroll to zoom. Close the window to continue (e.g. breakdown plots).
    """
    attach_vismpl(surface, style=style)
    _ = surface.evalpts
    render_kwargs: dict = {"plot": show}
    if save_path is not None:
        render_kwargs["filename"] = str(save_path)
    surface.render(colormap=colormap, **render_kwargs)


def demo_s_ribbon_patch(*, delta: float = 0.04) -> BSpline.Surface:
    """Ruled patch: S-shaped edge in u, linear blend in v (uses run_p4_to_p3 u-data)."""
    from nurbs_experimenting import S_SHAPED_DEGREE, S_SHAPED_KNOTS

    s_cps = s_shaped_control_points()
    n_u = s_cps.shape[0]
    n_v = 4
    v_offsets = np.array([0.0, 0.8, 1.6, 2.4])
    net = np.zeros((n_u, n_v, 3), dtype=float)
    for i in range(n_u):
        for j in range(n_v):
            net[i, j, 0] = s_cps[i, 0]
            net[i, j, 1] = s_cps[i, 1] + v_offsets[j]
            net[i, j, 2] = 0.15 * v_offsets[j] * np.sin(0.3 * s_cps[i, 0])

    kv = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    return build_bspline_surface(
        net,
        S_SHAPED_KNOTS,
        kv,
        S_SHAPED_DEGREE,
        3,
        delta=delta,
    )


def visualize_bspline_surface(
    surface: BSpline.Surface,
    *,
    title: str | None = None,
    save_stem: Path | None = None,
    num_iso: int = 5,
    show: bool = True,
    use_vismpl: bool = True,
    vismpl_style: Literal["surface", "wireframe"] = "surface",
) -> None:
    """
    VisMPL interactive 3D (optional) plus 2D / 1D breakdown figures.

    If ``save_stem`` is ``outputs/foo``, writes:
      - ``outputs/foo_vismpl.png`` — VisMPL 3D snapshot (when saving)
      - ``outputs/foo_2d.png`` — flat (x, y) projection with z contour
      - ``outputs/foo_1d2d_breakdown.png`` — iso-u / iso-v, knot axes, 3D wireframe
    """
    if use_vismpl:
        vismpl_save = None
        if save_stem is not None:
            vismpl_save = Path(save_stem).parent / f"{Path(save_stem).name}_vismpl.png"
        render_surface_vismpl(
            surface,
            style=vismpl_style,
            show=show,
            save_path=vismpl_save if save_stem is not None else None,
        )
        if vismpl_save is not None:
            print(f"Saved {vismpl_save}")

    # Breakdown uses matplotlib; skip second GUI if VisMPL window was already shown.
    breakdown_show = show and not use_vismpl
    grid = evalpts_grid(surface)
    ctrl2d = surface.ctrlpts2d
    pu, pv = surface.degree_u, surface.degree_v
    nu, nv = surface.ctrlpts_size_u, surface.ctrlpts_size_v
    base_title = title or f"B-spline surface: {nu}×{nv} CPs, p_u={pu}, p_v={pv}"

    if save_stem is not None:
        save_stem = Path(save_stem)
        save_stem.parent.mkdir(parents=True, exist_ok=True)

    # --- 2D-only figure ---
    fig2d, ax2d = plt.subplots(figsize=(8, 6))
    _plot_surface_xy_flat(ax2d, grid, ctrl2d, title=base_title + " (2D)")
    fig2d.tight_layout()
    if save_stem is not None:
        path_2d = save_stem.parent / f"{save_stem.name}_2d.png"
        fig2d.savefig(path_2d, dpi=150)
        print(f"Saved {path_2d}")

    # --- 1D / 2D breakdown ---
    u_iso = np.linspace(surface.domain[0][0], surface.domain[0][1], num_iso)
    v_iso = np.linspace(surface.domain[1][0], surface.domain[1][1], num_iso)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.2, 1.0, 0.45],
        width_ratios=[1.1, 1.0],
        hspace=0.35,
        wspace=0.28,
    )

    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    _plot_surface_2d(ax_3d, grid, ctrl2d, title=base_title + " (3D)")

    ax_xy = fig.add_subplot(gs[0, 1])
    _plot_surface_xy_flat(ax_xy, grid, ctrl2d, title="(x, y) projection")

    ax_iso_u = fig.add_subplot(gs[1, 0])
    for u in u_iso:
        pts = iso_u_curve(surface, float(u))
        ax_iso_u.plot(pts[:, 0], pts[:, 1], linewidth=1.2, label=f"u={u:.2f}")
    ax_iso_u.set_title("1D breakdown: iso-u curves (v varies)")
    ax_iso_u.set_aspect("equal", adjustable="datalim")
    ax_iso_u.legend(loc="best", fontsize=7, ncol=2)
    ax_iso_u.grid(True, alpha=0.3)

    ax_iso_v = fig.add_subplot(gs[1, 1])
    for v in v_iso:
        pts = iso_v_curve(surface, float(v))
        ax_iso_v.plot(pts[:, 0], pts[:, 1], linewidth=1.2, label=f"v={v:.2f}")
    ax_iso_v.set_title("1D breakdown: iso-v curves (u varies)")
    ax_iso_v.set_aspect("equal", adjustable="datalim")
    ax_iso_v.legend(loc="best", fontsize=7, ncol=2)
    ax_iso_v.grid(True, alpha=0.3)

    ax_ku = fig.add_subplot(gs[2, 0])
    _plot_knot_axis(ax_ku, np.asarray(surface.knotvector_u), pu, "u")
    ax_ku.set_title(f"Knot spans (u), p_u={pu}")

    ax_kv = fig.add_subplot(gs[2, 1])
    _plot_knot_axis(ax_kv, np.asarray(surface.knotvector_v), pv, "v")
    ax_kv.set_title(f"Knot spans (v), p_v={pv}")

    fig.suptitle(base_title + " — 1D/2D breakdown", fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.92, hspace=0.38, wspace=0.25)

    if save_stem is not None:
        path_bd = save_stem.parent / f"{save_stem.name}_1d2d_breakdown.png"
        fig.savefig(path_bd, dpi=150, bbox_inches="tight")
        print(f"Saved {path_bd}")

    if breakdown_show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2d)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize B-spline surfaces (geomdl).")
    parser.add_argument(
        "--demo",
        choices=("saddle", "s-ribbon"),
        default="saddle",
        help="Built-in demo surface (default: saddle)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("outputs/surface_demo"),
        help="Output path stem (writes *_2d.png and *_1d2d_breakdown.png)",
    )
    parser.add_argument("--no-show", action="store_true", help="Save PNGs only, no plot windows")
    parser.add_argument(
        "--no-vismpl",
        action="store_true",
        help="Skip VisMPL interactive 3D (matplotlib breakdown only)",
    )
    parser.add_argument(
        "--vismpl-style",
        choices=("surface", "wireframe"),
        default="surface",
        help="VisMPL 3D style: shaded tri surface or wireframe (default: surface)",
    )
    parser.add_argument(
        "--no-breakdown",
        action="store_true",
        help="VisMPL only; skip 1D/2D breakdown PNGs",
    )
    args = parser.parse_args()

    if args.demo == "s-ribbon":
        surf = demo_s_ribbon_patch()
        title = "S-ribbon patch (u: S-curve, v: blend)"
    else:
        surf = demo_saddle_patch()
        title = "Demo saddle patch"

    if args.no_breakdown:
        render_surface_vismpl(
            surf,
            style=args.vismpl_style,
            show=not args.no_show,
            save_path=args.save.parent / f"{args.save.name}_vismpl.png" if args.save else None,
        )
    else:
        visualize_bspline_surface(
            surf,
            title=title,
            save_stem=args.save,
            show=not args.no_show,
            use_vismpl=not args.no_vismpl,
            vismpl_style=args.vismpl_style,
        )


if __name__ == "__main__":
    main()
