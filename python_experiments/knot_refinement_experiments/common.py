"""Shared p6→p5 experiment utilities (geomdl, A5.11, comparison figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from geomdl import BSpline, NURBS, operations

from b_spline_curve_reduction import DegreeReduceCurve
from nurbs_degree_reduction import degree_reduce_nurbs, degree_reduce_unified
from visualize_bspline_curve import evaluate_bspline

P_IN = 6
P_OUT = 5
DELTA = 0.01
KNOT_INSERT_MULTIPLICITY = 2


def knot_tag(u: float) -> str:
    """Encode a knot value for filenames: 0.4 → ``04``, 0.25 → ``025``."""
    return f"{u:g}".replace(".", "")


def build_geomdl_curve(
    control_points: np.ndarray,
    knotvector: np.ndarray,
    degree: int,
    *,
    delta: float = DELTA,
    name: str = "",
) -> BSpline.Curve:
    """Polynomial B-spline curve in the xy plane (z = 0)."""
    qw = np.asarray(control_points, dtype=float)
    kv = np.asarray(knotvector, dtype=float)
    curve = BSpline.Curve()
    curve.degree = int(degree)
    curve.ctrlpts = [[float(x), float(y), 0.0] for x, y in qw]
    curve.knotvector = kv.tolist()
    curve.delta = delta
    if name:
        curve.name = name
    return curve


def build_geomdl_from_geometry(
    control_points: np.ndarray,
    knotvector: np.ndarray,
    degree: int,
    *,
    weights: np.ndarray | None = None,
    delta: float = DELTA,
    name: str = "",
) -> BSpline.Curve | NURBS.Curve:
    """Build a geomdl B-spline or rational NURBS curve from numpy geometry."""
    cp = np.asarray(control_points, dtype=float)
    kv = np.asarray(knotvector, dtype=float)
    if weights is None:
        return build_geomdl_curve(cp, kv, degree, delta=delta, name=name)

    w = np.asarray(weights, dtype=float)
    if cp.shape[1] == 2:
        ctrlpts = [[float(x), float(y), 0.0] for x, y in cp]
    elif cp.shape[1] == 3:
        ctrlpts = [[float(x), float(y), float(z)] for x, y, z in cp]
    else:
        raise ValueError("control_points must be (n, 2) or (n, 3)")

    curve = NURBS.Curve()
    curve.degree = int(degree)
    curve.ctrlpts = ctrlpts
    curve.weights = [float(v) for v in w]
    curve.knotvector = kv.tolist()
    curve.delta = float(delta)
    if name:
        curve.name = name
    return curve


def curve_to_numpy(curve: BSpline.Curve) -> tuple[np.ndarray, np.ndarray, int]:
    """Control points (n, 2), knot vector, degree."""
    qw = np.asarray(curve.ctrlpts, dtype=float)[:, :2]
    kv = np.asarray(curve.knotvector, dtype=float)
    return qw, kv, int(curve.degree)


def geometry_from_geomdl(
    curve: BSpline.Curve | NURBS.Curve,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray | None]:
    """``(Qw, U, p, weights)`` — ``weights`` is ``None`` for polynomial B-splines."""
    if isinstance(curve, NURBS.Curve):
        ctrl = np.asarray(curve.ctrlpts, dtype=float)
        if ctrl.shape[1] >= 3 and not np.allclose(ctrl[:, 2], 0.0):
            qw = ctrl[:, :3]
        else:
            qw = ctrl[:, :2]
        weights = np.asarray(curve.weights, dtype=float)
        kv = np.asarray(curve.knotvector, dtype=float)
        return qw, kv, int(curve.degree), weights

    qw, kv, p = curve_to_numpy(curve)
    return qw, kv, p, None


def knot_multiplicity_at(knotvector: np.ndarray, u: float) -> int:
    """Count occurrences of knot value ``u`` in a (possibly non-decreasing) knot vector."""
    kv = np.asarray(knotvector, dtype=float)
    return int(np.sum(np.isclose(kv, float(u))))


def insertions_for_knot(
    knotvector: np.ndarray,
    u: float,
    default_multiplicity: int,
) -> int:
    """
    Insertions for one explicit request.

    - Value not yet in ``U``: ``default_multiplicity`` (default 2).
    - Value already present: 1 (user asked for this knot; only the ×2 default is skipped).
    """
    if knot_multiplicity_at(knotvector, u) > 0:
        return 1
    return int(default_multiplicity)


def refine_knots(
    curve: BSpline.Curve,
    knots: tuple[float, ...],
    *,
    default_multiplicity: int = KNOT_INSERT_MULTIPLICITY,
) -> tuple[BSpline.Curve, list[tuple[float, int, int]]]:
    """
    Knot insertion via geomdl; updates ``curve`` in place.

    Returns the curve and a log of ``(u, existing_mult, n_inserted)`` per request.
    """
    if default_multiplicity < 1:
        raise ValueError("default_multiplicity must be >= 1")

    log: list[tuple[float, int, int]] = []
    for u in knots:
        u_val = float(u)
        kv = np.asarray(curve.knotvector, dtype=float)
        existing = knot_multiplicity_at(kv, u_val)
        n_insert = insertions_for_knot(kv, u_val, default_multiplicity)
        operations.insert_knot(curve, [u_val], [n_insert])
        log.append((u_val, existing, n_insert))
    return curve, log


COLOR_INPUT = "#4a7ab8"
COLOR_INPUT_CP = "#1a3a5c"
COLOR_REDUCED = "#e07a3a"
COLOR_REDUCED_CP = "#8b3a1a"
COLOR_ERR = "#5c4a9e"
COLOR_ERR_MAX = "#9e2a2a"


def degree_reduce_or_exit(
    n: int,
    p: int,
    u: np.ndarray,
    qw: np.ndarray,
    *,
    label: str,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    One-step degree reduction ``p → p-1``.

    Polynomial curves use A5.11 directly. NURBS curves lift to homogeneous space,
    run A5.11, then project back to ``(P, w)``.
    """
    if weights is not None:
        out = degree_reduce_nurbs(n, p, u, qw, weights)
        if out == 1:
            raise SystemExit(f"tolerance exceeded ({label})")
        pw, weights_out, uh, err = out
        print(f"--- {label} (homogeneous A5.11) ---")
        print("Pw:\n", pw)
        print("weights:", weights_out)
        print("Uh:", uh)
        print("error_array:", err)
        if np.any(err > 0):
            print("error_array (nonzero):", err[err > 0])
        return pw, uh, err, weights_out

    out = DegreeReduceCurve(n, p, u, qw)
    if out == 1:
        raise SystemExit(f"tolerance exceeded ({label})")
    pw, uh, err = out
    print(f"--- {label} ---")
    print("Pw:\n", pw)
    print("Uh:", uh)
    print("error_array:", err)
    if np.any(err > 0):
        print("error_array (nonzero):", err[err > 0])
    return pw, uh, err, None


def curve_pair_eval_points(
    qw: np.ndarray,
    u: np.ndarray,
    p_in: int,
    pw: np.ndarray,
    uh: np.ndarray,
    *,
    weights_in: np.ndarray | None = None,
    weights_out: np.ndarray | None = None,
    plot_dims: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Sampled input + reduced curve coordinates for axis limits (CPs excluded)."""
    _, xy_in = evaluate_bspline(qw, u, p_in, weights=weights_in)
    _, xy_out = evaluate_bspline(pw, uh, p_in - 1, weights=weights_out)
    di, dj = plot_dims
    return np.vstack([xy_in[:, [di, dj]], xy_out[:, [di, dj]]])


def square_axis_limits(points: np.ndarray, *, pad_frac: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    """Equal x/y span from evaluated curves so both panels share identical 1-1 boxes."""
    xmin, xmax = float(points[:, 0].min()), float(points[:, 0].max())
    ymin, ymax = float(points[:, 1].min()), float(points[:, 1].max())
    span = max(xmax - xmin, ymax - ymin) or 1.0
    span *= 1.0 + pad_frac
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * span
    return (cx - half, cx + half), (cy - half, cy + half)


def plot_curve_pair_on_ax(
    ax: plt.Axes,
    qw: np.ndarray,
    u: np.ndarray,
    p_in: int,
    pw: np.ndarray,
    uh: np.ndarray,
    *,
    title: str,
    input_label: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    weights_in: np.ndarray | None = None,
    weights_out: np.ndarray | None = None,
    plot_dims: tuple[int, int] = (0, 1),
) -> None:
    """Overlay input (blue) and reduced (orange) curves with control polygons."""
    _, xy_in = evaluate_bspline(qw, u, p_in, weights=weights_in)
    _, xy_out = evaluate_bspline(pw, uh, p_in - 1, weights=weights_out)
    di, dj = plot_dims
    p_out = p_in - 1

    ax.plot(xy_in[:, di], xy_in[:, dj], color=COLOR_INPUT, lw=2, label=input_label)
    ax.plot(xy_out[:, di], xy_out[:, dj], color=COLOR_REDUCED, lw=2, label=f"reduced p={p_out}")
    ax.plot(qw[:, di], qw[:, dj], "o--", color=COLOR_INPUT_CP, ms=4, lw=1, alpha=0.8)
    ax.plot(pw[:, di], pw[:, dj], "s--", color=COLOR_REDUCED_CP, ms=4, lw=1, alpha=0.8)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_error_array_on_ax(
    ax: plt.Axes,
    err: np.ndarray,
    u: np.ndarray,
    *,
    title: str,
) -> None:
    """Plot A5.11 error array with a dotted line at the maximum."""
    x = np.arange(err.size)
    err_max = float(np.max(err))
    ax.plot(x, err, "o-", color=COLOR_ERR, ms=3, lw=1.5, label="error")
    ax.axhline(err_max, color=COLOR_ERR_MAX, ls=":", lw=1.5, label=f"max = {err_max:.4g}")
    ax.set_xlabel("knot index")
    ax.set_ylabel("error")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(-0.5, err.size - 0.5)

    # Show every knot index and value on the x-axis.
    if u.size == err.size:
        tick_idx = np.arange(err.size)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([f"{u[i]:g}" for i in tick_idx], fontsize=7)
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_ha("center")
            label.set_va("top")


def render_comparison_figure(
    *,
    qw_orig: np.ndarray,
    u_orig: np.ndarray,
    pw_orig: np.ndarray,
    uh_orig: np.ndarray,
    err_orig: np.ndarray,
    qw_ref: np.ndarray,
    u_ref: np.ndarray,
    pw_ref: np.ndarray,
    uh_ref: np.ndarray,
    err_ref: np.ndarray,
    save_path: Path,
    show: bool,
    p_in: int = P_IN,
    weights_in_orig: np.ndarray | None = None,
    weights_out_orig: np.ndarray | None = None,
    weights_in_ref: np.ndarray | None = None,
    weights_out_ref: np.ndarray | None = None,
    plot_dims: tuple[int, int] = (0, 1),
    suptitle: str | None = None,
) -> None:
    """2×2 layout: curve pairs on top, error arrays below."""
    fig = plt.figure(figsize=(14.0, 10.0), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], hspace=0.45, wspace=0.12)

    ax_curves_orig = fig.add_subplot(gs[0, 0])
    ax_curves_ref = fig.add_subplot(gs[0, 1], sharex=ax_curves_orig, sharey=ax_curves_orig)
    ax_err_orig = fig.add_subplot(gs[1, 0])
    ax_err_ref = fig.add_subplot(gs[1, 1])

    p_ref = int(u_ref.size - qw_ref.shape[0] - 1)
    all_curve_pts = np.vstack(
        [
            curve_pair_eval_points(
                qw_orig,
                u_orig,
                p_in,
                pw_orig,
                uh_orig,
                weights_in=weights_in_orig,
                weights_out=weights_out_orig,
                plot_dims=plot_dims,
            ),
            curve_pair_eval_points(
                qw_ref,
                u_ref,
                p_ref,
                pw_ref,
                uh_ref,
                weights_in=weights_in_ref,
                weights_out=weights_out_ref,
                plot_dims=plot_dims,
            ),
        ]
    )
    xlim, ylim = square_axis_limits(all_curve_pts)

    axis_names = ("x", "y", "z")
    di, dj = plot_dims
    plane_suffix = ""
    if qw_orig.shape[1] > 2 or qw_ref.shape[1] > 2:
        plane_suffix = f" ({axis_names[di]}–{axis_names[dj]} projection)"

    plot_curve_pair_on_ax(
        ax_curves_orig,
        qw_orig,
        u_orig,
        p_in,
        pw_orig,
        uh_orig,
        title=f"original → p={p_in - 1}{plane_suffix}",
        input_label=f"original p={p_in}",
        xlim=xlim,
        ylim=ylim,
        weights_in=weights_in_orig,
        weights_out=weights_out_orig,
        plot_dims=plot_dims,
    )
    plot_curve_pair_on_ax(
        ax_curves_ref,
        qw_ref,
        u_ref,
        p_ref,
        pw_ref,
        uh_ref,
        title=f"refined → p={p_ref - 1}{plane_suffix}",
        input_label=f"refined p={p_ref} ({qw_ref.shape[0]} CPs)",
        xlim=xlim,
        ylim=ylim,
        weights_in=weights_in_ref,
        weights_out=weights_out_ref,
        plot_dims=plot_dims,
    )
    # Hide duplicate tick labels on the shared inner edge.
    ax_curves_ref.tick_params(labelleft=False, labelbottom=False)
    plot_error_array_on_ax(
        ax_err_orig,
        err_orig,
        u_orig,
        title="error array (original reduction)",
    )
    plot_error_array_on_ax(
        ax_err_ref,
        err_ref,
        u_ref,
        title="error array (refined reduction)",
    )

    title = suptitle or f"p{p_in} → p{p_in - 1} degree reduction comparison"
    fig.suptitle(title, fontsize=13, y=0.98)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)
