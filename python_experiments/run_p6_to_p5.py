"""
S-shaped curve: degree 6 → 5 comparison figure.

Left column: original input + degree reduction on the original.
Right column: knot-refined input + degree reduction on the refined curve.
Bottom row: per-column error arrays with a dotted line at each maximum.

Adjust ``KNOTS_TO_INSERT`` for knot-refinement (geomdl ``operations.insert_knot``).
New knot values are inserted twice by default; values already in ``U`` get one
insertion per explicit request (the default ×2 boost is skipped, not the insertion).

Tab completion (zsh/bash), after ``pip install argcomplete``::

    eval "$(register-python-argcomplete run_p6_to_p5.py)"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from geomdl import BSpline, operations

from b_spline_curve_reduction import DegreeReduceCurve
from visualize_bspline_curve import (
    evaluate_bspline,
    s_shaped_knots_p6,
    s_shaped_p6_control_points,
    multiple_peak_control_points,
    multiple_peak_knots,
)
 

# --- experiment knobs ---------------------------------------------------------
P_IN = 6
P_OUT = 5
DELTA = 0.01  # geomdl sampling density along the curve

# Knot-refinement experiment: list each knot to insert (repeat values = repeat insertions).
KNOTS_TO_INSERT: tuple[float, ...] = ()
KNOT_INSERT_MULTIPLICITY = 2  # insertions for values not yet present in U
OUTPUT_DIR = Path("knot_refinement_experiment_outputs")


def knot_tag(u: float) -> str:
    """Encode a knot value for filenames: 0.4 → ``04``, 0.25 → ``025``."""
    return f"{u:g}".replace(".", "")


def auto_save_path(
    knots: tuple[float, ...],
    knot_multiplicity: int,
) -> Path:
    """
    Auto-name experiment PNGs for easy tracking.

    Examples
    --------
    no knots        → ``knot_refinement_experiment_outputs/p6_to_p5_baseline.png``
    -k 0.4          → ``.../p6_to_p5_04x2.png``
    -k 0.4 0.75 -m 3 → ``.../p6_to_p5_04_075x3.png``
    """
    if not knots:
        return OUTPUT_DIR / "p6_to_p5_baseline.png"
    knots_part = "_".join(knot_tag(k) for k in knots)
    return OUTPUT_DIR / f"p6_to_p5_{knots_part}x{knot_multiplicity}.png"


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


def curve_to_numpy(curve: BSpline.Curve) -> tuple[np.ndarray, np.ndarray, int]:
    """Control points (n, 2), knot vector, degree."""
    qw = np.asarray(curve.ctrlpts, dtype=float)[:, :2]
    kv = np.asarray(curve.knotvector, dtype=float)
    return qw, kv, int(curve.degree)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return pw, uh, err


def curve_pair_eval_points(
    qw: np.ndarray,
    u: np.ndarray,
    p_in: int,
    pw: np.ndarray,
    uh: np.ndarray,
) -> np.ndarray:
    """Sampled input + reduced curve coordinates for axis limits (CPs excluded)."""
    _, xy_in = evaluate_bspline(qw, u, p_in)
    _, xy_out = evaluate_bspline(pw, uh, p_in - 1)
    return np.vstack([xy_in, xy_out])


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
) -> None:
    """Overlay input (blue) and reduced (orange) curves with control polygons."""
    _, xy_in = evaluate_bspline(qw, u, p_in)
    _, xy_out = evaluate_bspline(pw, uh, p_in - 1)

    ax.plot(xy_in[:, 0], xy_in[:, 1], color=COLOR_INPUT, lw=2, label=input_label)
    p_out = p_in - 1
    ax.plot(xy_out[:, 0], xy_out[:, 1], color=COLOR_REDUCED, lw=2, label=f"reduced p={p_out}")
    ax.plot(qw[:, 0], qw[:, 1], "o--", color=COLOR_INPUT_CP, ms=4, lw=1, alpha=0.8)
    ax.plot(pw[:, 0], pw[:, 1], "s--", color=COLOR_REDUCED_CP, ms=4, lw=1, alpha=0.8)
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
            curve_pair_eval_points(qw_orig, u_orig, P_IN, pw_orig, uh_orig),
            curve_pair_eval_points(qw_ref, u_ref, p_ref, pw_ref, uh_ref),
        ]
    )
    xlim, ylim = square_axis_limits(all_curve_pts)

    plot_curve_pair_on_ax(
        ax_curves_orig,
        qw_orig,
        u_orig,
        P_IN,
        pw_orig,
        uh_orig,
        title=f"original → p={P_OUT}",
        input_label=f"original p={P_IN}",
        xlim=xlim,
        ylim=ylim,
    )
    plot_curve_pair_on_ax(
        ax_curves_ref,
        qw_ref,
        u_ref,
        p_ref,
        pw_ref,
        uh_ref,
        title=f"refined → p={p_ref - 1}",
        input_label=f"refined p={p_ref} ({qw_ref.shape[0]} CPs)",
        xlim=xlim,
        ylim=ylim,
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

    fig.suptitle("p6 → p5 degree reduction comparison", fontsize=13, y=0.98)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="p6→p5 S-curve (geomdl + A5.11).")
    parser.add_argument(
        "-k",
        "--knots",
        "--insert-knots",
        nargs="*",
        type=float,
        default=None,
        metavar="U",
        dest="knots",
        help="Knots to insert before reduction (overrides KNOTS_TO_INSERT)",
    )
    parser.add_argument(
        "-m",
        "--knot-multiplicity",
        "--insert-multiplicity",
        type=int,
        default=KNOT_INSERT_MULTIPLICITY,
        metavar="N",
        dest="knot_multiplicity",
        help="Insertions for new knot values; existing values get 1 per request (default: 2)",
    )
    parser.add_argument(
        "-o",
        "--save",
        type=Path,
        default=None,
        metavar="PATH",
        help="output PNG (default: auto-named under knot_refinement_experiment_outputs/)",
    )
    parser.add_argument(
        "-n",
        "--no-show",
        action="store_true",
        help="Save only, no window",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args(argv)

    knots_to_insert = tuple(args.knots) if args.knots is not None else KNOTS_TO_INSERT
    save_path = (
        args.save
        if args.save is not None
        else auto_save_path(knots_to_insert, args.knot_multiplicity)
    )
    print(f"save path: {save_path}")

    # 1. Original geometry (9 CPs, two-span knot vector from visualize_bspline_curve).
    qw = multiple_peak_control_points()
    u = multiple_peak_knots()
    print("Qw:\n", qw)
    print("U:", u)

    # 2. Optional knot insertion on a copy (geometry unchanged, more CPs).
    curve_refined = build_geomdl_curve(qw, u, P_IN, name="refined")
    if knots_to_insert:
        print(
            f"knot insertion (default ×{args.knot_multiplicity} for new values):",
            list(knots_to_insert),
        )
        curve_refined, knot_log = refine_knots(
            curve_refined,
            knots_to_insert,
            default_multiplicity=args.knot_multiplicity,
        )
        for u_val, existing, n_insert in knot_log:
            note = "new" if existing == 0 else "existing"
            print(f"  {u_val:g} ({note}): mult {existing} → {existing + n_insert} (+{n_insert})")
        qw_ref, u_ref, p_in = curve_to_numpy(curve_refined)
        print(f"after refinement: {qw_ref.shape[0]} CPs, len(U)={u_ref.size}, p={p_in}")
        print("Qw (refined):\n", qw_ref)
        print("U (refined):", u_ref)
    else:
        qw_ref, u_ref, p_in = qw, u, P_IN

    # 3. Degree reduction on original and refined inputs (Piegl & Tiller A5.11).
    pw_orig, uh_orig, err_orig = degree_reduce_or_exit(
        qw.shape[0], P_IN, u, qw, label="original reduction"
    )
    pw_ref, uh_ref, err_ref = degree_reduce_or_exit(
        qw_ref.shape[0], p_in, u_ref, qw_ref, label="refined reduction"
    )

    # 4. Side-by-side curves with error arrays underneath.
    render_comparison_figure(
        qw_orig=qw,
        u_orig=u,
        pw_orig=pw_orig,
        uh_orig=uh_orig,
        err_orig=err_orig,
        qw_ref=qw_ref,
        u_ref=u_ref,
        pw_ref=pw_ref,
        uh_ref=uh_ref,
        err_ref=err_ref,
        save_path=save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
