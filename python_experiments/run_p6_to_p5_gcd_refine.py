"""
p6 → p5 with GCD-based knot refinement before degree reduction.

Grid step ``h = max(d / r, min_step)`` where ``d`` is the GCD of unique knot spans,
``r`` is the resolution (default 1 → ``h = d``; ``-r 2`` → ``h = d/2``), and
``min_step`` defaults to 0.05.

    .venv/bin/python run_p6_to_p5_gcd_refine.py
    .venv/bin/python run_p6_to_p5_gcd_refine.py -r 2 -M 2 -n
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

from knot_refinement_algorithms.gcd_knot_refinement import DEFAULT_MIN_STEP, gcd_new_knots
from knot_refinement_algorithms.knot_refinement_common import DEFAULT_INSERT_MULTIPLICITY, apply_new_knots
from run_p6_to_p5 import (
    OUTPUT_DIR,
    P_IN,
    build_geomdl_curve,
    curve_to_numpy,
    degree_reduce_or_exit,
    render_comparison_figure,
)
from visualize_bspline_curve import multiple_peak_control_points, multiple_peak_knots


def auto_save_path_gcd(resolution: int, insert_multiplicity: int) -> Path:
    r_part = "" if resolution == 1 else f"_r{resolution}"
    return OUTPUT_DIR / f"p6_to_p5_gcd{r_part}_m{insert_multiplicity}.png"


def apply_gcd_refinement(
    curve,
    knotvector: np.ndarray,
    degree: int,
    *,
    resolution: int,
    insert_multiplicity: int,
    min_step: float,
) -> tuple[list[tuple[float, int]], Fraction, Fraction, Fraction, list[float]]:
    """
    Insert GCD-grid knots not already in ``U``.

    Returns log entries ``(u, n_inserted)``, ``d``, ``h``, and full new-knot list.
    """
    new_knots, d, h_req, h, grid = gcd_new_knots(
        knotvector, degree, resolution=resolution, min_step=min_step
    )

    print(f"unique span GCD d = {float(d):g}  ({d})")
    print(f"resolution r = {resolution}, requested step = {float(h_req):g}  ({h_req})")
    if h != h_req:
        print(f"min step floor = {min_step:g} → actual step h = {float(h):g}  ({h})")
    else:
        print(f"step h = {float(h):g}  ({h})")
    print(f"refinement grid ({len(grid)} points): {[float(t) for t in grid]}")

    log = apply_new_knots(
        curve,
        new_knots,
        insert_multiplicity,
        algorithm="GCD",
    )
    return log, d, h, new_knots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="p6→p5 with GCD knot refinement + A5.11 degree reduction.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=None,
        metavar="R",
        help="integer >= 2: grid step h = d/r (default: h = d)",
    )
    parser.add_argument(
        "--min-step",
        type=float,
        default=DEFAULT_MIN_STEP,
        metavar="H",
        help=f"minimum uniform grid gap (default: {DEFAULT_MIN_STEP})",
    )
    parser.add_argument(
        "-M",
        "--insert-multiplicity",
        type=int,
        default=DEFAULT_INSERT_MULTIPLICITY,
        metavar="M",
        dest="insert_multiplicity",
        help=f"insertions per new grid knot, 1..p (default: {DEFAULT_INSERT_MULTIPLICITY})",
    )
    parser.add_argument(
        "-o",
        "--save",
        type=Path,
        default=None,
        help="output PNG (default: auto-named under knot_refinement_experiment_outputs/)",
    )
    parser.add_argument(
        "-n",
        "--no-show",
        action="store_true",
        help="save only, no window",
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

    resolution = 1 if args.resolution is None else int(args.resolution)
    if args.resolution is not None and resolution < 2:
        raise SystemExit("--resolution / -r must be an integer >= 2 when provided")
    if not 1 <= args.insert_multiplicity <= P_IN:
        raise SystemExit(f"--insert-multiplicity must be in [1, {P_IN}] (degree p)")
    if args.min_step <= 0:
        raise SystemExit("--min-step must be positive")

    save_path = (
        args.save
        if args.save is not None
        else auto_save_path_gcd(resolution, args.insert_multiplicity)
    )
    print(f"save path: {save_path}")

    qw = multiple_peak_control_points()
    u = multiple_peak_knots()
    print("Qw:\n", qw)
    print("U:", u)

    curve_refined = build_geomdl_curve(qw, u, P_IN, name="gcd-refined")
    apply_gcd_refinement(
        curve_refined,
        u,
        P_IN,
        resolution=resolution,
        insert_multiplicity=args.insert_multiplicity,
        min_step=args.min_step,
    )
    qw_ref, u_ref, p_in = curve_to_numpy(curve_refined)
    print(f"after GCD refinement: {qw_ref.shape[0]} CPs, len(U)={u_ref.size}, p={p_in}")
    print("Qw (refined):\n", qw_ref)
    print("U (refined):", u_ref)

    pw_orig, uh_orig, err_orig = degree_reduce_or_exit(
        qw.shape[0], P_IN, u, qw, label="original reduction"
    )
    pw_ref, uh_ref, err_ref = degree_reduce_or_exit(
        qw_ref.shape[0], p_in, u_ref, qw_ref, label="GCD-refined reduction"
    )

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
