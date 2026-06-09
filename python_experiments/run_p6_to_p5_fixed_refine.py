"""
p6 → p5 with fixed-span knot refinement before degree reduction.

Uniform grid step ``d`` on ``[U_p, U_n]``; existing knots untouched, new grid
points inserted at multiplicity ``M`` (default 2).

    .venv/bin/python run_p6_to_p5_fixed_refine.py -d 0.05 -n
    .venv/bin/python run_p6_to_p5_fixed_refine.py -d 0.1 -M 3 -n
"""

from __future__ import annotations

import argparse
from pathlib import Path

from knot_refinement_algorithms.fixed_knot_refinement import DEFAULT_STEP_D, fixed_new_knots
from knot_refinement_algorithms.knot_refinement_common import DEFAULT_INSERT_MULTIPLICITY, apply_new_knots
from run_p6_to_p5 import (
    OUTPUT_DIR,
    P_IN,
    build_geomdl_curve,
    curve_to_numpy,
    degree_reduce_or_exit,
    knot_tag,
    render_comparison_figure,
)
from visualize_bspline_curve import multiple_peak_control_points, multiple_peak_knots


def auto_save_path_fixed(step_d: float, insert_multiplicity: int) -> Path:
    d_part = knot_tag(step_d)
    return OUTPUT_DIR / f"p6_to_p5_fixed_d{d_part}_m{insert_multiplicity}.png"


def apply_fixed_refinement(
    curve,
    knotvector,
    degree: int,
    *,
    step_d: float,
    insert_multiplicity: int,
) -> list[tuple[float, int]]:
    new_knots, d, grid = fixed_new_knots(knotvector, degree, step_d=step_d)

    print(f"fixed-span step d = {d:g}")
    print(f"refinement grid ({len(grid)} points): {[float(t) for t in grid]}")

    return apply_new_knots(
        curve,
        new_knots,
        insert_multiplicity,
        algorithm="fixed-span",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="p6→p5 with fixed-span knot refinement + A5.11 degree reduction.",
    )
    parser.add_argument(
        "-d",
        "--step",
        type=float,
        default=DEFAULT_STEP_D,
        metavar="D",
        dest="step_d",
        help=f"uniform grid step on [U_p, U_n] (default: {DEFAULT_STEP_D})",
    )
    parser.add_argument(
        "-M",
        "--insert-multiplicity",
        "-m",
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

    if args.step_d <= 0:
        raise SystemExit("--step / -d must be positive")
    if not 1 <= args.insert_multiplicity <= P_IN:
        raise SystemExit(f"--insert-multiplicity must be in [1, {P_IN}] (degree p)")

    save_path = (
        args.save
        if args.save is not None
        else auto_save_path_fixed(args.step_d, args.insert_multiplicity)
    )
    print(f"save path: {save_path}")

    qw = multiple_peak_control_points()
    u = multiple_peak_knots()
    print("Qw:\n", qw)
    print("U:", u)

    curve_refined = build_geomdl_curve(qw, u, P_IN, name="fixed-refined")
    apply_fixed_refinement(
        curve_refined,
        u,
        P_IN,
        step_d=args.step_d,
        insert_multiplicity=args.insert_multiplicity,
    )
    qw_ref, u_ref, p_in = curve_to_numpy(curve_refined)
    print(f"after fixed-span refinement: {qw_ref.shape[0]} CPs, len(U)={u_ref.size}, p={p_in}")
    print("Qw (refined):\n", qw_ref)
    print("U (refined):", u_ref)

    pw_orig, uh_orig, err_orig = degree_reduce_or_exit(
        qw.shape[0], P_IN, u, qw, label="original reduction"
    )
    pw_ref, uh_ref, err_ref = degree_reduce_or_exit(
        qw_ref.shape[0], p_in, u_ref, qw_ref, label="fixed-span refined reduction"
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
