"""
p6 → p5 with adaptive error-driven knot refinement before degree reduction.

Each iteration degree-reduces the current curve, finds the max-error knot index,
and inserts a knot midway to the next unique knot value. The right-hand figure
shows the **final** refined curve and its reduction error array.

    .venv/bin/python run_p6_to_p5_adaptive_refine.py -n
    .venv/bin/python run_p6_to_p5_adaptive_refine.py -i 5 -M 2 -n
"""

from __future__ import annotations

import argparse
from pathlib import Path

from knot_refinement_algorithms.adaptive_error_knot_refinement import (
    DEFAULT_MAX_ITERATIONS,
    adaptive_error_refine,
)
from knot_refinement_algorithms.knot_refinement_common import DEFAULT_INSERT_MULTIPLICITY
from run_p6_to_p5 import (
    OUTPUT_DIR,
    P_IN,
    build_geomdl_curve,
    curve_to_numpy,
    degree_reduce_or_exit,
    render_comparison_figure,
)
from visualize_bspline_curve import multiple_peak_control_points, multiple_peak_knots


def auto_save_path_adaptive(max_iterations: int, insert_multiplicity: int) -> Path:
    return OUTPUT_DIR / f"p6_to_p5_adaptive_i{max_iterations}_m{insert_multiplicity}.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="p6→p5 with adaptive error knot refinement + A5.11 degree reduction.",
    )
    parser.add_argument(
        "-i",
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        metavar="N",
        dest="max_iterations",
        help=f"max refinement iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "-M",
        "--insert-multiplicity",
        type=int,
        default=DEFAULT_INSERT_MULTIPLICITY,
        metavar="M",
        dest="insert_multiplicity",
        help=f"insertions per new midpoint knot, 1..p (default: {DEFAULT_INSERT_MULTIPLICITY})",
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

    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be >= 1")
    if not 1 <= args.insert_multiplicity <= P_IN:
        raise SystemExit(f"--insert-multiplicity must be in [1, {P_IN}] (degree p)")

    save_path = (
        args.save
        if args.save is not None
        else auto_save_path_adaptive(args.max_iterations, args.insert_multiplicity)
    )
    print(f"save path: {save_path}")

    qw = multiple_peak_control_points()
    u = multiple_peak_knots()
    print("Qw:\n", qw)
    print("U:", u)

    curve_refined = build_geomdl_curve(qw, u, P_IN, name="adaptive-refined")
    print(f"adaptive refinement (max {args.max_iterations} iterations):")
    adaptive_error_refine(
        curve_refined,
        qw,
        u,
        P_IN,
        max_iterations=args.max_iterations,
        insert_multiplicity=args.insert_multiplicity,
    )

    qw_ref, u_ref, p_in = curve_to_numpy(curve_refined)
    print(f"after adaptive refinement: {qw_ref.shape[0]} CPs, len(U)={u_ref.size}, p={p_in}")
    print("Qw (refined):\n", qw_ref)
    print("U (refined):", u_ref)

    pw_orig, uh_orig, err_orig = degree_reduce_or_exit(
        qw.shape[0], P_IN, u, qw, label="original reduction"
    )
    pw_ref, uh_ref, err_ref = degree_reduce_or_exit(
        qw_ref.shape[0], p_in, u_ref, qw_ref, label="adaptive-refined reduction (final)"
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
