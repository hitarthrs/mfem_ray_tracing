"""
p6 → p5 with fixed-span knot refinement before degree reduction.

Uniform grid step ``d`` on ``[U_p, U_n]``; existing knots untouched, new grid
points inserted at multiplicity ``M`` (default 2).

    .venv/bin/python knot_refinement_experiments/fixed/run.py -d 0.05 -n
    .venv/bin/python knot_refinement_experiments/fixed/run.py --curve s_shaped --degree 5 -d 0.1 -M 3 -n
    .venv/bin/python knot_refinement_experiments/fixed/run.py --curve simple_semicircle --degree 3 -d 0.125 -n
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _install_python_experiments_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        bootstrap = parent / "_bootstrap.py"
        if bootstrap.is_file():
            spec = importlib.util.spec_from_file_location("knot_refinement_bootstrap", bootstrap)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load {bootstrap}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return
    raise RuntimeError("knot_refinement_experiments/_bootstrap.py not found")


_install_python_experiments_path()

from knot_refinement_algorithms.fixed_knot_refinement import DEFAULT_STEP_D, fixed_new_knots
from knot_refinement_algorithms.knot_refinement_common import DEFAULT_INSERT_MULTIPLICITY, apply_new_knots
from knot_refinement_experiments.common import (
    build_geomdl_from_geometry,
    degree_reduce_or_exit,
    geometry_from_geomdl,
    knot_tag,
    render_comparison_figure,
)
from knot_refinement_experiments.experiment_cli import (
    add_example_arguments,
    degree_reduce_prefix,
    example_label,
    load_input_geometry,
)
from knot_refinement_experiments.paths import ensure_output_dir

OUTPUT_DIR = ensure_output_dir("fixed")


def auto_save_path_fixed(
    curve: str,
    p_in: int,
    step_d: float,
    insert_multiplicity: int,
) -> Path:
    d_part = knot_tag(step_d)
    prefix = degree_reduce_prefix(p_in, curve)
    return OUTPUT_DIR / f"{prefix}_fixed_d{d_part}_m{insert_multiplicity}.png"


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
    add_example_arguments(parser)
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
        help="output PNG (default: knot_refinement_experiments/fixed/outputs/)",
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
    geom = load_input_geometry(args.curve, degree=args.degree)
    p_in = geom.p
    if not 1 <= args.insert_multiplicity <= p_in:
        raise SystemExit(f"--insert-multiplicity must be in [1, {p_in}] (degree p)")

    save_path = (
        args.save
        if args.save is not None
        else auto_save_path_fixed(args.curve, p_in, args.step_d, args.insert_multiplicity)
    )
    print(f"example: {example_label(args.curve, p_in, geom=geom)}")
    print(f"save path: {save_path}")
    if geom.weights is not None:
        print(f"weights: min={float(geom.weights.min()):g} max={float(geom.weights.max()):g}")
    print("Qw:\n", geom.qw)
    print("U:", geom.u)

    curve_refined = build_geomdl_from_geometry(
        geom.qw, geom.u, p_in, weights=geom.weights, name="fixed-refined"
    )
    apply_fixed_refinement(
        curve_refined,
        geom.u,
        p_in,
        step_d=args.step_d,
        insert_multiplicity=args.insert_multiplicity,
    )
    qw_ref, u_ref, p_in, weights_ref = geometry_from_geomdl(curve_refined)
    print(f"after fixed-span refinement: {qw_ref.shape[0]} CPs, len(U)={u_ref.size}, p={p_in}")
    print("Qw (refined):\n", qw_ref)
    print("U (refined):", u_ref)

    pw_orig, uh_orig, err_orig, w_orig_out = degree_reduce_or_exit(
        geom.qw.shape[0], p_in, geom.u, geom.qw, label="original reduction", weights=geom.weights
    )
    pw_ref, uh_ref, err_ref, w_ref_out = degree_reduce_or_exit(
        qw_ref.shape[0], p_in, u_ref, qw_ref, label="fixed-span refined reduction", weights=weights_ref
    )

    render_comparison_figure(
        qw_orig=geom.qw,
        u_orig=geom.u,
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
        p_in=p_in,
        weights_in_orig=geom.weights,
        weights_out_orig=w_orig_out,
        weights_in_ref=weights_ref,
        weights_out_ref=w_ref_out,
        suptitle=f"p{p_in} → p{p_in - 1} degree reduction ({example_label(args.curve, p_in, geom=geom)})",
    )


if __name__ == "__main__":
    main()
