"""
p6 → p5 with adaptive error-driven knot refinement before degree reduction.

Each outer iteration degree-reduces the current curve and inserts midpoint knot(s)
at the highest-error span(s). Use ``-b`` for batched insertion (several knots per
outer iteration, one A5.11 pass each).

    .venv/bin/python knot_refinement_experiments/adaptive/run.py -n
    .venv/bin/python knot_refinement_experiments/adaptive/run.py -i 5 -M 2 -n
    .venv/bin/python knot_refinement_experiments/adaptive/run.py -i 10 -b 4 -n
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

from knot_refinement_algorithms.adaptive_error_knot_refinement import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_ITERATIONS,
    adaptive_error_refine,
)
from knot_refinement_algorithms.knot_refinement_common import DEFAULT_INSERT_MULTIPLICITY
from knot_refinement_experiments.common import (
    build_geomdl_from_geometry,
    degree_reduce_or_exit,
    geometry_from_geomdl,
    render_comparison_figure,
)
from knot_refinement_experiments.experiment_cli import (
    add_example_arguments,
    degree_reduce_prefix,
    example_label,
    load_input_geometry,
)
from knot_refinement_experiments.paths import ensure_output_dir

OUTPUT_DIR = ensure_output_dir("adaptive")


def auto_save_path_adaptive(
    curve: str,
    p_in: int,
    max_iterations: int,
    insert_multiplicity: int,
    batch_size: int,
) -> Path:
    prefix = degree_reduce_prefix(p_in, curve)
    batch_part = "" if batch_size == 1 else f"_b{batch_size}"
    return OUTPUT_DIR / f"{prefix}_adaptive_i{max_iterations}{batch_part}_m{insert_multiplicity}.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="p6→p5 with adaptive error knot refinement + A5.11 degree reduction.",
    )
    add_example_arguments(parser)
    parser.add_argument(
        "-i",
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        metavar="N",
        dest="max_iterations",
        help=f"max outer refinement iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="B",
        dest="batch_size",
        help=(
            f"midpoint knots inserted per outer iteration (default: {DEFAULT_BATCH_SIZE}; "
            "1 = original serial behaviour)"
        ),
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
        help="output PNG (default: knot_refinement_experiments/adaptive/outputs/)",
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
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    geom = load_input_geometry(args.curve, degree=args.degree)
    p_in = geom.p
    if not 1 <= args.insert_multiplicity <= p_in:
        raise SystemExit(f"--insert-multiplicity must be in [1, {p_in}] (degree p)")

    save_path = (
        args.save
        if args.save is not None
        else auto_save_path_adaptive(
            args.curve, p_in, args.max_iterations, args.insert_multiplicity, args.batch_size
        )
    )
    print(f"example: {example_label(args.curve, p_in, geom=geom)}")
    print(f"save path: {save_path}")
    print("Qw:\n", geom.qw)
    print("U:", geom.u)

    curve_refined = build_geomdl_from_geometry(
        geom.qw, geom.u, p_in, weights=geom.weights, name="adaptive-refined"
    )
    adaptive_error_refine(
        curve_refined,
        geom.qw,
        geom.u,
        p_in,
        max_iterations=args.max_iterations,
        insert_multiplicity=args.insert_multiplicity,
        batch_size=args.batch_size,
        weights=geom.weights,
    )

    qw_ref, u_ref, p_in, weights_ref = geometry_from_geomdl(curve_refined)
    print(f"after adaptive refinement: {qw_ref.shape[0]} CPs, len(U)={u_ref.size}, p={p_in}")
    print("Qw (refined):\n", qw_ref)
    print("U (refined):", u_ref)

    pw_orig, uh_orig, err_orig, w_orig_out = degree_reduce_or_exit(
        geom.qw.shape[0], p_in, geom.u, geom.qw, label="original reduction", weights=geom.weights
    )
    pw_ref, uh_ref, err_ref, w_ref_out = degree_reduce_or_exit(
        qw_ref.shape[0], p_in, u_ref, qw_ref, label="adaptive-refined reduction (final)", weights=weights_ref
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
        suptitle=f"p{p_in} → p{p_in - 1} ({example_label(args.curve, p_in, geom=geom)})",
    )


if __name__ == "__main__":
    main()
