"""
Surface degree reduction with adaptive error-driven knot refinement.

    .venv/bin/python surface_knot_refinement_experiments/adaptive/run.py -n
    .venv/bin/python surface_knot_refinement_experiments/adaptive/run.py -s circular_s_shaped_crown -i 5 -n
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
            spec = importlib.util.spec_from_file_location("surface_knot_bootstrap", bootstrap)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load {bootstrap}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return
    raise RuntimeError("surface_knot_refinement_experiments/_bootstrap.py not found")


_install_python_experiments_path()

from knot_refinement_algorithms.adaptive_error_knot_refinement import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_ITERATIONS,
)
from knot_refinement_algorithms.adaptive_error_surface_refinement import (
    adaptive_error_surface_refine,
)
from knot_refinement_algorithms.surface_knot_refinement_common import DEFAULT_INSERT_MULTIPLICITY
from surface_knot_refinement_experiments.common import (
    build_geomdl_from_surface_geometry,
    degree_reduce_surface_or_exit,
    geometry_from_geomdl_surface,
    render_surface_comparison_figure,
)
from surface_knot_refinement_experiments.experiment_cli import (
    add_surface_argument,
    degree_reduce_prefix,
    example_label,
    load_input_geometry,
)
from surface_knot_refinement_experiments.paths import ensure_output_dir

OUTPUT_DIR = ensure_output_dir("adaptive")


def auto_save_path_adaptive(
    surface: str,
    p_u: int,
    p_v: int,
    max_iterations: int,
    insert_multiplicity: int,
    batch_size: int,
) -> Path:
    prefix = degree_reduce_prefix(p_u, p_v, surface)
    batch_part = "" if batch_size == 1 else f"_b{batch_size}"
    return OUTPUT_DIR / f"{prefix}_adaptive_i{max_iterations}{batch_part}_m{insert_multiplicity}.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Surface adaptive error knot refinement + degree reduction.",
    )
    add_surface_argument(parser)
    parser.add_argument(
        "-i",
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        dest="max_iterations",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        dest="batch_size",
    )
    parser.add_argument(
        "-M",
        "--insert-multiplicity",
        type=int,
        default=DEFAULT_INSERT_MULTIPLICITY,
        dest="insert_multiplicity",
    )
    parser.add_argument("-o", "--save", type=Path, default=None)
    parser.add_argument("-n", "--no-show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    geom = load_input_geometry(args.surface)
    p_u, p_v = geom.p_u, geom.p_v
    max_p = max(p_u, p_v)
    if not 1 <= args.insert_multiplicity <= max_p:
        raise SystemExit(f"--insert-multiplicity must be in [1, {max_p}]")

    save_path = args.save or auto_save_path_adaptive(
        args.surface,
        p_u,
        p_v,
        args.max_iterations,
        args.insert_multiplicity,
        args.batch_size,
    )
    print(f"example: {example_label(args.surface, p_u, p_v, geom=geom)}")
    print(f"save path: {save_path}")

    surf_ref = build_geomdl_from_surface_geometry(
        geom.qw, geom.u, geom.v, p_u, p_v, weights=geom.weights, name="adaptive-refined"
    )
    adaptive_error_surface_refine(
        surf_ref,
        geom.qw,
        geom.u,
        geom.v,
        p_u,
        p_v,
        max_iterations=args.max_iterations,
        insert_multiplicity=args.insert_multiplicity,
        batch_size=args.batch_size,
        weights=geom.weights,
    )

    qw_ref, u_ref, v_ref, p_u_ref, p_v_ref, weights_ref = geometry_from_geomdl_surface(surf_ref)
    print(f"after adaptive refinement: {qw_ref.shape[0]}×{qw_ref.shape[1]} CPs")

    _, _, _, err_u_orig, err_v_orig, _ = degree_reduce_surface_or_exit(
        geom.qw.shape[0],
        geom.qw.shape[1],
        p_u,
        p_v,
        geom.u,
        geom.v,
        geom.qw,
        label="original reduction",
        weights=geom.weights,
    )
    _, _, _, err_u_ref, err_v_ref, _ = degree_reduce_surface_or_exit(
        qw_ref.shape[0],
        qw_ref.shape[1],
        p_u_ref,
        p_v_ref,
        u_ref,
        v_ref,
        qw_ref,
        label="adaptive-refined reduction (final)",
        weights=weights_ref,
    )

    render_surface_comparison_figure(
        qw_orig=geom.qw,
        u_orig=geom.u,
        v_orig=geom.v,
        p_u=p_u,
        p_v=p_v,
        err_u_orig=err_u_orig,
        err_v_orig=err_v_orig,
        qw_ref=qw_ref,
        u_ref=u_ref,
        v_ref=v_ref,
        p_u_ref=p_u_ref,
        p_v_ref=p_v_ref,
        err_u_ref=err_u_ref,
        err_v_ref=err_v_ref,
        save_path=save_path,
        show=not args.no_show,
        weights_orig=geom.weights,
        weights_ref=weights_ref,
        suptitle=(
            f"p{p_u}p{p_v} → p{p_u - 1}p{p_v - 1} adaptive "
            f"({example_label(args.surface, p_u, p_v, geom=geom)})"
        ),
    )


if __name__ == "__main__":
    main()
