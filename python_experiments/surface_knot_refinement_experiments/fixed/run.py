"""
Surface degree reduction with fixed-span knot refinement.

    .venv/bin/python surface_knot_refinement_experiments/fixed/run.py -d 0.1 -n
    .venv/bin/python surface_knot_refinement_experiments/fixed/run.py -s right_angled_squared --axis u -d 0.15 -n
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

from knot_refinement_algorithms.fixed_knot_refinement import DEFAULT_STEP_D
from knot_refinement_algorithms.fixed_surface_knot_refinement import apply_fixed_surface_refinement
from knot_refinement_algorithms.surface_knot_refinement_common import DEFAULT_INSERT_MULTIPLICITY
from surface_knot_refinement_experiments.common import (
    build_geomdl_from_surface_geometry,
    degree_reduce_surface_or_exit,
    geometry_from_geomdl_surface,
    knot_tag,
    render_surface_comparison_figure,
)
from surface_knot_refinement_experiments.experiment_cli import (
    add_surface_argument,
    degree_reduce_prefix,
    example_label,
    load_input_geometry,
)
from surface_knot_refinement_experiments.paths import ensure_output_dir

OUTPUT_DIR = ensure_output_dir("fixed")


def auto_save_path_fixed(
    surface: str,
    p_u: int,
    p_v: int,
    step_d: float,
    insert_multiplicity: int,
    axis: str,
) -> Path:
    d_part = knot_tag(step_d)
    prefix = degree_reduce_prefix(p_u, p_v, surface)
    axis_part = "" if axis == "both" else f"_{axis}"
    return OUTPUT_DIR / f"{prefix}_fixed{axis_part}_d{d_part}_m{insert_multiplicity}.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Surface fixed-span knot refinement + degree reduction.",
    )
    add_surface_argument(parser)
    parser.add_argument(
        "-d",
        "--step",
        type=float,
        default=DEFAULT_STEP_D,
        dest="step_d",
        help=f"uniform grid step on active interval (default: {DEFAULT_STEP_D})",
    )
    parser.add_argument(
        "-M",
        "--insert-multiplicity",
        type=int,
        default=DEFAULT_INSERT_MULTIPLICITY,
        dest="insert_multiplicity",
    )
    parser.add_argument(
        "--axis",
        choices=("both", "u", "v"),
        default="both",
        help="refine u knots, v knots, or both (default: both)",
    )
    parser.add_argument("-o", "--save", type=Path, default=None)
    parser.add_argument("-n", "--no-show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.step_d <= 0:
        raise SystemExit("--step must be positive")

    geom = load_input_geometry(args.surface)
    p_u, p_v = geom.p_u, geom.p_v
    max_p = max(p_u, p_v)
    if not 1 <= args.insert_multiplicity <= max_p:
        raise SystemExit(f"--insert-multiplicity must be in [1, {max_p}]")

    save_path = args.save or auto_save_path_fixed(
        args.surface, p_u, p_v, args.step_d, args.insert_multiplicity, args.axis
    )
    print(f"example: {example_label(args.surface, p_u, p_v, geom=geom)}")
    print(f"save path: {save_path}")

    surf_ref = build_geomdl_from_surface_geometry(
        geom.qw, geom.u, geom.v, p_u, p_v, weights=geom.weights, name="fixed-refined"
    )
    apply_fixed_surface_refinement(
        surf_ref,
        geom.u,
        geom.v,
        p_u,
        p_v,
        step_d=args.step_d,
        insert_multiplicity=args.insert_multiplicity,
        axis=args.axis,
    )

    qw_ref, u_ref, v_ref, p_u_ref, p_v_ref, weights_ref = geometry_from_geomdl_surface(surf_ref)
    print(f"after fixed-span refinement: {qw_ref.shape[0]}×{qw_ref.shape[1]} CPs")

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
        label="fixed-span refined reduction",
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
            f"p{p_u}p{p_v} → p{p_u - 1}p{p_v - 1} fixed-span "
            f"({example_label(args.surface, p_u, p_v, geom=geom)})"
        ),
    )


if __name__ == "__main__":
    main()
