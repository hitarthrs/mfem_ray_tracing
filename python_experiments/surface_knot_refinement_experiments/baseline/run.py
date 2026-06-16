"""
Manual knot insertion before surface degree reduction.

    .venv/bin/python surface_knot_refinement_experiments/baseline/run.py -n
    .venv/bin/python surface_knot_refinement_experiments/baseline/run.py -s right_angled_squared -ku 0.33 -kv 0.66 -n
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

from surface_knot_refinement_experiments.common import (
    KNOT_INSERT_MULTIPLICITY,
    build_geomdl_from_surface_geometry,
    degree_reduce_surface_or_exit,
    geometry_from_geomdl_surface,
    knot_tag,
    refine_surface_knots_u,
    refine_surface_knots_v,
    render_surface_comparison_figure,
)
from surface_knot_refinement_experiments.experiment_cli import (
    add_surface_argument,
    degree_reduce_prefix,
    example_label,
    load_input_geometry,
)
from surface_knot_refinement_experiments.paths import ensure_output_dir

OUTPUT_DIR = ensure_output_dir("baseline")


def auto_save_path(
    surface: str,
    p_u: int,
    p_v: int,
    u_knots: tuple[float, ...],
    v_knots: tuple[float, ...],
    knot_multiplicity: int,
) -> Path:
    prefix = degree_reduce_prefix(p_u, p_v, surface)
    if not u_knots and not v_knots:
        return OUTPUT_DIR / f"{prefix}_baseline.png"
    parts: list[str] = []
    if u_knots:
        parts.append("ku" + "_".join(knot_tag(k) for k in u_knots))
    if v_knots:
        parts.append("kv" + "_".join(knot_tag(k) for k in v_knots))
    return OUTPUT_DIR / f"{prefix}_{'_'.join(parts)}x{knot_multiplicity}.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Surface baseline (manual knot insertion).")
    add_surface_argument(parser)
    parser.add_argument("-ku", "--u-knots", nargs="*", type=float, default=None, dest="u_knots")
    parser.add_argument("-kv", "--v-knots", nargs="*", type=float, default=None, dest="v_knots")
    parser.add_argument(
        "-m",
        "--knot-multiplicity",
        type=int,
        default=KNOT_INSERT_MULTIPLICITY,
        dest="knot_multiplicity",
    )
    parser.add_argument("-o", "--save", type=Path, default=None)
    parser.add_argument("-n", "--no-show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    geom = load_input_geometry(args.surface)
    p_u, p_v = geom.p_u, geom.p_v
    u_knots = tuple(args.u_knots or ())
    v_knots = tuple(args.v_knots or ())
    save_path = args.save or auto_save_path(
        args.surface, p_u, p_v, u_knots, v_knots, args.knot_multiplicity
    )

    print(f"example: {example_label(args.surface, p_u, p_v, geom=geom)}")
    print(f"save path: {save_path}")

    surf_ref = build_geomdl_from_surface_geometry(
        geom.qw, geom.u, geom.v, p_u, p_v, weights=geom.weights, name="refined"
    )

    if u_knots:
        print(f"u-knot insertion (×{args.knot_multiplicity}):", list(u_knots))
        refine_surface_knots_u(surf_ref, u_knots, default_multiplicity=args.knot_multiplicity)
    if v_knots:
        print(f"v-knot insertion (×{args.knot_multiplicity}):", list(v_knots))
        refine_surface_knots_v(surf_ref, v_knots, default_multiplicity=args.knot_multiplicity)

    qw_ref, u_ref, v_ref, p_u_ref, p_v_ref, weights_ref = geometry_from_geomdl_surface(surf_ref)
    if u_knots or v_knots:
        print(f"after refinement: {qw_ref.shape[0]}×{qw_ref.shape[1]} CPs")

    r_orig, uh_orig, vh_orig, err_u_orig, err_v_orig, w_orig_out = degree_reduce_surface_or_exit(
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
    r_ref, uh_ref, vh_ref, err_u_ref, err_v_ref, w_ref_out = degree_reduce_surface_or_exit(
        qw_ref.shape[0],
        qw_ref.shape[1],
        p_u_ref,
        p_v_ref,
        u_ref,
        v_ref,
        qw_ref,
        label="refined reduction",
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
            f"p{p_u}p{p_v} → p{p_u - 1}p{p_v - 1} "
            f"({example_label(args.surface, p_u, p_v, geom=geom)})"
        ),
    )


if __name__ == "__main__":
    main()
