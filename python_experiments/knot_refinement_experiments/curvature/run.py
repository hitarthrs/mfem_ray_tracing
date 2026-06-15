"""
p6 → p5 with curvature-adaptive knot refinement before degree reduction.

    .venv/bin/python knot_refinement_experiments/curvature/run.py -n
    .venv/bin/python knot_refinement_experiments/curvature/run.py --march-mode u -n
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

from knot_refinement_algorithms.curvature_knot_refinement import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_H_MAX,
    DEFAULT_H_MIN,
    DEFAULT_MARCH_MODE,
    DEFAULT_N_SAMPLES,
    MarchMode,
)
from knot_refinement_algorithms.knot_refinement_common import DEFAULT_INSERT_MULTIPLICITY
from knot_refinement_experiments.common import (
    build_geomdl_from_geometry,
    degree_reduce_or_exit,
    geometry_from_geomdl,
    knot_tag,
    render_comparison_figure,
)
from knot_refinement_experiments.curvature.helpers import (
    apply_curvature_refinement,
    profile_save_path,
    save_curvature_profile,
)
from knot_refinement_experiments.experiment_cli import (
    add_example_arguments,
    degree_reduce_prefix,
    example_label,
    load_input_geometry,
)
from knot_refinement_experiments.paths import ensure_output_dir

OUTPUT_DIR = ensure_output_dir("curvature")


def auto_save_path_curvature(
    curve: str,
    p_in: int,
    h_min: float,
    h_max: float,
    alpha: float,
    insert_multiplicity: int,
    march_mode: MarchMode,
) -> Path:
    mode_part = "" if march_mode == "arc" else "_u"
    prefix = degree_reduce_prefix(p_in, curve)
    return (
        OUTPUT_DIR
        / f"{prefix}_curvature_hmin{knot_tag(h_min)}_hmax{knot_tag(h_max)}_a{alpha:g}{mode_part}_m{insert_multiplicity}.png"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="p6→p5 with curvature-adaptive knot refinement + A5.11 degree reduction.",
    )
    add_example_arguments(parser)
    parser.add_argument("--march-mode", choices=["arc", "u"], default=DEFAULT_MARCH_MODE)
    parser.add_argument("--h-min", type=float, default=DEFAULT_H_MIN)
    parser.add_argument("--h-max", type=float, default=DEFAULT_H_MAX)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("-M", "--insert-multiplicity", type=int, default=DEFAULT_INSERT_MULTIPLICITY, dest="insert_multiplicity")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("-o", "--save", type=Path, default=None)
    parser.add_argument("-n", "--no-show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass
    args = parser.parse_args(argv)

    if args.h_min <= 0 or args.h_max <= 0:
        raise SystemExit("--h-min and --h-max must be positive")
    if args.h_min > args.h_max:
        raise SystemExit("--h-min must be <= --h-max")
    geom = load_input_geometry(args.curve, degree=args.degree)
    p_in = geom.p
    if not 1 <= args.insert_multiplicity <= p_in:
        raise SystemExit(f"--insert-multiplicity must be in [1, {p_in}]")

    save_path = args.save or auto_save_path_curvature(
        args.curve, p_in, args.h_min, args.h_max, args.alpha,
        args.insert_multiplicity, args.march_mode,
    )
    print(f"example: {example_label(args.curve, p_in, geom=geom)}")
    print(f"save path: {save_path}")

    curve_refined = build_geomdl_from_geometry(
        geom.qw, geom.u, p_in, weights=geom.weights, name="curvature-refined"
    )
    _, u_samples, s_samples, kappa, spacing_samples, grid = apply_curvature_refinement(
        curve_refined, geom.u, p_in,
        h_min=args.h_min, h_max=args.h_max, alpha=args.alpha, beta=args.beta,
        n_samples=args.n_samples, insert_multiplicity=args.insert_multiplicity,
        march_mode=args.march_mode,
    )

    if args.profile:
        save_curvature_profile(
            u_samples, s_samples, kappa, spacing_samples, grid,
            march_mode=args.march_mode, save_path=profile_save_path(save_path),
        )

    qw_ref, u_ref, p_in, weights_ref = geometry_from_geomdl(curve_refined)
    pw_orig, uh_orig, err_orig, w_orig_out = degree_reduce_or_exit(
        geom.qw.shape[0], p_in, geom.u, geom.qw, label="original reduction", weights=geom.weights
    )
    pw_ref, uh_ref, err_ref, w_ref_out = degree_reduce_or_exit(
        qw_ref.shape[0], p_in, u_ref, qw_ref, label="curvature-refined reduction", weights=weights_ref
    )

    render_comparison_figure(
        qw_orig=geom.qw, u_orig=geom.u, pw_orig=pw_orig, uh_orig=uh_orig, err_orig=err_orig,
        qw_ref=qw_ref, u_ref=u_ref, pw_ref=pw_ref, uh_ref=uh_ref, err_ref=err_ref,
        save_path=save_path, show=not args.no_show, p_in=p_in,
        weights_in_orig=geom.weights, weights_out_orig=w_orig_out,
        weights_in_ref=weights_ref, weights_out_ref=w_ref_out,
        suptitle=f"p{p_in} → p{p_in - 1} ({example_label(args.curve, p_in, geom=geom)})",
    )


if __name__ == "__main__":
    main()
