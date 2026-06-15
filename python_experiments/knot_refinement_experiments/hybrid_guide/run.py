"""
p6 → p5 curvature-guided adaptive refinement (no curvature knot insertion).

Curvature samples κ(u) each outer iteration to guide adaptive insertion.
Use ``--guide-mode`` to pick the guidance strategy.

    .venv/bin/python knot_refinement_experiments/hybrid_guide/run.py -n
    .venv/bin/python knot_refinement_experiments/hybrid_guide/run.py --guide-mode additive --alpha 2 -i 10 -n
    .venv/bin/python knot_refinement_experiments/hybrid_guide/run.py --guide-mode tiebreaker --tie-epsilon 0.1 -n
    .venv/bin/python knot_refinement_experiments/hybrid_guide/run.py --guide-mode biased_site -i 5 -n
    .venv/bin/python knot_refinement_experiments/hybrid_guide/run.py --guide-mode regional_budget -b 4 -n
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

from knot_refinement_algorithms.adaptive_error_knot_refinement import DEFAULT_BATCH_SIZE
from knot_refinement_algorithms.curvature_guided_adaptive_knot_refinement import (
    DEFAULT_CURVATURE_ALPHA,
    DEFAULT_CURVATURE_BETA,
    DEFAULT_GUIDE_MODE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TIE_EPSILON,
    GuideMode,
    curvature_guided_adaptive_refine,
)
from knot_refinement_algorithms.curvature_knot_refinement import (
    DEFAULT_H_MAX,
    DEFAULT_H_MIN,
    DEFAULT_N_SAMPLES,
)
from knot_refinement_algorithms.knot_refinement_common import DEFAULT_INSERT_MULTIPLICITY
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

OUTPUT_DIR = ensure_output_dir("hybrid_guide")

GUIDE_MODE_CHOICES: tuple[GuideMode, ...] = (
    "multiplicative",
    "additive",
    "tiebreaker",
    "biased_site",
    "regional_budget",
)


def auto_save_path_hybrid_guide(
    curve: str,
    p_in: int,
    max_iterations: int,
    insert_multiplicity: int,
    batch_size: int,
    guide_mode: GuideMode,
    curvature_alpha: float,
    curvature_beta: float,
    tie_epsilon: float,
    h_min: float,
    h_max: float,
) -> Path:
    prefix = degree_reduce_prefix(p_in, curve)
    batch_part = "" if batch_size == 1 else f"_b{batch_size}"
    mode_part = "" if guide_mode == "multiplicative" else f"_{guide_mode}"
    alpha_part = ""
    if guide_mode in ("multiplicative", "additive", "regional_budget") and curvature_alpha > 0:
        alpha_part = f"_a{curvature_alpha:g}"
    elif guide_mode == "tiebreaker":
        alpha_part = f"_te{tie_epsilon:g}"
    beta_part = ""
    if curvature_beta != 1.0 and guide_mode in ("multiplicative", "additive", "tiebreaker", "regional_budget"):
        beta_part = f"_kb{curvature_beta:g}"
    zone_part = ""
    if guide_mode == "regional_budget":
        zone_part = f"_hmin{knot_tag(h_min)}_hmax{knot_tag(h_max)}"
    return OUTPUT_DIR / (
        f"{prefix}_hybrid_guide{mode_part}_i{max_iterations}{batch_part}"
        f"{alpha_part}{beta_part}{zone_part}_m{insert_multiplicity}.png"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "p6→p5 curvature-guided adaptive refinement: κ guides span ranking "
            "and/or insertion sites (no curvature CPs)."
        ),
    )
    add_example_arguments(parser)
    parser.add_argument(
        "--guide-mode",
        choices=GUIDE_MODE_CHOICES,
        default=DEFAULT_GUIDE_MODE,
        dest="guide_mode",
        help=(
            "multiplicative: err*(1+ακ̂); additive: err+α·max(err)·κ̂; "
            "tiebreaker: κ breaks error ties within ε·max(err); "
            "biased_site: error rank + κ-peak insert; "
            f"regional_budget: zone quotas from κ spacing (default: {DEFAULT_GUIDE_MODE})"
        ),
    )
    parser.add_argument(
        "-i",
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        dest="max_iterations",
        help=f"max outer refinement iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        dest="batch_size",
        help=f"distinct insertion sites per outer iteration (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "-M",
        "--insert-multiplicity",
        type=int,
        default=DEFAULT_INSERT_MULTIPLICITY,
        dest="insert_multiplicity",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_CURVATURE_ALPHA,
        dest="curvature_alpha",
        help=(
            "multiplicative/additive/regional spacing weight "
            f"(default: {DEFAULT_CURVATURE_ALPHA})"
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_CURVATURE_BETA,
        dest="curvature_beta",
        help=f"κ exponent in scores/spacing (default: {DEFAULT_CURVATURE_BETA})",
    )
    parser.add_argument(
        "--tie-epsilon",
        type=float,
        default=DEFAULT_TIE_EPSILON,
        dest="tie_epsilon",
        help=(
            "tiebreaker: error band as fraction of max(err) "
            f"(default: {DEFAULT_TIE_EPSILON})"
        ),
    )
    parser.add_argument(
        "--h-min",
        type=float,
        default=DEFAULT_H_MIN,
        dest="h_min",
        help=f"regional_budget: min Δu zone spacing (default: {DEFAULT_H_MIN})",
    )
    parser.add_argument(
        "--h-max",
        type=float,
        default=DEFAULT_H_MAX,
        dest="h_max",
        help=f"regional_budget: max Δu zone spacing (default: {DEFAULT_H_MAX})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        dest="n_samples",
        help=f"κ(u) samples per outer iteration (default: {DEFAULT_N_SAMPLES})",
    )
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

    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.curvature_beta < 0:
        raise SystemExit("--beta must be >= 0")
    if args.tie_epsilon < 0:
        raise SystemExit("--tie-epsilon must be >= 0")
    if args.n_samples < 3:
        raise SystemExit("--n-samples must be >= 3")
    if args.h_min <= 0 or args.h_max <= 0 or args.h_min > args.h_max:
        raise SystemExit("require 0 < --h-min <= --h-max")

    geom = load_input_geometry(args.curve, degree=args.degree)
    p_in = geom.p
    if not 1 <= args.insert_multiplicity <= p_in:
        raise SystemExit(f"--insert-multiplicity must be in [1, {p_in}]")

    save_path = args.save or auto_save_path_hybrid_guide(
        args.curve,
        p_in,
        args.max_iterations,
        args.insert_multiplicity,
        args.batch_size,
        args.guide_mode,
        args.curvature_alpha,
        args.curvature_beta,
        args.tie_epsilon,
        args.h_min,
        args.h_max,
    )
    print(f"example: {example_label(args.curve, p_in, geom=geom)}")
    print(f"save path: {save_path}")

    curve_refined = build_geomdl_from_geometry(
        geom.qw, geom.u, p_in, weights=geom.weights, name="hybrid-guide-refined"
    )
    curvature_guided_adaptive_refine(
        curve_refined,
        geom.qw,
        geom.u,
        p_in,
        max_iterations=args.max_iterations,
        insert_multiplicity=args.insert_multiplicity,
        batch_size=args.batch_size,
        guide_mode=args.guide_mode,
        curvature_alpha=args.curvature_alpha,
        curvature_beta=args.curvature_beta,
        tie_epsilon=args.tie_epsilon,
        n_samples=args.n_samples,
        h_min=args.h_min,
        h_max=args.h_max,
        weights=geom.weights,
    )

    qw_ref, u_ref, p_in, weights_ref = geometry_from_geomdl(curve_refined)
    print(f"after guided refinement: {qw_ref.shape[0]} CPs, len(U)={u_ref.size}, p={p_in}")

    pw_orig, uh_orig, err_orig, w_orig_out = degree_reduce_or_exit(
        geom.qw.shape[0], p_in, geom.u, geom.qw, label="original reduction", weights=geom.weights
    )
    pw_ref, uh_ref, err_ref, w_ref_out = degree_reduce_or_exit(
        qw_ref.shape[0], p_in, u_ref, qw_ref, label="hybrid-guide final reduction", weights=weights_ref
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
