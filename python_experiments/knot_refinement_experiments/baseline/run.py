"""
Manual knot insertion before p6→p5 degree reduction.

    .venv/bin/python knot_refinement_experiments/baseline/run.py -n
    .venv/bin/python knot_refinement_experiments/baseline/run.py --curve s_shaped --degree 4 -k 0.4 0.75 -n
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
from knot_refinement_experiments.common import (
    KNOT_INSERT_MULTIPLICITY,
    build_geomdl_from_geometry,
    degree_reduce_or_exit,
    geometry_from_geomdl,
    knot_tag,
    refine_knots,
    render_comparison_figure,
)
from knot_refinement_experiments.experiment_cli import (
    add_example_arguments,
    degree_reduce_prefix,
    example_label,
    load_input_geometry,
)
from knot_refinement_experiments.paths import ensure_output_dir

OUTPUT_DIR = ensure_output_dir("baseline")
KNOTS_TO_INSERT: tuple[float, ...] = ()


def auto_save_path(
    curve: str,
    p_in: int,
    knots: tuple[float, ...],
    knot_multiplicity: int,
) -> Path:
    prefix = degree_reduce_prefix(p_in, curve)
    if not knots:
        return OUTPUT_DIR / f"{prefix}_baseline.png"
    knots_part = "_".join(knot_tag(k) for k in knots)
    return OUTPUT_DIR / f"{prefix}_{knots_part}x{knot_multiplicity}.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="p6→p5 baseline (manual knot insertion).")
    add_example_arguments(parser)
    parser.add_argument("-k", "--knots", "--insert-knots", nargs="*", type=float, default=None, dest="knots")
    parser.add_argument(
        "-m",
        "--knot-multiplicity",
        "--insert-multiplicity",
        type=int,
        default=KNOT_INSERT_MULTIPLICITY,
        dest="knot_multiplicity",
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

    knots_to_insert = tuple(args.knots) if args.knots is not None else KNOTS_TO_INSERT
    geom = load_input_geometry(args.curve, degree=args.degree)
    p_in = geom.p
    save_path = args.save or auto_save_path(args.curve, p_in, knots_to_insert, args.knot_multiplicity)
    print(f"example: {example_label(args.curve, p_in, geom=geom)}")
    print(f"save path: {save_path}")
    print("Qw:\n", geom.qw)
    print("U:", geom.u)

    curve_refined = build_geomdl_from_geometry(
        geom.qw, geom.u, p_in, weights=geom.weights, name="refined"
    )
    if knots_to_insert:
        print(f"knot insertion (default ×{args.knot_multiplicity} for new values):", list(knots_to_insert))
        curve_refined, knot_log = refine_knots(
            curve_refined, knots_to_insert, default_multiplicity=args.knot_multiplicity
        )
        for u_val, existing, n_insert in knot_log:
            note = "new" if existing == 0 else "existing"
            print(f"  {u_val:g} ({note}): mult {existing} → {existing + n_insert} (+{n_insert})")
        qw_ref, u_ref, p_ref, weights_ref = geometry_from_geomdl(curve_refined)
        print(f"after refinement: {qw_ref.shape[0]} CPs, len(U)={u_ref.size}, p={p_ref}")
    else:
        qw_ref, u_ref, p_ref, weights_ref = geom.qw, geom.u, p_in, geom.weights

    pw_orig, uh_orig, err_orig, w_orig_out = degree_reduce_or_exit(
        geom.qw.shape[0], p_in, geom.u, geom.qw, label="original reduction", weights=geom.weights
    )
    pw_ref, uh_ref, err_ref, w_ref_out = degree_reduce_or_exit(
        qw_ref.shape[0], p_ref, u_ref, qw_ref, label="refined reduction", weights=weights_ref
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
