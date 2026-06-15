"""
Sweep adaptive refinement iterations and plot final max reduction error.

Runs ``run_p6_to_p5_adaptive_refine`` logic for ``-i 1 .. N`` and plots
max(error_array) from the final refined p6→p5 reduction vs iteration count.

    .venv/bin/python knot_refinement_experiments/adaptive/plot_max_error_sweep.py -n
    .venv/bin/python knot_refinement_experiments/adaptive/plot_max_error_sweep.py --max-i 10 -M 2 -n
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

from nurbs_degree_reduction import degree_reduce_unified
from knot_refinement_algorithms.adaptive_error_knot_refinement import adaptive_error_refine
from knot_refinement_algorithms.knot_refinement_common import DEFAULT_INSERT_MULTIPLICITY
from knot_refinement_experiments.common import build_geomdl_from_geometry, geometry_from_geomdl
from knot_refinement_experiments.experiment_cli import (
    add_example_arguments,
    degree_reduce_prefix,
    example_label,
    load_input_geometry,
)
from knot_refinement_experiments.paths import ensure_output_dir

OUTPUT_DIR = ensure_output_dir("adaptive")


def reduction_max_error(
    n: int,
    p: int,
    u: np.ndarray,
    qw: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> float:
    out = degree_reduce_unified(n, p, u, qw, weights=weights)
    if out == 1:
        return float("nan")
    _, _, err = out
    return float(np.max(err))


def run_adaptive_case(
    curve: str,
    degree: int,
    max_iterations: int,
    insert_multiplicity: int,
) -> tuple[float, float, int]:
    """Return (max_err_original, max_err_refined, num_cps_after_refine)."""
    geom = load_input_geometry(curve, degree=degree)

    err_orig = reduction_max_error(geom.qw.shape[0], geom.p, geom.u, geom.qw, weights=geom.weights)

    curve_obj = build_geomdl_from_geometry(
        geom.qw, geom.u, geom.p, weights=geom.weights, name="adaptive-refined"
    )
    adaptive_error_refine(
        curve_obj,
        geom.qw,
        geom.u,
        geom.p,
        max_iterations=max_iterations,
        insert_multiplicity=insert_multiplicity,
        weights=geom.weights,
    )
    qw_ref, u_ref, p_ref, weights_ref = geometry_from_geomdl(curve_obj)
    err_ref = reduction_max_error(qw_ref.shape[0], p_ref, u_ref, qw_ref, weights=weights_ref)
    return err_orig, err_ref, int(qw_ref.shape[0])


def plot_max_error_sweep(
    iterations: np.ndarray,
    err_orig: np.ndarray,
    err_ref: np.ndarray,
    n_cps: np.ndarray,
    *,
    insert_multiplicity: int,
    save_path: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    ax.plot(
        iterations,
        err_ref,
        "o-",
        color="#e07a3a",
        lw=2,
        ms=7,
        label="max error after adaptive refine → reduce",
    )
    ax.axhline(
        float(err_orig[0]),
        color="#4a7ab8",
        ls="--",
        lw=1.5,
        label=f"max error original → reduce ({err_orig[0]:.4g})",
    )

    ax.set_xlabel("adaptive refinement iterations (-i)")
    ax.set_ylabel("max(error_array)")
    ax.set_title(
        f"Adaptive refinement sweep (insert mult = {insert_multiplicity}, p6→p5)"
    )
    ax.set_xticks(iterations)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    for i, (x, y, n) in enumerate(zip(iterations, err_ref, n_cps)):
        if np.isfinite(y):
            ax.annotate(f"{n} CPs", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot max error vs adaptive -i sweep.")
    add_example_arguments(parser)
    parser.add_argument(
        "--max-i",
        type=int,
        default=10,
        help="largest iteration count to sweep (runs 1..max-i, default: 10)",
    )
    parser.add_argument(
        "-M",
        "--insert-multiplicity",
        type=int,
        default=DEFAULT_INSERT_MULTIPLICITY,
        dest="insert_multiplicity",
        help=f"insert multiplicity (default: {DEFAULT_INSERT_MULTIPLICITY})",
    )
    parser.add_argument(
        "-o",
        "--save",
        type=Path,
        default=None,
        help="output PNG path",
    )
    parser.add_argument("-n", "--no-show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_i < 1:
        raise SystemExit("--max-i must be >= 1")
    geom = load_input_geometry(args.curve, degree=args.degree)
    p_in = geom.p
    if not 1 <= args.insert_multiplicity <= p_in:
        raise SystemExit(f"--insert-multiplicity must be in [1, {p_in}]")

    prefix = degree_reduce_prefix(p_in, args.curve)
    save_path = (
        args.save
        if args.save is not None
        else OUTPUT_DIR / f"{prefix}_adaptive_max_error_sweep_i1_{args.max_i}_m{args.insert_multiplicity}.png"
    )

    iterations = np.arange(1, args.max_i + 1, dtype=int)
    err_orig_list: list[float] = []
    err_ref_list: list[float] = []
    n_cps_list: list[int] = []

    print(
        f"sweep: {example_label(args.curve, p_in)}, i = 1..{args.max_i}, M = {args.insert_multiplicity}"
    )
    for i in iterations:
        err_orig, err_ref, n_cps = run_adaptive_case(
            args.curve, args.degree, i, args.insert_multiplicity
        )
        err_orig_list.append(err_orig)
        err_ref_list.append(err_ref)
        n_cps_list.append(n_cps)
        print(f"  i={i:2d}: max_err_orig={err_orig:.6g}, max_err_ref={err_ref:.6g}, CPs={n_cps}")

    plot_max_error_sweep(
        iterations,
        np.asarray(err_orig_list),
        np.asarray(err_ref_list),
        np.asarray(n_cps_list),
        insert_multiplicity=args.insert_multiplicity,
        save_path=save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
