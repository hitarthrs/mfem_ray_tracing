"""
Smoke test: surface degree reduction + error heatmap visualization.

    .venv/bin/python surface_knot_refinement_experiments/plot_error_demo.py -n
    .venv/bin/python surface_knot_refinement_experiments/plot_error_demo.py -s semicircle_plateau_shell -n
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
    degree_reduce_surface_or_exit,
    render_surface_comparison_figure,
)
from surface_knot_refinement_experiments.experiment_cli import (
    add_surface_argument,
    example_label,
    load_input_geometry,
)
from surface_knot_refinement_experiments.paths import ensure_output_dir

OUTPUT_DIR = ensure_output_dir("demo")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Surface error heatmap demo (no refinement).")
    add_surface_argument(parser)
    parser.add_argument("-o", "--save", type=Path, default=None)
    parser.add_argument("-n", "--no-show", action="store_true")
    args = parser.parse_args(argv)

    geom = load_input_geometry(args.surface)
    p_u, p_v = geom.p_u, geom.p_v
    save_path = args.save or OUTPUT_DIR / f"p{p_u}p{p_v}_to_p{p_u - 1}p{p_v - 1}_{args.surface}_error_demo.png"

    print(f"example: {example_label(args.surface, p_u, p_v, geom=geom)}")
    print(f"save path: {save_path}")

    r, uh, vh, err_u, err_v, w_out = degree_reduce_surface_or_exit(
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

    render_surface_comparison_figure(
        qw_orig=geom.qw,
        u_orig=geom.u,
        v_orig=geom.v,
        p_u=p_u,
        p_v=p_v,
        err_u_orig=err_u,
        err_v_orig=err_v,
        qw_ref=geom.qw,
        u_ref=geom.u,
        v_ref=geom.v,
        p_u_ref=p_u,
        p_v_ref=p_v,
        err_u_ref=err_u,
        err_v_ref=err_v,
        save_path=save_path,
        show=not args.no_show,
        weights_orig=geom.weights,
        weights_ref=geom.weights,
        suptitle=f"Error heatmap demo — {example_label(args.surface, p_u, p_v, geom=geom)}",
    )


if __name__ == "__main__":
    main()
