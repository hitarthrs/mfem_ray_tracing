#!/usr/bin/env python3
"""
Demo: degree-4 NURBS examples reduced to degree 2 in two steps (Algorithm 1).

Algorithm 2 backends:

- ``single`` — one cut per step (``dummy_approach``); up to 2^2 = 4 leaves.
- ``multiple_cuts`` — cuts at several ``u`` per step (``dummy_multiple_cuts``);
  with default ``(0.25, 0.5, 0.75)`` expect up to 4^2 = 16 leaves.
- ``approach_1`` — peak-error adaptive split per step (``approach_1``); leaf
  count depends on ``max_error`` and geometry.

    cd python_experiments
    .venv/bin/python -m multiple_step_degree_reduction.demo_degree_4_to_2
    .venv/bin/python -m multiple_step_degree_reduction.demo_degree_4_to_2 \\
        --backend approach_1 -c s_shaped --max-error 0.15 -n
    .venv/bin/python -m multiple_step_degree_reduction.demo_degree_4_to_2 --all -n \\
        --backend multiple_cuts

    .venv/bin/python multiple_step_degree_reduction/degree_reduce_multiple_steps.py \\
        --backend approach_1 -c s_shaped --max-error 0.15 -n
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pathlib import Path

_PYTHON_EXPERIMENTS = Path(__file__).resolve().parent.parent
if str(_PYTHON_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_PYTHON_EXPERIMENTS))

import matplotlib.pyplot as plt
import numpy as np

from knot_refinement_experiments.common import (
    COLOR_INPUT,
    COLOR_INPUT_CP,
    build_geomdl_from_geometry,
)
from multiple_step_degree_reduction.degree_reduce_multiple_steps import (
    degree_reduce_multiple_steps,
)
from nurbs_curve_examples import curves_for_degree, load_example, load_example_weights
from single_step_degree_reduction.dummy_approach import (
    DummySplitReduceStep,
    curve_properties_from_curve,
    plot_curve_overlay,
)
from single_step_degree_reduction.dummy_multiple_cuts import (
    DEFAULT_SPLIT_US,
    DummyMultipleCutsStep,
)
from single_step_degree_reduction.approach_1 import (
    DEFAULT_MAX_DEPTH as APPROACH_1_DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_SPAN_WIDTH as APPROACH_1_DEFAULT_MIN_SPAN_WIDTH,
    PeakErrorApproach1Step,
)
from single_step_degree_reduction.protocol import DegreeReductionSingleStep

DEGREE_IN = 4
DEGREE_OUT = 2
N_STEPS = DEGREE_IN - DEGREE_OUT
BACKEND_CHOICES = ("single", "multiple_cuts", "approach_1")
MAX_LEGEND_LEAVES = 10


def degree_4_example_names() -> tuple[str, ...]:
    """All ``nurbs_curve_examples`` curve names defined at degree 4."""
    return curves_for_degree(DEGREE_IN)


def build_degree_4_curve(name: str):
    qw, knotvector, degree = load_example(name, degree=DEGREE_IN)
    weights = load_example_weights(name, degree=DEGREE_IN)
    return build_geomdl_from_geometry(
        qw,
        knotvector,
        degree,
        weights=weights,
        name=f"{name}_p{degree}",
    )


def build_single_step_backend(
    backend_name: str,
    split_us: Sequence[float],
    *,
    max_depth: int = APPROACH_1_DEFAULT_MAX_DEPTH,
    min_span_width: float = APPROACH_1_DEFAULT_MIN_SPAN_WIDTH,
) -> DegreeReductionSingleStep:
    if backend_name == "single":
        if len(split_us) != 1:
            raise ValueError(
                f"backend 'single' requires exactly one --split-u value, got {len(split_us)}"
            )
        return DummySplitReduceStep(split_u=float(split_us[0]))
    if backend_name == "multiple_cuts":
        return DummyMultipleCutsStep(split_us=tuple(float(u) for u in split_us))
    if backend_name == "approach_1":
        return PeakErrorApproach1Step(
            max_depth=int(max_depth),
            min_span_width=float(min_span_width),
        )
    raise ValueError(f"unknown backend {backend_name!r}")


def expected_max_leaves(backend_name: str, split_us: Sequence[float]) -> int | None:
    if backend_name == "approach_1":
        return None
    branches = 2 if backend_name == "single" else len(split_us) + 1
    return int(branches**N_STEPS)


def default_save_path(curve_name: str, backend_name: str) -> Path:
    from multiple_step_degree_reduction.paths import resolve_curve_save_path

    path = resolve_curve_save_path(
        None,
        name=f"degree_4_to_2_{curve_name}_{backend_name}",
        show=False,
    )
    assert path is not None
    return path


def plot_degree_4_to_2_result(
    input_curve,
    result,
    *,
    curve_name: str,
    backend_name: str,
    split_us: Sequence[float] | None,
    max_error: float,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    cmap = plt.get_cmap("tab20")
    fig, (ax_input, ax_leaves) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    plot_curve_overlay(
        ax_input,
        input_curve,
        color=COLOR_INPUT,
        cp_color=COLOR_INPUT_CP,
        label=f"input p={DEGREE_IN} ({len(input_curve.ctrlpts)} CPs)",
        lw=2.4,
    )
    ax_input.set_title(f"{curve_name}: degree-{DEGREE_IN} input")
    ax_input.set_aspect("equal", adjustable="datalim")
    ax_input.grid(True, alpha=0.3)
    ax_input.legend(loc="best", fontsize=8)

    show_legend = len(result.segments) <= MAX_LEGEND_LEAVES
    for index, leaf in enumerate(result.segments):
        color = cmap(index % 20)
        props = curve_properties_from_curve(leaf.curve)
        label = (
            f"leaf {index + 1} p={props.degree} ({props.n_control_points} CPs, "
            f"err={leaf.total_error:.4g})"
            if show_legend
            else "_nolegend_"
        )
        plot_curve_overlay(
            ax_leaves,
            leaf.curve,
            color=color,
            cp_color=color,
            label=label,
            lw=1.8,
            ls="-",
            cp_marker="o",
        )

    tol_label = "inf" if not np.isfinite(max_error) else f"{max_error:g}"
    if backend_name == "approach_1":
        leaves_detail = f"peak-error adaptive, max_error={tol_label}"
    else:
        cuts_label = ", ".join(f"{u:g}" for u in (split_us or ()))
        leaves_detail = f"cuts: {cuts_label}, max_error={tol_label}"
    ax_leaves.set_title(
        f"{len(result.segments)} degree-{DEGREE_OUT} leaves "
        f"({backend_name}, {leaves_detail})"
    )
    ax_leaves.set_aspect("equal", adjustable="datalim")
    ax_leaves.grid(True, alpha=0.3)
    if show_legend:
        ax_leaves.legend(loc="best", fontsize=7)
    else:
        ax_leaves.text(
            0.02,
            0.98,
            f"{len(result.segments)} leaves (legend omitted)",
            transform=ax_leaves.transAxes,
            va="top",
            fontsize=8,
        )

    fig.suptitle(
        f"Algorithm 1: {curve_name} p={DEGREE_IN} → p={DEGREE_OUT} "
        f"via {backend_name}",
        fontsize=11,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"saved {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def run_degree_4_to_2(
    curve_name: str,
    *,
    backend_name: str = "single",
    split_us: Sequence[float] = (0.5,),
    max_error: float = float("inf"),
    max_depth: int = APPROACH_1_DEFAULT_MAX_DEPTH,
    min_span_width: float = APPROACH_1_DEFAULT_MIN_SPAN_WIDTH,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Run multi-step reduction and optional plot for one degree-4 example."""
    if curve_name not in degree_4_example_names():
        raise ValueError(
            f"unknown degree-{DEGREE_IN} example {curve_name!r}; "
            f"choose from {degree_4_example_names()}"
        )

    curve = build_degree_4_curve(curve_name)
    backend = build_single_step_backend(
        backend_name,
        split_us,
        max_depth=max_depth,
        min_span_width=min_span_width,
    )
    max_leaves = expected_max_leaves(backend_name, split_us)

    print(
        f"\n=== {curve_name} ({backend_name}) ===\n"
        f"input: degree={curve.degree} n_cp={len(curve.ctrlpts)} domain={curve.domain}"
    )
    if backend_name == "approach_1":
        print(
            f"peak-error step: max_depth={max_depth}, min_span_width={min_span_width:g}"
        )
    else:
        print(f"cuts per step: {tuple(float(u) for u in split_us)}")
    leaves_hint = (
        "variable (depends on max_error and splits)"
        if max_leaves is None
        else f"up to {max_leaves}"
    )
    print(
        f"target: degree={DEGREE_OUT} in {N_STEPS} steps, "
        f"max_error={max_error}, {leaves_hint} leaves"
    )

    result = degree_reduce_multiple_steps(
        curve,
        N_STEPS,
        float(max_error),
        single_step_backend=backend,
    )

    print(f"leaves: {len(result.segments)}")
    for index, leaf in enumerate(result.segments):
        props = curve_properties_from_curve(leaf.curve)
        print(
            f"  leaf{index}: degree={props.degree} n_cp={props.n_control_points} "
            f"total_error={leaf.total_error:.6g}"
        )

    out_path = save_path
    if out_path is None and not show:
        out_path = default_save_path(curve_name, backend_name)

    plot_degree_4_to_2_result(
        curve,
        result,
        curve_name=curve_name,
        backend_name=backend_name,
        split_us=None if backend_name == "approach_1" else split_us,
        max_error=float(max_error),
        save_path=out_path,
        show=show,
    )


def _default_split_us(backend_name: str) -> tuple[float, ...]:
    if backend_name == "multiple_cuts":
        return DEFAULT_SPLIT_US
    if backend_name == "approach_1":
        return ()
    return (0.5,)


def main() -> None:
    example_names = degree_4_example_names()
    parser = argparse.ArgumentParser(
        description=(
            f"Multi-step reduction of degree-{DEGREE_IN} nurbs examples "
            f"to degree-{DEGREE_OUT}."
        ),
    )
    parser.add_argument(
        "-c",
        "--curve",
        choices=example_names,
        default=None,
        help=f"single degree-{DEGREE_IN} example (default: s_shaped, or all with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"run every degree-{DEGREE_IN} example ({', '.join(example_names)})",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="single",
        help="Algorithm 2 backend (default: single)",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=float("inf"),
        help="global error budget per branch (default: inf; try 0.15 for approach_1)",
    )
    parser.add_argument(
        "--split-u",
        type=float,
        nargs="+",
        default=None,
        metavar="U",
        help="split parameter(s) for single/multiple_cuts backends only "
        "(default: 0.5 for single, 0.25 0.5 0.75 for multiple_cuts)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=APPROACH_1_DEFAULT_MAX_DEPTH,
        help=f"approach_1: max recursive splits per branch (default: {APPROACH_1_DEFAULT_MAX_DEPTH})",
    )
    parser.add_argument(
        "--min-span-width",
        type=float,
        default=APPROACH_1_DEFAULT_MIN_SPAN_WIDTH,
        help=f"approach_1: minimum parameter span before forced accept (default: {APPROACH_1_DEFAULT_MIN_SPAN_WIDTH:g})",
    )
    parser.add_argument(
        "-o",
        "--save",
        type=Path,
        default=None,
        help="output image (single-curve runs only; use with one of -c or default)",
    )
    parser.add_argument("-n", "--no-show", action="store_true")
    args = parser.parse_args()

    if args.all and args.curve is not None:
        parser.error("use either --all or -c/--curve, not both")
    if args.all and args.save is not None:
        parser.error("--save applies to a single curve; omit it when using --all")

    split_us = (
        tuple(float(u) for u in args.split_u)
        if args.split_u is not None
        else _default_split_us(args.backend)
    )
    if args.backend == "single" and len(split_us) != 1:
        parser.error("backend 'single' requires exactly one --split-u value")
    if args.backend == "approach_1" and args.split_u is not None:
        parser.error("--split-u is not used with backend 'approach_1'")

    show = not args.no_show
    curves_to_run = list(example_names) if args.all else [args.curve or example_names[0]]

    for name in curves_to_run:
        save_path = args.save if len(curves_to_run) == 1 else None
        run_degree_4_to_2(
            name,
            backend_name=args.backend,
            split_us=split_us,
            max_error=float(args.max_error),
            max_depth=int(args.max_depth),
            min_span_width=float(args.min_span_width),
            save_path=save_path,
            show=show,
        )


if __name__ == "__main__":
    main()
