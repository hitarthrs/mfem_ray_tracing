"""
Dummy Algorithm 2: split at several parameter values, then A5.11 on each piece.

Unlike ``dummy_approach`` (one cut → two segments), this backend cuts each
incoming curve at multiple ``u`` sites in its current parameter domain, producing
``len(split_us) + 1`` segments per single-step call, each reduced independently.

Run::

    cd python_experiments
    .venv/bin/python -m single_step_degree_reduction.dummy_multiple_cuts
    .venv/bin/python -m single_step_degree_reduction.dummy_multiple_cuts --split-u 0.33 0.66 -n
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_PYTHON_EXPERIMENTS = Path(__file__).resolve().parent.parent
if str(_PYTHON_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_PYTHON_EXPERIMENTS))

import numpy as np

from knot_refinement_algorithms.recursive_curve_degree_reduction import extract_subcurve

try:
    from .dummy_approach import (
        curve_properties_from_curve,
        plot_curve_overlay,
        point_at_u,
        reduce_curve_properties_once,
    )
    from .protocol import (
        Curve,
        SingleStepReductionResult,
        single_step_result_from_lists,
    )
except ImportError:
    from single_step_degree_reduction.dummy_approach import (
        curve_properties_from_curve,
        plot_curve_overlay,
        point_at_u,
        reduce_curve_properties_once,
    )
    from single_step_degree_reduction.protocol import (
        Curve,
        SingleStepReductionResult,
        single_step_result_from_lists,
    )

DEFAULT_SPLIT_US: tuple[float, ...] = (0.25, 0.5, 0.75)


def normalize_split_sites(
    split_us: Sequence[float],
    u_min: float,
    u_max: float,
    *,
    atol: float = 1e-9,
) -> tuple[float, ...]:
    """
    Sort, deduplicate, and validate interior split parameters.

    Each returned site lies strictly between ``u_min`` and ``u_max``.
    """
    if not split_us:
        raise ValueError("split_us must contain at least one parameter value")

    u0, u1 = float(u_min), float(u_max)
    if u1 <= u0 + atol:
        raise ValueError(f"curve domain [{u0}, {u1}] is degenerate")

    cuts: list[float] = []
    for raw in sorted(float(u) for u in split_us):
        if raw <= u0 + atol or raw >= u1 - atol:
            raise ValueError(
                f"split parameter u={raw:g} must lie strictly inside "
                f"domain [{u0:g}, {u1:g}]"
            )
        if cuts and abs(raw - cuts[-1]) <= atol:
            continue
        cuts.append(raw)
    return tuple(cuts)


def split_curve_into_segments(
    curve: Curve,
    split_us: Sequence[float] = DEFAULT_SPLIT_US,
    *,
    atol: float = 1e-9,
) -> tuple[Curve, ...]:
    """
    Split ``curve`` at ``split_us``; return pieces in increasing parameter order.

    ``split_us`` are expressed in the current curve parameter domain
    (typically ``[0, 1]`` after prior geomdl splits).
    """
    u_min, u_max = float(curve.domain[0]), float(curve.domain[1])
    cuts = normalize_split_sites(split_us, u_min, u_max, atol=atol)
    edges = (u_min, *cuts, u_max)
    pieces: list[Curve] = []
    for index in range(len(edges) - 1):
        lo, hi = float(edges[index]), float(edges[index + 1])
        if hi - lo <= atol:
            continue
        pieces.append(extract_subcurve(curve, lo, hi, atol=atol))
    if not pieces:
        raise RuntimeError("split produced no non-degenerate segments")
    return tuple(pieces)


def dummy_multiple_cuts_single_step(
    initial_curve: Curve,
    max_error: float,
    *,
    split_us: Sequence[float] = DEFAULT_SPLIT_US,
) -> SingleStepReductionResult:
    """
    Split at multiple ``u`` values, run one A5.11 pass per piece.

    Returns ``len(unique split_us) + 1`` reduced segments (fewer only if a
    sub-interval is degenerate).
    """
    props = curve_properties_from_curve(initial_curve)
    if props.degree < 2:
        raise ValueError(
            f"dummy multiple-cuts requires input degree >= 2, got {props.degree}"
        )

    pieces = split_curve_into_segments(initial_curve, split_us)
    reduced_curves: list[Curve] = []
    errors: list[float] = []

    for index, piece in enumerate(pieces):
        piece_props = curve_properties_from_curve(piece)
        reduced, err = reduce_curve_properties_once(
            piece_props,
            tol=max_error,
            name=f"dummy-mc-{index}",
        )
        reduced_curves.append(reduced)
        errors.append(err)

    return single_step_result_from_lists(reduced_curves, errors)


@dataclass
class DummyMultipleCutsStep:
    """Callable backend matching :class:`DegreeReductionSingleStep`."""

    split_us: tuple[float, ...] = DEFAULT_SPLIT_US

    def __call__(
        self,
        initial_curve: Curve,
        max_error: float,
    ) -> SingleStepReductionResult:
        return dummy_multiple_cuts_single_step(
            initial_curve,
            max_error,
            split_us=self.split_us,
        )


def plot_dummy_multiple_cuts(
    curve: Curve,
    result: SingleStepReductionResult,
    *,
    split_us: Sequence[float] = DEFAULT_SPLIT_US,
    curve_name: str = "curve",
    max_error: float = float("inf"),
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Input with cut markers; overlay each piece with its reduction."""
    import matplotlib.pyplot as plt

    from knot_refinement_experiments.common import COLOR_INPUT, COLOR_INPUT_CP

    pieces = split_curve_into_segments(curve, split_us)
    cuts = normalize_split_sites(
        split_us,
        float(curve.domain[0]),
        float(curve.domain[1]),
    )
    p_in = int(curve.degree)
    p_out = p_in - 1
    cmap = plt.get_cmap("tab10")

    fig, (ax_input, ax_overlay) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    plot_curve_overlay(
        ax_input,
        curve,
        color=COLOR_INPUT,
        cp_color=COLOR_INPUT_CP,
        label=f"input p={p_in} ({len(curve.ctrlpts)} CPs)",
        lw=2.4,
    )
    for cut in cuts:
        pt = point_at_u(curve, cut)
        ax_input.plot(
            pt[0],
            pt[1],
            "x",
            color="crimson",
            ms=9,
            mew=2.0,
        )
    ax_input.plot([], [], "x", color="crimson", label=f"cuts ({len(cuts)})")
    ax_input.set_title(f"{curve_name}: input p={p_in}")
    ax_input.set_aspect("equal", adjustable="datalim")
    ax_input.grid(True, alpha=0.3)
    ax_input.legend(loc="best", fontsize=8)

    for index, (piece, segment) in enumerate(zip(pieces, result.segments, strict=True)):
        color = cmap(index % 10)
        plot_curve_overlay(
            ax_overlay,
            piece,
            color=color,
            cp_color=color,
            label=f"seg {index + 1} input p={p_in} ({len(piece.ctrlpts)} CPs)",
            lw=2.2,
            ls="-",
            cp_marker="o",
        )
        plot_curve_overlay(
            ax_overlay,
            segment.curve,
            color=color,
            cp_color=color,
            label=(
                f"seg {index + 1} reduced p={p_out} "
                f"({len(segment.curve.ctrlpts)} CPs, err={segment.segment_error:.4g})"
            ),
            lw=2.0,
            ls="--",
            cp_marker="s",
        )

    tol_label = "inf" if not np.isfinite(max_error) else f"{max_error:g}"
    cuts_label = ", ".join(f"{u:g}" for u in cuts)
    ax_overlay.set_title(
        f"{len(result.segments)} segments (cuts: {cuts_label}; max_error={tol_label})"
    )
    ax_overlay.set_aspect("equal", adjustable="datalim")
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.legend(loc="best", fontsize=7)

    fig.suptitle(
        f"dummy multiple-cuts: {curve_name}  |  p={p_in} → p={p_out}",
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


def _main() -> None:
    import argparse

    from nurbs_curve_examples import curves_for_degree, load_example, load_example_weights
    try:
        from .paths import resolve_curve_save_path
    except ImportError:
        from single_step_degree_reduction.paths import resolve_curve_save_path

    try:
        from .protocol import degree_reduction_single_step
    except ImportError:
        from single_step_degree_reduction.protocol import degree_reduction_single_step

    from knot_refinement_experiments.common import build_geomdl_from_geometry

    parser = argparse.ArgumentParser(
        description="Dummy multi-cut single-step degree reduction demo.",
    )
    parser.add_argument(
        "-c",
        "--curve",
        choices=curves_for_degree(4),
        default="s_shaped",
        help="degree-4 nurbs_curve_examples curve",
    )
    parser.add_argument(
        "--split-u",
        type=float,
        nargs="+",
        default=list(DEFAULT_SPLIT_US),
        metavar="U",
        help=f"interior split parameters (default: {' '.join(f'{u:g}' for u in DEFAULT_SPLIT_US)})",
    )
    parser.add_argument("--max-error", type=float, default=float("inf"))
    parser.add_argument("-o", "--save", type=Path, default=None)
    parser.add_argument("-n", "--no-show", action="store_true")
    args = parser.parse_args()

    split_us = tuple(float(u) for u in args.split_u)
    qw, knotvector, degree = load_example(args.curve, degree=4)
    weights = load_example_weights(args.curve, degree=4)
    curve = build_geomdl_from_geometry(
        qw,
        knotvector,
        degree,
        weights=weights,
        name=args.curve,
    )

    backend = DummyMultipleCutsStep(split_us=split_us)
    result = degree_reduction_single_step(curve, float(args.max_error), backend=backend)

    print(f"input: {args.curve} degree={curve.degree} domain={curve.domain}")
    print(f"cuts: {split_us}")
    print(f"segments: {len(result.segments)}")
    for index, segment in enumerate(result.segments):
        props = curve_properties_from_curve(segment.curve)
        print(
            f"  seg{index}: degree={props.degree} n_cp={props.n_control_points} "
            f"error={segment.segment_error:.6g}"
        )

    save_path = resolve_curve_save_path(
        args.save,
        name=f"dummy_multiple_cuts_{args.curve}",
        show=not args.no_show,
    )
    plot_dummy_multiple_cuts(
        curve,
        result,
        split_us=split_us,
        curve_name=args.curve,
        max_error=float(args.max_error),
        save_path=save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    _main()
