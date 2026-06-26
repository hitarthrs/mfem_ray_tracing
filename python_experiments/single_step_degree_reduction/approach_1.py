"""
Approach 1 (single-step): peak-error driven ``p → p-1`` reduction.

Pipeline on each high-order piece:

  1. Run A5.11 once (probe) and read ``error_array``.
  2. If ``max(error_array) <= max_error`` → accept one reduced segment for this interval.
  3. Else split at the midpoint of the max-error knot span, process the **left**
     sub-interval first, then the **right** (left-to-right freeze order).

Splitting uses ``extract_subcurve_global``; split sites come from
``midpoint_knot_for_max_error`` (same rule as adaptive knot refinement).

Run::

    cd python_experiments
    .venv/bin/python -m single_step_degree_reduction.approach_1
    .venv/bin/python -m single_step_degree_reduction.approach_1 --all -n
    .venv/bin/python -m single_step_degree_reduction.approach_1 -c multiple_peak --max-error 0.2
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_PYTHON_EXPERIMENTS = Path(__file__).resolve().parent.parent
if str(_PYTHON_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_PYTHON_EXPERIMENTS))

import numpy as np

from knot_refinement_algorithms.adaptive_error_knot_refinement import (
    midpoint_knot_for_max_error,
)
from knot_refinement_algorithms.recursive_curve_degree_reduction import (
    extract_subcurve_global,
)

from b_spline_curve_reduction import DegreeReduceCurve
from knot_refinement_experiments.common import build_geomdl_from_geometry
from nurbs_degree_reduction import degree_reduce_nurbs

try:
    from .dummy_approach import (
        CurveProperties,
        curve_properties_from_curve,
        plot_curve_overlay,
        segment_error_from_reduction,
    )
    from .protocol import (
        Curve,
        SingleStepReductionResult,
        single_step_result_from_lists,
    )
except ImportError:
    from single_step_degree_reduction.dummy_approach import (
        CurveProperties,
        curve_properties_from_curve,
        plot_curve_overlay,
        segment_error_from_reduction,
    )
    from single_step_degree_reduction.protocol import (
        Curve,
        SingleStepReductionResult,
        single_step_result_from_lists,
    )

DEFAULT_MAX_DEPTH = 25
DEFAULT_MIN_SPAN_WIDTH = 1e-8
MAX_LEGEND_SEGMENTS = 10


@dataclass(frozen=True)
class _PeakErrorLeaf:
    """Internal accepted leaf with root parameter interval."""

    curve: Curve
    segment_error: float
    u_domain: tuple[float, float]


def local_param_to_global(
    u_local: float,
    piece_global: tuple[float, float],
    local_domain: tuple[float, float],
) -> float:
    """Affine map from the current geomdl domain into root/global parameters."""
    g0, g1 = piece_global
    l0, l1 = local_domain
    if abs(l1 - l0) < 1e-15:
        return float(g0)
    t = (float(u_local) - l0) / (l1 - l0)
    return float(g0 + t * (g1 - g0))


def _max_error(error_array: np.ndarray) -> float:
    err = np.asarray(error_array, dtype=float)
    if err.size == 0:
        return 0.0
    return float(np.max(err))


def _run_a5_11(
    props: CurveProperties,
    *,
    tol: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Run A5.11; return ``(Pw, Uh, weights_out, error_array)``."""
    n = props.n_control_points
    p = props.degree
    if p < 2:
        raise ValueError(f"degree reduction requires degree >= 2, got {p}")

    if props.weights is None:
        out = DegreeReduceCurve(n, p, props.knotvector, props.control_points, tol=tol)
        if out == 1:
            raise RuntimeError("A5.11 tolerance exceeded on polynomial segment")
        pw, uh, err = out
        return pw, uh, None, np.asarray(err, dtype=float)

    out = degree_reduce_nurbs(
        n,
        p,
        props.knotvector,
        props.control_points,
        props.weights,
        tol=tol,
    )
    if out == 1:
        raise RuntimeError("A5.11 tolerance exceeded on NURBS segment")
    pw, weights_out, uh, err = out
    return pw, uh, weights_out, np.asarray(err, dtype=float)


def _build_reduced_curve(
    props: CurveProperties,
    pw: np.ndarray,
    uh: np.ndarray,
    weights_out: np.ndarray | None,
    err: np.ndarray,
    *,
    name: str,
) -> tuple[Curve, float]:
    reduced = build_geomdl_from_geometry(
        pw,
        uh,
        props.degree - 1,
        weights=weights_out,
        name=name,
    )
    return reduced, segment_error_from_reduction(err)


def _process_high_order_piece(
    high_order_curve: Curve,
    piece_global: tuple[float, float],
    *,
    max_error: float,
    max_depth: int,
    min_span_width: float,
    depth: int = 0,
    leaf_counter: list[int] | None = None,
) -> list[_PeakErrorLeaf]:
    """
    Recursively reduce one high-order interval; return leaves left-to-right.
    """
    counter = leaf_counter if leaf_counter is not None else [0]
    g0, g1 = float(piece_global[0]), float(piece_global[1])
    span_width = g1 - g0
    if span_width < min_span_width:
        raise RuntimeError(f"piece [{g0:g}, {g1:g}] narrower than min_span_width")

    props = curve_properties_from_curve(high_order_curve)
    if props.degree < 2:
        raise ValueError(f"peak-error step requires degree >= 2, got {props.degree}")

    pw_probe, uh_probe, w_probe, err_probe = _run_a5_11(props, tol=float("inf"))
    _ = pw_probe, uh_probe, w_probe
    peak_err = _max_error(err_probe)

    def accept_piece(*, tol: float, name: str) -> tuple[Curve, float]:
        pw, uh, w_out, err = _run_a5_11(props, tol=tol)
        return _build_reduced_curve(props, pw, uh, w_out, err, name=name)

    def accept_forced(reason: str) -> list[_PeakErrorLeaf]:
        idx = counter[0]
        counter[0] += 1
        reduced, seg_err = accept_piece(tol=max_error, name=f"peak-leaf-{idx}")
        _ = reason
        return [
            _PeakErrorLeaf(
                curve=reduced,
                segment_error=float(seg_err),
                u_domain=(g0, g1),
            )
        ]

    if peak_err <= max_error:
        idx = counter[0]
        counter[0] += 1
        reduced, seg_err = accept_piece(tol=max_error, name=f"peak-leaf-{idx}")
        return [
            _PeakErrorLeaf(
                curve=reduced,
                segment_error=float(seg_err),
                u_domain=(g0, g1),
            )
        ]

    site = midpoint_knot_for_max_error(props.knotvector, props.degree, err_probe)
    if site is None:
        return accept_forced("no valid midpoint for max-error knot")

    _m, _u_at, _u_next, u_mid = site
    local_domain = (float(high_order_curve.domain[0]), float(high_order_curve.domain[1]))
    u_split_local = float(u_mid)
    l0, l1 = local_domain
    eps = 1e-9 * max(1.0, abs(l1 - l0))

    if u_split_local <= l0 + eps or u_split_local >= l1 - eps:
        return accept_forced("split site on piece boundary")

    u_split_global = local_param_to_global(u_split_local, piece_global, local_domain)
    left_width = u_split_global - g0
    right_width = g1 - u_split_global
    if left_width < min_span_width or right_width < min_span_width:
        return accept_forced("split would produce sub-span below min_span_width")

    if depth >= max_depth:
        return accept_forced("max_depth reached")

    left_global = (g0, u_split_global)
    right_global = (u_split_global, g1)

    left_high = extract_subcurve_global(
        high_order_curve,
        piece_global,
        left_global[0],
        left_global[1],
    )
    right_high = extract_subcurve_global(
        high_order_curve,
        piece_global,
        right_global[0],
        right_global[1],
    )

    leaves_left = _process_high_order_piece(
        left_high,
        left_global,
        max_error=max_error,
        max_depth=max_depth,
        min_span_width=min_span_width,
        depth=depth + 1,
        leaf_counter=counter,
    )
    leaves_right = _process_high_order_piece(
        right_high,
        right_global,
        max_error=max_error,
        max_depth=max_depth,
        min_span_width=min_span_width,
        depth=depth + 1,
        leaf_counter=counter,
    )
    return leaves_left + leaves_right


def peak_error_single_step(
    initial_curve: Curve,
    max_error: float,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_span_width: float = DEFAULT_MIN_SPAN_WIDTH,
) -> SingleStepReductionResult:
    """
    Algorithm 2 — peak-error, left-to-right single degree drop.

    Returns one or more degree ``(p-1)`` segments covering the input curve.
    """
    if max_error < 0:
        raise ValueError(f"max_error must be non-negative, got {max_error}")

    piece_global = (
        float(initial_curve.domain[0]),
        float(initial_curve.domain[1]),
    )
    leaves = _process_high_order_piece(
        initial_curve,
        piece_global,
        max_error=float(max_error),
        max_depth=int(max_depth),
        min_span_width=float(min_span_width),
    )
    leaves_sorted = sorted(leaves, key=lambda leaf: leaf.u_domain[0])
    return single_step_result_from_lists(
        [leaf.curve for leaf in leaves_sorted],
        [leaf.segment_error for leaf in leaves_sorted],
    )


@dataclass
class PeakErrorApproach1Step:
    """Callable backend matching :class:`DegreeReductionSingleStep`."""

    max_depth: int = DEFAULT_MAX_DEPTH
    min_span_width: float = DEFAULT_MIN_SPAN_WIDTH

    def __call__(
        self,
        initial_curve: Curve,
        max_error: float,
    ) -> SingleStepReductionResult:
        return peak_error_single_step(
            initial_curve,
            max_error,
            max_depth=self.max_depth,
            min_span_width=self.min_span_width,
        )


def plot_peak_error_single_step(
    input_curve: Curve,
    result: SingleStepReductionResult,
    *,
    curve_name: str = "curve",
    max_error: float = float("inf"),
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Left: full input; right: accepted ``p-1`` segments overlaid left-to-right."""
    import matplotlib.pyplot as plt

    from knot_refinement_experiments.common import COLOR_INPUT, COLOR_INPUT_CP

    p_in = int(input_curve.degree)
    p_out = p_in - 1
    cmap = plt.get_cmap("tab20")
    show_legend = len(result.segments) <= MAX_LEGEND_SEGMENTS

    fig, (ax_input, ax_reduced) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    plot_curve_overlay(
        ax_input,
        input_curve,
        color=COLOR_INPUT,
        cp_color=COLOR_INPUT_CP,
        label=f"input p={p_in} ({len(input_curve.ctrlpts)} CPs)",
        lw=2.4,
    )
    ax_input.set_title(f"{curve_name}: degree-{p_in} input")
    ax_input.set_aspect("equal", adjustable="datalim")
    ax_input.grid(True, alpha=0.3)
    ax_input.legend(loc="best", fontsize=8)

    for index, segment in enumerate(result.segments):
        color = cmap(index % 20)
        label = (
            f"seg {index + 1} p={p_out} ({len(segment.curve.ctrlpts)} CPs, "
            f"err={segment.segment_error:.4g})"
            if show_legend
            else "_nolegend_"
        )
        plot_curve_overlay(
            ax_reduced,
            segment.curve,
            color=color,
            cp_color=color,
            label=label,
            lw=2.0,
            ls="-",
            cp_marker="o",
        )

    tol_label = "inf" if not np.isfinite(max_error) else f"{max_error:g}"
    ax_reduced.set_title(
        f"{len(result.segments)} accepted degree-{p_out} segments "
        f"(peak-error, max_error={tol_label})"
    )
    ax_reduced.set_aspect("equal", adjustable="datalim")
    ax_reduced.grid(True, alpha=0.3)
    if show_legend:
        ax_reduced.legend(loc="best", fontsize=7)
    else:
        ax_reduced.text(
            0.02,
            0.98,
            f"{len(result.segments)} segments (legend omitted)",
            transform=ax_reduced.transAxes,
            va="top",
            fontsize=8,
        )

    fig.suptitle(
        f"approach 1 (peak error): {curve_name}  |  p={p_in} → p={p_out}",
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


def run_degree_4_example(
    curve_name: str,
    *,
    max_error: float,
    max_depth: int = DEFAULT_MAX_DEPTH,
    save_path: Path | None = None,
    show: bool = True,
) -> SingleStepReductionResult:
    """Build one degree-4 example, run peak-error step, print + plot."""
    from knot_refinement_experiments.common import build_geomdl_from_geometry
    from nurbs_curve_examples import curves_for_degree, load_example, load_example_weights

    if curve_name not in curves_for_degree(4):
        raise ValueError(
            f"unknown degree-4 example {curve_name!r}; choose from {curves_for_degree(4)}"
        )

    qw, knotvector, degree = load_example(curve_name, degree=4)
    weights = load_example_weights(curve_name, degree=4)
    curve = build_geomdl_from_geometry(
        qw,
        knotvector,
        degree,
        weights=weights,
        name=curve_name,
    )

    result = peak_error_single_step(curve, float(max_error), max_depth=max_depth)

    print(f"\n=== {curve_name} (approach 1, peak error) ===")
    print(
        f"input: degree={curve.degree} n_cp={len(curve.ctrlpts)} "
        f"domain={curve.domain}"
    )
    print(f"max_error={max_error}")
    print(f"segments: {len(result.segments)}")
    for index, segment in enumerate(result.segments):
        props = curve_properties_from_curve(segment.curve)
        print(
            f"  seg{index}: degree={props.degree} n_cp={props.n_control_points} "
            f"error={segment.segment_error:.6g}"
        )

    plot_peak_error_single_step(
        curve,
        result,
        curve_name=curve_name,
        max_error=float(max_error),
        save_path=save_path,
        show=show,
    )
    return result


def _main() -> None:
    import argparse

    from nurbs_curve_examples import curves_for_degree

    example_names = curves_for_degree(4)
    parser = argparse.ArgumentParser(
        description="Approach 1: peak-error single-step reduction on degree-4 examples.",
    )
    parser.add_argument(
        "-c",
        "--curve",
        choices=example_names,
        default=None,
        help="degree-4 example (default: s_shaped, or all with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"run all degree-4 examples ({', '.join(example_names)})",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=0.15,
        help="peak A5.11 error tolerance per accepted segment (default: 0.15)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"maximum recursive splits per branch (default: {DEFAULT_MAX_DEPTH})",
    )
    parser.add_argument("-o", "--save", type=Path, default=None)
    parser.add_argument("-n", "--no-show", action="store_true")
    args = parser.parse_args()

    if args.all and args.curve is not None:
        parser.error("use either --all or -c/--curve, not both")
    if args.all and args.save is not None:
        parser.error("--save applies to a single curve; omit it when using --all")

    try:
        from .paths import resolve_curve_save_path
    except ImportError:
        from single_step_degree_reduction.paths import resolve_curve_save_path

    show = not args.no_show
    curves_to_run = list(example_names) if args.all else [args.curve or example_names[0]]

    for name in curves_to_run:
        save_path = resolve_curve_save_path(
            args.save if len(curves_to_run) == 1 else None,
            name=f"approach1_peak_error_{name}",
            show=show,
        )

        run_degree_4_example(
            name,
            max_error=float(args.max_error),
            max_depth=int(args.max_depth),
            save_path=save_path,
            show=show,
        )


if __name__ == "__main__":
    _main()
