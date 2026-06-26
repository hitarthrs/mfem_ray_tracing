"""
Approach 2 (single-step): top-k peak-error splits with tolerance filter.

On each high-order piece:

  1. Run A5.11 once (probe) and read ``error_array``.
  2. If ``max(error_array) <= max_error`` → accept one reduced segment.
  3. Else collect up to ``top_k`` **distinct** split midpoints from knot spans with
     ``error_array[m] > max_error`` (filter C).
  4. Partition the piece at **all** planned sites in one pass (strategy A), then
     accept-or-recurse on each sub-interval left-to-right.

Split sites use ``midpoint_knot_for_index`` (same rule as adaptive knot refinement).
Geometry extraction uses ``extract_subcurve_global``.

Run::

    cd python_experiments
    .venv/bin/python -m single_step_degree_reduction.approach_2
    .venv/bin/python -m single_step_degree_reduction.approach_2 --all -n
    .venv/bin/python -m single_step_degree_reduction.approach_2 -c multiple_peak --top-k 3

Compare speed vs approach 1::

    .venv/bin/python -m single_step_degree_reduction.compare_approach_1_vs_2
    .venv/bin/python -m single_step_degree_reduction.compare_approach_1_vs_2 --all
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
    midpoint_knot_for_index,
    midpoint_knot_for_max_error,
)
from knot_refinement_algorithms.recursive_curve_degree_reduction import (
    extract_subcurve_global,
)

try:
    from . import approach_1 as _a1
    from .dummy_approach import curve_properties_from_curve, plot_curve_overlay
    from .protocol import Curve, SingleStepReductionResult, single_step_result_from_lists
except ImportError:
    import single_step_degree_reduction.approach_1 as _a1
    from single_step_degree_reduction.dummy_approach import (
        curve_properties_from_curve,
        plot_curve_overlay,
    )
    from single_step_degree_reduction.protocol import (
        Curve,
        SingleStepReductionResult,
        single_step_result_from_lists,
    )

DEFAULT_MAX_DEPTH = _a1.DEFAULT_MAX_DEPTH
DEFAULT_MIN_SPAN_WIDTH = _a1.DEFAULT_MIN_SPAN_WIDTH
MAX_LEGEND_SEGMENTS = _a1.MAX_LEGEND_SEGMENTS
DEFAULT_TOP_K = 3

_PeakErrorLeaf = _a1._PeakErrorLeaf
local_param_to_global = _a1.local_param_to_global
_max_error = _a1._max_error
_run_a5_11 = _a1._run_a5_11
_build_reduced_curve = _a1._build_reduced_curve


def plan_top_k_split_sites(
    knotvector: np.ndarray,
    degree: int,
    error_array: np.ndarray,
    max_error: float,
    top_k: int,
    *,
    atol: float = 1e-9,
) -> list[tuple[int, float, float, float, float]]:
    """
    Plan up to ``top_k`` distinct midpoint splits for spans above ``max_error``.

    Returns ``(m, err_m, u_at, u_next, u_mid)`` sorted by increasing ``u_mid``.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    err = np.asarray(error_array, dtype=float)
    u = np.asarray(knotvector, dtype=float)
    if err.size == 0:
        return []

    order = np.argsort(err)[::-1]
    planned: list[tuple[int, float, float, float, float]] = []
    seen_mids: list[float] = []

    for m in order:
        if float(err[m]) <= float(max_error):
            break
        if len(planned) >= top_k:
            break

        site = midpoint_knot_for_index(u, int(degree), int(m))
        if site is None:
            continue

        u_at, u_next, u_mid = site
        if any(abs(float(u_mid) - s) <= atol for s in seen_mids):
            continue

        planned.append((int(m), float(err[m]), float(u_at), float(u_next), float(u_mid)))
        seen_mids.append(float(u_mid))

    planned.sort(key=lambda row: row[4])
    return planned


def _global_split_positions(
    planned: list[tuple[int, float, float, float, float]],
    *,
    high_order_curve: Curve,
    piece_global: tuple[float, float],
    min_span_width: float,
) -> list[float] | None:
    """Map planned local midpoints to strictly interior global split positions."""
    g0, g1 = float(piece_global[0]), float(piece_global[1])
    local_domain = (float(high_order_curve.domain[0]), float(high_order_curve.domain[1]))
    l0, l1 = local_domain
    eps = 1e-9 * max(1.0, abs(l1 - l0), abs(g1 - g0))

    globals_out: list[float] = []
    for _m, _err_m, _u_at, _u_next, u_mid in planned:
        u_split_local = float(u_mid)
        if u_split_local <= l0 + eps or u_split_local >= l1 - eps:
            continue

        u_split_global = local_param_to_global(u_split_local, piece_global, local_domain)
        if u_split_global <= g0 + eps or u_split_global >= g1 - eps:
            continue
        if globals_out and u_split_global <= globals_out[-1] + eps:
            continue
        if u_split_global - (globals_out[-1] if globals_out else g0) < min_span_width:
            continue
        if g1 - u_split_global < min_span_width:
            continue

        globals_out.append(float(u_split_global))

    if not globals_out:
        return None
    return globals_out


def _binary_fallback_split(
    high_order_curve: Curve,
    piece_global: tuple[float, float],
    props,
    err_probe: np.ndarray,
    *,
    min_span_width: float,
) -> float | None:
    """Fallback to a single max-error midpoint when top-k planning yields nothing."""
    site = midpoint_knot_for_max_error(props.knotvector, props.degree, err_probe)
    if site is None:
        return None

    _m, _u_at, _u_next, u_mid = site
    local_domain = (float(high_order_curve.domain[0]), float(high_order_curve.domain[1]))
    l0, l1 = local_domain
    eps = 1e-9 * max(1.0, abs(l1 - l0))

    u_split_local = float(u_mid)
    if u_split_local <= l0 + eps or u_split_local >= l1 - eps:
        return None

    u_split_global = local_param_to_global(u_split_local, piece_global, local_domain)
    g0, g1 = float(piece_global[0]), float(piece_global[1])
    if u_split_global - g0 < min_span_width or g1 - u_split_global < min_span_width:
        return None
    return float(u_split_global)


def _process_high_order_piece(
    high_order_curve: Curve,
    piece_global: tuple[float, float],
    *,
    max_error: float,
    top_k: int,
    max_depth: int,
    min_span_width: float,
    depth: int = 0,
    leaf_counter: list[int] | None = None,
) -> list[_PeakErrorLeaf]:
    """Recursively reduce one high-order interval; return leaves left-to-right."""
    counter = leaf_counter if leaf_counter is not None else [0]
    g0, g1 = float(piece_global[0]), float(piece_global[1])
    span_width = g1 - g0
    if span_width < min_span_width:
        raise RuntimeError(f"piece [{g0:g}, {g1:g}] narrower than min_span_width")

    props = curve_properties_from_curve(high_order_curve)
    if props.degree < 2:
        raise ValueError(f"top-k peak-error step requires degree >= 2, got {props.degree}")

    pw_probe, uh_probe, w_probe, err_probe = _run_a5_11(props, tol=float("inf"))
    _ = pw_probe, uh_probe, w_probe
    peak_err = _max_error(err_probe)

    def accept_piece(*, tol: float, name: str):
        pw, uh, w_out, err = _run_a5_11(props, tol=tol)
        return _build_reduced_curve(props, pw, uh, w_out, err, name=name)

    def accept_forced(reason: str) -> list[_PeakErrorLeaf]:
        idx = counter[0]
        counter[0] += 1
        reduced, seg_err = accept_piece(tol=max_error, name=f"topk-leaf-{idx}")
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
        reduced, seg_err = accept_piece(tol=max_error, name=f"topk-leaf-{idx}")
        return [
            _PeakErrorLeaf(
                curve=reduced,
                segment_error=float(seg_err),
                u_domain=(g0, g1),
            )
        ]

    if depth >= max_depth:
        return accept_forced("max_depth reached")

    planned = plan_top_k_split_sites(
        props.knotvector,
        props.degree,
        err_probe,
        max_error,
        top_k,
    )
    split_globals = (
        _global_split_positions(
            planned,
            high_order_curve=high_order_curve,
            piece_global=piece_global,
            min_span_width=min_span_width,
        )
        if planned
        else None
    )

    if split_globals is None:
        u_fallback = _binary_fallback_split(
            high_order_curve,
            piece_global,
            props,
            err_probe,
            min_span_width=min_span_width,
        )
        if u_fallback is None:
            return accept_forced("no valid split site")
        split_globals = [u_fallback]

    edges = [g0, *split_globals, g1]
    leaves: list[_PeakErrorLeaf] = []
    for index in range(len(edges) - 1):
        sub_lo, sub_hi = float(edges[index]), float(edges[index + 1])
        if sub_hi - sub_lo < min_span_width:
            continue

        sub_global = (sub_lo, sub_hi)
        sub_high = extract_subcurve_global(
            high_order_curve,
            piece_global,
            sub_global[0],
            sub_global[1],
        )
        leaves.extend(
            _process_high_order_piece(
                sub_high,
                sub_global,
                max_error=max_error,
                top_k=top_k,
                max_depth=max_depth,
                min_span_width=min_span_width,
                depth=depth + 1,
                leaf_counter=counter,
            )
        )

    if not leaves:
        return accept_forced("partition produced no sub-pieces")
    return leaves


def top_k_peak_error_single_step(
    initial_curve: Curve,
    max_error: float,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_span_width: float = DEFAULT_MIN_SPAN_WIDTH,
) -> SingleStepReductionResult:
    """
    Algorithm 2 — top-k peak-error filter + multi-split single degree drop.

    Returns one or more degree ``(p-1)`` segments covering the input curve.
    """
    if max_error < 0:
        raise ValueError(f"max_error must be non-negative, got {max_error}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    piece_global = (
        float(initial_curve.domain[0]),
        float(initial_curve.domain[1]),
    )
    leaves = _process_high_order_piece(
        initial_curve,
        piece_global,
        max_error=float(max_error),
        top_k=int(top_k),
        max_depth=int(max_depth),
        min_span_width=float(min_span_width),
    )
    leaves_sorted = sorted(leaves, key=lambda leaf: leaf.u_domain[0])
    return single_step_result_from_lists(
        [leaf.curve for leaf in leaves_sorted],
        [leaf.segment_error for leaf in leaves_sorted],
    )


@dataclass
class PeakErrorTopKApproach2Step:
    """Callable backend matching :class:`DegreeReductionSingleStep`."""

    top_k: int = DEFAULT_TOP_K
    max_depth: int = DEFAULT_MAX_DEPTH
    min_span_width: float = DEFAULT_MIN_SPAN_WIDTH

    def __call__(
        self,
        initial_curve: Curve,
        max_error: float,
    ) -> SingleStepReductionResult:
        return top_k_peak_error_single_step(
            initial_curve,
            max_error,
            top_k=self.top_k,
            max_depth=self.max_depth,
            min_span_width=self.min_span_width,
        )


def plot_top_k_peak_error_single_step(
    input_curve: Curve,
    result: SingleStepReductionResult,
    *,
    curve_name: str = "curve",
    max_error: float = float("inf"),
    top_k: int = DEFAULT_TOP_K,
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
        f"(top-{top_k} peak-error, max_error={tol_label})"
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
        f"approach 2 (top-{top_k} peak error): {curve_name}  |  p={p_in} → p={p_out}",
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
    top_k: int = DEFAULT_TOP_K,
    max_depth: int = DEFAULT_MAX_DEPTH,
    save_path: Path | None = None,
    show: bool = True,
) -> SingleStepReductionResult:
    """Build one degree-4 example, run top-k peak-error step, print + plot."""
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

    result = top_k_peak_error_single_step(
        curve,
        float(max_error),
        top_k=top_k,
        max_depth=max_depth,
    )

    print(f"\n=== {curve_name} (approach 2, top-{top_k} peak error) ===")
    print(
        f"input: degree={curve.degree} n_cp={len(curve.ctrlpts)} "
        f"domain={curve.domain}"
    )
    print(f"max_error={max_error}, top_k={top_k}")
    print(f"segments: {len(result.segments)}")
    for index, segment in enumerate(result.segments):
        props = curve_properties_from_curve(segment.curve)
        print(
            f"  seg{index}: degree={props.degree} n_cp={props.n_control_points} "
            f"error={segment.segment_error:.6g}"
        )

    plot_top_k_peak_error_single_step(
        curve,
        result,
        curve_name=curve_name,
        max_error=float(max_error),
        top_k=top_k,
        save_path=save_path,
        show=show,
    )
    return result


def _main() -> None:
    import argparse

    from nurbs_curve_examples import curves_for_degree

    example_names = curves_for_degree(4)
    parser = argparse.ArgumentParser(
        description="Approach 2: top-k peak-error single-step reduction on degree-4 examples.",
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
        help="A5.11 error tolerance per accepted segment (default: 0.15)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"max distinct split sites per partition pass (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"maximum recursive partition depth per branch (default: {DEFAULT_MAX_DEPTH})",
    )
    parser.add_argument("-o", "--save", type=Path, default=None)
    parser.add_argument("-n", "--no-show", action="store_true")
    args = parser.parse_args()

    if args.all and args.curve is not None:
        parser.error("use either --all or -c/--curve, not both")
    if args.all and args.save is not None:
        parser.error("--save applies to a single curve; omit it when using --all")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    try:
        from .paths import resolve_curve_save_path
    except ImportError:
        from single_step_degree_reduction.paths import resolve_curve_save_path

    show = not args.no_show
    curves_to_run = list(example_names) if args.all else [args.curve or example_names[0]]

    for name in curves_to_run:
        save_path = resolve_curve_save_path(
            args.save if len(curves_to_run) == 1 else None,
            name=f"approach2_top{args.top_k}_{name}",
            show=show,
        )

        run_degree_4_example(
            name,
            max_error=float(args.max_error),
            top_k=int(args.top_k),
            max_depth=int(args.max_depth),
            save_path=save_path,
            show=show,
        )


if __name__ == "__main__":
    _main()
