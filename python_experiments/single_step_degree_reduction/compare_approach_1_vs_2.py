#!/usr/bin/env python3
"""
Benchmark approach 1 (binary peak-error) vs approach 2 (top-k multi-split).

Reports wall time, segment count, and max segment error for each run.
Use ``--repeat`` for a rough average (first run may include import/warmup cost).

    cd python_experiments
    .venv/bin/python -m single_step_degree_reduction.compare_approach_1_vs_2
    .venv/bin/python -m single_step_degree_reduction.compare_approach_1_vs_2 --all
    .venv/bin/python -m single_step_degree_reduction.compare_approach_1_vs_2 \\
        -c multiple_peak --max-error 0.15 --top-k 3 --repeat 5
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_PYTHON_EXPERIMENTS = Path(__file__).resolve().parent.parent
if str(_PYTHON_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_PYTHON_EXPERIMENTS))

import numpy as np

from knot_refinement_experiments.common import build_geomdl_from_geometry
from nurbs_curve_examples import curves_for_degree, load_example, load_example_weights
from single_step_degree_reduction.approach_1 import peak_error_single_step
from single_step_degree_reduction.approach_2 import (
    DEFAULT_TOP_K,
    top_k_peak_error_single_step,
)
from single_step_degree_reduction.protocol import SingleStepReductionResult


@dataclass(frozen=True)
class BenchRow:
    approach: str
    curve: str
    segments: int
    max_segment_error: float
    mean_segment_error: float
    elapsed_s: float
    repeats: int


def build_degree_4_curve(name: str):
    qw, knotvector, degree = load_example(name, degree=4)
    weights = load_example_weights(name, degree=4)
    return build_geomdl_from_geometry(
        qw,
        knotvector,
        degree,
        weights=weights,
        name=name,
    )


def _summarize_result(result: SingleStepReductionResult) -> tuple[int, float, float]:
    errors = [float(seg.segment_error) for seg in result.segments]
    if not errors:
        return 0, 0.0, 0.0
    return len(errors), float(max(errors)), float(np.mean(errors))


def _time_run(fn, repeats: int) -> tuple[SingleStepReductionResult, float]:
    # One warmup call (not timed) so repeat=1 measures steady state after imports.
    result = fn()
    if repeats <= 1:
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        return result, elapsed

    start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    elapsed = (time.perf_counter() - start) / repeats
    return result, elapsed


def bench_curve(
    curve_name: str,
    *,
    max_error: float,
    top_k: int,
    max_depth: int,
    repeats: int,
) -> tuple[BenchRow, BenchRow]:
    curve = build_degree_4_curve(curve_name)

    def run_a1():
        return peak_error_single_step(
            curve,
            float(max_error),
            max_depth=max_depth,
        )

    def run_a2():
        return top_k_peak_error_single_step(
            curve,
            float(max_error),
            top_k=top_k,
            max_depth=max_depth,
        )

    res1, t1 = _time_run(run_a1, repeats)
    res2, t2 = _time_run(run_a2, repeats)

    n1, max1, mean1 = _summarize_result(res1)
    n2, max2, mean2 = _summarize_result(res2)

    row1 = BenchRow(
        approach="approach_1",
        curve=curve_name,
        segments=n1,
        max_segment_error=max1,
        mean_segment_error=mean1,
        elapsed_s=t1,
        repeats=repeats,
    )
    row2 = BenchRow(
        approach=f"approach_2 (k={top_k})",
        curve=curve_name,
        segments=n2,
        max_segment_error=max2,
        mean_segment_error=mean2,
        elapsed_s=t2,
        repeats=repeats,
    )
    return row1, row2


def _print_row(row: BenchRow) -> None:
    repeat_label = f"avg/{row.repeats}" if row.repeats > 1 else "1 run"
    print(
        f"  {row.approach:22s}  "
        f"segments={row.segments:3d}  "
        f"max_err={row.max_segment_error:.6g}  "
        f"mean_err={row.mean_segment_error:.6g}  "
        f"time={row.elapsed_s * 1000:.2f} ms ({repeat_label})"
    )


def _print_speedup(row1: BenchRow, row2: BenchRow) -> None:
    if row1.elapsed_s <= 0 or row2.elapsed_s <= 0:
        return
    ratio = row2.elapsed_s / row1.elapsed_s
    faster = "approach_1" if ratio > 1 else "approach_2"
    factor = ratio if ratio > 1 else 1.0 / ratio
    print(f"  speed: {faster} ~{factor:.2f}x faster than the other")


def main() -> None:
    example_names = curves_for_degree(4)
    parser = argparse.ArgumentParser(
        description="Compare approach 1 vs approach 2 (single-step, degree 4).",
    )
    parser.add_argument(
        "-c",
        "--curve",
        choices=example_names,
        default=None,
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--max-error",
        type=float,
        default=0.15,
        help="A5.11 tolerance per accepted segment (default: 0.15)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"approach 2 split budget (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=25,
        help="max recursive depth for both approaches (default: 25)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="timed repetitions per approach (default: 3, averaged)",
    )
    args = parser.parse_args()

    if args.all and args.curve is not None:
        parser.error("use either --all or -c/--curve, not both")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    curves = list(example_names) if args.all else [args.curve or example_names[0]]

    print(
        f"settings: max_error={args.max_error}, top_k={args.top_k}, "
        f"max_depth={args.max_depth}, repeat={args.repeat}"
    )

    all_rows: list[tuple[BenchRow, BenchRow]] = []
    for name in curves:
        print(f"\n=== {name} ===")
        row1, row2 = bench_curve(
            name,
            max_error=float(args.max_error),
            top_k=int(args.top_k),
            max_depth=int(args.max_depth),
            repeats=int(args.repeat),
        )
        _print_row(row1)
        _print_row(row2)
        _print_speedup(row1, row2)
        all_rows.append((row1, row2))

    if len(all_rows) > 1:
        total_t1 = sum(r1.elapsed_s for r1, _ in all_rows)
        total_t2 = sum(r2.elapsed_s for _, r2 in all_rows)
        total_n1 = sum(r1.segments for r1, _ in all_rows)
        total_n2 = sum(r2.segments for _, r2 in all_rows)
        print("\n=== totals (all curves) ===")
        print(f"  approach_1: {total_n1} segments, {total_t1 * 1000:.1f} ms")
        print(f"  approach_2: {total_n2} segments, {total_t2 * 1000:.1f} ms")
        if total_t1 > 0 and total_t2 > 0:
            ratio = total_t2 / total_t1
            faster = "approach_1" if ratio > 1 else "approach_2"
            factor = ratio if ratio > 1 else 1.0 / ratio
            print(f"  speed: {faster} ~{factor:.2f}x faster overall")


if __name__ == "__main__":
    main()
