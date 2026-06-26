"""
Multi-step B-spline curve degree reduction (Algorithm 1).

Breadth-first queue over curve segments. Each step delegates to a
:class:`~single_step_degree_reduction.protocol.DegreeReductionSingleStep`
backend (e.g. dummy split-at-u + A5.11).
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

_PYTHON_EXPERIMENTS = Path(__file__).resolve().parent.parent
if str(_PYTHON_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_PYTHON_EXPERIMENTS))

from single_step_degree_reduction.protocol import Curve, DegreeReductionSingleStep


@dataclass(frozen=True)
class ReducedCurveLeaf:
    """One output curve at the target degree with cumulative error along its path."""

    curve: Curve
    total_error: float
    steps_taken: int


@dataclass(frozen=True)
class MultipleStepReductionResult:
    """All accepted leaves from a multi-step reduction run."""

    segments: tuple[ReducedCurveLeaf, ...]

    @property
    def curves(self) -> tuple[Curve, ...]:
        return tuple(leaf.curve for leaf in self.segments)


@dataclass(frozen=True)
class _QueueItem:
    curve: Curve
    current_degree: int
    remaining_budget: float
    consumed_error: float


class MultipleStepReductionFailure(Exception):
    """Raised when ``n_steps`` is invalid for the input degree."""


def degree_reduce_multiple_steps(
    initial_curve: Curve,
    n_steps: int,
    max_error: float,
    *,
    single_step_backend: DegreeReductionSingleStep,
) -> MultipleStepReductionResult:
    """
    Reduce ``initial_curve`` by ``n_steps`` degree levels using ``single_step_backend``.

    Parameters
    ----------
    initial_curve :
        Input B-spline / NURBS curve of degree ``s``.
    n_steps :
        Number of degree-reduction steps (``s → s - n_steps``).
    max_error :
        Global error budget subtracted along each branch.
    single_step_backend :
        Algorithm 2 implementation (one ``p → p-1`` step, possibly splitting).

    Returns
    -------
    MultipleStepReductionResult
        One leaf per queue branch that reaches ``s - n_steps``.

    Raises
    ------
    ValueError
        If ``max_error`` or ``n_steps`` is negative.
    MultipleStepReductionFailure
        If ``n_steps > s - 1``.
    """
    if max_error < 0:
        raise ValueError(f"max_error must be non-negative, got {max_error}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")

    input_degree = int(initial_curve.degree)
    if n_steps == 0:
        return MultipleStepReductionResult(
            segments=(ReducedCurveLeaf(initial_curve, 0.0, 0),)
        )
    if n_steps > input_degree - 1:
        raise MultipleStepReductionFailure(
            f"cannot reduce {n_steps} steps from degree {input_degree} "
            f"(maximum is {input_degree - 1})"
        )

    target_degree = input_degree - n_steps
    queue: deque[_QueueItem] = deque(
        [_QueueItem(initial_curve, input_degree, float(max_error), 0.0)]
    )
    leaves: list[ReducedCurveLeaf] = []

    while queue:
        item = queue.popleft()

        if item.current_degree == target_degree:
            leaves.append(
                ReducedCurveLeaf(
                    curve=item.curve,
                    total_error=float(item.consumed_error),
                    steps_taken=int(n_steps),
                )
            )
            continue

        step_result = single_step_backend(item.curve, item.remaining_budget)
        for segment in step_result.segments:
            next_budget = item.remaining_budget - segment.segment_error
            if next_budget < 0:
                continue
            queue.append(
                _QueueItem(
                    curve=segment.curve,
                    current_degree=item.current_degree - 1,
                    remaining_budget=next_budget,
                    consumed_error=item.consumed_error + segment.segment_error,
                )
            )

    return MultipleStepReductionResult(segments=tuple(leaves))


def _main() -> None:
    from multiple_step_degree_reduction.demo_degree_4_to_2 import main as demo_main

    demo_main()


if __name__ == "__main__":
    _main()
