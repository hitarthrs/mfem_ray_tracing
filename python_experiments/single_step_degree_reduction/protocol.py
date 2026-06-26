"""
Protocol for single-step B-spline curve degree reduction (Algorithm 2).

One call reduces degree by one (``p → p-1``) and may return one or more curve
segments. Each segment carries a non-negative *segment error* — the error cost
attributed to that piece for this step. ``DegreeReduceMultipleSteps`` (Algorithm 1)
subtracts these costs from a running error budget as it walks the processing queue.

Implementations live in this package (e.g. a dummy splitter for orchestration
tests, A5.11 pass-through, recursive span split, knot-refine-then-reduce).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from geomdl import BSpline, NURBS

Curve = BSpline.Curve | NURBS.Curve


@dataclass(frozen=True)
class ReducedCurveSegment:
    """One curve piece produced by a single degree-reduction step."""

    curve: Curve
    segment_error: float


@dataclass(frozen=True)
class SingleStepReductionResult:
    """
    Output of :func:`degree_reduction_single_step` / :class:`DegreeReductionSingleStep`.

    ``segments`` and ``segment_errors`` are parallel: index ``i`` of one matches
    index ``i`` of the other.
    """

    segments: tuple[ReducedCurveSegment, ...]

    @property
    def curves(self) -> tuple[Curve, ...]:
        return tuple(segment.curve for segment in self.segments)

    @property
    def segment_errors(self) -> tuple[float, ...]:
        return tuple(segment.segment_error for segment in self.segments)


@runtime_checkable
class DegreeReductionSingleStep(Protocol):
    """
    Callable strategy for Algorithm 2.

    Parameters
    ----------
    initial_curve :
        Input B-spline or NURBS curve of degree ``p >= 1``.
    max_error :
        Upper bound on acceptable *segment error* for pieces returned from this
        step. Implementations must not emit segments with ``segment_error > max_error``.

    Returns
    -------
    SingleStepReductionResult
        One or more segments, each at degree ``p - 1``.
    """

    def __call__(
        self,
        initial_curve: Curve,
        max_error: float,
    ) -> SingleStepReductionResult: ...


def single_step_result_from_lists(
    curves: list[Curve] | tuple[Curve, ...],
    segment_errors: list[float] | tuple[float, ...],
) -> SingleStepReductionResult:
    """Build a :class:`SingleStepReductionResult` from parallel curve / error lists."""
    curve_list = tuple(curves)
    error_list = tuple(float(e) for e in segment_errors)
    if len(curve_list) != len(error_list):
        raise ValueError(
            f"curves length {len(curve_list)} != segment_errors length {len(error_list)}"
        )
    return SingleStepReductionResult(
        segments=tuple(
            ReducedCurveSegment(curve=c, segment_error=e)
            for c, e in zip(curve_list, error_list, strict=True)
        )
    )


def validate_single_step_result(
    result: SingleStepReductionResult,
    *,
    input_curve: Curve,
    max_error: float,
) -> None:
    """
    Check protocol invariants. Raises :class:`ValueError` on violation.

    Call from tests and orchestration code when wiring a new backend.
    """
    if max_error < 0:
        raise ValueError(f"max_error must be non-negative, got {max_error}")

    if not result.segments:
        raise ValueError("single-step result must contain at least one segment")

    input_degree = int(input_curve.degree)
    if input_degree < 1:
        raise ValueError(f"input curve degree must be >= 1, got {input_degree}")
    expected_degree = input_degree - 1

    for index, segment in enumerate(result.segments):
        if segment.segment_error < 0:
            raise ValueError(
                f"segment {index}: segment_error must be non-negative, "
                f"got {segment.segment_error}"
            )
        if segment.segment_error > max_error:
            raise ValueError(
                f"segment {index}: segment_error {segment.segment_error} "
                f"exceeds max_error {max_error}"
            )
        out_degree = int(segment.curve.degree)
        if out_degree != expected_degree:
            raise ValueError(
                f"segment {index}: expected degree {expected_degree}, got {out_degree}"
            )


def degree_reduction_single_step(
    initial_curve: Curve,
    max_error: float,
    *,
    backend: DegreeReductionSingleStep,
) -> SingleStepReductionResult:
    """
    Algorithm 2 entry point.

    Dispatches to ``backend`` and validates the returned result against
    ``initial_curve`` and ``max_error``.
    """
    result = backend(initial_curve, max_error)
    validate_single_step_result(
        result,
        input_curve=initial_curve,
        max_error=max_error,
    )
    return result
