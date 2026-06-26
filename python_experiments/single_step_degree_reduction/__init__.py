"""Single-step B-spline curve degree reduction strategies (Algorithm 2)."""

from .protocol import (
    Curve,
    DegreeReductionSingleStep,
    ReducedCurveSegment,
    SingleStepReductionResult,
    degree_reduction_single_step,
    single_step_result_from_lists,
    validate_single_step_result,
)

__all__ = [
    "Curve",
    "DegreeReductionSingleStep",
    "ReducedCurveSegment",
    "SingleStepReductionResult",
    "degree_reduction_single_step",
    "single_step_result_from_lists",
    "validate_single_step_result",
]
