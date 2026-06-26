"""Multi-step B-spline curve degree reduction (Algorithm 1)."""

from .degree_reduce_multiple_steps import (
    MultipleStepReductionFailure,
    MultipleStepReductionResult,
    ReducedCurveLeaf,
    degree_reduce_multiple_steps,
)

__all__ = [
    "MultipleStepReductionFailure",
    "MultipleStepReductionResult",
    "ReducedCurveLeaf",
    "degree_reduce_multiple_steps",
]
