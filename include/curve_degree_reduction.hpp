#ifndef CURVE_DEGREE_REDUCTION_HPP
#define CURVE_DEGREE_REDUCTION_HPP

// Public API: adaptive single-step curve degree reduction (approach 1).

#include "curve_reduction_types.hpp"

namespace mfem_raytracing
{

/** Reduce p -> p-1 with peak-error driven knot splitting; may return multiple segments. */
SingleStepReductionResult PeakErrorSingleStep(
    const CurveData &initial_curve,
    double max_error,
    const PeakErrorSingleStepOptions &options = {});

/**
 * Optimized variant of PeakErrorSingleStep().
 *
 * Uses the same reduction logic and returns the same segment ordering, but
 * avoids redundant accepted-piece reductions and can evaluate left/right
 * recursion branches concurrently.
 */
SingleStepReductionResult PeakErrorSingleStepOptimized(
    const CurveData &initial_curve,
    double max_error,
    const PeakErrorSingleStepOptions &options = {});

} // namespace mfem_raytracing

#endif
