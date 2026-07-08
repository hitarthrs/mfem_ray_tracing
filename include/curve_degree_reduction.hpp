#ifndef CURVE_DEGREE_REDUCTION_HPP
#define CURVE_DEGREE_REDUCTION_HPP

#include "curve_reduction_types.hpp"

namespace mfem_raytracing
{

SingleStepReductionResult PeakErrorSingleStep(
    const CurveData &initial_curve,
    double max_error,
    const PeakErrorSingleStepOptions &options = {});

} // namespace mfem_raytracing

#endif
