#ifndef SURFACE_DEGREE_REDUCTION_HPP
#define SURFACE_DEGREE_REDUCTION_HPP

#include "surface_reduction_types.hpp"

namespace mfem_raytracing
{

SingleStepSurfaceReductionResult PeakErrorSurfaceSingleStep(
    const SurfaceData &initial_surface,
    double max_error,
    const PeakErrorSurfaceSingleStepOptions &options = {});

} // namespace mfem_raytracing

#endif
