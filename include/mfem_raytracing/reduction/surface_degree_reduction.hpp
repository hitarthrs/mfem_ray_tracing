#ifndef SURFACE_DEGREE_REDUCTION_HPP
#define SURFACE_DEGREE_REDUCTION_HPP

#include "mfem_raytracing/reduction/surface_reduction_types.hpp"

namespace mfem_raytracing
{

SingleStepSurfaceReductionResult PeakErrorSurfaceSingleStep(
    const SurfaceData &initial_surface,
    double max_error,
    const PeakErrorSurfaceSingleStepOptions &options = {});

/**
 * Optimized variant of PeakErrorSurfaceSingleStep().
 *
 * Preserves segment ordering and geometry, but parallelizes independent curve
 * reductions across u/v passes and can evaluate recursive split branches
 * concurrently.
 */
SingleStepSurfaceReductionResult PeakErrorSurfaceSingleStepOptimized(
    const SurfaceData &initial_surface,
    double max_error,
    const PeakErrorSurfaceSingleStepOptions &options = {});

} // namespace mfem_raytracing

#endif
