#ifndef MFEM_RAYTRACING_PIPELINE_REDUCE_HPP
#define MFEM_RAYTRACING_PIPELINE_REDUCE_HPP

// Public Stage 1: one NURBS/B-spline patch → watertight bilinear leaves.
// Always compete + coalesce with hard seams ON. Soft coalesce is not exposed.

#include "mfem_raytracing/reduction/bilinear_leaf_extraction.hpp"
#include "mfem_raytracing/reduction/hard_seam_bilinearization.hpp"
#include "mfem_raytracing/reduction/surface_reduction_types.hpp"

#include <vector>

namespace mfem_raytracing
{

struct PatchLeafReductionOptions
{
    /// 0 uses the conforming reducer's hardware-concurrency default.
    unsigned threads = 0;
    /// Error accounting / Eq. 5.30 / refinement guards.
    PeakErrorSurfaceSingleStepOptions single_step;
};

/// Reduce one tensor-product patch to true 2×2 bilinear leaves under `max_error`.
/// Hard seams (original unique knots) are always retained through coalesce.
BilinearLeafCollection ReducePatchToBilinearLeaves(
    const SurfaceData &surface, double max_error,
    const PatchLeafReductionOptions &options = {});

/// Catalog helper: map ReducePatchToBilinearLeaves over selected patch ids.
/// Equivalent to the historical BilinearizePatchesWithHardSeams entry point.
HardSeamBilinearizationResult ReducePatchesToBilinearLeaves(
    const SurfacePatchCatalog &catalog, const std::vector<int> &patch_ids,
    const HardSeamBilinearizationOptions &options);

} // namespace mfem_raytracing

#endif
