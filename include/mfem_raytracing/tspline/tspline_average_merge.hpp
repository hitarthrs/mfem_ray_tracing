#ifndef MFEM_RAYTRACING_TSPLINE_AVERAGE_MERGE_HPP
#define MFEM_RAYTRACING_TSPLINE_AVERAGE_MERGE_HPP

#include "mfem_raytracing/tspline/tspline_bilinear_ops.hpp"
#include "mfem_raytracing/tspline/tspline_leaf_assembly.hpp"

#include <string>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

/// A copied leaf retains its stable source-assembly index.  The copied control
/// net may be modified by the seam merge while its provenance stays intact.
struct SeamRegionLeaf
{
    std::size_t source_leaf_index = 0;
    LeafPatch leaf;
};

struct SeamRegion
{
    int patch_id = -1;
    std::string role;
    BoundarySide side = BoundarySide::UMin;
    ParameterRange u_range;
    ParameterRange v_range;
    std::vector<SeamRegionLeaf> leaves;
};

struct AverageSeamMergeOptions
{
    double parameter_tolerance = 1e-9;
    double geometry_tolerance = 1e-6;
};

/// Result of the symmetric Section-6 homogeneous average merge.  Neither side
/// is a master: both outer seam rows are reseated to `merged_controls`.
struct AverageMergedSeam
{
    SeamRegion region_a;
    SeamRegion region_b;
    BoundaryChain chain_a;
    BoundaryChain chain_b;
    bool b_reversed = false;
    /// Common normalized seam parameters in ascending orientation of A.
    std::vector<double> union_parameters;
    std::vector<HomogeneousPoint> merged_controls;
    double max_pre_merge_disagreement = 0.0;
    double max_displacement_a = 0.0;
    double max_displacement_b = 0.0;
};

SeamRegion MakeSeamRegion(const BilinearLeafAssembly &assembly,
                          const BoundaryBand &band);

AverageMergedSeam AverageMergeSeam(const BilinearLeafAssembly &assembly_a,
                                   const BoundaryBand &band_a,
                                   const BilinearLeafAssembly &assembly_b,
                                   const BoundaryBand &band_b,
                                   const AverageSeamMergeOptions &options = {});

} // namespace tspline
} // namespace mfem_raytracing

#endif
