#ifndef MFEM_RAYTRACING_TSPLINE_CORNER_COLLAR_HPP
#define MFEM_RAYTRACING_TSPLINE_CORNER_COLLAR_HPP

#include "mfem_raytracing/tspline/tspline_average_merge.hpp"
#include "mfem_raytracing/tspline/tspline_shell_watertightness.hpp"

#include <string>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

/// One side of an already averaged interface.  The collar builder consumes
/// these constraints together: that is what lets a single patch-corner cell
/// satisfy both incident seams without being emitted twice.
struct BoundaryMergeConstraint
{
    std::string interface_id;
    int patch_id = -1;
    BoundarySide side = BoundarySide::UMin;
    BoundaryChain chain;
    bool reversed = false;
    std::vector<double> union_parameters;
    std::vector<HomogeneousPoint> merged_controls;
    double seam_displacement = 0.0;
};

struct CornerCollarLeaf
{
    LeafPatch leaf;
    SourceLeafRef source;
    std::vector<std::string> interface_ids;
    double seam_displacement = 0.0;
};

struct CornerCollarOptions
{
    double parameter_tolerance = 1e-9;
    double geometry_tolerance = 1e-6;
};

/// Exact, non-overlapping replacement for full pairwise seam strips.
///
/// Every source leaf is emitted once (possibly split by exact degree-one knot
/// insertion).  A side collar follows one merged seam; a corner collar is one
/// refined source cell that follows both incident merged seams.  Endpoint
/// assignments are connected through interface parameter ownership, then
/// averaged homogeneously once, so all interfaces meeting at a corner use the
/// same projective vertex.
std::vector<CornerCollarLeaf> BuildExactBoundaryCornerCollar(
    const std::vector<BilinearLeafAssembly> &assemblies,
    const std::vector<BoundaryMergeConstraint> &constraints,
    const CornerCollarOptions &options = {});

} // namespace tspline
} // namespace mfem_raytracing

#endif
