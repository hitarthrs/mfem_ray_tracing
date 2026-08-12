#ifndef MFEM_RAYTRACING_TSPLINE_LEAF_ASSEMBLY_HPP
#define MFEM_RAYTRACING_TSPLINE_LEAF_ASSEMBLY_HPP

#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "mfem_raytracing/tspline/tspline_seam_types.hpp"

#include <string>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

struct ParameterRange
{
    double lo = 0.0;
    double hi = 0.0;
};

/// Bilinear leaves belonging to one original source patch.  The leaves remain
/// in source JSON order, while every query uses cached parameter bounds.
struct BilinearLeafAssembly
{
    int patch_id = -1;
    std::string role;
    std::vector<LeafPatch> leaves;
    ParameterRange u_range;
    ParameterRange v_range;
};

/// A requested number of rows next to one patch boundary.  It deliberately
/// preserves leaf indices into the parent assembly for future ownership and
/// provenance accounting.
struct BoundaryBand
{
    int patch_id = -1;
    BoundarySide side = BoundarySide::UMin;
    std::vector<std::size_t> leaf_indices;
    ParameterRange normal_range;
};

struct BoundaryChainSegment
{
    std::size_t leaf_index = 0;
    ParameterRange tangent_range;
    std::array<HomogeneousPoint, 2> endpoints;
};

/// The ordered, C0 boundary polyline of a selected band.  This is the direct
/// input to a later average/C0 merge, without rediscovering leaves or corners.
struct BoundaryChain
{
    int patch_id = -1;
    BoundarySide side = BoundarySide::UMin;
    double boundary_value = 0.0;
    ParameterRange tangent_range;
    std::vector<BoundaryChainSegment> segments;
};

struct BoundaryChainOptions
{
    double parameter_tolerance = 1e-9;
    double geometry_tolerance = 1e-6;
};

std::vector<BilinearLeafAssembly> BuildBilinearLeafAssemblies(
    const LeafPatchScene &scene);
const BilinearLeafAssembly &FindBilinearLeafAssembly(
    const std::vector<BilinearLeafAssembly> &assemblies, int patch_id);

BoundaryBand SelectBoundaryBand(const BilinearLeafAssembly &assembly,
                                BoundarySide side, std::size_t rows = 1,
                                double parameter_tolerance = 1e-9);
BoundaryChain BuildBoundaryChain(const BilinearLeafAssembly &assembly,
                                 const BoundaryBand &band,
                                 const BoundaryChainOptions &options = {});

} // namespace tspline
} // namespace mfem_raytracing

#endif
