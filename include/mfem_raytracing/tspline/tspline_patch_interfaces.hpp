#ifndef MFEM_RAYTRACING_TSPLINE_PATCH_INTERFACES_HPP
#define MFEM_RAYTRACING_TSPLINE_PATCH_INTERFACES_HPP

#include "mfem_raytracing/reduction/hard_seam_bilinearization.hpp"
#include "mfem_raytracing/tspline/tspline_seam_types.hpp"

#include <array>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

/// One source NURBS boundary, ordered in its native tangent parameter.
struct NurbsBoundaryCurve
{
    int degree = 0;
    std::vector<double> knots;
    std::vector<HomogeneousPoint> controls;
};

struct BoundaryCurveMatch
{
    bool matches = false;
    /// True when B must be traversed in reverse to align it with A.
    bool b_reversed = false;
    double max_point_disagreement = 0.0;
    double max_weight_disagreement = 0.0;
};

struct PatchInterface
{
    int patch_a = -1;
    BoundarySide side_a = BoundarySide::UMin;
    int patch_b = -1;
    BoundarySide side_b = BoundarySide::UMin;
    bool b_reversed = false;
    double max_point_disagreement = 0.0;
    double max_weight_disagreement = 0.0;
};

struct InterfaceDiscoveryOptions
{
    /// Absolute control-point/weight matching tolerance, matching the Python case study.
    double tolerance = 1e-5;
};

/// Cached source edges for a catalog.  Build once, then reuse for interface
/// discovery, later seam validation, or a master-curve merge without repeatedly
/// slicing the surface control nets.
struct PatchBoundaryIndexEntry
{
    int patch_id = -1;
    std::array<NurbsBoundaryCurve, 4> boundaries;
};

struct PatchBoundaryIndex
{
    std::vector<PatchBoundaryIndexEntry> patches;
};

NurbsBoundaryCurve ExtractNurbsBoundaryCurve(const SurfaceData &surface,
                                              BoundarySide side);
BoundaryCurveMatch MatchNurbsBoundaryCurves(const NurbsBoundaryCurve &a,
                                            const NurbsBoundaryCurve &b,
                                            double tolerance = 1e-5);
PatchBoundaryIndex BuildPatchBoundaryIndex(const SurfacePatchCatalog &catalog);

/// Discover all pairwise matching NURBS edges in a patch catalog.
std::vector<PatchInterface> DiscoverPatchInterfaces(
    const PatchBoundaryIndex &index,
    const InterfaceDiscoveryOptions &options = {});
std::vector<PatchInterface> DiscoverPatchInterfaces(
    const SurfacePatchCatalog &catalog,
    const InterfaceDiscoveryOptions &options = {});

} // namespace tspline
} // namespace mfem_raytracing

#endif
