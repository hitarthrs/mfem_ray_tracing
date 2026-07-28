#ifndef MFEM_RAYTRACING_TSPLINE_SHELL_WATERTIGHTNESS_HPP
#define MFEM_RAYTRACING_TSPLINE_SHELL_WATERTIGHTNESS_HPP

#include "embree/leaf_patch_loader.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

struct SourceLeafRef
{
    int patch_id = -1;
    std::size_t leaf_index = 0;

    bool operator<(const SourceLeafRef &other) const
    {
        return patch_id != other.patch_id ? patch_id < other.patch_id : leaf_index < other.leaf_index;
    }
    bool operator==(const SourceLeafRef &other) const
    {
        return patch_id == other.patch_id && leaf_index == other.leaf_index;
    }
};

/// A final RT leaf plus the source face and construction owner that produced it.
/// Several baked subcells may share one owner/source; multiple distinct owners
/// for a source leaf are an ambiguous seam-corner conflict.
struct WatertightLeaf
{
    std::size_t final_leaf_index = 0;
    LeafPatch leaf;
    SourceLeafRef source;
    std::string owner_id;
    std::size_t strip_face_index = 0;
};

struct WatertightnessOptions
{
    double relative_tolerance = 1e-11;
    double absolute_tolerance = 0.0;
    bool require_closed = true;
    bool require_single_source_owner = true;
    std::size_t max_reported_issues = 64;
};

enum class WatertightnessIssueKind
{
    MissingSourceOwner,
    MultipleSourceOwners,
    DegenerateEdge,
    OpenEdgeSpan,
    NonManifoldEdgeSpan,
    GeometricMismatch,
};

struct WatertightnessIssue
{
    WatertightnessIssueKind kind = WatertightnessIssueKind::OpenEdgeSpan;
    SourceLeafRef source;
    std::string owner_id;
    std::size_t final_leaf_index = 0;
    double gap = 0.0;
};

struct WatertightnessReport
{
    bool watertight = false;
    double geometry_tolerance = 0.0;
    double max_edge_gap = 0.0;
    std::size_t expected_source_leaf_count = 0;
    std::size_t unowned_source_leaf_count = 0;
    std::size_t multiply_owned_source_leaf_count = 0;
    std::size_t atomic_edge_span_count = 0;
    std::size_t open_edge_span_count = 0;
    std::size_t nonmanifold_edge_span_count = 0;
    std::size_t geometric_mismatch_count = 0;
    std::size_t degenerate_edge_count = 0;
    std::vector<WatertightnessIssue> issues;
};

WatertightnessReport CheckShellWatertightness(
    const std::vector<WatertightLeaf> &leaves,
    const std::vector<SourceLeafRef> &expected_sources,
    const WatertightnessOptions &options = {});

void RequireWatertightForRayTracing(const WatertightnessReport &report);
const char *WatertightnessIssueKindName(WatertightnessIssueKind kind);

} // namespace tspline
} // namespace mfem_raytracing

#endif
