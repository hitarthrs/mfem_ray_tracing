#ifndef MFEM_RAYTRACING_TSPLINE_SHELL_COMPOSER_HPP
#define MFEM_RAYTRACING_TSPLINE_SHELL_COMPOSER_HPP

#include "mfem_raytracing/tspline/tspline_degree_one_bake.hpp"
#include "mfem_raytracing/tspline/tspline_corner_collar.hpp"
#include "mfem_raytracing/tspline/tspline_error_accounting.hpp"
#include "mfem_raytracing/tspline/tspline_patch_interfaces.hpp"
#include "mfem_raytracing/tspline/tspline_shell_watertightness.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

enum class CornerOwnershipPolicy
{
    /// Partition and refine the source boundary rows exactly.  Each output
    /// cell has exactly one source owner; corner cells satisfy all incident
    /// seam constraints together.  This is the production default.
    ExactBoundaryCornerCollar,
    /// Build full local strips for diagnostics, but never certify an ambiguous
    /// source leaf for RT. This is the safe production default.
    RejectAmbiguous,
    /// Retain Python-compatible overlapping strips for regression comparison.
    /// The watertightness gate still reports the overlap and blocks RT.
    CompatibilityOverlap,
};

struct SeamClaim
{
    SourceLeafRef source;
    std::string interface_id;
    StripRegion region = StripRegion::A;
};

struct CornerConflict
{
    SourceLeafRef source;
    std::vector<SeamClaim> claims;
};

struct SeamOwnershipPlan
{
    std::size_t raw_claim_count = 0;
    std::size_t unique_claimed_source_leaf_count = 0;
    std::map<SourceLeafRef, std::vector<SeamClaim>> claims_by_source;
    std::vector<CornerConflict> conflicts;
};

struct ShellStripResult
{
    PatchInterface interface;
    std::string interface_id;
    AverageMergedSeam merge;
    LocalDegreeOneStrip strip;
    BakedDegreeOneStrip baked;
    BakeVerification verification;
};

struct BakedShellLeaf
{
    LeafPatch leaf;
    SourceLeafRef source;
    std::string owner_id;
    std::size_t strip_face_index = 0;
    std::array<double, 4> local_subrect = {};
    LeafErrorBreakdown error;
};

struct ShellBuildOptions
{
    std::size_t seam_band_rows = 1;
    CornerOwnershipPolicy corner_policy = CornerOwnershipPolicy::ExactBoundaryCornerCollar;
    InterfaceDiscoveryOptions interface_discovery;
    AverageSeamMergeOptions average_merge;
    LocalStripBuildOptions strip_build;
    DegreeOneBakeOptions bake;
    ErrorValidationOptions error_validation;
    WatertightnessOptions watertightness;
};

struct BakedTsplineShell
{
    std::string surface_name;
    std::vector<BakedShellLeaf> leaves;
    std::vector<ShellStripResult> strips;
    SeamOwnershipPlan ownership;
    ErrorValidationReport errors;
    WatertightnessReport watertightness;
    CornerOwnershipPolicy corner_policy = CornerOwnershipPolicy::ExactBoundaryCornerCollar;

    bool ReadyForRayTracing() const;
    std::vector<LeafPatch> RuntimeLeaves() const;
};

std::string MakeInterfaceId(const PatchInterface &interface);
SeamOwnershipPlan MakeSeamOwnershipPlan(
    const std::vector<std::pair<PatchInterface, std::pair<BoundaryBand, BoundaryBand>>> &bands);

/// Assemble interiors plus all local baked seam strips.
/// Prefer the staged public API: ConnectPatchLeaves → BuildMultiPatchTMesh →
/// BakeForRayTracing. This entry point remains as a thin Connect→Build→Bake
/// wrapper for existing callers.
BakedTsplineShell ComposeBakedTsplineShell(const LeafPatchScene &input,
                                           const SurfacePatchCatalog &catalog,
                                           const ShellBuildOptions &options = {});
void RequireShellReadyForRayTracing(const BakedTsplineShell &shell);

} // namespace tspline
} // namespace mfem_raytracing

#endif
