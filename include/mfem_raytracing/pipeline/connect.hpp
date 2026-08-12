#ifndef MFEM_RAYTRACING_PIPELINE_CONNECT_HPP
#define MFEM_RAYTRACING_PIPELINE_CONNECT_HPP

// Public Stage 2: connect per-patch bilinear leaves across a catalog.

#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "mfem_raytracing/reduction/hard_seam_bilinearization.hpp"
#include "mfem_raytracing/tspline/tspline_leaf_assembly.hpp"
#include "mfem_raytracing/tspline/tspline_shell_composer.hpp"

#include <string>
#include <utility>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

/// Catalog-wide seam topology + merged geometry, before T-mesh materialization.
struct SeamAssembly
{
    std::string surface_name;
    ShellBuildOptions options;
    std::vector<BilinearLeafAssembly> assemblies;
    std::vector<PatchInterface> interfaces;
    std::vector<std::pair<PatchInterface, std::pair<BoundaryBand, BoundaryBand>>> bands;
    SeamOwnershipPlan ownership;
    /// One entry per interface: merge filled; local strip mesh filled by Build.
    std::vector<ShellStripResult> strips;
    std::vector<SourceLeafRef> expected_sources;
};

/// Discover interfaces, select seam bands, average-merge seams, plan ownership.
SeamAssembly ConnectPatchLeaves(const LeafPatchScene &input,
                                const SurfacePatchCatalog &catalog,
                                const ShellBuildOptions &options = {});

} // namespace tspline
} // namespace mfem_raytracing

#endif
