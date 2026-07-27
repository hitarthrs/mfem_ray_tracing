#ifndef MFEM_RAYTRACING_HARD_SEAM_BILINEARIZATION_HPP
#define MFEM_RAYTRACING_HARD_SEAM_BILINEARIZATION_HPP

// Reusable full-patch hard-seam bilinearization.  This is the C++ counterpart
// to the Python pipe demo, but accepts any catalog of tensor-product NURBS
// patches: original unique knots remain cell boundaries, each patch is reduced
// independently in world coordinates, and no inter-patch sewing is performed.

#include "bilinear_leaf_extraction.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace mfem_raytracing
{

struct SurfacePatchDescriptor
{
    int id = -1;
    std::string name;
    std::string role;
    int quarter = -1;
    int volume_patch = -1;
    int attribute = -1;
    SurfaceData surface;
};

struct SurfacePatchCatalog
{
    std::string mesh;
    std::string description;
    std::vector<SurfacePatchDescriptor> patches;
};

/// Load the patch-catalog JSON schema used by pipe_nurbs_border_patches.json.
SurfacePatchCatalog LoadSurfacePatchCatalogJson(const std::string &json_path);

/// Locate a catalog patch by its stable id, or throw std::out_of_range.
const SurfacePatchDescriptor &FindSurfacePatch(const SurfacePatchCatalog &catalog, int id);

struct HardSeamBilinearizationOptions
{
    /// Global error limit per source patch. Reductions use the library defaults:
    /// approach_4 weighted-late budgets, sum error accounting, and Eq. 5.30.
    double max_error = 0.0;
    /// 0 uses the conforming reducer's hardware-concurrency default.
    unsigned threads = 0;
};

struct HardSeamPatchBilinearization
{
    SurfacePatchDescriptor source;
    int n_steps = 0;
    std::size_t n_competing = 0;
    std::vector<double> hard_knots_u;
    std::vector<double> hard_knots_v;
    std::vector<BilinearLeafPatch> leaves;
};

struct HardSeamBilinearizationResult
{
    std::string mesh;
    double max_error = 0.0;
    std::vector<HardSeamPatchBilinearization> patches;

    std::size_t LeafCount() const;
    AxisAlignedBounds SceneBBox() const;
};

/// Reduce selected source patches to independent, conforming bilinear leaves.
/// Input knots are hard seams; the operation never modifies source positions
/// through cage centering and never performs inter-patch sewing.
HardSeamBilinearizationResult BilinearizePatchesWithHardSeams(
    const SurfacePatchCatalog &catalog, const std::vector<int> &patch_ids,
    const HardSeamBilinearizationOptions &options);

/// Write the self-describing hard-seam leaf JSON consumed by LoadLeafPatchScene
/// and the Embree utilities. Leaves carry their source patch id and role.
void WriteHardSeamBilinearizationJson(std::ostream &os,
                                      const HardSeamBilinearizationResult &result,
                                      const std::string &surface_name);

} // namespace mfem_raytracing

#endif
