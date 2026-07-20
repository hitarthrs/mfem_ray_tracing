#ifndef BILINEAR_LEAF_EXTRACTION_HPP
#define BILINEAR_LEAF_EXTRACTION_HPP

// Bilinear leaf patches + axis-aligned bounding boxes for BVH / ray tracing.
//
// For degree-(1, 1) tensor-product patches with non-negative weights, the AABB
// of the four corner control points bounds the surface exactly on [0, 1]^2.
// Port of python_experiments/multiple_step_degree_reduction_surfaces/leaf_bvh.py
// plus the JSON schema of demo_leaf_bboxes.py.

#include "embree/bilinear_patch_geometry.hpp"
#include "surface_multistep_reduction.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace mfem_raytracing
{

constexpr double kDefaultWeightTolerance = 1e-12;

/// One bilinear leaf ready for BVH insertion and ray-patch intersection.
/// `bbox` is exact when degrees are (1, 1) and all weights are non-negative.
struct BilinearLeafPatch
{
    int index = -1;
    SurfaceData surface;
    std::pair<double, double> u_domain_local = {0.0, 1.0};
    std::pair<double, double> v_domain_local = {0.0, 1.0};
    std::pair<double, double> u_domain_global = {0.0, 1.0};
    std::pair<double, double> v_domain_global = {0.0, 1.0};
    double total_error = 0.0;
    AxisAlignedBounds bbox;
    bool weights_were_clamped = false;
};

/// All bilinear leaves from one multi-step reduction run.
struct BilinearLeafCollection
{
    std::vector<BilinearLeafPatch> leaves;
    double max_error = 0.0;
    int n_steps = 0;
    std::string backend_name = "approach_4";

    /// Union of all leaf boxes; only meaningful when !leaves.empty().
    AxisAlignedBounds SceneBBox() const;
};

/// Backend name ("approach_1" | "approach_3" | "approach_4" | "approach_5")
/// to budget policy. Throws std::invalid_argument for unknown names.
ErrorBudgetPolicy BudgetPolicyFromBackendName(const std::string &backend_name);

bool WeightsAreNonNegative(const std::vector<std::vector<double>> &weights,
                           double tol = kDefaultWeightTolerance);

/// Clamp weights below -tol to `floor`. Returns true when clamping occurred.
bool ClampWeightsNonNegative(std::vector<std::vector<double>> &weights,
                             double floor = 0.0,
                             double tol = kDefaultWeightTolerance);

/// Exact AABB for a bilinear patch from its 2x2 Cartesian control net.
/// Throws std::invalid_argument unless degrees are (1, 1) with a 2x2 net.
AxisAlignedBounds BilinearControlNetAABB(const SurfaceData &surface);

/// Build one leaf patch from a reduction leaf: sanitize weights (clamp
/// negatives when `enforce_nonneg_weights`, else throw) and compute the AABB.
BilinearLeafPatch ExtractBilinearLeafPatch(const ReducedSurfaceLeaf &leaf,
                                           int index,
                                           bool enforce_nonneg_weights = true,
                                           double weight_tol = kDefaultWeightTolerance);

struct BilinearLeafReductionOptions
{
    /// Multi-step budget policy by backend name (default approach_4 = weighted-late).
    std::string backend_name = "approach_4";
    PeakErrorSurfaceSingleStepOptions single_step;
    bool enforce_nonneg_weights = true;
    /// Watertight tensor-grid driver (DegreeReduceMultipleStepsConforming)
    /// instead of the legacy non-conforming per-branch splitting.
    bool conforming = false;
    /// Greedy full-line coalescing after the conforming grid build.
    bool coalesce = true;
};

/// Full pipeline: multi-step reduction to degrees (1, 1) + leaf extraction.
BilinearLeafCollection ReduceSurfaceToBilinearLeaves(
    const SurfaceData &initial_surface,
    int n_steps,
    double max_error,
    const BilinearLeafReductionOptions &options = {});

/// Write the collection in the demo_leaf_bboxes.py JSON schema, readable by
/// LoadLeafPatchScene() and the bilinear_ray_tracer.html viewer.
void WriteLeafBBoxJson(std::ostream &os,
                       const BilinearLeafCollection &collection,
                       const std::string &surface_name);

} // namespace mfem_raytracing

#endif
