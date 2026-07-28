#ifndef MFEM_RAYTRACING_EMBREE_LEAF_PATCH_LOADER_HPP
#define MFEM_RAYTRACING_EMBREE_LEAF_PATCH_LOADER_HPP

#include "embree/bilinear_patch_geometry.hpp"
#include "surface_reduction_types.hpp"

#include <string>
#include <vector>

namespace mfem_raytracing
{

/// One bilinear leaf from a multi-step surface degree reduction, as exported by
/// python_experiments/multiple_step_degree_reduction_surfaces (d4_leaf_bboxes.json).
struct LeafPatch
{
    BilinearPatchPrimitive patch;
    int index = -1;
    /// Source catalog patch, when provided by a hard-seam export.
    int patch_id = -1;
    /// Exporter role, when provided (e.g. interior, seam-exact, seam-phantom).
    std::string role = "unknown";
    /// Primitive construction kind, when supplied by a baked T-spline exporter.
    std::string kind = "bilinear";
    /// Stable seam/interface label, if this leaf was produced by a seam bake.
    std::string interface_id;
    /// Where this leaf's [0, 1]^2 parameters live on the original surface.
    double u_domain_global[2] = {0.0, 1.0};
    double v_domain_global[2] = {0.0, 1.0};
    double total_error = 0.0;
    AxisAlignedBounds bbox;
};

/// A full leaf-patch scene loaded from JSON.
struct LeafPatchScene
{
    std::string surface_name;
    double max_error = 0.0;
    AxisAlignedBounds scene_bbox;
    std::vector<LeafPatch> leaves;

    /// Baked T-spline shell JSON may carry an explicit pre-RT certificate.
    /// Legacy leaf JSON has no declaration and remains permitted by default.
    bool declares_rt_certification = false;
    bool rt_certified = true;

    /// Just the patches, in leaf order, ready for EmbreeRayTracer::RegisterPatches.
    std::vector<BilinearPatchPrimitive> Patches() const;

    /// Refuse an explicitly diagnostic-only baked shell at an RT boundary.
    /// `allow_diagnostic_shell` is intentionally opt-in for regression work;
    /// it has no effect on ordinary legacy scenes that carry no certificate.
    void RequireRayTracingCertified(bool allow_diagnostic_shell = false) const;
};

/// Load a leaf-bbox JSON export ("leaves" array of degree-(1,1) patches).
/// Throws std::runtime_error on I/O or parse failure.
LeafPatchScene LoadLeafPatchScene(const std::string &json_path);

/// Load a full input NURBS surface from JSON (as written by
/// python_experiments/export_surface_inputs.py):
///   { "name", "degree_u", "degree_v", "control_points"[nu][nv][3],
///     "weights" (null or [nu][nv]), "knotvector_u", "knotvector_v" }
/// Domains are derived from the clamped knot vectors.
SurfaceData LoadSurfaceDataJson(const std::string &json_path);

} // namespace mfem_raytracing

#endif
