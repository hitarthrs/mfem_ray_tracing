#ifndef MFEM_RAYTRACING_PIPELINE_MULTI_PATCH_TMESH_HPP
#define MFEM_RAYTRACING_PIPELINE_MULTI_PATCH_TMESH_HPP

// Public Stage 3: always-materialized multi-patch degree-1 T-mesh mid-product.

#include "mfem_raytracing/pipeline/connect.hpp"
#include "mfem_raytracing/tspline/tspline.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

struct MultiPatchTMeshComponent
{
    enum class Kind
    {
        SeamStrip,
        InteriorPatch,
    };

    Kind kind = Kind::InteriorPatch;
    int patch_id = -1;
    std::string interface_id;
    TMesh mesh;
    std::vector<LocalTSplineFace> faces;
};

/// Real multi-patch T-mesh mid-product (non-RT evaluable; bake separately for RT).
struct MultiPatchTMesh
{
    SeamAssembly assembly;
    std::vector<MultiPatchTMeshComponent> components;

    /// Parameter-disjoint catalog of all components as one TMesh (each seam
    /// strip already shares controls internally). Prefer EvaluateComponent for
    /// meaningful local (s,t); EvaluateUnified is for island-chart queries.
    TMesh unified;
    bool has_unified = false;

    /// Evaluate a single component with a degree-1 (default) or higher T-spline.
    Point3 EvaluateComponent(std::size_t component_index, double s, double t,
                             int degree = 1) const;

    /// Evaluate on the unified mesh when has_unified is true.
    Point3 EvaluateUnified(double s, double t, int degree = 1) const;

    /// Degree-1 leaf extraction over every component (and unified when present).
    std::vector<TSplineLeaf> ExtractAllDegreeOneLeaves() const;
};

/// Materialize local strip T-meshes + interior leaf nets, then unify controls.
MultiPatchTMesh BuildMultiPatchTMesh(SeamAssembly assembly);

} // namespace tspline
} // namespace mfem_raytracing

#endif
