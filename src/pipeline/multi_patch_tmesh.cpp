#include "mfem_raytracing/pipeline/multi_patch_tmesh.hpp"

#include "mfem_raytracing/embree/bilinear_patch_geometry.hpp"
#include "mfem_raytracing/tspline/tspline_strip_builder.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

constexpr double kUnifyParamGap = 2.0;

Point3 CornerPosition(const LeafPatch &leaf, BilinearCorner corner)
{
    const int i = static_cast<int>(corner);
    return {leaf.patch.control_points[i][0], leaf.patch.control_points[i][1],
            leaf.patch.control_points[i][2]};
}

double CornerWeight(const LeafPatch &leaf, BilinearCorner corner)
{
    if (!leaf.patch.rational) { return 1.0; }
    return leaf.patch.weights[static_cast<int>(corner)];
}

TMesh MakeInteriorLeafTMesh(const LeafPatch &leaf)
{
    TMesh mesh;
    const double s0 = leaf.u_domain_global[0];
    const double s1 = leaf.u_domain_global[1];
    const double t0 = leaf.v_domain_global[0];
    const double t1 = leaf.v_domain_global[1];
    const std::size_t i00 =
        mesh.AddControlPoint({s0, t0}, CornerPosition(leaf, BilinearCorner::P00),
                             CornerWeight(leaf, BilinearCorner::P00));
    const std::size_t i01 =
        mesh.AddControlPoint({s0, t1}, CornerPosition(leaf, BilinearCorner::P01),
                             CornerWeight(leaf, BilinearCorner::P01));
    const std::size_t i10 =
        mesh.AddControlPoint({s1, t0}, CornerPosition(leaf, BilinearCorner::P10),
                             CornerWeight(leaf, BilinearCorner::P10));
    const std::size_t i11 =
        mesh.AddControlPoint({s1, t1}, CornerPosition(leaf, BilinearCorner::P11),
                             CornerWeight(leaf, BilinearCorner::P11));
    mesh.AddEdge(i00, i10);
    mesh.AddEdge(i10, i11);
    mesh.AddEdge(i11, i01);
    mesh.AddEdge(i01, i00);
    mesh.Validate();
    return mesh;
}

/// Append `source` into `dest` as a disjoint (s,t) island. Seam-local shared
/// controls already live inside each strip component; cross-component vertex
/// merging would break TMesh rectilinearity in the unified parameter chart.
void AppendComponentIsland(TMesh &dest, const TMesh &source, double s_offset, double t_offset)
{
    std::vector<std::size_t> old_to_new(source.ControlPoints().size(),
                                        static_cast<std::size_t>(-1));
    for (std::size_t i = 0; i < source.ControlPoints().size(); ++i)
    {
        const ControlPoint &cp = source.ControlPoints()[i];
        const Point2 parameter = {cp.parameter[0] + s_offset, cp.parameter[1] + t_offset};
        old_to_new[i] = dest.AddControlPoint(parameter, cp.position, cp.weight);
    }
    for (const Edge &edge : source.Edges())
    {
        dest.AddEdge(old_to_new[edge.first], old_to_new[edge.second]);
    }
}

} // namespace

Point3 MultiPatchTMesh::EvaluateComponent(std::size_t component_index, double s, double t,
                                          int degree) const
{
    if (component_index >= components.size())
    {
        throw std::out_of_range("MultiPatchTMesh::EvaluateComponent: bad component index");
    }
    const TSplineSurface surface(components[component_index].mesh, degree);
    return surface.Evaluate(s, t);
}

Point3 MultiPatchTMesh::EvaluateUnified(double s, double t, int degree) const
{
    if (!has_unified)
    {
        throw std::logic_error("MultiPatchTMesh::EvaluateUnified: unified mesh was not built");
    }
    const TSplineSurface surface(unified, degree);
    return surface.Evaluate(s, t);
}

std::vector<TSplineLeaf> MultiPatchTMesh::ExtractAllDegreeOneLeaves() const
{
    std::vector<TSplineLeaf> leaves;
    for (const MultiPatchTMeshComponent &component : components)
    {
        if (component.mesh.ControlPoints().empty()) { continue; }
        const std::vector<TSplineLeaf> part = ExtractDegreeOneLeaves(component.mesh);
        leaves.insert(leaves.end(), part.begin(), part.end());
    }
    return leaves;
}

MultiPatchTMesh BuildMultiPatchTMesh(SeamAssembly assembly)
{
    MultiPatchTMesh result;
    result.assembly = std::move(assembly);
    ShellBuildOptions &options = result.assembly.options;

    // Always materialize local degree-1 strip T-meshes (non-RT mid-product).
    // Each strip already shares seam controls in its local TMesh.
    for (ShellStripResult &strip : result.assembly.strips)
    {
        strip.strip = BuildLocalDegreeOneStrip(strip.merge, options.strip_build);
        MultiPatchTMeshComponent component;
        component.kind = MultiPatchTMeshComponent::Kind::SeamStrip;
        component.interface_id = strip.interface_id;
        component.mesh = strip.strip.mesh;
        component.faces = strip.strip.faces;
        result.components.push_back(std::move(component));
    }

    // Interior leaves (not claimed by any seam band) as per-leaf T-meshes.
    std::set<SourceLeafRef> claimed;
    for (const auto &[source, claims] : result.assembly.ownership.claims_by_source)
    {
        (void)claims;
        claimed.insert(source);
    }
    for (const BilinearLeafAssembly &patch_assembly : result.assembly.assemblies)
    {
        for (std::size_t leaf_index = 0; leaf_index < patch_assembly.leaves.size(); ++leaf_index)
        {
            const SourceLeafRef source = {patch_assembly.patch_id, leaf_index};
            if (claimed.count(source) != 0) { continue; }
            MultiPatchTMeshComponent component;
            component.kind = MultiPatchTMeshComponent::Kind::InteriorPatch;
            component.patch_id = patch_assembly.patch_id;
            component.mesh = MakeInteriorLeafTMesh(patch_assembly.leaves[leaf_index]);
            result.components.push_back(std::move(component));
        }
    }

    // Unified chart: parameter-disjoint islands (shared world controls remain
    // inside each seam-strip component). Useful for catalog-style export /
    // EvaluateUnified over a single TMesh object.
    double s_offset = 0.0;
    for (const MultiPatchTMeshComponent &component : result.components)
    {
        const TMesh &source = component.mesh;
        if (source.ControlPoints().empty()) { continue; }
        double s_min = source.ControlPoints().front().parameter[0];
        double s_max = s_min;
        double t_min = source.ControlPoints().front().parameter[1];
        for (const ControlPoint &cp : source.ControlPoints())
        {
            s_min = std::min(s_min, cp.parameter[0]);
            s_max = std::max(s_max, cp.parameter[0]);
            t_min = std::min(t_min, cp.parameter[1]);
        }
        AppendComponentIsland(result.unified, source, s_offset - s_min, -t_min);
        s_offset += (s_max - s_min) + kUnifyParamGap;
    }
    if (!result.unified.ControlPoints().empty())
    {
        result.unified.Validate();
        result.has_unified = true;
    }
    return result;
}

} // namespace tspline
} // namespace mfem_raytracing
