#include "mfem_raytracing/tspline/tspline_patch_interfaces.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

constexpr BoundarySide kSides[] = {
    BoundarySide::UMin, BoundarySide::UMax, BoundarySide::VMin, BoundarySide::VMax,
};

std::size_t SideIndex(BoundarySide side)
{
    return static_cast<std::size_t>(side);
}

HomogeneousPoint ToHomogeneous(const std::vector<double> &point, double weight)
{
    if (point.size() != 3)
    {
        throw std::invalid_argument("NURBS boundary control points must be 3D");
    }
    return {weight * point[0], weight * point[1], weight * point[2], weight};
}

double CartesianDistance(const HomogeneousPoint &a, const HomogeneousPoint &b)
{
    if (std::abs(a[3]) <= 1e-14 || std::abs(b[3]) <= 1e-14)
    {
        throw std::invalid_argument("NURBS boundary has a zero weight");
    }
    const double dx = a[0] / a[3] - b[0] / b[3];
    const double dy = a[1] / a[3] - b[1] / b[3];
    const double dz = a[2] / a[3] - b[2] / b[3];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

BoundaryCurveMatch Compare(const NurbsBoundaryCurve &a, const NurbsBoundaryCurve &b,
                           bool reverse_b, double tolerance)
{
    BoundaryCurveMatch result;
    if (a.degree != b.degree || a.controls.size() != b.controls.size()) { return result; }
    for (std::size_t i = 0; i < a.controls.size(); ++i)
    {
        const HomogeneousPoint &pa = a.controls[i];
        const HomogeneousPoint &pb = b.controls[reverse_b ? b.controls.size() - i - 1 : i];
        result.max_point_disagreement = std::max(result.max_point_disagreement,
                                                  CartesianDistance(pa, pb));
        result.max_weight_disagreement = std::max(result.max_weight_disagreement,
                                                   std::abs(pa[3] - pb[3]));
    }
    result.matches = result.max_point_disagreement <= tolerance &&
                     result.max_weight_disagreement <= tolerance;
    result.b_reversed = reverse_b;
    return result;
}

} // namespace

NurbsBoundaryCurve ExtractNurbsBoundaryCurve(const SurfaceData &surface, BoundarySide side)
{
    const int normal_axis = BoundaryNormalAxis(side);
    const int tangent_axis = BoundaryTangentAxis(side);
    if (surface.degree_u < 0 || surface.degree_v < 0 || surface.control_points.empty() ||
        surface.control_points.front().empty())
    {
        throw std::invalid_argument("cannot extract boundary from an empty surface");
    }
    const std::size_t nu = surface.control_points.size();
    const std::size_t nv = surface.control_points.front().size();
    for (const auto &row : surface.control_points)
    {
        if (row.size() != nv) { throw std::invalid_argument("surface control net is ragged"); }
    }
    const bool rational = surface.IsRational();
    if (rational && surface.weights.size() != nu)
    {
        throw std::invalid_argument("surface weights do not match its control net");
    }
    if (rational)
    {
        for (const auto &row : surface.weights)
        {
            if (row.size() != nv)
            {
                throw std::invalid_argument("surface weights do not match its control net");
            }
        }
    }

    NurbsBoundaryCurve result;
    result.degree = tangent_axis == 0 ? surface.degree_u : surface.degree_v;
    result.knots = tangent_axis == 0 ? surface.knotvector_u : surface.knotvector_v;
    const std::size_t count = tangent_axis == 0 ? nu : nv;
    result.controls.reserve(count);
    const std::size_t fixed = BoundaryIsMaximum(side) ?
        (normal_axis == 0 ? nu - 1 : nv - 1) : 0;
    for (std::size_t tangent = 0; tangent < count; ++tangent)
    {
        const std::size_t iu = normal_axis == 0 ? fixed : tangent;
        const std::size_t iv = normal_axis == 0 ? tangent : fixed;
        const double weight = rational ? surface.weights[iu][iv] : 1.0;
        result.controls.push_back(ToHomogeneous(surface.control_points[iu][iv], weight));
    }
    return result;
}

BoundaryCurveMatch MatchNurbsBoundaryCurves(const NurbsBoundaryCurve &a,
                                            const NurbsBoundaryCurve &b,
                                            double tolerance)
{
    if (tolerance < 0.0) { throw std::invalid_argument("boundary tolerance must be non-negative"); }
    BoundaryCurveMatch direct = Compare(a, b, false, tolerance);
    if (direct.matches) { return direct; }
    BoundaryCurveMatch reversed = Compare(a, b, true, tolerance);
    if (reversed.matches) { return reversed; }
    return direct.max_point_disagreement <= reversed.max_point_disagreement ? direct : reversed;
}

std::vector<PatchInterface> DiscoverPatchInterfaces(const SurfacePatchCatalog &catalog,
                                                     const InterfaceDiscoveryOptions &options)
{
    return DiscoverPatchInterfaces(BuildPatchBoundaryIndex(catalog), options);
}

PatchBoundaryIndex BuildPatchBoundaryIndex(const SurfacePatchCatalog &catalog)
{
    PatchBoundaryIndex result;
    result.patches.reserve(catalog.patches.size());
    for (const SurfacePatchDescriptor &patch : catalog.patches)
    {
        PatchBoundaryIndexEntry entry;
        entry.patch_id = patch.id;
        for (const BoundarySide side : kSides)
        {
            entry.boundaries[SideIndex(side)] = ExtractNurbsBoundaryCurve(patch.surface, side);
        }
        result.patches.push_back(std::move(entry));
    }
    std::sort(result.patches.begin(), result.patches.end(), [](const auto &a, const auto &b) {
        return a.patch_id < b.patch_id;
    });
    return result;
}

std::vector<PatchInterface> DiscoverPatchInterfaces(const PatchBoundaryIndex &index,
                                                     const InterfaceDiscoveryOptions &options)
{
    if (options.tolerance < 0.0)
    {
        throw std::invalid_argument("interface discovery tolerance must be non-negative");
    }
    std::vector<PatchInterface> found;
    for (std::size_t i = 0; i < index.patches.size(); ++i)
    {
        for (std::size_t j = i + 1; j < index.patches.size(); ++j)
        {
            for (const BoundarySide side_a : kSides)
            {
                for (const BoundarySide side_b : kSides)
                {
                    const BoundaryCurveMatch match =
                        MatchNurbsBoundaryCurves(index.patches[i].boundaries[SideIndex(side_a)],
                                                  index.patches[j].boundaries[SideIndex(side_b)],
                                                  options.tolerance);
                    if (match.matches)
                    {
                        found.push_back({index.patches[i].patch_id, side_a,
                                         index.patches[j].patch_id, side_b,
                                         match.b_reversed, match.max_point_disagreement,
                                         match.max_weight_disagreement});
                    }
                }
            }
        }
    }
    return found;
}

} // namespace tspline
} // namespace mfem_raytracing
