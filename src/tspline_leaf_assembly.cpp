#include "tspline_leaf_assembly.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

ParameterRange Domain(const LeafPatch &leaf, int axis)
{
    return axis == 0 ? ParameterRange{leaf.u_domain_global[0], leaf.u_domain_global[1]}
                     : ParameterRange{leaf.v_domain_global[0], leaf.v_domain_global[1]};
}

double Endpoint(const LeafPatch &leaf, int axis, bool maximum)
{
    const ParameterRange range = Domain(leaf, axis);
    return maximum ? range.hi : range.lo;
}

HomogeneousPoint ToHomogeneous(const LeafPatch &leaf, int corner)
{
    const double weight = leaf.patch.rational ? leaf.patch.weights[corner] : 1.0;
    if (std::abs(weight) <= 1e-14)
    {
        throw std::invalid_argument("bilinear leaf has a zero projective weight");
    }
    return {weight * leaf.patch.control_points[corner][0],
            weight * leaf.patch.control_points[corner][1],
            weight * leaf.patch.control_points[corner][2], weight};
}

double Distance(const HomogeneousPoint &a, const HomogeneousPoint &b)
{
    const double dx = a[0] / a[3] - b[0] / b[3];
    const double dy = a[1] / a[3] - b[1] / b[3];
    const double dz = a[2] / a[3] - b[2] / b[3];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void SortAndUnique(std::vector<double> &values, double tolerance)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end(), [tolerance](double a, double b) {
        return std::abs(a - b) <= tolerance;
    }), values.end());
}

} // namespace

std::vector<BilinearLeafAssembly> BuildBilinearLeafAssemblies(const LeafPatchScene &scene)
{
    std::map<int, BilinearLeafAssembly> grouped;
    for (const LeafPatch &leaf : scene.leaves)
    {
        BilinearLeafAssembly &assembly = grouped[leaf.patch_id];
        assembly.patch_id = leaf.patch_id;
        if (assembly.leaves.empty()) { assembly.role = leaf.role; }
        else if (assembly.role != leaf.role) { assembly.role = "mixed"; }
        assembly.leaves.push_back(leaf);
    }

    std::vector<BilinearLeafAssembly> result;
    result.reserve(grouped.size());
    for (auto &entry : grouped)
    {
        BilinearLeafAssembly assembly = std::move(entry.second);
        if (assembly.leaves.empty()) { continue; }
        assembly.u_range = {std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity()};
        assembly.v_range = assembly.u_range;
        for (const LeafPatch &leaf : assembly.leaves)
        {
            const ParameterRange u = Domain(leaf, 0);
            const ParameterRange v = Domain(leaf, 1);
            if (!(u.hi > u.lo) || !(v.hi > v.lo))
            {
                throw std::invalid_argument("bilinear leaf has an empty parameter domain");
            }
            assembly.u_range.lo = std::min(assembly.u_range.lo, u.lo);
            assembly.u_range.hi = std::max(assembly.u_range.hi, u.hi);
            assembly.v_range.lo = std::min(assembly.v_range.lo, v.lo);
            assembly.v_range.hi = std::max(assembly.v_range.hi, v.hi);
        }
        result.push_back(std::move(assembly));
    }
    return result;
}

const BilinearLeafAssembly &FindBilinearLeafAssembly(
    const std::vector<BilinearLeafAssembly> &assemblies, int patch_id)
{
    const auto it = std::find_if(assemblies.begin(), assemblies.end(), [patch_id](const auto &assembly) {
        return assembly.patch_id == patch_id;
    });
    if (it == assemblies.end())
    {
        throw std::out_of_range("no bilinear leaf assembly for patch " + std::to_string(patch_id));
    }
    return *it;
}

BoundaryBand SelectBoundaryBand(const BilinearLeafAssembly &assembly, BoundarySide side,
                                std::size_t rows, double parameter_tolerance)
{
    if (rows == 0) { throw std::invalid_argument("boundary band needs at least one row"); }
    if (parameter_tolerance < 0.0)
    {
        throw std::invalid_argument("boundary-band tolerance must be non-negative");
    }
    const int normal_axis = BoundaryNormalAxis(side);
    std::vector<double> breaks;
    breaks.reserve(2 * assembly.leaves.size());
    for (const LeafPatch &leaf : assembly.leaves)
    {
        const ParameterRange range = Domain(leaf, normal_axis);
        breaks.push_back(range.lo);
        breaks.push_back(range.hi);
    }
    SortAndUnique(breaks, parameter_tolerance);
    if (breaks.size() < 2) { throw std::invalid_argument("assembly has no parameter rows"); }
    const bool high = BoundaryIsMaximum(side);
    const std::size_t cut_index = high ?
        (breaks.size() > rows ? breaks.size() - rows - 1 : 0) :
        std::min(rows, breaks.size() - 1);
    const double cut = breaks[cut_index];

    BoundaryBand result;
    result.patch_id = assembly.patch_id;
    result.side = side;
    result.normal_range = high ? ParameterRange{cut, breaks.back()}
                               : ParameterRange{breaks.front(), cut};
    for (std::size_t i = 0; i < assembly.leaves.size(); ++i)
    {
        const ParameterRange range = Domain(assembly.leaves[i], normal_axis);
        const bool belongs = high ? range.lo >= cut - parameter_tolerance
                                  : range.hi <= cut + parameter_tolerance;
        if (belongs) { result.leaf_indices.push_back(i); }
    }
    if (result.leaf_indices.empty()) { throw std::runtime_error("boundary band selected no leaves"); }
    return result;
}

BoundaryChain BuildBoundaryChain(const BilinearLeafAssembly &assembly, const BoundaryBand &band,
                                 const BoundaryChainOptions &options)
{
    if (band.patch_id != assembly.patch_id)
    {
        throw std::invalid_argument("boundary band belongs to another assembly");
    }
    if (options.parameter_tolerance < 0.0 || options.geometry_tolerance < 0.0)
    {
        throw std::invalid_argument("boundary-chain tolerances must be non-negative");
    }
    const int normal_axis = BoundaryNormalAxis(band.side);
    const int tangent_axis = BoundaryTangentAxis(band.side);
    const bool high = BoundaryIsMaximum(band.side);
    const ParameterRange assembly_normal = normal_axis == 0 ? assembly.u_range : assembly.v_range;
    const double boundary = high ? assembly_normal.hi : assembly_normal.lo;

    BoundaryChain result;
    result.patch_id = assembly.patch_id;
    result.side = band.side;
    result.boundary_value = boundary;
    const std::array<int, 2> corners = BoundaryCornerIndices(band.side);
    for (const std::size_t leaf_index : band.leaf_indices)
    {
        if (leaf_index >= assembly.leaves.size())
        {
            throw std::out_of_range("boundary band leaf index is outside its assembly");
        }
        const LeafPatch &leaf = assembly.leaves[leaf_index];
        if (std::abs(Endpoint(leaf, normal_axis, high) - boundary) > options.parameter_tolerance)
        {
            continue;
        }
        const ParameterRange tangent = Domain(leaf, tangent_axis);
        result.segments.push_back({leaf_index, tangent,
                                   {ToHomogeneous(leaf, corners[0]),
                                    ToHomogeneous(leaf, corners[1])}});
    }
    if (result.segments.empty()) { throw std::runtime_error("boundary band has no leaves on its outer edge"); }
    std::sort(result.segments.begin(), result.segments.end(), [](const auto &a, const auto &b) {
        if (a.tangent_range.lo != b.tangent_range.lo) { return a.tangent_range.lo < b.tangent_range.lo; }
        return a.leaf_index < b.leaf_index;
    });

    result.tangent_range = {result.segments.front().tangent_range.lo,
                            result.segments.back().tangent_range.hi};
    for (std::size_t i = 0; i < result.segments.size(); ++i)
    {
        const BoundaryChainSegment &segment = result.segments[i];
        if (!(segment.tangent_range.hi > segment.tangent_range.lo))
        {
            throw std::runtime_error("boundary chain has an empty tangent segment");
        }
        if (i == 0) { continue; }
        const BoundaryChainSegment &previous = result.segments[i - 1];
        if (std::abs(segment.tangent_range.lo - previous.tangent_range.hi) >
            options.parameter_tolerance)
        {
            throw std::runtime_error("boundary leaves are not a contiguous parameter chain");
        }
        if (Distance(previous.endpoints[1], segment.endpoints[0]) > options.geometry_tolerance)
        {
            throw std::runtime_error("boundary leaves are not C0 in world coordinates");
        }
    }
    return result;
}

} // namespace tspline
} // namespace mfem_raytracing
