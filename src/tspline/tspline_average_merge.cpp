#include "mfem_raytracing/tspline/tspline_average_merge.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

struct HomogeneousPolyline
{
    std::vector<double> parameters;
    std::vector<HomogeneousPoint> controls;
};

ParameterRange Domain(const LeafPatch &leaf, int axis)
{
    return axis == 0 ? ParameterRange{leaf.u_domain_global[0], leaf.u_domain_global[1]}
                     : ParameterRange{leaf.v_domain_global[0], leaf.v_domain_global[1]};
}

HomogeneousPolyline PolylineFromChain(const BoundaryChain &chain, bool reversed)
{
    if (chain.segments.empty() || !(chain.tangent_range.hi > chain.tangent_range.lo))
    {
        throw std::invalid_argument("cannot make a polyline from an empty boundary chain");
    }
    HomogeneousPolyline result;
    result.parameters.reserve(chain.segments.size() + 1);
    result.controls.reserve(chain.segments.size() + 1);
    const double span = chain.tangent_range.hi - chain.tangent_range.lo;
    result.parameters.push_back(0.0);
    result.controls.push_back(chain.segments.front().endpoints[0]);
    for (const BoundaryChainSegment &segment : chain.segments)
    {
        result.parameters.push_back((segment.tangent_range.hi - chain.tangent_range.lo) / span);
        result.controls.push_back(segment.endpoints[1]);
    }
    if (!reversed) { return result; }

    std::reverse(result.controls.begin(), result.controls.end());
    for (double &parameter : result.parameters) { parameter = 1.0 - parameter; }
    std::reverse(result.parameters.begin(), result.parameters.end());
    return result;
}

HomogeneousPoint EvaluatePolyline(const HomogeneousPolyline &polyline, double parameter,
                                  double tolerance)
{
    if (polyline.parameters.size() < 2 || polyline.parameters.size() != polyline.controls.size())
    {
        throw std::invalid_argument("invalid homogeneous polyline");
    }
    parameter = std::min(std::max(parameter, polyline.parameters.front()), polyline.parameters.back());
    const auto upper = std::upper_bound(polyline.parameters.begin(), polyline.parameters.end(), parameter);
    std::size_t i = upper == polyline.parameters.begin() ? 0 :
                    static_cast<std::size_t>(upper - polyline.parameters.begin() - 1);
    if (i + 1 >= polyline.parameters.size()) { return polyline.controls.back(); }
    const double lo = polyline.parameters[i];
    const double hi = polyline.parameters[i + 1];
    if (hi - lo <= tolerance) { throw std::runtime_error("homogeneous polyline has an empty segment"); }
    return InterpolateHomogeneous(polyline.controls[i], polyline.controls[i + 1],
                                  (parameter - lo) / (hi - lo));
}

std::vector<double> UnionParameters(const HomogeneousPolyline &a, const HomogeneousPolyline &b,
                                    double tolerance)
{
    std::vector<double> result = a.parameters;
    result.insert(result.end(), b.parameters.begin(), b.parameters.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end(), [tolerance](double x, double y) {
        return std::abs(x - y) <= tolerance;
    }), result.end());
    if (result.size() < 2)
    {
        throw std::runtime_error("merged seam needs at least two distinct parameters");
    }
    result.front() = 0.0;
    result.back() = 1.0;
    return result;
}

bool ShouldReverse(const BoundaryChain &a, const BoundaryChain &b)
{
    const HomogeneousPoint &a0 = a.segments.front().endpoints[0];
    const HomogeneousPoint &a1 = a.segments.back().endpoints[1];
    const HomogeneousPoint &b0 = b.segments.front().endpoints[0];
    const HomogeneousPoint &b1 = b.segments.back().endpoints[1];
    const double direct = HomogeneousCartesianDistance(a0, b0) +
                          HomogeneousCartesianDistance(a1, b1);
    const double reverse = HomogeneousCartesianDistance(a0, b1) +
                           HomogeneousCartesianDistance(a1, b0);
    return reverse < direct;
}

double ReseatOuterSeam(SeamRegion &region, const BoundaryChain &chain, bool reversed,
                       const HomogeneousPolyline &merged, double tolerance)
{
    const std::array<int, 2> corners = BoundaryCornerIndices(region.side);
    const double span = chain.tangent_range.hi - chain.tangent_range.lo;
    std::unordered_map<std::size_t, SeamRegionLeaf *> by_source;
    for (SeamRegionLeaf &leaf : region.leaves) { by_source.emplace(leaf.source_leaf_index, &leaf); }

    double displacement = 0.0;
    for (const BoundaryChainSegment &segment : chain.segments)
    {
        const auto found = by_source.find(segment.leaf_index);
        if (found == by_source.end())
        {
            throw std::runtime_error("merged seam chain references a leaf outside its region");
        }
        const double native[2] = {segment.tangent_range.lo, segment.tangent_range.hi};
        HomogeneousBilinearNet net = HomogenizeBilinearLeaf(found->second->leaf);
        for (int endpoint = 0; endpoint < 2; ++endpoint)
        {
            double parameter = (native[endpoint] - chain.tangent_range.lo) / span;
            if (reversed) { parameter = 1.0 - parameter; }
            const HomogeneousPoint target = EvaluatePolyline(merged, parameter, tolerance);
            displacement = std::max(displacement,
                                    HomogeneousCartesianDistance(net[corners[endpoint]], target));
            net[corners[endpoint]] = target;
        }
        SetHomogeneousBilinearNet(found->second->leaf, net);
    }
    return displacement;
}

} // namespace

SeamRegion MakeSeamRegion(const BilinearLeafAssembly &assembly, const BoundaryBand &band)
{
    if (assembly.patch_id != band.patch_id)
    {
        throw std::invalid_argument("cannot make a seam region from another patch's band");
    }
    SeamRegion result;
    result.patch_id = assembly.patch_id;
    result.role = assembly.role;
    result.side = band.side;
    result.u_range = {std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity()};
    result.v_range = result.u_range;
    result.leaves.reserve(band.leaf_indices.size());
    for (const std::size_t source_index : band.leaf_indices)
    {
        if (source_index >= assembly.leaves.size())
        {
            throw std::out_of_range("seam-region source leaf index is outside its assembly");
        }
        const LeafPatch &leaf = assembly.leaves[source_index];
        result.leaves.push_back({source_index, leaf});
        const ParameterRange u = Domain(leaf, 0);
        const ParameterRange v = Domain(leaf, 1);
        result.u_range.lo = std::min(result.u_range.lo, u.lo);
        result.u_range.hi = std::max(result.u_range.hi, u.hi);
        result.v_range.lo = std::min(result.v_range.lo, v.lo);
        result.v_range.hi = std::max(result.v_range.hi, v.hi);
    }
    if (result.leaves.empty()) { throw std::invalid_argument("cannot make an empty seam region"); }
    return result;
}

AverageMergedSeam AverageMergeSeam(const BilinearLeafAssembly &assembly_a,
                                   const BoundaryBand &band_a,
                                   const BilinearLeafAssembly &assembly_b,
                                   const BoundaryBand &band_b,
                                   const AverageSeamMergeOptions &options)
{
    if (options.parameter_tolerance < 0.0 || options.geometry_tolerance < 0.0)
    {
        throw std::invalid_argument("average merge tolerances must be non-negative");
    }
    const BoundaryChainOptions chain_options = {options.parameter_tolerance,
                                                options.geometry_tolerance};
    AverageMergedSeam result;
    result.chain_a = BuildBoundaryChain(assembly_a, band_a, chain_options);
    result.chain_b = BuildBoundaryChain(assembly_b, band_b, chain_options);
    result.region_a = MakeSeamRegion(assembly_a, band_a);
    result.region_b = MakeSeamRegion(assembly_b, band_b);
    result.b_reversed = ShouldReverse(result.chain_a, result.chain_b);

    const HomogeneousPolyline a = PolylineFromChain(result.chain_a, false);
    const HomogeneousPolyline b = PolylineFromChain(result.chain_b, result.b_reversed);
    result.union_parameters = UnionParameters(a, b, options.parameter_tolerance);
    result.merged_controls.reserve(result.union_parameters.size());
    for (const double parameter : result.union_parameters)
    {
        const HomogeneousPoint ha = EvaluatePolyline(a, parameter, options.parameter_tolerance);
        const HomogeneousPoint hb = EvaluatePolyline(b, parameter, options.parameter_tolerance);
        result.max_pre_merge_disagreement = std::max(
            result.max_pre_merge_disagreement, HomogeneousCartesianDistance(ha, hb));
        result.merged_controls.push_back(InterpolateHomogeneous(ha, hb, 0.5));
    }
    const HomogeneousPolyline merged = {result.union_parameters, result.merged_controls};
    result.max_displacement_a = ReseatOuterSeam(result.region_a, result.chain_a, false, merged,
                                                 options.parameter_tolerance);
    result.max_displacement_b = ReseatOuterSeam(result.region_b, result.chain_b, result.b_reversed,
                                                 merged, options.parameter_tolerance);
    return result;
}

} // namespace tspline
} // namespace mfem_raytracing
