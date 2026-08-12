#include "mfem_raytracing/tspline/tspline_corner_collar.hpp"

#include "mfem_raytracing/tspline/tspline_bilinear_ops.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

struct WorkingLeaf
{
    LeafPatch leaf;
    SourceLeafRef source;
};

struct Assignment
{
    std::size_t leaf_index = 0;
    int corner = 0;
    HomogeneousPoint target{};
    std::string interface_id;
    long long parameter_key = 0;
    double displacement = 0.0;
};

class DisjointSet
{
public:
    explicit DisjointSet(std::size_t n) : parent_(n), rank_(n, 0)
    {
        for (std::size_t i = 0; i < n; ++i) { parent_[i] = i; }
    }

    std::size_t Find(std::size_t a)
    {
        if (parent_[a] != a) { parent_[a] = Find(parent_[a]); }
        return parent_[a];
    }

    void Unite(std::size_t a, std::size_t b)
    {
        a = Find(a); b = Find(b);
        if (a == b) { return; }
        if (rank_[a] < rank_[b]) { std::swap(a, b); }
        parent_[b] = a;
        if (rank_[a] == rank_[b]) { ++rank_[a]; }
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<int> rank_;
};

ParameterRange Domain(const LeafPatch &leaf, int axis)
{
    return axis == 0 ? ParameterRange{leaf.u_domain_global[0], leaf.u_domain_global[1]}
                     : ParameterRange{leaf.v_domain_global[0], leaf.v_domain_global[1]};
}

bool Touches(const LeafPatch &leaf, BoundarySide side, const BoundaryChain &chain, double tolerance)
{
    const int axis = BoundaryNormalAxis(side);
    const ParameterRange domain = Domain(leaf, axis);
    const double value = chain.boundary_value;
    return BoundaryIsMaximum(side) ? std::abs(domain.hi - value) <= tolerance
                                   : std::abs(domain.lo - value) <= tolerance;
}

HomogeneousPoint EvaluateMerged(const BoundaryMergeConstraint &constraint, double oriented_parameter,
                                double tolerance)
{
    const std::vector<double> &parameters = constraint.union_parameters;
    const std::vector<HomogeneousPoint> &controls = constraint.merged_controls;
    if (parameters.size() < 2 || parameters.size() != controls.size())
    {
        throw std::invalid_argument("invalid merged boundary constraint");
    }
    oriented_parameter = std::max(parameters.front(), std::min(oriented_parameter, parameters.back()));
    const auto upper = std::upper_bound(parameters.begin(), parameters.end(), oriented_parameter);
    const std::size_t i = upper == parameters.begin() ? 0 :
        static_cast<std::size_t>(upper - parameters.begin() - 1);
    if (i + 1 >= parameters.size()) { return controls.back(); }
    const double lo = parameters[i];
    const double hi = parameters[i + 1];
    if (hi - lo <= tolerance) { throw std::runtime_error("empty merged boundary segment"); }
    return InterpolateHomogeneous(controls[i], controls[i + 1],
                                  (oriented_parameter - lo) / (hi - lo));
}

std::vector<double> NativeBreaks(const BoundaryMergeConstraint &constraint)
{
    const double lo = constraint.chain.tangent_range.lo;
    const double span = constraint.chain.tangent_range.hi - lo;
    if (!(span > 0.0)) { throw std::invalid_argument("boundary constraint has an empty tangent range"); }
    std::vector<double> result;
    result.reserve(constraint.union_parameters.size());
    for (const double p : constraint.union_parameters)
    {
        result.push_back(lo + (constraint.reversed ? 1.0 - p : p) * span);
    }
    return result;
}

void AppendUnique(std::vector<std::string> &values, const std::string &value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) { values.push_back(value); }
}

} // namespace

std::vector<CornerCollarLeaf> BuildExactBoundaryCornerCollar(
    const std::vector<BilinearLeafAssembly> &assemblies,
    const std::vector<BoundaryMergeConstraint> &constraints,
    const CornerCollarOptions &options)
{
    if (options.parameter_tolerance < 0.0 || options.geometry_tolerance < 0.0)
    {
        throw std::invalid_argument("corner-collar tolerances must be non-negative");
    }
    const double parameter_tolerance = std::max(options.parameter_tolerance, 1e-14);

    std::map<int, std::vector<const BoundaryMergeConstraint *>> by_patch;
    for (const BoundaryMergeConstraint &constraint : constraints)
    {
        if (constraint.patch_id < 0) { throw std::invalid_argument("boundary constraint has no patch id"); }
        by_patch[constraint.patch_id].push_back(&constraint);
    }

    std::vector<WorkingLeaf> working;
    for (const BilinearLeafAssembly &assembly : assemblies)
    {
        const auto found = by_patch.find(assembly.patch_id);
        const std::vector<const BoundaryMergeConstraint *> empty;
        const auto &patch_constraints = found == by_patch.end() ? empty : found->second;
        for (std::size_t source_index = 0; source_index < assembly.leaves.size(); ++source_index)
        {
            const LeafPatch &source_leaf = assembly.leaves[source_index];
            std::vector<LeafPatch> pieces = {source_leaf};
            for (int axis = 0; axis < 2; ++axis)
            {
                std::vector<double> breaks;
                for (const BoundaryMergeConstraint *constraint : patch_constraints)
                {
                    if (BoundaryTangentAxis(constraint->side) != axis ||
                        !Touches(source_leaf, constraint->side, constraint->chain, parameter_tolerance))
                    {
                        continue;
                    }
                    const std::vector<double> native = NativeBreaks(*constraint);
                    breaks.insert(breaks.end(), native.begin(), native.end());
                }
                if (breaks.empty()) { continue; }
                std::vector<LeafPatch> refined;
                for (const LeafPatch &piece : pieces)
                {
                    const std::vector<LeafPatch> split = RefineRationalBilinearLeafToBreaks(
                        piece, axis, breaks, parameter_tolerance);
                    refined.insert(refined.end(), split.begin(), split.end());
                }
                pieces = std::move(refined);
            }
            for (LeafPatch &piece : pieces)
            {
                working.push_back({std::move(piece), {assembly.patch_id, source_index}});
            }
        }
    }

    std::vector<Assignment> assignments;
    std::vector<std::set<std::string>> leaf_interfaces(working.size());
    for (std::size_t leaf_index = 0; leaf_index < working.size(); ++leaf_index)
    {
        WorkingLeaf &working_leaf = working[leaf_index];
        const auto found = by_patch.find(working_leaf.source.patch_id);
        if (found == by_patch.end()) { continue; }
        const HomogeneousBilinearNet original = HomogenizeBilinearLeaf(working_leaf.leaf);
        for (const BoundaryMergeConstraint *constraint : found->second)
        {
            if (!Touches(working_leaf.leaf, constraint->side, constraint->chain, parameter_tolerance))
            {
                continue;
            }
            const int tangent_axis = BoundaryTangentAxis(constraint->side);
            const ParameterRange tangent = Domain(working_leaf.leaf, tangent_axis);
            if (tangent.lo < constraint->chain.tangent_range.lo - parameter_tolerance ||
                tangent.hi > constraint->chain.tangent_range.hi + parameter_tolerance)
            {
                continue;
            }
            const double span = constraint->chain.tangent_range.hi - constraint->chain.tangent_range.lo;
            const std::array<int, 2> corners = BoundaryCornerIndices(constraint->side);
            const double native[2] = {tangent.lo, tangent.hi};
            for (int endpoint = 0; endpoint < 2; ++endpoint)
            {
                double p = (native[endpoint] - constraint->chain.tangent_range.lo) / span;
                p = std::max(0.0, std::min(1.0, p));
                const double oriented_p = constraint->reversed ? 1.0 - p : p;
                const HomogeneousPoint target = EvaluateMerged(*constraint, oriented_p,
                                                                parameter_tolerance);
                assignments.push_back({leaf_index, corners[endpoint], target, constraint->interface_id,
                                       static_cast<long long>(std::llround(oriented_p / parameter_tolerance)),
                                       HomogeneousCartesianDistance(original[corners[endpoint]], target)});
            }
            leaf_interfaces[leaf_index].insert(constraint->interface_id);
        }
    }

    DisjointSet sets(assignments.size());
    std::map<std::pair<std::size_t, int>, std::size_t> destination;
    std::map<std::pair<std::string, long long>, std::size_t> interface_point;
    for (std::size_t i = 0; i < assignments.size(); ++i)
    {
        const Assignment &assignment = assignments[i];
        const auto destination_inserted = destination.emplace(
            std::make_pair(assignment.leaf_index, assignment.corner), i);
        if (!destination_inserted.second) { sets.Unite(i, destination_inserted.first->second); }
        const auto interface_inserted = interface_point.emplace(
            std::make_pair(assignment.interface_id, assignment.parameter_key), i);
        if (!interface_inserted.second) { sets.Unite(i, interface_inserted.first->second); }
    }

    std::map<std::size_t, HomogeneousPoint> sums;
    std::map<std::size_t, std::size_t> counts;
    for (std::size_t i = 0; i < assignments.size(); ++i)
    {
        const std::size_t root = sets.Find(i);
        HomogeneousPoint &sum = sums[root];
        for (int c = 0; c < 4; ++c) { sum[c] += assignments[i].target[c]; }
        ++counts[root];
    }
    std::vector<double> max_displacement(working.size(), 0.0);
    std::map<std::pair<std::size_t, int>, HomogeneousPoint> final_controls;
    for (std::size_t i = 0; i < assignments.size(); ++i)
    {
        const std::size_t root = sets.Find(i);
        HomogeneousPoint average = sums[root];
        for (double &value : average) { value /= static_cast<double>(counts[root]); }
        final_controls[{assignments[i].leaf_index, assignments[i].corner}] = average;
    }
    for (const auto &[destination_key, target] : final_controls)
    {
        const std::size_t leaf_index = destination_key.first;
        const int corner = destination_key.second;
        HomogeneousBilinearNet net = HomogenizeBilinearLeaf(working[leaf_index].leaf);
        max_displacement[leaf_index] = std::max(max_displacement[leaf_index],
                                                HomogeneousCartesianDistance(net[corner], target));
        net[corner] = target;
        SetHomogeneousBilinearNet(working[leaf_index].leaf, net);
    }

    std::vector<CornerCollarLeaf> result;
    result.reserve(working.size());
    for (std::size_t i = 0; i < working.size(); ++i)
    {
        CornerCollarLeaf output;
        output.leaf = std::move(working[i].leaf);
        output.source = working[i].source;
        output.interface_ids.assign(leaf_interfaces[i].begin(), leaf_interfaces[i].end());
        output.seam_displacement = max_displacement[i];
        if (output.interface_ids.empty())
        {
            output.leaf.role = "interior";
            output.leaf.kind = "interior";
        }
        else if (output.interface_ids.size() == 1)
        {
            output.leaf.role = "seam-exact";
            output.leaf.kind = "side-collar";
            output.leaf.interface_id = output.interface_ids.front();
        }
        else
        {
            output.leaf.role = "seam-corner";
            output.leaf.kind = "corner-collar";
            output.leaf.interface_id = output.interface_ids.front();
        }
        result.push_back(std::move(output));
    }
    return result;
}

} // namespace tspline
} // namespace mfem_raytracing
