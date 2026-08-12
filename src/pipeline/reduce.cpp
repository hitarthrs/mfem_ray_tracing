#include "mfem_raytracing/pipeline/reduce.hpp"

#include "mfem_raytracing/reduction/surface_conforming_reduction.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mfem_raytracing
{
namespace
{

constexpr double kHardSeamTolerance = 1e-12;

std::vector<double> UniqueKnots(const std::vector<double> &knots)
{
    std::vector<double> unique = knots;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end(),
                             [](double a, double b) {
                                 return std::fabs(a - b) <= kHardSeamTolerance;
                             }),
                 unique.end());
    return unique;
}

bool CrossesHardSeam(double lo, double hi, const std::vector<double> &knots)
{
    for (const double knot : knots)
    {
        if (lo + kHardSeamTolerance < knot && knot < hi - kHardSeamTolerance)
        {
            return true;
        }
    }
    return false;
}

} // namespace

BilinearLeafCollection ReducePatchToBilinearLeaves(const SurfaceData &surface,
                                                   double max_error,
                                                   const PatchLeafReductionOptions &options)
{
    if (max_error < 0.0)
    {
        throw std::invalid_argument("ReducePatchToBilinearLeaves: max_error must be non-negative");
    }
    const int n_steps = std::max(surface.degree_u - 1, surface.degree_v - 1);
    if (n_steps < 1)
    {
        throw std::invalid_argument(
            "ReducePatchToBilinearLeaves: surface is already degree (1,1)");
    }

    const std::vector<double> hard_u = UniqueKnots(surface.knotvector_u);
    const std::vector<double> hard_v = UniqueKnots(surface.knotvector_v);

    ConformingReductionOptions reduction;
    reduction.budget_policy = ErrorBudgetPolicy::WeightedLate;
    reduction.single_step = options.single_step;
    reduction.coalesce = true;
    reduction.hard_seams = true;
    reduction.target_degree_u = 1;
    reduction.target_degree_v = 1;
    reduction.threads = options.threads;

    const MultipleStepSurfaceReductionResult coalesced =
        DegreeReduceMultipleStepsConforming(surface, max_error, reduction);

    BilinearLeafCollection collection;
    collection.max_error = max_error;
    collection.n_steps = n_steps;
    collection.backend_name = "approach_4_competing_coalesced_hard_seams";
    collection.n_competing = coalesced.n_competing;
    collection.seconds_compete = coalesced.seconds_compete;
    collection.seconds_coalesce = coalesced.seconds_coalesce;
    collection.leaves.reserve(coalesced.segments.size());
    for (std::size_t i = 0; i < coalesced.segments.size(); ++i)
    {
        const ReducedSurfaceLeaf &leaf = coalesced.segments[i];
        if (leaf.surface.degree_u != 1 || leaf.surface.degree_v != 1 ||
            leaf.surface.NumControlPointsU() != 2 || leaf.surface.NumControlPointsV() != 2 ||
            CrossesHardSeam(leaf.u_domain_global.first, leaf.u_domain_global.second, hard_u) ||
            CrossesHardSeam(leaf.v_domain_global.first, leaf.v_domain_global.second, hard_v))
        {
            throw std::runtime_error(
                "ReducePatchToBilinearLeaves: invalid coalesced leaf (hard-seam violation)");
        }
        if (leaf.total_error > max_error + 1e-10)
        {
            throw std::runtime_error("ReducePatchToBilinearLeaves: leaf exceeds max_error");
        }
        collection.leaves.push_back(ExtractBilinearLeafPatch(leaf, static_cast<int>(i)));
    }
    return collection;
}

HardSeamBilinearizationResult ReducePatchesToBilinearLeaves(
    const SurfacePatchCatalog &catalog, const std::vector<int> &patch_ids,
    const HardSeamBilinearizationOptions &options)
{
    return BilinearizePatchesWithHardSeams(catalog, patch_ids, options);
}

} // namespace mfem_raytracing
