// Multi-step surface degree reduction driver (Algorithm 1).
//
// Faithful port of python_experiments/multiple_step_degree_reduction_surfaces/
// degree_reduce_multiple_steps.py. Segment domains returned by the single-step
// backend are expressed in the parent surface's parameterization; this driver
// maps them into the root surface's (u, v) rectangle branch by branch.

#include "surface_multistep_reduction.hpp"

#include "curve_reduction_domain.hpp"
#include "surface_degree_reduction.hpp"

#include <cmath>
#include <deque>
#include <limits>
#include <sstream>

namespace mfem_raytracing
{
namespace
{

struct QueueItem
{
    SurfaceData surface;
    int current_degree_u = 0;
    int current_degree_v = 0;
    double remaining_budget = 0.0;
    double consumed_error = 0.0;
    std::pair<double, double> u_domain_global;
    std::pair<double, double> v_domain_global;
};

// Map a single-step segment into the root-surface (u, v) parameter rectangle.
void SegmentGlobalDomains(const ReducedSurfaceSegment &segment,
                          const SurfaceData &parent_surface,
                          const std::pair<double, double> &parent_u_global,
                          const std::pair<double, double> &parent_v_global,
                          std::pair<double, double> &child_u_global,
                          std::pair<double, double> &child_v_global)
{
    child_u_global =
        MapIntervalToGlobal(segment.u_domain, parent_surface.u_domain, parent_u_global);
    child_v_global =
        MapIntervalToGlobal(segment.v_domain, parent_surface.v_domain, parent_v_global);
}

} // namespace

MultipleStepSurfaceReductionResult DegreeReduceMultipleSteps(
    const SurfaceData &initial_surface,
    int n_steps,
    double max_error,
    const MultiStepSurfaceReductionOptions &options)
{
    if (max_error < 0.0)
    {
        throw std::invalid_argument("DegreeReduceMultipleSteps: max_error must be non-negative");
    }
    if (n_steps < 0)
    {
        throw std::invalid_argument("DegreeReduceMultipleSteps: n_steps must be non-negative");
    }

    const int input_degree_u = initial_surface.degree_u;
    const int input_degree_v = initial_surface.degree_v;
    const std::pair<double, double> root_u_domain = initial_surface.u_domain;
    const std::pair<double, double> root_v_domain = initial_surface.v_domain;

    if (n_steps == 0)
    {
        return {{ReducedSurfaceLeaf{initial_surface, 0.0, 0, root_u_domain, root_v_domain}}};
    }

    const int max_steps_u = input_degree_u - 1;
    const int max_steps_v = input_degree_v - 1;
    if (n_steps > max_steps_u || n_steps > max_steps_v)
    {
        std::ostringstream oss;
        oss << "cannot reduce " << n_steps << " steps from degrees (" << input_degree_u << ", "
            << input_degree_v << "); maximum is (" << max_steps_u << ", " << max_steps_v << ")";
        throw MultipleStepSurfaceReductionFailure(oss.str());
    }

    const SurfaceSingleStepBackend backend =
        options.backend
            ? options.backend
            : [&options](const SurfaceData &surface, double tol) {
                  return PeakErrorSurfaceSingleStep(surface, tol, options.single_step);
              };

    const int target_degree_u = input_degree_u - n_steps;
    const int target_degree_v = input_degree_v - n_steps;
    const bool finite_budget = std::isfinite(max_error);

    std::deque<QueueItem> queue;
    queue.push_back(QueueItem{initial_surface,
                              input_degree_u,
                              input_degree_v,
                              max_error,
                              0.0,
                              root_u_domain,
                              root_v_domain});

    std::vector<ReducedSurfaceLeaf> leaves;

    while (!queue.empty())
    {
        const QueueItem item = std::move(queue.front());
        queue.pop_front();

        if (item.current_degree_u == target_degree_u && item.current_degree_v == target_degree_v)
        {
            leaves.push_back(ReducedSurfaceLeaf{item.surface,
                                                item.consumed_error,
                                                n_steps,
                                                item.u_domain_global,
                                                item.v_domain_global});
            continue;
        }

        const int step_index = input_degree_u - item.current_degree_u + 1;
        const double step_tol = StepErrorBudget(max_error,
                                                n_steps,
                                                step_index,
                                                item.remaining_budget,
                                                options.budget_policy);
        const SingleStepSurfaceReductionResult step_result = backend(item.surface, step_tol);
        for (const ReducedSurfaceSegment &segment : step_result.segments)
        {
            const double next_consumed = item.consumed_error + segment.segment_error;
            if (finite_budget && next_consumed > max_error)
            {
                continue;
            }
            const double next_budget = finite_budget
                                           ? max_error - next_consumed
                                           : std::numeric_limits<double>::infinity();
            std::pair<double, double> child_u_global;
            std::pair<double, double> child_v_global;
            SegmentGlobalDomains(segment,
                                 item.surface,
                                 item.u_domain_global,
                                 item.v_domain_global,
                                 child_u_global,
                                 child_v_global);
            queue.push_back(QueueItem{segment.surface,
                                      item.current_degree_u - 1,
                                      item.current_degree_v - 1,
                                      next_budget,
                                      next_consumed,
                                      child_u_global,
                                      child_v_global});
        }
    }

    return {std::move(leaves)};
}

} // namespace mfem_raytracing
