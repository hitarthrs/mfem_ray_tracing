#ifndef SURFACE_MULTISTEP_REDUCTION_HPP
#define SURFACE_MULTISTEP_REDUCTION_HPP

// Multi-step B-spline / NURBS surface degree reduction (Algorithm 1).
//
// Breadth-first queue over tensor-product surface patches. Each step performs
// (p_u, p_v) -> (p_u - 1, p_v - 1) via the adaptive single-step backend
// (PeakErrorSurfaceSingleStep), which may split the patch. Port of
// python_experiments/multiple_step_degree_reduction_surfaces/degree_reduce_multiple_steps.py.

#include "mfem_raytracing/reduction/curve_reduction_error_budget.hpp"
#include "mfem_raytracing/reduction/surface_reduction_types.hpp"

#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfem_raytracing
{

/// One output surface at the target degrees with cumulative error along its path.
struct ReducedSurfaceLeaf
{
    SurfaceData surface;
    double total_error = 0.0;
    int steps_taken = 0;
    /// Axis-aligned rectangle on the *original* input surface covered by this leaf.
    std::pair<double, double> u_domain_global = {0.0, 1.0};
    std::pair<double, double> v_domain_global = {0.0, 1.0};
};

/// All accepted leaves from a multi-step surface reduction run.
struct MultipleStepSurfaceReductionResult
{
    std::vector<ReducedSurfaceLeaf> segments;
    /// Conforming competing-grid cell count before coalescing (equal to
    /// `segments.size()` when coalesce was disabled).
    std::size_t n_competing = 0;
    /// Wall time of competing grid build (seconds).
    double seconds_compete = 0.0;
    /// Wall time of hard-seam coalesce pass (seconds); 0 when coalesce off.
    double seconds_coalesce = 0.0;
};

/// Raised when n_steps is invalid for the input degrees.
class MultipleStepSurfaceReductionFailure : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

/// Single-step backend signature: reduce (p_u, p_v) -> (p_u - 1, p_v - 1) under
/// the given tolerance, possibly splitting into several segments.
using SurfaceSingleStepBackend =
    std::function<SingleStepSurfaceReductionResult(const SurfaceData &, double)>;

struct MultiStepSurfaceReductionOptions
{
    /// How each single-step tolerance is derived from the global budget.
    ErrorBudgetPolicy budget_policy = ErrorBudgetPolicy::Cumulative;
    /// Options forwarded to the default PeakErrorSurfaceSingleStep backend.
    PeakErrorSurfaceSingleStepOptions single_step;
    /// Custom backend; when empty, PeakErrorSurfaceSingleStep(single_step) is used.
    SurfaceSingleStepBackend backend;
};

/**
 * Reduce `initial_surface` by `n_steps` degree levels in u and v.
 *
 * `max_error` is the global error budget for each branch (cumulative cap);
 * children whose cumulative error would exceed it are dropped.
 *
 * Throws std::invalid_argument for negative `max_error`/`n_steps` and
 * MultipleStepSurfaceReductionFailure when `n_steps` exceeds
 * min(degree_u, degree_v) - 1.
 */
MultipleStepSurfaceReductionResult DegreeReduceMultipleStepsNonConforming(
    const SurfaceData &initial_surface,
    int n_steps,
    double max_error,
    const MultiStepSurfaceReductionOptions &options = {});

} // namespace mfem_raytracing

#endif
