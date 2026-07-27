#ifndef SURFACE_CONFORMING_REDUCTION_HPP
#define SURFACE_CONFORMING_REDUCTION_HPP

// Watertight multi-step surface degree reduction to bilinear leaves.
//
// Unlike DegreeReduceMultipleStepsNonConforming (per-branch local splits ->
// T-junctions and cracked leaf edges), every split here is a full iso-line
// across the whole surface, so the decomposition is always a conforming tensor
// grid: neighbours share identical edge control data and the deterministic,
// endpoint-preserving A5.11 reduction gives bitwise-identical shared edges (C0
// watertight) at every step. Ports the Python pipeline
//   multiple_step_degree_reduction_surfaces/competing_reduction.py  (grid)
//   multiple_step_degree_reduction_surfaces/coalesce_reduction.py   (A+C pass)
// including competing u/v refinement, per-cell budgets, greedy full-line
// coalescing with hard seams + corner welding, and (p, q) inputs via
// skip-direction steps.

#include "curve_reduction_error_budget.hpp"
#include "surface_multistep_reduction.hpp"
#include "surface_reduction_types.hpp"

#include <utility>

namespace mfem_raytracing
{

struct ConformingReductionOptions
{
    ErrorBudgetPolicy budget_policy = ErrorBudgetPolicy::WeightedLate; // approach_4
    /// Error accounting / Eq. 5.30 / refinement guards (sum + correction default).
    PeakErrorSurfaceSingleStepOptions single_step;
    /// Greedy full-line removal after the grid build (keeps watertightness).
    bool coalesce = true;
    /// Keep original unique knots as unremovable coalesce seams. Prevents
    /// multi-span degree-1 nets (non-2×2 bilinears) when the input has interior
    /// knots (e.g. the 9×9 rational torus). Matches the Python hard-seam
    /// coalesce. No effect when the only unique knots are the domain endpoints.
    bool hard_seams = true;
    /// Target degrees; (1, 1) yields bilinear leaves. Inputs may be (p, q) with
    /// p != q — once a direction reaches its target it is skipped.
    int target_degree_u = 1;
    int target_degree_v = 1;
    /// Worker threads for the grid-build passes (per-cell reductions/probes are
    /// independent). 0 = std::thread::hardware_concurrency(); 1 = serial. Output
    /// is bitwise-identical regardless of thread count. Coalescing stays serial.
    unsigned threads = 0;
};

/**
 * Reduce `initial_surface` (degrees (p, q)) to the target degrees on a
 * conforming tensor grid under the global `max_error` budget.
 *
 * Leaves cover the full parameter rectangle (over-budget cells are kept with a
 * warning rather than dropped, so the result stays watertight) and shared leaf
 * edges are exactly C0.
 */
MultipleStepSurfaceReductionResult DegreeReduceMultipleStepsConforming(
    const SurfaceData &initial_surface,
    double max_error,
    const ConformingReductionOptions &options = {});

/// Reduce one (merged) block to the target degrees WITHOUT splitting; returns
/// the reduced surface and its accumulated error (used by coalescing + tests).
/// When `weights_ok` is non-null it is set to false if any pass of any step
/// drove a reduced weight below the positivity floor: such a block is not an
/// acceptable merge even when its error fits the budget (its control-net AABB
/// and Eq. 5.30 accounting are void).
std::pair<SurfaceData, double> ReduceBlockNoSplit(const SurfaceData &block,
                                                  int target_degree_u,
                                                  int target_degree_v,
                                                  SurfaceErrorCombination error_combination,
                                                  bool rational_tol_correction,
                                                  bool *weights_ok = nullptr);

} // namespace mfem_raytracing

#endif
