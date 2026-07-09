#ifndef CURVE_REDUCTION_TYPES_HPP
#define CURVE_REDUCTION_TYPES_HPP

// Shared curve geometry and single-step reduction result types.

#include <utility>
#include <vector>

namespace mfem_raytracing
{

/** B-spline / NURBS curve in raw control-point form (weights empty => polynomial). */
struct CurveData
{
    int degree = 0;
    int dim = 0;
    std::vector<std::vector<double>> control_points;
    std::vector<double> weights;
    std::vector<double> knotvector;
    std::pair<double, double> domain = {0.0, 1.0};

    bool IsRational() const;
    int NumControlPoints() const;
};

/** One accepted p -> p-1 piece after adaptive splitting. */
struct ReducedCurveSegment
{
    CurveData curve;
    double segment_error = 0.0;
    std::pair<double, double> u_domain = {0.0, 1.0}; // global root coordinates
};

/** Output of a single-step adaptive reduction (one or more segments). */
struct SingleStepReductionResult
{
    std::vector<ReducedCurveSegment> segments;
};

/** Recursion limits for peak-error splitting (approach 1). */
struct PeakErrorSingleStepOptions
{
    int max_depth = 25;
    double min_span_width = 1e-8;
};

} // namespace mfem_raytracing

#endif
