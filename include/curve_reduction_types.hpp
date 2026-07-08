#ifndef CURVE_REDUCTION_TYPES_HPP
#define CURVE_REDUCTION_TYPES_HPP

#include <utility>
#include <vector>

namespace mfem_raytracing
{

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

struct ReducedCurveSegment
{
    CurveData curve;
    double segment_error = 0.0;
    std::pair<double, double> u_domain = {0.0, 1.0};
};

struct SingleStepReductionResult
{
    std::vector<ReducedCurveSegment> segments;
};

struct PeakErrorSingleStepOptions
{
    int max_depth = 25;
    double min_span_width = 1e-8;
};

} // namespace mfem_raytracing

#endif
