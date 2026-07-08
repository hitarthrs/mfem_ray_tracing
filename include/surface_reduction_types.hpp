#ifndef SURFACE_REDUCTION_TYPES_HPP
#define SURFACE_REDUCTION_TYPES_HPP

#include <utility>
#include <vector>

namespace mfem_raytracing
{

struct SurfaceData
{
    int degree_u = 0;
    int degree_v = 0;
    int dim = 0;
    std::vector<std::vector<std::vector<double>>> control_points;
    std::vector<std::vector<double>> weights;
    std::vector<double> knotvector_u;
    std::vector<double> knotvector_v;
    std::pair<double, double> u_domain = {0.0, 1.0};
    std::pair<double, double> v_domain = {0.0, 1.0};

    bool IsRational() const;
    int NumControlPointsU() const;
    int NumControlPointsV() const;
};

struct ReducedSurfaceSegment
{
    SurfaceData surface;
    double segment_error = 0.0;
    std::pair<double, double> u_domain = {0.0, 1.0};
    std::pair<double, double> v_domain = {0.0, 1.0};
};

struct SingleStepSurfaceReductionResult
{
    std::vector<ReducedSurfaceSegment> segments;
};

struct PeakErrorSurfaceSingleStepOptions
{
    int max_depth = 25;
    double min_span_width = 1e-8;
};

} // namespace mfem_raytracing

#endif
