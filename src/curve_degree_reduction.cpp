/**
 * Adaptive single-step curve reduction (Python approach_1).
 *
 * Probes one-step A5.11, splits at the midpoint of the worst knot span when
 * peak error exceeds the budget, then recurses on left/right subcurves.
 */

#include "curve_degree_reduction.hpp"

#include "curve_reduction_domain.hpp"

#include "b_spline_curve_reduction.hpp"
#include "nurbs_degree_reduction.hpp"

#include <algorithm>
#include <future>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>

namespace mfem_raytracing
{
namespace
{

constexpr double kParamTol = 1e-9;

struct ReductionOutput
{
    CurveData curve;
    std::vector<double> error_array;
    double segment_error = 0.0;
};

struct ParallelConfig
{
    unsigned int max_async_depth = 0;
};

double SegmentError(const std::vector<double> &error_array)
{
    double peak = 0.0;
    for (double err : error_array)
    {
        peak = std::max(peak, err);
    }
    return peak;
}

// Dispatch to polynomial or NURBS one-step kernel (p -> p-1).
ReductionOutput ReduceCurveOneStep(const CurveData &curve, double tol)
{
    ValidateCurveData(curve, "ReduceCurveOneStep");
    if (curve.degree < 2)
    {
        throw std::invalid_argument("ReduceCurveOneStep: degree must be >= 2");
    }

    ReductionOutput output;

    if (curve.IsRational())
    {
        std::vector<std::vector<double>> reduced_cp;
        std::vector<double> reduced_weights;
        std::vector<double> reduced_knots;
        const bool ok = DegreeReduceNURBCurve(
            curve.NumControlPoints(),
            curve.degree,
            curve.knotvector,
            curve.control_points,
            curve.weights,
            reduced_cp,
            reduced_weights,
            reduced_knots,
            output.error_array,
            tol);
        if (!ok)
        {
            throw std::runtime_error("ReduceCurveOneStep: NURBS reduction exceeded tolerance");
        }

        output.curve.degree = curve.degree - 1;
        output.curve.dim = curve.dim;
        output.curve.control_points = reduced_cp;
        output.curve.weights = reduced_weights;
        output.curve.knotvector = reduced_knots;
        output.curve.domain = CurveDomain(reduced_knots, output.curve.degree);
    }
    else
    {
        std::vector<std::vector<double>> reduced_cp;
        std::vector<double> reduced_knots;
        const bool ok = DegreeReduceCurve(
            curve.NumControlPoints(),
            curve.degree,
            curve.knotvector,
            curve.control_points,
            curve.dim,
            reduced_cp,
            reduced_knots,
            output.error_array,
            tol);
        if (!ok)
        {
            throw std::runtime_error(
                "ReduceCurveOneStep: polynomial reduction exceeded tolerance");
        }

        output.curve.degree = curve.degree - 1;
        output.curve.dim = curve.dim;
        output.curve.control_points = reduced_cp;
        output.curve.knotvector = reduced_knots;
        output.curve.domain = CurveDomain(reduced_knots, output.curve.degree);
    }

    output.segment_error = SegmentError(output.error_array);
    return output;
}

// Split site: midpoint of the knot span containing the largest A5.11 error entry.
std::optional<std::tuple<int, double, double, double>>
MidpointKnotForMaxError(const std::vector<double> &knotvector,
                        int degree,
                        const std::vector<double> &error_array)
{
    (void)degree;
    if (error_array.empty())
    {
        return std::nullopt;
    }

    const auto max_iter = std::max_element(error_array.begin(), error_array.end());
    if (*max_iter <= 0.0)
    {
        return std::nullopt;
    }

    const int m = static_cast<int>(std::distance(error_array.begin(), max_iter));
    if (m < 0 || m >= static_cast<int>(knotvector.size()))
    {
        return std::nullopt;
    }

    const double u_at = knotvector[static_cast<std::size_t>(m)];
    for (std::size_t i = static_cast<std::size_t>(m + 1); i < knotvector.size(); ++i)
    {
        const double u_next = knotvector[i];
        if (u_next > u_at + kParamTol)
        {
            return std::make_tuple(m, u_at, u_next, 0.5 * (u_at + u_next));
        }
    }
    return std::nullopt;
}

// Recursive core; piece_global tracks this subcurve in root parameter coordinates.
void PeakErrorSingleStepImpl(const CurveData &curve,
                             const std::pair<double, double> &piece_global,
                             double max_error,
                             const PeakErrorSingleStepOptions &options,
                             int depth,
                             std::vector<ReducedCurveSegment> &segments)
{
    if (curve.degree < 2)
    {
        throw std::invalid_argument("PeakErrorSingleStepImpl: degree must be >= 2");
    }

    const double span_width = curve.domain.second - curve.domain.first;
    if (span_width < options.min_span_width)
    {
        throw std::runtime_error(
            "PeakErrorSingleStepImpl: span narrower than min_span_width");
    }

    const ReductionOutput probe =
        ReduceCurveOneStep(curve, std::numeric_limits<double>::infinity());
    const double peak_error = SegmentError(probe.error_array);

    const auto accept_piece = [&](double tol) {
        const ReductionOutput accepted = ReduceCurveOneStep(curve, tol);
        segments.push_back({accepted.curve, accepted.segment_error, piece_global});
    };

    const auto force_accept = [&]() {
        segments.push_back({probe.curve, probe.segment_error, piece_global});
    };

    if (peak_error <= max_error)
    {
        accept_piece(max_error);
        return;
    }

    const auto site =
        MidpointKnotForMaxError(curve.knotvector, curve.degree, probe.error_array);
    if (!site.has_value())
    {
        force_accept();
        return;
    }

    const double u_split_local = std::get<3>(*site);
    if (u_split_local <= curve.domain.first + kParamTol ||
        u_split_local >= curve.domain.second - kParamTol)
    {
        force_accept();
        return;
    }

    const double u_split_global =
        GlobalParamFromLocal(u_split_local, piece_global, curve.domain);
    if ((u_split_global - piece_global.first) < options.min_span_width ||
        (piece_global.second - u_split_global) < options.min_span_width)
    {
        force_accept();
        return;
    }
    if (depth >= options.max_depth)
    {
        force_accept();
        return;
    }

    const std::pair<double, double> left_global = {piece_global.first, u_split_global};
    const std::pair<double, double> right_global = {u_split_global, piece_global.second};

    const CurveData left = ExtractSubcurveGlobal(
        curve, piece_global, left_global.first, left_global.second);
    const CurveData right = ExtractSubcurveGlobal(
        curve, piece_global, right_global.first, right_global.second);

    PeakErrorSingleStepImpl(left, left_global, max_error, options, depth + 1, segments);
    PeakErrorSingleStepImpl(right, right_global, max_error, options, depth + 1, segments);
}

ParallelConfig MakeParallelConfig()
{
    ParallelConfig config;
    const unsigned int hw = std::thread::hardware_concurrency();
    if (hw <= 1)
    {
        config.max_async_depth = 0;
        return config;
    }

    unsigned int depth = 0;
    unsigned int tasks = 1;
    while (tasks < hw)
    {
        tasks <<= 1U;
        ++depth;
    }
    config.max_async_depth = depth;
    return config;
}

std::vector<ReducedCurveSegment> PeakErrorSingleStepOptimizedImpl(
    const CurveData &curve,
    const std::pair<double, double> &piece_global,
    double max_error,
    const PeakErrorSingleStepOptions &options,
    int depth,
    const ParallelConfig &parallel)
{
    if (curve.degree < 2)
    {
        throw std::invalid_argument("PeakErrorSingleStepOptimizedImpl: degree must be >= 2");
    }

    const double span_width = curve.domain.second - curve.domain.first;
    if (span_width < options.min_span_width)
    {
        throw std::runtime_error(
            "PeakErrorSingleStepOptimizedImpl: span narrower than min_span_width");
    }

    const ReductionOutput probe =
        ReduceCurveOneStep(curve, std::numeric_limits<double>::infinity());
    const double peak_error = SegmentError(probe.error_array);

    // If the probe already satisfies the requested tolerance, reuse it directly.
    if (peak_error <= max_error)
    {
        return {{probe.curve, probe.segment_error, piece_global}};
    }

    const auto site =
        MidpointKnotForMaxError(curve.knotvector, curve.degree, probe.error_array);
    if (!site.has_value())
    {
        return {{probe.curve, probe.segment_error, piece_global}};
    }

    const double u_split_local = std::get<3>(*site);
    if (u_split_local <= curve.domain.first + kParamTol ||
        u_split_local >= curve.domain.second - kParamTol)
    {
        return {{probe.curve, probe.segment_error, piece_global}};
    }

    const double u_split_global =
        GlobalParamFromLocal(u_split_local, piece_global, curve.domain);
    if ((u_split_global - piece_global.first) < options.min_span_width ||
        (piece_global.second - u_split_global) < options.min_span_width ||
        depth >= options.max_depth)
    {
        return {{probe.curve, probe.segment_error, piece_global}};
    }

    const std::pair<double, double> left_global = {piece_global.first, u_split_global};
    const std::pair<double, double> right_global = {u_split_global, piece_global.second};
    const CurveData left = ExtractSubcurveGlobal(
        curve, piece_global, left_global.first, left_global.second);
    const CurveData right = ExtractSubcurveGlobal(
        curve, piece_global, right_global.first, right_global.second);

    std::vector<ReducedCurveSegment> left_segments;
    std::vector<ReducedCurveSegment> right_segments;

    const bool use_async =
        parallel.max_async_depth > 0 &&
        static_cast<unsigned int>(depth) < parallel.max_async_depth &&
        span_width > 4.0 * options.min_span_width;
    if (use_async)
    {
        auto left_future = std::async(std::launch::async,
                                      [&]() {
                                          return PeakErrorSingleStepOptimizedImpl(
                                              left,
                                              left_global,
                                              max_error,
                                              options,
                                              depth + 1,
                                              parallel);
                                      });
        right_segments = PeakErrorSingleStepOptimizedImpl(
            right, right_global, max_error, options, depth + 1, parallel);
        left_segments = left_future.get();
    }
    else
    {
        left_segments = PeakErrorSingleStepOptimizedImpl(
            left, left_global, max_error, options, depth + 1, parallel);
        right_segments = PeakErrorSingleStepOptimizedImpl(
            right, right_global, max_error, options, depth + 1, parallel);
    }

    left_segments.insert(left_segments.end(), right_segments.begin(), right_segments.end());
    return left_segments;
}

} // namespace

SingleStepReductionResult PeakErrorSingleStep(const CurveData &initial_curve,
                                              double max_error,
                                              const PeakErrorSingleStepOptions &options)
{
    if (max_error < 0.0)
    {
        throw std::invalid_argument("PeakErrorSingleStep: max_error must be non-negative");
    }

    ValidateCurveData(initial_curve, "PeakErrorSingleStep");

    std::vector<ReducedCurveSegment> segments;
    PeakErrorSingleStepImpl(
        initial_curve, initial_curve.domain, max_error, options, 0, segments);
    std::sort(segments.begin(),
              segments.end(),
              [](const ReducedCurveSegment &lhs, const ReducedCurveSegment &rhs) {
                  return lhs.u_domain.first < rhs.u_domain.first;
              });
    return {segments};
}

SingleStepReductionResult PeakErrorSingleStepOptimized(
    const CurveData &initial_curve,
    double max_error,
    const PeakErrorSingleStepOptions &options)
{
    if (max_error < 0.0)
    {
        throw std::invalid_argument(
            "PeakErrorSingleStepOptimized: max_error must be non-negative");
    }

    ValidateCurveData(initial_curve, "PeakErrorSingleStepOptimized");

    const ParallelConfig parallel = MakeParallelConfig();
    std::vector<ReducedCurveSegment> segments = PeakErrorSingleStepOptimizedImpl(
        initial_curve, initial_curve.domain, max_error, options, 0, parallel);
    std::sort(segments.begin(),
              segments.end(),
              [](const ReducedCurveSegment &lhs, const ReducedCurveSegment &rhs) {
                  return lhs.u_domain.first < rhs.u_domain.first;
              });
    return {segments};
}

} // namespace mfem_raytracing
