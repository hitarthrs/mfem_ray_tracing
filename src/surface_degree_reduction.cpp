#include "surface_degree_reduction.hpp"

#include "curve_reduction_domain.hpp"
#include "curve_reduction_types.hpp"
#include "nurbs_degree_reduction.hpp"
#include "surface_reduction_domain.hpp"

#include "b_spline_curve_reduction.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace mfem_raytracing
{
namespace
{

constexpr double kParamTol = 1e-9;

struct UPassState
{
    SurfaceData surface;
    double u_pass_error = 0.0;
    std::pair<double, double> u_domain = {0.0, 1.0};
    std::pair<double, double> v_domain = {0.0, 1.0};
};

struct ParallelConfig
{
    unsigned int max_async_depth = 0;
    bool parallel_passes = false;
};

struct ReducedCurveDataWithError
{
    CurveData curve;
    std::vector<double> error_array;
};

double MaxError(const std::vector<double> &error_array)
{
    double peak = 0.0;
    for (double value : error_array)
    {
        peak = std::max(peak, value);
    }
    return peak;
}

std::optional<std::tuple<int, double, double, double>>
MidpointKnotForMaxError(const std::vector<double> &knotvector,
                        const std::vector<double> &error_array)
{
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

std::vector<double> ProbeIsoUColumn(const SurfaceData &surface, int column_index, double tol)
{
    CurveData curve;
    curve.degree = surface.degree_u;
    curve.dim = surface.dim;
    curve.knotvector = surface.knotvector_u;
    curve.domain = surface.u_domain;
    curve.control_points.resize(static_cast<std::size_t>(surface.NumControlPointsU()));
    for (int i = 0; i < surface.NumControlPointsU(); ++i)
    {
        curve.control_points[static_cast<std::size_t>(i)] =
            surface.control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(column_index)];
    }

    std::vector<double> error_array;
    if (surface.IsRational())
    {
        curve.weights.resize(static_cast<std::size_t>(surface.NumControlPointsU()));
        for (int i = 0; i < surface.NumControlPointsU(); ++i)
        {
            curve.weights[static_cast<std::size_t>(i)] =
                surface.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(column_index)];
        }

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
            error_array,
            tol);
        if (!ok)
        {
            throw std::runtime_error("ProbeIsoUColumn: NURBS reduction exceeded tolerance");
        }
        return error_array;
    }

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
        error_array,
        tol);
    if (!ok)
    {
        throw std::runtime_error("ProbeIsoUColumn: polynomial reduction exceeded tolerance");
    }
    return error_array;
}

std::vector<double> ProbeIsoVRow(const SurfaceData &surface, int row_index, double tol)
{
    CurveData curve;
    curve.degree = surface.degree_v;
    curve.dim = surface.dim;
    curve.knotvector = surface.knotvector_v;
    curve.domain = surface.v_domain;
    curve.control_points = surface.control_points[static_cast<std::size_t>(row_index)];

    std::vector<double> error_array;
    if (surface.IsRational())
    {
        curve.weights = surface.weights[static_cast<std::size_t>(row_index)];

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
            error_array,
            tol);
        if (!ok)
        {
            throw std::runtime_error("ProbeIsoVRow: NURBS reduction exceeded tolerance");
        }
        return error_array;
    }

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
        error_array,
        tol);
    if (!ok)
    {
        throw std::runtime_error("ProbeIsoVRow: polynomial reduction exceeded tolerance");
    }
    return error_array;
}

std::vector<double> AggregateUPassProbeErrors(const SurfaceData &surface)
{
    std::vector<double> error_u(surface.knotvector_u.size(), 0.0);
    for (int j = 0; j < surface.NumControlPointsV(); ++j)
    {
        const std::vector<double> err = ProbeIsoUColumn(surface, j, std::numeric_limits<double>::infinity());
        const std::size_t n_err = std::min(error_u.size(), err.size());
        for (std::size_t i = 0; i < n_err; ++i)
        {
            error_u[i] = std::max(error_u[i], err[i]);
        }
    }
    return error_u;
}

std::vector<double> AggregateVPassProbeErrors(const SurfaceData &surface)
{
    std::vector<double> error_v(surface.knotvector_v.size(), 0.0);
    for (int i = 0; i < surface.NumControlPointsU(); ++i)
    {
        const std::vector<double> err = ProbeIsoVRow(surface, i, std::numeric_limits<double>::infinity());
        const std::size_t n_err = std::min(error_v.size(), err.size());
        for (std::size_t k = 0; k < n_err; ++k)
        {
            error_v[k] = std::max(error_v[k], err[k]);
        }
    }
    return error_v;
}

ReducedCurveDataWithError ReduceColumnCurve(const SurfaceData &surface, int column_index, double tol)
{
    CurveData curve;
    curve.degree = surface.degree_u;
    curve.dim = surface.dim;
    curve.knotvector = surface.knotvector_u;
    curve.domain = surface.u_domain;
    curve.control_points.resize(static_cast<std::size_t>(surface.NumControlPointsU()));
    for (int i = 0; i < surface.NumControlPointsU(); ++i)
    {
        curve.control_points[static_cast<std::size_t>(i)] =
            surface.control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(column_index)];
    }

    std::vector<double> error_array;
    CurveData reduced_curve;
    if (surface.IsRational())
    {
        curve.weights.resize(static_cast<std::size_t>(surface.NumControlPointsU()));
        for (int i = 0; i < surface.NumControlPointsU(); ++i)
        {
            curve.weights[static_cast<std::size_t>(i)] =
                surface.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(column_index)];
        }

        std::vector<std::vector<double>> reduced_cp;
        std::vector<double> reduced_weights;
        std::vector<double> reduced_knots;
        const bool ok = DegreeReduceNURBCurve(curve.NumControlPoints(),
                                              curve.degree,
                                              curve.knotvector,
                                              curve.control_points,
                                              curve.weights,
                                              reduced_cp,
                                              reduced_weights,
                                              reduced_knots,
                                              error_array,
                                              tol);
        if (!ok)
        {
            throw std::runtime_error("ReduceColumnCurve: NURBS column exceeded tolerance");
        }
        reduced_curve.degree = surface.degree_u - 1;
        reduced_curve.dim = surface.dim;
        reduced_curve.control_points = reduced_cp;
        reduced_curve.weights = reduced_weights;
        reduced_curve.knotvector = reduced_knots;
        reduced_curve.domain = CurveDomain(reduced_knots, reduced_curve.degree);
    }
    else
    {
        std::vector<std::vector<double>> reduced_cp;
        std::vector<double> reduced_knots;
        const bool ok = DegreeReduceCurve(curve.NumControlPoints(),
                                          curve.degree,
                                          curve.knotvector,
                                          curve.control_points,
                                          curve.dim,
                                          reduced_cp,
                                          reduced_knots,
                                          error_array,
                                          tol);
        if (!ok)
        {
            throw std::runtime_error("ReduceColumnCurve: polynomial column exceeded tolerance");
        }
        reduced_curve.degree = surface.degree_u - 1;
        reduced_curve.dim = surface.dim;
        reduced_curve.control_points = reduced_cp;
        reduced_curve.knotvector = reduced_knots;
        reduced_curve.domain = CurveDomain(reduced_knots, reduced_curve.degree);
    }

    return {reduced_curve, error_array};
}

ReducedCurveDataWithError ReduceRowCurve(const SurfaceData &surface, int row_index, double tol)
{
    CurveData curve;
    curve.degree = surface.degree_v;
    curve.dim = surface.dim;
    curve.knotvector = surface.knotvector_v;
    curve.domain = surface.v_domain;
    curve.control_points = surface.control_points[static_cast<std::size_t>(row_index)];

    std::vector<double> error_array;
    CurveData reduced_curve;
    if (surface.IsRational())
    {
        curve.weights = surface.weights[static_cast<std::size_t>(row_index)];

        std::vector<std::vector<double>> reduced_cp;
        std::vector<double> reduced_weights;
        std::vector<double> reduced_knots;
        const bool ok = DegreeReduceNURBCurve(curve.NumControlPoints(),
                                              curve.degree,
                                              curve.knotvector,
                                              curve.control_points,
                                              curve.weights,
                                              reduced_cp,
                                              reduced_weights,
                                              reduced_knots,
                                              error_array,
                                              tol);
        if (!ok)
        {
            throw std::runtime_error("ReduceRowCurve: NURBS row exceeded tolerance");
        }
        reduced_curve.degree = surface.degree_v - 1;
        reduced_curve.dim = surface.dim;
        reduced_curve.control_points = reduced_cp;
        reduced_curve.weights = reduced_weights;
        reduced_curve.knotvector = reduced_knots;
        reduced_curve.domain = CurveDomain(reduced_knots, reduced_curve.degree);
    }
    else
    {
        std::vector<std::vector<double>> reduced_cp;
        std::vector<double> reduced_knots;
        const bool ok = DegreeReduceCurve(curve.NumControlPoints(),
                                          curve.degree,
                                          curve.knotvector,
                                          curve.control_points,
                                          curve.dim,
                                          reduced_cp,
                                          reduced_knots,
                                          error_array,
                                          tol);
        if (!ok)
        {
            throw std::runtime_error("ReduceRowCurve: polynomial row exceeded tolerance");
        }
        reduced_curve.degree = surface.degree_v - 1;
        reduced_curve.dim = surface.dim;
        reduced_curve.control_points = reduced_cp;
        reduced_curve.knotvector = reduced_knots;
        reduced_curve.domain = CurveDomain(reduced_knots, reduced_curve.degree);
    }

    return {reduced_curve, error_array};
}

ParallelConfig MakeParallelConfig()
{
    ParallelConfig config;
    const unsigned int hw = std::thread::hardware_concurrency();
    if (hw <= 1)
    {
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
    config.parallel_passes = hw > 1;
    return config;
}

UPassState ReduceUPassOnly(const SurfaceData &surface,
                          const std::pair<double, double> &u_domain,
                          const std::pair<double, double> &v_domain,
                          double tol)
{
    SurfaceData reduced;
    reduced.degree_u = surface.degree_u - 1;
    reduced.degree_v = surface.degree_v;
    reduced.dim = surface.dim;
    reduced.knotvector_v = surface.knotvector_v;
    reduced.v_domain = surface.v_domain;
    reduced.control_points.clear();

    std::vector<double> error_u(surface.knotvector_u.size(), 0.0);
    std::vector<std::vector<CurveData>> reduced_columns;
    reduced_columns.reserve(static_cast<std::size_t>(surface.NumControlPointsV()));

    CurveData reference;
    bool have_reference = false;

    for (int j = 0; j < surface.NumControlPointsV(); ++j)
    {
        CurveData curve;
        curve.degree = surface.degree_u;
        curve.dim = surface.dim;
        curve.knotvector = surface.knotvector_u;
        curve.domain = surface.u_domain;
        curve.control_points.resize(static_cast<std::size_t>(surface.NumControlPointsU()));
        for (int i = 0; i < surface.NumControlPointsU(); ++i)
        {
            curve.control_points[static_cast<std::size_t>(i)] =
                surface.control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }

        std::vector<double> err_j;
        CurveData reduced_curve;
        if (surface.IsRational())
        {
            curve.weights.resize(static_cast<std::size_t>(surface.NumControlPointsU()));
            for (int i = 0; i < surface.NumControlPointsU(); ++i)
            {
                curve.weights[static_cast<std::size_t>(i)] =
                    surface.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            }

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
                err_j,
                tol);
            if (!ok)
            {
                throw std::runtime_error("ReduceUPassOnly: NURBS column exceeded tolerance");
            }
            reduced_curve.degree = surface.degree_u - 1;
            reduced_curve.dim = surface.dim;
            reduced_curve.control_points = reduced_cp;
            reduced_curve.weights = reduced_weights;
            reduced_curve.knotvector = reduced_knots;
            reduced_curve.domain = CurveDomain(reduced_knots, reduced_curve.degree);
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
                err_j,
                tol);
            if (!ok)
            {
                throw std::runtime_error("ReduceUPassOnly: polynomial column exceeded tolerance");
            }
            reduced_curve.degree = surface.degree_u - 1;
            reduced_curve.dim = surface.dim;
            reduced_curve.control_points = reduced_cp;
            reduced_curve.knotvector = reduced_knots;
            reduced_curve.domain = CurveDomain(reduced_knots, reduced_curve.degree);
        }

        if (!have_reference)
        {
            reference = reduced_curve;
            have_reference = true;
        }
        else if (reference.knotvector != reduced_curve.knotvector)
        {
            throw std::runtime_error("ReduceUPassOnly: Uh mismatch across columns");
        }

        const std::size_t n_err = std::min(error_u.size(), err_j.size());
        for (std::size_t i = 0; i < n_err; ++i)
        {
            error_u[i] = std::max(error_u[i], err_j[i]);
        }
        reduced_columns.push_back({reduced_curve});
    }

    reduced.knotvector_u = reference.knotvector;
    reduced.u_domain = reference.domain;
    reduced.control_points.assign(static_cast<std::size_t>(reference.NumControlPoints()),
                                  std::vector<std::vector<double>>(
                                      static_cast<std::size_t>(surface.NumControlPointsV())));
    if (surface.IsRational())
    {
        reduced.weights.assign(static_cast<std::size_t>(reference.NumControlPoints()),
                               std::vector<double>(
                                   static_cast<std::size_t>(surface.NumControlPointsV()), 1.0));
    }

    for (int j = 0; j < surface.NumControlPointsV(); ++j)
    {
        const CurveData &reduced_curve = reduced_columns[static_cast<std::size_t>(j)][0];
        for (int i = 0; i < reference.NumControlPoints(); ++i)
        {
            reduced.control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                reduced_curve.control_points[static_cast<std::size_t>(i)];
            if (surface.IsRational())
            {
                reduced.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    reduced_curve.weights[static_cast<std::size_t>(i)];
            }
        }
    }

    return {reduced, MaxError(error_u), u_domain, v_domain};
}

UPassState ReduceUPassOnlyOptimized(const SurfaceData &surface,
                                   const std::pair<double, double> &u_domain,
                                   const std::pair<double, double> &v_domain,
                                   double tol,
                                   const ParallelConfig &parallel)
{
    SurfaceData reduced;
    reduced.degree_u = surface.degree_u - 1;
    reduced.degree_v = surface.degree_v;
    reduced.dim = surface.dim;
    reduced.knotvector_v = surface.knotvector_v;
    reduced.v_domain = surface.v_domain;

    const int n_cols = surface.NumControlPointsV();
    std::vector<ReducedCurveDataWithError> reduced_columns(static_cast<std::size_t>(n_cols));

    auto process_range = [&](int begin, int end) {
        for (int j = begin; j < end; ++j)
        {
            reduced_columns[static_cast<std::size_t>(j)] = ReduceColumnCurve(surface, j, tol);
        }
    };

    if (parallel.parallel_passes && n_cols >= 4)
    {
        const int mid = n_cols / 2;
        auto future = std::async(std::launch::async, [&]() { process_range(0, mid); });
        process_range(mid, n_cols);
        future.get();
    }
    else
    {
        process_range(0, n_cols);
    }

    CurveData reference;
    bool have_reference = false;
    std::vector<double> error_u(surface.knotvector_u.size(), 0.0);
    for (int j = 0; j < n_cols; ++j)
    {
        const auto &result = reduced_columns[static_cast<std::size_t>(j)];
        if (!have_reference)
        {
            reference = result.curve;
            have_reference = true;
        }
        else if (reference.knotvector != result.curve.knotvector)
        {
            throw std::runtime_error("ReduceUPassOnlyOptimized: Uh mismatch across columns");
        }
        const std::size_t n_err = std::min(error_u.size(), result.error_array.size());
        for (std::size_t i = 0; i < n_err; ++i)
        {
            error_u[i] = std::max(error_u[i], result.error_array[i]);
        }
    }

    reduced.knotvector_u = reference.knotvector;
    reduced.u_domain = reference.domain;
    reduced.control_points.assign(static_cast<std::size_t>(reference.NumControlPoints()),
                                  std::vector<std::vector<double>>(
                                      static_cast<std::size_t>(surface.NumControlPointsV())));
    if (surface.IsRational())
    {
        reduced.weights.assign(static_cast<std::size_t>(reference.NumControlPoints()),
                               std::vector<double>(
                                   static_cast<std::size_t>(surface.NumControlPointsV()), 1.0));
    }

    for (int j = 0; j < n_cols; ++j)
    {
        const CurveData &reduced_curve = reduced_columns[static_cast<std::size_t>(j)].curve;
        for (int i = 0; i < reference.NumControlPoints(); ++i)
        {
            reduced.control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                reduced_curve.control_points[static_cast<std::size_t>(i)];
            if (surface.IsRational())
            {
                reduced.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    reduced_curve.weights[static_cast<std::size_t>(i)];
            }
        }
    }

    return {reduced, MaxError(error_u), u_domain, v_domain};
}

ReducedSurfaceSegment ReduceVPassOnly(const UPassState &u_state, double tol)
{
    const SurfaceData &surface = u_state.surface;
    SurfaceData reduced;
    reduced.degree_u = surface.degree_u;
    reduced.degree_v = surface.degree_v - 1;
    reduced.dim = surface.dim;
    reduced.knotvector_u = surface.knotvector_u;
    reduced.u_domain = surface.u_domain;

    std::vector<double> error_v(surface.knotvector_v.size(), 0.0);
    std::vector<CurveData> reduced_rows;
    reduced_rows.reserve(static_cast<std::size_t>(surface.NumControlPointsU()));

    CurveData reference;
    bool have_reference = false;

    for (int i = 0; i < surface.NumControlPointsU(); ++i)
    {
        CurveData curve;
        curve.degree = surface.degree_v;
        curve.dim = surface.dim;
        curve.knotvector = surface.knotvector_v;
        curve.domain = surface.v_domain;
        curve.control_points = surface.control_points[static_cast<std::size_t>(i)];

        std::vector<double> err_i;
        CurveData reduced_curve;
        if (surface.IsRational())
        {
            curve.weights = surface.weights[static_cast<std::size_t>(i)];

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
                err_i,
                tol);
            if (!ok)
            {
                throw std::runtime_error("ReduceVPassOnly: NURBS row exceeded tolerance");
            }
            reduced_curve.degree = surface.degree_v - 1;
            reduced_curve.dim = surface.dim;
            reduced_curve.control_points = reduced_cp;
            reduced_curve.weights = reduced_weights;
            reduced_curve.knotvector = reduced_knots;
            reduced_curve.domain = CurveDomain(reduced_knots, reduced_curve.degree);
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
                err_i,
                tol);
            if (!ok)
            {
                throw std::runtime_error("ReduceVPassOnly: polynomial row exceeded tolerance");
            }
            reduced_curve.degree = surface.degree_v - 1;
            reduced_curve.dim = surface.dim;
            reduced_curve.control_points = reduced_cp;
            reduced_curve.knotvector = reduced_knots;
            reduced_curve.domain = CurveDomain(reduced_knots, reduced_curve.degree);
        }

        if (!have_reference)
        {
            reference = reduced_curve;
            have_reference = true;
        }
        else if (reference.knotvector != reduced_curve.knotvector)
        {
            throw std::runtime_error("ReduceVPassOnly: Vh mismatch across rows");
        }

        const std::size_t n_err = std::min(error_v.size(), err_i.size());
        for (std::size_t k = 0; k < n_err; ++k)
        {
            error_v[k] = std::max(error_v[k], err_i[k]);
        }
        reduced_rows.push_back(reduced_curve);
    }

    reduced.knotvector_v = reference.knotvector;
    reduced.v_domain = reference.domain;
    reduced.control_points.resize(static_cast<std::size_t>(surface.NumControlPointsU()));
    if (surface.IsRational())
    {
        reduced.weights.resize(static_cast<std::size_t>(surface.NumControlPointsU()));
    }

    for (int i = 0; i < surface.NumControlPointsU(); ++i)
    {
        reduced.control_points[static_cast<std::size_t>(i)] =
            reduced_rows[static_cast<std::size_t>(i)].control_points;
        if (surface.IsRational())
        {
            reduced.weights[static_cast<std::size_t>(i)] =
                reduced_rows[static_cast<std::size_t>(i)].weights;
        }
    }

    return {reduced,
            std::max(u_state.u_pass_error, MaxError(error_v)),
            u_state.u_domain,
            u_state.v_domain};
}

ReducedSurfaceSegment ReduceVPassOnlyOptimized(const UPassState &u_state,
                                               double tol,
                                               const ParallelConfig &parallel)
{
    const SurfaceData &surface = u_state.surface;
    SurfaceData reduced;
    reduced.degree_u = surface.degree_u;
    reduced.degree_v = surface.degree_v - 1;
    reduced.dim = surface.dim;
    reduced.knotvector_u = surface.knotvector_u;
    reduced.u_domain = surface.u_domain;

    const int n_rows = surface.NumControlPointsU();
    std::vector<ReducedCurveDataWithError> reduced_rows(static_cast<std::size_t>(n_rows));

    auto process_range = [&](int begin, int end) {
        for (int i = begin; i < end; ++i)
        {
            reduced_rows[static_cast<std::size_t>(i)] = ReduceRowCurve(surface, i, tol);
        }
    };

    if (parallel.parallel_passes && n_rows >= 4)
    {
        const int mid = n_rows / 2;
        auto future = std::async(std::launch::async, [&]() { process_range(0, mid); });
        process_range(mid, n_rows);
        future.get();
    }
    else
    {
        process_range(0, n_rows);
    }

    CurveData reference;
    bool have_reference = false;
    std::vector<double> error_v(surface.knotvector_v.size(), 0.0);
    for (int i = 0; i < n_rows; ++i)
    {
        const auto &result = reduced_rows[static_cast<std::size_t>(i)];
        if (!have_reference)
        {
            reference = result.curve;
            have_reference = true;
        }
        else if (reference.knotvector != result.curve.knotvector)
        {
            throw std::runtime_error("ReduceVPassOnlyOptimized: Vh mismatch across rows");
        }
        const std::size_t n_err = std::min(error_v.size(), result.error_array.size());
        for (std::size_t k = 0; k < n_err; ++k)
        {
            error_v[k] = std::max(error_v[k], result.error_array[k]);
        }
    }

    reduced.knotvector_v = reference.knotvector;
    reduced.v_domain = reference.domain;
    reduced.control_points.resize(static_cast<std::size_t>(surface.NumControlPointsU()));
    if (surface.IsRational())
    {
        reduced.weights.resize(static_cast<std::size_t>(surface.NumControlPointsU()));
    }

    for (int i = 0; i < n_rows; ++i)
    {
        reduced.control_points[static_cast<std::size_t>(i)] =
            reduced_rows[static_cast<std::size_t>(i)].curve.control_points;
        if (surface.IsRational())
        {
            reduced.weights[static_cast<std::size_t>(i)] =
                reduced_rows[static_cast<std::size_t>(i)].curve.weights;
        }
    }

    return {reduced,
            std::max(u_state.u_pass_error, MaxError(error_v)),
            u_state.u_domain,
            u_state.v_domain};
}

std::vector<UPassState> UPassWithApproach1(const SurfaceData &surface,
                                           const std::pair<double, double> &u_domain_global,
                                           const std::pair<double, double> &v_domain_global,
                                           double max_error,
                                           const PeakErrorSurfaceSingleStepOptions &options,
                                           int depth)
{
    if (surface.degree_u < 2)
    {
        throw std::invalid_argument("UPassWithApproach1: degree_u must be >= 2");
    }

    const std::vector<double> err_probe = AggregateUPassProbeErrors(surface);
    const double peak_err = MaxError(err_probe);

    auto accept_u_pass = [&](double tol) {
        return ReduceUPassOnly(surface, u_domain_global, v_domain_global, tol);
    };

    auto force_accept = [&]() {
        return std::vector<UPassState>{accept_u_pass(std::numeric_limits<double>::infinity())};
    };

    if (peak_err <= max_error)
    {
        return {accept_u_pass(max_error)};
    }

    const auto site = MidpointKnotForMaxError(surface.knotvector_u, err_probe);
    if (!site.has_value())
    {
        return force_accept();
    }

    const double u_mid_local = std::get<3>(*site);
    const double u_split_global =
        GlobalParamFromLocal(u_mid_local, u_domain_global, surface.u_domain);
    const double u0 = u_domain_global.first;
    const double u1 = u_domain_global.second;
    const double eps = 1e-9 * std::max(1.0, std::abs(u1 - u0));
    if (u_split_global <= u0 + eps || u_split_global >= u1 - eps)
    {
        return force_accept();
    }
    if (u_split_global - u0 < options.min_span_width ||
        u1 - u_split_global < options.min_span_width ||
        depth >= options.max_depth)
    {
        return force_accept();
    }

    const double u_split_local =
        LocalSurfaceParamFromGlobal(u_split_global, u_domain_global, surface.u_domain);
    const auto split = SplitSurfaceU(surface, u_split_local);
    std::vector<UPassState> left_states = UPassWithApproach1(split.first,
                                                             {u0, u_split_global},
                                                             v_domain_global,
                                                             max_error,
                                                             options,
                                                             depth + 1);
    std::vector<UPassState> right_states = UPassWithApproach1(split.second,
                                                              {u_split_global, u1},
                                                              v_domain_global,
                                                              max_error,
                                                              options,
                                                              depth + 1);
    left_states.insert(left_states.end(), right_states.begin(), right_states.end());
    return left_states;
}

std::vector<ReducedSurfaceSegment> VPassWithApproach1(
    const UPassState &u_state,
    double max_error,
    const PeakErrorSurfaceSingleStepOptions &options,
    int depth)
{
    const SurfaceData &surface = u_state.surface;
    if (surface.degree_v < 2)
    {
        throw std::invalid_argument("VPassWithApproach1: degree_v must be >= 2");
    }

    const std::vector<double> err_probe = AggregateVPassProbeErrors(surface);
    const double peak_err = MaxError(err_probe);

    auto accept_full = [&](double tol) {
        return ReduceVPassOnly(u_state, tol);
    };

    auto force_accept = [&]() {
        return std::vector<ReducedSurfaceSegment>{accept_full(std::numeric_limits<double>::infinity())};
    };

    if (peak_err <= max_error)
    {
        return {accept_full(max_error)};
    }

    const auto site = MidpointKnotForMaxError(surface.knotvector_v, err_probe);
    if (!site.has_value())
    {
        return force_accept();
    }

    const double v_mid_local = std::get<3>(*site);
    const double v_split_global =
        GlobalParamFromLocal(v_mid_local, u_state.v_domain, surface.v_domain);
    const double v0 = u_state.v_domain.first;
    const double v1 = u_state.v_domain.second;
    const double eps = 1e-9 * std::max(1.0, std::abs(v1 - v0));
    if (v_split_global <= v0 + eps || v_split_global >= v1 - eps)
    {
        return force_accept();
    }
    if (v_split_global - v0 < options.min_span_width ||
        v1 - v_split_global < options.min_span_width ||
        depth >= options.max_depth)
    {
        return force_accept();
    }

    const double v_split_local =
        LocalSurfaceParamFromGlobal(v_split_global, u_state.v_domain, surface.v_domain);
    const auto split = SplitSurfaceV(surface, v_split_local);

    UPassState left_state = u_state;
    left_state.surface = split.first;
    left_state.v_domain = {v0, v_split_global};

    UPassState right_state = u_state;
    right_state.surface = split.second;
    right_state.v_domain = {v_split_global, v1};

    std::vector<ReducedSurfaceSegment> left_segments =
        VPassWithApproach1(left_state, max_error, options, depth + 1);
    std::vector<ReducedSurfaceSegment> right_segments =
        VPassWithApproach1(right_state, max_error, options, depth + 1);
    left_segments.insert(left_segments.end(), right_segments.begin(), right_segments.end());
    return left_segments;
}

std::vector<UPassState> UPassWithApproach1Optimized(const SurfaceData &surface,
                                                    const std::pair<double, double> &u_domain_global,
                                                    const std::pair<double, double> &v_domain_global,
                                                    double max_error,
                                                    const PeakErrorSurfaceSingleStepOptions &options,
                                                    int depth,
                                                    const ParallelConfig &parallel)
{
    if (surface.degree_u < 2)
    {
        throw std::invalid_argument("UPassWithApproach1Optimized: degree_u must be >= 2");
    }

    const std::vector<double> err_probe = AggregateUPassProbeErrors(surface);
    const double peak_err = MaxError(err_probe);

    auto accept_u_pass = [&](double tol) {
        return ReduceUPassOnlyOptimized(surface, u_domain_global, v_domain_global, tol, parallel);
    };
    auto force_accept = [&]() {
        return std::vector<UPassState>{accept_u_pass(std::numeric_limits<double>::infinity())};
    };

    if (peak_err <= max_error)
    {
        return {accept_u_pass(max_error)};
    }

    const auto site = MidpointKnotForMaxError(surface.knotvector_u, err_probe);
    if (!site.has_value())
    {
        return force_accept();
    }

    const double u_mid_local = std::get<3>(*site);
    const double u_split_global =
        GlobalParamFromLocal(u_mid_local, u_domain_global, surface.u_domain);
    const double u0 = u_domain_global.first;
    const double u1 = u_domain_global.second;
    const double eps = 1e-9 * std::max(1.0, std::abs(u1 - u0));
    if (u_split_global <= u0 + eps || u_split_global >= u1 - eps ||
        u_split_global - u0 < options.min_span_width ||
        u1 - u_split_global < options.min_span_width ||
        depth >= options.max_depth)
    {
        return force_accept();
    }

    const double u_split_local =
        LocalSurfaceParamFromGlobal(u_split_global, u_domain_global, surface.u_domain);
    const auto split = SplitSurfaceU(surface, u_split_local);

    std::vector<UPassState> left_states;
    std::vector<UPassState> right_states;
    const bool use_async =
        parallel.max_async_depth > 0 &&
        static_cast<unsigned int>(depth) < parallel.max_async_depth;
    if (use_async)
    {
        auto left_future = std::async(std::launch::async,
                                      [&]() {
                                          return UPassWithApproach1Optimized(split.first,
                                                                             {u0, u_split_global},
                                                                             v_domain_global,
                                                                             max_error,
                                                                             options,
                                                                             depth + 1,
                                                                             parallel);
                                      });
        right_states = UPassWithApproach1Optimized(split.second,
                                                   {u_split_global, u1},
                                                   v_domain_global,
                                                   max_error,
                                                   options,
                                                   depth + 1,
                                                   parallel);
        left_states = left_future.get();
    }
    else
    {
        left_states = UPassWithApproach1Optimized(split.first,
                                                  {u0, u_split_global},
                                                  v_domain_global,
                                                  max_error,
                                                  options,
                                                  depth + 1,
                                                  parallel);
        right_states = UPassWithApproach1Optimized(split.second,
                                                   {u_split_global, u1},
                                                   v_domain_global,
                                                   max_error,
                                                   options,
                                                   depth + 1,
                                                   parallel);
    }
    left_states.insert(left_states.end(), right_states.begin(), right_states.end());
    return left_states;
}

std::vector<ReducedSurfaceSegment> VPassWithApproach1Optimized(
    const UPassState &u_state,
    double max_error,
    const PeakErrorSurfaceSingleStepOptions &options,
    int depth,
    const ParallelConfig &parallel)
{
    const SurfaceData &surface = u_state.surface;
    if (surface.degree_v < 2)
    {
        throw std::invalid_argument("VPassWithApproach1Optimized: degree_v must be >= 2");
    }

    const std::vector<double> err_probe = AggregateVPassProbeErrors(surface);
    const double peak_err = MaxError(err_probe);

    auto accept_full = [&](double tol) {
        return ReduceVPassOnlyOptimized(u_state, tol, parallel);
    };
    auto force_accept = [&]() {
        return std::vector<ReducedSurfaceSegment>{accept_full(std::numeric_limits<double>::infinity())};
    };

    if (peak_err <= max_error)
    {
        return {accept_full(max_error)};
    }

    const auto site = MidpointKnotForMaxError(surface.knotvector_v, err_probe);
    if (!site.has_value())
    {
        return force_accept();
    }

    const double v_mid_local = std::get<3>(*site);
    const double v_split_global =
        GlobalParamFromLocal(v_mid_local, u_state.v_domain, surface.v_domain);
    const double v0 = u_state.v_domain.first;
    const double v1 = u_state.v_domain.second;
    const double eps = 1e-9 * std::max(1.0, std::abs(v1 - v0));
    if (v_split_global <= v0 + eps || v_split_global >= v1 - eps ||
        v_split_global - v0 < options.min_span_width ||
        v1 - v_split_global < options.min_span_width ||
        depth >= options.max_depth)
    {
        return force_accept();
    }

    const double v_split_local =
        LocalSurfaceParamFromGlobal(v_split_global, u_state.v_domain, surface.v_domain);
    const auto split = SplitSurfaceV(surface, v_split_local);

    UPassState left_state = u_state;
    left_state.surface = split.first;
    left_state.v_domain = {v0, v_split_global};
    UPassState right_state = u_state;
    right_state.surface = split.second;
    right_state.v_domain = {v_split_global, v1};

    std::vector<ReducedSurfaceSegment> left_segments;
    std::vector<ReducedSurfaceSegment> right_segments;
    const bool use_async =
        parallel.max_async_depth > 0 &&
        static_cast<unsigned int>(depth) < parallel.max_async_depth;
    if (use_async)
    {
        auto left_future = std::async(std::launch::async,
                                      [&]() {
                                          return VPassWithApproach1Optimized(left_state,
                                                                             max_error,
                                                                             options,
                                                                             depth + 1,
                                                                             parallel);
                                      });
        right_segments = VPassWithApproach1Optimized(
            right_state, max_error, options, depth + 1, parallel);
        left_segments = left_future.get();
    }
    else
    {
        left_segments = VPassWithApproach1Optimized(
            left_state, max_error, options, depth + 1, parallel);
        right_segments = VPassWithApproach1Optimized(
            right_state, max_error, options, depth + 1, parallel);
    }
    left_segments.insert(left_segments.end(), right_segments.begin(), right_segments.end());
    return left_segments;
}

} // namespace

SingleStepSurfaceReductionResult PeakErrorSurfaceSingleStep(
    const SurfaceData &initial_surface,
    double max_error,
    const PeakErrorSurfaceSingleStepOptions &options)
{
    if (max_error < 0.0)
    {
        throw std::invalid_argument(
            "PeakErrorSurfaceSingleStep: max_error must be non-negative");
    }

    ValidateSurfaceData(initial_surface, "PeakErrorSurfaceSingleStep");

    std::vector<UPassState> u_states = UPassWithApproach1(initial_surface,
                                                          initial_surface.u_domain,
                                                          initial_surface.v_domain,
                                                          max_error,
                                                          options,
                                                          0);
    std::vector<ReducedSurfaceSegment> segments;
    for (const UPassState &u_state : u_states)
    {
        std::vector<ReducedSurfaceSegment> piece_segments =
            VPassWithApproach1(u_state, max_error, options, 0);
        segments.insert(segments.end(), piece_segments.begin(), piece_segments.end());
    }

    std::sort(segments.begin(),
              segments.end(),
              [](const ReducedSurfaceSegment &lhs, const ReducedSurfaceSegment &rhs) {
                  if (lhs.u_domain.first != rhs.u_domain.first)
                  {
                      return lhs.u_domain.first < rhs.u_domain.first;
                  }
                  return lhs.v_domain.first < rhs.v_domain.first;
              });
    return {segments};
}

SingleStepSurfaceReductionResult PeakErrorSurfaceSingleStepOptimized(
    const SurfaceData &initial_surface,
    double max_error,
    const PeakErrorSurfaceSingleStepOptions &options)
{
    if (max_error < 0.0)
    {
        throw std::invalid_argument(
            "PeakErrorSurfaceSingleStepOptimized: max_error must be non-negative");
    }

    ValidateSurfaceData(initial_surface, "PeakErrorSurfaceSingleStepOptimized");

    const ParallelConfig parallel = MakeParallelConfig();
    std::vector<UPassState> u_states = UPassWithApproach1Optimized(initial_surface,
                                                                   initial_surface.u_domain,
                                                                   initial_surface.v_domain,
                                                                   max_error,
                                                                   options,
                                                                   0,
                                                                   parallel);
    std::vector<ReducedSurfaceSegment> segments;
    for (const UPassState &u_state : u_states)
    {
        std::vector<ReducedSurfaceSegment> piece_segments =
            VPassWithApproach1Optimized(u_state, max_error, options, 0, parallel);
        segments.insert(segments.end(), piece_segments.begin(), piece_segments.end());
    }

    std::sort(segments.begin(),
              segments.end(),
              [](const ReducedSurfaceSegment &lhs, const ReducedSurfaceSegment &rhs) {
                  if (lhs.u_domain.first != rhs.u_domain.first)
                  {
                      return lhs.u_domain.first < rhs.u_domain.first;
                  }
                  return lhs.v_domain.first < rhs.v_domain.first;
              });
    return {segments};
}

} // namespace mfem_raytracing
