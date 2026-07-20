#include "surface_degree_reduction.hpp"

#include "curve_reduction_domain.hpp"
#include "curve_reduction_error_budget.hpp"
#include "curve_reduction_types.hpp"
#include "nurbs_degree_reduction.hpp"
#include "surface_conforming_reduction.hpp"
#include "surface_reduction_domain.hpp"

#include "b_spline_curve_reduction.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <limits>
#include <map>
#include <optional>
#include <set>
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

// Rational A5.11 reduction with optional Piegl & Tiller Eq. 5.30 correction.
// When `correct`, `tol` is treated as a Cartesian bound: the homogeneous error
// is accumulated against tol / factor and rescaled back by factor, where
// factor = (1 + |P|max) / w_min (convex-hull bound over the control net).
bool DegreeReduceNURBCurve530(int n_control_points,
                              int degree,
                              const std::vector<double> &U,
                              const std::vector<std::vector<double>> &cp,
                              const std::vector<double> &weights,
                              std::vector<std::vector<double>> &pw,
                              std::vector<double> &weights_out,
                              std::vector<double> &uh,
                              std::vector<double> &error_array,
                              double tol,
                              bool correct)
{
    if (!correct)
    {
        return DegreeReduceNURBCurve(n_control_points, degree, U, cp, weights, pw,
                                     weights_out, uh, error_array, tol);
    }
    double w_min = std::numeric_limits<double>::infinity();
    bool constant_weights = true;
    for (double w : weights)
    {
        w_min = std::min(w_min, w);
        if (std::abs(w - weights.front()) > 1e-12 * std::max(1.0, std::abs(weights.front())))
        {
            constant_weights = false;
        }
    }
    // The (1+|P|max) term of Eq. 5.30 comes from weight variation; for a curve
    // with constant weights (polynomial in this direction) the exact factor is
    // 1/w_min, so the rational correction does not over-tighten polynomial rows.
    double factor = 1.0 / w_min;
    if (!constant_weights)
    {
        double p_max = 0.0;
        for (const std::vector<double> &pt : cp)
        {
            double sq = 0.0;
            for (double c : pt)
            {
                sq += c * c;
            }
            p_max = std::max(p_max, std::sqrt(sq));
        }
        factor = (1.0 + p_max) / w_min;
    }
    const double hom_tol =
        std::isfinite(tol) ? tol / factor : std::numeric_limits<double>::infinity();
    if (!DegreeReduceNURBCurve(n_control_points, degree, U, cp, weights, pw, weights_out, uh,
                               error_array, hom_tol))
    {
        return false;
    }
    for (double &e : error_array)
    {
        e *= factor;
    }
    return true;
}

std::vector<double> ProbeIsoUColumn(const SurfaceData &surface,
                                    int column_index,
                                    double tol,
                                    bool rational_tol_correction)
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
        const bool ok = DegreeReduceNURBCurve530(
            curve.NumControlPoints(),
            curve.degree,
            curve.knotvector,
            curve.control_points,
            curve.weights,
            reduced_cp,
            reduced_weights,
            reduced_knots,
            error_array,
            tol,
            rational_tol_correction);
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

std::vector<double> ProbeIsoVRow(const SurfaceData &surface,
                                 int row_index,
                                 double tol,
                                 bool rational_tol_correction)
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
        const bool ok = DegreeReduceNURBCurve530(
            curve.NumControlPoints(),
            curve.degree,
            curve.knotvector,
            curve.control_points,
            curve.weights,
            reduced_cp,
            reduced_weights,
            reduced_knots,
            error_array,
            tol,
            rational_tol_correction);
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

std::vector<double> AggregateUPassProbeErrors(const SurfaceData &surface,
                                              bool rational_tol_correction)
{
    std::vector<double> error_u(surface.knotvector_u.size(), 0.0);
    for (int j = 0; j < surface.NumControlPointsV(); ++j)
    {
        const std::vector<double> err = ProbeIsoUColumn(
            surface, j, std::numeric_limits<double>::infinity(), rational_tol_correction);
        const std::size_t n_err = std::min(error_u.size(), err.size());
        for (std::size_t i = 0; i < n_err; ++i)
        {
            error_u[i] = std::max(error_u[i], err[i]);
        }
    }
    return error_u;
}

std::vector<double> AggregateVPassProbeErrors(const SurfaceData &surface,
                                              bool rational_tol_correction)
{
    std::vector<double> error_v(surface.knotvector_v.size(), 0.0);
    for (int i = 0; i < surface.NumControlPointsU(); ++i)
    {
        const std::vector<double> err = ProbeIsoVRow(
            surface, i, std::numeric_limits<double>::infinity(), rational_tol_correction);
        const std::size_t n_err = std::min(error_v.size(), err.size());
        for (std::size_t k = 0; k < n_err; ++k)
        {
            error_v[k] = std::max(error_v[k], err[k]);
        }
    }
    return error_v;
}

ReducedCurveDataWithError ReduceColumnCurve(const SurfaceData &surface,
                                            int column_index,
                                            double tol,
                                            bool rational_tol_correction)
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
        const bool ok = DegreeReduceNURBCurve530(curve.NumControlPoints(),
                                                 curve.degree,
                                                 curve.knotvector,
                                                 curve.control_points,
                                                 curve.weights,
                                                 reduced_cp,
                                                 reduced_weights,
                                                 reduced_knots,
                                                 error_array,
                                                 tol,
                                                 rational_tol_correction);
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

ReducedCurveDataWithError ReduceRowCurve(const SurfaceData &surface,
                                         int row_index,
                                         double tol,
                                         bool rational_tol_correction)
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
        const bool ok = DegreeReduceNURBCurve530(curve.NumControlPoints(),
                                                 curve.degree,
                                                 curve.knotvector,
                                                 curve.control_points,
                                                 curve.weights,
                                                 reduced_cp,
                                                 reduced_weights,
                                                 reduced_knots,
                                                 error_array,
                                                 tol,
                                                 rational_tol_correction);
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
                          double tol,
                          bool rational_tol_correction)
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
            const bool ok = DegreeReduceNURBCurve530(
                curve.NumControlPoints(),
                curve.degree,
                curve.knotvector,
                curve.control_points,
                curve.weights,
                reduced_cp,
                reduced_weights,
                reduced_knots,
                err_j,
                tol,
                rational_tol_correction);
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
                                   const ParallelConfig &parallel,
                                   bool rational_tol_correction)
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
            reduced_columns[static_cast<std::size_t>(j)] =
                ReduceColumnCurve(surface, j, tol, rational_tol_correction);
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

// Combine the u-pass and v-pass deviations into the leaf segment error.
inline double CombineSurfaceError(double u_pass_error,
                                  double v_pass_error,
                                  SurfaceErrorCombination combo)
{
    return combo == SurfaceErrorCombination::Sum ? (u_pass_error + v_pass_error)
                                                 : std::max(u_pass_error, v_pass_error);
}

ReducedSurfaceSegment ReduceVPassOnly(const UPassState &u_state,
                                      double tol,
                                      bool rational_tol_correction,
                                      SurfaceErrorCombination error_combination)
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
            const bool ok = DegreeReduceNURBCurve530(
                curve.NumControlPoints(),
                curve.degree,
                curve.knotvector,
                curve.control_points,
                curve.weights,
                reduced_cp,
                reduced_weights,
                reduced_knots,
                err_i,
                tol,
                rational_tol_correction);
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
            CombineSurfaceError(u_state.u_pass_error, MaxError(error_v), error_combination),
            u_state.u_domain,
            u_state.v_domain};
}

ReducedSurfaceSegment ReduceVPassOnlyOptimized(const UPassState &u_state,
                                               double tol,
                                               const ParallelConfig &parallel,
                                               bool rational_tol_correction,
                                               SurfaceErrorCombination error_combination)
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
            reduced_rows[static_cast<std::size_t>(i)] =
                ReduceRowCurve(surface, i, tol, rational_tol_correction);
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
            CombineSurfaceError(u_state.u_pass_error, MaxError(error_v), error_combination),
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

    const std::vector<double> err_probe =
        AggregateUPassProbeErrors(surface, options.rational_tol_correction);
    const double peak_err = MaxError(err_probe);

    auto accept_u_pass = [&](double tol) {
        return ReduceUPassOnly(surface, u_domain_global, v_domain_global, tol,
                               options.rational_tol_correction);
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

    // "Sum": the u-pass already spent u_pass_error, so the v-pass fits in what's
    // left and the leaf error is the true sum u+v. "Max": v-pass uses the full
    // budget and the leaf error is max(u, v).
    const double v_budget = options.error_combination == SurfaceErrorCombination::Sum
                                ? std::max(0.0, max_error - u_state.u_pass_error)
                                : max_error;

    const std::vector<double> err_probe =
        AggregateVPassProbeErrors(surface, options.rational_tol_correction);
    const double peak_err = MaxError(err_probe);

    auto accept_full = [&](double tol) {
        return ReduceVPassOnly(u_state, tol, options.rational_tol_correction,
                               options.error_combination);
    };

    auto force_accept = [&]() {
        return std::vector<ReducedSurfaceSegment>{accept_full(std::numeric_limits<double>::infinity())};
    };

    if (peak_err <= v_budget)
    {
        return {accept_full(v_budget)};
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

    const std::vector<double> err_probe =
        AggregateUPassProbeErrors(surface, options.rational_tol_correction);
    const double peak_err = MaxError(err_probe);

    auto accept_u_pass = [&](double tol) {
        return ReduceUPassOnlyOptimized(surface, u_domain_global, v_domain_global, tol, parallel,
                                        options.rational_tol_correction);
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

    const double v_budget = options.error_combination == SurfaceErrorCombination::Sum
                                ? std::max(0.0, max_error - u_state.u_pass_error)
                                : max_error;

    const std::vector<double> err_probe =
        AggregateVPassProbeErrors(surface, options.rational_tol_correction);
    const double peak_err = MaxError(err_probe);

    auto accept_full = [&](double tol) {
        return ReduceVPassOnlyOptimized(u_state, tol, parallel, options.rational_tol_correction,
                                        options.error_combination);
    };
    auto force_accept = [&]() {
        return std::vector<ReducedSurfaceSegment>{accept_full(std::numeric_limits<double>::infinity())};
    };

    if (peak_err <= v_budget)
    {
        return {accept_full(v_budget)};
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

// ============================================================================
//  Conforming (watertight) multi-step reduction: competing global-line
//  refinement + greedy full-line coalescing + corner welding.
//  Port of competing_reduction.py / coalesce_reduction.py; lives in this TU to
//  reuse the pass internals above (probes, ReduceUPassOnly / ReduceVPassOnly).
// ============================================================================

namespace
{

struct ConfCell
{
    SurfaceData surface;
    std::pair<double, double> u_range;
    std::pair<double, double> v_range;
    double consumed = 0.0;
    double step_tol = std::numeric_limits<double>::infinity();
    double u_err = 0.0;
    bool forced = false;
    // Cached probe arrays for the current `surface`, invalidated on any surface
    // change (split -> new child, or a pass reduction). A cell that no inserted
    // line passes through keeps its probes across refinement iterations.
    std::vector<double> u_probe;
    std::vector<double> v_probe;
    bool has_u_probe = false;
    bool has_v_probe = false;
};

// Parallel-for over [0, n): chunked std::async, falls back to serial for small
// n or a single thread. Callers must ensure fn(i) touches only element i.
template <typename Fn>
void ParallelForIndices(std::size_t n, unsigned threads, Fn fn)
{
    if (threads <= 1 || n < 2)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            fn(i);
        }
        return;
    }
    const unsigned t = static_cast<unsigned>(std::min<std::size_t>(threads, n));
    const std::size_t chunk = (n + t - 1) / t;
    std::vector<std::future<void>> futures;
    futures.reserve(t);
    for (unsigned k = 0; k < t; ++k)
    {
        const std::size_t begin = static_cast<std::size_t>(k) * chunk;
        const std::size_t end = std::min(n, begin + chunk);
        if (begin >= end)
        {
            break;
        }
        futures.push_back(std::async(std::launch::async, [begin, end, &fn]() {
            for (std::size_t i = begin; i < end; ++i)
            {
                fn(i);
            }
        }));
    }
    for (auto &f : futures)
    {
        f.get();
    }
}

// Snap a local split parameter onto an existing interior knot when within
// tolerance: parameters that travel through nested local<->global affine maps
// can drift a few ulps off a knot, and splitting at a not-quite-equal value
// next to an existing knot mis-slices the piece knot vector.
double SnapLocalParamToKnot(double t,
                            const std::vector<double> &knots,
                            double lo,
                            double hi)
{
    const double tol = 1e-9 * std::max(1.0, std::abs(hi - lo));
    double best = t;
    double best_dist = std::numeric_limits<double>::infinity();
    for (double k : knots)
    {
        if (k > lo && k < hi)
        {
            const double d = std::abs(k - t);
            if (d <= tol && d < best_dist)
            {
                best = k;
                best_dist = d;
            }
        }
    }
    return best;
}

ConfCell MakeChildCell(const ConfCell &parent,
                       SurfaceData surface,
                       const std::pair<double, double> &u_range,
                       const std::pair<double, double> &v_range)
{
    ConfCell child = parent;
    child.surface = std::move(surface);
    child.u_range = u_range;
    child.v_range = v_range;
    // Child surface differs from the parent's -> its cached probes are stale.
    child.u_probe.clear();
    child.v_probe.clear();
    child.has_u_probe = false;
    child.has_v_probe = false;
    return child;
}

// Fill (once) and return the cached u/v probe arrays for a cell's current surface.
const std::vector<double> &CachedUProbe(ConfCell &cell, bool rtc)
{
    if (!cell.has_u_probe)
    {
        cell.u_probe = AggregateUPassProbeErrors(cell.surface, rtc);
        cell.has_u_probe = true;
    }
    return cell.u_probe;
}

const std::vector<double> &CachedVProbe(ConfCell &cell, bool rtc)
{
    if (!cell.has_v_probe)
    {
        cell.v_probe = AggregateVPassProbeErrors(cell.surface, rtc);
        cell.has_v_probe = true;
    }
    return cell.v_probe;
}

// Split every cell whose axis range strictly contains x (a global iso-line).
std::vector<ConfCell> InsertConfLine(const std::vector<ConfCell> &cells,
                                     double x,
                                     bool axis_u,
                                     double min_span_width)
{
    std::vector<ConfCell> out;
    out.reserve(cells.size() + 4);
    for (const ConfCell &cell : cells)
    {
        const auto range = axis_u ? cell.u_range : cell.v_range;
        const double lo = range.first;
        const double hi = range.second;
        const double eps = 1e-12 * std::max(1.0, std::abs(hi - lo));
        if (lo + eps < x && x < hi - eps && (x - lo) >= min_span_width &&
            (hi - x) >= min_span_width)
        {
            if (axis_u)
            {
                double local = LocalSurfaceParamFromGlobal(x, cell.u_range, cell.surface.u_domain);
                local = SnapLocalParamToKnot(local, cell.surface.knotvector_u,
                                             cell.surface.u_domain.first,
                                             cell.surface.u_domain.second);
                auto split = SplitSurfaceU(cell.surface, local);
                out.push_back(MakeChildCell(cell, std::move(split.first), {lo, x}, cell.v_range));
                out.push_back(MakeChildCell(cell, std::move(split.second), {x, hi}, cell.v_range));
            }
            else
            {
                double local = LocalSurfaceParamFromGlobal(x, cell.v_range, cell.surface.v_domain);
                local = SnapLocalParamToKnot(local, cell.surface.knotvector_v,
                                             cell.surface.v_domain.first,
                                             cell.surface.v_domain.second);
                auto split = SplitSurfaceV(cell.surface, local);
                out.push_back(MakeChildCell(cell, std::move(split.first), cell.u_range, {lo, x}));
                out.push_back(MakeChildCell(cell, std::move(split.second), cell.u_range, {x, hi}));
            }
        }
        else
        {
            out.push_back(cell);
        }
    }
    return out;
}

// Global split coordinate from the peak-error midpoint rule, or nullopt.
std::optional<double> ProposeConfSplit(const std::vector<double> &err_probe,
                                       const std::vector<double> &knotvector,
                                       const std::pair<double, double> &cell_range,
                                       const std::pair<double, double> &local_domain,
                                       double min_span_width)
{
    const auto site = MidpointKnotForMaxError(knotvector, err_probe);
    if (!site.has_value())
    {
        return std::nullopt;
    }
    const double mid_local = std::get<3>(*site);
    const double x = GlobalParamFromLocal(mid_local, cell_range, local_domain);
    const double lo = cell_range.first;
    const double hi = cell_range.second;
    const double eps = 1e-9 * std::max(1.0, std::abs(hi - lo));
    if (x <= lo + eps || x >= hi - eps || (x - lo) < min_span_width ||
        (hi - x) < min_span_width)
    {
        return std::nullopt;
    }
    return x;
}

// One conforming degree step. `reduce_u` / `reduce_v` support (p, q) inputs:
// a direction that already reached its target degree is skipped entirely.
std::vector<ConfCell> ConformingStep(std::vector<ConfCell> cells,
                                     const PeakErrorSurfaceSingleStepOptions &opt,
                                     bool reduce_u,
                                     bool reduce_v,
                                     unsigned threads)
{
    const bool is_sum = opt.error_combination == SurfaceErrorCombination::Sum;
    const bool rtc = opt.rational_tol_correction;

    // ---- competing refinement (probes on the unreduced cells) --------------
    for (int iter = 0; iter <= opt.max_depth; ++iter)
    {
        // Parallel: fill each active cell's probe cache (touches only that cell;
        // cells whose surface is unchanged from a prior iteration are no-ops).
        ParallelForIndices(cells.size(), threads, [&](std::size_t i) {
            ConfCell &cell = cells[i];
            if (cell.forced)
            {
                return;
            }
            if (reduce_u)
            {
                CachedUProbe(cell, rtc);
            }
            if (reduce_v)
            {
                CachedVProbe(cell, rtc);
            }
        });

        // Serial: cheap decision logic on cached probes (deterministic order).
        std::set<double> u_props;
        std::set<double> v_props;
        for (ConfCell &cell : cells)
        {
            if (cell.forced)
            {
                continue;
            }
            const double eu = reduce_u ? MaxError(cell.u_probe) : 0.0;
            const double ev = reduce_v ? MaxError(cell.v_probe) : 0.0;
            const bool ok = is_sum ? (eu + ev <= cell.step_tol)
                                   : (std::max(eu, ev) <= cell.step_tol);
            if (ok)
            {
                continue;
            }
            bool placed = false;
            const bool u_first = eu >= ev;
            for (int attempt = 0; attempt < 2 && !placed; ++attempt)
            {
                const bool try_u = (attempt == 0) ? u_first : !u_first;
                if (try_u && reduce_u)
                {
                    const auto x = ProposeConfSplit(cell.u_probe, cell.surface.knotvector_u,
                                                    cell.u_range, cell.surface.u_domain,
                                                    opt.min_span_width);
                    if (x.has_value())
                    {
                        u_props.insert(*x);
                        placed = true;
                    }
                }
                else if (!try_u && reduce_v)
                {
                    const auto x = ProposeConfSplit(cell.v_probe, cell.surface.knotvector_v,
                                                    cell.v_range, cell.surface.v_domain,
                                                    opt.min_span_width);
                    if (x.has_value())
                    {
                        v_props.insert(*x);
                        placed = true;
                    }
                }
            }
            if (!placed)
            {
                cell.forced = true;
            }
        }
        if (u_props.empty() && v_props.empty())
        {
            break;
        }
        for (double x : u_props)
        {
            cells = InsertConfLine(cells, x, true, opt.min_span_width);
        }
        for (double x : v_props)
        {
            cells = InsertConfLine(cells, x, false, opt.min_span_width);
        }
    }

    // ---- u-pass (parallel reduction; each cell writes only itself) ----------
    ParallelForIndices(cells.size(), threads, [&](std::size_t i) {
        ConfCell &cell = cells[i];
        if (reduce_u)
        {
            const double eu = MaxError(CachedUProbe(cell, rtc));
            const double tol = (eu <= cell.step_tol) ? cell.step_tol
                                                     : std::numeric_limits<double>::infinity();
            if (!std::isfinite(tol))
            {
                std::fprintf(stderr,
                             "conforming u-pass forced accept on u=[%g,%g] v=[%g,%g]\n",
                             cell.u_range.first, cell.u_range.second,
                             cell.v_range.first, cell.v_range.second);
            }
            UPassState state =
                ReduceUPassOnly(cell.surface, cell.u_range, cell.v_range, tol, rtc);
            cell.surface = std::move(state.surface);
            cell.u_err = state.u_pass_error;
        }
        else
        {
            cell.u_err = 0.0;
        }
        // surface changed -> invalidate probe caches for the v phase
        cell.has_u_probe = false;
        cell.has_v_probe = false;
        cell.forced = false;
    });

    auto v_budget = [&](const ConfCell &cell) {
        return is_sum ? std::max(0.0, cell.step_tol - cell.u_err) : cell.step_tol;
    };

    // ---- v safety refinement (real intermediates, not the pre-u proxy) ------
    if (reduce_v)
    {
        for (int iter = 0; iter <= opt.max_depth; ++iter)
        {
            ParallelForIndices(cells.size(), threads, [&](std::size_t i) {
                if (!cells[i].forced)
                {
                    CachedVProbe(cells[i], rtc);
                }
            });

            std::set<double> props;
            for (ConfCell &cell : cells)
            {
                if (cell.forced)
                {
                    continue;
                }
                if (MaxError(cell.v_probe) <= v_budget(cell))
                {
                    continue;
                }
                const auto x = ProposeConfSplit(cell.v_probe, cell.surface.knotvector_v,
                                                cell.v_range, cell.surface.v_domain,
                                                opt.min_span_width);
                if (x.has_value())
                {
                    props.insert(*x);
                }
                else
                {
                    cell.forced = true;
                }
            }
            if (props.empty())
            {
                break;
            }
            for (double x : props)
            {
                // children copy the parent's u_err via MakeChildCell
                cells = InsertConfLine(cells, x, false, opt.min_span_width);
            }
        }
    }

    // ---- v-pass (parallel reduction; each cell writes only itself) ----------
    ParallelForIndices(cells.size(), threads, [&](std::size_t i) {
        ConfCell &cell = cells[i];
        if (reduce_v)
        {
            const double budget = v_budget(cell);
            const double ev = MaxError(CachedVProbe(cell, rtc));
            const double tol =
                (ev <= budget) ? budget : std::numeric_limits<double>::infinity();
            if (!std::isfinite(tol))
            {
                std::fprintf(stderr,
                             "conforming v-pass forced accept on u=[%g,%g] v=[%g,%g]\n",
                             cell.u_range.first, cell.u_range.second,
                             cell.v_range.first, cell.v_range.second);
            }
            UPassState state;
            state.surface = cell.surface;
            state.u_pass_error = cell.u_err;
            state.u_domain = cell.u_range;
            state.v_domain = cell.v_range;
            ReducedSurfaceSegment segment =
                ReduceVPassOnly(state, tol, rtc, opt.error_combination);
            cell.surface = std::move(segment.surface);
            cell.consumed += segment.segment_error; // combined u/v per option
        }
        else
        {
            cell.consumed += cell.u_err;
        }
        cell.u_err = 0.0;
        cell.forced = false;
    });
    return cells;
}

// Extract the sub-patch of `surface` over the global rectangle, snapping split
// parameters to existing knots (drifted grid-line guard).
SurfaceData ExtractBlockFromOriginal(const SurfaceData &surface,
                                     double u0,
                                     double u1,
                                     double v0,
                                     double v1)
{
    SurfaceData s = surface;
    auto trim = [](SurfaceData current, double lo, double hi,
                   std::pair<double, double> global, bool axis_u) {
        const double eps = 1e-9 * std::max(1.0, std::abs(global.second - global.first));
        if (hi < global.second - eps)
        {
            const auto &knots = axis_u ? current.knotvector_u : current.knotvector_v;
            const auto dom = axis_u ? current.u_domain : current.v_domain;
            double local = LocalSurfaceParamFromGlobal(hi, global, dom);
            local = SnapLocalParamToKnot(local, knots, dom.first, dom.second);
            current = axis_u ? SplitSurfaceU(current, local).first
                             : SplitSurfaceV(current, local).first;
            global.second = hi;
        }
        if (lo > global.first + eps)
        {
            const auto &knots = axis_u ? current.knotvector_u : current.knotvector_v;
            const auto dom = axis_u ? current.u_domain : current.v_domain;
            double local = LocalSurfaceParamFromGlobal(lo, global, dom);
            local = SnapLocalParamToKnot(local, knots, dom.first, dom.second);
            current = axis_u ? SplitSurfaceU(current, local).second
                             : SplitSurfaceV(current, local).second;
        }
        return current;
    };
    s = trim(std::move(s), u0, u1, surface.u_domain, true);
    s = trim(std::move(s), v0, v1, surface.v_domain, false);
    return s;
}

long long WeldKey(double value)
{
    return std::llround(value * 1e9);
}

// Force every leaf sharing a grid vertex onto one canonical corner control
// point/weight so shared bilinear edges are bitwise identical (exactly C0).
// Coalesced blocks are re-extracted independently from the original surface,
// so shared corners can disagree at ~1e-14 without this.
void WeldLeafCorners(std::vector<ReducedSurfaceLeaf> &leaves)
{
    struct Canon
    {
        std::vector<double> point;
        double weight = 1.0;
        bool has_weight = false;
    };
    std::map<std::pair<long long, long long>, Canon> canon;

    auto corners = [](const ReducedSurfaceLeaf &leaf) {
        const double u0 = leaf.u_domain_global.first;
        const double u1 = leaf.u_domain_global.second;
        const double v0 = leaf.v_domain_global.first;
        const double v1 = leaf.v_domain_global.second;
        return std::array<std::tuple<long long, long long, int, int>, 4>{
            std::tuple<long long, long long, int, int>{WeldKey(u0), WeldKey(v0), 0, 0},
            std::tuple<long long, long long, int, int>{WeldKey(u1), WeldKey(v0), 1, 0},
            std::tuple<long long, long long, int, int>{WeldKey(u0), WeldKey(v1), 0, 1},
            std::tuple<long long, long long, int, int>{WeldKey(u1), WeldKey(v1), 1, 1},
        };
    };

    for (const ReducedSurfaceLeaf &leaf : leaves)
    {
        for (const auto &[ku, kv, i, j] : corners(leaf))
        {
            const auto key = std::make_pair(ku, kv);
            if (canon.find(key) == canon.end())
            {
                Canon c;
                c.point = leaf.surface.control_points[static_cast<std::size_t>(i)]
                                                     [static_cast<std::size_t>(j)];
                if (leaf.surface.IsRational())
                {
                    c.weight = leaf.surface.weights[static_cast<std::size_t>(i)]
                                                   [static_cast<std::size_t>(j)];
                    c.has_weight = true;
                }
                canon.emplace(key, std::move(c));
            }
        }
    }
    for (ReducedSurfaceLeaf &leaf : leaves)
    {
        for (const auto &[ku, kv, i, j] : corners(leaf))
        {
            const Canon &c = canon.at(std::make_pair(ku, kv));
            leaf.surface.control_points[static_cast<std::size_t>(i)]
                                       [static_cast<std::size_t>(j)] = c.point;
            if (leaf.surface.IsRational() && c.has_weight)
            {
                leaf.surface.weights[static_cast<std::size_t>(i)]
                                    [static_cast<std::size_t>(j)] = c.weight;
            }
        }
    }
}

} // namespace

std::pair<SurfaceData, double> ReduceBlockNoSplit(const SurfaceData &block,
                                                  int target_degree_u,
                                                  int target_degree_v,
                                                  SurfaceErrorCombination error_combination,
                                                  bool rational_tol_correction)
{
    SurfaceData s = block;
    double total = 0.0;
    const double inf = std::numeric_limits<double>::infinity();
    while (s.degree_u > target_degree_u || s.degree_v > target_degree_v)
    {
        double u_err = 0.0;
        if (s.degree_u > target_degree_u)
        {
            UPassState state =
                ReduceUPassOnly(s, s.u_domain, s.v_domain, inf, rational_tol_correction);
            s = std::move(state.surface);
            u_err = state.u_pass_error;
        }
        if (s.degree_v > target_degree_v)
        {
            UPassState state;
            state.surface = s;
            state.u_pass_error = u_err;
            state.u_domain = s.u_domain;
            state.v_domain = s.v_domain;
            ReducedSurfaceSegment segment =
                ReduceVPassOnly(state, inf, rational_tol_correction, error_combination);
            s = std::move(segment.surface);
            total += segment.segment_error;
        }
        else
        {
            total += u_err;
        }
    }
    return {std::move(s), total};
}

namespace
{

// Greedy full-line removal on the conforming grid: a line is removable when
// every merged block across it still reduces (without splitting) within the
// budget. Removing full lines keeps the partition a tensor grid (watertight).
//
// When options.hard_seams is set, the original surface's unique knots are
// injected into the grid and never removed, and any candidate block that would
// span multiple input knot spans (or reduce to a non-2×2 degree-(1,1) net) is
// rejected — matching python coalesce_reduction.py.
std::vector<ReducedSurfaceLeaf> CoalesceConformingGrid(
    const SurfaceData &original,
    const std::vector<ReducedSurfaceLeaf> &grid_leaves,
    double max_error,
    const ConformingReductionOptions &options,
    int n_steps,
    unsigned threads)
{
    constexpr double kHardSeamTol = 1e-12;

    auto unique_knots = [](const std::vector<double> &kv) {
        std::set<double> s(kv.begin(), kv.end());
        return std::vector<double>(s.begin(), s.end());
    };
    const std::vector<double> ku_hard = unique_knots(original.knotvector_u);
    const std::vector<double> kv_hard = unique_knots(original.knotvector_v);

    auto crosses_hard_seam = [&](double lo, double hi, const std::vector<double> &hard) {
        if (!options.hard_seams)
        {
            return false;
        }
        for (double k : hard)
        {
            if (lo + kHardSeamTol < k && k < hi - kHardSeamTol)
            {
                return true;
            }
        }
        return false;
    };

    auto is_true_bilinear = [&](const SurfaceData &s) {
        return s.degree_u == options.target_degree_u && s.degree_v == options.target_degree_v &&
               s.NumControlPointsU() == options.target_degree_u + 1 &&
               s.NumControlPointsV() == options.target_degree_v + 1;
    };

    std::set<double> u_set;
    std::set<double> v_set;
    for (const auto &leaf : grid_leaves)
    {
        u_set.insert(leaf.u_domain_global.first);
        u_set.insert(leaf.u_domain_global.second);
        v_set.insert(leaf.v_domain_global.first);
        v_set.insert(leaf.v_domain_global.second);
    }
    if (options.hard_seams)
    {
        u_set.insert(ku_hard.begin(), ku_hard.end());
        v_set.insert(kv_hard.begin(), kv_hard.end());
    }
    std::vector<double> us(u_set.begin(), u_set.end());
    std::vector<double> vs(v_set.begin(), v_set.end());

    const auto combo = options.single_step.error_combination;
    const bool rtc = options.single_step.rational_tol_correction;
    const int tu = options.target_degree_u;
    const int tv = options.target_degree_v;

    // u-strip cache: each [u0, u1] x full-v extraction is reused for every
    // v-block inside it (identical split operations, memoized).
    std::map<std::pair<long long, long long>, SurfaceData> strips;
    auto u_strip = [&](double u0, double u1) -> const SurfaceData & {
        const auto key = std::make_pair(WeldKey(u0), WeldKey(u1));
        auto it = strips.find(key);
        if (it == strips.end())
        {
            it = strips
                     .emplace(key, ExtractBlockFromOriginal(original, u0, u1,
                                                            original.v_domain.first,
                                                            original.v_domain.second))
                     .first;
        }
        return it->second;
    };

    // Cached (surface, error). error = +inf marks a rejected (non-bilinear /
    // multi-span) block so line-removal treats it as non-removable.
    std::map<std::array<long long, 4>, std::pair<SurfaceData, double>> blocks;
    auto block = [&](double u0, double u1, double v0, double v1)
        -> const std::pair<SurfaceData, double> & {
        const std::array<long long, 4> key = {WeldKey(u0), WeldKey(u1), WeldKey(v0),
                                              WeldKey(v1)};
        auto it = blocks.find(key);
        if (it == blocks.end())
        {
            if (crosses_hard_seam(u0, u1, ku_hard) || crosses_hard_seam(v0, v1, kv_hard))
            {
                it = blocks
                         .emplace(key,
                                  std::make_pair(SurfaceData{},
                                                 std::numeric_limits<double>::infinity()))
                         .first;
            }
            else
            {
                const SurfaceData &strip = u_strip(u0, u1);
                SurfaceData sub = ExtractBlockFromOriginal(
                    strip, strip.u_domain.first, strip.u_domain.second, v0, v1);
                auto reduced = ReduceBlockNoSplit(sub, tu, tv, combo, rtc);
                if (options.hard_seams && !is_true_bilinear(reduced.first))
                {
                    reduced.second = std::numeric_limits<double>::infinity();
                }
                it = blocks.emplace(key, std::move(reduced)).first;
            }
        }
        return it->second;
    };

    // The greedy line-removal short-circuits (a non-removable line fails on its
    // first over-budget block), so it already does near-minimal work per line;
    // speculative parallel warming only over-computes. Kept serial on purpose.
    (void)threads;
    auto u_line_removable = [&](std::size_t k) {
        if (crosses_hard_seam(us[k - 1], us[k + 1], ku_hard))
        {
            return false;
        }
        for (std::size_t j = 0; j + 1 < vs.size(); ++j)
        {
            if (block(us[k - 1], us[k + 1], vs[j], vs[j + 1]).second > max_error)
            {
                return false;
            }
        }
        return true;
    };
    auto v_line_removable = [&](std::size_t k) {
        if (crosses_hard_seam(vs[k - 1], vs[k + 1], kv_hard))
        {
            return false;
        }
        for (std::size_t i = 0; i + 1 < us.size(); ++i)
        {
            if (block(us[i], us[i + 1], vs[k - 1], vs[k + 1]).second > max_error)
            {
                return false;
            }
        }
        return true;
    };

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (std::size_t k = 1; k + 1 < us.size(); ++k)
        {
            if (u_line_removable(k))
            {
                us.erase(us.begin() + static_cast<std::ptrdiff_t>(k));
                changed = true;
                break;
            }
        }
        if (changed)
        {
            continue;
        }
        for (std::size_t k = 1; k + 1 < vs.size(); ++k)
        {
            if (v_line_removable(k))
            {
                vs.erase(vs.begin() + static_cast<std::ptrdiff_t>(k));
                changed = true;
                break;
            }
        }
    }

    std::vector<ReducedSurfaceLeaf> leaves;
    leaves.reserve((us.size() - 1) * (vs.size() - 1));
    for (std::size_t i = 0; i + 1 < us.size(); ++i)
    {
        for (std::size_t j = 0; j + 1 < vs.size(); ++j)
        {
            const auto &b = block(us[i], us[i + 1], vs[j], vs[j + 1]);
            if (options.hard_seams &&
                (!std::isfinite(b.second) || !is_true_bilinear(b.first)))
            {
                throw std::runtime_error(
                    "CoalesceConformingGrid: cell is not a true bilinear "
                    "(hard-seam / reduce failure)");
            }
            leaves.push_back(ReducedSurfaceLeaf{b.first,
                                                b.second,
                                                n_steps,
                                                {us[i], us[i + 1]},
                                                {vs[j], vs[j + 1]}});
        }
    }
    WeldLeafCorners(leaves);
    return leaves;
}

} // namespace

MultipleStepSurfaceReductionResult DegreeReduceMultipleStepsConforming(
    const SurfaceData &initial_surface,
    double max_error,
    const ConformingReductionOptions &options)
{
    if (max_error < 0.0)
    {
        throw std::invalid_argument(
            "DegreeReduceMultipleStepsConforming: max_error must be non-negative");
    }
    if (options.target_degree_u < 1 || options.target_degree_v < 1)
    {
        throw std::invalid_argument(
            "DegreeReduceMultipleStepsConforming: target degrees must be >= 1");
    }
    if (initial_surface.degree_u < options.target_degree_u ||
        initial_surface.degree_v < options.target_degree_v)
    {
        throw std::invalid_argument(
            "DegreeReduceMultipleStepsConforming: surface degrees below target");
    }
    ValidateSurfaceData(initial_surface, "DegreeReduceMultipleStepsConforming");

    const int n_steps = std::max(initial_surface.degree_u - options.target_degree_u,
                                 initial_surface.degree_v - options.target_degree_v);
    unsigned threads = options.threads;
    if (threads == 0)
    {
        threads = std::max(1u, std::thread::hardware_concurrency());
    }

#ifdef MFEM_CONFORMING_TIMING
    const auto t_start = std::chrono::steady_clock::now();
#endif
    std::vector<ConfCell> cells;
    cells.push_back(ConfCell{initial_surface,
                             initial_surface.u_domain,
                             initial_surface.v_domain});
    if (n_steps == 0)
    {
        return {{ReducedSurfaceLeaf{initial_surface, 0.0, 0, initial_surface.u_domain,
                                    initial_surface.v_domain}}};
    }

    const bool finite_budget = std::isfinite(max_error);
    for (int step_index = 1; step_index <= n_steps; ++step_index)
    {
        for (ConfCell &cell : cells)
        {
            const double remaining =
                finite_budget ? std::max(0.0, max_error - cell.consumed)
                              : std::numeric_limits<double>::infinity();
            cell.step_tol = StepErrorBudget(max_error, n_steps, step_index, remaining,
                                            options.budget_policy);
            cell.forced = false;
        }
        const bool reduce_u = cells.front().surface.degree_u > options.target_degree_u;
        const bool reduce_v = cells.front().surface.degree_v > options.target_degree_v;
        cells = ConformingStep(std::move(cells), options.single_step, reduce_u, reduce_v,
                               threads);
    }
#ifdef MFEM_CONFORMING_TIMING
    const auto t_grid = std::chrono::steady_clock::now();
#endif

    std::vector<ReducedSurfaceLeaf> leaves;
    if (options.coalesce)
    {
        std::vector<ReducedSurfaceLeaf> grid_leaves;
        grid_leaves.reserve(cells.size());
        for (const ConfCell &cell : cells)
        {
            grid_leaves.push_back(ReducedSurfaceLeaf{cell.surface, cell.consumed, n_steps,
                                                     cell.u_range, cell.v_range});
        }
        leaves = CoalesceConformingGrid(initial_surface, grid_leaves, max_error, options,
                                        n_steps, threads);
    }
    else
    {
        leaves.reserve(cells.size());
        for (const ConfCell &cell : cells)
        {
            leaves.push_back(ReducedSurfaceLeaf{cell.surface, cell.consumed, n_steps,
                                                cell.u_range, cell.v_range});
        }
    }

    std::sort(leaves.begin(), leaves.end(),
              [](const ReducedSurfaceLeaf &a, const ReducedSurfaceLeaf &b) {
                  if (a.u_domain_global.first != b.u_domain_global.first)
                  {
                      return a.u_domain_global.first < b.u_domain_global.first;
                  }
                  return a.v_domain_global.first < b.v_domain_global.first;
              });

#ifdef MFEM_CONFORMING_TIMING
    const auto t_end = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[timing] grid-build %.2fs  coalesce %.2fs\n",
                 std::chrono::duration<double>(t_grid - t_start).count(),
                 std::chrono::duration<double>(t_end - t_grid).count());
#endif
    int over = 0;
    for (const auto &leaf : leaves)
    {
        over += (finite_budget && leaf.total_error > max_error) ? 1 : 0;
    }
    if (over > 0)
    {
        std::fprintf(stderr,
                     "DegreeReduceMultipleStepsConforming: %d leaves exceed max_error=%g "
                     "(kept for watertightness)\n",
                     over, max_error);
    }
    return {std::move(leaves)};
}

} // namespace mfem_raytracing
