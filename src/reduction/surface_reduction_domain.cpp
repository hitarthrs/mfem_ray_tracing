#include "mfem_raytracing/reduction/surface_reduction_domain.hpp"

#include "mfem_raytracing/reduction/curve_reduction_domain.hpp"
#include "mfem_raytracing/reduction/curve_reduction_types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mfem_raytracing
{
namespace
{

constexpr double kZeroTol = 1e-15;
constexpr double kParamTol = 1e-9;

CurveData MakeColumnCurve(const SurfaceData &surface, int j)
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
    if (surface.IsRational())
    {
        curve.weights.resize(static_cast<std::size_t>(surface.NumControlPointsU()));
        for (int i = 0; i < surface.NumControlPointsU(); ++i)
        {
            curve.weights[static_cast<std::size_t>(i)] =
                surface.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
    }
    return curve;
}

CurveData MakeRowCurve(const SurfaceData &surface, int i)
{
    CurveData curve;
    curve.degree = surface.degree_v;
    curve.dim = surface.dim;
    curve.knotvector = surface.knotvector_v;
    curve.domain = surface.v_domain;
    curve.control_points = surface.control_points[static_cast<std::size_t>(i)];
    if (surface.IsRational())
    {
        curve.weights = surface.weights[static_cast<std::size_t>(i)];
    }
    return curve;
}

void ReparameterizeKnots(std::vector<double> &knotvector,
                         const std::pair<double, double> &old_domain,
                         const std::pair<double, double> &new_domain)
{
    const double old_lo = old_domain.first;
    const double old_hi = old_domain.second;
    const double new_lo = new_domain.first;
    const double new_hi = new_domain.second;
    if (std::abs(old_hi - old_lo) < kZeroTol)
    {
        std::fill(knotvector.begin(), knotvector.end(), new_lo);
        return;
    }
    for (double &knot : knotvector)
    {
        const double t = (knot - old_lo) / (old_hi - old_lo);
        knot = new_lo + t * (new_hi - new_lo);
    }
}

void VerifyMatchingCurveGeometry(const CurveData &reference,
                                 const CurveData &candidate,
                                 const char *label)
{
    if (reference.degree != candidate.degree ||
        reference.knotvector.size() != candidate.knotvector.size())
    {
        throw std::runtime_error(std::string(label) + ": knot vector mismatch");
    }
    for (std::size_t k = 0; k < reference.knotvector.size(); ++k)
    {
        if (std::abs(reference.knotvector[k] - candidate.knotvector[k]) > 1e-12)
        {
            throw std::runtime_error(std::string(label) + ": inconsistent split knot vectors");
        }
    }
}

} // namespace

bool SurfaceData::IsRational() const
{
    return !weights.empty();
}

int SurfaceData::NumControlPointsU() const
{
    return static_cast<int>(control_points.size());
}

int SurfaceData::NumControlPointsV() const
{
    return control_points.empty() ? 0 : static_cast<int>(control_points.front().size());
}

void ValidateSurfaceData(const SurfaceData &surface, const char *label)
{
    if (surface.degree_u < 1 || surface.degree_v < 1)
    {
        throw std::invalid_argument(std::string(label) + ": degrees must be >= 1");
    }
    if (surface.dim <= 0)
    {
        throw std::invalid_argument(std::string(label) + ": dim must be positive");
    }
    if (surface.control_points.empty() || surface.control_points.front().empty())
    {
        throw std::invalid_argument(std::string(label) + ": control net is empty");
    }

    const int n_u = surface.NumControlPointsU();
    const int n_v = surface.NumControlPointsV();
    for (const auto &row : surface.control_points)
    {
        if (static_cast<int>(row.size()) != n_v)
        {
            throw std::invalid_argument(std::string(label) + ": ragged control net");
        }
        for (const auto &point : row)
        {
            if (static_cast<int>(point.size()) != surface.dim)
            {
                throw std::invalid_argument(std::string(label) +
                                            ": inconsistent control point dimension");
            }
        }
    }

    if (surface.IsRational())
    {
        if (static_cast<int>(surface.weights.size()) != n_u)
        {
            throw std::invalid_argument(std::string(label) + ": weight row count mismatch");
        }
        for (const auto &row : surface.weights)
        {
            if (static_cast<int>(row.size()) != n_v)
            {
                throw std::invalid_argument(std::string(label) + ": weight column count mismatch");
            }
        }
    }

    if (static_cast<int>(surface.knotvector_u.size()) != n_u + surface.degree_u + 1)
    {
        throw std::invalid_argument(std::string(label) + ": invalid u knot vector length");
    }
    if (static_cast<int>(surface.knotvector_v.size()) != n_v + surface.degree_v + 1)
    {
        throw std::invalid_argument(std::string(label) + ": invalid v knot vector length");
    }

    if (surface.u_domain.first > surface.u_domain.second + kParamTol ||
        surface.v_domain.first > surface.v_domain.second + kParamTol)
    {
        throw std::invalid_argument(std::string(label) + ": invalid surface domain ordering");
    }
}

double LocalSurfaceParamFromGlobal(double t_global,
                                   const std::pair<double, double> &global_domain,
                                   const std::pair<double, double> &local_domain)
{
    const double g0 = global_domain.first;
    const double g1 = global_domain.second;
    const double l0 = local_domain.first;
    const double l1 = local_domain.second;
    if (std::abs(g1 - g0) < kZeroTol)
    {
        return l0;
    }
    return l0 + (t_global - g0) / (g1 - g0) * (l1 - l0);
}

std::pair<SurfaceData, SurfaceData> SplitSurfaceU(const SurfaceData &surface, double u)
{
    ValidateSurfaceData(surface, "SplitSurfaceU");
    if (u <= surface.u_domain.first + kParamTol || u >= surface.u_domain.second - kParamTol)
    {
        throw std::invalid_argument("SplitSurfaceU: split must lie strictly inside the u domain");
    }

    SurfaceData left;
    SurfaceData right;
    left.degree_u = right.degree_u = surface.degree_u;
    left.degree_v = right.degree_v = surface.degree_v;
    left.dim = right.dim = surface.dim;
    left.knotvector_v = right.knotvector_v = surface.knotvector_v;
    left.v_domain = right.v_domain = surface.v_domain;

    CurveData left_ref;
    CurveData right_ref;
    bool have_ref = false;

    const int n_v = surface.NumControlPointsV();
    left.control_points.resize(0);
    right.control_points.resize(0);

    std::vector<std::vector<CurveData>> split_columns(static_cast<std::size_t>(n_v));
    for (int j = 0; j < n_v; ++j)
    {
        auto split = SplitCurve(MakeColumnCurve(surface, j), u);
        if (surface.IsRational())
        {
            if (split.first.weights.empty())
            {
                split.first.weights.assign(static_cast<std::size_t>(split.first.NumControlPoints()), 1.0);
            }
            if (split.second.weights.empty())
            {
                split.second.weights.assign(static_cast<std::size_t>(split.second.NumControlPoints()), 1.0);
            }
        }
        split_columns[static_cast<std::size_t>(j)] = {split.first, split.second};
        if (!have_ref)
        {
            left_ref = split.first;
            right_ref = split.second;
            have_ref = true;
        }
        else
        {
            VerifyMatchingCurveGeometry(left_ref, split.first, "SplitSurfaceU");
            VerifyMatchingCurveGeometry(right_ref, split.second, "SplitSurfaceU");
        }
    }

    left.knotvector_u = left_ref.knotvector;
    right.knotvector_u = right_ref.knotvector;
    left.u_domain = left_ref.domain;
    right.u_domain = right_ref.domain;
    left.control_points.assign(static_cast<std::size_t>(left_ref.NumControlPoints()),
                               std::vector<std::vector<double>>(static_cast<std::size_t>(n_v)));
    right.control_points.assign(static_cast<std::size_t>(right_ref.NumControlPoints()),
                                std::vector<std::vector<double>>(static_cast<std::size_t>(n_v)));

    if (surface.IsRational())
    {
        left.weights.assign(static_cast<std::size_t>(left_ref.NumControlPoints()),
                            std::vector<double>(static_cast<std::size_t>(n_v), 1.0));
        right.weights.assign(static_cast<std::size_t>(right_ref.NumControlPoints()),
                             std::vector<double>(static_cast<std::size_t>(n_v), 1.0));
    }

    for (int j = 0; j < n_v; ++j)
    {
        const CurveData &left_curve = split_columns[static_cast<std::size_t>(j)][0];
        const CurveData &right_curve = split_columns[static_cast<std::size_t>(j)][1];
        for (int i = 0; i < left_curve.NumControlPoints(); ++i)
        {
            left.control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                left_curve.control_points[static_cast<std::size_t>(i)];
            if (surface.IsRational())
            {
                left.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    left_curve.weights[static_cast<std::size_t>(i)];
            }
        }
        for (int i = 0; i < right_curve.NumControlPoints(); ++i)
        {
            right.control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                right_curve.control_points[static_cast<std::size_t>(i)];
            if (surface.IsRational())
            {
                right.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    right_curve.weights[static_cast<std::size_t>(i)];
            }
        }
    }

    ReparameterizeKnots(left.knotvector_u, left.u_domain, {0.0, 1.0});
    ReparameterizeKnots(right.knotvector_u, right.u_domain, {0.0, 1.0});
    left.u_domain = {0.0, 1.0};
    right.u_domain = {0.0, 1.0};

    return {left, right};
}

std::pair<SurfaceData, SurfaceData> SplitSurfaceV(const SurfaceData &surface, double v)
{
    ValidateSurfaceData(surface, "SplitSurfaceV");
    if (v <= surface.v_domain.first + kParamTol || v >= surface.v_domain.second - kParamTol)
    {
        throw std::invalid_argument("SplitSurfaceV: split must lie strictly inside the v domain");
    }

    SurfaceData left;
    SurfaceData right;
    left.degree_u = right.degree_u = surface.degree_u;
    left.degree_v = right.degree_v = surface.degree_v;
    left.dim = right.dim = surface.dim;
    left.knotvector_u = right.knotvector_u = surface.knotvector_u;
    left.u_domain = right.u_domain = surface.u_domain;

    CurveData left_ref;
    CurveData right_ref;
    bool have_ref = false;

    const int n_u = surface.NumControlPointsU();
    std::vector<std::vector<CurveData>> split_rows(static_cast<std::size_t>(n_u));
    for (int i = 0; i < n_u; ++i)
    {
        auto split = SplitCurve(MakeRowCurve(surface, i), v);
        if (surface.IsRational())
        {
            if (split.first.weights.empty())
            {
                split.first.weights.assign(static_cast<std::size_t>(split.first.NumControlPoints()), 1.0);
            }
            if (split.second.weights.empty())
            {
                split.second.weights.assign(static_cast<std::size_t>(split.second.NumControlPoints()), 1.0);
            }
        }
        split_rows[static_cast<std::size_t>(i)] = {split.first, split.second};
        if (!have_ref)
        {
            left_ref = split.first;
            right_ref = split.second;
            have_ref = true;
        }
        else
        {
            VerifyMatchingCurveGeometry(left_ref, split.first, "SplitSurfaceV");
            VerifyMatchingCurveGeometry(right_ref, split.second, "SplitSurfaceV");
        }
    }

    left.knotvector_v = left_ref.knotvector;
    right.knotvector_v = right_ref.knotvector;
    left.v_domain = left_ref.domain;
    right.v_domain = right_ref.domain;
    left.control_points.assign(static_cast<std::size_t>(n_u),
                               std::vector<std::vector<double>>(
                                   static_cast<std::size_t>(left_ref.NumControlPoints())));
    right.control_points.assign(static_cast<std::size_t>(n_u),
                                std::vector<std::vector<double>>(
                                    static_cast<std::size_t>(right_ref.NumControlPoints())));

    if (surface.IsRational())
    {
        left.weights.assign(static_cast<std::size_t>(n_u),
                            std::vector<double>(static_cast<std::size_t>(left_ref.NumControlPoints()), 1.0));
        right.weights.assign(static_cast<std::size_t>(n_u),
                             std::vector<double>(static_cast<std::size_t>(right_ref.NumControlPoints()), 1.0));
    }

    for (int i = 0; i < n_u; ++i)
    {
        const CurveData &left_curve = split_rows[static_cast<std::size_t>(i)][0];
        const CurveData &right_curve = split_rows[static_cast<std::size_t>(i)][1];
        left.control_points[static_cast<std::size_t>(i)] = left_curve.control_points;
        right.control_points[static_cast<std::size_t>(i)] = right_curve.control_points;
        if (surface.IsRational())
        {
            left.weights[static_cast<std::size_t>(i)] = left_curve.weights;
            right.weights[static_cast<std::size_t>(i)] = right_curve.weights;
        }
    }

    ReparameterizeKnots(left.knotvector_v, left.v_domain, {0.0, 1.0});
    ReparameterizeKnots(right.knotvector_v, right.v_domain, {0.0, 1.0});
    left.v_domain = {0.0, 1.0};
    right.v_domain = {0.0, 1.0};

    return {left, right};
}

} // namespace mfem_raytracing
