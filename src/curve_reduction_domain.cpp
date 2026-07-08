#include "curve_reduction_domain.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mfem_raytracing
{
namespace
{

using Point = std::vector<double>;

constexpr double kZeroTol = 1e-15;
constexpr double kParamTol = 1e-9;

bool IsNondecreasing(const std::vector<double> &values)
{
    for (std::size_t i = 1; i < values.size(); ++i)
    {
        if (values[i] + kZeroTol < values[i - 1])
        {
            return false;
        }
    }
    return true;
}

std::vector<Point> ToHomogeneous(const CurveData &curve)
{
    const int homog_dim = curve.dim + 1;
    std::vector<Point> homogeneous(curve.control_points.size(),
                                   Point(static_cast<std::size_t>(homog_dim), 0.0));
    for (std::size_t i = 0; i < curve.control_points.size(); ++i)
    {
        const double w = curve.IsRational() ? curve.weights[i] : 1.0;
        for (int d = 0; d < curve.dim; ++d)
        {
            homogeneous[i][static_cast<std::size_t>(d)] =
                curve.control_points[i][static_cast<std::size_t>(d)] * w;
        }
        homogeneous[i][static_cast<std::size_t>(curve.dim)] = w;
    }
    return homogeneous;
}

CurveData FromHomogeneous(const std::vector<Point> &homogeneous,
                          const std::vector<double> &knotvector,
                          int degree,
                          int spatial_dim)
{
    if (homogeneous.empty())
    {
        throw std::invalid_argument("FromHomogeneous: empty control net");
    }

    CurveData curve;
    curve.degree = degree;
    curve.dim = spatial_dim;
    curve.knotvector = knotvector;
    curve.domain = CurveDomain(knotvector, degree);
    curve.control_points.assign(homogeneous.size(),
                                Point(static_cast<std::size_t>(spatial_dim), 0.0));
    curve.weights.assign(homogeneous.size(), 1.0);

    bool all_unit_weights = true;
    for (std::size_t i = 0; i < homogeneous.size(); ++i)
    {
        const double w = homogeneous[i][static_cast<std::size_t>(spatial_dim)];
        if (std::abs(w) < kZeroTol)
        {
            throw std::invalid_argument("FromHomogeneous: zero weight encountered");
        }
        curve.weights[i] = w;
        all_unit_weights = all_unit_weights && std::abs(w - 1.0) <= 1e-12;
        for (int d = 0; d < spatial_dim; ++d)
        {
            curve.control_points[i][static_cast<std::size_t>(d)] =
                homogeneous[i][static_cast<std::size_t>(d)] / w;
        }
    }

    if (all_unit_weights)
    {
        curve.weights.clear();
    }
    return curve;
}

int FindSpan(const std::vector<double> &U, int degree, double u)
{
    const int n = static_cast<int>(U.size()) - degree - 2;
    if (u >= U[static_cast<std::size_t>(n + 1)] - kParamTol)
    {
        return n;
    }
    if (u <= U[static_cast<std::size_t>(degree)] + kParamTol)
    {
        return degree;
    }

    int low = degree;
    int high = n + 1;
    int mid = (low + high) / 2;
    while (u < U[static_cast<std::size_t>(mid)] ||
           u >= U[static_cast<std::size_t>(mid + 1)])
    {
        if (u < U[static_cast<std::size_t>(mid)])
        {
            high = mid;
        }
        else
        {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    return mid;
}

int KnotMultiplicity(const std::vector<double> &U, double u)
{
    int multiplicity = 0;
    for (double value : U)
    {
        if (std::abs(value - u) <= kParamTol)
        {
            ++multiplicity;
        }
    }
    return multiplicity;
}

void InsertKnotOnce(std::vector<Point> &control_points,
                    std::vector<double> &knotvector,
                    int degree,
                    double u)
{
    const int n = static_cast<int>(control_points.size()) - 1;
    const int k = FindSpan(knotvector, degree, u);

    int s = 0;
    for (double knot : knotvector)
    {
        if (std::abs(knot - u) <= kParamTol)
        {
            ++s;
        }
    }
    if (s > degree)
    {
        throw std::invalid_argument("InsertKnotOnce: knot multiplicity exceeds degree");
    }

    std::vector<Point> next_points(static_cast<std::size_t>(n + 2),
                                   Point(control_points.front().size(), 0.0));
    std::vector<double> next_knots(static_cast<std::size_t>(knotvector.size() + 1), 0.0);

    for (int i = 0; i <= k; ++i)
    {
        next_knots[static_cast<std::size_t>(i)] = knotvector[static_cast<std::size_t>(i)];
    }
    next_knots[static_cast<std::size_t>(k + 1)] = u;
    for (int i = k + 1; i < static_cast<int>(knotvector.size()); ++i)
    {
        next_knots[static_cast<std::size_t>(i + 1)] = knotvector[static_cast<std::size_t>(i)];
    }

    for (int i = 0; i <= k - degree; ++i)
    {
        next_points[static_cast<std::size_t>(i)] = control_points[static_cast<std::size_t>(i)];
    }
    for (int i = k - s; i <= n; ++i)
    {
        next_points[static_cast<std::size_t>(i + 1)] = control_points[static_cast<std::size_t>(i)];
    }
    for (int i = k - degree + 1; i <= k - s; ++i)
    {
        const double denom =
            knotvector[static_cast<std::size_t>(i + degree)] - knotvector[static_cast<std::size_t>(i)];
        const double alpha =
            (std::abs(denom) <= kZeroTol) ? 0.0
                                          : (u - knotvector[static_cast<std::size_t>(i)]) / denom;
        for (std::size_t d = 0; d < control_points.front().size(); ++d)
        {
            next_points[static_cast<std::size_t>(i)][d] =
                alpha * control_points[static_cast<std::size_t>(i)][d] +
                (1.0 - alpha) * control_points[static_cast<std::size_t>(i - 1)][d];
        }
    }

    control_points.swap(next_points);
    knotvector.swap(next_knots);
}

} // namespace

bool CurveData::IsRational() const
{
    return !weights.empty();
}

int CurveData::NumControlPoints() const
{
    return static_cast<int>(control_points.size());
}

std::pair<double, double> CurveDomain(const std::vector<double> &knotvector, int degree)
{
    if (degree < 0 || static_cast<int>(knotvector.size()) < degree + 2)
    {
        throw std::invalid_argument("CurveDomain: invalid degree or knot vector");
    }
    const int last_cp = static_cast<int>(knotvector.size()) - degree - 2;
    return {knotvector[static_cast<std::size_t>(degree)],
            knotvector[static_cast<std::size_t>(last_cp + 1)]};
}

void ValidateCurveData(const CurveData &curve, const char *label)
{
    if (curve.degree < 1)
    {
        throw std::invalid_argument(std::string(label) + ": degree must be >= 1");
    }
    if (curve.dim <= 0)
    {
        throw std::invalid_argument(std::string(label) + ": dim must be positive");
    }
    if (curve.control_points.empty())
    {
        throw std::invalid_argument(std::string(label) + ": control point list is empty");
    }
    for (const auto &point : curve.control_points)
    {
        if (static_cast<int>(point.size()) != curve.dim)
        {
            throw std::invalid_argument(std::string(label) +
                                        ": inconsistent control point dimension");
        }
    }
    if (!curve.weights.empty() && curve.weights.size() != curve.control_points.size())
    {
        throw std::invalid_argument(std::string(label) +
                                    ": weights/control point count mismatch");
    }
    if (static_cast<int>(curve.knotvector.size()) !=
        curve.NumControlPoints() + curve.degree + 1)
    {
        throw std::invalid_argument(std::string(label) +
                                    ": knot vector length must be n+p+1");
    }
    if (!IsNondecreasing(curve.knotvector))
    {
        throw std::invalid_argument(std::string(label) +
                                    ": knot vector must be nondecreasing");
    }
}

double LocalParamFromGlobal(double u_global,
                            const std::pair<double, double> &piece_global,
                            const std::pair<double, double> &local_domain)
{
    const double g0 = piece_global.first;
    const double g1 = piece_global.second;
    const double l0 = local_domain.first;
    const double l1 = local_domain.second;
    if (std::abs(g1 - g0) < kZeroTol)
    {
        return l0;
    }
    const double t = (u_global - g0) / (g1 - g0);
    return l0 + t * (l1 - l0);
}

double GlobalParamFromLocal(double u_local,
                            const std::pair<double, double> &piece_global,
                            const std::pair<double, double> &local_domain)
{
    const double g0 = piece_global.first;
    const double g1 = piece_global.second;
    const double l0 = local_domain.first;
    const double l1 = local_domain.second;
    if (std::abs(l1 - l0) < kZeroTol)
    {
        return g0;
    }
    const double t = (u_local - l0) / (l1 - l0);
    return g0 + t * (g1 - g0);
}

std::pair<double, double> MapIntervalToGlobal(
    const std::pair<double, double> &interval,
    const std::pair<double, double> &local_domain,
    const std::pair<double, double> &global_domain)
{
    return {GlobalParamFromLocal(interval.first, global_domain, local_domain),
            GlobalParamFromLocal(interval.second, global_domain, local_domain)};
}

CurveData ReparameterizeCurve(const CurveData &curve,
                              const std::pair<double, double> &new_domain)
{
    CurveData out = curve;
    const auto [old_lo, old_hi] = curve.domain;
    const auto [new_lo, new_hi] = new_domain;
    if (std::abs(old_hi - old_lo) < kZeroTol)
    {
        std::fill(out.knotvector.begin(), out.knotvector.end(), new_lo);
    }
    else
    {
        for (double &knot : out.knotvector)
        {
            const double t = (knot - old_lo) / (old_hi - old_lo);
            knot = new_lo + t * (new_hi - new_lo);
        }
    }
    out.domain = new_domain;
    return out;
}

std::pair<CurveData, CurveData> SplitCurve(const CurveData &curve, double u)
{
    ValidateCurveData(curve, "SplitCurve");
    const auto [domain_lo, domain_hi] = curve.domain;
    if (u <= domain_lo + kParamTol || u >= domain_hi - kParamTol)
    {
        throw std::invalid_argument(
            "SplitCurve: split parameter must lie strictly inside the domain");
    }

    std::vector<Point> homogeneous = ToHomogeneous(curve);
    std::vector<double> knots = curve.knotvector;

    const int degree = curve.degree;
    const int k = FindSpan(curve.knotvector, curve.degree, u);
    const int s = KnotMultiplicity(curve.knotvector, u);
    const int inserts = degree - s;
    for (int r = 0; r < inserts; ++r)
    {
        InsertKnotOnce(homogeneous, knots, degree, u);
    }

    const int split_cp_index = k - s;
    std::vector<Point> left_homogeneous(homogeneous.begin(),
                                        homogeneous.begin() + split_cp_index + 1);
    std::vector<Point> right_homogeneous(homogeneous.begin() + split_cp_index,
                                         homogeneous.end());

    std::vector<double> left_knots(knots.begin(), knots.begin() + k + inserts + 1);
    left_knots.push_back(u);

    std::vector<double> right_knots;
    right_knots.push_back(u);
    right_knots.insert(right_knots.end(), knots.begin() + (k - s + 1), knots.end());

    return {FromHomogeneous(left_homogeneous, left_knots, degree, curve.dim),
            FromHomogeneous(right_homogeneous, right_knots, degree, curve.dim)};
}

CurveData ExtractSubcurveGlobal(const CurveData &curve,
                                const std::pair<double, double> &piece_global,
                                double u0_global,
                                double u1_global)
{
    ValidateCurveData(curve, "ExtractSubcurveGlobal");
    if (u0_global > u1_global + kParamTol)
    {
        throw std::invalid_argument("ExtractSubcurveGlobal: u0 must be <= u1");
    }

    CurveData piece = curve;
    const std::pair<double, double> local_domain = curve.domain;
    const double u0_local = LocalParamFromGlobal(u0_global, piece_global, local_domain);
    const double u1_local = LocalParamFromGlobal(u1_global, piece_global, local_domain);

    if (u0_local > piece.domain.first + kParamTol)
    {
        piece = SplitCurve(piece, u0_local).second;
    }
    if (u1_local < piece.domain.second - kParamTol)
    {
        const double remapped_u1 =
            LocalParamFromGlobal(u1_global, {u0_global, piece_global.second}, piece.domain);
        piece = SplitCurve(piece, remapped_u1).first;
    }
    return ReparameterizeCurve(piece, {0.0, 1.0});
}

} // namespace mfem_raytracing
