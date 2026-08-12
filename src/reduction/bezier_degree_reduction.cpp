#include "mfem_raytracing/reduction/bezier_degree_reduction.hpp"

#include <cmath>
#include <stdexcept>

namespace
{

double Binomial(int n, int k)
{
    if (k < 0 || k > n)
    {
        return 0.0;
    }
    if (k == 0 || k == n)
    {
        return 1.0;
    }
    k = std::min(k, n - k);
    double c = 1.0;
    for (int i = 1; i <= k; ++i)
    {
        c = c * static_cast<double>(n - k + i) / static_cast<double>(i);
    }
    return c;
}

double Bernstein(int i, int p, double u)
{
    return Binomial(p, i) * std::pow(u, i) * std::pow(1.0 - u, p - i);
}

double MaxBernstein(int i, int p, int samples = 512)
{
    double peak = 0.0;
    for (int s = 0; s <= samples; ++s)
    {
        const double u = static_cast<double>(s) / static_cast<double>(samples);
        peak = std::max(peak, Bernstein(i, p, u));
    }
    return peak;
}

double MaxAbsBernsteinDifference(int i, int j, int p, int samples = 512)
{
    double peak = 0.0;
    for (int s = 0; s <= samples; ++s)
    {
        const double u = static_cast<double>(s) / static_cast<double>(samples);
        peak = std::max(peak, std::abs(Bernstein(i, p, u) - Bernstein(j, p, u)));
    }
    return peak;
}

double Alpha(int i, int p)
{
    return static_cast<double>(i) / static_cast<double>(p);
}

void ValidateControlPoints(const std::vector<std::vector<double>> &bpts, int dim)
{
    if (dim <= 0)
    {
        throw std::invalid_argument("BezierDegreeReduce: dim must be positive");
    }
    if (bpts.size() < 3)
    {
        throw std::invalid_argument(
            "BezierDegreeReduce: need at least 3 control points (p >= 2)");
    }
    for (const auto &pt : bpts)
    {
        if (static_cast<int>(pt.size()) != dim)
        {
            throw std::invalid_argument(
                "BezierDegreeReduce: all control points must have length dim");
        }
    }
}

std::vector<double> Sub(const std::vector<double> &a,
                        const std::vector<double> &b,
                        int dim)
{
    std::vector<double> r(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d)
    {
        r[static_cast<std::size_t>(d)] = a[static_cast<std::size_t>(d)] - b[static_cast<std::size_t>(d)];
    }
    return r;
}

std::vector<double> ScaleAdd(const std::vector<double> &a,
                             const std::vector<double> &b,
                             double sa,
                             double sb,
                             int dim)
{
    std::vector<double> r(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d)
    {
        r[static_cast<std::size_t>(d)] =
            sa * a[static_cast<std::size_t>(d)] + sb * b[static_cast<std::size_t>(d)];
    }
    return r;
}

std::vector<double> Scale(const std::vector<double> &a, double s, int dim)
{
    std::vector<double> r(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d)
    {
        r[static_cast<std::size_t>(d)] = s * a[static_cast<std::size_t>(d)];
    }
    return r;
}

double NormL2(const std::vector<double> &a, int dim)
{
    double s = 0.0;
    for (int d = 0; d < dim; ++d)
    {
        const double v = a[static_cast<std::size_t>(d)];
        s += v * v;
    }
    return std::sqrt(s);
}

std::vector<double> ForwardControlPoint(const std::vector<double> &q_i,
                                        const std::vector<double> &p_prev,
                                        int i,
                                        int p,
                                        int dim)
{
    const double a = Alpha(i, p);
    const std::vector<double> num = Sub(q_i, Scale(p_prev, a, dim), dim);
    return Scale(num, 1.0 / (1.0 - a), dim);
}

std::vector<double> BackwardControlPoint(const std::vector<double> &q_ip1,
                                         const std::vector<double> &p_next,
                                         int i,
                                         int p,
                                         int dim)
{
    const double a = Alpha(i + 1, p);
    const std::vector<double> num = Sub(q_ip1, Scale(p_next, 1.0 - a, dim), dim);
    return Scale(num, 1.0 / a, dim);
}

double ComputeMaxErrorEven(const std::vector<std::vector<double>> &q,
                           const std::vector<std::vector<double>> &reduced,
                           int degree,
                           int r,
                           int dim)
{
    const std::vector<double> mid =
        ScaleAdd(reduced[static_cast<std::size_t>(r)],
                 reduced[static_cast<std::size_t>(r + 1)],
                 0.5,
                 0.5,
                 dim);
    const std::vector<double> err_vec =
        Sub(q[static_cast<std::size_t>(r + 1)], mid, dim);
    const double err_norm = NormL2(err_vec, dim);
    return err_norm * MaxBernstein(r + 1, degree);
}

double ComputeMaxErrorOdd(const std::vector<double> &p_left,
                          const std::vector<double> &p_right,
                          int r,
                          int p,
                          int dim)
{
    const std::vector<double> diff = Sub(p_left, p_right, dim);
    const double diff_norm = NormL2(diff, dim);
    const double scale = 0.5 * (1.0 - Alpha(r, p));
    return scale * diff_norm * MaxAbsBernsteinDifference(r, r + 1, p);
}

}  // namespace

void BezierDegreeReduce(const std::vector<std::vector<double>> &bpts,
                        int dim,
                        std::vector<std::vector<double>> &reduced_bpts,
                        double &max_err)
{
    ValidateControlPoints(bpts, dim);

    // Piegl notation: input has degree n with n+1 control points (Q_0..Q_n).
    // Odd/even branching and alpha_i = i/n use n (degree), NOT the CP count n+1.
    const int degree = static_cast<int>(bpts.size()) - 1;
    const int r = (degree - 1) / 2;

    reduced_bpts.assign(static_cast<std::size_t>(degree),
                        std::vector<double>(static_cast<std::size_t>(dim)));

    reduced_bpts[0] = bpts[0];
    for (int i = 1; i <= r; ++i)
    {
        reduced_bpts[static_cast<std::size_t>(i)] =
            ForwardControlPoint(bpts[static_cast<std::size_t>(i)],
                                reduced_bpts[static_cast<std::size_t>(i - 1)],
                                i,
                                degree,
                                dim);
    }

    reduced_bpts[static_cast<std::size_t>(degree - 1)] =
        bpts[static_cast<std::size_t>(degree)];
    for (int i = degree - 2; i >= r + 1; --i)
    {
        reduced_bpts[static_cast<std::size_t>(i)] =
            BackwardControlPoint(bpts[static_cast<std::size_t>(i + 1)],
                                 reduced_bpts[static_cast<std::size_t>(i + 1)],
                                 i,
                                 degree,
                                 dim);
    }

    if (degree % 2 == 1)
    {
        const std::vector<double> p_left =
            ForwardControlPoint(bpts[static_cast<std::size_t>(r)],
                                reduced_bpts[static_cast<std::size_t>(r - 1)],
                                r,
                                degree,
                                dim);
        const std::vector<double> p_right =
            BackwardControlPoint(bpts[static_cast<std::size_t>(r + 1)],
                                 reduced_bpts[static_cast<std::size_t>(r + 1)],
                                 r,
                                 degree,
                                 dim);
        reduced_bpts[static_cast<std::size_t>(r)] =
            ScaleAdd(p_left, p_right, 0.5, 0.5, dim);
        max_err = ComputeMaxErrorOdd(p_left, p_right, r, degree, dim);
    }
    else
    {
        max_err = ComputeMaxErrorEven(bpts, reduced_bpts, degree, r, dim);
    }
}
