#include "b_spline_curve_reduction.hpp"

#include "bezier_degree_reduction.hpp"

#include <cmath>
#include <stdexcept>

namespace
{

using Vec = std::vector<double>;

double Distance(const Vec &p, const Vec &q)
{
    double s = 0.0;
    for (std::size_t d = 0; d < p.size(); ++d)
    {
        const double diff = p[d] - q[d];
        s += diff * diff;
    }
    return std::sqrt(s);
}

Vec LinComb(const Vec &a, double sa, const Vec &b, double sb)
{
    Vec r(a.size());
    for (std::size_t d = 0; d < a.size(); ++d)
    {
        r[d] = sa * a[d] + sb * b[d];
    }
    return r;
}

Vec ScalePoint(const Vec &a, double s)
{
    Vec r(a.size());
    for (std::size_t d = 0; d < a.size(); ++d)
    {
        r[d] = s * a[d];
    }
    return r;
}

Vec SubPoint(const Vec &a, const Vec &b)
{
    Vec r(a.size());
    for (std::size_t d = 0; d < a.size(); ++d)
    {
        r[d] = a[d] - b[d];
    }
    return r;
}

Vec DivPoint(const Vec &a, double denom)
{
    Vec r(a.size());
    for (std::size_t d = 0; d < a.size(); ++d)
    {
        r[d] = a[d] / denom;
    }
    return r;
}

}  // namespace

bool DegreeReduceCurve(int n_control_points,
                       int degree,
                       const std::vector<double> &U,
                       const std::vector<std::vector<double>> &Qw,
                       int dim,
                       std::vector<std::vector<double>> &Pw,
                       std::vector<double> &Uh,
                       std::vector<double> &error_array,
                       double tol)
{
    const int p = degree;
    if (p < 2)
    {
        throw std::invalid_argument("DegreeReduceCurve requires input degree p >= 2");
    }
    if (dim <= 0)
    {
        throw std::invalid_argument("DegreeReduceCurve: dim must be positive");
    }

    const int n = n_control_points;
    const int expected_u = n + p + 1;
    if (static_cast<int>(U.size()) != expected_u)
    {
        throw std::invalid_argument("DegreeReduceCurve: knot vector length mismatch");
    }
    if (static_cast<int>(Qw.size()) != n)
    {
        throw std::invalid_argument("DegreeReduceCurve: control point count mismatch");
    }
    for (const auto &pt : Qw)
    {
        if (static_cast<int>(pt.size()) != dim)
        {
            throw std::invalid_argument(
                "DegreeReduceCurve: all control points must have length dim");
        }
    }

    const int ph = p - 1;
    const int m = static_cast<int>(U.size()) - 1;

    int mh = ph;
    int kind = ph + 1;
    int r = -1;
    int a = p;
    int b = p + 1;
    int cind = 1;
    int mult = p;

    std::vector<Vec> qw_work(static_cast<std::size_t>(m + 1), Vec(static_cast<std::size_t>(dim)));
    for (int i = 0; i < n; ++i)
    {
        qw_work[static_cast<std::size_t>(i)] = Qw[static_cast<std::size_t>(i)];
    }
    if (n > 0)
    {
        for (int i = n; i <= m; ++i)
        {
            qw_work[static_cast<std::size_t>(i)] = qw_work[static_cast<std::size_t>(n - 1)];
        }
    }

    Pw.assign(static_cast<std::size_t>(n + p), Vec(static_cast<std::size_t>(dim)));
    Pw[0] = qw_work[0];

    const int uh_len = m + ph + 2;
    Uh.assign(static_cast<std::size_t>(uh_len), 0.0);
    for (int i = 0; i <= ph; ++i)
    {
        Uh[static_cast<std::size_t>(i)] = U[0];
    }

    std::vector<Vec> bpts(static_cast<std::size_t>(p + 1), Vec(static_cast<std::size_t>(dim)));
    for (int i = 0; i <= p; ++i)
    {
        bpts[static_cast<std::size_t>(i)] = qw_work[static_cast<std::size_t>(i)];
    }

    std::vector<Vec> next_bpts(static_cast<std::size_t>(p - 1), Vec(static_cast<std::size_t>(dim)));
    std::vector<Vec> rbpts(static_cast<std::size_t>(p + 1), Vec(static_cast<std::size_t>(dim)));
    std::vector<double> alphas(static_cast<std::size_t>(p - 1), 0.0);
    error_array.assign(static_cast<std::size_t>(m + 1), 0.0);

    while (b < m)
    {
        int i = b;
        while (b < m && U[static_cast<std::size_t>(b)] == U[static_cast<std::size_t>(b + 1)])
        {
            ++b;
        }

        mult = b - i + 1;
        mh += mult - 1;

        const int old_r = r;
        r = p - mult;

        int lbz = (old_r > 0) ? ((old_r + 2) / 2) : 1;

        if (r > 0)
        {
            const double numer = U[static_cast<std::size_t>(b)] - U[static_cast<std::size_t>(a)];
            for (int k = p; k >= mult + 1; --k)
            {
                alphas[static_cast<std::size_t>(k - mult - 1)] =
                    numer / (U[static_cast<std::size_t>(a + k)] - U[static_cast<std::size_t>(a)]);
            }

            for (int j = 1; j <= r; ++j)
            {
                const int save = r - j;
                const int s = mult + j;
                for (int k = p; k >= s; --k)
                {
                    const double alpha = alphas[static_cast<std::size_t>(k - s)];
                    bpts[static_cast<std::size_t>(k)] =
                        LinComb(bpts[static_cast<std::size_t>(k)],
                                alpha,
                                bpts[static_cast<std::size_t>(k - 1)],
                                1.0 - alpha);
                }
                next_bpts[static_cast<std::size_t>(save)] = bpts[static_cast<std::size_t>(p)];
            }
        }

        std::vector<Vec> reduced_bpts;
        double max_err = 0.0;
        BezierDegreeReduce(bpts, dim, reduced_bpts, max_err);
        for (int k = 0; k < p; ++k)
        {
            rbpts[static_cast<std::size_t>(k)] = reduced_bpts[static_cast<std::size_t>(k)];
        }

        error_array[static_cast<std::size_t>(a)] += max_err;
        if (error_array[static_cast<std::size_t>(a)] > tol)
        {
            return false;
        }

        if (old_r > 0)
        {
            int first = kind;
            int last = kind;

            for (int k_rem = 0; k_rem < old_r; ++k_rem)
            {
                i = first;
                int j = last;
                int kj = j - kind;

                while (j - i > k_rem)
                {
                    const double alfa =
                        (U[static_cast<std::size_t>(a)] - Uh[static_cast<std::size_t>(i - 1)]) /
                        (U[static_cast<std::size_t>(b)] - Uh[static_cast<std::size_t>(i - 1)]);
                    const double beta =
                        (U[static_cast<std::size_t>(a)] - Uh[static_cast<std::size_t>(j - k_rem - 1)]) /
                        (U[static_cast<std::size_t>(b)] - Uh[static_cast<std::size_t>(j - k_rem - 1)]);

                    Pw[static_cast<std::size_t>(i - 1)] =
                        DivPoint(SubPoint(Pw[static_cast<std::size_t>(i - 1)],
                                          ScalePoint(Pw[static_cast<std::size_t>(i - 2)], 1.0 - alfa)),
                                 alfa);
                    rbpts[static_cast<std::size_t>(kj)] =
                        DivPoint(SubPoint(rbpts[static_cast<std::size_t>(kj)],
                                          ScalePoint(rbpts[static_cast<std::size_t>(kj + 1)], beta)),
                                 1.0 - beta);
                    ++i;
                    --j;
                    --kj;
                }

                double Br = 0.0;
                if (j - i < k_rem)
                {
                    Br = Distance(Pw[static_cast<std::size_t>(i - 2)],
                                  rbpts[static_cast<std::size_t>(kj + 1)]);
                }
                else
                {
                    const double delta =
                        (U[static_cast<std::size_t>(a)] - Uh[static_cast<std::size_t>(i - 1)]) /
                        (U[static_cast<std::size_t>(b)] - Uh[static_cast<std::size_t>(i - 1)]);
                    const Vec A =
                        LinComb(rbpts[static_cast<std::size_t>(kj + 1)],
                                delta,
                                Pw[static_cast<std::size_t>(i - 2)],
                                1.0 - delta);
                    Br = Distance(Pw[static_cast<std::size_t>(i - 1)], A);
                }

                const int K = a + old_r - k_rem;
                const int q = (2 * p - k_rem + 1) / 2;
                const int L = K - q;
                for (int ii = L; ii <= a; ++ii)
                {
                    error_array[static_cast<std::size_t>(ii)] += Br;
                    if (error_array[static_cast<std::size_t>(ii)] > tol)
                    {
                        return false;
                    }
                }

                --first;
                ++last;
            }

            cind = i - 1;
        }

        if (a != p)
        {
            for (int t = 0; t < ph - old_r; ++t)
            {
                Uh[static_cast<std::size_t>(kind)] = U[static_cast<std::size_t>(a)];
                ++kind;
            }
        }

        for (int ii = lbz; ii <= ph; ++ii)
        {
            Pw[static_cast<std::size_t>(cind)] = rbpts[static_cast<std::size_t>(ii)];
            ++cind;
        }

        if (b < m)
        {
            if (r > 0)
            {
                for (int ii = 0; ii < r; ++ii)
                {
                    bpts[static_cast<std::size_t>(ii)] = next_bpts[static_cast<std::size_t>(ii)];
                }
            }
            for (int ii = r; ii <= p; ++ii)
            {
                bpts[static_cast<std::size_t>(ii)] =
                    qw_work[static_cast<std::size_t>(b - p + ii)];
            }
            a = b;
            ++b;
        }
        else
        {
            for (int ii = 0; ii <= ph; ++ii)
            {
                Uh[static_cast<std::size_t>(kind + ii)] = U[static_cast<std::size_t>(b)];
            }
        }
    }

    const int n_active = cind;
    const int uh_active = kind + ph + 1;
    Pw.resize(static_cast<std::size_t>(n_active));
    Uh.resize(static_cast<std::size_t>(uh_active));
    return true;
}
