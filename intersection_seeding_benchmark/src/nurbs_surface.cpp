#include "nurbs_surface.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace seeding_benchmark
{

namespace
{

// FindSpan: index of the knot span containing u (Piegl & Tiller A2.1), clamped
// so u exactly at the right domain end lands on the last non-degenerate span.
int FindSpan(int n, int p, double u, const std::vector<double> &U)
{
    if (u >= U[static_cast<std::size_t>(n) + 1])
    {
        return n;
    }
    if (u <= U[static_cast<std::size_t>(p)])
    {
        return p;
    }
    int low = p;
    int high = n + 1;
    int mid = (low + high) / 2;
    while (u < U[static_cast<std::size_t>(mid)] ||
           u >= U[static_cast<std::size_t>(mid) + 1])
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

// DersBasisFuns for the first derivative only (n_ders = 1), Piegl & Tiller A2.3.
// ders[0][0..p] = basis function values, ders[1][0..p] = first derivatives.
void DersBasisFuns(int span,
                   double u,
                   int p,
                   const std::vector<double> &U,
                   std::array<std::vector<double>, 2> &ders)
{
    const int pp = p + 1;
    std::vector<std::vector<double>> ndu(pp, std::vector<double>(pp, 0.0));
    std::vector<double> left(pp, 0.0);
    std::vector<double> right(pp, 0.0);

    ndu[0][0] = 1.0;
    for (int j = 1; j <= p; ++j)
    {
        left[j] = u - U[static_cast<std::size_t>(span + 1 - j)];
        right[j] = U[static_cast<std::size_t>(span + j)] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r)
        {
            ndu[j][r] = right[r + 1] + left[j - r];
            const double temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    for (int j = 0; j <= p; ++j)
    {
        ders[0][static_cast<std::size_t>(j)] = ndu[j][p];
    }

    // First derivatives.
    for (int r = 0; r <= p; ++r)
    {
        int s1 = 0;
        int s2 = 1;
        std::array<std::vector<double>, 2> a{std::vector<double>(pp, 0.0),
                                             std::vector<double>(pp, 0.0)};
        a[0][0] = 1.0;

        const int k = 1;  // derivative order
        double d = 0.0;
        const int rk = r - k;
        const int pk = p - k;
        if (r >= k)
        {
            a[static_cast<std::size_t>(s2)][0] = a[static_cast<std::size_t>(s1)][0] / ndu[pk + 1][rk];
            d = a[static_cast<std::size_t>(s2)][0] * ndu[rk][pk];
        }
        const int j1 = (rk >= -1) ? 1 : -rk;
        const int j2 = (r - 1 <= pk) ? (k - 1) : (p - r);
        for (int j = j1; j <= j2; ++j)
        {
            a[static_cast<std::size_t>(s2)][j] =
                (a[static_cast<std::size_t>(s1)][j] - a[static_cast<std::size_t>(s1)][j - 1]) /
                ndu[pk + 1][rk + j];
            d += a[static_cast<std::size_t>(s2)][j] * ndu[rk + j][pk];
        }
        if (r <= pk)
        {
            a[static_cast<std::size_t>(s2)][k] =
                -a[static_cast<std::size_t>(s1)][k - 1] / ndu[pk + 1][r];
            d += a[static_cast<std::size_t>(s2)][k] * ndu[r][pk];
        }
        ders[1][static_cast<std::size_t>(r)] = d;
        std::swap(s1, s2);
    }

    // Scale first derivatives by p (factorial factor for order 1).
    for (int j = 0; j <= p; ++j)
    {
        ders[1][static_cast<std::size_t>(j)] *= static_cast<double>(p);
    }
}

}  // namespace

NurbsSurface::NurbsSurface(mfem_raytracing::SurfaceData surface)
    : surface_(std::move(surface))
{
    nu_ = surface_.NumControlPointsU();
    nv_ = surface_.NumControlPointsV();
    if (nu_ < surface_.degree_u + 1 || nv_ < surface_.degree_v + 1)
    {
        throw std::invalid_argument("NurbsSurface: too few control points for the degree");
    }
    if (static_cast<int>(surface_.knotvector_u.size()) != nu_ + surface_.degree_u + 1 ||
        static_cast<int>(surface_.knotvector_v.size()) != nv_ + surface_.degree_v + 1)
    {
        throw std::invalid_argument("NurbsSurface: knot vector length inconsistent with net");
    }
}

SurfaceSample NurbsSurface::Evaluate(double u, double v) const
{
    const int pu = surface_.degree_u;
    const int pv = surface_.degree_v;
    const int nu_idx = nu_ - 1;
    const int nv_idx = nv_ - 1;

    const int su = FindSpan(nu_idx, pu, u, surface_.knotvector_u);
    const int sv = FindSpan(nv_idx, pv, v, surface_.knotvector_v);

    std::array<std::vector<double>, 2> du{std::vector<double>(pu + 1, 0.0),
                                          std::vector<double>(pu + 1, 0.0)};
    std::array<std::vector<double>, 2> dv{std::vector<double>(pv + 1, 0.0),
                                          std::vector<double>(pv + 1, 0.0)};
    DersBasisFuns(su, u, pu, surface_.knotvector_u, du);
    DersBasisFuns(sv, v, pv, surface_.knotvector_v, dv);

    const bool rational = surface_.IsRational();

    // Homogeneous accumulators: A = sum N_u N_v w P, w = sum N_u N_v w.
    // (00) value, (10) d/du, (01) d/dv.
    Vec3 A00{{0.0, 0.0, 0.0}}, A10{{0.0, 0.0, 0.0}}, A01{{0.0, 0.0, 0.0}};
    double w00 = 0.0, w10 = 0.0, w01 = 0.0;

    for (int a = 0; a <= pu; ++a)
    {
        const int i = su - pu + a;
        const double Nu = du[0][static_cast<std::size_t>(a)];
        const double dNu = du[1][static_cast<std::size_t>(a)];
        for (int b = 0; b <= pv; ++b)
        {
            const int j = sv - pv + b;
            const double Nv = dv[0][static_cast<std::size_t>(b)];
            const double dNv = dv[1][static_cast<std::size_t>(b)];

            const double w = rational
                                 ? surface_.weights[static_cast<std::size_t>(i)]
                                                   [static_cast<std::size_t>(j)]
                                 : 1.0;
            const auto &P = surface_.control_points[static_cast<std::size_t>(i)]
                                                   [static_cast<std::size_t>(j)];

            const double b00 = Nu * Nv;
            const double b10 = dNu * Nv;
            const double b01 = Nu * dNv;

            for (int c = 0; c < 3; ++c)
            {
                const double wp = w * P[static_cast<std::size_t>(c)];
                A00[static_cast<std::size_t>(c)] += b00 * wp;
                A10[static_cast<std::size_t>(c)] += b10 * wp;
                A01[static_cast<std::size_t>(c)] += b01 * wp;
            }
            w00 += b00 * w;
            w10 += b10 * w;
            w01 += b01 * w;
        }
    }

    SurfaceSample out;
    const double inv_w = 1.0 / w00;
    for (int c = 0; c < 3; ++c)
    {
        const double S = A00[static_cast<std::size_t>(c)] * inv_w;
        out.S[static_cast<std::size_t>(c)] = S;
        // Quotient rule: dS = (A' - w' S) / w.
        out.dSdu[static_cast<std::size_t>(c)] =
            (A10[static_cast<std::size_t>(c)] - w10 * S) * inv_w;
        out.dSdv[static_cast<std::size_t>(c)] =
            (A01[static_cast<std::size_t>(c)] - w01 * S) * inv_w;
    }
    return out;
}

Vec3 NurbsSurface::Point(double u, double v) const
{
    return Evaluate(u, v).S;
}

Vec3 NurbsSurface::Normal(double u, double v) const
{
    const SurfaceSample s = Evaluate(u, v);
    Vec3 n{{s.dSdu[1] * s.dSdv[2] - s.dSdu[2] * s.dSdv[1],
            s.dSdu[2] * s.dSdv[0] - s.dSdu[0] * s.dSdv[2],
            s.dSdu[0] * s.dSdv[1] - s.dSdu[1] * s.dSdv[0]}};
    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len > 1e-300)
    {
        n[0] /= len;
        n[1] /= len;
        n[2] /= len;
    }
    return n;
}

}  // namespace seeding_benchmark
