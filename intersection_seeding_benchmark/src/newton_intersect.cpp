#include "newton_intersect.hpp"

#include <cmath>

namespace seeding_benchmark
{

namespace
{

double Dot(const Vec3 &a, const Vec3 &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 Cross(const Vec3 &a, const Vec3 &b)
{
    return {{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

// Orthonormal frame with e_z along the (unit) ray direction.
struct Frame
{
    Vec3 ex, ey, ez;
};

Frame BuildFrame(const Vec3 &dir)
{
    Frame f;
    f.ez = dir;
    Vec3 ax{{0.0, 0.0, 0.0}};
    if (std::abs(dir[0]) > 0.9)
    {
        ax[1] = 1.0;
    }
    else
    {
        ax[0] = 1.0;
    }
    f.ex = Cross(f.ez, ax);
    const double n = std::sqrt(Dot(f.ex, f.ex));
    f.ex[0] /= n;
    f.ex[1] /= n;
    f.ex[2] /= n;
    f.ey = Cross(f.ex, f.ez);
    return f;
}

}  // namespace

NewtonResult NewtonIntersect(const NurbsSurface &surface,
                             const Ray &ray,
                             double u0,
                             double v0,
                             const NewtonConfig &config)
{
    const Frame frame = BuildFrame(ray.dir);
    const Vec3 &O = ray.origin;
    const Vec3 &D = ray.dir;

    const auto ud = surface.UDomain();
    const auto vd = surface.VDomain();
    const double u_span = ud.second - ud.first;
    const double v_span = vd.second - vd.first;
    const double u_lo = ud.first - config.escape_margin * u_span;
    const double u_hi = ud.second + config.escape_margin * u_span;
    const double v_lo = vd.first - config.escape_margin * v_span;
    const double v_hi = vd.second + config.escape_margin * v_span;

    NewtonResult res;
    double u = u0;
    double v = v0;

    for (int iter = 0; iter < config.max_iter; ++iter)
    {
        if (std::isnan(u) || std::isnan(v) || u < u_lo || u > u_hi || v < v_lo ||
            v > v_hi)
        {
            res.status = NewtonStatus::Diverged;
            res.iterations = iter;
            res.u = u;
            res.v = v;
            return res;
        }

        const SurfaceSample s = surface.Evaluate(u, v);
        Vec3 rel{{s.S[0] - O[0], s.S[1] - O[1], s.S[2] - O[2]}};
        const double t = Dot(rel, D);
        Vec3 r{{rel[0] - t * D[0], rel[1] - t * D[1], rel[2] - t * D[2]}};

        const double rx = Dot(r, frame.ex);
        const double ry = Dot(r, frame.ey);
        const double res_norm = std::hypot(rx, ry);

        res.iterations = iter + 1;
        res.residual = res_norm;
        res.u = u;
        res.v = v;
        res.t = t;

        if (res_norm < config.residual_tol)
        {
            res.status = NewtonStatus::Converged;
            res.valid_hit = (u >= ud.first && u <= ud.second && v >= vd.first &&
                             v <= vd.second && t >= ray.tmin && t <= ray.tmax);
            return res;
        }

        // dr/du = dS/du - (dS/du . D) D, and likewise for v.
        const double dtdu = Dot(s.dSdu, D);
        const double dtdv = Dot(s.dSdv, D);
        Vec3 drdu{{s.dSdu[0] - dtdu * D[0], s.dSdu[1] - dtdu * D[1],
                   s.dSdu[2] - dtdu * D[2]}};
        Vec3 drdv{{s.dSdv[0] - dtdv * D[0], s.dSdv[1] - dtdv * D[1],
                   s.dSdv[2] - dtdv * D[2]}};

        const double J00 = Dot(drdu, frame.ex);
        const double J01 = Dot(drdv, frame.ex);
        const double J10 = Dot(drdu, frame.ey);
        const double J11 = Dot(drdv, frame.ey);

        const double det = J00 * J11 - J01 * J10;
        if (std::abs(det) < config.singular_det)
        {
            res.status = NewtonStatus::Diverged;
            return res;
        }

        // Solve J [du, dv]^T = -[rx, ry].
        u += (-rx * J11 + ry * J01) / det;
        v += (-J00 * ry + J10 * rx) / det;
    }

    res.status = NewtonStatus::MaxIter;
    return res;
}

}  // namespace seeding_benchmark
