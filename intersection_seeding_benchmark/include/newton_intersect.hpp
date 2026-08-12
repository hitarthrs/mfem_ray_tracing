#ifndef SEEDING_BENCHMARK_NEWTON_INTERSECT_HPP
#define SEEDING_BENCHMARK_NEWTON_INTERSECT_HPP

// Newton-Raphson ray / NURBS-surface intersection on the analytic surface.
//
// Residual formulation (same as src/bilinear_intersection.cpp RefineSurfaceRayHit):
// build an orthonormal frame with e_z along the ray, project the miss vector
// r = S(u,v) - O - t D onto (e_x, e_y), and drive that 2D residual to zero with a
// 2x2 Newton step. t = (S - O) . D closes the third equation.
//
// The solver is instrumented for the seeding benchmark: iteration count,
// convergence status, and final residual are reported so a naive seed and a
// bilinear-patch seed can be compared on identical solver settings.

#include "nurbs_surface.hpp"

namespace seeding_benchmark
{

struct Ray
{
    Vec3 origin{{0.0, 0.0, 0.0}};
    Vec3 dir{{0.0, 0.0, 1.0}};  ///< unit direction
    double tmin = 0.0;
    double tmax = 1e30;
};

enum class NewtonStatus
{
    Converged,  ///< residual fell below tolerance
    MaxIter,    ///< exhausted iterations without converging
    Diverged,   ///< singular Jacobian, NaN, or parameter escaped the domain box
};

struct NewtonConfig
{
    int max_iter = 50;
    double residual_tol = 1e-10;  ///< physical distance tolerance on the 2D residual
    /// How far (in units of the domain span) a parameter may roam outside the
    /// valid domain before the solve is declared diverged.
    double escape_margin = 1.0;
    double singular_det = 1e-14;
};

struct NewtonResult
{
    NewtonStatus status = NewtonStatus::Diverged;
    int iterations = 0;
    double residual = 0.0;  ///< final 2D residual norm (physical distance)
    double u = 0.0;
    double v = 0.0;
    double t = 0.0;
    /// True when the solve converged, the final (u, v) is inside the true domain,
    /// and t is inside [tmin, tmax] -- i.e. a genuine on-patch hit.
    bool valid_hit = false;
};

/// Run Newton from the given seed (u0, v0). The surface's domain defines the
/// escape box and the in-domain check for `valid_hit`.
NewtonResult NewtonIntersect(const NurbsSurface &surface,
                             const Ray &ray,
                             double u0,
                             double v0,
                             const NewtonConfig &config);

}  // namespace seeding_benchmark

#endif
