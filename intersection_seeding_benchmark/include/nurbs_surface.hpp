#ifndef SEEDING_BENCHMARK_NURBS_SURFACE_HPP
#define SEEDING_BENCHMARK_NURBS_SURFACE_HPP

// Self-contained analytic NURBS/B-spline surface evaluator (point + first
// derivatives) over an mfem_raytracing::SurfaceData control net.
//
// This is the "true" surface the Newton-Raphson ray-surface solver runs on.
// It is deliberately independent of MFEM's per-element IsoparametricTransformation:
// the bilinear leaf approximation carries (u, v) domains in the *whole-patch*
// parameter space (a wall patch spans multiple knot spans in v), so both the
// truth and the seed must live in that same global parameter domain.
//
// Standard clamped-knot de Boor / Cox--de Boor basis evaluation (Piegl & Tiller,
// "The NURBS Book"). Rational surfaces use the homogeneous quotient rule.

#include "mfem_raytracing/reduction/surface_reduction_types.hpp"

#include <array>

namespace seeding_benchmark
{

using Vec3 = std::array<double, 3>;

/// Point and first partial derivatives of a surface at a parameter location.
struct SurfaceSample
{
    Vec3 S{{0.0, 0.0, 0.0}};      ///< S(u, v)
    Vec3 dSdu{{0.0, 0.0, 0.0}};   ///< dS/du
    Vec3 dSdv{{0.0, 0.0, 0.0}};   ///< dS/dv
};

/// Analytic NURBS surface backed by an mfem_raytracing::SurfaceData.
///
/// Parameters (u, v) live in the clamped knot-vector domain
/// [UDomain().first, UDomain().second] x [VDomain().first, VDomain().second].
class NurbsSurface
{
public:
    explicit NurbsSurface(mfem_raytracing::SurfaceData surface);

    /// Evaluate S(u, v) only.
    Vec3 Point(double u, double v) const;

    /// Evaluate S and dS/du, dS/dv.
    SurfaceSample Evaluate(double u, double v) const;

    /// Unit surface normal (dS/du x dS/dv, normalized). Zero vector if degenerate.
    Vec3 Normal(double u, double v) const;

    std::pair<double, double> UDomain() const { return surface_.u_domain; }
    std::pair<double, double> VDomain() const { return surface_.v_domain; }
    double UMid() const { return 0.5 * (surface_.u_domain.first + surface_.u_domain.second); }
    double VMid() const { return 0.5 * (surface_.v_domain.first + surface_.v_domain.second); }

    const mfem_raytracing::SurfaceData &Data() const { return surface_; }

private:
    mfem_raytracing::SurfaceData surface_;
    int nu_ = 0;  ///< number of control points in u
    int nv_ = 0;  ///< number of control points in v
};

}  // namespace seeding_benchmark

#endif
