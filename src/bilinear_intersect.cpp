// Embree user-geometry callbacks for clamped bilinear NURBS/B-spline leaf patches.
//
// Patch layout and corner indexing live in embree/bilinear_patch_geometry.hpp.
// Ray data (origin, direction, tnear, tfar) is single-precision per RTCRay/RTCHit.
//
// Intersection uses a ray-aligned frame (e_z || direction).  For a rational patch,
// each corner carries a 4D homogeneous control point P^H = (wX, wY, wZ, w).  Rather
// than storing 4-vectors explicitly, we form weighted offsets M = w * (P - O) and a
// bilinear weight W(u, v).  The surface point relative to the ray origin is
//   F(u, v) - O = M(u, v) / W(u, v),
// so the hit equations dot(F - O, e_x) = 0 and dot(F - O, e_y) = 0 become the
// bilinear constraints M_x(u, v) = 0 and M_y(u, v) = 0.  Because e_z is a *unit*
// vector, M_z(u, v) / W(u, v) is the physical distance from the ray origin to the
// hit point; Embree's ray parameter t (from P = org + t * dir, with dir generally
// unnormalized) is that distance divided by |dir|.
// Setting w = 1 at every corner recovers the polynomial Cartesian case.

#include "bilinear_intersect.hpp"

#include "embree/bilinear_patch_geometry.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace mfem_raytracing
{
namespace
{

constexpr float kDegeneracyEps = 1e-6f;
constexpr float kIntersectionTol = 1e-5f;

struct Vec3f
{
    float x, y, z;
};

struct Vec2f
{
    float x, y;
};

// Scalar bilinear f(u, v) = a + b*u + c*v + d*u*v.
struct BilinearCoeff
{
    float a, b, c, d;
};

// Relative tolerances derived from the magnitude of this ray/patch's coefficients,
// so thresholds tuned for unit-scale geometry still make sense for patches (or
// scenes) that are much larger or smaller.
struct SolveTolerances
{
    float coeff_degeneracy;   // vs. M-coefficient magnitudes (weight*length units)
    float coeff_intersection; // vs. M-coefficient magnitudes (weight*length units)
    float weight_degeneracy;  // vs. W-coefficient magnitudes (weight units)
    float quad_degeneracy;    // vs. squared M-coefficient magnitudes
};

struct BilinearPatchHitResult
{
    bool hit = false;
    float t = std::numeric_limits<float>::infinity();
    float u = 0.0f;
    float v = 0.0f;
    Vec3f Ng = {0.0f, 0.0f, 0.0f};
};

inline Vec3f operator+(const Vec3f &a, const Vec3f &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3f operator-(const Vec3f &a, const Vec3f &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3f operator*(float s, const Vec3f &a)
{
    return {s * a.x, s * a.y, s * a.z};
}

inline float Dot(const Vec3f &a, const Vec3f &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3f Cross(const Vec3f &a, const Vec3f &b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline float CrossZ(const Vec2f &a, const Vec2f &b)
{
    return a.x * b.y - a.y * b.x;
}

inline BilinearCoeff FromCorners(float f00, float f10, float f01, float f11)
{
    return {f00, f10 - f00, f01 - f00, f11 - f10 - f01 + f00};
}

inline float Eval(const BilinearCoeff &f, float u, float v)
{
    return f.a + f.b * u + f.c * v + f.d * u * v;
}

inline float MaxAbsCoeff(const BilinearCoeff &f)
{
    return std::max(std::max(std::fabs(f.a), std::fabs(f.b)), std::max(std::fabs(f.c), std::fabs(f.d)));
}

inline Vec3f CornerPoint(const BilinearPatchPrimitive &patch, BilinearCorner corner)
{
    const int i = static_cast<int>(corner);
    return {static_cast<float>(patch.control_points[i][0]),
            static_cast<float>(patch.control_points[i][1]),
            static_cast<float>(patch.control_points[i][2])};
}

inline float CornerWeight(const BilinearPatchPrimitive &patch, BilinearCorner corner)
{
    if (!patch.rational)
    {
        return 1.0f;
    }
    const int i = static_cast<int>(corner);
    return static_cast<float>(patch.weights[i]);
}

// dot(w * (P - O), axis) — one component of the weighted offset M at a corner.
inline float WeightedRayScalar(const Vec3f &origin,
                               const Vec3f &axis,
                               const Vec3f &point,
                               float weight)
{
    return Dot(weight * (point - origin), axis);
}

// Build an orthonormal frame with e_z aligned to direction; also returns |direction|
// so callers can convert the frame-relative hit distance into Embree's ray parameter.
inline bool BuildRayFrame(const Vec3f &direction, Vec3f &ex, Vec3f &ey, Vec3f &ez, float &dir_len)
{
    const float dir_norm2 = Dot(direction, direction);
    if (dir_norm2 <= 0.0f)
    {
        return false;
    }
    dir_len = std::sqrt(dir_norm2);
    ez = (1.0f / dir_len) * direction;

    const Vec3f helper = (std::fabs(ez.x) > 0.9f) ? Vec3f{0.0f, 1.0f, 0.0f} : Vec3f{1.0f, 0.0f, 0.0f};
    ex = Cross(ez, helper);
    const float ex_norm2 = Dot(ex, ex);
    if (ex_norm2 <= 0.0f)
    {
        return false;
    }
    ex = (1.0f / std::sqrt(ex_norm2)) * ex;
    ey = Cross(ex, ez);
    return true;
}

// For a fixed u root of M_x = M_y = 0, solve for v and accept if inside the patch.
// `inv_dir_len` converts the frame-relative hit distance (M_z / W) into Embree's
// ray parameter t, since e_z is a unit vector but Embree's dir is not.
inline bool TryCandidate(float u,
                         const BilinearCoeff &mx,
                         const BilinearCoeff &my,
                         const BilinearCoeff &mz,
                         const BilinearCoeff &W,
                         float tnear,
                         float inv_dir_len,
                         const SolveTolerances &tol,
                         float &best_t,
                         float &best_u,
                         float &best_v)
{
    const float denom_x = mx.c + mx.d * u;
    const float denom_y = my.c + my.d * u;

    float v;
    if (std::fabs(denom_x) >= std::fabs(denom_y))
    {
        if (std::fabs(denom_x) < tol.coeff_degeneracy)
        {
            return false;
        }
        v = -(mx.a + mx.b * u) / denom_x;
    }
    else
    {
        if (std::fabs(denom_y) < tol.coeff_degeneracy)
        {
            return false;
        }
        v = -(my.a + my.b * u) / denom_y;
    }

    // Verify both transverse equations (redundant if algebra is exact).
    if (std::fabs(Eval(mx, u, v)) > tol.coeff_intersection || std::fabs(Eval(my, u, v)) > tol.coeff_intersection)
    {
        return false;
    }

    if (u < -kIntersectionTol || u > 1.0f + kIntersectionTol ||
        v < -kIntersectionTol || v > 1.0f + kIntersectionTol)
    {
        return false;
    }

    const float w = Eval(W, u, v);
    if (std::fabs(w) < tol.weight_degeneracy)
    {
        return false;
    }

    const float t = Eval(mz, u, v) * inv_dir_len / w;
    if (t < tnear || t > best_t)
    {
        return false;
    }

    best_t = t;
    best_u = std::min(1.0f, std::max(0.0f, u));
    best_v = std::min(1.0f, std::max(0.0f, v));
    return true;
}

inline bool SolveBilinearHit(const BilinearCoeff &mx,
                             const BilinearCoeff &my,
                             const BilinearCoeff &mz,
                             const BilinearCoeff &W,
                             float tnear,
                             float inv_dir_len,
                             const SolveTolerances &tol,
                             float &best_t,
                             float &best_u,
                             float &best_v)
{
    // Requiring the transverse vectors (a + b*u, c + d*u) [i.e. M(u, 0) and
    // dM/dv, restricted to the u-root] to be parallel — Cross_z(...) = 0 — gives
    // A*u^2 + B*u + C = 0, the same resultant as in the polynomial (Cartesian) case.
    const Vec2f a = {mx.a, my.a};
    const Vec2f b = {mx.b, my.b};
    const Vec2f c = {mx.c, my.c};
    const Vec2f d = {mx.d, my.d};

    const float A = CrossZ(b, d);
    const float B = CrossZ(a, d) + CrossZ(b, c);
    const float C = CrossZ(a, c);

    bool hit_found = false;

    if (std::fabs(A) > tol.quad_degeneracy)
    {
        const float disc = B * B - 4.0f * A * C;
        if (disc >= 0.0f)
        {
            // Numerically stable quadratic formula: avoids catastrophic cancellation
            // in (-B +/- sqrt(disc)) when B^2 >> 4AC.
            const float sqrt_disc = std::sqrt(disc);
            const float q = -0.5f * (B + std::copysign(sqrt_disc, B));
            const float u0 = (q != 0.0f) ? (q / A) : 0.0f;
            const float u1 = (q != 0.0f) ? (C / q) : 0.0f;
            hit_found |= TryCandidate(u0, mx, my, mz, W, tnear, inv_dir_len, tol, best_t, best_u, best_v);
            hit_found |= TryCandidate(u1, mx, my, mz, W, tnear, inv_dir_len, tol, best_t, best_u, best_v);
        }
    }
    else if (std::fabs(B) > tol.quad_degeneracy)
    {
        hit_found |= TryCandidate(-C / B, mx, my, mz, W, tnear, inv_dir_len, tol, best_t, best_u, best_v);
    }

    return hit_found;
}

BilinearPatchHitResult BilinearPatchResult(const BilinearPatchPrimitive &patch,
                                          const Vec3f &ray_origin,
                                          const Vec3f &ray_direction,
                                          float tnear,
                                          float tfar)
{
    BilinearPatchHitResult result;

    // build the ray frame
    Vec3f ex, ey, ez;
    float dir_len;
    if (!BuildRayFrame(ray_direction, ex, ey, ez, dir_len))
    {
        return result;
    }
    const float inv_dir_len = 1.0f / dir_len;
    
    // per-corner weighted ray scalars from homogeneous control points P^H = (wX, wY, wZ, w).
    const auto corner_scalar = [&](BilinearCorner corner, const Vec3f &axis) {
        return WeightedRayScalar(ray_origin, axis, CornerPoint(patch, corner), CornerWeight(patch, corner));
    };

    const BilinearCoeff mx = FromCorners(corner_scalar(BilinearCorner::P00, ex),
                                         corner_scalar(BilinearCorner::P10, ex),
                                         corner_scalar(BilinearCorner::P01, ex),
                                         corner_scalar(BilinearCorner::P11, ex));
    const BilinearCoeff my = FromCorners(corner_scalar(BilinearCorner::P00, ey),
                                         corner_scalar(BilinearCorner::P10, ey),
                                         corner_scalar(BilinearCorner::P01, ey),
                                         corner_scalar(BilinearCorner::P11, ey));
    const BilinearCoeff mz = FromCorners(corner_scalar(BilinearCorner::P00, ez),
                                         corner_scalar(BilinearCorner::P10, ez),
                                         corner_scalar(BilinearCorner::P01, ez),
                                         corner_scalar(BilinearCorner::P11, ez));
    const BilinearCoeff W = FromCorners(CornerWeight(patch, BilinearCorner::P00),
                                        CornerWeight(patch, BilinearCorner::P10),
                                        CornerWeight(patch, BilinearCorner::P01),
                                        CornerWeight(patch, BilinearCorner::P11));

    const float coeff_scale = std::max(std::max(MaxAbsCoeff(mx), MaxAbsCoeff(my)), MaxAbsCoeff(mz));
    const float weight_scale = MaxAbsCoeff(W);

    const float cs = std::max(coeff_scale, std::numeric_limits<float>::min());
    const float ws = std::max(weight_scale, std::numeric_limits<float>::min());

    const SolveTolerances tol{
        kDegeneracyEps * cs,
        kIntersectionTol * cs,
        kDegeneracyEps * ws,
        kDegeneracyEps * cs * cs,
    };

    float best_t = tfar;
    float best_u = 0.0f;
    float best_v = 0.0f;
    if (!SolveBilinearHit(mx, my, mz, W, tnear, inv_dir_len, tol, best_t, best_u, best_v))
    {
        return result;
    }

    // Geometric normal ∝ (W*dM/du - Wu*M) x (W*dM/dv - Wv*M), reconstructed in world
    // space from the ray-frame weighted-offset bilinears.
    const float w_hit = Eval(W, best_u, best_v);
    const float wu_hit = W.b + W.d * best_v;
    const float wv_hit = W.c + W.d * best_u;
    const Vec3f m_hit = Eval(mx, best_u, best_v) * ex + Eval(my, best_u, best_v) * ey + Eval(mz, best_u, best_v) * ez;
    const Vec3f mu_hit = (mx.b + mx.d * best_v) * ex + (my.b + my.d * best_v) * ey + (mz.b + mz.d * best_v) * ez;
    const Vec3f mv_hit = (mx.c + mx.d * best_u) * ex + (my.c + my.d * best_u) * ey + (mz.c + mz.d * best_u) * ez;
    const Vec3f tangent_u = w_hit * mu_hit - wu_hit * m_hit;
    const Vec3f tangent_v = w_hit * mv_hit - wv_hit * m_hit;

    result.hit = true;
    result.t = best_t;
    result.u = best_u;
    result.v = best_v;
    result.Ng = Cross(tangent_u, tangent_v);
    return result;
}

} // namespace

void BilinearPatchBoundsFunc(const RTCBoundsFunctionArguments *args)
{
    const auto *user_data = static_cast<const BilinearPatchGeometryData *>(args->geometryUserPtr);
    const BilinearPatchPrimitive &patch = user_data->prim_ref_buffer[args->primID];

    // Rational surface lies in the convex hull of Cartesian corners — only true
    // for positive weights. A non-positive weight breaks the hull argument and
    // silently produces a wrong (too small) BVH box, so assert the precondition.
    float min_xyz[3] = {+std::numeric_limits<float>::infinity(),
                        +std::numeric_limits<float>::infinity(),
                        +std::numeric_limits<float>::infinity()};
    float max_xyz[3] = {-std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()};

    for (int corner = 0; corner < kBilinearPatchCorners; ++corner)
    {
        assert((!patch.rational || patch.weights[corner] > 0.0) &&
               "BilinearPatchBoundsFunc: convex-hull bound requires positive rational weights");
        for (int axis = 0; axis < 3; ++axis)
        {
            const float value = static_cast<float>(patch.control_points[corner][axis]);
            min_xyz[axis] = std::min(min_xyz[axis], value);
            max_xyz[axis] = std::max(max_xyz[axis], value);
        }
    }

    const float bump = static_cast<float>(user_data->box_bump);
    args->bounds_o->lower_x = min_xyz[0] - bump;
    args->bounds_o->lower_y = min_xyz[1] - bump;
    args->bounds_o->lower_z = min_xyz[2] - bump;
    args->bounds_o->upper_x = max_xyz[0] + bump;
    args->bounds_o->upper_y = max_xyz[1] + bump;
    args->bounds_o->upper_z = max_xyz[2] + bump;
}

void BilinearPatchIntersectionFunc(const RTCIntersectFunctionNArguments *args)
{
    if (!args->valid[0])
    {
        return;
    }
    assert(args->N == 1);

    const auto *user_data = static_cast<const BilinearPatchGeometryData *>(args->geometryUserPtr);
    const BilinearPatchPrimitive &patch = user_data->prim_ref_buffer[args->primID];

    // args->rayhit is RTCRayHitN* (an opaque packet type); reinterpret_cast is the
    // documented way to view it as a single RTCRayHit when N == 1.
    RTCRayHit *rayhit = (RTCRayHit*)args->rayhit;
    RTCRay &ray = rayhit->ray;
    RTCHit &hit = rayhit->hit;

    const Vec3f origin = {ray.org_x, ray.org_y, ray.org_z};
    const Vec3f direction = {ray.dir_x, ray.dir_y, ray.dir_z};

    const BilinearPatchHitResult result =
        BilinearPatchResult(patch, origin, direction, ray.tnear, ray.tfar);
    if (!result.hit)
    {
        return;
    }

    ray.tfar = result.t;
    hit.u = result.u;
    hit.v = result.v;
    hit.geomID = args->geomID;
    hit.primID = args->primID;
    for (unsigned int level = 0; level < RTC_MAX_INSTANCE_LEVEL_COUNT; ++level)
    {
        hit.instID[level] = args->context->instID[level];
    }

    hit.Ng_x = result.Ng.x;
    hit.Ng_y = result.Ng.y;
    hit.Ng_z = result.Ng.z;
}

void BilinearPatchOccludedFunc(const RTCOccludedFunctionNArguments *args)
{
    if (!args->valid[0])
    {
        return;
    }
    assert(args->N == 1);
    const auto *user_data = static_cast<const BilinearPatchGeometryData *>(args->geometryUserPtr);
    const BilinearPatchPrimitive &patch = user_data->prim_ref_buffer[args->primID];

    // args->ray is RTCRayN*; reinterpret as a single RTCRay because N == 1.
    RTCRay *ray = (RTCRay *)args->ray;
    const Vec3f origin = {ray->org_x, ray->org_y, ray->org_z};
    const Vec3f direction = {ray->dir_x, ray->dir_y, ray->dir_z};

    const BilinearPatchHitResult result =
        BilinearPatchResult(patch, origin, direction, ray->tnear, ray->tfar);

    // Embree convention: an occluded ray is marked by setting tfar to -inf.
    if (result.hit)
    {
        ray->tfar = -std::numeric_limits<float>::infinity();
    }
}

} // namespace mfem_raytracing
