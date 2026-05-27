#include "bilinear_intersection.hpp"

#include <cmath>

namespace
{

constexpr double kDegeneracyEps = 1e-14;
constexpr double kIntersectionTol = 1e-8;
constexpr double kRefineTol = 1e-12;
constexpr int kRefineMaxIter = 25;

// 2D cross product of u and v in the xy plane.
double CrossZ(const mfem::Vector &u, const mfem::Vector &v)
{
    return u(0) * v(1) - u(1) * v(0);
}

// Build an orthonormal basis with ez aligned to the ray.
void BuildRayFrame(const mfem::Vector &direction, RayFrame &frame)
{
    frame.ex.SetSize(3);
    frame.ey.SetSize(3);
    frame.ez.SetSize(3);
    frame.ez = direction;

    // Pick a helper axis not nearly parallel to the ray.
    mfem::Vector ax(3);
    ax = 0.0;
    if (std::abs(direction(0)) > 0.9) // if the ray is very close to the x-axis, use the y-axis
    {
        ax(1) = 1.0;
    }
    else
    {
        ax(0) = 1.0;
    }

    frame.ez.cross3D(ax, frame.ex);
    const double ex_norm = frame.ex.Norml2();
    MFEM_ASSERT(ex_norm > 0.0, "Failed to build ray-aligned frame.");
    frame.ex /= ex_norm;

    frame.ex.cross3D(frame.ez, frame.ey);
}

// Convert a point from physical space to ray-aligned coordinates.
void ToRaySpace(const mfem::Vector &point,
                const mfem::Vector &origin,
                const RayFrame &frame,
                mfem::Vector &q)
{
    q.SetSize(3);
    mfem::Vector relative(point);
    relative -= origin;

    q(0) = relative(0) * frame.ex(0) + relative(1) * frame.ex(1) + relative(2) * frame.ex(2);
    q(1) = relative(0) * frame.ey(0) + relative(1) * frame.ey(1) + relative(2) * frame.ey(2);
    q(2) = relative(0) * frame.ez(0) + relative(1) * frame.ez(1) + relative(2) * frame.ez(2);
}

// Convert a delta in ray-aligned coordinates to physical space.
void FromRayDelta(const mfem::Vector &dq, const RayFrame &frame, mfem::Vector &dp)
{
    dp.SetSize(3);
    dp(0) = dq(0) * frame.ex(0) + dq(1) * frame.ey(0) + dq(2) * frame.ez(0);
    dp(1) = dq(0) * frame.ex(1) + dq(1) * frame.ey(1) + dq(2) * frame.ez(1);
    dp(2) = dq(0) * frame.ex(2) + dq(1) * frame.ey(2) + dq(2) * frame.ez(2);
}

void ComputeCoefficients(const mfem::Vector &q00,
                         const mfem::Vector &q10,
                         const mfem::Vector &q01,
                         const mfem::Vector &q11,
                         mfem::Vector &a,
                         mfem::Vector &b,
                         mfem::Vector &c,
                         mfem::Vector &d)
{
    // S(u,v) = a + b*u + c*v + d*u*v
    a = q00;
    b = q10;
    b -= q00;
    c = q01;
    c -= q00;
    d = q11;
    d -= q10;
    d -= q01;
    d += q00;
}

double RayParameter(const mfem::Vector &point,
                    const mfem::Vector &origin,
                    const mfem::Vector &direction)
{
    mfem::Vector delta(point);
    delta -= origin;
    return delta(0) * direction(0) + delta(1) * direction(1) + delta(2) * direction(2);
}

void ComputePhysicalNormal(mfem::IsoparametricTransformation &FTr,
                           double u,
                           double v,
                           mfem::Vector &normal)
{
    mfem::IntegrationPoint ip;
    ip.Set3(u, v, 0.0);
    FTr.SetIntPoint(&ip);

    const mfem::DenseMatrix &jac = FTr.Jacobian();
    mfem::Vector su(3), sv(3);
    su(0) = jac(0, 0);
    su(1) = jac(1, 0);
    su(2) = jac(2, 0);
    sv(0) = jac(0, 1);
    sv(1) = jac(1, 1);
    sv(2) = jac(2, 1);
    su.cross3D(sv, normal);

    const double n_norm = normal.Norml2();
    if (n_norm > kDegeneracyEps)
    {
        normal /= n_norm;
    }
}

// Newton refinement on the true boundary map F(u,v): t = dot(F - O, D), D unit.
bool RefineSurfaceRayHit(mfem::IsoparametricTransformation &FTr,
                         const Ray &ray,
                         const RayFrame &frame,
                         double &u,
                         double &v,
                         double &t)
{
    const mfem::Vector &O = ray.GetOrigin();
    const mfem::Vector &D = ray.GetDirection();

    mfem::IntegrationPoint ip;
    mfem::Vector F(3), r(3), dFdu(3), dFdv(3), drdu(3), drdv(3);

    for (int iter = 0; iter < kRefineMaxIter; ++iter)
    {
        ip.Set3(u, v, 0.0);
        FTr.SetIntPoint(&ip);
        FTr.Transform(ip, F);

        t = RayParameter(F, O, D);

        r = F;
        r.Add(-1.0, O);
        r.Add(-t, D);

        const double rx = r(0) * frame.ex(0) + r(1) * frame.ex(1) + r(2) * frame.ex(2);
        const double ry = r(0) * frame.ey(0) + r(1) * frame.ey(1) + r(2) * frame.ey(2);
        const double res_norm = std::hypot(rx, ry);
        if (res_norm < kRefineTol)
        {
            return true;
        }

        const mfem::DenseMatrix &jac = FTr.Jacobian();
        dFdu(0) = jac(0, 0);
        dFdu(1) = jac(1, 0);
        dFdu(2) = jac(2, 0);
        dFdv(0) = jac(0, 1);
        dFdv(1) = jac(1, 1);
        dFdv(2) = jac(2, 1);

        const double dtd_u = dFdu(0) * D(0) + dFdu(1) * D(1) + dFdu(2) * D(2);
        const double dtd_v = dFdv(0) * D(0) + dFdv(1) * D(1) + dFdv(2) * D(2);

        drdu = dFdu;
        drdu.Add(-dtd_u, D);
        drdv = dFdv;
        drdv.Add(-dtd_v, D);

        const double J00 = drdu(0) * frame.ex(0) + drdu(1) * frame.ex(1) + drdu(2) * frame.ex(2);
        const double J01 = drdv(0) * frame.ex(0) + drdv(1) * frame.ex(1) + drdv(2) * frame.ex(2);
        const double J10 = drdu(0) * frame.ey(0) + drdu(1) * frame.ey(1) + drdu(2) * frame.ey(2);
        const double J11 = drdv(0) * frame.ey(0) + drdv(1) * frame.ey(1) + drdv(2) * frame.ey(2);

        const double det = J00 * J11 - J01 * J10;
        if (std::abs(det) < kDegeneracyEps)
        {
            return false;
        }

        u += (-rx * J11 + ry * J01) / det;
        v += (-J00 * ry + J10 * rx) / det;
    }

    return false;
}

void CommitHit(double u,
               double v,
               double t,
               const Ray &ray,
               mfem::IsoparametricTransformation &FTr,
               FaceHitInformation &best)
{
    best.hit = true;
    best.t_intersection = t;
    best.u = u;
    best.v = v;
    best.local_coords.SetSize(3);
    ray.Evaluate(t, best.local_coords);
    ComputePhysicalNormal(FTr, u, v, best.normal);
}

void ComputeNormal(const mfem::Vector &b,
                   const mfem::Vector &c,
                   const mfem::Vector &d,
                   double u,
                   double v,
                   const RayFrame &frame,
                   mfem::Vector &normal)
{
    mfem::Vector b_world(3), c_world(3), d_world(3);
    FromRayDelta(b, frame, b_world);
    FromRayDelta(c, frame, c_world);
    FromRayDelta(d, frame, d_world);

    mfem::Vector su(b_world), sv(c_world), scaled_d(d_world);
    scaled_d *= v;
    su += scaled_d;  // dS/du

    scaled_d = d_world;
    scaled_d *= u;
    sv += scaled_d;  // dS/dv

    su.cross3D(sv, normal);
    const double n_norm = normal.Norml2();
    if (n_norm > kDegeneracyEps)
    {
        normal /= n_norm;
    }
}

bool TryCandidate(double u,
                  const mfem::Vector &a,
                  const mfem::Vector &b,
                  const mfem::Vector &c,
                  const mfem::Vector &d,
                  const Ray &ray,
                  const RayFrame &frame,
                  FaceHitInformation &best)
{
    // Intersection requires S_x = S_y = 0; isolate v from the better-conditioned equation.
    const double denom_x = c(0) + d(0) * u;
    const double denom_y = c(1) + d(1) * u;

    double v = 0.0;
    if (std::abs(denom_x) >= std::abs(denom_y))
    {
        if (std::abs(denom_x) < kDegeneracyEps)
        {
            return false;
        }
        v = -(a(0) + b(0) * u) / denom_x;
    }
    else
    {
        if (std::abs(denom_y) < kDegeneracyEps)
        {
            return false;
        }
        v = -(a(1) + b(1) * u) / denom_y;
    }

    const double sx = a(0) + b(0) * u + c(0) * v + d(0) * u * v;
    const double sy = a(1) + b(1) * u + c(1) * v + d(1) * u * v;
    if (std::abs(sx) > kIntersectionTol || std::abs(sy) > kIntersectionTol)
    {
        return false;
    }
    const double t = a(2) + b(2) * u + c(2) * v + d(2) * u * v;  // ray parameter in aligned frame

    // Face and ray segment trimming.
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
    {
        return false;
    }
    if (t < ray.GetTMin() || t > ray.GetTMax())
    {
        return false;
    }

    if (best.hit && t >= best.t_intersection)
    {
        return false;
    }

    best.hit = true;
    best.t_intersection = t;
    best.u = u;
    best.v = v;
    best.local_coords.SetSize(3);
    ray.Evaluate(t, best.local_coords);
    ComputeNormal(b, c, d, u, v, frame, best.normal);
    return true;
}

void ScanUCandidates(const mfem::Vector &a,
                     const mfem::Vector &b,
                     const mfem::Vector &c,
                     const mfem::Vector &d,
                     const Ray &ray,
                     const RayFrame &frame,
                     FaceHitInformation &best)
{
    constexpr int nu = 256;
    for (int i = 0; i <= nu; ++i)
    {
        TryCandidate(static_cast<double>(i) / nu, a, b, c, d, ray, frame, best);
    }
}

void SolveForRoots(double A,
                   double B,
                   double C,
                   const mfem::Vector &a,
                   const mfem::Vector &b,
                   const mfem::Vector &c,
                   const mfem::Vector &d,
                   const Ray &ray,
                   const RayFrame &frame,
                   FaceHitInformation &best)
{
    // Eliminating v from S_x = S_y = 0 gives A*u^2 + B*u + C = 0.
    if (std::abs(A) > kDegeneracyEps)
    {
        const double disc = B * B - 4.0 * A * C;
        if (disc >= 0.0)
        {
            const double sqrt_disc = std::sqrt(disc);
            TryCandidate((-B + sqrt_disc) / (2.0 * A), a, b, c, d, ray, frame, best);
            TryCandidate((-B - sqrt_disc) / (2.0 * A), a, b, c, d, ray, frame, best);
        }
    }
    else if (std::abs(B) > kDegeneracyEps)
    {
        TryCandidate(-C / B, a, b, c, d, ray, frame, best);
    }

    // Quadratic roots can lie outside [0, 1]; scan u on the face parameter range.
    ScanUCandidates(a, b, c, d, ray, frame, best);
}

void PhysicalBilinearSearch(const Ray &ray,
                            mfem::IsoparametricTransformation &FTr,
                            FaceHitInformation &best)
{
    const mfem::Vector &O = ray.GetOrigin();
    const mfem::Vector &D = ray.GetDirection();

    mfem::IntegrationPoint ip;
    constexpr int n = 256;

    double best_dist = 1e300;
    double seed_u = 0.0;
    double seed_v = 0.0;
    double seed_t = 0.0;

    for (int i = 0; i <= n; ++i)
    {
        for (int j = 0; j <= n; ++j)
        {
            const double u = static_cast<double>(i) / n;
            const double v = static_cast<double>(j) / n;

            ip.Set3(u, v, 0.0);
            FTr.SetIntPoint(&ip);
            mfem::Vector P(3);
            FTr.Transform(ip, P);

            const double t = RayParameter(P, O, D);
            if (t < ray.GetTMin() || t > ray.GetTMax())
            {
                continue;
            }

            mfem::Vector on_ray(3);
            ray.Evaluate(t, on_ray);
            on_ray -= P;
            const double dist = on_ray.Norml2();
            if (dist < best_dist)
            {
                best_dist = dist;
                seed_u = u;
                seed_v = v;
                seed_t = t;
            }
        }
    }

    constexpr double seed_tol = 0.05;
    if (best_dist > seed_tol)
    {
        return;
    }

    CommitHit(seed_u, seed_v, seed_t, ray, FTr, best);
}

}  // namespace

FaceHitInformation BilinearIntersection(const Ray &ray, const mfem::Mesh &mesh, int face_index)
{
    FaceHitInformation result;

    // Step 0: four reference corners of the boundary face in physical space.
    mfem::IsoparametricTransformation FTr;
    mesh.GetBdrElementTransformation(face_index, &FTr);

    mfem::IntegrationPoint ip;
    mfem::Vector P00(3), P10(3), P01(3), P11(3);

    ip.Set3(0.0, 0.0, 0.0);
    FTr.SetIntPoint(&ip);
    FTr.Transform(ip, P00);
    ip.Set3(1.0, 0.0, 0.0);
    FTr.SetIntPoint(&ip);
    FTr.Transform(ip, P10);
    ip.Set3(0.0, 1.0, 0.0);
    FTr.SetIntPoint(&ip);
    FTr.Transform(ip, P01);
    ip.Set3(1.0, 1.0, 0.0);
    FTr.SetIntPoint(&ip);
    FTr.Transform(ip, P11);

    const mfem::Vector &ray_origin = ray.GetOrigin();
    const mfem::Vector &ray_direction = ray.GetDirection();

    RayFrame frame;
    BuildRayFrame(ray_direction, frame);

    // Step 1: express the bilinear patch in ray-aligned coordinates.
    mfem::Vector q00(3), q10(3), q01(3), q11(3);
    ToRaySpace(P00, ray_origin, frame, q00);
    ToRaySpace(P10, ray_origin, frame, q10);
    ToRaySpace(P01, ray_origin, frame, q01);
    ToRaySpace(P11, ray_origin, frame, q11);

    mfem::Vector a(3), b(3), c(3), d(3);
    ComputeCoefficients(q00, q10, q01, q11, a, b, c, d);

    // Step 2: quadratic coefficients from 2D cross products (z-component in ray frame).
    const double A = CrossZ(b, d);
    const double B = CrossZ(a, d) + CrossZ(b, c);
    const double C = CrossZ(a, c);

    SolveForRoots(A, B, C, a, b, c, d, ray, frame, result);

    if (!result.hit)
    {
        PhysicalBilinearSearch(ray, FTr, result);
    }

    if (result.hit)
    {
        double u = result.u;
        double v = result.v;
        double t = result.t_intersection;
        if (RefineSurfaceRayHit(FTr, ray, frame, u, v, t) && u >= 0.0 && u <= 1.0 && v >= 0.0 &&
            v <= 1.0 && t >= ray.GetTMin() && t <= ray.GetTMax())
        {
            CommitHit(u, v, t, ray, FTr, result);
        }
    }

    return result;
}
