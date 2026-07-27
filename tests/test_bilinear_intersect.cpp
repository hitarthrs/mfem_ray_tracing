// Unit tests for the Embree user-geometry callbacks in src/bilinear_intersect.cpp.
//
// These call BilinearPatchIntersectionFunc / BilinearPatchBoundsFunc directly with
// hand-built Embree argument structs — no RTCDevice/RTCScene is needed since the
// callbacks only read/write their POD arguments.

#include "bilinear_intersect.hpp"
#include "embree/bilinear_patch_geometry.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <limits>

using namespace mfem_raytracing;

namespace
{

#if defined(MFEM_RAYTRACING_EMBREE4)
using RTCTestQueryContext = RTCRayQueryContext;
inline void InitTestQueryContext(RTCTestQueryContext &ctx) { rtcInitRayQueryContext(&ctx); }
#else
using RTCTestQueryContext = RTCIntersectContext;
inline void InitTestQueryContext(RTCTestQueryContext &ctx) { rtcInitIntersectContext(&ctx); }
#endif

// A flat, non-rational patch lying in the z = z_value plane, spanning
// [-half_extent, half_extent] in x and y (u in [0,1] -> x, v in [0,1] -> y).
BilinearPatchPrimitive MakeFlatPatch(double z_value, double half_extent = 10.0)
{
    BilinearPatchPrimitive patch;
    auto set_corner = [&](BilinearCorner c, double x, double y) {
        const int i = static_cast<int>(c);
        patch.control_points[i][0] = x;
        patch.control_points[i][1] = y;
        patch.control_points[i][2] = z_value;
    };
    set_corner(BilinearCorner::P00, -half_extent, -half_extent);
    set_corner(BilinearCorner::P10, half_extent, -half_extent);
    set_corner(BilinearCorner::P01, -half_extent, half_extent);
    set_corner(BilinearCorner::P11, half_extent, half_extent);
    patch.rational = false;
    return patch;
}

// Bundles the RTCRayHit + geometry data + query context a single test needs, and
// pre-fills RTCHit fields with sentinels so tests can tell "left untouched" apart
// from "genuinely computed to this value".
struct TestFixture
{
    RTCRayHit rayhit{};
    BilinearPatchGeometryData geometry{};
    RTCTestQueryContext context{};
    int valid_lane = -1; // Embree convention: -1 means the lane is active.

    TestFixture(const BilinearPatchPrimitive &patch,
                float ox, float oy, float oz,
                float dx, float dy, float dz,
                float tnear = 0.0f,
                float tfar = std::numeric_limits<float>::infinity())
    {
        rayhit.ray.org_x = ox;
        rayhit.ray.org_y = oy;
        rayhit.ray.org_z = oz;
        rayhit.ray.dir_x = dx;
        rayhit.ray.dir_y = dy;
        rayhit.ray.dir_z = dz;
        rayhit.ray.tnear = tnear;
        rayhit.ray.tfar = tfar;

        rayhit.hit.Ng_x = -999.0f;
        rayhit.hit.Ng_y = -999.0f;
        rayhit.hit.Ng_z = -999.0f;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

        geometry.prim_ref_buffer = &patch;
        geometry.primitive_count = 1;
        geometry.box_bump = 0.0;

        InitTestQueryContext(context);
    }

    // Runs BilinearPatchIntersectionFunc and reports whether ray.tfar was updated
    // (i.e. a hit was recorded), tagging the ray with the given instance id.
    bool RunIntersect(unsigned int inst_id = RTC_INVALID_GEOMETRY_ID)
    {
        context.instID[0] = inst_id;

        RTCIntersectFunctionNArguments args{};
        args.valid = &valid_lane;
        args.geometryUserPtr = &geometry;
        args.primID = 0;
        args.context = &context;
        args.rayhit = reinterpret_cast<RTCRayHitN *>(&rayhit);
        args.N = 1;
        args.geomID = 7;

        const float tfar_before = rayhit.ray.tfar;
        BilinearPatchIntersectionFunc(&args);
        return rayhit.ray.tfar != tfar_before;
    }
};

// Bug: t was reported as physical distance along the (unit) ray-aligned frame
// instead of Embree's t in P = org + t*dir with dir left unnormalized.
void TestRayParameterUsesUnnormalizedDirectionScale()
{
    // Patch at z = 5, ray direction (0, 0, 2) has length 2, so the physical
    // distance to the patch (5) corresponds to Embree parameter t = 5 / 2 = 2.5.
    BilinearPatchPrimitive patch = MakeFlatPatch(5.0);
    TestFixture t(patch, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f);

    CHECK(t.RunIntersect());
    CHECK_NEAR(t.rayhit.ray.tfar, 2.5, 1e-4);
    CHECK_NEAR(t.rayhit.hit.u, 0.5, 1e-3);
    CHECK_NEAR(t.rayhit.hit.v, 0.5, 1e-3);
}

// Same geometry, but tfar is set between the correct answer (2.5) and the
// pre-fix, unscaled answer (5.0). The old code would compare its unscaled t
// against this tfar and wrongly report a miss.
void TestRayParameterRespectsTFarAfterScaling()
{
    BilinearPatchPrimitive patch = MakeFlatPatch(5.0);
    TestFixture t(patch, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 3.0f);

    CHECK(t.RunIntersect());
    CHECK_NEAR(t.rayhit.ray.tfar, 2.5, 1e-4);
}

// Bug: hit.instID was never copied from the query context.
void TestInstanceIdIsPropagated()
{
    BilinearPatchPrimitive patch = MakeFlatPatch(1.0);
    TestFixture t(patch, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    CHECK(t.RunIntersect(/*inst_id=*/42u));
    CHECK(t.rayhit.hit.instID[0] == 42u);
}

// Bug: hit.Ng_x/y/z were never written, leaving whatever was in the RTCHit.
void TestGeometricNormalIsComputed()
{
    BilinearPatchPrimitive patch = MakeFlatPatch(1.0);
    TestFixture t(patch, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    CHECK(t.RunIntersect());

    // The sentinel (-999) must have been overwritten.
    CHECK(t.rayhit.hit.Ng_x != -999.0f || t.rayhit.hit.Ng_y != -999.0f || t.rayhit.hit.Ng_z != -999.0f);

    // The patch lies in a z = const plane, so the geometric normal must be
    // parallel to the z axis and non-zero.
    CHECK_NEAR(t.rayhit.hit.Ng_x, 0.0, 1e-4);
    CHECK_NEAR(t.rayhit.hit.Ng_y, 0.0, 1e-4);
    CHECK(std::fabs(t.rayhit.hit.Ng_z) > 1e-4f);
}

// Bug: args->valid[0] was never checked, so an inactive ray lane would still be
// processed (and could clobber ray/hit state that the caller expects untouched).
void TestInactiveLaneIsSkipped()
{
    BilinearPatchPrimitive patch = MakeFlatPatch(1.0);
    TestFixture t(patch, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    t.valid_lane = 0; // inactive lane

    CHECK(!t.RunIntersect());
    CHECK(t.rayhit.hit.instID[0] == RTC_INVALID_GEOMETRY_ID);
}

// Regression: transverse solver thresholds must depend on the leaf footprint,
// not the distance along the ray.  Before the split scale, this 0.3-wide flat
// patch at z=1e6 was declared degenerate because mz.a dominated the threshold.
void TestFarSmallPatchIsNotDeclaredDegenerate()
{
    BilinearPatchPrimitive patch = MakeFlatPatch(1.0e6, 0.15);
    TestFixture t(patch, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    CHECK(t.RunIntersect());
    CHECK_NEAR(t.rayhit.ray.tfar, 1.0e6, 1e-2);
    CHECK_NEAR(t.rayhit.hit.u, 0.5, 1e-3);
    CHECK_NEAR(t.rayhit.hit.v, 0.5, 1e-3);
}

void TestDirectDiagnosticReportsDomainRejection()
{
    BilinearPatchPrimitive patch = MakeFlatPatch(1.0, 1.0);
    const float origin[3] = {5.0f, 0.0f, 0.0f};
    const float direction[3] = {0.0f, 0.0f, 1.0f};
    const BilinearPatchRayHit result =
        IntersectBilinearPatchDirect(patch, origin, direction, 0.0f, 10.0f);

    CHECK(!result.hit);
    CHECK((result.reject_reasons & BilinearRejectDomain) != 0);
}

void TestBoundsAreRoundedOutward()
{
    BilinearPatchPrimitive patch = MakeFlatPatch(2.0, 1.0);
    BilinearPatchGeometryData geometry{};
    geometry.prim_ref_buffer = &patch;
    geometry.primitive_count = 1;
    geometry.box_bump = 0.0;
    RTCBounds bounds{};
    RTCBoundsFunctionArguments args{};
    args.geometryUserPtr = &geometry;
    args.primID = 0;
    args.bounds_o = &bounds;

    BilinearPatchBoundsFunc(&args);

    CHECK(bounds.lower_x < -1.0f);
    CHECK(bounds.lower_y < -1.0f);
    CHECK(bounds.lower_z < 2.0f);
    CHECK(bounds.upper_x > 1.0f);
    CHECK(bounds.upper_y > 1.0f);
    CHECK(bounds.upper_z > 2.0f);
}

} // namespace

void TestBilinearIntersect()
{
    TestRayParameterUsesUnnormalizedDirectionScale();
    TestRayParameterRespectsTFarAfterScaling();
    TestInstanceIdIsPropagated();
    TestGeometricNormalIsComputed();
    TestInactiveLaneIsSkipped();
    TestFarSmallPatchIsNotDeclaredDegenerate();
    TestDirectDiagnosticReportsDomainRejection();
    TestBoundsAreRoundedOutward();
}
