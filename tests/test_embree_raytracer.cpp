// Tests for EmbreeRayTracer: scene management, first-hit and occlusion queries
// against bilinear patch user geometries, plus loading the leaf-patch JSON
// exported by python_experiments/multiple_step_degree_reduction_surfaces.

#include "embree/leaf_patch_loader.hpp"
#include "embree/raytracer.hpp"
#include "test_helpers.hpp"

#include <cmath>

using namespace mfem_raytracing;

namespace
{

constexpr const char *kLeafJsonPath =
    "python_experiments/multiple_step_degree_reduction_surfaces/outputs/d4_leaf_bboxes.json";

BilinearPatchPrimitive MakeFlatPatch(double z_value, double half_extent = 1.0)
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
    return patch;
}

void TestSingleFlatPatchIntersectAndOcclude()
{
    EmbreeRayTracer tracer;
    const unsigned int geom_id = tracer.RegisterPatches({MakeFlatPatch(2.0)});
    tracer.CommitScene();
    CHECK(tracer.PatchCount() == 1);

    const double origin[3] = {0.25, -0.5, 0.0};
    const double direction[3] = {0.0, 0.0, 1.0};

    const RayHitRecord hit = tracer.Intersect(origin, direction);
    CHECK(hit.hit);
    CHECK_NEAR(hit.t, 2.0, 1e-4);
    CHECK(hit.geom_id == geom_id);
    CHECK(hit.prim_id == 0);
    // u maps x in [-1, 1], v maps y in [-1, 1].
    CHECK_NEAR(hit.u, 0.625, 1e-3);
    CHECK_NEAR(hit.v, 0.25, 1e-3);
    // Flat z-plane: normal parallel to z.
    CHECK_NEAR(hit.Ng[0], 0.0, 1e-4);
    CHECK_NEAR(hit.Ng[1], 0.0, 1e-4);
    CHECK(std::fabs(hit.Ng[2]) > 1e-4);
    CHECK(tracer.GetPatch(hit.geom_id, hit.prim_id) != nullptr);

    CHECK(tracer.Occluded(origin, direction));
    // Segment ends before the patch: not occluded.
    CHECK(!tracer.Occluded(origin, direction, 0.0, 1.5));
    // Ray starts past the patch: not occluded.
    CHECK(!tracer.Occluded(origin, direction, 2.5));

    // Ray misses the patch laterally.
    const double outside[3] = {5.0, 5.0, 0.0};
    CHECK(!tracer.Intersect(outside, direction).hit);
    CHECK(!tracer.Occluded(outside, direction));
}

void TestNearestOfTwoGeometriesWins()
{
    EmbreeRayTracer tracer;
    const unsigned int far_id = tracer.RegisterPatches({MakeFlatPatch(5.0)});
    const unsigned int near_id = tracer.RegisterPatches({MakeFlatPatch(2.0)});
    tracer.CommitScene();
    CHECK(tracer.PatchCount() == 2);
    CHECK(far_id != near_id);

    const double origin[3] = {0.0, 0.0, 0.0};
    const double direction[3] = {0.0, 0.0, 1.0};
    const RayHitRecord hit = tracer.Intersect(origin, direction);
    CHECK(hit.hit);
    CHECK(hit.geom_id == near_id);
    CHECK_NEAR(hit.t, 2.0, 1e-4);

    // Restrict the segment to start beyond the near patch: far patch is hit.
    const RayHitRecord far_hit = tracer.Intersect(origin, direction, 3.0);
    CHECK(far_hit.hit);
    CHECK(far_hit.geom_id == far_id);
    CHECK_NEAR(far_hit.t, 5.0, 1e-4);

    // Continuing past the first hit must report both surfaces.
    const std::vector<RayHitRecord> all = tracer.IntersectAll(origin, direction);
    CHECK(all.size() == 2);
    CHECK(all[0].geom_id == near_id);
    CHECK_NEAR(all[0].t, 2.0, 1e-4);
    CHECK(all[1].geom_id == far_id);
    CHECK_NEAR(all[1].t, 5.0, 1e-4);
}

void TestUnnormalizedDirectionThroughScene()
{
    EmbreeRayTracer tracer;
    tracer.RegisterPatches({MakeFlatPatch(4.0)});
    tracer.CommitScene();

    const double origin[3] = {0.0, 0.0, 0.0};
    const double direction[3] = {0.0, 0.0, 8.0}; // |dir| = 8, patch at distance 4
    const RayHitRecord hit = tracer.Intersect(origin, direction);
    CHECK(hit.hit);
    CHECK_NEAR(hit.t, 0.5, 1e-4);
}

void TestLeafSceneLoader()
{
    const LeafPatchScene scene = LoadLeafPatchScene(kLeafJsonPath);
    CHECK(scene.surface_name == "d4_s_shaped_peak_saddle");
    // Exact leaf count depends on the Python export; just require a real scene.
    CHECK(scene.leaves.size() >= 10);

    // Every leaf's stored bbox must equal the AABB of its (non-rational)
    // control net — validates control-point parsing end to end.
    for (const LeafPatch &leaf : scene.leaves)
    {
        CHECK(!leaf.patch.rational);
        for (int axis = 0; axis < 3; ++axis)
        {
            double lo = leaf.patch.control_points[0][axis];
            double hi = lo;
            for (int corner = 1; corner < kBilinearPatchCorners; ++corner)
            {
                lo = std::min(lo, leaf.patch.control_points[corner][axis]);
                hi = std::max(hi, leaf.patch.control_points[corner][axis]);
            }
            CHECK_NEAR(lo, leaf.bbox.min[axis], 1e-9);
            CHECK_NEAR(hi, leaf.bbox.max[axis], 1e-9);
        }
    }
}

void TestLeafSceneRayGrid()
{
    const LeafPatchScene scene = LoadLeafPatchScene(kLeafJsonPath);

    EmbreeRayTracer tracer;
    tracer.RegisterPatches(scene.Patches());
    tracer.CommitScene();
    CHECK(tracer.PatchCount() == scene.leaves.size());

    const double x0 = scene.scene_bbox.min[0], x1 = scene.scene_bbox.max[0];
    const double y0 = scene.scene_bbox.min[1], y1 = scene.scene_bbox.max[1];
    const double z_top = scene.scene_bbox.max[2] + 1.0;
    const double z_span = (z_top - scene.scene_bbox.min[2]) + 1.0;

    const int n = 64;
    int hits = 0;
    for (int iy = 0; iy < n; ++iy)
    {
        for (int ix = 0; ix < n; ++ix)
        {
            // Sample strictly inside the domain to avoid boundary-seam cases.
            const double fx = (ix + 0.5) / n;
            const double fy = (iy + 0.5) / n;
            const double origin[3] = {x0 + fx * (x1 - x0), y0 + fy * (y1 - y0), z_top};
            const double direction[3] = {0.0, 0.0, -1.0};

            const RayHitRecord hit = tracer.Intersect(origin, direction, 0.0, z_span);
            const bool occluded = tracer.Occluded(origin, direction, 0.0, z_span);
            CHECK(hit.hit == occluded);
            if (!hit.hit)
            {
                continue;
            }
            ++hits;

            // The hit leaf's bbox (padded by a small tolerance) must contain
            // the hit point.
            const LeafPatch &leaf = scene.leaves[hit.prim_id];
            const double point[3] = {origin[0], origin[1], origin[2] - hit.t};
            const double tol = 1e-3;
            for (int axis = 0; axis < 3; ++axis)
            {
                CHECK(point[axis] >= leaf.bbox.min[axis] - tol);
                CHECK(point[axis] <= leaf.bbox.max[axis] + tol);
            }

            // Patch-local uv must be in [0, 1]^2 (clamped by the intersector).
            CHECK(hit.u >= 0.0 && hit.u <= 1.0);
            CHECK(hit.v >= 0.0 && hit.v <= 1.0);
        }
    }

    // The surface covers the whole xy footprint of the scene bbox, so nearly
    // every interior ray must find it (leaf seams may drop a measure-zero few).
    CHECK(hits > n * n * 95 / 100);
}

} // namespace

void TestEmbreeRayTracer()
{
    TestSingleFlatPatchIntersectAndOcclude();
    TestNearestOfTwoGeometriesWins();
    TestUnnormalizedDirectionThroughScene();
    TestLeafSceneLoader();
    TestLeafSceneRayGrid();
}
