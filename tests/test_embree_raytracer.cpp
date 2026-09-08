// Tests for EmbreeRayTracer: scene management, first-hit and occlusion queries
// against bilinear patch user geometries, plus loading the leaf-patch JSON
// fixtures under tests/test-jsons/.

#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "mfem_raytracing/embree/raytracer.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstring>
#include <future>

using namespace mfem_raytracing;

namespace
{

constexpr const char *kLeafJsonPath = "tests/test-jsons/d4_leaf_bboxes.json";

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

BilinearPatchPrimitive MakeFlatPatchRange(double x0, double x1, double y0, double y1, double z_value)
{
    BilinearPatchPrimitive patch;
    auto set_corner = [&](BilinearCorner c, double x, double y) {
        const int i = static_cast<int>(c);
        patch.control_points[i][0] = x;
        patch.control_points[i][1] = y;
        patch.control_points[i][2] = z_value;
    };
    set_corner(BilinearCorner::P00, x0, y0);
    set_corner(BilinearCorner::P10, x1, y0);
    set_corner(BilinearCorner::P01, x0, y1);
    set_corner(BilinearCorner::P11, x1, y1);
    return patch;
}

BilinearPatchPrimitive MakeSaddlePatch()
{
    // z = x*y over [-1, 1]^2. The line x=y, z=0 touches at (0, 0, 0).
    BilinearPatchPrimitive patch;
    const auto set_corner = [&](BilinearCorner c, double x, double y, double z) {
        const int i = static_cast<int>(c);
        patch.control_points[i][0] = x;
        patch.control_points[i][1] = y;
        patch.control_points[i][2] = z;
    };
    set_corner(BilinearCorner::P00, -1.0, -1.0, 1.0);
    set_corner(BilinearCorner::P10, 1.0, -1.0, -1.0);
    set_corner(BilinearCorner::P01, -1.0, 1.0, -1.0);
    set_corner(BilinearCorner::P11, 1.0, 1.0, 1.0);
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

    const std::vector<RayHitRecord> brute_all = tracer.IntersectAllBruteForce(origin, direction);
    CHECK(brute_all.size() == 2);
    CHECK_NEAR(brute_all[0].t, 2.0, 1e-4);
    CHECK_NEAR(brute_all[1].t, 5.0, 1e-4);
}

void TestIntersectAllClustersSharedCoverageButPreservesDistinctWalls()
{
    EmbreeRayTracer tracer;
    // The first two leaves deliberately overlap at near-equal t. The third is
    // farther than the mixed dedupe tolerance and must remain a separate wall.
    tracer.RegisterPatches({MakeFlatPatch(2.0), MakeFlatPatch(2.000005), MakeFlatPatch(2.00005)});
    tracer.CommitScene();

    const double origin[3] = {0.0, 0.0, 0.0};
    const double direction[3] = {0.0, 0.0, 1.0};
    const std::vector<RayHitRecord> all = tracer.IntersectAll(origin, direction);

    CHECK(all.size() == 2);
    CHECK_NEAR(all[0].t, 2.0, 1e-4);
    CHECK_NEAR(all[1].t, 2.00005, 1e-4);
}

void TestSharedEdgeAndNearEdgeRaysAcrossDistances()
{
    EmbreeRayTracer tracer;
    tracer.RegisterPatches({MakeFlatPatchRange(-1.0, 0.0, -1.0, 1.0, 0.0),
                            MakeFlatPatchRange(0.0, 1.0, -1.0, 1.0, 0.0)});
    tracer.CommitScene();

    const double direction[3] = {0.0, 0.0, 1.0};
    const double edge_offsets[] = {-4.0 * std::numeric_limits<float>::epsilon(), 0.0,
                                   4.0 * std::numeric_limits<float>::epsilon()};
    const double distances[] = {1.0, 1000.0, 1.0e6};
    for (const double distance : distances)
    {
        for (const double offset : edge_offsets)
        {
            const double origin[3] = {offset, 0.0, -distance};
            const RayHitRecord embree_hit = tracer.Intersect(origin, direction);
            const RayHitRecord direct_hit = tracer.IntersectBruteForce(origin, direction);
            CHECK(embree_hit.hit);
            CHECK(direct_hit.hit);
            CHECK_NEAR(embree_hit.t, direct_hit.t, 1e-4 * std::max(1.0, distance));

            const std::vector<RayHitRecord> all = tracer.IntersectAll(origin, direction);
            CHECK(all.size() == 1);
            CHECK_NEAR(all[0].t, distance, 1e-4 * std::max(1.0, distance));
        }
    }
}

void TestGrazingTangentAndNearTangentRays()
{
    EmbreeRayTracer tracer;
    tracer.RegisterPatches({MakeSaddlePatch()});
    tracer.CommitScene();

    const double direction[3] = {1.0, 1.0, 0.0};
    const double tangent_origin[3] = {-1.0, -1.0, 0.0};
    const RayHitRecord tangent = tracer.Intersect(tangent_origin, direction);
    const RayHitRecord tangent_direct = tracer.IntersectBruteForce(tangent_origin, direction);
    CHECK(tangent.hit);
    CHECK(tangent_direct.hit);
    CHECK_NEAR(tangent.t, 1.0, 1e-4);
    CHECK_NEAR(tangent.t, tangent_direct.t, 1e-4);
    CHECK(tracer.IntersectAll(tangent_origin, direction).size() == 1);

    // Lift the tangent ray slightly: the quadratic has two separate roots.
    const double near_tangent_origin[3] = {-1.0, -1.0, 1e-4};
    const std::vector<RayHitRecord> near_tangent_all = tracer.IntersectAll(near_tangent_origin, direction);
    CHECK(near_tangent_all.size() == 2);
    CHECK(near_tangent_all[1].t > near_tangent_all[0].t);

    // Lower it: the ray does not reach the saddle.
    const double miss_origin[3] = {-1.0, -1.0, -1e-4};
    CHECK(!tracer.Intersect(miss_origin, direction).hit);
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

void TestCertifiedLeafSceneRegistration()
{
    LeafPatchScene diagnostic;
    diagnostic.declares_rt_certification = true;
    diagnostic.rt_certified = false;
    LeafPatch leaf;
    leaf.patch = MakeFlatPatch(2.0);
    diagnostic.leaves.push_back(leaf);

    bool rejected = false;
    try
    {
        EmbreeRayTracer tracer;
        tracer.RegisterLeafPatchScene(diagnostic);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    CHECK(rejected);

    EmbreeRayTracer tracer;
    tracer.RegisterLeafPatchScene(diagnostic, true);
    tracer.CommitScene();
    CHECK(tracer.PatchCount() == 1);
}

void TestRayQueryDiagnosticsCountCallbackOutcomes()
{
    EmbreeRayTracer tracer;
    tracer.RegisterPatches({MakeFlatPatch(2.0)});
    tracer.CommitScene();

    const double direction[3] = {0.0, 0.0, 1.0};
    RayQueryDiagnostics diagnostics;

    const double hit_origin[3] = {0.0, 0.0, 0.0};
    CHECK(tracer.Intersect(hit_origin, direction, 0.0,
                           std::numeric_limits<double>::infinity(), &diagnostics).hit);
    CHECK(diagnostics.kernel_invocations >= 1);
    CHECK(diagnostics.reported_hits >= 1);
    CHECK(diagnostics.kernel_rejections + diagnostics.reported_hits == diagnostics.kernel_invocations);

    const RayHitRecord brute_hit = tracer.IntersectBruteForce(hit_origin, direction);
    CHECK(brute_hit.hit);
    CHECK_NEAR(brute_hit.t, 2.0, 1e-4);

    // The patch lies beyond this ray segment.  Embree is permitted to invoke
    // user geometry callbacks before their final t-range rejection, so this
    // must be counted as a callback-level rejection rather than assumed to be
    // a BVH miss.
    CHECK(!tracer.Intersect(hit_origin, direction, 0.0, 1.0, &diagnostics).hit);
    CHECK(diagnostics.kernel_invocations >= 1);
    CHECK(diagnostics.kernel_rejections == diagnostics.kernel_invocations);
    CHECK(diagnostics.reported_hits == 0);

    // This lateral query reaches the callback but the solver rejects it, which
    // is the other state the diagnostic output needs to distinguish.
    const double solver_miss_origin[3] = {5.0, 5.0, 0.0};
    CHECK(!tracer.Intersect(solver_miss_origin, direction, 0.0,
                            std::numeric_limits<double>::infinity(), &diagnostics).hit);
    CHECK(diagnostics.kernel_invocations >= 1);
    CHECK(diagnostics.kernel_rejections == diagnostics.kernel_invocations);
    CHECK(diagnostics.reported_hits == 0);
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
        // Older leaf exports did not carry a role; parsing remains backwards-compatible.
        CHECK(leaf.role == "unknown");
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

void CheckSameHit(const RayHitRecord &a, const RayHitRecord &b)
{
    CHECK(a.hit == b.hit);
    if (!a.hit || !b.hit) { return; }
    CHECK(a.t == b.t);
    CHECK(a.u == b.u);
    CHECK(a.v == b.v);
    CHECK(a.geom_id == b.geom_id);
    CHECK(a.prim_id == b.prim_id);
    for (int i = 0; i < 3; ++i) { CHECK(a.Ng[i] == b.Ng[i]); }
}

void CheckSamePatch(const BilinearPatchPrimitive &a, const BilinearPatchPrimitive &b)
{
    CHECK(a.rational == b.rational);
    CHECK(std::memcmp(a.control_points, b.control_points, sizeof(a.control_points)) == 0);
    CHECK(std::memcmp(a.weights, b.weights, sizeof(a.weights)) == 0);
}

void CompareStorageQueries(const std::vector<BilinearPatchPrimitive> &patches)
{
    EmbreeRayTracer compact;
    EmbreeRayTracer dense(PatchStoragePolicy::Dense);
    compact.RegisterPatches(patches);
    dense.RegisterPatches(patches);
    compact.CommitScene();
    dense.CommitScene();
    CHECK(compact.StorageStatistics().indexed_geometries == 1);
    // Include misses, exact grid edges/corners, oblique and unnormalized rays.
    for (int y = -1; y <= 9; ++y)
    {
        for (int x = -1; x <= 9; ++x)
        {
            const double origin[3] = {x * 0.5, y * 0.5, -2.0};
            for (const double dx : {0.0, 0.125})
            {
                const double dir[3] = {dx, 0.0, 2.0};
                CheckSameHit(compact.Intersect(origin, dir), dense.Intersect(origin, dir));
                CheckSameHit(compact.IntersectBruteForce(origin, dir), dense.IntersectBruteForce(origin, dir));
                CHECK(compact.Occluded(origin, dir) == dense.Occluded(origin, dir));
                CHECK(compact.Occluded(origin, dir, 0.0, 0.5) == dense.Occluded(origin, dir, 0.0, 0.5));
                const auto a = compact.IntersectAll(origin, dir);
                const auto b = dense.IntersectAll(origin, dir);
                CHECK(a.size() == b.size());
                for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) { CheckSameHit(a[i], b[i]); }
                const auto aa = compact.IntersectAllBruteForce(origin, dir);
                const auto bb = dense.IntersectAllBruteForce(origin, dir);
                CHECK(aa.size() == bb.size());
                for (std::size_t i = 0; i < std::min(aa.size(), bb.size()); ++i) { CheckSameHit(aa[i], bb[i]); }
            }
        }
    }
    CHECK(compact.StorageStatistics().compatibility_patch_bytes == 0);
}

void TestIndexedStorage()
{
    std::vector<BilinearPatchPrimitive> patches;
    for (int layer = 0; layer < 2; ++layer)
    {
        for (int y = 0; y < 4; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                auto patch = MakeFlatPatchRange(x, x + 1, y, y + 1, layer);
                patch.rational = ((x + y) % 2 == 0);
                for (int c = 0; c < 4; ++c)
                {
                    patch.control_points[c][2] += 0.125 * patch.control_points[c][0] * patch.control_points[c][1];
                    patch.weights[c] = c + 1.0;
                }
                patches.push_back(patch);
            }
        }
    }
    CompareStorageQueries(patches);
    CompareStorageQueries(LoadLeafPatchScene(kLeafJsonPath).Patches());

    EmbreeRayTracer tracer;
    const auto id = tracer.RegisterPatches(patches);
    tracer.CommitScene();
    const auto stats = tracer.StorageStatistics();
    CHECK(stats.indexed_geometries == 1);
    CHECK(stats.buffer_bytes < stats.dense_equivalent_bytes / 2);
    for (std::size_t i = 0; i < patches.size(); ++i)
    {
        BilinearPatchPrimitive copy;
        CHECK(tracer.CopyPatch(id, static_cast<unsigned int>(i), copy));
        CheckSamePatch(copy, patches[i]);
    }
    CHECK(tracer.StorageStatistics().compatibility_patch_bytes == 0);
    const auto *first = tracer.GetPatch(id, 0);
    CHECK(first != nullptr);
    // Concurrent compatibility lookups must produce stable, exact pointers.
    std::vector<std::future<const BilinearPatchPrimitive *>> readers;
    for (int i = 0; i < 8; ++i)
    {
        readers.push_back(std::async(std::launch::async, [&tracer, id]() { return tracer.GetPatch(id, 0); }));
    }
    for (auto &reader : readers) { CHECK(reader.get() == first); }
    for (std::size_t i = 1; i < patches.size(); ++i) { tracer.GetPatch(id, static_cast<unsigned int>(i)); }
    tracer.RegisterPatches({MakeFlatPatch(8.0)});
    tracer.CommitScene();
    CHECK(tracer.GetPatch(id, 0) == first);
    if (first) { CheckSamePatch(*first, patches[0]); }
    BilinearPatchPrimitive copy;
    CHECK(!tracer.CopyPatch(id, 99999, copy));
    CHECK(tracer.GetPatch(id, 99999) == nullptr);
    CHECK(tracer.GetPatch(RTC_INVALID_GEOMETRY_ID, 0) == nullptr);

    // A disconnected patch costs more indexed; AutoIndexed keeps the dense path.
    EmbreeRayTracer small;
    small.RegisterPatches({MakeFlatPatch(0.0)});
    CHECK(small.StorageStatistics().dense_geometries == 1);
    CHECK(small.StorageStatistics().buffer_bytes == sizeof(BilinearPatchPrimitive));
    EmbreeRayTracer empty;
    empty.RegisterPatches({});
    CHECK(empty.PatchCount() == 0);
    CHECK(empty.StorageStatistics().buffer_bytes == 0);

    // No approximate merging: signed zero, sub-float differences and even
    // unused NaN weights survive the encoding exactly. Do not trace NaN data.
    auto special = MakeFlatPatch(0.0);
    special.control_points[0][2] = -0.0;
    special.control_points[1][0] = std::nextafter(special.control_points[0][0], 0.0);
    special.weights[0] = std::numeric_limits<double>::quiet_NaN();
    IndexedBilinearPatchStorage exact({special});
    CheckSamePatch(exact.Load(0), special);
}

void TestCachedFramesAndBvhOptions()
{
    std::vector<BilinearPatchPrimitive> patches;
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            auto patch = MakeFlatPatchRange(x, x + 1, y, y + 1, 1.0);
            patch.rational = true;
            for (int c = 0; c < 4; ++c) { patch.weights[c] = c + 1.0; }
            patches.push_back(patch);
        }
    }
    struct Query { double origin[3]; double direction[3]; };
    const Query queries[] = {
        {{0.37, 0.23, -1.0}, {0.0, 0.0, 1.0}},
        {{0.37, 0.23, -1.0}, {-0.0, 0.0, 1.0}},
        {{0.37, 0.23, -1.0}, {0.0, 0.0, 2.0}},
        {{0.37, 0.23, -1.0}, {0.125, 0.25, 1.0}},
        {{0.37, 0.23, -1.0}, {0.125, 0.25, -1.0}},
        {{0.37, 0.23, 2.0}, {0.0, -0.0, -1.0}},
        {{0.37, 0.23, -1.0}, {0.0, 1.0, 0.0}},
        {{4.37, 0.23, -1.0}, {-1.0, 0.125, 1.0}},
    };
    for (const auto quality : {RTC_BUILD_QUALITY_LOW, RTC_BUILD_QUALITY_MEDIUM, RTC_BUILD_QUALITY_HIGH})
    {
        for (const bool compact : {false, true})
        {
            EmbreeRayTracer tracer(PatchStoragePolicy::AutoIndexed, {quality, compact});
            tracer.RegisterPatches(patches);
            tracer.CommitScene();
            CHECK(tracer.StorageStatistics().indexed_geometries == 1);
            std::vector<RayHitRecord> expected;
            for (const auto &query : queries)
            {
                // Brute force uses the original, uncached dense-value solver.
                const auto direct = tracer.IntersectBruteForce(query.origin, query.direction);
                expected.push_back(direct);
                CheckSameHit(tracer.Intersect(query.origin, query.direction), direct);
                CHECK(tracer.Occluded(query.origin, query.direction) == direct.hit);
            }
            std::vector<std::future<bool>> workers;
            for (std::size_t offset = 0; offset < 4; ++offset)
            {
                workers.push_back(std::async(std::launch::async, [&, offset]() {
                    for (std::size_t step = 0; step < 100; ++step)
                    {
                        const auto index = (offset + step) % expected.size();
                        const auto &q = queries[index];
                        const auto hit = tracer.Intersect(q.origin, q.direction);
                        const auto &ref = expected[index];
                        if (hit.hit != ref.hit || (hit.hit &&
                            (hit.t != ref.t || hit.u != ref.u || hit.v != ref.v ||
                             hit.Ng[0] != ref.Ng[0] || hit.Ng[1] != ref.Ng[1] || hit.Ng[2] != ref.Ng[2])))
                        { return false; }
                    }
                    return true;
                }));
            }
            for (auto &worker : workers) { CHECK(worker.get()); }
        }
    }
    bool rejected = false;
    try { EmbreeRayTracer invalid(PatchStoragePolicy::AutoIndexed, {RTC_BUILD_QUALITY_REFIT, false}); }
    catch (const std::invalid_argument &) { rejected = true; }
    CHECK(rejected);
}

} // namespace

void TestEmbreeRayTracer()
{
    TestIndexedStorage();
    TestCachedFramesAndBvhOptions();
    TestSingleFlatPatchIntersectAndOcclude();
    TestNearestOfTwoGeometriesWins();
    TestIntersectAllClustersSharedCoverageButPreservesDistinctWalls();
    TestSharedEdgeAndNearEdgeRaysAcrossDistances();
    TestGrazingTangentAndNearTangentRays();
    TestUnnormalizedDirectionThroughScene();
    TestCertifiedLeafSceneRegistration();
    TestRayQueryDiagnosticsCountCallbackOutcomes();
    TestLeafSceneLoader();
    TestLeafSceneRayGrid();
}
