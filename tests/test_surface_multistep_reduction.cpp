/**
 * @file test_surface_multistep_reduction.cpp
 * @brief Multi-step surface reduction + bilinear leaf extraction (Algorithm 1).
 *
 * The end-to-end golden fixture is the Python pipeline export
 *   python_experiments/multiple_step_degree_reduction_surfaces/outputs/d4_leaf_bboxes.json
 * regenerated via
 *   .venv/bin/python -m multiple_step_degree_reduction_surfaces.demo_leaf_bboxes \
 *       -s d4_s_shaped_peak_saddle --backend approach_4 --max-error 1.0 \
 *       --json multiple_step_degree_reduction_surfaces/outputs/d4_leaf_bboxes.json
 */

#include "bilinear_leaf_extraction.hpp"
#include "embree/leaf_patch_loader.hpp"
#include "surface_conforming_reduction.hpp"
#include "surface_golden_cases.hpp"
#include "surface_multistep_reduction.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

namespace
{

#include "generated_surface_degree_reduction_cases.inc"

constexpr const char *kGoldenLeafJson =
    "python_experiments/multiple_step_degree_reduction_surfaces/outputs/d4_leaf_bboxes.json";

mfem_raytracing::SurfaceData PeakSaddleInput()
{
    return MakeSurfaceData(SurfaceApproach1SShapedPeakSaddle().input);
}

mfem_raytracing::SurfaceData SemicircleInput()
{
    return MakeSurfaceData(SurfaceApproach1SemicirclePlateauShell().input);
}

double LeafCoverage(const mfem_raytracing::MultipleStepSurfaceReductionResult &result)
{
    double area = 0.0;
    for (const auto &leaf : result.segments)
    {
        area += (leaf.u_domain_global.second - leaf.u_domain_global.first) *
                (leaf.v_domain_global.second - leaf.v_domain_global.first);
    }
    return area;
}

void TestZeroStepsReturnsInput()
{
    const mfem_raytracing::SurfaceData input = PeakSaddleInput();
    const auto result = mfem_raytracing::DegreeReduceMultipleStepsNonConforming(input, 0, 1.0);
    CHECK(static_cast<int>(result.segments.size()) == 1);
    const auto &leaf = result.segments.front();
    CHECK(leaf.surface.degree_u == input.degree_u);
    CHECK(leaf.surface.degree_v == input.degree_v);
    CHECK_NEAR(leaf.total_error, 0.0, 0.0);
    CHECK(leaf.steps_taken == 0);
    CHECK_NEAR(leaf.u_domain_global.first, input.u_domain.first, 0.0);
    CHECK_NEAR(leaf.u_domain_global.second, input.u_domain.second, 0.0);
}

void TestInvalidArgumentsThrow()
{
    const mfem_raytracing::SurfaceData input = PeakSaddleInput();

    bool threw = false;
    try
    {
        mfem_raytracing::DegreeReduceMultipleStepsNonConforming(input, -1, 1.0);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try
    {
        mfem_raytracing::DegreeReduceMultipleStepsNonConforming(input, 1, -0.5);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);

    // degree 4 allows at most 3 steps
    threw = false;
    try
    {
        mfem_raytracing::DegreeReduceMultipleStepsNonConforming(input, 4, 1.0);
    }
    catch (const mfem_raytracing::MultipleStepSurfaceReductionFailure &)
    {
        threw = true;
    }
    CHECK(threw);
}

void TestBudgetRespectedAndTargetDegreesReached()
{
    const mfem_raytracing::SurfaceData input = PeakSaddleInput();
    const double max_error = 1.0;

    mfem_raytracing::MultiStepSurfaceReductionOptions options;
    options.budget_policy = mfem_raytracing::ErrorBudgetPolicy::WeightedLate;
    const auto result =
        mfem_raytracing::DegreeReduceMultipleStepsNonConforming(input, 3, max_error, options);

    CHECK(!result.segments.empty());
    for (const auto &leaf : result.segments)
    {
        CHECK(leaf.surface.degree_u == 1);
        CHECK(leaf.surface.degree_v == 1);
        CHECK(leaf.steps_taken == 3);
        CHECK(leaf.total_error <= max_error);
        CHECK(leaf.u_domain_global.first >= input.u_domain.first - 1e-12);
        CHECK(leaf.u_domain_global.second <= input.u_domain.second + 1e-12);
        CHECK(leaf.u_domain_global.first < leaf.u_domain_global.second);
        CHECK(leaf.v_domain_global.first < leaf.v_domain_global.second);
    }
}

// sum accounting + Eq. 5.30 (the new defaults) must both refine more than the
// legacy max/homogeneous path, and never leave holes in the parameter domain.
void TestErrorCombinationAndRationalCorrection()
{
    // Polynomial surface: 5.30 is a no-op, but "sum" must split at least as much
    // as "max" (it reserves budget for the v-pass).
    {
        const mfem_raytracing::SurfaceData input = PeakSaddleInput();
        mfem_raytracing::MultiStepSurfaceReductionOptions sum_opts;
        sum_opts.budget_policy = mfem_raytracing::ErrorBudgetPolicy::WeightedLate;
        sum_opts.single_step.error_combination = mfem_raytracing::SurfaceErrorCombination::Sum;

        mfem_raytracing::MultiStepSurfaceReductionOptions max_opts = sum_opts;
        max_opts.single_step.error_combination = mfem_raytracing::SurfaceErrorCombination::Max;

        const auto sum_res = mfem_raytracing::DegreeReduceMultipleStepsNonConforming(input, 3, 0.3, sum_opts);
        const auto max_res = mfem_raytracing::DegreeReduceMultipleStepsNonConforming(input, 3, 0.3, max_opts);
        CHECK(sum_res.segments.size() >= max_res.segments.size());
        CHECK_NEAR(LeafCoverage(sum_res), 1.0, 1e-9);
        CHECK_NEAR(LeafCoverage(max_res), 1.0, 1e-9);
    }

    // Rational surface: enabling Eq. 5.30 tightens the effective tolerance and
    // must produce strictly more leaves, still with full coverage.
    {
        const mfem_raytracing::SurfaceData input = SemicircleInput();
        mfem_raytracing::MultiStepSurfaceReductionOptions corrected;
        corrected.budget_policy = mfem_raytracing::ErrorBudgetPolicy::WeightedLate;
        corrected.single_step.rational_tol_correction = true;

        mfem_raytracing::MultiStepSurfaceReductionOptions uncorrected = corrected;
        uncorrected.single_step.rational_tol_correction = false;

        const auto with_530 = mfem_raytracing::DegreeReduceMultipleStepsNonConforming(input, 3, 2.0, corrected);
        const auto without = mfem_raytracing::DegreeReduceMultipleStepsNonConforming(input, 3, 2.0, uncorrected);
        CHECK(with_530.segments.size() > without.segments.size());
        CHECK_NEAR(LeafCoverage(with_530), 1.0, 1e-9);
        CHECK_NEAR(LeafCoverage(without), 1.0, 1e-9);
    }
}

void TestBackendPolicyMapping()
{
    using mfem_raytracing::BudgetPolicyFromBackendName;
    using mfem_raytracing::ErrorBudgetPolicy;
    CHECK(BudgetPolicyFromBackendName("approach_1") == ErrorBudgetPolicy::Cumulative);
    CHECK(BudgetPolicyFromBackendName("approach_3") == ErrorBudgetPolicy::EqualPerStep);
    CHECK(BudgetPolicyFromBackendName("approach_4") == ErrorBudgetPolicy::WeightedLate);
    CHECK(BudgetPolicyFromBackendName("approach_5") == ErrorBudgetPolicy::Geometric);

    bool threw = false;
    try
    {
        BudgetPolicyFromBackendName("approach_2");
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);
}

void TestBilinearControlNetAABB()
{
    mfem_raytracing::SurfaceData patch;
    patch.degree_u = 1;
    patch.degree_v = 1;
    patch.dim = 3;
    patch.control_points = {{{0.0, 0.0, 2.0}, {0.0, 1.0, -1.0}},
                            {{3.0, 0.0, 0.5}, {3.0, 1.0, 4.0}}};
    patch.knotvector_u = {0.0, 0.0, 1.0, 1.0};
    patch.knotvector_v = {0.0, 0.0, 1.0, 1.0};

    const auto bbox = mfem_raytracing::BilinearControlNetAABB(patch);
    CHECK_NEAR(bbox.min[0], 0.0, 0.0);
    CHECK_NEAR(bbox.min[1], 0.0, 0.0);
    CHECK_NEAR(bbox.min[2], -1.0, 0.0);
    CHECK_NEAR(bbox.max[0], 3.0, 0.0);
    CHECK_NEAR(bbox.max[1], 1.0, 0.0);
    CHECK_NEAR(bbox.max[2], 4.0, 0.0);
}

void TestWeightClamping()
{
    std::vector<std::vector<double>> weights = {{1.0, -0.25}, {0.5, 2.0}};
    CHECK(!mfem_raytracing::WeightsAreNonNegative(weights));
    CHECK(mfem_raytracing::ClampWeightsNonNegative(weights));
    CHECK(mfem_raytracing::WeightsAreNonNegative(weights));
    CHECK_NEAR(weights[0][1], 0.0, 0.0);
    CHECK_NEAR(weights[1][0], 0.5, 0.0);
    // idempotent on clean weights
    CHECK(!mfem_raytracing::ClampWeightsNonNegative(weights));
}

// End-to-end: C++ pipeline must reproduce the Python golden leaf JSON.
void TestGoldenLeafBBoxes()
{
    mfem_raytracing::LeafPatchScene golden;
    try
    {
        golden = mfem_raytracing::LoadLeafPatchScene(kGoldenLeafJson);
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL cannot load golden JSON: " << e.what() << std::endl;
        ++g_failures;
        return;
    }

    const mfem_raytracing::SurfaceData input = PeakSaddleInput();
    mfem_raytracing::BilinearLeafReductionOptions options;
    options.backend_name = "approach_4";
    const auto collection =
        mfem_raytracing::ReduceSurfaceToBilinearLeaves(input, 3, 1.0, options);

    CHECK(collection.leaves.size() == golden.leaves.size());
    if (collection.leaves.size() != golden.leaves.size())
    {
        return;
    }

    const auto scene = collection.SceneBBox();
    for (int axis = 0; axis < 3; ++axis)
    {
        CHECK_NEAR(scene.min[axis], golden.scene_bbox.min[axis], 1e-8);
        CHECK_NEAR(scene.max[axis], golden.scene_bbox.max[axis], 1e-8);
    }

    const int corner_order[2][2] = {
        {static_cast<int>(mfem_raytracing::BilinearCorner::P00),
         static_cast<int>(mfem_raytracing::BilinearCorner::P01)},
        {static_cast<int>(mfem_raytracing::BilinearCorner::P10),
         static_cast<int>(mfem_raytracing::BilinearCorner::P11)},
    };

    for (std::size_t k = 0; k < collection.leaves.size(); ++k)
    {
        const auto &got = collection.leaves[k];
        const auto &expected = golden.leaves[k];

        CHECK(got.index == expected.index);
        CHECK_NEAR(got.total_error, expected.total_error, 1e-8);
        CHECK_NEAR(got.u_domain_global.first, expected.u_domain_global[0], 1e-10);
        CHECK_NEAR(got.u_domain_global.second, expected.u_domain_global[1], 1e-10);
        CHECK_NEAR(got.v_domain_global.first, expected.v_domain_global[0], 1e-10);
        CHECK_NEAR(got.v_domain_global.second, expected.v_domain_global[1], 1e-10);
        for (int axis = 0; axis < 3; ++axis)
        {
            CHECK_NEAR(got.bbox.min[axis], expected.bbox.min[axis], 1e-8);
            CHECK_NEAR(got.bbox.max[axis], expected.bbox.max[axis], 1e-8);
        }
        CHECK(got.surface.IsRational() == expected.patch.rational);
        // control points: SurfaceData [i=u][j=v] vs loader's BilinearCorner layout
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                const auto &got_cp = got.surface.control_points[static_cast<std::size_t>(i)]
                                                               [static_cast<std::size_t>(j)];
                const double *exp_cp = expected.patch.control_points[corner_order[i][j]];
                for (int d = 0; d < 3; ++d)
                {
                    CHECK_NEAR(got_cp[static_cast<std::size_t>(d)], exp_cp[d], 1e-8);
                }
            }
        }
    }
}

// The C++ JSON writer output must be loadable by LoadLeafPatchScene and
// round-trip the same geometry (the "works with existing Embree tooling" contract).
void TestJsonWriterRoundTrip()
{
    const mfem_raytracing::SurfaceData input = PeakSaddleInput();
    const auto collection = mfem_raytracing::ReduceSurfaceToBilinearLeaves(input, 3, 1.0);

    const std::string path = "build_test_leaf_bboxes_roundtrip.json";
    {
        std::ofstream out(path);
        CHECK(static_cast<bool>(out));
        mfem_raytracing::WriteLeafBBoxJson(out, collection, "d4_s_shaped_peak_saddle");
    }

    mfem_raytracing::LeafPatchScene reloaded;
    try
    {
        reloaded = mfem_raytracing::LoadLeafPatchScene(path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL JSON round-trip load: " << e.what() << std::endl;
        ++g_failures;
        std::remove(path.c_str());
        return;
    }

    CHECK(reloaded.surface_name == "d4_s_shaped_peak_saddle");
    CHECK(reloaded.leaves.size() == collection.leaves.size());
    if (reloaded.leaves.size() == collection.leaves.size())
    {
        for (std::size_t k = 0; k < collection.leaves.size(); ++k)
        {
            CHECK_NEAR(reloaded.leaves[k].total_error, collection.leaves[k].total_error, 1e-12);
            for (int axis = 0; axis < 3; ++axis)
            {
                CHECK_NEAR(reloaded.leaves[k].bbox.min[axis],
                           collection.leaves[k].bbox.min[axis],
                           1e-12);
                CHECK_NEAR(reloaded.leaves[k].bbox.max[axis],
                           collection.leaves[k].bbox.max[axis],
                           1e-12);
            }
        }
    }
    std::remove(path.c_str());
}

// The conforming (watertight) driver: full coverage, bilinear leaves, and every
// shared grid vertex bitwise identical across all leaves touching it (which,
// for straight-edged bilinear patches, is exactly C0 watertightness).
void CheckConformingResult(const mfem_raytracing::MultipleStepSurfaceReductionResult &result)
{
    CHECK(!result.segments.empty());
    double area = 0.0;
    std::map<std::pair<long long, long long>, std::vector<double>> corner_map;
    int corner_mismatches = 0;
    for (const auto &leaf : result.segments)
    {
        CHECK(leaf.surface.degree_u == 1);
        CHECK(leaf.surface.degree_v == 1);
        area += (leaf.u_domain_global.second - leaf.u_domain_global.first) *
                (leaf.v_domain_global.second - leaf.v_domain_global.first);
        const double us[2] = {leaf.u_domain_global.first, leaf.u_domain_global.second};
        const double vs[2] = {leaf.v_domain_global.first, leaf.v_domain_global.second};
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                const auto key = std::make_pair(std::llround(us[i] * 1e9),
                                                std::llround(vs[j] * 1e9));
                const auto &cp = leaf.surface.control_points[static_cast<std::size_t>(i)]
                                                            [static_cast<std::size_t>(j)];
                auto it = corner_map.find(key);
                if (it == corner_map.end())
                {
                    corner_map.emplace(key, cp);
                }
                else if (it->second != cp) // bitwise equality required
                {
                    ++corner_mismatches;
                }
            }
        }
    }
    CHECK_NEAR(area, 1.0, 1e-9);
    CHECK(corner_mismatches == 0);
}

/// Piegl–Tiller 9-CP rational unit circle in the xy-plane, scaled to `radius`
/// and lifted to height `z`. If `radius < 0`, reflect through the z-axis.
void FillHorizontalCircle(double radius, double z, std::vector<std::vector<double>> &cps_out,
                          std::vector<double> &w_out)
{
    const double r = std::fabs(radius);
    const double s = (radius < 0.0) ? -1.0 : 1.0;
    const double w = std::sqrt(2.0) / 2.0;
    const double xy[9][2] = {{r, 0},  {r, r},  {0, r},  {-r, r}, {-r, 0},
                             {-r, -r}, {0, -r}, {r, -r}, {r, 0}};
    const double wt[9] = {1, w, 1, w, 1, w, 1, w, 1};
    cps_out.resize(9);
    w_out.resize(9);
    for (int i = 0; i < 9; ++i)
    {
        cps_out[static_cast<std::size_t>(i)] = {s * xy[i][0], s * xy[i][1], z};
        w_out[static_cast<std::size_t>(i)] = wt[i];
    }
}

/// Exact single-patch rational torus (R=2, r=0.7), matching
/// python_experiments/nurbs_surface_examples/torus.py.
mfem_raytracing::SurfaceData MakeTorusSurface(double major_radius = 2.0,
                                              double minor_radius = 0.7)
{
    // Profile circle in xz: unit circle in xy mapped (x,y)->(R+x, 0, y).
    std::vector<std::vector<double>> profile_xy;
    std::vector<double> beta;
    FillHorizontalCircle(minor_radius, 0.0, profile_xy, beta);
    std::vector<std::vector<double>> profile(9);
    for (int j = 0; j < 9; ++j)
    {
        profile[static_cast<std::size_t>(j)] = {major_radius + profile_xy[static_cast<std::size_t>(j)][0],
                                                0.0,
                                                profile_xy[static_cast<std::size_t>(j)][1]};
    }

    const double w_corner = std::sqrt(2.0) / 2.0;
    const double alpha[9] = {1, w_corner, 1, w_corner, 1, w_corner, 1, w_corner, 1};

    mfem_raytracing::SurfaceData s;
    s.degree_u = 2;
    s.degree_v = 2;
    s.dim = 3;
    s.knotvector_u = {0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1};
    s.knotvector_v = s.knotvector_u;
    s.u_domain = {0.0, 1.0};
    s.v_domain = {0.0, 1.0};
    s.control_points.resize(9);
    s.weights.resize(9);
    for (int i = 0; i < 9; ++i)
    {
        s.control_points[static_cast<std::size_t>(i)].resize(9);
        s.weights[static_cast<std::size_t>(i)].resize(9);
    }
    for (int j = 0; j < 9; ++j)
    {
        const double xj = profile[static_cast<std::size_t>(j)][0];
        const double zj = profile[static_cast<std::size_t>(j)][2];
        std::vector<std::vector<double>> circle;
        std::vector<double> unused_w;
        FillHorizontalCircle(xj, zj, circle, unused_w);
        for (int i = 0; i < 9; ++i)
        {
            s.control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                circle[static_cast<std::size_t>(i)];
            s.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                alpha[i] * beta[static_cast<std::size_t>(j)];
        }
    }
    return s;
}

bool IntervalCrossesHardSeam(double lo, double hi, const std::vector<double> &hard,
                             double tol = 1e-12)
{
    for (double k : hard)
    {
        if (lo + tol < k && k < hi - tol)
        {
            return true;
        }
    }
    return false;
}

void TestConformingTorusHardSeams()
{
    const mfem_raytracing::SurfaceData torus = MakeTorusSurface();
    CHECK(torus.degree_u == 2);
    CHECK(torus.degree_v == 2);
    CHECK(torus.NumControlPointsU() == 9);
    CHECK(torus.NumControlPointsV() == 9);
    CHECK(torus.IsRational());
    // Spot-check: outer equator CP matches Python torus (R+r = 2.7).
    CHECK_NEAR(torus.control_points[0][0][0], 2.7, 1e-12);
    CHECK_NEAR(torus.control_points[0][0][1], 0.0, 1e-12);
    CHECK_NEAR(torus.control_points[0][0][2], 0.0, 1e-12);

    const double max_error = 0.1;
    mfem_raytracing::ConformingReductionOptions opts;
    opts.coalesce = true;
    opts.hard_seams = true;
    const auto result =
        mfem_raytracing::DegreeReduceMultipleStepsConforming(torus, max_error, opts);

    CheckConformingResult(result);

    const std::vector<double> hard = {0.0, 0.25, 0.5, 0.75, 1.0};
    for (const auto &leaf : result.segments)
    {
        CHECK(leaf.surface.NumControlPointsU() == 2);
        CHECK(leaf.surface.NumControlPointsV() == 2);
        CHECK(leaf.total_error <= max_error + 1e-12);
        CHECK(!IntervalCrossesHardSeam(leaf.u_domain_global.first, leaf.u_domain_global.second,
                                       hard));
        CHECK(!IntervalCrossesHardSeam(leaf.v_domain_global.first, leaf.v_domain_global.second,
                                       hard));
    }

    // Interior hard seams must appear as cell boundaries (never removed).
    auto has_boundary = [&](double k, bool u_dir) {
        for (const auto &leaf : result.segments)
        {
            const auto &dom = u_dir ? leaf.u_domain_global : leaf.v_domain_global;
            if (std::fabs(dom.first - k) < 1e-12 || std::fabs(dom.second - k) < 1e-12)
            {
                return true;
            }
        }
        return false;
    };
    for (double k : {0.25, 0.5, 0.75})
    {
        CHECK(has_boundary(k, true));
        CHECK(has_boundary(k, false));
    }

    // Sanity: torus needs many leaves at this tolerance; hard seams alone give
    // at least 4×4 knot-span cells, and refinement adds more.
    CHECK(result.segments.size() >= 16);
    CHECK(result.segments.size() <= 5000);
}

void TestConformingWatertightAndCoalesce()
{
    const mfem_raytracing::SurfaceData input = PeakSaddleInput();

    mfem_raytracing::ConformingReductionOptions grid_only;
    grid_only.coalesce = false;
    const auto grid =
        mfem_raytracing::DegreeReduceMultipleStepsConforming(input, 1.0, grid_only);
    CheckConformingResult(grid);

    mfem_raytracing::ConformingReductionOptions coalesced;
    coalesced.coalesce = true;
    const auto compact =
        mfem_raytracing::DegreeReduceMultipleStepsConforming(input, 1.0, coalesced);
    CheckConformingResult(compact);

    // coalescing may only shrink the grid, and both stay within budget
    CHECK(compact.segments.size() <= grid.segments.size());
    for (const auto &leaf : compact.segments)
    {
        CHECK(leaf.total_error <= 1.0 + 1e-12);
    }
}

// (p, q) input with p != q: reduce (3, 2) -> (1, 1); the v direction finishes a
// step earlier and is skipped.
void TestConformingAsymmetricDegrees()
{
    mfem_raytracing::SurfaceData surface;
    surface.degree_u = 3;
    surface.degree_v = 2;
    surface.dim = 3;
    surface.knotvector_u = {0, 0, 0, 0, 1, 1, 1, 1};
    surface.knotvector_v = {0, 0, 0, 1, 1, 1};
    surface.u_domain = {0.0, 1.0};
    surface.v_domain = {0.0, 1.0};
    const int n_u = 4;
    const int n_v = 3;
    surface.control_points.resize(n_u);
    for (int i = 0; i < n_u; ++i)
    {
        surface.control_points[static_cast<std::size_t>(i)].resize(n_v);
        for (int j = 0; j < n_v; ++j)
        {
            const double x = static_cast<double>(i) / (n_u - 1);
            const double y = static_cast<double>(j) / (n_v - 1);
            surface.control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                {x, y, 0.4 * x * x + 0.3 * y - 0.2 * x * y};
        }
    }

    const auto result = mfem_raytracing::DegreeReduceMultipleStepsConforming(surface, 0.05);
    CheckConformingResult(result);
    for (const auto &leaf : result.segments)
    {
        CHECK(leaf.steps_taken == 2); // max(3-1, 2-1)
    }
}

} // namespace

void TestSurfaceMultistepReduction()
{
    TestZeroStepsReturnsInput();
    TestInvalidArgumentsThrow();
    TestBudgetRespectedAndTargetDegreesReached();
    TestErrorCombinationAndRationalCorrection();
    TestBackendPolicyMapping();
    TestBilinearControlNetAABB();
    TestWeightClamping();
    TestGoldenLeafBBoxes();
    TestJsonWriterRoundTrip();
    TestConformingWatertightAndCoalesce();
    TestConformingAsymmetricDegrees();
    TestConformingTorusHardSeams();
}
