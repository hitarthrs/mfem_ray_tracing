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
#include "surface_golden_cases.hpp"
#include "surface_multistep_reduction.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace
{

#include "generated_surface_degree_reduction_cases.inc"

constexpr const char *kGoldenLeafJson =
    "python_experiments/multiple_step_degree_reduction_surfaces/outputs/d4_leaf_bboxes.json";

mfem_raytracing::SurfaceData PeakSaddleInput()
{
    return MakeSurfaceData(SurfaceApproach1SShapedPeakSaddle().input);
}

void TestZeroStepsReturnsInput()
{
    const mfem_raytracing::SurfaceData input = PeakSaddleInput();
    const auto result = mfem_raytracing::DegreeReduceMultipleSteps(input, 0, 1.0);
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
        mfem_raytracing::DegreeReduceMultipleSteps(input, -1, 1.0);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try
    {
        mfem_raytracing::DegreeReduceMultipleSteps(input, 1, -0.5);
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
        mfem_raytracing::DegreeReduceMultipleSteps(input, 4, 1.0);
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
        mfem_raytracing::DegreeReduceMultipleSteps(input, 3, max_error, options);

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

} // namespace

void TestSurfaceMultistepReduction()
{
    TestZeroStepsReturnsInput();
    TestInvalidArgumentsThrow();
    TestBudgetRespectedAndTargetDegreesReached();
    TestBackendPolicyMapping();
    TestBilinearControlNetAABB();
    TestWeightClamping();
    TestGoldenLeafBBoxes();
    TestJsonWriterRoundTrip();
}
