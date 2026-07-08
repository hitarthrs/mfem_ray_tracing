/**
 * @file test_surface_degree_reduction.cpp
 * @brief Regression tests for adaptive single-step surface reduction.
 *
 * Generated fixtures come from Python example surfaces:
 *   python_experiments/export_cpp_golden_cases.py
 *     --surface-output tests/generated_surface_degree_reduction_cases.inc
 */

#include "surface_degree_reduction.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace
{

using SurfaceCP = std::vector<std::vector<std::vector<double>>>;
using WeightNet = std::vector<std::vector<double>>;
using Domain = std::pair<double, double>;

struct SurfaceGolden
{
    int degree_u;
    int degree_v;
    int dim;
    SurfaceCP control_points;
    WeightNet weights;
    std::vector<double> knotvector_u;
    std::vector<double> knotvector_v;
};

struct SingleStepSurfaceSegmentGolden
{
    SurfaceGolden surface;
    double segment_error;
    Domain u_domain;
    Domain v_domain;
};

struct SingleStepSurfaceGoldenCase
{
    std::string name;
    SurfaceGolden input;
    double max_error;
    std::vector<SingleStepSurfaceSegmentGolden> segments;
};

void CheckSurfaceControlPoints(const SurfaceCP &got, const SurfaceCP &expected, double tol)
{
    CHECK(static_cast<int>(got.size()) == static_cast<int>(expected.size()));
    const int n_u = std::min(static_cast<int>(got.size()), static_cast<int>(expected.size()));
    for (int i = 0; i < n_u; ++i)
    {
        CHECK(static_cast<int>(got[static_cast<std::size_t>(i)].size()) ==
              static_cast<int>(expected[static_cast<std::size_t>(i)].size()));
        const int n_v = std::min(static_cast<int>(got[static_cast<std::size_t>(i)].size()),
                                 static_cast<int>(expected[static_cast<std::size_t>(i)].size()));
        for (int j = 0; j < n_v; ++j)
        {
            CHECK(static_cast<int>(got[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].size()) ==
                  static_cast<int>(expected[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].size()));
            const int dim = std::min(
                static_cast<int>(got[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].size()),
                static_cast<int>(expected[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].size()));
            for (int d = 0; d < dim; ++d)
            {
                CHECK_NEAR(got[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(d)],
                           expected[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(d)],
                           tol);
            }
        }
    }
}

void CheckWeightNet(const WeightNet &got, const WeightNet &expected, double tol)
{
    CHECK(static_cast<int>(got.size()) == static_cast<int>(expected.size()));
    const int n_u = std::min(static_cast<int>(got.size()), static_cast<int>(expected.size()));
    for (int i = 0; i < n_u; ++i)
    {
        CHECK(static_cast<int>(got[static_cast<std::size_t>(i)].size()) ==
              static_cast<int>(expected[static_cast<std::size_t>(i)].size()));
        const int n_v = std::min(static_cast<int>(got[static_cast<std::size_t>(i)].size()),
                                 static_cast<int>(expected[static_cast<std::size_t>(i)].size()));
        for (int j = 0; j < n_v; ++j)
        {
            CHECK_NEAR(got[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                       expected[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                       tol);
        }
    }
}

void CheckKnots(const std::vector<double> &got,
                const std::vector<double> &expected,
                double tol)
{
    CHECK(static_cast<int>(got.size()) == static_cast<int>(expected.size()));
    const int n = std::min(static_cast<int>(got.size()), static_cast<int>(expected.size()));
    for (int i = 0; i < n; ++i)
    {
        CHECK_NEAR(got[static_cast<std::size_t>(i)],
                   expected[static_cast<std::size_t>(i)],
                   tol);
    }
}

mfem_raytracing::SurfaceData MakeSurfaceData(const SurfaceGolden &golden)
{
    mfem_raytracing::SurfaceData surface;
    surface.degree_u = golden.degree_u;
    surface.degree_v = golden.degree_v;
    surface.dim = golden.dim;
    surface.control_points = golden.control_points;
    surface.weights = golden.weights;
    surface.knotvector_u = golden.knotvector_u;
    surface.knotvector_v = golden.knotvector_v;
    surface.u_domain = {golden.knotvector_u[static_cast<std::size_t>(golden.degree_u)],
                        golden.knotvector_u[golden.knotvector_u.size() -
                                            static_cast<std::size_t>(golden.degree_u) - 1]};
    surface.v_domain = {golden.knotvector_v[static_cast<std::size_t>(golden.degree_v)],
                        golden.knotvector_v[golden.knotvector_v.size() -
                                            static_cast<std::size_t>(golden.degree_v) - 1]};
    return surface;
}

void CheckSurfaceData(const mfem_raytracing::SurfaceData &got,
                      const SurfaceGolden &expected,
                      double tol = 1e-9)
{
    CHECK(got.degree_u == expected.degree_u);
    CHECK(got.degree_v == expected.degree_v);
    CHECK(got.dim == expected.dim);
    CheckSurfaceControlPoints(got.control_points, expected.control_points, tol);
    CheckWeightNet(got.weights, expected.weights, tol);
    CheckKnots(got.knotvector_u, expected.knotvector_u, 1e-12);
    CheckKnots(got.knotvector_v, expected.knotvector_v, 1e-12);
}

#include "generated_surface_degree_reduction_cases.inc"

void RunSingleStepSurfaceCase(const SingleStepSurfaceGoldenCase &golden)
{
    const mfem_raytracing::SurfaceData input = MakeSurfaceData(golden.input);
    const mfem_raytracing::SingleStepSurfaceReductionResult result =
        mfem_raytracing::PeakErrorSurfaceSingleStep(input, golden.max_error);

    CHECK(static_cast<int>(result.segments.size()) == static_cast<int>(golden.segments.size()));
    const int count = std::min(static_cast<int>(result.segments.size()),
                               static_cast<int>(golden.segments.size()));
    for (int i = 0; i < count; ++i)
    {
        CheckSurfaceData(result.segments[static_cast<std::size_t>(i)].surface,
                         golden.segments[static_cast<std::size_t>(i)].surface,
                         1e-8);
        CHECK_NEAR(result.segments[static_cast<std::size_t>(i)].segment_error,
                   golden.segments[static_cast<std::size_t>(i)].segment_error,
                   1e-8);
        CHECK_NEAR(result.segments[static_cast<std::size_t>(i)].u_domain.first,
                   golden.segments[static_cast<std::size_t>(i)].u_domain.first,
                   1e-10);
        CHECK_NEAR(result.segments[static_cast<std::size_t>(i)].u_domain.second,
                   golden.segments[static_cast<std::size_t>(i)].u_domain.second,
                   1e-10);
        CHECK_NEAR(result.segments[static_cast<std::size_t>(i)].v_domain.first,
                   golden.segments[static_cast<std::size_t>(i)].v_domain.first,
                   1e-10);
        CHECK_NEAR(result.segments[static_cast<std::size_t>(i)].v_domain.second,
                   golden.segments[static_cast<std::size_t>(i)].v_domain.second,
                   1e-10);
    }
}

} // namespace

void TestSurfaceDegreeReduction()
{
    RunSingleStepSurfaceCase(SurfaceApproach1SShapedPeakSaddle());
    RunSingleStepSurfaceCase(SurfaceApproach1MultiplePeakUniformShell());
    RunSingleStepSurfaceCase(SurfaceApproach1SemicirclePlateauShell());
    RunSingleStepSurfaceCase(SurfaceApproach1SemicircleSShapedCrown());
}
