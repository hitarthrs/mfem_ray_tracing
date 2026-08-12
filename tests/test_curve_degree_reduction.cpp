/**
 * @file test_curve_degree_reduction.cpp
 * @brief Regression tests for adaptive single-step curve reduction.
 *
 * Generated fixtures come from Python example curves:
 *   python_experiments/export_cpp_golden_cases.py
 *     --output tests/generated_nurbs_golden_cases.inc
 *     --curve-output tests/generated_curve_degree_reduction_cases.inc
 */

#include "mfem_raytracing/reduction/curve_degree_reduction.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace
{

using CP = std::vector<std::vector<double>>;
using Domain = std::pair<double, double>;

struct CurveGolden
{
    int degree;
    int dim;
    CP control_points;
    std::vector<double> weights;
    std::vector<double> knotvector;
};

struct SingleStepSegmentGolden
{
    CurveGolden curve;
    double segment_error;
    Domain u_domain;
};

struct SingleStepGoldenCase
{
    std::string name;
    CurveGolden input;
    double max_error;
    std::vector<SingleStepSegmentGolden> segments;
};

void CheckControlPoints(const CP &got, const CP &expected, double tol)
{
    CHECK(static_cast<int>(got.size()) == static_cast<int>(expected.size()));
    const int n = std::min(static_cast<int>(got.size()), static_cast<int>(expected.size()));
    for (int i = 0; i < n; ++i)
    {
        CHECK(static_cast<int>(got[static_cast<std::size_t>(i)].size()) ==
              static_cast<int>(expected[static_cast<std::size_t>(i)].size()));
        const int dim = std::min(static_cast<int>(got[static_cast<std::size_t>(i)].size()),
                                 static_cast<int>(expected[static_cast<std::size_t>(i)].size()));
        for (int d = 0; d < dim; ++d)
        {
            CHECK_NEAR(got[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)],
                       expected[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)],
                       tol);
        }
    }
}

void CheckWeights(const std::vector<double> &got,
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

mfem_raytracing::CurveData MakeCurveData(const CurveGolden &golden)
{
    mfem_raytracing::CurveData curve;
    curve.degree = golden.degree;
    curve.dim = golden.dim;
    curve.control_points = golden.control_points;
    curve.weights = golden.weights;
    curve.knotvector = golden.knotvector;
    curve.domain = {golden.knotvector[static_cast<std::size_t>(golden.degree)],
                    golden.knotvector[golden.knotvector.size() - static_cast<std::size_t>(golden.degree) - 1]};
    return curve;
}

void CheckCurveData(const mfem_raytracing::CurveData &got,
                    const CurveGolden &expected,
                    double tol = 1e-9)
{
    CHECK(got.degree == expected.degree);
    CHECK(got.dim == expected.dim);
    CheckControlPoints(got.control_points, expected.control_points, tol);
    CheckWeights(got.weights, expected.weights, tol);
    CheckKnots(got.knotvector, expected.knotvector, 1e-12);
}

#include "generated_curve_degree_reduction_cases.inc"

void RunSingleStepCase(const SingleStepGoldenCase &golden)
{
    const mfem_raytracing::CurveData input = MakeCurveData(golden.input);
    const mfem_raytracing::SingleStepReductionResult result =
        mfem_raytracing::PeakErrorSingleStep(input, golden.max_error);
    CHECK(static_cast<int>(result.segments.size()) == static_cast<int>(golden.segments.size()));

    const int count = std::min(static_cast<int>(result.segments.size()),
                               static_cast<int>(golden.segments.size()));
    for (int i = 0; i < count; ++i)
    {
        CheckCurveData(result.segments[static_cast<std::size_t>(i)].curve,
                       golden.segments[static_cast<std::size_t>(i)].curve,
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
    }
}

} // namespace

void TestCurveDegreeReduction()
{
    RunSingleStepCase(Approach1P4SShaped());
    RunSingleStepCase(Approach1P4MultiplePeak());
    RunSingleStepCase(Approach1P4SinglePeakUniform());
    RunSingleStepCase(Approach1P4Semicircle());
}
