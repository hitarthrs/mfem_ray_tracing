/**
 * @file test_nurbs_degree_reduction.cpp
 * @brief Unit tests for NURBS degree reduction via mfem::NURBSPatch.
 *
 * Golden expected values are generated from Python:
 *   python_experiments/export_cpp_golden_cases.py
 *     --output tests/generated_nurbs_golden_cases.inc
 */

#include "nurbs_degree_reduction.hpp"
#include "test_helpers.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace
{

using CP = std::vector<std::vector<double>>;

/** Expected outputs from a single Python-generated reduction case. */
struct GoldenCase
{
    std::string name;
    int n;
    int p;
    int dim;
    CP qw;
    std::vector<double> weights;
    std::vector<double> u;
    CP pw_expected;
    std::vector<double> weights_expected;
    std::vector<double> uh_expected;
    std::vector<std::pair<int, double>> nonzero_err;
    double max_err_expected;
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

void CheckKnotVector(const std::vector<double> &got,
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

#include "generated_nurbs_golden_cases.inc"

mfem::KnotVector MakeKnotVector(int degree,
                                int n_control_points,
                                const std::vector<double> &knots)
{
    std::ostringstream oss;
    oss << degree << " " << n_control_points;
    for (double knot : knots)
    {
        oss << " " << knot;
    }
    std::istringstream iss(oss.str());
    return mfem::KnotVector(iss);
}

/** Pack (w·P, w) into MFEM's flat homogeneous layout. Uses w = 1 when weights empty. */
std::vector<double> PackHomogeneousFlat(const CP &control_points,
                                        const std::vector<double> &weights,
                                        int spatial_dim)
{
    const int homog_dim = spatial_dim + 1;
    const std::size_t n = control_points.size();
    std::vector<double> flat(n * static_cast<std::size_t>(homog_dim));

    for (std::size_t i = 0; i < n; ++i)
    {
        const double w = weights.empty() ? 1.0 : weights[i];
        for (int d = 0; d < spatial_dim; ++d)
        {
            flat[i * static_cast<std::size_t>(homog_dim) + static_cast<std::size_t>(d)] =
                control_points[i][static_cast<std::size_t>(d)] * w;
        }
        flat[i * static_cast<std::size_t>(homog_dim) + static_cast<std::size_t>(spatial_dim)] = w;
    }
    return flat;
}

/** Build a 1D mfem::NURBSPatch from golden input data. */
mfem::NURBSPatch MakePatchFromGolden(const GoldenCase &golden)
{
    mfem::KnotVector kv = MakeKnotVector(golden.p, golden.n, golden.u);
    const std::vector<double> flat =
        PackHomogeneousFlat(golden.qw, golden.weights, golden.dim);

    mfem::Array<const mfem::KnotVector *> kvs(1);
    kvs[0] = &kv;
    return mfem::NURBSPatch(kvs, golden.dim + 1, flat.data());
}

/** Build patch, run DegreeReduceNURBCurve, compare to Python goldens. */
void RunGoldenCase(const GoldenCase &golden, double cp_tol = 1e-9, double err_tol = 1e-9)
{
    mfem::NURBSPatch patch = MakePatchFromGolden(golden);

    std::vector<std::vector<double>> pw;
    std::vector<double> weights_out;
    std::vector<double> uh;
    std::vector<double> error_array;

    const bool ok = DegreeReduceNURBCurve(patch, pw, weights_out, uh, error_array);
    CHECK(ok);

    CheckControlPoints(pw, golden.pw_expected, cp_tol);
    CheckKnotVector(uh, golden.uh_expected, 1e-12);
    CheckWeights(weights_out, golden.weights_expected, cp_tol);

    double max_err = 0.0;
    for (double e : error_array)
    {
        max_err = std::max(max_err, e);
    }
    CHECK_NEAR(max_err, golden.max_err_expected, err_tol);

    for (const auto &[idx, expected] : golden.nonzero_err)
    {
        CHECK(idx >= 0 && idx < static_cast<int>(error_array.size()));
        CHECK_NEAR(error_array[static_cast<std::size_t>(idx)], expected, err_tol);
    }
}

}  // namespace

void TestNURBSDegreeReduction()
{
    RunGoldenCase(P6ToP5MultiplePeakUnified());
    RunGoldenCase(P4ToP3SShapedUnified());
    RunGoldenCase(P4ToP3MultiplePeakUnified());
    RunGoldenCase(P4ToP3SinglePeakUniformUnified());
    RunGoldenCase(P4ToP3SinglePeakTrailingUnified());
    RunGoldenCase(P4ToP3SemicircleRational());
    RunGoldenCase(P6ToP5CircleRational());
}
