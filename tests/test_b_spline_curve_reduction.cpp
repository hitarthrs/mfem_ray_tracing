#include "mfem_raytracing/reduction/b_spline_curve_reduction.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace
{

using CP = std::vector<std::vector<double>>;

struct GoldenCase
{
    std::string name;
    int n;
    int p;
    int dim;
    CP qw;
    std::vector<double> u;
    CP pw_expected;
    std::vector<double> uh_expected;
    std::vector<std::pair<int, double>> nonzero_err;
    double max_err_expected;
};

void CheckControlPoints(const CP &got,
                        const CP &expected,
                        double tol,
                        const char *label)
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
    (void)label;
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

void RunGoldenCase(const GoldenCase &golden, double cp_tol = 1e-9, double err_tol = 1e-9)
{
    std::vector<std::vector<double>> pw;
    std::vector<double> uh;
    std::vector<double> error_array;

    const bool ok = DegreeReduceCurve(
        golden.n, golden.p, golden.u, golden.qw, golden.dim, pw, uh, error_array);
    CHECK(ok);

    CheckControlPoints(pw, golden.pw_expected, cp_tol, golden.name.c_str());
    CheckKnotVector(uh, golden.uh_expected, 1e-12);

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

GoldenCase P6ToP5MultiplePeak()
{
    return GoldenCase{
        "p6_to_p5_multiple_peak",
        16,
        6,
        2,
        {{0.0, 5.0},
         {2.0, 8.0},
         {5.0, 2.0},
         {6.2, 3.5},
         {7.0, 19.0},
         {7.5, 4.0},
         {8.0, -12.0},
         {8.5, 4.0},
         {9.0, 19.0},
         {10.0, 3.0},
         {11.0, 4.0},
         {14.0, 4.0},
         {17.0, 8.5},
         {20.0, 6.0},
         {22.0, 2.5},
         {24.0, 5.0}},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5,
         0.75, 0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        {{0.0, 5.0},
         {2.4, 8.6},
         {6.3, -1.2999999999999992},
         {6.632499999999999, 22.9125},
         {7.536250000000001, -7.487499999999999},
         {8.46625, -7.50625},
         {9.187499999999996, 17.556249999999995},
         {10.593750000000002, -3.612499999999998},
         {13.04999999999999, 5.487499999999999},
         {19.200000000000003, 8.0},
         {21.599999999999998, 2.0000000000000004},
         {24.0, 5.0}},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        {{5, 5.759202045693916},
         {6, 6.154962911884611},
         {7, 5.759202045693917},
         {8, 14.23489175306435},
         {9, 15.642118369780171},
         {10, 8.475689707370432},
         {11, 11.372733256943864},
         {12, 12.614414714648387},
         {13, 2.8970435495734312},
         {14, 2.8970435495734312},
         {15, 3.305878148237661}},
        15.642118369780171};
}

GoldenCase P4ToP3MultiplePeak()
{
    return GoldenCase{
        "p4_to_p3_multiple_peak",
        18,
        4,
        2,
        {{-0.067694, 5.50859},
         {3.45525, 6.31394},
         {7.03265, 0.424671},
         {6.61319, 14.8938},
         {7.49642, 4.0892},
         {7.57152, -0.552611},
         {8.04518, -8.84188},
         {8.35854, -0.796542},
         {8.5875, 4.59629},
         {9.04086, 10.4526},
         {9.32774, 10.0187},
         {9.97791, 5.21322},
         {11.0497, 3.25742},
         {11.8088, 4.17042},
         {14.6364, 4.88271},
         {16.8373, 10.8403},
         {21.2725, 1.98826},
         {23.9667, 4.87204}},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.375, 0.375, 0.5, 0.5,
         0.5, 0.625, 0.625, 0.75, 0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0},
        {{-0.067694, 5.50859},
         {4.629564666666666, 6.582389999999999},
         {6.416916666666668, 17.294822222222223},
         {7.594556666666667, 2.8886888888888884},
         {7.890973333333332, -12.242524722222221},
         {8.511940000000001, 3.620238333333333},
         {9.11642, 11.428651666666667},
         {9.902236666666669, 4.930148333333335},
         {11.49462222222222, 4.0912766666666665},
         {15.264755555555555, 5.0409966666666675},
         {20.374433333333332, 1.027},
         {23.9667, 4.87204}},
        {0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.375, 0.5, 0.5, 0.625, 0.75, 0.75,
         1.0, 1.0, 1.0, 1.0},
        {{4, 4.354669125823956},
         {5, 1.7763568394002505e-15},
         {6, 1.0649998068265611},
         {7, 1.1981438150641397},
         {8, 1.0649998068265594},
         {9, 1.136051225352972},
         {11, 1.6796802757205789},
         {12, 1.7603758763511779},
         {13, 1.6796802757205809},
         {14, 1.9139467450308758},
         {15, 3.552713678800501e-15},
         {16, 3.552713678800501e-15},
         {17, 2.9504481459901353}},
        4.354669125823956};
}

GoldenCase P3ToP2SimpleSinglePeakTrailing()
{
    return GoldenCase{
        "p3_to_p2_simple_single_peak_trailing",
        6,
        3,
        2,
        {{0.0, 0.0},
         {0.5, 3.0},
         {2.0, 3.0},
         {2.5, 0.0},
         {6.0, -1.0},
         {9.0, -1.0}},
        {0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0},
        {{0.0, 0.0},
         {1.3125, 4.125},
         {3.5625, -0.875},
         {9.0, -1.0}},
        {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0},
        {{3, 0.35545072812313877},
         {4, 0.22534695471649932},
         {5, 0.4073651076271593}},
        0.4073651076271593};
}

}  // namespace

void TestBSplineCurveReduction()
{
    RunGoldenCase(P6ToP5MultiplePeak());
    RunGoldenCase(P4ToP3MultiplePeak());
    RunGoldenCase(P3ToP2SimpleSinglePeakTrailing());
}
