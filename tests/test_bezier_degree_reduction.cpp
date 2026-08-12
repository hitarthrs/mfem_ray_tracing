#include "mfem_raytracing/reduction/bezier_degree_reduction.hpp"
#include "test_helpers.hpp"

#include <cmath>

namespace
{

std::vector<std::vector<double>> ElevateDegreeOnce(
    const std::vector<std::vector<double>> &lower,
    int dim)
{
    const int p = static_cast<int>(lower.size());
    std::vector<std::vector<double>> elevated(static_cast<std::size_t>(p + 1),
                                              std::vector<double>(static_cast<std::size_t>(dim)));
    elevated[0] = lower[0];
    elevated[static_cast<std::size_t>(p)] = lower[static_cast<std::size_t>(p - 1)];
    for (int i = 1; i < p; ++i)
    {
        const double a = static_cast<double>(i) / static_cast<double>(p);
        elevated[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(dim));
        for (int d = 0; d < dim; ++d)
        {
            elevated[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] =
                a * lower[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] +
                (1.0 - a) * lower[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(d)];
        }
    }
    return elevated;
}

void TestQuadraticLineExactReduction()
{
    const int dim = 3;
    const std::vector<std::vector<double>> bpts = {
        {0.0, 0.0, 0.0},
        {4.0, 0.0, 0.0},
        {8.0, 0.0, 0.0},
    };

    std::vector<std::vector<double>> reduced;
    double max_err = -1.0;
    BezierDegreeReduce(bpts, dim, reduced, max_err);

    CHECK(static_cast<int>(reduced.size()) == 2);
    CHECK_NEAR(reduced[0][0], 0.0, 1e-12);
    CHECK_NEAR(reduced[1][0], 8.0, 1e-12);
    CHECK_NEAR(max_err, 0.0, 1e-10);
}

std::vector<double> EvaluateBezier(const std::vector<std::vector<double>> &bpts,
                                    double u,
                                    int dim)
{
    const int p = static_cast<int>(bpts.size()) - 1;
    std::vector<double> value(static_cast<std::size_t>(dim), 0.0);
    for (int i = 0; i <= p; ++i)
    {
        double coeff = 1.0;
        for (int j = 1; j <= i; ++j)
        {
            coeff *= static_cast<double>(p - j + 1) / static_cast<double>(j);
        }
        coeff *= std::pow(u, i) * std::pow(1.0 - u, p - i);
        for (int d = 0; d < dim; ++d)
        {
            value[static_cast<std::size_t>(d)] +=
                coeff * bpts[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)];
        }
    }
    return value;
}

double MaxCurveDeviation(const std::vector<std::vector<double>> &high,
                         const std::vector<std::vector<double>> &low,
                         int dim,
                         int samples = 64)
{
    double peak = 0.0;
    for (int s = 0; s <= samples; ++s)
    {
        const double u = static_cast<double>(s) / static_cast<double>(samples);
        const std::vector<double> a = EvaluateBezier(high, u, dim);
        const std::vector<double> b = EvaluateBezier(low, u, dim);
        double dist_sq = 0.0;
        for (int d = 0; d < dim; ++d)
        {
            const double delta =
                a[static_cast<std::size_t>(d)] - b[static_cast<std::size_t>(d)];
            dist_sq += delta * delta;
        }
        peak = std::max(peak, std::sqrt(dist_sq));
    }
    return peak;
}

void TestElevatedLineReportsApproximation()
{
    const int dim = 2;
    const std::vector<std::vector<double>> line = {
        {0.0, 0.0},
        {10.0, 0.0},
    };

    std::vector<std::vector<double>> cubic = ElevateDegreeOnce(
        ElevateDegreeOnce(line, dim), dim);

    CHECK(static_cast<int>(cubic.size()) == 4);

    std::vector<std::vector<double>> reduced;
    double max_err = 0.0;
    BezierDegreeReduce(cubic, dim, reduced, max_err);

    CHECK(static_cast<int>(reduced.size()) == 3);
    CHECK_NEAR(reduced[0][0], 0.0, 1e-12);
    CHECK_NEAR(reduced[2][0], 10.0, 1e-12);
    CHECK(max_err > 0.0);
    CHECK(max_err + 1e-10 >= MaxCurveDeviation(cubic, reduced, dim));
}

void TestCubicNonCollinearPositiveError()
{
    const int dim = 2;
    const std::vector<std::vector<double>> bpts = {
        {0.0, 0.0},
        {1.0, 2.0},
        {2.0, -1.0},
        {3.0, 1.0},
    };

    std::vector<std::vector<double>> reduced;
    double max_err = 0.0;
    BezierDegreeReduce(bpts, dim, reduced, max_err);

    CHECK(static_cast<int>(reduced.size()) == 3);
    CHECK(max_err > 0.0);
}

void TestSymmetricCubicExactReduction()
{
    const int dim = 2;
    const std::vector<std::vector<double>> bpts = {
        {0.0, 0.0},
        {3.0, 6.0},
        {6.0, 6.0},
        {9.0, 0.0},
    };

    std::vector<std::vector<double>> reduced;
    double max_err = -1.0;
    BezierDegreeReduce(bpts, dim, reduced, max_err);

    CHECK(static_cast<int>(reduced.size()) == 3);
    CHECK_NEAR(reduced[1][0], 4.5, 1e-12);
    CHECK_NEAR(reduced[1][1], 9.0, 1e-12);
    CHECK_NEAR(max_err, 0.0, 1e-10);
    CHECK(MaxCurveDeviation(bpts, reduced, dim) < 1e-10);
}

void TestSymmetricQuinticExactReduction()
{
    const int dim = 2;
    const std::vector<std::vector<double>> bpts = {
        {0.0, 0.0},
        {3.2, 12.8},
        {6.4, 19.2},
        {9.6, 19.2},
        {12.8, 12.8},
        {16.0, 0.0},
    };

    std::vector<std::vector<double>> reduced;
    double max_err = -1.0;
    BezierDegreeReduce(bpts, dim, reduced, max_err);

    CHECK(static_cast<int>(reduced.size()) == 5);
    CHECK_NEAR(reduced[1][0], 4.0, 1e-12);
    CHECK_NEAR(reduced[1][1], 16.0, 1e-12);
    CHECK_NEAR(reduced[2][0], 8.0, 1e-12);
    CHECK_NEAR(reduced[2][1], 64.0 / 3.0, 1e-10);
    CHECK_NEAR(reduced[3][0], 12.0, 1e-12);
    CHECK_NEAR(reduced[3][1], 16.0, 1e-12);
    CHECK_NEAR(max_err, 0.0, 1e-10);
    CHECK(MaxCurveDeviation(bpts, reduced, dim) < 1e-10);
}

void TestRejectTooFewPoints()
{
    bool threw = false;
    try
    {
        const std::vector<std::vector<double>> bpts = {{0.0}, {1.0}};
        std::vector<std::vector<double>> reduced;
        double max_err = 0.0;
        BezierDegreeReduce(bpts, 1, reduced, max_err);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

void TestBezierDegreeReduction()
{
    TestQuadraticLineExactReduction();
    TestSymmetricCubicExactReduction();
    TestSymmetricQuinticExactReduction();
    TestElevatedLineReportsApproximation();
    TestCubicNonCollinearPositiveError();
    TestRejectTooFewPoints();
}
