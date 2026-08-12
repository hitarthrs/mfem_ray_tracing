#include "mfem_raytracing/reduction/curve_degree_reduction.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace
{

using CP = std::vector<std::vector<double>>;

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
    std::pair<double, double> u_domain;
};

struct SingleStepGoldenCase
{
    std::string name;
    CurveGolden input;
    double max_error;
    std::vector<SingleStepSegmentGolden> segments;
};

#include "generated_curve_degree_reduction_cases.inc"

mfem_raytracing::CurveData MakeCurveData(const CurveGolden &golden)
{
    mfem_raytracing::CurveData curve;
    curve.degree = golden.degree;
    curve.dim = golden.dim;
    curve.control_points = golden.control_points;
    curve.weights = golden.weights;
    curve.knotvector = golden.knotvector;
    curve.domain = {golden.knotvector[static_cast<std::size_t>(golden.degree)],
                    golden.knotvector[golden.knotvector.size() -
                                      static_cast<std::size_t>(golden.degree) - 1]};
    return curve;
}

bool NearlyEqual(double a, double b, double tol)
{
    return std::abs(a - b) <= tol;
}

bool SameResult(const mfem_raytracing::SingleStepReductionResult &lhs,
                const mfem_raytracing::SingleStepReductionResult &rhs)
{
    if (lhs.segments.size() != rhs.segments.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < lhs.segments.size(); ++i)
    {
        const auto &a = lhs.segments[i];
        const auto &b = rhs.segments[i];
        if (a.curve.degree != b.curve.degree ||
            a.curve.dim != b.curve.dim ||
            a.curve.knotvector.size() != b.curve.knotvector.size() ||
            a.curve.control_points.size() != b.curve.control_points.size() ||
            a.curve.weights.size() != b.curve.weights.size())
        {
            return false;
        }
        if (!NearlyEqual(a.segment_error, b.segment_error, 1e-12) ||
            !NearlyEqual(a.u_domain.first, b.u_domain.first, 1e-12) ||
            !NearlyEqual(a.u_domain.second, b.u_domain.second, 1e-12))
        {
            return false;
        }
        for (std::size_t k = 0; k < a.curve.knotvector.size(); ++k)
        {
            if (!NearlyEqual(a.curve.knotvector[k], b.curve.knotvector[k], 1e-12))
            {
                return false;
            }
        }
        for (std::size_t p = 0; p < a.curve.control_points.size(); ++p)
        {
            if (a.curve.control_points[p].size() != b.curve.control_points[p].size())
            {
                return false;
            }
            for (std::size_t d = 0; d < a.curve.control_points[p].size(); ++d)
            {
                if (!NearlyEqual(a.curve.control_points[p][d], b.curve.control_points[p][d], 1e-12))
                {
                    return false;
                }
            }
        }
        for (std::size_t w = 0; w < a.curve.weights.size(); ++w)
        {
            if (!NearlyEqual(a.curve.weights[w], b.curve.weights[w], 1e-12))
            {
                return false;
            }
        }
    }
    return true;
}

template <typename Fn>
double TimeMs(Fn &&fn, int iterations)
{
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        fn();
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

void BenchmarkCase(const SingleStepGoldenCase &golden, int iterations)
{
    const mfem_raytracing::CurveData input = MakeCurveData(golden.input);

    const auto baseline_once =
        mfem_raytracing::PeakErrorSingleStep(input, golden.max_error);
    const auto optimized_once =
        mfem_raytracing::PeakErrorSingleStepOptimized(input, golden.max_error);
    if (!SameResult(baseline_once, optimized_once))
    {
        std::cerr << "Result mismatch for " << golden.name << std::endl;
        std::exit(1);
    }

    const double baseline_ms = TimeMs(
        [&]() {
            auto result = mfem_raytracing::PeakErrorSingleStep(input, golden.max_error);
            (void)result;
        },
        iterations);
    const double optimized_ms = TimeMs(
        [&]() {
            auto result =
                mfem_raytracing::PeakErrorSingleStepOptimized(input, golden.max_error);
            (void)result;
        },
        iterations);

    std::cout << golden.name << "\n";
    std::cout << "  iterations: " << iterations << "\n";
    std::cout << "  baseline:   " << baseline_ms << " ms total, "
              << (baseline_ms / iterations) << " ms/call\n";
    std::cout << "  optimized:  " << optimized_ms << " ms total, "
              << (optimized_ms / iterations) << " ms/call\n";
    std::cout << "  speedup:    " << (baseline_ms / optimized_ms) << "x\n";
}

} // namespace

int main()
{
    BenchmarkCase(Approach1P4MultiplePeak(), 2000);
    BenchmarkCase(Approach1P4Semicircle(), 2000);
    return 0;
}
