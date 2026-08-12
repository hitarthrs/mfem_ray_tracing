#include "mfem_raytracing/reduction/surface_degree_reduction.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
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

#include "generated_surface_degree_reduction_cases.inc"

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

bool NearlyEqual(double a, double b, double tol)
{
    return std::abs(a - b) <= tol;
}

bool SameResult(const mfem_raytracing::SingleStepSurfaceReductionResult &lhs,
                const mfem_raytracing::SingleStepSurfaceReductionResult &rhs)
{
    if (lhs.segments.size() != rhs.segments.size())
    {
        return false;
    }

    for (std::size_t s = 0; s < lhs.segments.size(); ++s)
    {
        const auto &a = lhs.segments[s];
        const auto &b = rhs.segments[s];
        if (a.surface.degree_u != b.surface.degree_u ||
            a.surface.degree_v != b.surface.degree_v ||
            a.surface.dim != b.surface.dim ||
            a.surface.knotvector_u.size() != b.surface.knotvector_u.size() ||
            a.surface.knotvector_v.size() != b.surface.knotvector_v.size() ||
            a.surface.control_points.size() != b.surface.control_points.size() ||
            a.surface.weights.size() != b.surface.weights.size())
        {
            return false;
        }
        if (!NearlyEqual(a.segment_error, b.segment_error, 1e-12) ||
            !NearlyEqual(a.u_domain.first, b.u_domain.first, 1e-12) ||
            !NearlyEqual(a.u_domain.second, b.u_domain.second, 1e-12) ||
            !NearlyEqual(a.v_domain.first, b.v_domain.first, 1e-12) ||
            !NearlyEqual(a.v_domain.second, b.v_domain.second, 1e-12))
        {
            return false;
        }
        for (std::size_t i = 0; i < a.surface.knotvector_u.size(); ++i)
        {
            if (!NearlyEqual(a.surface.knotvector_u[i], b.surface.knotvector_u[i], 1e-12))
            {
                return false;
            }
        }
        for (std::size_t i = 0; i < a.surface.knotvector_v.size(); ++i)
        {
            if (!NearlyEqual(a.surface.knotvector_v[i], b.surface.knotvector_v[i], 1e-12))
            {
                return false;
            }
        }
        for (std::size_t i = 0; i < a.surface.control_points.size(); ++i)
        {
            if (a.surface.control_points[i].size() != b.surface.control_points[i].size())
            {
                return false;
            }
            for (std::size_t j = 0; j < a.surface.control_points[i].size(); ++j)
            {
                if (a.surface.control_points[i][j].size() != b.surface.control_points[i][j].size())
                {
                    return false;
                }
                for (std::size_t d = 0; d < a.surface.control_points[i][j].size(); ++d)
                {
                    if (!NearlyEqual(a.surface.control_points[i][j][d],
                                     b.surface.control_points[i][j][d],
                                     1e-12))
                    {
                        return false;
                    }
                }
            }
        }
        for (std::size_t i = 0; i < a.surface.weights.size(); ++i)
        {
            if (a.surface.weights[i].size() != b.surface.weights[i].size())
            {
                return false;
            }
            for (std::size_t j = 0; j < a.surface.weights[i].size(); ++j)
            {
                if (!NearlyEqual(a.surface.weights[i][j], b.surface.weights[i][j], 1e-12))
                {
                    return false;
                }
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

void BenchmarkCase(const SingleStepSurfaceGoldenCase &golden,
                   double max_error,
                   int iterations)
{
    const mfem_raytracing::SurfaceData input = MakeSurfaceData(golden.input);

    const auto baseline_once =
        mfem_raytracing::PeakErrorSurfaceSingleStep(input, max_error);
    const auto optimized_once =
        mfem_raytracing::PeakErrorSurfaceSingleStepOptimized(input, max_error);
    if (!SameResult(baseline_once, optimized_once))
    {
        std::cerr << "Result mismatch for " << golden.name << std::endl;
        std::exit(1);
    }

    const double baseline_ms = TimeMs(
        [&]() {
            auto result = mfem_raytracing::PeakErrorSurfaceSingleStep(input, max_error);
            (void)result;
        },
        iterations);
    const double optimized_ms = TimeMs(
        [&]() {
            auto result =
                mfem_raytracing::PeakErrorSurfaceSingleStepOptimized(input, max_error);
            (void)result;
        },
        iterations);

    std::cout << golden.name << "\n";
    std::cout << "  max_error:  " << max_error << "\n";
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
    BenchmarkCase(SurfaceApproach1SShapedPeakSaddle(), 1e-2, 200);
    BenchmarkCase(SurfaceApproach1SemicircleSShapedCrown(), 1e-2, 200);
    return 0;
}
