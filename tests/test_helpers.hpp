#ifndef TEST_HELPERS_HPP
#define TEST_HELPERS_HPP

#include "mfem.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

inline int g_failures = 0;

inline void Check(bool cond, const char *expr, const char *file, int line)
{
    if (!cond)
    {
        std::cerr << "FAIL " << file << ":" << line << "  " << expr << std::endl;
        ++g_failures;
    }
}

inline void CheckNear(double a, double b, double tol, const char *expr,
                      const char *file, int line)
{
    if (std::abs(a - b) > tol)
    {
        std::cerr << "FAIL " << file << ":" << line << "  " << expr
                  << "  (got " << a << ", expected " << b << ")" << std::endl;
        ++g_failures;
    }
}

#define CHECK(cond) Check((cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) CheckNear((a), (b), (tol), #a " near " #b, __FILE__, __LINE__)

/** Copy MFEM knot values (with multiplicities) into a std::vector. */
inline std::vector<double> KnotsFromKV(const mfem::KnotVector &kv)
{
    std::vector<double> knots(kv.Size());
    for (int i = 0; i < kv.Size(); ++i)
    {
        knots[i] = kv[i];
    }
    return knots;
}

/**
 * Spline degree p for ElementExtractors (matrix size (p+1) x (p+1)).
 * For knot vectors loaded from MFEM NURBS meshes, GetOrder() matches
 * the degree used by ElementExtractors (verified on segment-nurbs.mesh).
 */
inline int SplineDegreeFromKV(const mfem::KnotVector &kv)
{
    return kv.GetOrder();
}

void TestCartesianMesh();
void TestNURBSMesh();
void TestBSplineCurveReduction();
void TestBezierDegreeReduction();
void TestNURBSDegreeReduction();
void TestCurveDegreeReduction();
void TestSurfaceDegreeReduction();
void TestSurfaceMultistepReduction();
void TestTSpline();

#ifdef MFEM_RAYTRACING_ENABLE_EMBREE
void TestBilinearIntersect();
void TestEmbreeRayTracer();
#endif

#endif
