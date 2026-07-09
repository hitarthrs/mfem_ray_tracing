#ifndef TEST_HELPERS_HPP
#define TEST_HELPERS_HPP

#include "mfem.hpp"
#include "ray.hpp"

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

/** Single-sample wrapper around mfem::Mesh::FindPoints for tests. */
inline bool FindElemAndIP(const Ray &ray,
                          mfem::Mesh &mesh,
                          double t,
                          int &elem,
                          mfem::IntegrationPoint &ip)
{
    const int sd = mesh.SpaceDimension();
    mfem::DenseMatrix point_mat(sd, 1);
    mfem::Vector point;
    ray.Evaluate(t, point);
    for (int d = 0; d < sd; ++d)
    {
        point_mat(d, 0) = point(d);
    }

    mfem::Array<int> elem_arr(1);
    mfem::Array<mfem::IntegrationPoint> ips(1);
    mesh.FindPoints(point_mat, elem_arr, ips);

    elem = elem_arr[0];
    ip = ips[0];
    return elem >= 0;
}

void TestRay();
void TestIntersectAABB();
void TestCartesianMesh();
void TestRayTrace();
void TestNURBSMesh();
void TestBilinearIntersection();
void TestElementExtractor();
void TestKroneckerProduct();
void TestIENGenerator();
void TestIDGenerator();
void TestLMGenerator();
void TestIGAReferenceConnectivity();
void TestBdrPatchExtractor();
void TestBSplineCurveReduction();
void TestBezierDegreeReduction();
void TestNURBSDegreeReduction();
void TestCurveDegreeReduction();
void TestSurfaceDegreeReduction();
void TestSurfaceMultistepReduction();

#ifdef MFEM_RAYTRACING_ENABLE_EMBREE
void TestBilinearIntersect();
void TestEmbreeRayTracer();
#endif

#endif
