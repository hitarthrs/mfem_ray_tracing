#include "test_helpers.hpp"

#include <iostream>

int main()
{
    std::cout << "Running Cartesian mesh tests..." << std::endl;
    TestCartesianMesh();

    std::cout << "Running NURBS mesh tests..." << std::endl;
    TestNURBSMesh();

    std::cout << "Running B-spline curve degree reduction tests..." << std::endl;
    TestBSplineCurveReduction();

    std::cout << "Running Bezier degree reduction tests..." << std::endl;
    TestBezierDegreeReduction();

    std::cout << "Running NURBS degree reduction tests..." << std::endl;
    TestNURBSDegreeReduction();

    std::cout << "Running curve degree reduction tests..." << std::endl;
    TestCurveDegreeReduction();

    std::cout << "Running surface degree reduction tests..." << std::endl;
    TestSurfaceDegreeReduction();

    std::cout << "Running multi-step surface reduction tests..." << std::endl;
    TestSurfaceMultistepReduction();

#ifdef MFEM_RAYTRACING_ENABLE_EMBREE
    std::cout << "Running Embree bilinear patch intersection tests..." << std::endl;
    TestBilinearIntersect();

    std::cout << "Running Embree ray tracer tests..." << std::endl;
    TestEmbreeRayTracer();
#endif

    if (g_failures == 0)
    {
        std::cout << "All tests passed." << std::endl;
        return 0;
    }

    std::cerr << g_failures << " test(s) failed." << std::endl;
    return 1;
}
