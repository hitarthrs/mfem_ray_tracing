#include "test_helpers.hpp"

#include <iostream>

int main()
{
    std::cout << "Running Ray tests..." << std::endl;
    TestRay();

    std::cout << "Running IntersectAABB tests..." << std::endl;
    TestIntersectAABB();

    std::cout << "Running Cartesian mesh tests..." << std::endl;
    TestCartesianMesh();

    std::cout << "Running ray trace tests..." << std::endl;
    TestRayTrace();

    std::cout << "Running NURBS mesh tests..." << std::endl;
    TestNURBSMesh();

    std::cout << "Running bilinear intersection tests..." << std::endl;
    TestBilinearIntersection();

    std::cout << "Running element extractor tests..." << std::endl;
    TestElementExtractor();

    if (g_failures == 0)
    {
        std::cout << "All tests passed." << std::endl;
        return 0;
    }

    std::cerr << g_failures << " test(s) failed." << std::endl;
    return 1;
}
