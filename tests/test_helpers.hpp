#ifndef TEST_HELPERS_HPP
#define TEST_HELPERS_HPP

#include <cmath>
#include <iostream>
#include <string>

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

void TestRay();
void TestIntersectAABB();
void TestCartesianMesh();
void TestRayTrace();

#endif
