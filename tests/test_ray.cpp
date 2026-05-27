#include "ray.hpp"
#include "test_helpers.hpp"

#include <cmath>

namespace
{

// Evaluate(0) returns the ray origin unchanged.
void TestEvaluateAtOrigin()
{
    mfem::Vector origin(3);
    mfem::Vector direction(3);
    origin = 1.0;
    origin[0] = 2.0;
    origin[1] = -1.0;
    origin[2] = 0.5;
    direction = 0.0;
    direction[0] = 3.0;  // becomes unit +x after normalization

    const Ray ray(origin, direction);
    mfem::Vector point(3);
    ray.Evaluate(0.0, point);

    CHECK_NEAR(point[0], 2.0, 1e-12);
    CHECK_NEAR(point[1], -1.0, 1e-12);
    CHECK_NEAR(point[2], 0.5, 1e-12);
}

// Evaluate(t) advances along the normalized direction by distance t.
void TestEvaluateAlongDirection()
{
    mfem::Vector origin(3);
    mfem::Vector direction(3);
    origin = 0.0;
    direction = 0.0;
    direction[1] = 2.0;  // unit +y

    const Ray ray(origin, direction);
    mfem::Vector point(3);
    ray.Evaluate(2.5, point);

    CHECK_NEAR(point[0], 0.0, 1e-12);
    CHECK_NEAR(point[1], 2.5, 1e-12);
    CHECK_NEAR(point[2], 0.0, 1e-12);
}

// Constructor stores a unit-length direction vector.
void TestDirectionNormalized()
{
    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin = 0.0;
    direction[0] = 3.0;
    direction[1] = 4.0;

    const Ray ray(origin, direction);
    const mfem::Vector &d = ray.GetDirection();

    CHECK_NEAR(d.Norml2(), 1.0, 1e-12);
    CHECK_NEAR(d[0], 0.6, 1e-12);
    CHECK_NEAR(d[1], 0.8, 1e-12);
}

// Default active segment is [0, 1] and weight is preserved.
void TestDefaultSegmentAndWeight()
{
    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin = 0.0;
    direction[0] = 1.0;

    const Ray ray(origin, direction, 0.75);

    CHECK_NEAR(ray.GetTMin(), 0.0, 1e-12);
    CHECK_NEAR(ray.GetTMax(), 1.0, 1e-12);
    CHECK_NEAR(ray.GetWeight(), 0.75, 1e-12);
}

}  // namespace

void TestRay()
{
    TestEvaluateAtOrigin();
    TestEvaluateAlongDirection();
    TestDirectionNormalized();
    TestDefaultSegmentAndWeight();
}
