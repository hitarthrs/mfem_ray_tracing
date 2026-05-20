#include "intersectAABB.hpp"
#include "ray.hpp"
#include "test_helpers.hpp"

namespace
{

// One quad cell on [0,1] x [0,1].
mfem::Mesh UnitSquareMesh()
{
    return mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
}

void TestHitCenterRay()
{
    mfem::Mesh mesh = UnitSquareMesh();

    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin[0] = 0.5;
    origin[1] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;

    Ray ray(origin, direction);
    double t_entry = 0.0;
    double t_exit = 0.0;

    CHECK(IntersectAABB(ray, mesh, t_entry, t_exit));
    CHECK_NEAR(t_entry, 0.0, 1e-10);
    CHECK_NEAR(t_exit, 0.5, 1e-10);
}

void TestMissOutsideBox()
{
    mfem::Mesh mesh = UnitSquareMesh();

    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin[0] = 2.0;
    origin[1] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;

    Ray ray(origin, direction);
    double t_entry = 0.0;
    double t_exit = 0.0;

    CHECK(!IntersectAABB(ray, mesh, t_entry, t_exit));
}

void TestParallelAxisInside()
{
    mfem::Mesh mesh = UnitSquareMesh();

    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin[0] = 0.5;
    origin[1] = 0.5;
    direction[0] = 0.0;
    direction[1] = 1.0;

    Ray ray(origin, direction);
    double t_entry = 0.0;
    double t_exit = 0.0;

    CHECK(IntersectAABB(ray, mesh, t_entry, t_exit));
    CHECK_NEAR(t_entry, 0.0, 1e-10);   // clipped from t = -0.5 at y = 0
    CHECK_NEAR(t_exit, 0.5, 1e-10);    // leaves box at y = 1 from center (0.5, 0.5)
}

void TestClipToRaySegment()
{
    mfem::Mesh mesh = UnitSquareMesh();

    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin[0] = 0.5;
    origin[1] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;

    Ray ray(origin, direction);
    ray.SetTMin(0.2);
    ray.SetTMax(0.4);

    double t_entry = 0.0;
    double t_exit = 0.0;

    CHECK(IntersectAABB(ray, mesh, t_entry, t_exit));
    CHECK_NEAR(t_entry, 0.2, 1e-10);
    CHECK_NEAR(t_exit, 0.4, 1e-10);
}

}  // namespace

void TestIntersectAABB()
{
    TestHitCenterRay();
    TestMissOutsideBox();
    TestParallelAxisInside();
    TestClipToRaySegment();
}
