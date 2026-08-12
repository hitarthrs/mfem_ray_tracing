#include "intersectAABB.hpp"
#include "ray.hpp"
#include "test_helpers.hpp"

#include <algorithm>

namespace
{

// One quad cell on [0,1] x [0,1].
mfem::Mesh UnitSquareMesh()
{
    return mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
}

// Ray from cell center along +x hits the unit square with t in [0, 0.5].
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

// Ray starting outside the mesh bounding box is rejected.
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

// Axis-aligned ray inside the box is clipped to the default [t_min, t_max] segment.
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

// 3D ray through the center of a unit hex exits the global bbox at t = 0.5.
void TestHit3DUnitCube()
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        1, 1, 1, mfem::Element::HEXAHEDRON, true, 1.0, 1.0, 1.0);

    mfem::Vector origin(3);
    mfem::Vector direction(3);
    origin[0] = 0.5;
    origin[1] = 0.5;
    origin[2] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;
    direction[2] = 0.0;

    Ray ray(origin, direction);
    double t_entry = 0.0;
    double t_exit = 0.0;

    CHECK(IntersectAABB(ray, mesh, t_entry, t_exit));
    CHECK_NEAR(t_entry, 0.0, 1e-10);
    CHECK_NEAR(t_exit, 0.5, 1e-10);
}

// Ray through the bbox center of a loaded Cartesian mesh intersects the global AABB.
void TestHitLoadedCartesianMesh()
{
    mfem::Mesh mesh("meshes/cartesian/cartesian_mesh_3D_n10x10x10_s2x1x1.mesh", 1, 1);

    mfem::Vector bbox_min, bbox_max;
    mesh.GetBoundingBox(bbox_min, bbox_max);

    mfem::Vector origin(3);
    mfem::Vector direction(3);
    for (int d = 0; d < 3; ++d)
    {
        origin[d] = 0.5 * (bbox_min(d) + bbox_max(d));
        direction[d] = (d == 0) ? 1.0 : 0.0;
    }

    Ray ray(origin, direction);
    double t_entry = 0.0;
    double t_exit = 0.0;

    CHECK(IntersectAABB(ray, mesh, t_entry, t_exit));
    CHECK(t_exit > t_entry);
    CHECK(t_exit > 0.0);
}

// IntersectAABB matches slab times from mesh.GetBoundingBox, clipped to the ray segment.
void TestBBoxConsistentWithMFEM()
{
    mfem::Mesh mesh = UnitSquareMesh();

    mfem::Vector bbox_min, bbox_max;
    mesh.GetBoundingBox(bbox_min, bbox_max);

    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin[0] = 0.5;
    origin[1] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;

    const double t_entry_box =
        (bbox_min[0] - origin[0]) / direction[0];
    const double t_exit_box =
        (bbox_max[0] - origin[0]) / direction[0];

    Ray ray(origin, direction);
    const double t_entry_expected = std::max(t_entry_box, ray.GetTMin());
    const double t_exit_expected = std::min(t_exit_box, ray.GetTMax());

    double t_entry = 0.0;
    double t_exit = 0.0;

    CHECK(IntersectAABB(ray, mesh, t_entry, t_exit));
    CHECK_NEAR(t_entry, t_entry_expected, 1e-10);
    CHECK_NEAR(t_exit, t_exit_expected, 1e-10);
}

// Intersection interval is clipped to user SetTMin/SetTMax on the ray.
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
    TestHit3DUnitCube();
    TestHitLoadedCartesianMesh();
    TestBBoxConsistentWithMFEM();
    TestMissOutsideBox();
    TestParallelAxisInside();
    TestClipToRaySegment();
}
