#include "ray.hpp"
#include "ray_tracing.hpp"
#include "test_helpers.hpp"

namespace
{

mfem::Mesh Mesh2x1UnitSquare()
{
    return mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
}

double MinCellSpacing2D(const mfem::Mesh &mesh, int nx, int ny, double sx, double sy)
{
    const double hx = sx / nx;
    const double hy = sy / ny;
    return (hx < hy) ? hx : hy;
}

// Horizontal ray crosses two quads in a 2x1 mesh with expected cell indices.
void TestTraceTwoCellsAlongX()
{
    mfem::Mesh mesh = Mesh2x1UnitSquare();

    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin[0] = 0.25;
    origin[1] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;

    Ray ray(origin, direction);
    ray.SetTMin(0.0);
    ray.SetTMax(1.0);

    const double dt = 0.05 * MinCellSpacing2D(mesh, 2, 1, 1.0, 1.0);

    std::vector<RayCellCrossings> crossings;
    CHECK(TraceFindPoints(ray, mesh, dt, crossings));
    CHECK(crossings.size() == 2u);
    CHECK(crossings[0].cell_index == 0);
    CHECK(crossings[1].cell_index == 1);
    CHECK_NEAR(crossings[0].t_entry, 0.0, 1e-9);
    // t_exit is the first sample where the cell changes (<= face at t=0.5)
    CHECK(crossings[0].t_exit > 0.15);
    CHECK(crossings[0].t_exit < 0.55);
    CHECK_NEAR(crossings[1].t_exit, 0.75, 0.06);
}

// Ray confined to one quad produces a single crossing segment.
void TestTraceSingleCell()
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
        1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);

    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin[0] = 0.5;
    origin[1] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;

    Ray ray(origin, direction);
    ray.SetTMin(0.0);
    ray.SetTMax(1.0);

    const double dt = 0.1 * MinCellSpacing2D(mesh, 1, 1, 1.0, 1.0);

    std::vector<RayCellCrossings> crossings;
    CHECK(TraceFindPoints(ray, mesh, dt, crossings));
    CHECK(crossings.size() == 1u);
    CHECK(crossings[0].cell_index == 0);
    CHECK_NEAR(crossings[0].t_entry, 0.0, 1e-9);
    CHECK_NEAR(crossings[0].t_exit, 0.5, 0.06);
}

// Ray outside the mesh bbox yields no crossings.
void TestTraceMiss()
{
    mfem::Mesh mesh = Mesh2x1UnitSquare();

    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin[0] = 2.0;
    origin[1] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;

    Ray ray(origin, direction);

    std::vector<RayCellCrossings> crossings;
    CHECK(!TraceFindPoints(ray, mesh, 0.05, crossings));
    CHECK(crossings.empty());
}

// FindPoints locates an interior point in cell 0 with reference coords in [0, 1]^2.
void TestFindPointsUnitCellCenter()
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
        1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);

    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin[0] = 0.25;
    origin[1] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;

    const Ray ray(origin, direction);
    int elem = -1;
    mfem::IntegrationPoint ip;
    CHECK(FindElemAndIP(ray, mesh, 0.0, elem, ip));
    CHECK(elem == 0);
    CHECK(ip.x >= 0.0 && ip.x <= 1.0);
    CHECK(ip.y >= 0.0 && ip.y <= 1.0);
}

// Points just left and right of x = 0.5 map to cells 0 and 1.
void TestFindPointsTwoCellBoundary()
{
    mfem::Mesh mesh = Mesh2x1UnitSquare();

    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin[0] = 0.49;
    origin[1] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;

    const Ray ray(origin, direction);
    int elem = -1;
    mfem::IntegrationPoint ip;
    CHECK(FindElemAndIP(ray, mesh, 0.0, elem, ip));
    CHECK(elem == 0);

    origin[0] = 0.51;
    const Ray ray_right(origin, direction);
    CHECK(FindElemAndIP(ray_right, mesh, 0.0, elem, ip));
    CHECK(elem == 1);
}

// 3D hex mesh: ray along x visits two cells (smoke test for SpaceDimension 3).
void TestTrace3DSmoke()
{
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        2, 1, 1, mfem::Element::HEXAHEDRON, true, 1.0, 1.0, 1.0);

    mfem::Vector origin(3);
    mfem::Vector direction(3);
    origin[0] = 0.25;
    origin[1] = 0.5;
    origin[2] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;
    direction[2] = 0.0;

    Ray ray(origin, direction);
    ray.SetTMin(0.0);
    ray.SetTMax(1.0);

    std::vector<RayCellCrossings> crossings;
    CHECK(TraceFindPoints(ray, mesh, 0.05, crossings));
    CHECK(crossings.size() == 2u);
    CHECK(crossings[0].cell_index == 0);
    CHECK(crossings[1].cell_index == 1);
}

// Crossing segments use distinct cells with non-decreasing entry times.
void TestTraceCellIndicesMonotonic()
{
    mfem::Mesh mesh = Mesh2x1UnitSquare();

    mfem::Vector origin(2);
    mfem::Vector direction(2);
    origin[0] = 0.1;
    origin[1] = 0.5;
    direction[0] = 1.0;
    direction[1] = 0.0;

    Ray ray(origin, direction);
    ray.SetTMin(0.0);
    ray.SetTMax(1.0);

    std::vector<RayCellCrossings> crossings;
    CHECK(TraceFindPoints(ray, mesh, 0.05, crossings));

    for (std::size_t k = 1; k < crossings.size(); ++k)
    {
        CHECK(crossings[k].cell_index != crossings[k - 1].cell_index);
        CHECK(crossings[k].t_entry >= crossings[k - 1].t_entry);
    }
}

}  // namespace

void TestRayTrace()
{
    TestFindPointsUnitCellCenter();
    TestFindPointsTwoCellBoundary();
    TestTrace3DSmoke();
    TestTraceTwoCellsAlongX();
    TestTraceSingleCell();
    TestTraceMiss();
    TestTraceCellIndicesMonotonic();
}
