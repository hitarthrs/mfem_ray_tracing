#include "bilinear_intersection.hpp"
#include "ray.hpp"
#include "test_helpers.hpp"

#include <cmath>

namespace
{

mfem::Mesh UnitCubeMesh()
{
    return mfem::Mesh::MakeCartesian3D(
        1, 1, 1, mfem::Element::HEXAHEDRON, true, 1.0, 1.0, 1.0);
}

int FindBoundaryFaceOnPlane(const mfem::Mesh &mesh, int axis, double value, double tol = 1e-10)
{
    mfem::IsoparametricTransformation FTr;
    mfem::IntegrationPoint ip;
    mfem::Vector corner(3);

    for (int face = 0; face < mesh.GetNBE(); ++face)
    {
        mesh.GetBdrElementTransformation(face, &FTr);

        bool matches = true;
        const double corner_params[4][3] = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {1.0, 1.0, 0.0},
        };

        for (const auto &params : corner_params)
        {
            ip.Set3(params[0], params[1], params[2]);
            FTr.SetIntPoint(&ip);
            FTr.Transform(ip, corner);
            if (std::abs(corner(axis) - value) > tol)
            {
                matches = false;
                break;
            }
        }

        if (matches)
        {
            return face;
        }
    }

    return -1;
}

void TestHitFrontFace()
{
    mfem::Mesh mesh = UnitCubeMesh();
    const int z0_face = FindBoundaryFaceOnPlane(mesh, 2, 0.0);
    CHECK(z0_face >= 0);

    mfem::Vector origin(3);
    mfem::Vector direction(3);
    origin(0) = 0.5;
    origin(1) = 0.5;
    origin(2) = -1.0;
    direction(0) = 0.0;
    direction(1) = 0.0;
    direction(2) = 1.0;

    Ray ray(origin, direction);
    ray.SetTMin(0.0);
    ray.SetTMax(10.0);

    const FaceHitInformation hit = BilinearIntersection(ray, mesh, z0_face);
    CHECK(hit.hit);
    CHECK_NEAR(hit.t_intersection, 1.0, 1e-12);
    CHECK_NEAR(hit.u, 0.5, 1e-12);
    CHECK_NEAR(hit.v, 0.5, 1e-12);
    CHECK_NEAR(hit.local_coords(0), 0.5, 1e-12);
    CHECK_NEAR(hit.local_coords(1), 0.5, 1e-12);
    CHECK_NEAR(hit.local_coords(2), 0.0, 1e-12);
}

void TestMissOutsidePatch()
{
    mfem::Mesh mesh = UnitCubeMesh();
    const int z0_face = FindBoundaryFaceOnPlane(mesh, 2, 0.0);
    CHECK(z0_face >= 0);

    mfem::Vector origin(3);
    mfem::Vector direction(3);
    origin(0) = 2.0;
    origin(1) = 0.5;
    origin(2) = -1.0;
    direction(0) = 0.0;
    direction(1) = 0.0;
    direction(2) = 1.0;

    Ray ray(origin, direction);
    const FaceHitInformation hit = BilinearIntersection(ray, mesh, z0_face);
    CHECK(!hit.hit);
}

void TestClipTMax()
{
    mfem::Mesh mesh = UnitCubeMesh();
    const int z0_face = FindBoundaryFaceOnPlane(mesh, 2, 0.0);
    CHECK(z0_face >= 0);

    mfem::Vector origin(3);
    mfem::Vector direction(3);
    origin(0) = 0.5;
    origin(1) = 0.5;
    origin(2) = -1.0;
    direction(0) = 0.0;
    direction(1) = 0.0;
    direction(2) = 1.0;

    Ray ray(origin, direction);
    ray.SetTMin(0.0);
    ray.SetTMax(0.5);

    const FaceHitInformation hit = BilinearIntersection(ray, mesh, z0_face);
    CHECK(!hit.hit);
}

void TestObliqueHit()
{
    mfem::Mesh mesh = UnitCubeMesh();
    const int x0_face = FindBoundaryFaceOnPlane(mesh, 0, 0.0);
    CHECK(x0_face >= 0);

    mfem::Vector origin(3);
    mfem::Vector direction(3);
    origin(0) = -1.0;
    origin(1) = 0.25;
    origin(2) = 0.75;
    direction(0) = 1.0;
    direction(1) = 0.0;
    direction(2) = 0.0;

    Ray ray(origin, direction);
    ray.SetTMin(0.0);
    ray.SetTMax(10.0);

    const FaceHitInformation hit = BilinearIntersection(ray, mesh, x0_face);
    CHECK(hit.hit);
    CHECK_NEAR(hit.t_intersection, 1.0, 1e-12);
    CHECK_NEAR(hit.local_coords(0), 0.0, 1e-12);
    CHECK_NEAR(hit.local_coords(1), 0.25, 1e-12);
    CHECK_NEAR(hit.local_coords(2), 0.75, 1e-12);
    CHECK(hit.normal.Norml2() > 0.0);
}

void TestWarpedSaddleHit()
{
    mfem::Mesh mesh = UnitCubeMesh();
    
    // 1. Find the top Z=1 face BEFORE we warp it, while it is still perfectly flat
    // so our FindBoundaryFaceOnPlane utility can easily identify it.
    const int z1_face = FindBoundaryFaceOnPlane(mesh, 2, 1.0);
    CHECK(z1_face >= 0);

    // 2. Warp the mesh to create a true bilinear saddle!
    // We apply a transformation that twists the Z-axis coordinates based on X and Y.
    mesh.Transform([](const mfem::Vector& x, mfem::Vector& p) {
        p = x;
        // The top face z=1 becomes z = 1 + 0.5*x*y
        p(2) += 0.5 * x(0) * x(1); 
    });

    // 3. Fire a ray straight down at the center of this newly warped face
    mfem::Vector origin(3);
    mfem::Vector direction(3);
    
    origin(0) = 0.5;
    origin(1) = 0.5;
    origin(2) = 3.0; // Starting high above the mesh
    
    direction(0) = 0.0;
    direction(1) = 0.0;
    direction(2) = -1.0; // Pointing straight down

    Ray ray(origin, direction);
    ray.SetTMin(0.0);
    ray.SetTMax(10.0);

    // 4. Run the intersection
    const FaceHitInformation hit = BilinearIntersection(ray, mesh, z1_face);
    
    CHECK(hit.hit);
    
    // 5. Verify the exact math of the warp
    // The center of the reference face is at u = 0.5, v = 0.5.
    // In our physical space, X and Y are 0.5.
    // The warped Z at this exact spot is: 1.0 + (0.5 * 0.5 * 0.5) = 1.125
    // The ray travels from Z = 3.0 down to Z = 1.125, meaning travel distance t = 1.875
    
    CHECK_NEAR(hit.t_intersection, 1.875, 1e-12);
    CHECK_NEAR(hit.u, 0.5, 1e-12);
    CHECK_NEAR(hit.v, 0.5, 1e-12);
    
    CHECK_NEAR(hit.local_coords(0), 0.5, 1e-12);
    CHECK_NEAR(hit.local_coords(1), 0.5, 1e-12);
    CHECK_NEAR(hit.local_coords(2), 1.125, 1e-12);
    
    // Ensure it generated a valid surface normal
    CHECK(hit.normal.Norml2() > 0.0);
}

}  // namespace

void TestBilinearIntersection()
{
    TestHitFrontFace();
    TestMissOutsidePatch();
    TestClipTMax();
    TestObliqueHit();
    TestWarpedSaddleHit();
}
