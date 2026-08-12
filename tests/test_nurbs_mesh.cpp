#include "mfem_raytracing/mesh/nurbs_mesh_operations.hpp"
#include "test_helpers.hpp"

#include <iostream>
#include <string>

namespace
{

double BoxVolume(const ElementBoundingBox &bbox)
{
    double vol = 1.0;
    for (int d = 0; d < bbox.min.Size(); ++d)
    {
        vol *= (bbox.max(d) - bbox.min(d));
    }
    return vol;
}

bool IsInsideGlobalBox(const ElementBoundingBox &elem,
                       const mfem::Vector &global_min,
                       const mfem::Vector &global_max,
                       double tol)
{
    for (int d = 0; d < elem.min.Size(); ++d)
    {
        if (elem.min(d) < global_min(d) - tol || elem.max(d) > global_max(d) + tol)
        {
            return false;
        }
    }
    return true;
}

// Per-element AABBs match NE, lie inside the global bbox, and have positive volume.
void TestElementBoundingBoxes(mfem::Mesh &mesh, int expected_ne)
{
    const auto bboxes = GetElementBoundingBoxes(mesh);
    CHECK(static_cast<int>(bboxes.size()) == expected_ne);
    CHECK(static_cast<int>(bboxes.size()) == mesh.GetNE());

    mfem::Vector global_min, global_max;
    mesh.GetBoundingBox(global_min, global_max);

    for (const auto &bbox : bboxes)
    {
        CHECK(bbox.min.Size() == bbox.max.Size());
        for (int d = 0; d < bbox.min.Size(); ++d)
        {
            CHECK(bbox.min(d) <= bbox.max(d));
        }

        CHECK(IsInsideGlobalBox(bbox, global_min, global_max, 1e-10));
        CHECK(BoxVolume(bbox) > 0.0);
    }
}

bool IsInsideElementBox(const ElementBoundingBox &bbox,
                        const mfem::Vector &point,
                        double tol)
{
    for (int d = 0; d < point.Size(); ++d)
    {
        if (point(d) < bbox.min(d) - tol || point(d) > bbox.max(d) + tol)
        {
            return false;
        }
    }
    return true;
}

// 1D segment NURBS mesh is detected and has one valid element bounding box.
void TestSegmentNURBSMesh()
{
    mfem::Mesh mesh("meshes/iga/segment-nurbs.mesh", 1, 1);
    CHECK(IsNURBSMesh(mesh));
    CHECK(mesh.GetNE() == 1);
    TestElementBoundingBoxes(mesh, 1);
}

// MFEM exposes the expected clamped knot vector on segment-nurbs.mesh.
void TestKnotVectorSmokeSegment()
{
    mfem::Mesh mesh("meshes/iga/segment-nurbs.mesh", 1, 1);
    const mfem::KnotVector *kv = mesh.NURBSext->GetKnotVector(0);
    CHECK(kv != nullptr);
    CHECK(kv->Size() == 4);
    CHECK_NEAR((*kv)[0], 0.0, 1e-12);
    CHECK_NEAR((*kv)[3], 1.0, 1e-12);
}

// Finer geometry refinement does not enlarge the sampled element bounding box.
void TestRefinementAffectsBBox()
{
    mfem::Mesh mesh("meshes/iga/square-nurbs.mesh", 1, 1);
    const auto bboxes_ref1 = GetElementBoundingBoxes(mesh, 1);
    const auto bboxes_ref2 = GetElementBoundingBoxes(mesh, 2);
    CHECK(static_cast<int>(bboxes_ref1.size()) >= 1);
    CHECK(bboxes_ref1.size() == bboxes_ref2.size());

    const double vol_ref1 = BoxVolume(bboxes_ref1[0]);
    const double vol_ref2 = BoxVolume(bboxes_ref2[0]);
    CHECK(vol_ref1 > 0.0);
    CHECK(vol_ref2 <= vol_ref1 + 1e-10);
}

// A ref-corner point from ElementTransformation lies inside that element's AABB.
void TestElementTransformInsideBBox()
{
    mfem::Mesh mesh("meshes/iga/square-nurbs.mesh", 1, 1);
    const auto bboxes = GetElementBoundingBoxes(mesh);

    mfem::ElementTransformation *T = mesh.GetElementTransformation(0);
    mfem::IntegrationPoint ip;
    ip.x = 0.0;
    ip.y = 0.0;
    T->SetIntPoint(&ip);

    mfem::Vector point(3);
    T->Transform(ip, point);

    CHECK(IsInsideElementBox(bboxes[0], point, 1e-8));
}

}  // namespace

void TestNURBSMesh()
{
    TestSegmentNURBSMesh();
    TestKnotVectorSmokeSegment();
    TestRefinementAffectsBBox();
    TestElementTransformInsideBBox();

    const std::string pipe_mesh = "meshes/iga/pipe-nurbs.mesh";
    const std::string cart_mesh =
        "meshes/cartesian/cartesian_mesh_3D_n10x10x10_s2x1x1.mesh";

    // Pipe NURBS patch mesh: IsNURBSMesh, patch count, and per-element bboxes.
    mfem::Mesh pipe(pipe_mesh.c_str(), 1, 1);
    CHECK(IsNURBSMesh(pipe));
    CHECK(pipe.NURBSext != nullptr);
    CHECK(pipe.NURBSext->GetNP() == 4);
    TestElementBoundingBoxes(pipe, 8);

    // Large Cartesian mesh: not NURBS, but bbox count matches element count.
    mfem::Mesh cart(cart_mesh.c_str(), 1, 1);
    CHECK(!IsNURBSMesh(cart));
    CHECK(cart.NURBSext == nullptr);
    TestElementBoundingBoxes(cart, 1000);

    const auto pipe_bboxes = GetElementBoundingBoxes(pipe);
    std::cout << "  pipe-nurbs.mesh: IsNURBSMesh=" << IsNURBSMesh(pipe)
              << ", patches=" << pipe.NURBSext->GetNP()
              << ", NE=" << pipe.GetNE() << std::endl;
    std::cout << "  pipe element bounding boxes:" << std::endl;
    PrintElementBoundingBoxes(pipe_bboxes, std::cout);

    std::cout << "  cartesian mesh:  IsNURBSMesh=" << IsNURBSMesh(cart)
              << ", NE=" << cart.GetNE()
              << ", bbox count=" << GetElementBoundingBoxes(cart).size()
              << std::endl;
}
