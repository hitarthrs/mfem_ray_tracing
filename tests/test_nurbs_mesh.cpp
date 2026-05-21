#include "nurbs_mesh_operations.hpp"
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

void TestElementBoundingBoxes(mfem::Mesh &mesh, int expected_ne)
{
    const auto bboxes = GetElementBoundingBoxes(mesh);
    CHECK(static_cast<int>(bboxes.size()) == expected_ne);

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

}  // namespace

void TestNURBSMesh()
{
    const std::string pipe_mesh = "meshes/iga/pipe-nurbs.mesh";
    const std::string cart_mesh =
        "meshes/cartesian/cartesian_mesh_3D_n10x10x10_s2x1x1.mesh";

    mfem::Mesh pipe(pipe_mesh.c_str(), 1, 1);
    CHECK(IsNURBSMesh(pipe));
    CHECK(pipe.NURBSext != nullptr);
    CHECK(pipe.NURBSext->GetNP() == 4);
    TestElementBoundingBoxes(pipe, 8);

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
