#include "ien_generator.hpp"
#include "test_helpers.hpp"

#include <algorithm>

namespace
{

// Uniform p=q=1, open knots → one element, four corner control points 0..3.
void TestBilinearSingleElementIEN()
{
    const std::vector<double> knots = {0.0, 0.0, 1.0, 1.0};
    const IENGenerator gen(knots, knots, 1, 1);

    CHECK(gen.GetNumElementsU() == 1);
    CHECK(gen.GetNumElementsV() == 1);
    CHECK(gen.GetTotalElements() == 1);
    CHECK(gen.GetLocalNodesPerElement() == 4);
    CHECK(gen.GetNumControlPointsU() == 2);
    CHECK(gen.GetNumControlPointsV() == 2);

    CHECK(gen.GetGlobalDof(0, 0) == 0);
    CHECK(gen.GetGlobalDof(0, 1) == 1);
    CHECK(gen.GetGlobalDof(0, 2) == 2);
    CHECK(gen.GetGlobalDof(0, 3) == 3);

    const mfem::Table table = gen.MakeElementToDofTable();
    CHECK(table.Size() == 1);
    CHECK(table.RowSize(0) == 4);
    const int *dofs = table.GetRow(0);
    CHECK(dofs[0] == 0);
    CHECK(dofs[1] == 1);
    CHECK(dofs[2] == 2);
    CHECK(dofs[3] == 3);
}

// square-nurbs.mesh: one patch, bilinear in ξ and η.
void TestIENFromSquareNURBSPatch()
{
    mfem::Mesh mesh("meshes/iga/square-nurbs.mesh", 1, 1);
    CHECK(mesh.NURBSext != nullptr);

    const IENGenerator gen = IENGeneratorFromPatch(mesh, 0);
    CHECK(gen.GetTotalElements() == 1);
    CHECK(gen.GetLocalNodesPerElement() == 4);

    mfem::Array<const mfem::KnotVector *> kv;
    mesh.NURBSext->GetPatchKnotVectors(0, kv);
    CHECK(kv.Size() >= 2);

    const IENGenerator gen_kv(*kv[0], *kv[1]);
    CHECK(gen_kv.GetTotalElements() == gen.GetTotalElements());
    CHECK(gen.GetGlobalDof(0, 0) == gen_kv.GetGlobalDof(0, 0));
}

// Two knot spans in u and v (quadratic open knots) → 2×2 = 4 elements, 9 nodes each.
void TestQuadraticTwoByTwoElements()
{
    const std::vector<double> knots = {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const IENGenerator gen(knots, knots, 2, 2);

    CHECK(gen.GetNumElementsU() == 2);
    CHECK(gen.GetNumElementsV() == 2);
    CHECK(gen.GetTotalElements() == 4);
    CHECK(gen.GetLocalNodesPerElement() == 9);
    CHECK(gen.GetNumControlPointsU() == 4);
    CHECK(gen.GetNumControlPointsV() == 4);

    // First element: u/v span at cp 0; corners (i,j) = (0,0),(2,0),(0,2),(2,2).
    CHECK(gen.GetGlobalDof(0, 0) == 0);
    CHECK(gen.GetGlobalDof(0, 2) == 2);
    CHECK(gen.GetGlobalDof(0, 6) == 8);
    CHECK(gen.GetGlobalDof(0, 8) == 10);
}

// Knots {0,0,0,1/3,2/3,1,1,1}, quadratic in u and v → 3×3 elements, max A = 24.
void TestQuadraticThreeSpanKnotVectorIEN()
{
    const std::vector<double> knots = {
        0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0};
    const IENGenerator gen(knots, knots, 2, 2);

    CHECK(gen.GetNumElementsU() == 3);
    CHECK(gen.GetNumElementsV() == 3);
    CHECK(gen.GetTotalElements() == 9);
    CHECK(gen.GetNumControlPointsU() == 5);
    CHECK(gen.GetNumControlPointsV() == 5);

    int max_A = -1;
    const auto ien = gen.Generate2DIEN();
    for (const auto &row : ien)
    {
        for (const int a : row)
        {
            max_A = std::max(max_A, a);
        }
    }
    CHECK(max_A == 24);
}

}  // namespace

void TestIENGenerator()
{
    TestBilinearSingleElementIEN();
    TestIENFromSquareNURBSPatch();
    TestQuadraticTwoByTwoElements();
    TestQuadraticThreeSpanKnotVectorIEN();
}
