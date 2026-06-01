#include "element_extractor.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <vector>

namespace
{

bool IsIdentity(const mfem::DenseMatrix &mat, double tol = 1e-12)
{
    if (mat.Height() != mat.Width())
    {
        return false;
    }
    for (int i = 0; i < mat.Height(); ++i)
    {
        for (int j = 0; j < mat.Width(); ++j)
        {
            const double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(mat(i, j) - expected) > tol)
            {
                return false;
            }
        }
    }
    return true;
}

void CheckMatrixEntry(const mfem::DenseMatrix &mat,
                      int row,
                      int col,
                      double expected,
                      double tol = 1e-12)
{
    CHECK(row >= 0 && row < mat.Height());
    CHECK(col >= 0 && col < mat.Width());
    CHECK_NEAR(mat(row, col), expected, tol);
}

// One knot span {0,0,0,1,1,1}: Algorithm 1 returns 2 passes, PerSpan returns 1.
void TestExtractorsPerSpanSingleQuadraticSpan()
{
    const std::vector<double> knots = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const auto all = ElementExtractors(knots, 2);
    const auto per_span = ElementExtractorsPerSpan(knots, 2, 1);

    CHECK(static_cast<int>(all.size()) == 2);
    CHECK(static_cast<int>(per_span.size()) == 1);
    CHECK(per_span[0].matrix.Height() == 3);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            CHECK_NEAR(per_span[0].matrix(i, j), all[0].matrix(i, j), 1e-12);
        }
    }
}

// Hand-crafted quadratic knot vector yields known 3x3 extraction matrices.
void TestQuadraticThreeElementKnotVector()
{
    const std::vector<double> knots = {
        0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0};
    const auto ops = ElementExtractors(knots, 2);

    CHECK(static_cast<int>(ops.size()) >= 2);
    CHECK(ops[0].matrix.Height() == 3);
    CHECK(ops[0].matrix.Width() == 3);
    CHECK(ops[1].matrix.Height() == 3);
    CHECK(ops[1].matrix.Width() == 3);

    // C0 = [[1, 0, 0], [0, 1, 1/2], [0, 0, 1/2]]
    CheckMatrixEntry(ops[0].matrix, 0, 0, 1.0);
    CheckMatrixEntry(ops[0].matrix, 0, 1, 0.0);
    CheckMatrixEntry(ops[0].matrix, 0, 2, 0.0);
    CheckMatrixEntry(ops[0].matrix, 1, 0, 0.0);
    CheckMatrixEntry(ops[0].matrix, 1, 1, 1.0);
    CheckMatrixEntry(ops[0].matrix, 1, 2, 0.5);
    CheckMatrixEntry(ops[0].matrix, 2, 0, 0.0);
    CheckMatrixEntry(ops[0].matrix, 2, 1, 0.0);
    CheckMatrixEntry(ops[0].matrix, 2, 2, 0.5);

    // C1 = [[1/2, 0, 0], [1/2, 1, 1/2], [0, 0, 1/2]]
    CheckMatrixEntry(ops[1].matrix, 0, 0, 0.5);
    CheckMatrixEntry(ops[1].matrix, 0, 1, 0.0);
    CheckMatrixEntry(ops[1].matrix, 0, 2, 0.0);
    CheckMatrixEntry(ops[1].matrix, 1, 0, 0.5);
    CheckMatrixEntry(ops[1].matrix, 1, 1, 1.0);
    CheckMatrixEntry(ops[1].matrix, 1, 2, 0.5);
    CheckMatrixEntry(ops[1].matrix, 2, 0, 0.0);
    CheckMatrixEntry(ops[1].matrix, 2, 1, 0.0);
    CheckMatrixEntry(ops[1].matrix, 2, 2, 0.5);
}

// Single clamped cubic span produces identity on the first 4x4 extractor.
void TestSingleBezierElement()
{
    const std::vector<double> knots = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
    const auto ops = ElementExtractors(knots, 3);
    CHECK(static_cast<int>(ops.size()) == 2);
    CHECK(ops[0].matrix.Height() == 4);
    CHECK(ops[0].matrix.Width() == 4);
    CHECK(IsIdentity(ops[0].matrix));
}

// Two cubic spans yield three extractors; the first is not identity.
void TestTwoElementKnotVector()
{
    const std::vector<double> knots = {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0};
    const auto ops = ElementExtractors(knots, 3);
    CHECK(static_cast<int>(ops.size()) == 3);
    CHECK(ops[0].matrix.Height() == 4);
    CHECK(!IsIdentity(ops[0].matrix));
    CHECK(ops[1].matrix.Height() == 4);
}

// Linear knot vector with an interior knot produces three 2x2 extractors.
void TestLinearTwoElements()
{
    const std::vector<double> knots = {0.0, 0.0, 0.5, 1.0, 1.0};
    const auto ops = ElementExtractors(knots, 1);
    CHECK(static_cast<int>(ops.size()) == 3);
    CHECK(ops[0].matrix.Height() == 2);
    CHECK(ops[1].matrix.Height() == 2);
    CHECK(IsIdentity(ops[0].matrix) || !IsIdentity(ops[1].matrix));
}

// MFEM KnotVector order on segment-nurbs.mesh matches ElementExtractors degree p=1.
void TestSplineDegreeConventionFromSegmentNURBS()
{
    mfem::Mesh mesh("meshes/iga/segment-nurbs.mesh", 1, 1);
    CHECK(mesh.NURBSext != nullptr);

    const mfem::KnotVector *kv = mesh.NURBSext->GetKnotVector(0);
    CHECK(kv != nullptr);
    CHECK(kv->Size() == 4);
    CHECK_NEAR((*kv)[0], 0.0, 1e-12);
    CHECK_NEAR((*kv)[3], 1.0, 1e-12);

    const int p = SplineDegreeFromKV(*kv);
    CHECK(p == 1);
    CHECK(kv->GetOrder() == 1);
}

// Knots read from segment-nurbs.mesh give two extractors with identity on the first.
void TestExtractorFromSegmentNURBS()
{
    mfem::Mesh mesh("meshes/iga/segment-nurbs.mesh", 1, 1);
    const mfem::KnotVector *kv = mesh.NURBSext->GetKnotVector(0);
    const std::vector<double> knots = KnotsFromKV(*kv);
    const int p = SplineDegreeFromKV(*kv);

    const auto ops = ElementExtractors(knots, p);
    CHECK(static_cast<int>(ops.size()) == 2);
    CHECK(ops[0].matrix.Height() == p + 1);
    CHECK(ops[0].matrix.Width() == p + 1);
    CHECK(IsIdentity(ops[0].matrix));
}

// MFEM-sourced knots on segment-nurbs.mesh match hand-built {0,0,1,1} extraction matrices.
void TestExtractorMatchesHandLinear()
{
    const std::vector<double> hand_knots = {0.0, 0.0, 1.0, 1.0};
    const auto hand_ops = ElementExtractors(hand_knots, 1);

    mfem::Mesh mesh("meshes/iga/segment-nurbs.mesh", 1, 1);
    const mfem::KnotVector *kv = mesh.NURBSext->GetKnotVector(0);
    const auto mfem_ops = ElementExtractors(KnotsFromKV(*kv), SplineDegreeFromKV(*kv));

    CHECK(static_cast<int>(hand_ops.size()) == static_cast<int>(mfem_ops.size()));
    for (std::size_t e = 0; e < hand_ops.size(); ++e)
    {
        CHECK(hand_ops[e].matrix.Height() == mfem_ops[e].matrix.Height());
        for (int i = 0; i < hand_ops[e].matrix.Height(); ++i)
        {
            for (int j = 0; j < hand_ops[e].matrix.Width(); ++j)
            {
                CHECK_NEAR(hand_ops[e].matrix(i, j), mfem_ops[e].matrix(i, j), 1e-12);
            }
        }
    }
}

// Each parametric direction on square-nurbs.mesh matches the 1D segment extraction.
void TestExtractorSquareNURBSParamDirs()
{
    mfem::Mesh mesh("meshes/iga/square-nurbs.mesh", 1, 1);
    CHECK(mesh.NURBSext != nullptr);

    mfem::Array<const mfem::KnotVector *> patch_kv;
    mesh.NURBSext->GetPatchKnotVectors(0, patch_kv);
    CHECK(patch_kv.Size() >= 1);

    const std::vector<double> segment_knots = {0.0, 0.0, 1.0, 1.0};
    const auto segment_ops = ElementExtractors(segment_knots, 1);

    for (int d = 0; d < patch_kv.Size(); ++d)
    {
        const mfem::KnotVector *kv = patch_kv[d];
        const auto ops = ElementExtractors(KnotsFromKV(*kv), SplineDegreeFromKV(*kv));

        CHECK(static_cast<int>(ops.size()) == static_cast<int>(segment_ops.size()));
        CHECK(IsIdentity(ops[0].matrix));
        CHECK(ops[0].matrix.Height() == segment_ops[0].matrix.Height());
    }
}

}  // namespace

void TestElementExtractor()
{
    TestExtractorsPerSpanSingleQuadraticSpan();
    TestSplineDegreeConventionFromSegmentNURBS();
    TestExtractorFromSegmentNURBS();
    TestExtractorMatchesHandLinear();
    TestExtractorSquareNURBSParamDirs();
    TestSingleBezierElement();
    TestTwoElementKnotVector();
    TestLinearTwoElements();
    TestQuadraticThreeElementKnotVector();
}
