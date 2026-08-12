#include "bdr_patch_extractor.hpp"
#include "element_extractor.hpp"
#include "kronecker_product.hpp"
#include "test_helpers.hpp"

#include <cmath>

namespace
{

void CheckKnotsEqual(const std::vector<double> &a, const std::vector<double> &b, double tol)
{
    CHECK(static_cast<int>(a.size()) == static_cast<int>(b.size()));
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        CHECK_NEAR(a[i], b[i], tol);
    }
}

void CheckExtractorsEqual(const std::vector<ElementExtractor> &a,
                         const std::vector<ElementExtractor> &b,
                         double tol)
{
    CHECK(static_cast<int>(a.size()) == static_cast<int>(b.size()));
    for (std::size_t e = 0; e < a.size(); ++e)
    {
        CHECK(a[e].matrix.Height() == b[e].matrix.Height());
        CHECK(a[e].matrix.Width() == b[e].matrix.Width());
        for (int i = 0; i < a[e].matrix.Height(); ++i)
        {
            for (int j = 0; j < a[e].matrix.Width(); ++j)
            {
                CHECK_NEAR(a[e].matrix(i, j), b[e].matrix(i, j), tol);
            }
        }
    }
}

bool IsIdentityN(const mfem::DenseMatrix &mat, double tol)
{
    if (mat.Height() != mat.Width() || mat.Height() <= 0)
    {
        return false;
    }
    const int n = mat.Height();
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
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

bool IsIdentity3(const mfem::DenseMatrix &mat, double tol)
{
    if (mat.Height() != 3 || mat.Width() != 3)
    {
        return false;
    }
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
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

void CheckDiagonalWeight(const mfem::SparseMatrix &W, int k, double expected, double tol)
{
    CHECK(k >= 0 && k < W.Height() && k < W.Width());
    CHECK_NEAR(W(k, k), expected, tol);
    for (int j = 0; j < W.Width(); ++j)
    {
        if (j != k)
        {
            CHECK_NEAR(W(k, j), 0.0, tol);
        }
    }
}

// NURBS boundary patch 14 on pipe-nurbs.mesh: two z slabs (boundary elements 21, 22).
void TestPipeBdrPatch14TwoBlocks()
{
    mfem::Mesh mesh("meshes/iga/pipe-nurbs.mesh", 1, 1);

    const BdrPatchBlockData data = ExtractBdrPatchBlocks(mesh, 14);

    CHECK(data.nu == 3);
    CHECK(data.nv == 3);
    CHECK(data.ne_u == 1);
    CHECK(data.ne_v == 2);
    CHECK(static_cast<int>(data.elements.size()) == 2);

    const double tol = 1e-10;
    const double w_corner = 1.0;
    const double w_edge = 0.7071067811865475244;
    const double w_mid = 0.5;

    // Block 0 (IJK -1, 0): z in [0, 10] slab — matches boundary element 21.
    {
        const mfem::DenseMatrix &P = data.elements[0].P;
        const mfem::SparseMatrix &W = data.elements[0].W;

        CHECK(P.Height() == 9);
        CHECK(P.Width() == 3);
        CHECK(W.Height() == 9);
        CHECK(W.Width() == 9);

        CHECK_NEAR(P(0, 0), 0.0, tol);
        CHECK_NEAR(P(0, 1), -2.0, tol);
        CHECK_NEAR(P(0, 2), 0.0, tol);

        CHECK_NEAR(P(2, 0), 2.0, tol);
        CHECK_NEAR(P(2, 1), 0.0, tol);
        CHECK_NEAR(P(2, 2), 0.0, tol);

        CHECK_NEAR(P(8, 0), 2.0, tol);
        CHECK_NEAR(P(8, 1), 0.0, tol);
        CHECK_NEAR(P(8, 2), 10.0, tol);

        CheckDiagonalWeight(W, 0, w_corner, tol);
        CheckDiagonalWeight(W, 1, w_edge, tol);
        CheckDiagonalWeight(W, 2, w_corner, tol);
        CheckDiagonalWeight(W, 8, w_corner, tol);
    }

    // Block 1 (IJK -1, 2): z in [10, 20] slab — matches boundary element 22.
    {
        const mfem::DenseMatrix &P = data.elements[1].P;
        const mfem::SparseMatrix &W = data.elements[1].W;

        CHECK_NEAR(P(0, 0), 0.0, tol);
        CHECK_NEAR(P(0, 1), -2.0, tol);
        CHECK_NEAR(P(0, 2), 10.0, tol);

        CHECK_NEAR(P(2, 0), 2.0, tol);
        CHECK_NEAR(P(2, 1), 0.0, tol);
        CHECK_NEAR(P(2, 2), 10.0, tol);

        CHECK_NEAR(P(6, 0), 10.0, tol);
        CHECK_NEAR(P(6, 1), -2.0, tol);
        CHECK_NEAR(P(6, 2), 20.0, tol);

        CheckDiagonalWeight(W, 0, w_corner, tol);
        CheckDiagonalWeight(W, 4, w_mid, tol);
        CheckDiagonalWeight(W, 6, w_corner, tol);
    }
}

// Patch 14: kv_u = {0,0,0,1,1,1}, kv_v = {0,0,0,1,1,2,2,2}, quadratic (p=2).
void TestPipeBdrPatch14KnotsAndExtractors()
{
    mfem::Mesh mesh("meshes/iga/pipe-nurbs.mesh", 1, 1);

    const BdrPatchBlockData data = ExtractBdrPatchBlocks(mesh, 14);
    const double tol = 1e-12;

    const std::vector<double> knots_u = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const std::vector<double> knots_v = {
        0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0};

    CheckKnotsEqual(data.param_u.knots, knots_u, tol);
    CheckKnotsEqual(data.param_v.knots, knots_v, tol);

    CHECK(data.param_u.degree == 2);
    CHECK(data.param_v.degree == 2);
    CHECK(data.param_u.ne == 1);
    CHECK(data.param_v.ne == 2);

    const auto ref_u = ElementExtractorsPerSpan(knots_u, 2, 1);
    const auto ref_v = ElementExtractorsPerSpan(knots_v, 2, 2);
    CheckExtractorsEqual(data.param_u.extractors, ref_u, tol);
    CheckExtractorsEqual(data.param_v.extractors, ref_v, tol);

    CHECK(static_cast<int>(data.param_u.extractors.size()) == 1);
    CHECK(static_cast<int>(data.param_v.extractors.size()) == 2);
    CHECK(data.param_u.extractors[0].matrix.Height() == 3);
    CHECK(data.param_v.extractors[0].matrix.Height() == 3);

    for (const auto &op : data.param_u.extractors)
    {
        CHECK(IsIdentity3(op.matrix, tol));
    }
    for (const auto &op : data.param_v.extractors)
    {
        CHECK(IsIdentity3(op.matrix, tol));
    }
}

// C_e = C_v ⊗ C_u; patch 14 has 1×2 spans, all 1D extractors identity → C_e = I_9.
void TestPipeBdrPatch14KroneckerExtraction()
{
    mfem::Mesh mesh("meshes/iga/pipe-nurbs.mesh", 1, 1);
    const BdrPatchBlockData data = ExtractBdrPatchBlocks(mesh, 14);
    const double tol = 1e-12;

    CHECK(static_cast<int>(data.param_u.extractors.size()) == 1);
    CHECK(static_cast<int>(data.param_v.extractors.size()) == 2);
    CHECK(static_cast<int>(data.elements.size()) == 2);

    CHECK(data.elements[0].span_u == 0);
    CHECK(data.elements[0].span_v == 0);
    CHECK(data.elements[1].span_u == 0);
    CHECK(data.elements[1].span_v == 1);

    const mfem::DenseMatrix ref0 =
        KroneckerProduct(data.param_v.extractors[0].matrix,
                         data.param_u.extractors[0].matrix);
    const mfem::DenseMatrix ref1 =
        KroneckerProduct(data.param_v.extractors[1].matrix,
                         data.param_u.extractors[0].matrix);

    CHECK(data.elements[0].C_e.Height() == 9);
    CHECK(data.elements[0].C_e.Width() == 9);
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 9; ++j)
        {
            CHECK_NEAR(data.elements[0].C_e(i, j), ref0(i, j), tol);
            CHECK_NEAR(data.elements[1].C_e(i, j), ref1(i, j), tol);
        }
    }

    CHECK(IsIdentityN(data.elements[0].C_e, tol));
    CHECK(IsIdentityN(data.elements[1].C_e, tol));
}

// w_b_e = C_e^T w; patch 14 has C_e = I so w_b_e equals spline weights on the diagonal of W.
void TestPipeBdrPatch14BezierWeights()
{
    mfem::Mesh mesh("meshes/iga/pipe-nurbs.mesh", 1, 1);
    const BdrPatchBlockData data = ExtractBdrPatchBlocks(mesh, 14);
    const double tol = 1e-12;

    const double w_corner = 1.0;
    const double w_edge = 0.7071067811865475244;
    const double w_mid = 0.5;

    CHECK(data.elements[0].w_b_e.Size() == 9);
    CHECK(data.elements[1].w_b_e.Size() == 9);

    for (int a = 0; a < 9; ++a)
    {
        CHECK_NEAR(data.elements[0].w_b_e(a), data.elements[0].W(a, a), tol);
        CHECK_NEAR(data.elements[1].w_b_e(a), data.elements[1].W(a, a), tol);
    }

    CHECK_NEAR(data.elements[0].w_b_e(0), w_corner, tol);
    CHECK_NEAR(data.elements[0].w_b_e(1), w_edge, tol);
    CHECK_NEAR(data.elements[1].w_b_e(4), w_mid, tol);

    for (int e = 0; e < 2; ++e)
    {
        const mfem::SparseMatrix &Wb = data.elements[e].W_b;
        const mfem::Vector &wb = data.elements[e].w_b_e;
        CHECK(Wb.Height() == 9);
        CHECK(Wb.Width() == 9);
        for (int a = 0; a < 9; ++a)
        {
            CHECK_NEAR(Wb(a, a), wb(a), tol);
            for (int b = 0; b < 9; ++b)
            {
                if (a != b)
                {
                    CHECK_NEAR(Wb(a, b), 0.0, tol);
                }
            }
        }
    }
}

}  // namespace

// Q_e = (W_b)^{-1} C_e^T W P; patch 14 has C_e = I and w_b = w → Q_e = P.
void TestPipeBdrPatch14BezierControlPoints()
{
    mfem::Mesh mesh("meshes/iga/pipe-nurbs.mesh", 1, 1);
    const BdrPatchBlockData data = ExtractBdrPatchBlocks(mesh, 14);
    const double tol = 1e-10;

    for (int e = 0; e < 2; ++e)
    {
        const mfem::DenseMatrix &P = data.elements[e].P;
        const mfem::DenseMatrix &Q = data.elements[e].Q_e;
        CHECK(Q.Height() == P.Height());
        CHECK(Q.Width() == P.Width());
        for (int a = 0; a < P.Height(); ++a)
        {
            for (int d = 0; d < P.Width(); ++d)
            {
                CHECK_NEAR(Q(a, d), P(a, d), tol);
            }
        }
    }
}

void TestBdrPatchExtractor()
{
    TestPipeBdrPatch14TwoBlocks();
    TestPipeBdrPatch14KnotsAndExtractors();
    TestPipeBdrPatch14KroneckerExtraction();
    TestPipeBdrPatch14BezierWeights();
    TestPipeBdrPatch14BezierControlPoints();
}
