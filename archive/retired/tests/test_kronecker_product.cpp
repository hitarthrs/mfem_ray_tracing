#include "kronecker_product.hpp"
#include "test_helpers.hpp"

namespace
{

void CheckEntry(const mfem::DenseMatrix &mat, int row, int col, double expected, double tol = 1e-12)
{
    CHECK(row >= 0 && row < mat.Height());
    CHECK(col >= 0 && col < mat.Width());
    CHECK_NEAR(mat(row, col), expected, tol);
}

void FillC_eta_1(mfem::DenseMatrix &C)
{
    C.SetSize(3, 3);
    C(0, 0) = 1.0;
    C(0, 1) = 0.0;
    C(0, 2) = 0.0;
    C(1, 0) = 0.0;
    C(1, 1) = 1.0;
    C(1, 2) = 0.5;
    C(2, 0) = 0.0;
    C(2, 1) = 0.0;
    C(2, 2) = 0.5;
}

void FillC_xi_2(mfem::DenseMatrix &C)
{
    C.SetSize(3, 3);
    C(0, 0) = 0.5;
    C(0, 1) = 0.0;
    C(0, 2) = 0.0;
    C(1, 0) = 0.5;
    C(1, 1) = 1.0;
    C(1, 2) = 0.5;
    C(2, 0) = 0.0;
    C(2, 1) = 0.0;
    C(2, 2) = 0.5;
}

void CheckMatrix9x9(const mfem::DenseMatrix &mat, const double expected[9][9], double tol = 1e-12)
{
    CHECK(mat.Height() == 9);
    CHECK(mat.Width() == 9);
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 9; ++j)
        {
            CheckEntry(mat, i, j, expected[i][j], tol);
        }
    }
}

// 2×2 ⊗ 2×2 with a known block layout.
void TestKroneckerTwoByTwo()
{
    mfem::DenseMatrix A(2, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;

    mfem::DenseMatrix B(2, 2);
    B(0, 0) = 0.0;
    B(0, 1) = 5.0;
    B(1, 0) = 6.0;
    B(1, 1) = 7.0;

    const mfem::DenseMatrix K = KroneckerProduct(A, B);
    CHECK(K.Height() == 4);
    CHECK(K.Width() == 4);

    CheckEntry(K, 0, 0, 0.0);
    CheckEntry(K, 0, 1, 5.0);
    CheckEntry(K, 0, 2, 0.0);
    CheckEntry(K, 0, 3, 10.0);
    CheckEntry(K, 1, 0, 6.0);
    CheckEntry(K, 1, 1, 7.0);
    CheckEntry(K, 1, 2, 12.0);
    CheckEntry(K, 1, 3, 14.0);
    CheckEntry(K, 2, 0, 0.0);
    CheckEntry(K, 2, 1, 15.0);
    CheckEntry(K, 2, 2, 0.0);
    CheckEntry(K, 2, 3, 20.0);
    CheckEntry(K, 3, 0, 18.0);
    CheckEntry(K, 3, 1, 21.0);
    CheckEntry(K, 3, 2, 24.0);
    CheckEntry(K, 3, 3, 28.0);
}

// 1×1 ⊗ 2×2 equals scalar scaling of B.
void TestKroneckerScalarTimesB()
{
    mfem::DenseMatrix A(1, 1);
    A(0, 0) = 3.0;

    mfem::DenseMatrix B(2, 2);
    B(0, 0) = 1.0;
    B(0, 1) = 2.0;
    B(1, 0) = 3.0;
    B(1, 1) = 4.0;

    const mfem::DenseMatrix K = KroneckerProduct(A, B);
    CHECK(K.Height() == 2);
    CHECK(K.Width() == 2);
    CheckEntry(K, 0, 0, 3.0);
    CheckEntry(K, 0, 1, 6.0);
    CheckEntry(K, 1, 0, 9.0);
    CheckEntry(K, 1, 1, 12.0);
}

// I₂ ⊗ A is block-diagonal with two copies of A.
void TestKroneckerIdentityBlocks()
{
    mfem::DenseMatrix I2(2, 2);
    I2(0, 0) = 1.0;
    I2(1, 1) = 1.0;

    mfem::DenseMatrix A(2, 3);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;

    const mfem::DenseMatrix K = KroneckerProduct(I2, A);
    CHECK(K.Height() == 4);
    CHECK(K.Width() == 6);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            CheckEntry(K, i, j, A(i, j));
            CheckEntry(K, 2 + i, 3 + j, A(i, j));
        }
    }
    CheckEntry(K, 0, 3, 0.0);
    CheckEntry(K, 2, 0, 0.0);
}

// C^1 = C_eta^1 ⊗ C_xi^1 (identical 3×3 extraction operators).
void TestBezierExtractionKroneckerC1()
{
    mfem::DenseMatrix C_eta(3, 3);
    mfem::DenseMatrix C_xi(3, 3);
    FillC_eta_1(C_eta);
    FillC_eta_1(C_xi);  // C_xi^1 same as C_eta^1

    const mfem::DenseMatrix C1 = KroneckerProduct(C_eta, C_xi);

    static const double expected[9][9] = {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.25},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.25},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25},
    };
    CheckMatrix9x9(C1, expected);
}

// C^2 = C_eta^1 ⊗ C_xi^2.
void TestBezierExtractionKroneckerC2()
{
    mfem::DenseMatrix C_eta(3, 3);
    mfem::DenseMatrix C_xi(3, 3);
    FillC_eta_1(C_eta);
    FillC_xi_2(C_xi);

    const mfem::DenseMatrix C2 = KroneckerProduct(C_eta, C_xi);

    static const double expected[9][9] = {
        {0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.25},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25},
    };
    CheckMatrix9x9(C2, expected);
}

}  // namespace

void TestKroneckerProduct()
{
    TestKroneckerTwoByTwo();
    TestKroneckerScalarTimesB();
    TestKroneckerIdentityBlocks();
    TestBezierExtractionKroneckerC1();
    TestBezierExtractionKroneckerC2();
}
