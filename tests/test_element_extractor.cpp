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

void TestSingleBezierElement()
{
    const std::vector<double> knots = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
    const auto ops = ElementExtractors(knots, 3);
    CHECK(static_cast<int>(ops.size()) == 2);
    CHECK(ops[0].matrix.Height() == 4);
    CHECK(ops[0].matrix.Width() == 4);
    CHECK(IsIdentity(ops[0].matrix));
}

void TestTwoElementKnotVector()
{
    const std::vector<double> knots = {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0};
    const auto ops = ElementExtractors(knots, 3);
    CHECK(static_cast<int>(ops.size()) == 3);
    CHECK(ops[0].matrix.Height() == 4);
    CHECK(!IsIdentity(ops[0].matrix));
    CHECK(ops[1].matrix.Height() == 4);
}

void TestLinearTwoElements()
{
    const std::vector<double> knots = {0.0, 0.0, 0.5, 1.0, 1.0};
    const auto ops = ElementExtractors(knots, 1);
    CHECK(static_cast<int>(ops.size()) == 3);
    CHECK(ops[0].matrix.Height() == 2);
    CHECK(ops[1].matrix.Height() == 2);
    CHECK(IsIdentity(ops[0].matrix) || !IsIdentity(ops[1].matrix));
}

}  // namespace

void TestElementExtractor()
{
    TestSingleBezierElement();
    TestTwoElementKnotVector();
    TestLinearTwoElements();
    TestQuadraticThreeElementKnotVector();
}
