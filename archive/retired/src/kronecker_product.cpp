#include "kronecker_product.hpp"

#include <stdexcept>

mfem::DenseMatrix KroneckerProduct(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B)
{
    const int m = A.Height();
    const int n = A.Width();
    const int p = B.Height();
    const int q = B.Width();

    if (m <= 0 || n <= 0 || p <= 0 || q <= 0)
    {
        throw std::invalid_argument("KroneckerProduct requires non-empty input matrices");
    }

    mfem::DenseMatrix K(m * p, n * q);

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            const double a_ij = A(i, j);
            for (int k = 0; k < p; ++k)
            {
                for (int l = 0; l < q; ++l)
                {
                    K(i * p + k, j * q + l) = a_ij * B(k, l);
                }
            }
        }
    }

    return K;
}
