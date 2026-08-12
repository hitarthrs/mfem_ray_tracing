#ifndef KRONECKER_PRODUCT_HPP
#define KRONECKER_PRODUCT_HPP

#include "mfem.hpp"

/**
 * @brief Kronecker product K = A ⊗ B for dense MFEM matrices.
 *
 * If A is m×n and B is p×q, then K is (m·p)×(n·q) with
 *   K(i·p + k, j·q + l) = A(i, j) * B(k, l).
 */
mfem::DenseMatrix KroneckerProduct(const mfem::DenseMatrix &A, const mfem::DenseMatrix &B);

#endif
