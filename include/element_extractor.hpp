#ifndef ELEMENT_EXTRACTOR_HPP
#define ELEMENT_EXTRACTOR_HPP

#include "mfem.hpp"

#include <vector>

/**
 * @brief Local Bézier extraction operator for one B-spline knot span.
 *
 * Stores a (p+1) x (p+1) matrix C_e mapping Bernstein basis functions to
 * B-spline basis functions on element e: N = C_e B.
 */
struct ElementExtractor
{
    mfem::DenseMatrix matrix;
};

/**
 * @brief Computes local Bézier extraction operators for a 1D B-spline knot vector.
 *
 * Implements Algorithm 1 (knot insertion / extraction) for an open knot vector.
 * Each returned operator corresponds to one pass of the algorithm and has size
 * (p+1) x (p+1), where p is the spline degree.
 *
 * @param knot_vector Open or clamped knot vector U = {u_0, ..., u_{m-1}}.
 * @param degree Spline degree p (must be >= 1).
 * @return Vector of extraction operators C_e, e = 0, ..., nb-1.
 * @throws std::invalid_argument if the knot vector or degree is invalid.
 */
std::vector<ElementExtractor> ElementExtractors(const std::vector<double> &knot_vector,
                                                int degree);

#endif
