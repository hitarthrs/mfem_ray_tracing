#ifndef BEZIER_DEGREE_REDUCTION_HPP
#define BEZIER_DEGREE_REDUCTION_HPP

#include <cstddef>
#include <vector>

/**
 * @brief Polynomial Bézier degree reduction (Piegl & Tiller, w_i = 1).
 *
 * Input @a bpts has length p+1 (degree p >= 2). Output @a reduced_bpts has
 * length p (degree p-1). Always fills @a reduced_bpts and sets @a max_err to
 * the bound from eq. (5.43) (p even) or (5.44) (p odd), maximized over u in [0,1].
 *
 * Each control point is a vector of length @a dim (same dim for all points).
 */
void BezierDegreeReduce(const std::vector<std::vector<double>> &bpts,
                        int dim,
                        std::vector<std::vector<double>> &reduced_bpts,
                        double &max_err);

#endif
