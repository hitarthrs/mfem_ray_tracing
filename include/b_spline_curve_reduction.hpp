#ifndef B_SPLINE_CURVE_REDUCTION_HPP
#define B_SPLINE_CURVE_REDUCTION_HPP

#include <cstddef>
#include <vector>

/**
 * @brief B-spline curve degree reduction (Piegl & Tiller A5.11).
 *
 * Reduces a polynomial B-spline curve of degree @a degree (p) to degree p-1.
 *
 * @param n_control_points  Number of input control points n.
 * @param degree            Input degree p (>= 2).
 * @param U                 Knot vector, length n + p + 1.
 * @param Qw                Input controls, n rows of dim components.
 * @param dim               Spatial dimension of each control point.
 * @param Pw                Output reduced controls (active length returned).
 * @param Uh                Output reduced knot vector (active length returned).
 * @param error_array       Per-knot error accumulator, length m+1 where m = U.size()-1.
 * @param tol               Tolerance for early exit (use infinity to disable).
 * @return true on success, false if tolerance exceeded.
 */
bool DegreeReduceCurve(int n_control_points,
                       int degree,
                       const std::vector<double> &U,
                       const std::vector<std::vector<double>> &Qw,
                       int dim,
                       std::vector<std::vector<double>> &Pw,
                       std::vector<double> &Uh,
                       std::vector<double> &error_array,
                       double tol = 1e300);

#endif
