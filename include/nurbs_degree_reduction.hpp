#ifndef NURBS_DEGREE_REDUCTION_HPP
#define NURBS_DEGREE_REDUCTION_HPP

/**
 * @file nurbs_degree_reduction.hpp
 * @brief One-step rational NURBS curve degree reduction (Piegl & Tiller A5.11).
 *
 * Python reference: python_experiments/nurbs_degree_reduction.py
 *
 * Algorithm (homogeneous A5.11):
 *   1. Read patch B-net rows [w·P, w] from mfem::NURBSPatch (encodes control (P, w)).
 *   2. Run polynomial DegreeReduceCurve on those controls (dim = spatial_dim + 1).
 *   3. Project the reduced homogeneous controls back to Cartesian (P, W).
 *
 * Input is always a 1D mfem::NURBSPatch. Depends on b_spline_curve_reduction.hpp (A5.11).
 */

#include "mfem.hpp"

#include <vector>

/**
 * @brief One-step NURBS degree reduction p → p-1 on a 1D MFEM patch.
 *
 * @param patch         1D NURBS curve (GetNKV() == 1) with homogeneous controls.
 * @param pw_out        Reduced Cartesian controls (active length returned).
 * @param weights_out   Reduced weights (active length returned).
 * @param uh            Reduced knot vector (active length returned).
 * @param error_array   Per-knot error accumulator; length m + 1 with m = |U| - 1.
 * @param tol           Early-exit tolerance (use a large value to disable).
 * @return true on success, false if tolerance is exceeded.
 *
 * Knot vector and degree are read from patch.GetKV(0).
 */
bool DegreeReduceNURBCurve(mfem::NURBSPatch &patch,
                           std::vector<std::vector<double>> &pw_out,
                           std::vector<double> &weights_out,
                           std::vector<double> &uh,
                           std::vector<double> &error_array,
                           double tol = 1e300);

#endif
