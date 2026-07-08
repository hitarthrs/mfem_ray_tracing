/**
 * @file nurbs_degree_reduction.cpp
 * @brief Homogeneous A5.11 wrapper for rational NURBS curves in mfem::NURBSPatch.
 */

#include "nurbs_degree_reduction.hpp"

#include "b_spline_curve_reduction.hpp"

#include <cmath>
#include <stdexcept>

namespace
{

constexpr double kZeroWeightTol = 1e-15;

/** Copy MFEM knot values (with multiplicities) for DegreeReduceCurve. */
std::vector<double> KnotsFromKV(const mfem::KnotVector &kv)
{
    std::vector<double> knots(kv.Size());
    for (int i = 0; i < kv.Size(); ++i)
    {
        knots[i] = kv[i];
    }
    return knots;
}

/**
 * Read B-net rows from a 1D NURBSPatch for homogeneous A5.11.
 *
 * A rational control point is (P, w). On a NURBS mesh those live in separate
 * node/weight arrays, but mfem::NURBSPatch packs them as [w·P, w] in patch(i, :).
 * Copy that layout directly — no lift step needed before DegreeReduceCurve.
 */
std::vector<std::vector<double>> ReadHomogeneousFromPatch(const mfem::NURBSPatch &patch,
                                                          int n,
                                                          int homog_dim)
{
    std::vector<std::vector<double>> qw_hom(static_cast<std::size_t>(n),
                                           std::vector<double>(static_cast<std::size_t>(homog_dim)));
    for (int i = 0; i < n; ++i)
    {
        for (int d = 0; d < homog_dim; ++d)
        {
            qw_hom[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] = patch(i, d);
        }
    }
    return qw_hom;
}

/**
 * Project homogeneous controls [..., w] back to Cartesian (P, W).
 *
 * Row layout: spatial components occupy indices 0 .. spatial_dim-1; the NURBS
 * weight is the last entry. Mirrors Python project_from_homogeneous().
 */
void ProjectFromHomogeneous(const std::vector<std::vector<double>> &homogeneous,
                            std::vector<std::vector<double>> &control_points,
                            std::vector<double> &weights)
{
    if (homogeneous.empty() || homogeneous.front().size() < 2)
    {
        throw std::invalid_argument(
            "DegreeReduceNURBCurve: homogeneous controls must be (n, d+1)");
    }

    const std::size_t n = homogeneous.size();
    const std::size_t spatial_dim = homogeneous.front().size() - 1;

    control_points.assign(n, std::vector<double>(spatial_dim));
    weights.resize(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        if (homogeneous[i].size() != spatial_dim + 1)
        {
            throw std::invalid_argument(
                "DegreeReduceNURBCurve: inconsistent homogeneous row dimension");
        }

        const double w = homogeneous[i][spatial_dim];
        if (std::abs(w) < kZeroWeightTol)
        {
            throw std::invalid_argument(
                "DegreeReduceNURBCurve: zero homogeneous weight encountered");
        }

        weights[i] = w;
        // P_i = (w·P)_i / w
        for (std::size_t d = 0; d < spatial_dim; ++d)
        {
            control_points[i][d] = homogeneous[i][d] / w;
        }
    }
}

}  // namespace

bool DegreeReduceNURBCurve(int n_control_points,
                           int degree,
                           const std::vector<double> &U,
                           const std::vector<std::vector<double>> &control_points,
                           const std::vector<double> &weights,
                           std::vector<std::vector<double>> &pw_out,
                           std::vector<double> &weights_out,
                           std::vector<double> &uh,
                           std::vector<double> &error_array,
                           double tol)
{
    if (degree < 2)
    {
        throw std::invalid_argument("DegreeReduceNURBCurve: expected degree >= 2");
    }
    if (static_cast<int>(control_points.size()) != n_control_points)
    {
        throw std::invalid_argument("DegreeReduceNURBCurve: control point count mismatch");
    }
    if (weights.size() != control_points.size())
    {
        throw std::invalid_argument("DegreeReduceNURBCurve: weights/control point count mismatch");
    }
    if (control_points.empty())
    {
        throw std::invalid_argument("DegreeReduceNURBCurve: control point list is empty");
    }

    const int homog_dim = static_cast<int>(control_points.front().size()) + 1;
    if (homog_dim < 2)
    {
        throw std::invalid_argument("DegreeReduceNURBCurve: spatial dimension must be >= 1");
    }

    std::vector<std::vector<double>> qw_hom(
        static_cast<std::size_t>(n_control_points),
        std::vector<double>(static_cast<std::size_t>(homog_dim), 0.0));
    for (int i = 0; i < n_control_points; ++i)
    {
        if (control_points[static_cast<std::size_t>(i)].size() != static_cast<std::size_t>(homog_dim - 1))
        {
            throw std::invalid_argument("DegreeReduceNURBCurve: inconsistent control point dimension");
        }
        const double w = weights[static_cast<std::size_t>(i)];
        for (int d = 0; d < homog_dim - 1; ++d)
        {
            qw_hom[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] =
                control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] * w;
        }
        qw_hom[static_cast<std::size_t>(i)][static_cast<std::size_t>(homog_dim - 1)] = w;
    }

    std::vector<std::vector<double>> pw_hom;
    if (!DegreeReduceCurve(n_control_points,
                           degree,
                           U,
                           qw_hom,
                           homog_dim,
                           pw_hom,
                           uh,
                           error_array,
                           tol))
    {
        return false;
    }

    ProjectFromHomogeneous(pw_hom, pw_out, weights_out);
    return true;
}

bool DegreeReduceNURBCurve(mfem::NURBSPatch &patch,
                           std::vector<std::vector<double>> &pw_out,
                           std::vector<double> &weights_out,
                           std::vector<double> &uh,
                           std::vector<double> &error_array,
                           double tol)
{
    // Curve-only: one knot vector, not a tensor-product surface/volume patch.
    if (patch.GetNKV() != 1)
    {
        throw std::invalid_argument("DegreeReduceNURBCurve: expected a 1D NURBSPatch");
    }

    mfem::KnotVector &kv = *patch.GetKV(0);
    const int n = kv.GetNCP();
    const int degree = kv.GetOrder();       // MFEM GetOrder() == spline degree p
    const int homog_dim = patch.GetNC();    // number of components per CP (includes w)
    if (homog_dim < 2)
    {
        throw std::invalid_argument(
            "DegreeReduceNURBCurve: patch homogeneous dimension must be >= 2");
    }

    // --- Input: homogeneous controls and knot vector from the patch ------------
    const std::vector<double> U = KnotsFromKV(kv);
    const std::vector<std::vector<double>> qw_hom =
        ReadHomogeneousFromPatch(patch, n, homog_dim);

    std::vector<std::vector<double>> control_points(static_cast<std::size_t>(n),
                                                    std::vector<double>(static_cast<std::size_t>(homog_dim - 1), 0.0));
    std::vector<double> weights(static_cast<std::size_t>(n), 1.0);
    for (int i = 0; i < n; ++i)
    {
        const double w = qw_hom[static_cast<std::size_t>(i)][static_cast<std::size_t>(homog_dim - 1)];
        weights[static_cast<std::size_t>(i)] = w;
        for (int d = 0; d < homog_dim - 1; ++d)
        {
            control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] =
                qw_hom[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] / w;
        }
    }

    return DegreeReduceNURBCurve(
        n, degree, U, control_points, weights, pw_out, weights_out, uh, error_array, tol);
}
