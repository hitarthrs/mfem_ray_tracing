#include "mfem_raytracing/tspline/tspline_bilinear_ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

void UpdateBounds(LeafPatch &leaf)
{
    for (int axis = 0; axis < 3; ++axis)
    {
        leaf.bbox.min[axis] = leaf.patch.control_points[0][axis];
        leaf.bbox.max[axis] = leaf.patch.control_points[0][axis];
    }
    for (int corner = 1; corner < kBilinearPatchCorners; ++corner)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            leaf.bbox.min[axis] = std::min(leaf.bbox.min[axis], leaf.patch.control_points[corner][axis]);
            leaf.bbox.max[axis] = std::max(leaf.bbox.max[axis], leaf.patch.control_points[corner][axis]);
        }
    }
}

void SetDomain(LeafPatch &leaf, int axis, double lo, double hi)
{
    double *domain = axis == 0 ? leaf.u_domain_global : leaf.v_domain_global;
    domain[0] = lo;
    domain[1] = hi;
}

} // namespace

HomogeneousBilinearNet HomogenizeBilinearLeaf(const LeafPatch &leaf)
{
    HomogeneousBilinearNet result{};
    for (int corner = 0; corner < kBilinearPatchCorners; ++corner)
    {
        const double weight = leaf.patch.rational ? leaf.patch.weights[corner] : 1.0;
        if (!std::isfinite(weight) || std::abs(weight) <= 1e-14)
        {
            throw std::invalid_argument("bilinear leaf has a non-finite or zero projective weight");
        }
        result[corner] = {weight * leaf.patch.control_points[corner][0],
                          weight * leaf.patch.control_points[corner][1],
                          weight * leaf.patch.control_points[corner][2], weight};
    }
    return result;
}

void SetHomogeneousBilinearNet(LeafPatch &leaf, const HomogeneousBilinearNet &net)
{
    bool rational = leaf.patch.rational;
    for (int corner = 0; corner < kBilinearPatchCorners; ++corner)
    {
        const HomogeneousPoint &point = net[corner];
        if (!std::isfinite(point[3]) || std::abs(point[3]) <= 1e-14)
        {
            throw std::invalid_argument("cannot project a non-finite or zero homogeneous control point");
        }
        leaf.patch.control_points[corner][0] = point[0] / point[3];
        leaf.patch.control_points[corner][1] = point[1] / point[3];
        leaf.patch.control_points[corner][2] = point[2] / point[3];
        leaf.patch.weights[corner] = point[3];
        rational = rational || std::abs(point[3] - 1.0) > 1e-14;
    }
    leaf.patch.rational = rational;
    UpdateBounds(leaf);
}

HomogeneousPoint InterpolateHomogeneous(const HomogeneousPoint &a,
                                        const HomogeneousPoint &b, double alpha)
{
    if (!std::isfinite(alpha)) { throw std::invalid_argument("homogeneous interpolation alpha is not finite"); }
    HomogeneousPoint result{};
    for (int i = 0; i < 4; ++i) { result[i] = (1.0 - alpha) * a[i] + alpha * b[i]; }
    return result;
}

HomogeneousPoint EvaluateHomogeneousBilinear(const HomogeneousBilinearNet &net,
                                             double u, double v)
{
    const HomogeneousPoint lo = InterpolateHomogeneous(net[0], net[2], u);
    const HomogeneousPoint hi = InterpolateHomogeneous(net[1], net[3], u);
    return InterpolateHomogeneous(lo, hi, v);
}

std::array<double, 3> ProjectHomogeneous(const HomogeneousPoint &point)
{
    if (!std::isfinite(point[3]) || std::abs(point[3]) <= 1e-14)
    {
        throw std::invalid_argument("cannot project a non-finite or zero homogeneous point");
    }
    return {point[0] / point[3], point[1] / point[3], point[2] / point[3]};
}

double HomogeneousCartesianDistance(const HomogeneousPoint &a, const HomogeneousPoint &b)
{
    const auto pa = ProjectHomogeneous(a);
    const auto pb = ProjectHomogeneous(b);
    const double dx = pa[0] - pb[0];
    const double dy = pa[1] - pb[1];
    const double dz = pa[2] - pb[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::array<LeafPatch, 2> SplitRationalBilinearLeaf(const LeafPatch &leaf, int axis,
                                                    double parameter, double tolerance)
{
    if (axis != 0 && axis != 1) { throw std::invalid_argument("bilinear split axis must be 0 or 1"); }
    if (tolerance < 0.0) { throw std::invalid_argument("bilinear split tolerance must be non-negative"); }
    const double lo = axis == 0 ? leaf.u_domain_global[0] : leaf.v_domain_global[0];
    const double hi = axis == 0 ? leaf.u_domain_global[1] : leaf.v_domain_global[1];
    if (!(lo + tolerance < parameter && parameter < hi - tolerance))
    {
        throw std::invalid_argument("bilinear split parameter is not strictly inside the leaf");
    }
    const double alpha = (parameter - lo) / (hi - lo);
    const HomogeneousBilinearNet source = HomogenizeBilinearLeaf(leaf);
    HomogeneousBilinearNet first{};
    HomogeneousBilinearNet second{};
    if (axis == 0)
    {
        const HomogeneousPoint mid0 = InterpolateHomogeneous(source[0], source[2], alpha);
        const HomogeneousPoint mid1 = InterpolateHomogeneous(source[1], source[3], alpha);
        first = {source[0], source[1], mid0, mid1};
        second = {mid0, mid1, source[2], source[3]};
    }
    else
    {
        const HomogeneousPoint mid0 = InterpolateHomogeneous(source[0], source[1], alpha);
        const HomogeneousPoint mid1 = InterpolateHomogeneous(source[2], source[3], alpha);
        first = {source[0], mid0, source[2], mid1};
        second = {mid0, source[1], mid1, source[3]};
    }
    std::array<LeafPatch, 2> result = {leaf, leaf};
    SetDomain(result[0], axis, lo, parameter);
    SetDomain(result[1], axis, parameter, hi);
    SetHomogeneousBilinearNet(result[0], first);
    SetHomogeneousBilinearNet(result[1], second);
    return result;
}

std::vector<LeafPatch> RefineRationalBilinearLeafToBreaks(
    const LeafPatch &leaf, int axis, const std::vector<double> &breaks, double tolerance)
{
    if (axis != 0 && axis != 1) { throw std::invalid_argument("bilinear refinement axis must be 0 or 1"); }
    std::vector<double> sorted = breaks;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end(), [tolerance](double a, double b) {
        return std::abs(a - b) <= tolerance;
    }), sorted.end());
    std::vector<LeafPatch> pieces = {leaf};
    for (const double knot : sorted)
    {
        std::vector<LeafPatch> next;
        for (const LeafPatch &piece : pieces)
        {
            const double lo = axis == 0 ? piece.u_domain_global[0] : piece.v_domain_global[0];
            const double hi = axis == 0 ? piece.u_domain_global[1] : piece.v_domain_global[1];
            if (lo + tolerance < knot && knot < hi - tolerance)
            {
                const auto split = SplitRationalBilinearLeaf(piece, axis, knot, tolerance);
                next.push_back(split[0]);
                next.push_back(split[1]);
            }
            else
            {
                next.push_back(piece);
            }
        }
        pieces = std::move(next);
    }
    return pieces;
}

} // namespace tspline
} // namespace mfem_raytracing
