#ifndef MFEM_RAYTRACING_TSPLINE_BILINEAR_OPS_HPP
#define MFEM_RAYTRACING_TSPLINE_BILINEAR_OPS_HPP

#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "mfem_raytracing/tspline/tspline_seam_types.hpp"

#include <array>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

/// Rational-safe view of a bilinear control net in Embree corner order
/// (P00, P01, P10, P11).
using HomogeneousBilinearNet = std::array<HomogeneousPoint, 4>;

HomogeneousBilinearNet HomogenizeBilinearLeaf(const LeafPatch &leaf);
void SetHomogeneousBilinearNet(LeafPatch &leaf, const HomogeneousBilinearNet &net);
HomogeneousPoint InterpolateHomogeneous(const HomogeneousPoint &a,
                                        const HomogeneousPoint &b, double alpha);
HomogeneousPoint EvaluateHomogeneousBilinear(const HomogeneousBilinearNet &net,
                                             double u, double v);
std::array<double, 3> ProjectHomogeneous(const HomogeneousPoint &point);
double HomogeneousCartesianDistance(const HomogeneousPoint &a,
                                    const HomogeneousPoint &b);

/// Exact degree-1 rational knot insertion in one global leaf parameter.
/// `axis` is 0 for u or 1 for v; `parameter` must lie strictly inside the leaf.
std::array<LeafPatch, 2> SplitRationalBilinearLeaf(const LeafPatch &leaf, int axis,
                                                    double parameter,
                                                    double tolerance = 1e-12);

/// Split a leaf at every requested global knot strictly inside it.  The result
/// preserves the represented rational bilinear surface exactly.
std::vector<LeafPatch> RefineRationalBilinearLeafToBreaks(
    const LeafPatch &leaf, int axis, const std::vector<double> &breaks,
    double tolerance = 1e-12);

} // namespace tspline
} // namespace mfem_raytracing

#endif
