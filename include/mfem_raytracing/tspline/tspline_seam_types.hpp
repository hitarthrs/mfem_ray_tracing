#ifndef MFEM_RAYTRACING_TSPLINE_SEAM_TYPES_HPP
#define MFEM_RAYTRACING_TSPLINE_SEAM_TYPES_HPP

#include <array>
#include <cstddef>
#include <utility>

namespace mfem_raytracing
{
namespace tspline
{

/// A named boundary of a tensor-product patch or bilinear leaf assembly.
enum class BoundarySide
{
    UMin,
    UMax,
    VMin,
    VMax,
};

/// Projective control point used for rational-safe seam operations.
using HomogeneousPoint = std::array<double, 4>;

const char *BoundarySideName(BoundarySide side);
int BoundaryNormalAxis(BoundarySide side);
int BoundaryTangentAxis(BoundarySide side);
bool BoundaryIsMaximum(BoundarySide side);

/// Corner indices in ascending native tangent parameter.
std::array<int, 2> BoundaryCornerIndices(BoundarySide side);

} // namespace tspline
} // namespace mfem_raytracing

#endif
