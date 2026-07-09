#ifndef MFEM_RAYTRACING_BILINEAR_PATCH_GEOMETRY_HPP
#define MFEM_RAYTRACING_BILINEAR_PATCH_GEOMETRY_HPP

#include <cstddef>

namespace mfem_raytracing
{

/// Corner ordering for a clamped bilinear patch on ``[0, 1]^2``.
enum class BilinearCorner : int
{
    P00 = 0, ///< ``(u, v) = (0, 0)``
    P01 = 1, ///< ``(u, v) = (0, 1)``
    P10 = 2, ///< ``(u, v) = (1, 0)``
    P11 = 3, ///< ``(u, v) = (1, 1)``
};

constexpr int kBilinearPatchCorners = 4;

/// One degree-(1, 1) tensor-product patch (Cartesian control net + optional weights).
struct BilinearPatchPrimitive
{
    /// Cartesian control points ``P[i][0..2]``, indexed by :class:`BilinearCorner`.
    double control_points[kBilinearPatchCorners][3] = {};
    /// Rational weights; ignored when ``rational == false``.
    double weights[kBilinearPatchCorners] = {};
    bool rational = false;
};

/// Axis-aligned bounds used for Embree and BVH export.
struct AxisAlignedBounds
{
    double min[3] = {};
    double max[3] = {};
};

/// User payload attached to ``RTCGeometry`` via ``rtcSetGeometryUserData``.
struct BilinearPatchGeometryData
{
    /// Contiguous patch buffer indexed by Embree's ``primID``.
    const BilinearPatchPrimitive *prim_ref_buffer = nullptr;
    std::size_t primitive_count = 0;
    double box_bump = 0.0;
};

} // namespace mfem_raytracing

#endif
