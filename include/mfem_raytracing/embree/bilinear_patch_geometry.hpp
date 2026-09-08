#ifndef MFEM_RAYTRACING_BILINEAR_PATCH_GEOMETRY_HPP
#define MFEM_RAYTRACING_BILINEAR_PATCH_GEOMETRY_HPP

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>

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

/// Non-owning corner references for tracing without expanding indexed patches.
/// Valid only while the backing geometry storage remains alive and unchanged.
struct BilinearPatchView
{
    const double *control_points[4];
    const double *weights;
    bool rational;
};

/// Axis-aligned bounds used for Embree and BVH export.
struct AxisAlignedBounds
{
    double min[3] = {};
    double max[3] = {};
};

/// Lossless runtime encoding. Positions and complete weight tuples are shared
/// only when their double bit patterns match (no welding or quantization).
class IndexedBilinearPatchStorage
{
public:
    explicit IndexedBilinearPatchStorage(const std::vector<BilinearPatchPrimitive> &patches);
    BilinearPatchPrimitive Load(std::size_t index) const;
    BilinearPatchView View(std::size_t index) const
    {
        const auto &patch = patches_[index];
        return {{positions_[patch.corners[0]].data(), positions_[patch.corners[1]].data(),
                 positions_[patch.corners[2]].data(), positions_[patch.corners[3]].data()},
                weights_[patch.weights].data(), patch.rational};
    }
    std::size_t BufferBytes() const;
    std::size_t PatchCount() const { return patches_.size(); }

private:
    struct Patch
    {
        std::uint32_t corners[4];
        std::uint32_t weights;
        bool rational;
    };
    std::vector<std::array<double, 3>> positions_;
    std::vector<std::array<double, 4>> weights_;
    std::vector<Patch> patches_;
};

/// User payload attached to ``RTCGeometry`` via ``rtcSetGeometryUserData``.
struct BilinearPatchGeometryData
{
    /// Contiguous patch buffer indexed by Embree's ``primID``.
    const BilinearPatchPrimitive *prim_ref_buffer = nullptr;
    std::size_t primitive_count = 0;
    double box_bump = 0.0;
    /// Alternative to prim_ref_buffer; immutable for the geometry lifetime.
    const IndexedBilinearPatchStorage *indexed = nullptr;

    BilinearPatchView View(std::size_t index) const
    {
        if (indexed != nullptr) { return indexed->View(index); }
        const auto &patch = prim_ref_buffer[index];
        return {{patch.control_points[0], patch.control_points[1],
                 patch.control_points[2], patch.control_points[3]}, patch.weights, patch.rational};
    }

    const BilinearPatchPrimitive &Load(std::size_t index,
                                        BilinearPatchPrimitive &scratch) const
    {
        if (indexed != nullptr)
        {
            scratch = indexed->Load(index);
            return scratch;
        }
        return prim_ref_buffer[index];
    }
};

} // namespace mfem_raytracing

#endif
