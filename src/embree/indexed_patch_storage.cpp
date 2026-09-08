// Deduplicating storage for bilinear patches: shared position/weight tables plus
// per-primitive corner/weight indices. Tracing prefers View (header); Load expands.

#include "mfem_raytracing/embree/bilinear_patch_geometry.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace mfem_raytracing
{
namespace
{
template <std::size_t N>
struct BitsHash
{
    std::size_t operator()(const std::array<std::uint64_t, N> &key) const
    {
        std::size_t h = 0;
        for (const auto bits : key)
        {
            h ^= std::hash<std::uint64_t>{}(bits) + 0x9e3779b9u + (h << 6) + (h >> 2);
        }
        return h;
    }
};

template <std::size_t N>
using Index = std::unordered_map<std::array<std::uint64_t, N>, std::uint32_t, BitsHash<N>>;

// Bit-exact interning of N doubles (NaN/bit-pattern sensitive via uint64 keys).
template <std::size_t N>
std::uint32_t Intern(const double *values, Index<N> &index,
                     std::vector<std::array<double, N>> &table)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::array<std::uint64_t, N> key;
    std::memcpy(key.data(), values, sizeof(key));
    const auto found = index.find(key);
    if (found != index.end()) { return found->second; }
    if (table.size() >= std::numeric_limits<std::uint32_t>::max())
    {
        throw std::length_error("indexed bilinear storage exceeds 32-bit index capacity");
    }
    const auto id = static_cast<std::uint32_t>(table.size());
    std::array<double, N> entry;
    std::memcpy(entry.data(), values, sizeof(entry));
    table.push_back(entry);
    index.emplace(key, id);
    return id;
}
} // namespace

IndexedBilinearPatchStorage::IndexedBilinearPatchStorage(
    const std::vector<BilinearPatchPrimitive> &patches)
{
    // Construction tables are temporary: no hash-table overhead during tracing.
    Index<3> positions;
    Index<4> weights;
    patches_.reserve(patches.size());
    for (const auto &source : patches)
    {
        Patch patch{};
        for (int c = 0; c < 4; ++c)
        {
            patch.corners[c] = Intern(source.control_points[c], positions, positions_);
        }
        // Preserve even unused polynomial weights for exact API round-trips.
        patch.weights = Intern(source.weights, weights, weights_);
        patch.rational = source.rational;
        patches_.push_back(patch);
    }
    positions_.shrink_to_fit();
    weights_.shrink_to_fit();
}

// Expand indexed corners/weights into a dense primitive (View avoids this copy).
BilinearPatchPrimitive IndexedBilinearPatchStorage::Load(std::size_t index) const
{
    const Patch &encoded = patches_[index];
    BilinearPatchPrimitive patch;
    for (int c = 0; c < 4; ++c)
    {
        std::memcpy(patch.control_points[c], positions_[encoded.corners[c]].data(),
                    sizeof(patch.control_points[c]));
    }
    std::memcpy(patch.weights, weights_[encoded.weights].data(), sizeof(patch.weights));
    patch.rational = encoded.rational;
    return patch;
}

std::size_t IndexedBilinearPatchStorage::BufferBytes() const
{
    return positions_.capacity() * sizeof(positions_[0]) +
           weights_.capacity() * sizeof(weights_[0]) + patches_.capacity() * sizeof(Patch);
}
} // namespace mfem_raytracing
