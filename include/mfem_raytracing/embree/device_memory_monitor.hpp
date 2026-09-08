#ifndef MFEM_RAYTRACING_EMBREE_DEVICE_MEMORY_MONITOR_HPP
#define MFEM_RAYTRACING_EMBREE_DEVICE_MEMORY_MONITOR_HPP

#include "mfem_raytracing/embree/embree_interface.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <sys/types.h>

namespace mfem_raytracing
{

/// Tracks the bytes Embree allocates internally for a device: BVH nodes, leaf
/// blocks, and any buffer created through rtcSetNewGeometryBuffer. Buffers the
/// caller owns (shared buffers, user-geometry primitive arrays) are *not*
/// counted, so `live_bytes()` after a commit is the acceleration structure
/// proper.
///
/// Note that these are bytes, not nodes: Embree's internal allocator hands out
/// large slabs and sub-allocates nodes from them, so `alloc_events()` counts
/// slabs and is far smaller than the node count. Use BvhTreeStats for the
/// actual tree shape.
///
/// Embree allows one monitor per device; attach before building the scene.
class DeviceMemoryMonitor
{
public:
    /// Number of power-of-two buckets in the allocation-size histogram.
    static constexpr std::size_t kHistogramBuckets = 40;

    /// Install this monitor on `device`. Safe to call once per device.
    void AttachTo(RTCDevice device);

    /// Bytes currently held by the device (i.e. the committed BVH once the
    /// build scratch space has been released).
    std::int64_t live_bytes() const { return live_bytes_.load(); }

    /// High-water mark, which includes transient build allocations.
    std::int64_t peak_bytes() const { return peak_bytes_.load(); }

    /// Number of allocation callbacks seen; these are allocator slabs, not
    /// individual BVH nodes.
    std::int64_t alloc_events() const { return alloc_events_.load(); }
    std::int64_t free_events() const { return free_events_.load(); }
    std::int64_t largest_alloc() const { return largest_alloc_.load(); }

    /// Count of allocations whose size fell in [2^i, 2^(i+1)).
    std::int64_t histogram_bucket(std::size_t i) const { return histogram_[i].load(); }

    void Reset();

private:
    static bool MonitorFunc(void *user_ptr, ssize_t bytes, bool post);

    std::atomic<std::int64_t> live_bytes_{0};
    std::atomic<std::int64_t> peak_bytes_{0};
    std::atomic<std::int64_t> alloc_events_{0};
    std::atomic<std::int64_t> free_events_{0};
    std::atomic<std::int64_t> largest_alloc_{0};
    std::array<std::atomic<std::int64_t>, kHistogramBuckets> histogram_{};
};

} // namespace mfem_raytracing

#endif
