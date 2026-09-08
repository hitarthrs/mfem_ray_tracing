// Embree device memory monitor: live/peak bytes and slab-size histogram for
// allocations Embree makes internally (BVH nodes, leaf blocks, rtcSetNew*
// geometry buffers). Caller-owned shared buffers and user-geometry arrays are
// not reported here — account those separately from app-side sizes.

#include "mfem_raytracing/embree/device_memory_monitor.hpp"

namespace mfem_raytracing
{

bool DeviceMemoryMonitor::MonitorFunc(void *user_ptr, ssize_t bytes, bool post)
{
    // Embree calls with post == true after an allocation and post == false
    // before a free; counting one side of each pair keeps the running total
    // consistent without double counting.
    auto *self = static_cast<DeviceMemoryMonitor *>(user_ptr);
    const std::int64_t delta = static_cast<std::int64_t>(bytes);
    const std::int64_t live = self->live_bytes_.fetch_add(delta) + delta;

    if (delta > 0)
    {
        self->alloc_events_.fetch_add(1);

        std::int64_t largest = self->largest_alloc_.load();
        while (delta > largest &&
               !self->largest_alloc_.compare_exchange_weak(largest, delta))
        {
        }

        // Power-of-two size bucket: [2^i, 2^(i+1)).
        std::size_t bucket = 0;
        for (std::int64_t v = delta; v > 1 && bucket + 1 < kHistogramBuckets; v >>= 1)
        {
            ++bucket;
        }
        self->histogram_[bucket].fetch_add(1);
    }
    else if (delta < 0)
    {
        self->free_events_.fetch_add(1);
    }

    // Peak includes transient build scratch still live at this callback.
    std::int64_t peak = self->peak_bytes_.load();
    while (live > peak && !self->peak_bytes_.compare_exchange_weak(peak, live))
    {
    }
    (void)post;
    return true; // never refuse an allocation
}

void DeviceMemoryMonitor::Reset()
{
    live_bytes_.store(0);
    peak_bytes_.store(0);
    alloc_events_.store(0);
    free_events_.store(0);
    largest_alloc_.store(0);
    for (auto &bucket : histogram_)
    {
        bucket.store(0);
    }
}

void DeviceMemoryMonitor::AttachTo(RTCDevice device)
{
    // One monitor per device; attach before scene commit / BVH build.
    rtcSetDeviceMemoryMonitorFunction(device, &DeviceMemoryMonitor::MonitorFunc, this);
}

} // namespace mfem_raytracing
