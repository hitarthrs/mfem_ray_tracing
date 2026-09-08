#ifndef MFEM_RAYTRACING_EMBREE_BVH_TREE_STATS_HPP
#define MFEM_RAYTRACING_EMBREE_BVH_TREE_STATS_HPP

#include "mfem_raytracing/embree/embree_interface.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mfem_raytracing
{

/// Exact shape of a BVH over a given set of primitive boxes.
struct BvhTreeStats
{
    std::size_t primitives = 0;
    std::size_t inner_nodes = 0;
    std::size_t leaves = 0;
    /// Primitive references stored in leaves. Exceeds `primitives` when the
    /// high-quality builder splits a primitive across leaves.
    std::size_t leaf_primitive_refs = 0;
    std::size_t max_depth = 0;
    double mean_leaf_occupancy = 0.0;
};

/// Build a BVH over `boxes` with Embree's own SAH builder (rtcBuildBVH) and
/// count what it produces. This is the same builder the scene commit uses, run
/// with the branching factor and leaf size of the primitive type in question,
/// so the node and leaf counts are exact rather than inferred from bytes.
///
/// `branching_factor` should match the BVH width of the Embree build (4 for an
/// SSE2/NEON build, 8 with AVX). The leaf-size and SAH-block settings must
/// match the primitive type, or the proxy tree will not resemble the one the
/// scene commit builds:
///   Triangle4v      min = max = 4, sah_block_size = 4
///   Object (user)   min = max = 1, sah_block_size = 1
BvhTreeStats ComputeBvhTreeStats(RTCDevice device,
                                 const std::vector<RTCBuildPrimitive> &boxes,
                                 unsigned int branching_factor,
                                 unsigned int min_leaf_size,
                                 unsigned int max_leaf_size,
                                 unsigned int sah_block_size,
                                 RTCBuildQuality quality = RTC_BUILD_QUALITY_HIGH);

} // namespace mfem_raytracing

#endif
