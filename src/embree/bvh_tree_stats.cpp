// Proxy BVH build over primitive AABBs via rtcBuildBVH: callbacks record a
// CountNode tree, then Walk tallies exact inner/leaf/depth/occupancy stats.
// Used when device-memory bytes alone cannot describe tree shape.

#include "mfem_raytracing/embree/bvh_tree_stats.hpp"

#include "embree4/rtcore_builder.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mfem_raytracing
{
namespace
{

constexpr unsigned int kMaxChildren = 8;

/// Minimal node the counting callbacks build, so the finished tree can be
/// walked for depth and occupancy after rtcBuildBVH returns its root.
struct CountNode
{
    bool leaf = false;
    unsigned int count = 0; // children for an inner node, prims for a leaf
    CountNode *children[kMaxChildren] = {};
};

void *CreateNode(RTCThreadLocalAllocator allocator, unsigned int child_count, void *)
{
    void *raw = rtcThreadLocalAlloc(allocator, sizeof(CountNode), 16);
    auto *node = new (raw) CountNode();
    node->leaf = false;
    node->count = std::min(child_count, kMaxChildren);
    return node;
}

void SetNodeChildren(void *node_ptr, void **children, unsigned int child_count, void *)
{
    auto *node = static_cast<CountNode *>(node_ptr);
    node->count = std::min(child_count, kMaxChildren);
    for (unsigned int i = 0; i < node->count; ++i)
    {
        node->children[i] = static_cast<CountNode *>(children[i]);
    }
}

// Bounds are unused for topology counts; Embree still requires the callback.
void SetNodeBounds(void *, const RTCBounds **, unsigned int, void *) {}

void *CreateLeaf(RTCThreadLocalAllocator allocator, const RTCBuildPrimitive *,
                 std::size_t primitive_count, void *)
{
    void *raw = rtcThreadLocalAlloc(allocator, sizeof(CountNode), 16);
    auto *node = new (raw) CountNode();
    node->leaf = true;
    node->count = static_cast<unsigned int>(primitive_count);
    return node;
}

void SplitPrimitive(const RTCBuildPrimitive *primitive, unsigned int dimension,
                    float position, RTCBounds *left, RTCBounds *right, void *)
{
    // Spatial split used by the high-quality builder: clip the primitive box at
    // `position` along `dimension`.
    const float lower[3] = {primitive->lower_x, primitive->lower_y, primitive->lower_z};
    const float upper[3] = {primitive->upper_x, primitive->upper_y, primitive->upper_z};

    left->lower_x = lower[0];  left->lower_y = lower[1];  left->lower_z = lower[2];
    left->upper_x = upper[0];  left->upper_y = upper[1];  left->upper_z = upper[2];
    right->lower_x = lower[0]; right->lower_y = lower[1]; right->lower_z = lower[2];
    right->upper_x = upper[0]; right->upper_y = upper[1]; right->upper_z = upper[2];

    float *left_upper = &left->upper_x + dimension;
    float *right_lower = &right->lower_x + dimension;
    *left_upper = std::min(*left_upper, position);
    *right_lower = std::max(*right_lower, position);
}

// DFS over the proxy tree: depth, inner/leaf counts, and leaf prim-ref totals.
void Walk(const CountNode *node, std::size_t depth, BvhTreeStats &stats)
{
    if (node == nullptr)
    {
        return;
    }
    stats.max_depth = std::max(stats.max_depth, depth);
    if (node->leaf)
    {
        ++stats.leaves;
        stats.leaf_primitive_refs += node->count;
        return;
    }
    ++stats.inner_nodes;
    for (unsigned int i = 0; i < node->count; ++i)
    {
        Walk(node->children[i], depth + 1, stats);
    }
}

} // namespace

BvhTreeStats ComputeBvhTreeStats(RTCDevice device,
                                 const std::vector<RTCBuildPrimitive> &boxes,
                                 unsigned int branching_factor,
                                 unsigned int min_leaf_size,
                                 unsigned int max_leaf_size,
                                 unsigned int sah_block_size,
                                 RTCBuildQuality quality)
{
    BvhTreeStats stats;
    stats.primitives = boxes.size();
    if (boxes.empty())
    {
        return stats;
    }

    // The high-quality builder may split primitives, so it needs headroom past
    // the input count; Embree's own tutorials reserve ~2x for this.
    std::vector<RTCBuildPrimitive> scratch;
    scratch.reserve(boxes.size() * 2);
    scratch = boxes;
    scratch.resize(boxes.size() * 2);

    RTCBVH bvh = rtcNewBVH(device);

    // Same SAH knobs as the scene commit for the prim type under study.
    RTCBuildArguments args = rtcDefaultBuildArguments();
    args.byteSize = sizeof(args);
    args.buildQuality = quality;
    args.buildFlags = RTC_BUILD_FLAG_NONE;
    args.maxBranchingFactor = std::min(branching_factor, kMaxChildren);
    args.maxDepth = 1024;
    args.sahBlockSize = sah_block_size;
    args.minLeafSize = min_leaf_size;
    args.maxLeafSize = max_leaf_size;
    args.traversalCost = 1.0f;
    args.intersectionCost = 1.0f;
    args.bvh = bvh;
    args.primitives = scratch.data();
    args.primitiveCount = boxes.size();
    args.primitiveArrayCapacity = scratch.size();
    args.createNode = &CreateNode;
    args.setNodeChildren = &SetNodeChildren;
    args.setNodeBounds = &SetNodeBounds;
    args.createLeaf = &CreateLeaf;
    args.splitPrimitive = quality == RTC_BUILD_QUALITY_HIGH ? &SplitPrimitive : nullptr;
    args.buildProgress = nullptr;
    args.userPtr = nullptr;

    auto *root = static_cast<CountNode *>(rtcBuildBVH(&args));
    if (root == nullptr)
    {
        rtcReleaseBVH(bvh);
        throw std::runtime_error("rtcBuildBVH returned no root");
    }

    Walk(root, 1, stats);
    stats.mean_leaf_occupancy =
        stats.leaves ? static_cast<double>(stats.leaf_primitive_refs) / stats.leaves : 0.0;

    rtcReleaseBVH(bvh); // frees the nodes the callbacks allocated
    return stats;
}

} // namespace mfem_raytracing
