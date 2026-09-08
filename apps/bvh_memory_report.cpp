// Measure the Embree BVH memory footprint of the same surface represented two
// ways: as a tessellated triangle mesh, and as our bilinear patch primitives.
//
// BVH bytes come from Embree's own device memory monitor, which sees every
// internal allocation (nodes and leaf blocks). Geometry payload bytes -- the
// buffers the application owns and Embree only points at -- are accounted
// separately, because they are the part the memory monitor does not see.
//
// Usage:
//   bvh_memory_report --patches leaves.json [--stl mesh.stl] [--json out.json]
//
// Either input may be given alone; with both, a comparison block is printed.

#include "mfem_raytracing/embree/bvh_tree_stats.hpp"
#include "mfem_raytracing/embree/device_memory_monitor.hpp"
#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "mfem_raytracing/embree/raytracer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using mfem_raytracing::BvhTreeStats;
using mfem_raytracing::DeviceMemoryMonitor;

struct SceneMemory
{
    std::string label;
    std::size_t primitives = 0;
    std::size_t vertices = 0;      // triangle meshes only
    std::int64_t bvh_bytes = 0;    // Embree-internal, from the memory monitor
    std::int64_t peak_bytes = 0;   // includes transient build scratch
    std::size_t dense_payload_bytes = 0;
    std::int64_t payload_bytes = 0; // application-owned primitive storage
    double build_seconds = 0.0;

    std::int64_t alloc_events = 0;   // allocator slabs, not nodes
    std::int64_t largest_alloc = 0;

    BvhTreeStats tree;               // exact node/leaf counts, when requested
    bool has_tree = false;
    /// Bytes one leaf occupies for this primitive type: a Triangle4v block
    /// stores four triangles' vertices inline, while a user-geometry Object
    /// leaf stores only a (geomID, primID) reference.
    std::int64_t leaf_bytes = 0;

    std::int64_t total_bytes() const { return bvh_bytes + payload_bytes; }
};

/// Embree's BVH width: 4 for an SSE2/NEON build, 8 once AVX is enabled.
unsigned int g_branching_factor = 4;
/// Count the tree as well as the bytes.
bool g_tree_stats = false;

double MiB(std::int64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

constexpr float kInf = std::numeric_limits<float>::infinity();

RTCBuildPrimitive MakeBuildPrimitive(const float lower[3], const float upper[3],
                                     unsigned int geom_id, unsigned int prim_id)
{
    RTCBuildPrimitive p{};
    p.lower_x = lower[0]; p.lower_y = lower[1]; p.lower_z = lower[2];
    p.upper_x = upper[0]; p.upper_y = upper[1]; p.upper_z = upper[2];
    p.geomID = geom_id;
    p.primID = prim_id;
    return p;
}

// ---------------------------------------------------------------- STL loading

struct TriangleMesh
{
    std::vector<float> vertices;      // 3 floats per vertex
    std::vector<unsigned int> indices; // 3 indices per triangle
};

/// Weld an STL's per-facet vertices into an indexed mesh. STL stores every
/// facet's corners independently; a renderer would share them, so welding is
/// what makes the byte comparison honest.
class VertexWelder
{
public:
    unsigned int Add(TriangleMesh &mesh, float x, float y, float z)
    {
        const Key key{x, y, z};
        auto it = table_.find(key);
        if (it != table_.end())
        {
            return it->second;
        }
        const auto id = static_cast<unsigned int>(mesh.vertices.size() / 3);
        mesh.vertices.push_back(x);
        mesh.vertices.push_back(y);
        mesh.vertices.push_back(z);
        table_.emplace(key, id);
        return id;
    }

private:
    struct Key
    {
        float x, y, z;
        bool operator<(const Key &o) const
        {
            if (x != o.x) { return x < o.x; }
            if (y != o.y) { return y < o.y; }
            return z < o.z;
        }
    };
    std::map<Key, unsigned int> table_;
};

bool LooksLikeBinaryStl(std::ifstream &in)
{
    char header[6] = {};
    in.read(header, 5);
    in.clear();
    in.seekg(0);
    return std::strncmp(header, "solid", 5) != 0;
}

TriangleMesh LoadBinaryStl(std::ifstream &in)
{
    TriangleMesh mesh;
    VertexWelder welder;

    char header[80] = {};
    in.read(header, 80);
    std::uint32_t facet_count = 0;
    in.read(reinterpret_cast<char *>(&facet_count), sizeof(facet_count));

    mesh.indices.reserve(static_cast<std::size_t>(facet_count) * 3);
    for (std::uint32_t f = 0; f < facet_count; ++f)
    {
        float raw[12] = {};
        in.read(reinterpret_cast<char *>(raw), sizeof(raw));
        std::uint16_t attr = 0;
        in.read(reinterpret_cast<char *>(&attr), sizeof(attr));
        if (!in)
        {
            throw std::runtime_error("truncated binary STL");
        }
        for (int c = 0; c < 3; ++c)
        {
            mesh.indices.push_back(
                welder.Add(mesh, raw[3 + 3 * c], raw[4 + 3 * c], raw[5 + 3 * c]));
        }
    }
    return mesh;
}

TriangleMesh LoadAsciiStl(std::ifstream &in)
{
    TriangleMesh mesh;
    VertexWelder welder;

    std::string token;
    std::vector<unsigned int> facet;
    facet.reserve(3);
    while (in >> token)
    {
        if (token != "vertex")
        {
            continue;
        }
        float x = 0.0f, y = 0.0f, z = 0.0f;
        if (!(in >> x >> y >> z))
        {
            throw std::runtime_error("malformed vertex in ASCII STL");
        }
        facet.push_back(welder.Add(mesh, x, y, z));
        if (facet.size() == 3)
        {
            mesh.indices.insert(mesh.indices.end(), facet.begin(), facet.end());
            facet.clear();
        }
    }
    if (!facet.empty())
    {
        throw std::runtime_error("ASCII STL ended mid-facet");
    }
    return mesh;
}

TriangleMesh LoadStl(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("cannot open STL: " + path);
    }
    return LooksLikeBinaryStl(in) ? LoadBinaryStl(in) : LoadAsciiStl(in);
}

// ------------------------------------------------------------- measurements

SceneMemory MeasureTriangleScene(const TriangleMesh &mesh, const std::string &label)
{
    SceneMemory out;
    out.label = label;
    out.primitives = mesh.indices.size() / 3;
    out.vertices = mesh.vertices.size() / 3;

    DeviceMemoryMonitor monitor;
    RTCDevice device = rtcNewDevice(nullptr);
    monitor.AttachTo(device);
    // Attaching after rtcNewDevice misses the device's own fixed overhead;
    // zeroing here scopes the count to the scene build that follows.
    monitor.Reset();

    RTCScene scene = rtcNewScene(device);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    // Shared buffers: Embree points at our storage instead of copying it, so
    // the monitor reports the BVH alone and the payload stays explicit below.
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                               mesh.vertices.data(), 0, 3 * sizeof(float), out.vertices);
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                               mesh.indices.data(), 0, 3 * sizeof(unsigned int),
                               out.primitives);
    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);

    const auto t0 = std::chrono::steady_clock::now();
    rtcCommitScene(scene);
    const auto t1 = std::chrono::steady_clock::now();
    out.build_seconds = std::chrono::duration<double>(t1 - t0).count();

    out.bvh_bytes = monitor.live_bytes();
    out.peak_bytes = monitor.peak_bytes();
    out.alloc_events = monitor.alloc_events();
    out.largest_alloc = monitor.largest_alloc();
    out.payload_bytes =
        static_cast<std::int64_t>(mesh.vertices.size() * sizeof(float) +
                                  mesh.indices.size() * sizeof(unsigned int));

    if (g_tree_stats)
    {
        std::vector<RTCBuildPrimitive> boxes(out.primitives);
        for (std::size_t t = 0; t < out.primitives; ++t)
        {
            float lo[3] = {kInf, kInf, kInf};
            float hi[3] = {-kInf, -kInf, -kInf};
            for (int c = 0; c < 3; ++c)
            {
                const unsigned int v = mesh.indices[3 * t + c];
                for (int a = 0; a < 3; ++a)
                {
                    lo[a] = std::min(lo[a], mesh.vertices[3 * v + a]);
                    hi[a] = std::max(hi[a], mesh.vertices[3 * v + a]);
                }
            }
            boxes[t] = MakeBuildPrimitive(lo, hi, 0, static_cast<unsigned int>(t));
        }
        // Triangle4v packs four triangles per leaf, and the builder scores
        // split candidates in blocks of four to match.
        out.tree = mfem_raytracing::ComputeBvhTreeStats(device, boxes, g_branching_factor,
                                                        4, 4, 4);
        // Triangle4v: 4 lanes x 3 verts x 3 coords x 4 B, plus 4-wide geomID
        // and primID arrays.
        out.leaf_bytes = 4 * 3 * 3 * 4 + 4 * 4 + 4 * 4;
        out.has_tree = true;
    }

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
    return out;
}

SceneMemory MeasurePatchScene(const mfem_raytracing::LeafPatchScene &leaf_scene,
                              const std::string &label,
                              mfem_raytracing::PatchStoragePolicy storage_policy)
{
    SceneMemory out;
    out.label = label;
    out.primitives = leaf_scene.leaves.size();

    DeviceMemoryMonitor monitor;
    mfem_raytracing::EmbreeRayTracer tracer(storage_policy);
    monitor.AttachTo(tracer.Device());
    monitor.Reset();

    tracer.RegisterLeafPatchScene(leaf_scene, /*allow_diagnostic_shell=*/true);

    const auto t0 = std::chrono::steady_clock::now();
    tracer.CommitScene();
    const auto t1 = std::chrono::steady_clock::now();
    out.build_seconds = std::chrono::duration<double>(t1 - t0).count();

    out.bvh_bytes = monitor.live_bytes();
    out.peak_bytes = monitor.peak_bytes();
    out.alloc_events = monitor.alloc_events();
    out.largest_alloc = monitor.largest_alloc();
    const auto storage = tracer.StorageStatistics();
    out.payload_bytes = static_cast<std::int64_t>(storage.buffer_bytes);
    out.dense_payload_bytes = storage.dense_equivalent_bytes;

    if (g_tree_stats)
    {
        std::vector<RTCBuildPrimitive> boxes(out.primitives);
        for (std::size_t i = 0; i < out.primitives; ++i)
        {
            const mfem_raytracing::AxisAlignedBounds &b = leaf_scene.leaves[i].bbox;
            const float lo[3] = {static_cast<float>(b.min[0]), static_cast<float>(b.min[1]),
                                 static_cast<float>(b.min[2])};
            const float hi[3] = {static_cast<float>(b.max[0]), static_cast<float>(b.max[1]),
                                 static_cast<float>(b.max[2])};
            boxes[i] = MakeBuildPrimitive(lo, hi, 0, static_cast<unsigned int>(i));
        }
        // User-geometry Object leaves hold a single primitive reference.
        out.tree = mfem_raytracing::ComputeBvhTreeStats(tracer.Device(), boxes,
                                                        g_branching_factor, 1, 1, 1);
        out.leaf_bytes = 2 * sizeof(unsigned int); // geomID + primID
        out.has_tree = true;
    }
    return out;
}

// ------------------------------------------------------------------ printing

void PrintScene(const SceneMemory &s)
{
    std::printf("%s\n", s.label.c_str());
    std::printf("  primitives          %12zu\n", s.primitives);
    if (s.vertices > 0)
    {
        std::printf("  vertices            %12zu\n", s.vertices);
    }
    std::printf("  BVH (Embree)        %12lld B  %9.2f MiB  %8.1f B/prim\n",
                static_cast<long long>(s.bvh_bytes), MiB(s.bvh_bytes),
                s.primitives ? static_cast<double>(s.bvh_bytes) / s.primitives : 0.0);
    std::printf("  geometry payload    %12lld B  %9.2f MiB  %8.1f B/prim\n",
                static_cast<long long>(s.payload_bytes), MiB(s.payload_bytes),
                s.primitives ? static_cast<double>(s.payload_bytes) / s.primitives : 0.0);
    if (s.dense_payload_bytes > 0)
    {
        std::printf("  dense patch baseline %11zu B  payload reduction %.1f%%\n",
                    s.dense_payload_bytes,
                    100.0 * (1.0 - static_cast<double>(s.payload_bytes) / s.dense_payload_bytes));
    }
    std::printf("  total resident      %12lld B  %9.2f MiB  %8.1f B/prim\n",
                static_cast<long long>(s.total_bytes()), MiB(s.total_bytes()),
                s.primitives ? static_cast<double>(s.total_bytes()) / s.primitives : 0.0);
    std::printf("  build peak (device) %12lld B  %9.2f MiB\n",
                static_cast<long long>(s.peak_bytes), MiB(s.peak_bytes));
    std::printf("  build time          %12.3f s\n", s.build_seconds);
    std::printf("  allocator slabs     %12lld  (largest %lld B) -- slabs, not nodes\n",
                static_cast<long long>(s.alloc_events),
                static_cast<long long>(s.largest_alloc));
    if (s.has_tree)
    {
        std::printf("  tree: inner nodes   %12zu\n", s.tree.inner_nodes);
        std::printf("        leaves        %12zu  (mean %.2f prims/leaf)\n",
                    s.tree.leaves, s.tree.mean_leaf_occupancy);
        std::printf("        prim refs     %12zu  (%.2fx input, spatial splits)\n",
                    s.tree.leaf_primitive_refs,
                    s.primitives ? static_cast<double>(s.tree.leaf_primitive_refs) /
                                       s.primitives : 0.0);
        std::printf("        max depth     %12zu\n", s.tree.max_depth);
        std::printf("        nodes/prim    %12.3f\n",
                    s.primitives ? static_cast<double>(s.tree.inner_nodes + s.tree.leaves) /
                                       s.primitives : 0.0);
        // A BVH-N inner node stores N child pointers (8 B) plus N boxes
        // (6 floats), so 32 N bytes wide.
        const std::int64_t inner_node_bytes = 32 * g_branching_factor;
        const std::int64_t model_inner =
            static_cast<std::int64_t>(s.tree.inner_nodes) * inner_node_bytes;
        const std::int64_t model_leaf =
            static_cast<std::int64_t>(s.tree.leaves) * s.leaf_bytes;
        std::printf("        inner bytes   %12lld  (%zu x %lld B)\n",
                    static_cast<long long>(model_inner), s.tree.inner_nodes,
                    static_cast<long long>(inner_node_bytes));
        std::printf("        leaf bytes    %12lld  (%zu x %lld B)\n",
                    static_cast<long long>(model_leaf), s.tree.leaves,
                    static_cast<long long>(s.leaf_bytes));
        std::printf("        model total   %12lld  (%.0f%% of measured BVH)\n",
                    static_cast<long long>(model_inner + model_leaf),
                    s.bvh_bytes ? 100.0 * static_cast<double>(model_inner + model_leaf) /
                                      s.bvh_bytes : 0.0);
    }
    std::printf("\n");
}

void WriteJson(const std::string &path, const std::vector<SceneMemory> &scenes)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("cannot write JSON: " + path);
    }
    out << "{\n  \"scenes\": [\n";
    for (std::size_t i = 0; i < scenes.size(); ++i)
    {
        const SceneMemory &s = scenes[i];
        out << "    {\"label\": \"" << s.label << "\""
            << ", \"primitives\": " << s.primitives
            << ", \"vertices\": " << s.vertices
            << ", \"bvh_bytes\": " << s.bvh_bytes
            << ", \"payload_bytes\": " << s.payload_bytes
            << ", \"dense_payload_bytes\": " << s.dense_payload_bytes
            << ", \"total_bytes\": " << s.total_bytes()
            << ", \"peak_bytes\": " << s.peak_bytes
            << ", \"build_seconds\": " << s.build_seconds
            << ", \"alloc_events\": " << s.alloc_events
            << ", \"inner_nodes\": " << s.tree.inner_nodes
            << ", \"leaves\": " << s.tree.leaves
            << ", \"leaf_primitive_refs\": " << s.tree.leaf_primitive_refs
            << ", \"max_depth\": " << s.tree.max_depth << "}"
            << (i + 1 < scenes.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
}

void PrintUsage()
{
    std::fprintf(stderr,
                 "usage: bvh_memory_report [--patches leaves.json] [--stl mesh.stl]\n"
                 "                         [--json out.json] [--tree-stats]\n"
                 "                         [--branching-factor N] [--dense-patches]\n");
}

} // namespace

int main(int argc, char **argv)
{
    std::string patches_path;
    std::string stl_path;
    std::string json_path;
    auto storage_policy = mfem_raytracing::PatchStoragePolicy::AutoIndexed;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const bool has_value = (i + 1 < argc);
        if (arg == "--patches" && has_value) { patches_path = argv[++i]; }
        else if (arg == "--stl" && has_value) { stl_path = argv[++i]; }
        else if (arg == "--json" && has_value) { json_path = argv[++i]; }
        else if (arg == "--dense-patches") { storage_policy = mfem_raytracing::PatchStoragePolicy::Dense; }
        else if (arg == "--tree-stats") { g_tree_stats = true; }
        else if (arg == "--branching-factor" && has_value)
        {
            g_branching_factor = static_cast<unsigned int>(std::stoul(argv[++i]));
        }
        else { PrintUsage(); return 2; }
    }

    if (patches_path.empty() && stl_path.empty())
    {
        PrintUsage();
        return 2;
    }

    try
    {
        std::vector<SceneMemory> scenes;

        if (!stl_path.empty())
        {
            const TriangleMesh mesh = LoadStl(stl_path);
            scenes.push_back(MeasureTriangleScene(mesh, "triangle mesh  [" + stl_path + "]"));
        }
        if (!patches_path.empty())
        {
            const mfem_raytracing::LeafPatchScene leaf_scene =
                mfem_raytracing::LoadLeafPatchScene(patches_path);
            scenes.push_back(
                MeasurePatchScene(leaf_scene, "bilinear patches  [" + patches_path + "]", storage_policy));
        }

        std::printf("\n");
        for (const SceneMemory &s : scenes)
        {
            PrintScene(s);
        }

        if (scenes.size() == 2)
        {
            const SceneMemory &tri = scenes[0];
            const SceneMemory &pat = scenes[1];
            std::printf("ratio (triangles / patches)\n");
            std::printf("  primitives          %12.1fx\n",
                        static_cast<double>(tri.primitives) /
                            std::max<std::size_t>(pat.primitives, 1));
            std::printf("  BVH bytes           %12.1fx\n",
                        static_cast<double>(tri.bvh_bytes) /
                            std::max<std::int64_t>(pat.bvh_bytes, 1));
            std::printf("  total resident      %12.1fx\n\n",
                        static_cast<double>(tri.total_bytes()) /
                            std::max<std::int64_t>(pat.total_bytes(), 1));
        }

        if (!json_path.empty())
        {
            WriteJson(json_path, scenes);
            std::printf("wrote %s\n", json_path.c_str());
        }
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
