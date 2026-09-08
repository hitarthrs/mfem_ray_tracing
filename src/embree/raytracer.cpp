// Embree device/scene owner for bilinear user geometries: register patches,
// commit BVH, and query closest / all / occluded hits (plus brute-force checks).

#include "mfem_raytracing/embree/raytracer.hpp"

#include "mfem_raytracing/embree/bilinear_intersect.hpp"
#include "mfem_raytracing/embree/leaf_patch_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace mfem_raytracing
{
namespace
{

constexpr double kHitDedupeAbsoluteTolerance = 1e-5;
constexpr double kHitDedupeRelativeTolerance = 1e-6;

double HitDedupeTolerance(double a, double b)
{
    return std::max(kHitDedupeAbsoluteTolerance,
                    kHitDedupeRelativeTolerance * std::max(std::fabs(a), std::fabs(b)));
}

void EmbreeErrorFunc(void * /*user_ptr*/, RTCError code, const char *str)
{
    if (code != RTC_ERROR_NONE)
    {
        std::fprintf(stderr, "Embree error (%d): %s\n", static_cast<int>(code),
                     str ? str : "<no message>");
    }
}

} // namespace

EmbreeRayTracer::EmbreeRayTracer() : EmbreeRayTracer(PatchStoragePolicy::AutoIndexed) {}

EmbreeRayTracer::EmbreeRayTracer(PatchStoragePolicy storage_policy)
    : EmbreeRayTracer(storage_policy, BvhBuildOptions{}) {}

EmbreeRayTracer::EmbreeRayTracer(PatchStoragePolicy storage_policy,
                                 const BvhBuildOptions &build_options)
    : storage_policy_(storage_policy)
{
    if (build_options.quality != RTC_BUILD_QUALITY_LOW &&
        build_options.quality != RTC_BUILD_QUALITY_MEDIUM &&
        build_options.quality != RTC_BUILD_QUALITY_HIGH)
    {
        throw std::invalid_argument("BVH quality must be LOW, MEDIUM, or HIGH");
    }
    device_ = rtcNewDevice(nullptr);
    rtcSetDeviceErrorFunction(device_, EmbreeErrorFunc, nullptr);

    scene_ = rtcNewScene(device_);
    rtcSetSceneFlags(scene_, static_cast<RTCSceneFlags>(RTC_SCENE_FLAG_ROBUST |
                      (build_options.compact ? RTC_SCENE_FLAG_COMPACT : RTC_SCENE_FLAG_NONE)));
    rtcSetSceneBuildQuality(scene_, build_options.quality);
}

EmbreeRayTracer::~EmbreeRayTracer()
{
    if (scene_ != nullptr)
    {
        rtcReleaseScene(scene_);
    }
    if (device_ != nullptr)
    {
        rtcReleaseDevice(device_);
    }
}

unsigned int EmbreeRayTracer::RegisterPatches(std::vector<BilinearPatchPrimitive> patches,
                                              double box_bump)
{
    auto slot = std::make_unique<GeometrySlot>();
    slot->data.primitive_count = patches.size();
    // AutoIndexed: keep the indexed form only when it is strictly smaller than
    // a dense BilinearPatchPrimitive buffer (plus the storage object itself).
    if (storage_policy_ == PatchStoragePolicy::AutoIndexed && !patches.empty())
    {
        auto indexed = std::make_unique<IndexedBilinearPatchStorage>(patches);
        if (indexed->BufferBytes() + sizeof(IndexedBilinearPatchStorage) <
            patches.size() * sizeof(BilinearPatchPrimitive))
        {
            slot->indexed = std::move(indexed);
            slot->data.indexed = slot->indexed.get();
        }
    }
    // Dense fallback: Embree callbacks read prim_ref_buffer[primID] via View/Load.
    if (!slot->indexed)
    {
        slot->patches = std::move(patches);
        slot->data.prim_ref_buffer = slot->patches.data();
    }
    slot->data.box_bump = box_bump;

    // User geometry: bounds / intersect / occluded are the bilinear callbacks.
    RTCGeometry geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_USER);
    rtcSetGeometryUserPrimitiveCount(geometry, slot->data.primitive_count);
    rtcSetGeometryUserData(geometry, &slot->data);
    rtcSetGeometryBoundsFunction(geometry, &BilinearPatchBoundsFunc, nullptr);
    rtcSetGeometryIntersectFunction(geometry, &BilinearPatchIntersectionFunc);
    rtcSetGeometryOccludedFunction(geometry, &BilinearPatchOccludedFunc);
    rtcCommitGeometry(geometry);

    const unsigned int geom_id = rtcAttachGeometry(scene_, geometry);
    rtcReleaseGeometry(geometry); // scene now holds the reference

    geometry_slots_[geom_id] = std::move(slot);
    committed_ = false;
    return geom_id;
}

unsigned int EmbreeRayTracer::RegisterLeafPatchScene(const LeafPatchScene &scene,
                                                     bool allow_diagnostic_shell,
                                                     double box_bump)
{
    scene.RequireRayTracingCertified(allow_diagnostic_shell);
    return RegisterPatches(scene.Patches(), box_bump);
}

void EmbreeRayTracer::CommitScene()
{
    rtcCommitScene(scene_);
    committed_ = true;
}

RayHitRecord EmbreeRayTracer::Intersect(const double origin[3],
                                        const double direction[3],
                                        double tnear,
                                        double tfar,
                                        RayQueryDiagnostics *diagnostics) const
{
    RTCRayHit rayhit{};
    rayhit.ray.org_x = static_cast<float>(origin[0]);
    rayhit.ray.org_y = static_cast<float>(origin[1]);
    rayhit.ray.org_z = static_cast<float>(origin[2]);
    rayhit.ray.dir_x = static_cast<float>(direction[0]);
    rayhit.ray.dir_y = static_cast<float>(direction[1]);
    rayhit.ray.dir_z = static_cast<float>(direction[2]);
    rayhit.ray.tnear = static_cast<float>(tnear);
    rayhit.ray.tfar = static_cast<float>(tfar);
    rayhit.ray.mask = ~0u;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    for (unsigned int level = 0; level < RTC_MAX_INSTANCE_LEVEL_COUNT; ++level)
    {
        rayhit.hit.instID[level] = RTC_INVALID_GEOMETRY_ID;
    }

    if (diagnostics != nullptr)
    {
        *diagnostics = {};
    }
    struct DiagnosticsScope
    {
        explicit DiagnosticsScope(RayQueryDiagnostics *diagnostics)
            : previous(SetActiveRayQueryDiagnostics(diagnostics))
        {
        }
        ~DiagnosticsScope() { SetActiveRayQueryDiagnostics(previous); }

        RayQueryDiagnostics *previous;
    } diagnostics_scope(diagnostics);

    rtcIntersect1(scene_, &rayhit);

    RayHitRecord record;
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
    {
        return record;
    }
    record.hit = true;
    record.t = rayhit.ray.tfar;
    record.u = rayhit.hit.u;
    record.v = rayhit.hit.v;
    record.Ng[0] = rayhit.hit.Ng_x;
    record.Ng[1] = rayhit.hit.Ng_y;
    record.Ng[2] = rayhit.hit.Ng_z;
    record.geom_id = rayhit.hit.geomID;
    record.prim_id = rayhit.hit.primID;
    return record;
}

RayHitRecord EmbreeRayTracer::IntersectBruteForce(const double origin[3],
                                                  const double direction[3],
                                                  double tnear,
                                                  double tfar) const
{
    const float origin_f[3] = {static_cast<float>(origin[0]), static_cast<float>(origin[1]),
                               static_cast<float>(origin[2])};
    const float direction_f[3] = {static_cast<float>(direction[0]), static_cast<float>(direction[1]),
                                  static_cast<float>(direction[2])};
    const float tnear_f = static_cast<float>(tnear);
    float best_t = static_cast<float>(tfar);

    RayHitRecord record;
    for (const auto &entry : geometry_slots_)
    {
        const unsigned int geom_id = entry.first;
        const GeometrySlot &slot = *entry.second;
        for (std::size_t prim_id = 0; prim_id < slot.data.primitive_count; ++prim_id)
        {
            // Load materializes indexed patches into scratch; dense uses the buffer.
            BilinearPatchPrimitive scratch;
            const BilinearPatchRayHit hit = IntersectBilinearPatchDirect(
                slot.data.Load(prim_id, scratch), origin_f, direction_f, tnear_f, best_t);
            if (!hit.hit)
            {
                continue;
            }
            best_t = hit.t;
            record.hit = true;
            record.t = hit.t;
            record.u = hit.u;
            record.v = hit.v;
            record.Ng[0] = hit.Ng[0];
            record.Ng[1] = hit.Ng[1];
            record.Ng[2] = hit.Ng[2];
            record.geom_id = geom_id;
            record.prim_id = static_cast<unsigned int>(prim_id);
        }
    }
    return record;
}

std::vector<RayHitRecord> EmbreeRayTracer::IntersectAllBruteForce(
    const double origin[3], const double direction[3], double tnear, double tfar,
    std::size_t max_hits) const
{
    const float origin_f[3] = {static_cast<float>(origin[0]), static_cast<float>(origin[1]),
                               static_cast<float>(origin[2])};
    const float direction_f[3] = {static_cast<float>(direction[0]), static_cast<float>(direction[1]),
                                  static_cast<float>(direction[2])};
    const float tnear_f = static_cast<float>(tnear);
    const float tfar_f = static_cast<float>(tfar);

    std::vector<RayHitRecord> hits;
    for (const auto &entry : geometry_slots_)
    {
        const unsigned int geom_id = entry.first;
        const GeometrySlot &slot = *entry.second;
        for (std::size_t prim_id = 0; prim_id < slot.data.primitive_count; ++prim_id)
        {
            BilinearPatchPrimitive scratch;
            const BilinearPatchRayHit hit = IntersectBilinearPatchDirect(
                slot.data.Load(prim_id, scratch), origin_f, direction_f, tnear_f, tfar_f);
            if (!hit.hit)
            {
                continue;
            }
            RayHitRecord record;
            record.hit = true;
            record.t = hit.t;
            record.u = hit.u;
            record.v = hit.v;
            record.Ng[0] = hit.Ng[0];
            record.Ng[1] = hit.Ng[1];
            record.Ng[2] = hit.Ng[2];
            record.geom_id = geom_id;
            record.prim_id = static_cast<unsigned int>(prim_id);
            hits.push_back(record);
        }
    }
    std::sort(hits.begin(), hits.end(), [](const RayHitRecord &a, const RayHitRecord &b) {
        return a.t < b.t;
    });
    if (hits.size() > max_hits)
    {
        hits.resize(max_hits);
    }
    return hits;
}

std::vector<RayHitRecord> EmbreeRayTracer::IntersectAll(const double origin[3],
                                                        const double direction[3],
                                                        double tnear,
                                                        double tfar,
                                                        std::size_t max_hits) const
{
    std::vector<RayHitRecord> hits;
    hits.reserve(std::min(max_hits, std::size_t{8}));

    double cursor = tnear;
    while (hits.size() < max_hits && cursor < tfar)
    {
        const RayHitRecord hit = Intersect(origin, direction, cursor, tfar);
        if (!hit.hit)
        {
            break;
        }
        // Advance only to the next representable float t. The ray data seen by
        // Embree and the callback is float, so this excludes the exact same
        // candidate without skipping a distinct nearby surface.
        const float next_t = std::nextafter(static_cast<float>(hit.t),
                                            std::numeric_limits<float>::infinity());
        cursor = static_cast<double>(next_t);
        if (!(cursor > hit.t))
        {
            break; // defensive guard against a non-finite/non-advancing t
        }

        if (!hits.empty() && std::fabs(hit.t - hits.back().t) <= HitDedupeTolerance(hit.t, hits.back().t))
        {
            // The earlier record is the representative for this crossing.
            continue;
        }
        hits.push_back(hit);
    }
    return hits;
}

bool EmbreeRayTracer::Occluded(const double origin[3],
                               const double direction[3],
                               double tnear,
                               double tfar) const
{
    RTCRay ray{};
    ray.org_x = static_cast<float>(origin[0]);
    ray.org_y = static_cast<float>(origin[1]);
    ray.org_z = static_cast<float>(origin[2]);
    ray.dir_x = static_cast<float>(direction[0]);
    ray.dir_y = static_cast<float>(direction[1]);
    ray.dir_z = static_cast<float>(direction[2]);
    ray.tnear = static_cast<float>(tnear);
    ray.tfar = static_cast<float>(tfar);
    ray.mask = ~0u;

    rtcOccluded1(scene_, &ray);
    return ray.tfar == -std::numeric_limits<float>::infinity();
}

const BilinearPatchPrimitive *EmbreeRayTracer::GetPatch(unsigned int geom_id,
                                                        unsigned int prim_id) const
{
    const auto it = geometry_slots_.find(geom_id);
    if (it == geometry_slots_.end() || prim_id >= it->second->data.primitive_count)
    {
        return nullptr;
    }
    const GeometrySlot &slot = *it->second;
    if (!slot.indexed) { return &slot.patches[prim_id]; }
    // Indexed slots have no stable dense address; cache an exact copy for callers
    // that need a BilinearPatchPrimitive* (see also CopyPatch / Load).
    std::lock_guard<std::mutex> lock(slot.cache_mutex);
    auto found = slot.compatibility_cache.find(prim_id);
    if (found == slot.compatibility_cache.end())
    {
        found = slot.compatibility_cache.emplace(prim_id, slot.indexed->Load(prim_id)).first;
    }
    return &found->second;
}

bool EmbreeRayTracer::CopyPatch(unsigned int geom_id, unsigned int prim_id,
                                BilinearPatchPrimitive &patch) const
{
    const auto it = geometry_slots_.find(geom_id);
    if (it == geometry_slots_.end() || prim_id >= it->second->data.primitive_count)
    {
        return false;
    }
    BilinearPatchPrimitive scratch;
    patch = it->second->data.Load(prim_id, scratch);
    return true;
}

PatchStorageStatistics EmbreeRayTracer::StorageStatistics() const
{
    PatchStorageStatistics stats;
    for (const auto &entry : geometry_slots_)
    {
        const auto &slot = *entry.second;
        stats.dense_equivalent_bytes += slot.data.primitive_count * sizeof(BilinearPatchPrimitive);
        if (slot.indexed)
        {
            ++stats.indexed_geometries;
            stats.buffer_bytes += slot.indexed->BufferBytes();
        }
        else
        {
            ++stats.dense_geometries;
            stats.buffer_bytes += slot.patches.capacity() * sizeof(BilinearPatchPrimitive);
        }
        std::lock_guard<std::mutex> lock(slot.cache_mutex);
        stats.compatibility_patch_bytes += slot.compatibility_cache.size() * sizeof(BilinearPatchPrimitive);
    }
    return stats;
}

std::size_t EmbreeRayTracer::PatchCount() const
{
    std::size_t count = 0;
    for (const auto &entry : geometry_slots_)
    {
        count += entry.second->data.primitive_count;
    }
    return count;
}

} // namespace mfem_raytracing
