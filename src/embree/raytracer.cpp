#include "mfem_raytracing/embree/raytracer.hpp"

#include "mfem_raytracing/embree/bilinear_intersect.hpp"
#include "mfem_raytracing/embree/leaf_patch_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

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

EmbreeRayTracer::EmbreeRayTracer()
{
    device_ = rtcNewDevice(nullptr);
    rtcSetDeviceErrorFunction(device_, EmbreeErrorFunc, nullptr);

    scene_ = rtcNewScene(device_);
    rtcSetSceneFlags(scene_, RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(scene_, RTC_BUILD_QUALITY_HIGH);
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
    slot->patches = std::move(patches);
    slot->data.prim_ref_buffer = slot->patches.data();
    slot->data.primitive_count = slot->patches.size();
    slot->data.box_bump = box_bump;

    RTCGeometry geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_USER);
    rtcSetGeometryUserPrimitiveCount(geometry, slot->patches.size());
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
        for (std::size_t prim_id = 0; prim_id < slot.patches.size(); ++prim_id)
        {
            const BilinearPatchRayHit hit = IntersectBilinearPatchDirect(
                slot.patches[prim_id], origin_f, direction_f, tnear_f, best_t);
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
        for (std::size_t prim_id = 0; prim_id < slot.patches.size(); ++prim_id)
        {
            const BilinearPatchRayHit hit = IntersectBilinearPatchDirect(
                slot.patches[prim_id], origin_f, direction_f, tnear_f, tfar_f);
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
    if (it == geometry_slots_.end() || prim_id >= it->second->patches.size())
    {
        return nullptr;
    }
    return &it->second->patches[prim_id];
}

std::size_t EmbreeRayTracer::PatchCount() const
{
    std::size_t count = 0;
    for (const auto &entry : geometry_slots_)
    {
        count += entry.second->patches.size();
    }
    return count;
}

} // namespace mfem_raytracing
