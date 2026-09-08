#ifndef MFEM_RAYTRACING_EMBREE_RAYTRACER_HPP
#define MFEM_RAYTRACING_EMBREE_RAYTRACER_HPP

#include "mfem_raytracing/embree/bilinear_patch_geometry.hpp"
#include "mfem_raytracing/embree/embree_interface.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mfem_raytracing
{

struct LeafPatchScene;

enum class PatchStoragePolicy { AutoIndexed, Dense };

struct BvhBuildOptions
{
    RTCBuildQuality quality = RTC_BUILD_QUALITY_HIGH;
    bool compact = false;
};

struct PatchStorageStatistics
{
    std::size_t dense_equivalent_bytes = 0;
    /// Allocated primitive/table buffer capacity; excludes BVH, allocator and
    /// container bookkeeping, caller-owned scenes and temporary build tables.
    std::size_t buffer_bytes = 0;
    std::size_t indexed_geometries = 0;
    std::size_t dense_geometries = 0;
    /// GetPatch compatibility copies, separate from buffer_bytes. Map node
    /// bookkeeping and allocation overhead are not included here.
    std::size_t compatibility_patch_bytes = 0;
};

/// Result of a first-hit ray query against bilinear patch geometry.
struct RayHitRecord
{
    bool hit = false;
    /// Embree ray parameter of the hit: P = org + t * dir (dir unnormalized).
    double t = std::numeric_limits<double>::infinity();
    /// Patch-local parameters of the hit, clamped to [0, 1]^2.
    double u = 0.0;
    double v = 0.0;
    /// Unnormalized geometric normal at the hit.
    double Ng[3] = {0.0, 0.0, 0.0};
    unsigned int geom_id = RTC_INVALID_GEOMETRY_ID;
    unsigned int prim_id = RTC_INVALID_GEOMETRY_ID;
};

/// Per-query observations from the bilinear user-geometry callback.
/// A zero `kernel_invocations` count means the BVH never sent this ray to a
/// bilinear leaf callback. `kernel_rejections` counts callback invocations for
/// which the bilinear solver reported no acceptable candidate.
struct RayQueryDiagnostics
{
    std::uint64_t kernel_invocations = 0;
    std::uint64_t kernel_rejections = 0;
    std::uint64_t reported_hits = 0;
    std::uint64_t reject_invalid_ray = 0;
    std::uint64_t reject_no_root = 0;
    std::uint64_t reject_denominator = 0;
    std::uint64_t reject_residual = 0;
    std::uint64_t reject_domain = 0;
    std::uint64_t reject_weight = 0;
    std::uint64_t reject_t_range = 0;
};

/// Embree-backed first-hit / occlusion ray tracer over bilinear patch
/// user geometries (clamped degree-(1, 1) NURBS/B-spline leaf patches).
///
/// Usage:
///   EmbreeRayTracer tracer;
///   unsigned int geom_id = tracer.RegisterPatches(std::move(patches));
///   tracer.CommitScene();
///   RayHitRecord hit = tracer.Intersect(org, dir);
class EmbreeRayTracer
{
public:
    EmbreeRayTracer();
    explicit EmbreeRayTracer(PatchStoragePolicy storage_policy);
    /// Tune static BVH construction. Robust traversal remains enabled.
    EmbreeRayTracer(PatchStoragePolicy storage_policy, const BvhBuildOptions &build_options);
    ~EmbreeRayTracer();

    EmbreeRayTracer(const EmbreeRayTracer &) = delete;
    EmbreeRayTracer &operator=(const EmbreeRayTracer &) = delete;

    /// Register one user geometry holding `patches`; the tracer takes ownership
    /// of the patch data; AutoIndexed shares exact positions/weight tuples
    /// when that reduces buffer capacity, otherwise retains dense storage. `box_bump`
    /// pads every per-patch BVH box by a constant amount. Returns the Embree
    /// geometry id, which appears as RayHitRecord::geom_id in query results.
    unsigned int RegisterPatches(std::vector<BilinearPatchPrimitive> patches,
                                 double box_bump = 0.0);

    /// Register a JSON-derived leaf scene through its explicit RT admission
    /// gate.  Legacy scenes with no certificate are accepted; a baked shell
    /// that declares itself diagnostic-only requires an explicit override.
    /// The lower-level RegisterPatches overload remains available for callers
    /// constructing primitives programmatically.
    unsigned int RegisterLeafPatchScene(const LeafPatchScene &scene,
                                        bool allow_diagnostic_shell = false,
                                        double box_bump = 0.0);

    /// Build/rebuild the BVH. Must be called after RegisterPatches and before
    /// any queries.
    void CommitScene();

    /// First hit along org + t * dir for t in [tnear, tfar].
    RayHitRecord Intersect(const double origin[3],
                           const double direction[3],
                           double tnear = 0.0,
                           double tfar = std::numeric_limits<double>::infinity(),
                           RayQueryDiagnostics *diagnostics = nullptr) const;

    /// Diagnostic reference query: evaluate every registered patch directly,
    /// bypassing Embree's BVH. This is intentionally slow and returns only the
    /// closest direct hit, using the same float solver as the callback.
    RayHitRecord IntersectBruteForce(const double origin[3],
                                     const double direction[3],
                                     double tnear = 0.0,
                                     double tfar = std::numeric_limits<double>::infinity()) const;

    /// Diagnostic reference query: collect one direct candidate from every
    /// registered patch, sorted by t. Unlike IntersectAll, this never advances
    /// a cursor and therefore exposes overlapping leaf coverage.
    std::vector<RayHitRecord> IntersectAllBruteForce(
        const double origin[3], const double direction[3], double tnear = 0.0,
        double tfar = std::numeric_limits<double>::infinity(),
        std::size_t max_hits = std::numeric_limits<std::size_t>::max()) const;

    /// All distinct surface crossings along org + t * dir for t in [tnear,
    /// tfar], in increasing-t order. Traversal resumes at the next
    /// representable float t after each raw hit, then clusters nearly equal t
    /// values so shared leaf coverage does not produce duplicate crossings.
    /// Stops after `max_hits` retained crossings or when no further hit is found.
    std::vector<RayHitRecord> IntersectAll(const double origin[3],
                                           const double direction[3],
                                           double tnear = 0.0,
                                           double tfar = std::numeric_limits<double>::infinity(),
                                           std::size_t max_hits = 64) const;

    /// Whether any geometry occludes the segment t in [tnear, tfar].
    bool Occluded(const double origin[3],
                  const double direction[3],
                  double tnear = 0.0,
                  double tfar = std::numeric_limits<double>::infinity()) const;

    /// The patch behind a query result, or nullptr for an unknown id pair.
    /// For indexed geometry this lazily caches an exact compatibility copy.
    /// Returned pointers stay valid until tracer destruction. Use CopyPatch
    /// for bulk inspection without accumulating compatibility copies.
    const BilinearPatchPrimitive *GetPatch(unsigned int geom_id, unsigned int prim_id) const;

    bool CopyPatch(unsigned int geom_id, unsigned int prim_id,
                   BilinearPatchPrimitive &patch) const;
    PatchStorageStatistics StorageStatistics() const;

    /// Total number of patches across all registered geometries.
    std::size_t PatchCount() const;

    /// The underlying Embree device, for instrumentation such as
    /// DeviceMemoryMonitor. Attach any monitor before CommitScene().
    RTCDevice Device() const { return device_; }

private:
    struct GeometrySlot
    {
        std::vector<BilinearPatchPrimitive> patches;
        std::unique_ptr<IndexedBilinearPatchStorage> indexed;
        mutable std::mutex cache_mutex;
        mutable std::map<unsigned int, BilinearPatchPrimitive> compatibility_cache;
        BilinearPatchGeometryData data;
    };

    RTCDevice device_ = nullptr;
    RTCScene scene_ = nullptr;
    bool committed_ = false;
    PatchStoragePolicy storage_policy_ = PatchStoragePolicy::AutoIndexed;
    // Heap slots so the buffers Embree points at survive map rehashing.
    std::unordered_map<unsigned int, std::unique_ptr<GeometrySlot>> geometry_slots_;
};

} // namespace mfem_raytracing

#endif
