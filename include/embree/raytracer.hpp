#ifndef MFEM_RAYTRACING_EMBREE_RAYTRACER_HPP
#define MFEM_RAYTRACING_EMBREE_RAYTRACER_HPP

#include "embree/bilinear_patch_geometry.hpp"
#include "embree/embree_interface.hpp"

#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mfem_raytracing
{

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
    ~EmbreeRayTracer();

    EmbreeRayTracer(const EmbreeRayTracer &) = delete;
    EmbreeRayTracer &operator=(const EmbreeRayTracer &) = delete;

    /// Register one user geometry holding `patches`; the tracer takes ownership
    /// of the patch storage (Embree keeps raw pointers into it). `box_bump`
    /// pads every per-patch BVH box by a constant amount. Returns the Embree
    /// geometry id, which appears as RayHitRecord::geom_id in query results.
    unsigned int RegisterPatches(std::vector<BilinearPatchPrimitive> patches,
                                 double box_bump = 0.0);

    /// Build/rebuild the BVH. Must be called after RegisterPatches and before
    /// any queries.
    void CommitScene();

    /// First hit along org + t * dir for t in [tnear, tfar].
    RayHitRecord Intersect(const double origin[3],
                           const double direction[3],
                           double tnear = 0.0,
                           double tfar = std::numeric_limits<double>::infinity()) const;

    /// All hits along org + t * dir for t in [tnear, tfar], in increasing-t
    /// order. After each hit the search resumes just past that t so a ray can
    /// pierce successive patches (e.g. front then back of a torus). Stops after
    /// `max_hits` or when no further hit is found.
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
    const BilinearPatchPrimitive *GetPatch(unsigned int geom_id, unsigned int prim_id) const;

    /// Total number of patches across all registered geometries.
    std::size_t PatchCount() const;

private:
    struct GeometrySlot
    {
        std::vector<BilinearPatchPrimitive> patches;
        BilinearPatchGeometryData data;
    };

    RTCDevice device_ = nullptr;
    RTCScene scene_ = nullptr;
    bool committed_ = false;
    // Heap slots so the buffers Embree points at survive map rehashing.
    std::unordered_map<unsigned int, std::unique_ptr<GeometrySlot>> geometry_slots_;
};

} // namespace mfem_raytracing

#endif
