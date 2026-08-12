#ifndef MFEM_RAYTRACING_BILINEAR_INTERSECT_HPP
#define MFEM_RAYTRACING_BILINEAR_INTERSECT_HPP

#include "mfem_raytracing/embree/bilinear_patch_geometry.hpp"
#include "mfem_raytracing/embree/embree_interface.hpp"

namespace mfem_raytracing
{

struct RayQueryDiagnostics;

/// Reason bits observed when a patch callback cannot report a hit. Multiple
/// candidate roots can fail for different reasons, so these form a bit mask.
enum BilinearRejectReason : unsigned int
{
    BilinearRejectNone = 0u,
    BilinearRejectInvalidRay = 1u << 0,
    BilinearRejectNoRoot = 1u << 1,
    BilinearRejectDenominator = 1u << 2,
    BilinearRejectResidual = 1u << 3,
    BilinearRejectDomain = 1u << 4,
    BilinearRejectWeight = 1u << 5,
    BilinearRejectTRange = 1u << 6,
};

/// Result from directly evaluating one patch. This has the same float input
/// representation as the Embree callback and is intended for diagnostics.
struct BilinearPatchRayHit
{
    bool hit = false;
    float t = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float Ng[3] = {0.0f, 0.0f, 0.0f};
    unsigned int reject_reasons = BilinearRejectNone;
};

/// Set the thread-local diagnostics sink used by the user-geometry callback.
/// Returns the previously active sink so callers can restore nested queries.
RayQueryDiagnostics *SetActiveRayQueryDiagnostics(RayQueryDiagnostics *diagnostics);
RayQueryDiagnostics *GetActiveRayQueryDiagnostics();

BilinearPatchRayHit IntersectBilinearPatchDirect(const BilinearPatchPrimitive &patch,
                                                  const float origin[3],
                                                  const float direction[3],
                                                  float tnear,
                                                  float tfar);

void BilinearPatchBoundsFunc(const RTCBoundsFunctionArguments *args);
void BilinearPatchIntersectionFunc(const RTCIntersectFunctionNArguments *args);
void BilinearPatchOccludedFunc(const RTCOccludedFunctionNArguments *args);

} // namespace mfem_raytracing

#endif
