#ifndef MFEM_RAYTRACING_EMBREE3_HPP
#define MFEM_RAYTRACING_EMBREE3_HPP

#include "embree3/rtcore.h"

#ifndef MFEM_RAYTRACING_EMBREE3_WRAPPERS
#define MFEM_RAYTRACING_EMBREE3_WRAPPERS

// Match Embree 4 single-argument intersect/occluded signatures on Embree 3.
inline void rtcIntersect1(RTCScene scene, RTCRayHit *rayhit)
{
  RTCIntersectContext context;
  rtcInitIntersectContext(&context);
  rtcIntersect1(scene, &context, rayhit);
}

inline void rtcOccluded1(RTCScene scene, RTCRay *ray)
{
  RTCIntersectContext context;
  rtcInitIntersectContext(&context);
  rtcOccluded1(scene, &context, ray);
}

#endif

#endif
