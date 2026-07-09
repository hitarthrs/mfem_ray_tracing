#ifndef MFEM_RAYTRACING_BILINEAR_INTERSECT_HPP
#define MFEM_RAYTRACING_BILINEAR_INTERSECT_HPP

#include "embree/embree_interface.hpp"

namespace mfem_raytracing
{

void BilinearPatchBoundsFunc(const RTCBoundsFunctionArguments *args);
void BilinearPatchIntersectionFunc(const RTCIntersectFunctionNArguments *args);
void BilinearPatchOccludedFunc(const RTCOccludedFunctionNArguments *args);

} // namespace mfem_raytracing

#endif
