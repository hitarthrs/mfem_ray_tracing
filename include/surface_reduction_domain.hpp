#ifndef SURFACE_REDUCTION_DOMAIN_HPP
#define SURFACE_REDUCTION_DOMAIN_HPP

#include "surface_reduction_types.hpp"

#include <utility>

namespace mfem_raytracing
{

void ValidateSurfaceData(const SurfaceData &surface, const char *label);

std::pair<SurfaceData, SurfaceData> SplitSurfaceU(const SurfaceData &surface, double u);

std::pair<SurfaceData, SurfaceData> SplitSurfaceV(const SurfaceData &surface, double v);

double LocalSurfaceParamFromGlobal(double t_global,
                                   const std::pair<double, double> &global_domain,
                                   const std::pair<double, double> &local_domain);

} // namespace mfem_raytracing

#endif
