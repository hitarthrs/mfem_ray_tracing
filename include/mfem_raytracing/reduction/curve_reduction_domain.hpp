#ifndef CURVE_REDUCTION_DOMAIN_HPP
#define CURVE_REDUCTION_DOMAIN_HPP

// Curve validation, parameter remapping, knot insertion, and subcurve extraction.

#include "mfem_raytracing/reduction/curve_reduction_types.hpp"

#include <utility>
#include <vector>

namespace mfem_raytracing
{

std::pair<double, double> CurveDomain(const std::vector<double> &knotvector, int degree);

void ValidateCurveData(const CurveData &curve, const char *label);

// Map between a sub-piece's local domain and the root/global parameter line.
double LocalParamFromGlobal(double u_global,
                            const std::pair<double, double> &piece_global,
                            const std::pair<double, double> &local_domain);

double GlobalParamFromLocal(double u_local,
                            const std::pair<double, double> &piece_global,
                            const std::pair<double, double> &local_domain);

std::pair<double, double> MapIntervalToGlobal(
    const std::pair<double, double> &interval,
    const std::pair<double, double> &local_domain,
    const std::pair<double, double> &global_domain);

CurveData ReparameterizeCurve(const CurveData &curve,
                              const std::pair<double, double> &new_domain);

std::pair<CurveData, CurveData> SplitCurve(const CurveData &curve, double u);

// Extract [u0, u1] in global coords; result is reparameterized to [0, 1].
CurveData ExtractSubcurveGlobal(const CurveData &curve,
                                const std::pair<double, double> &piece_global,
                                double u0_global,
                                double u1_global);

} // namespace mfem_raytracing

#endif
