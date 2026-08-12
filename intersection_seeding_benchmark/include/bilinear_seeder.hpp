#ifndef SEEDING_BENCHMARK_BILINEAR_SEEDER_HPP
#define SEEDING_BENCHMARK_BILINEAR_SEEDER_HPP

// Per-patch bilinear-leaf seed provider.
//
// For one NURBS boundary patch, reduce it to a watertight set of degree-(1, 1)
// bilinear leaf patches at a given error tolerance, load those leaves into an
// Embree BVH, and answer ray queries with the (u, v) initial guess implied by
// the first-hit leaf -- mapped from the leaf's local [0, 1]^2 back to the
// patch's global parameter domain via the leaf's u/v_domain_global.
//
// This is the "bilinear-seeded" strategy's seed source; the BVH build cost is
// one-time per (patch, tolerance) and the per-ray lookup is timed by the caller.

#include "mfem_raytracing/embree/raytracer.hpp"
#include "newton_intersect.hpp"
#include "mfem_raytracing/reduction/surface_reduction_types.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace seeding_benchmark
{

struct SeedResult
{
    bool hit = false;   ///< a leaf was intersected
    double u = 0.0;     ///< seed in the patch's global u domain
    double v = 0.0;     ///< seed in the patch's global v domain
};

/// One patch's bilinear approximation + BVH, reusable across many ray queries.
class BilinearSeeder
{
public:
    /// Build the leaf approximation of `surface` at global error budget
    /// `max_error` and commit its Embree BVH. `box_bump` pads every leaf AABB by
    /// a constant amount so grazing rays that just miss a tight box still return
    /// a seed (0 = exact AABBs).
    BilinearSeeder(const mfem_raytracing::SurfaceData &surface,
                   double max_error,
                   double box_bump = 0.0);

    /// First-hit leaf seed for the ray, or {hit=false} on a miss.
    SeedResult Seed(const Ray &ray) const;

    std::size_t LeafCount() const { return leaf_u_domain_.size(); }
    double MaxError() const { return max_error_; }

private:
    double max_error_ = 0.0;
    std::unique_ptr<mfem_raytracing::EmbreeRayTracer> tracer_;
    unsigned int geom_id_ = 0;
    // Parallel to Embree primID: where each leaf's local [0,1]^2 maps globally.
    std::vector<std::pair<double, double>> leaf_u_domain_;
    std::vector<std::pair<double, double>> leaf_v_domain_;
};

}  // namespace seeding_benchmark

#endif
