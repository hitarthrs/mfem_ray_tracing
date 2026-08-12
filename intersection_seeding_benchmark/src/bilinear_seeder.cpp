#include "bilinear_seeder.hpp"

#include "mfem_raytracing/reduction/bilinear_leaf_extraction.hpp"
#include "mfem_raytracing/embree/bilinear_patch_geometry.hpp"

#include <algorithm>
#include <stdexcept>

namespace seeding_benchmark
{

namespace
{

using mfem_raytracing::BilinearCorner;

// geomdl [i][j] = [u][v] control net -> BilinearCorner layout, matching
// leaf_patch_loader.cpp CornerFromUV.
int CornerFromUV(int i, int j)
{
    if (i == 0)
    {
        return static_cast<int>(j == 0 ? BilinearCorner::P00 : BilinearCorner::P01);
    }
    return static_cast<int>(j == 0 ? BilinearCorner::P10 : BilinearCorner::P11);
}

mfem_raytracing::BilinearPatchPrimitive PrimitiveFromLeaf(
    const mfem_raytracing::SurfaceData &net)
{
    mfem_raytracing::BilinearPatchPrimitive prim;
    const bool rational = net.IsRational();
    prim.rational = rational;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            const int corner = CornerFromUV(i, j);
            const auto &P = net.control_points[static_cast<std::size_t>(i)]
                                              [static_cast<std::size_t>(j)];
            prim.control_points[corner][0] = P[0];
            prim.control_points[corner][1] = P[1];
            prim.control_points[corner][2] = P[2];
            prim.weights[corner] =
                rational
                    ? net.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]
                    : 1.0;
        }
    }
    return prim;
}

}  // namespace

BilinearSeeder::BilinearSeeder(const mfem_raytracing::SurfaceData &surface,
                               double max_error,
                               double box_bump)
    : max_error_(max_error)
{
    mfem_raytracing::BilinearLeafReductionOptions options;
    options.conforming = true;  // watertight tensor-grid driver
    options.coalesce = true;

    const int n_steps = std::max(surface.degree_u - 1, surface.degree_v - 1);
    if (n_steps < 1)
    {
        throw std::invalid_argument("BilinearSeeder: surface already degree (1,1)");
    }

    mfem_raytracing::BilinearLeafCollection collection =
        mfem_raytracing::ReduceSurfaceToBilinearLeaves(surface, n_steps, max_error,
                                                       options);

    std::vector<mfem_raytracing::BilinearPatchPrimitive> patches;
    patches.reserve(collection.leaves.size());
    leaf_u_domain_.reserve(collection.leaves.size());
    leaf_v_domain_.reserve(collection.leaves.size());

    for (const auto &leaf : collection.leaves)
    {
        patches.push_back(PrimitiveFromLeaf(leaf.surface));
        leaf_u_domain_.push_back(leaf.u_domain_global);
        leaf_v_domain_.push_back(leaf.v_domain_global);
    }

    tracer_ = std::make_unique<mfem_raytracing::EmbreeRayTracer>();
    geom_id_ = tracer_->RegisterPatches(std::move(patches), box_bump);
    tracer_->CommitScene();
}

SeedResult BilinearSeeder::Seed(const Ray &ray) const
{
    SeedResult out;
    const mfem_raytracing::RayHitRecord hit =
        tracer_->Intersect(ray.origin.data(), ray.dir.data(), ray.tmin, ray.tmax);
    if (!hit.hit || hit.geom_id != geom_id_)
    {
        return out;
    }

    const std::size_t prim = hit.prim_id;
    if (prim >= leaf_u_domain_.size())
    {
        return out;
    }

    const auto &ud = leaf_u_domain_[prim];
    const auto &vd = leaf_v_domain_[prim];
    out.hit = true;
    out.u = ud.first + hit.u * (ud.second - ud.first);
    out.v = vd.first + hit.v * (vd.second - vd.first);
    return out;
}

}  // namespace seeding_benchmark
