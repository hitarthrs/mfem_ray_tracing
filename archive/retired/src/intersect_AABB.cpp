#include "intersectAABB.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

// Ray–axis-aligned bounding box test using the slab method.
// The mesh global bounding box is obtained from MFEM; direction is assumed normalized
// (as enforced by Ray), so t values are distances along the ray.

bool IntersectAABB(const Ray &ray, mfem::Mesh &mesh, double &t_entry, double &t_exit)
{
    const mfem::Vector &origin = ray.GetOrigin();
    const mfem::Vector &direction = ray.GetDirection();

    // Global mesh AABB in physical (space) coordinates.
    mfem::Vector bbox_min, bbox_max;
    mesh.GetBoundingBox(bbox_min, bbox_max);

    const int dim = bbox_min.Size();
    constexpr double eps = 1e-14;  // treat |d_i| < eps as zero (avoids huge t from float noise)

    // Interval along the ray that lies inside the box. Start unbounded; each axis tightens it.
    double t_entry_box = -std::numeric_limits<double>::infinity();
    double t_exit_box = std::numeric_limits<double>::infinity();

    for (int i = 0; i < dim; ++i)
    {
        const double di = direction[i];
        const double oi = origin[i];

        // Ray parallel to the x_i slabs: o_i is fixed in t. No division by d_i.
        if (std::abs(di) < eps)
        {
            if (oi < bbox_min[i] || oi > bbox_max[i])
            {
                return false;  // outside the box on this axis and never crosses it
            }
            continue;  // inside slab on this axis; other axes still constrain t
        }

        // Parametric t where the ray hits the two planes x_i = bbox_min[i] and x_i = bbox_max[i].
        const double t1 = (bbox_min[i] - oi) / di;
        const double t2 = (bbox_max[i] - oi) / di;

        // Intersection with the box requires being inside all slabs:
        // t_entry_box = max(enter_i),  t_exit_box = min(exit_i).
        t_entry_box = std::max(t_entry_box, std::min(t1, t2));
        t_exit_box = std::min(t_exit_box, std::max(t1, t2));
    }

    if (t_entry_box > t_exit_box)
    {
        return false;  // intervals on each axis do not overlap
    }

    // Clip to the ray's active segment [GetTMin(), GetTMax()].
    t_entry = std::max(t_entry_box, ray.GetTMin());
    t_exit = std::min(t_exit_box, ray.GetTMax());

    return t_entry <= t_exit;
}
