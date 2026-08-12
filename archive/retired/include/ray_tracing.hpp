#ifndef RAY_TRACING_HPP
#define RAY_TRACING_HPP

#include "ray.hpp"
#include "mfem.hpp"
#include "intersectAABB.hpp"
#include "ray_cell_crossings.hpp"

#include <vector>

/// Sample the ray every @a dt along [t_entry, t_exit] (from IntersectAABB),
/// batched FindPoints per sample, then build one segment per visited element.
bool TraceFindPoints(const Ray &ray,
                     mfem::Mesh &mesh,
                     double dt,
                     std::vector<RayCellCrossings> &crossings);

#endif