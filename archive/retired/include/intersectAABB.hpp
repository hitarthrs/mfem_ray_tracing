#ifndef INTERSECT_AABB_HPP
#define INTERSECT_AABB_HPP

#include "ray.hpp"
#include "mfem.hpp"

bool IntersectAABB(const Ray &ray, mfem::Mesh &mesh, double &t_entry, double &t_exit);

#endif