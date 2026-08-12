#ifndef CARTESIAN_MESH_NAMING_HPP
#define CARTESIAN_MESH_NAMING_HPP

#include "mfem_raytracing/mesh/cartesian_mesh_spec.hpp"
#include <string>

std::string GenerateCartesianMeshName(const CartesianMeshSpec &spec);

#endif