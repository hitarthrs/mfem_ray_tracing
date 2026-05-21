#ifndef BILINEAR_INTERSECTION_HPP
#define BILINEAR_INTERSECTION_HPP

#include "mfem.hpp"
#include "ray.hpp"
#include "face_hit_information.hpp"

/**
 * @brief Computes direct analytical ray intersection with an element face (2nd-order bilinear patch).
 * * @param ray The physical unit-direction tracking ray.
 * @param mesh The master MFEM mesh object containing global geometry data storage.
 * @param face_index Global integer index of the specific boundary face being evaluated.
 * @return FaceHitInformation Structured object containing precise path lengths, coordinates, and normals.
 */

FaceHitInformation BilinearIntersection(const Ray &ray, const mfem::Mesh &mesh, int face_index);

#endif