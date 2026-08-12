#ifndef FACE_HIT_INFORMATION_HPP
#define FACE_HIT_INFORMATION_HPP

#include "mfem.hpp"
#include "ray.hpp"

// Lightweight struct for intersection resuts

struct FaceHitInformation
{
    bool hit = false; // Only true if ray hits face
    double t_intersection = -1.0; // Ray intersection travel distance
    double u = -1.0; // Parametric coordinate along the face (first parametric direction)
    double v = -1.0; // Parametric coordinate along the face (second parametric direction)
    mfem::Vector local_coords; // Local coordinates of the intersection point (x,y,z)
    mfem::Vector normal; // Normal vector at the intersection point
};

#endif