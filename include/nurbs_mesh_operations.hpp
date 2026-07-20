#ifndef NURBS_MESH_OPERATIONS_HPP
#define NURBS_MESH_OPERATIONS_HPP

#include "mfem.hpp"

#include <iosfwd>
#include <vector>

struct ElementBoundingBox
{
    mfem::Vector min;
    mfem::Vector max;
};

// Check if a mesh is a NURBS mesh
bool IsNURBSMesh(const mfem::Mesh &mesh);

// Axis-aligned bounding boxes for each mesh element (refined geometry sampling).
std::vector<ElementBoundingBox> GetElementBoundingBoxes(mfem::Mesh &mesh, int ref = 2);

// Print element index and [min, max] for each bounding box.
void PrintElementBoundingBoxes(const std::vector<ElementBoundingBox> &bboxes,
                               std::ostream &os,
                               int precision = 6);

#endif
