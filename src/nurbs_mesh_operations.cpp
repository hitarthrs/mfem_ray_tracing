#include "nurbs_mesh_operations.hpp"

#include <iomanip>
#include <limits>

namespace
{

ElementBoundingBox ComputeElementBoundingBox(mfem::Mesh &mesh, int elem, int ref)
{
    // ref coords -> physical coords for this element
    mfem::ElementTransformation *T = mesh.GetElementTransformation(elem);
    // sample points on the reference hex/quad (ref = sampling density)
    mfem::RefinedGeometry *RefG = mfem::GlobGeometryRefiner.Refine(mesh.GetElementBaseGeometry(elem), ref);

    // map samples to (x,y,z); columns of pointmat are physical points
    mfem::DenseMatrix pointmat;
    T->Transform(RefG->RefPts, pointmat);

    ElementBoundingBox bbox;
    const int sdim = pointmat.Height();
    const int npts = pointmat.Width();
    bbox.min.SetSize(sdim);
    bbox.max.SetSize(sdim);

    // start with an empty box
    for (int d = 0; d < sdim; ++d)
    {
        bbox.min(d) = std::numeric_limits<double>::infinity();
        bbox.max(d) = -std::numeric_limits<double>::infinity();
    }


    // grow the box to contain every sample
    for (int j = 0; j < npts; ++j)
    {
        for (int d = 0; d < sdim; ++d)
        {
            const double x = pointmat(d, j);
            if (x < bbox.min(d))
            {
                bbox.min(d) = x;
            }
            if (x > bbox.max(d))
            {
                bbox.max(d) = x;
            }
        }
    }

    return bbox;
}

}  // namespace

bool IsNURBSMesh(const mfem::Mesh &mesh)
{
    // NURBS meshes carry extra knot/control-point data here
    return mesh.NURBSext != nullptr;
}

std::vector<ElementBoundingBox> GetElementBoundingBoxes(mfem::Mesh &mesh, int ref)
{
    std::vector<ElementBoundingBox> bboxes;
    bboxes.reserve(mesh.GetNE());

    // one AABB per knot-span element
    for (int i = 0; i < mesh.GetNE(); ++i)
    {
        bboxes.push_back(ComputeElementBoundingBox(mesh, i, ref));
    }

    return bboxes;
}

void PrintElementBoundingBoxes(const std::vector<ElementBoundingBox> &bboxes,
                               std::ostream &os,
                               int precision)
{
    os << std::setprecision(precision);

    for (size_t i = 0; i < bboxes.size(); ++i)
    {
        const auto &bbox = bboxes[i];
        os << "  element " << i << ": min (";
        // print min corner
        for (int d = 0; d < bbox.min.Size(); ++d)
        {
            os << bbox.min(d);
            if (d + 1 < bbox.min.Size())
            {
                os << ", ";
            }
        }
        os << ")  max (";
        // print max corner
        for (int d = 0; d < bbox.max.Size(); ++d)
        {
            os << bbox.max(d);
            if (d + 1 < bbox.max.Size())
            {
                os << ", ";
            }
        }
        os << ")\n";
    }
}
