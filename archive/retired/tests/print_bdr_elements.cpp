#include "mfem.hpp"

#include <iostream>
#include <string>

namespace
{

void PrintKnotVector(std::ostream &os, const char *label, const mfem::KnotVector &kv)
{
    os << "    " << label << ": order=" << kv.GetOrder()
       << ", ncp=" << kv.GetNCP() << ", ne=" << kv.GetNE()
       << ", knots=[";
    for (int i = 0; i < kv.Size(); ++i)
    {
        if (i > 0)
        {
            os << ", ";
        }
        os << kv[i];
    }
    os << "]\n";
}

void PrintControlPoints(std::ostream &os,
                        const mfem::Array<int> &dofs,
                        const mfem::Vector &nodes,
                        const mfem::Vector &weights,
                        int vdim)
{
    os << "    control points (" << dofs.Size() << "):\n";
    for (int i = 0; i < dofs.Size(); ++i)
    {
        const int cp = dofs[i];
        os << "      [" << i << "] cp " << cp << "  w=" << weights(cp) << "  xyz=(";
        for (int d = 0; d < vdim; ++d)
        {
            if (d > 0)
            {
                os << ", ";
            }
            os << nodes(cp * vdim + d);
        }
        os << ")\n";
    }
}

}  // namespace

int main(int argc, char *argv[])
{
    const char *mesh_path = (argc > 1) ? argv[1] : "meshes/iga/pipe-nurbs.mesh";

    mfem::Mesh mesh(mesh_path, 1, 1);

    if (mesh.NURBSext == nullptr)
    {
        std::cerr << "Not a NURBS mesh: " << mesh_path << "\n";
        return 1;
    }
    if (mesh.GetNodes() == nullptr)
    {
        std::cerr << "Mesh has no nodal GridFunction: " << mesh_path << "\n";
        return 1;
    }

    mfem::Vector nodes;
    mesh.GetNodes(nodes);
    const int vdim = mesh.SpaceDimension();
    const mfem::Vector &weights = mesh.NURBSext->GetWeights();

    const mfem::FiniteElementSpace &fes = *mesh.GetNodes()->FESpace();
    mfem::Table *bel_dof = mesh.NURBSext->GetBdrElementDofTable();
    if (bel_dof == nullptr)
    {
        std::cerr << "No boundary element DOF table.\n";
        return 1;
    }

    std::cout << "Mesh: " << mesh_path << "\n";
    std::cout << "Boundary elements: " << mesh.GetNBE() << "\n\n";

    for (int be = 0; be < mesh.GetNBE(); ++be)
    {
        int adj_el = -1;
        int face_info = 0;
        mesh.GetBdrElementAdjacentElement(be, adj_el, face_info);

        const int vol_patch =
            (adj_el >= 0) ? mesh.NURBSext->GetElementPatch(adj_el) : -1;

        const mfem::FiniteElement *fe = fes.GetBE(be);
        const auto *nfe = dynamic_cast<const mfem::NURBSFiniteElement *>(fe);

        std::cout << "boundary element " << be << "\n";
        std::cout << "  attribute=" << mesh.GetBdrAttribute(be) << "\n";
        std::cout << "  adjacent volume element=" << adj_el << "\n";
        std::cout << "  volume patch=" << vol_patch << "\n";
        std::cout << "  face_info=" << face_info << "\n";

        if (nfe == nullptr)
        {
            std::cout << "  (not a NURBS boundary element)\n\n";
            continue;
        }

        std::cout << "  NURBS boundary patch=" << nfe->GetPatch() << "\n";
        std::cout << "  NURBS boundary element index=" << nfe->GetElement() << "\n";

        if (const int *ijk = nfe->GetIJK())
        {
            const int nkv = nfe->KnotVectors().Size();
            std::cout << "  IJK=(" << ijk[0];
            for (int d = 1; d < nkv; ++d)
            {
                std::cout << ", " << ijk[d];
            }
            std::cout << ")\n";
        }

        const mfem::Array<const mfem::KnotVector *> &kv = nfe->KnotVectors();
        for (int d = 0; d < kv.Size(); ++d)
        {
            PrintKnotVector(std::cout,
                            (d == 0) ? "kv_u" : (d == 1 ? "kv_v" : "kv_w"),
                            *kv[d]);
        }

        mfem::Array<int> dofs;
        bel_dof->GetRow(be, dofs);
        PrintControlPoints(std::cout, dofs, nodes, weights, vdim);
        std::cout << "\n";
    }

    return 0;
}
