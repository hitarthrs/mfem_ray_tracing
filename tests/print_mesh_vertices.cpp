#include "mfem.hpp"

#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
    const char *mesh_path = (argc > 1) ? argv[1] : "meshes/iga/pipe-nurbs.mesh";

    mfem::Mesh mesh(mesh_path, 1, 1);

    std::cout << "Mesh: " << mesh_path << "\n";
    std::cout << "Dimension: " << mesh.Dimension()
              << ", SpaceDimension: " << mesh.SpaceDimension() << "\n";
    std::cout << "GetNV() (vertices): " << mesh.GetNV() << "\n";
    std::cout << "GetNE() (elements): " << mesh.GetNE() << "\n";
    std::cout << "NURBS: " << (mesh.NURBSext != nullptr ? "yes" : "no") << "\n\n";

    std::cout << "--- Mesh vertices (GetVertex) ---\n";
    for (int v = 0; v < mesh.GetNV(); ++v)
    {
        const double *coords = mesh.GetVertex(v);
        std::cout << "  vertex " << v << ": (";
        for (int d = 0; d < mesh.SpaceDimension(); ++d)
        {
            std::cout << coords[d];
            if (d + 1 < mesh.SpaceDimension())
            {
                std::cout << ", ";
            }
        }
        std::cout << ")\n";
    }

    if (mesh.GetNodes())
    {
        mfem::Vector node_coord;
        mesh.GetNodes(node_coord);
        const int vdim = mesh.SpaceDimension();
        const int ncp = node_coord.Size() / vdim;
        std::cout << "\n--- NURBS control points (GetNodes) ---\n";
        std::cout << "  count=" << ncp << "\n";
        for (int i = 0; i < ncp; ++i)
        {
            std::cout << "  cp " << i << ": (";
            for (int d = 0; d < vdim; ++d)
            {
                std::cout << node_coord(i * vdim + d);
                if (d + 1 < vdim)
                {
                    std::cout << ", ";
                }
            }
            std::cout << ")\n";
        }
    }

    return 0;
}
