#include "bdr_patch_extractor.hpp"

#include "mfem.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

void PrintMatrix(std::ostream &os, const char *prefix, const mfem::DenseMatrix &M)
{
    os << prefix << "(" << M.Height() << " x " << M.Width() << ")\n";
    for (int i = 0; i < M.Height(); ++i)
    {
        os << prefix;
        for (int j = 0; j < M.Width(); ++j)
        {
            if (j > 0)
            {
                os << "  ";
            }
            os << M(i, j);
        }
        os << "\n";
    }
}

void PrintSparseMatrix(std::ostream &os, const char *prefix, const mfem::SparseMatrix &M)
{
    os << prefix << "(" << M.Height() << " x " << M.Width() << ")\n";
    for (int i = 0; i < M.Height(); ++i)
    {
        os << prefix;
        for (int j = 0; j < M.Width(); ++j)
        {
            if (j > 0)
            {
                os << "  ";
            }
            os << M(i, j);
        }
        os << "\n";
    }
}

void PrintKnots(std::ostream &os, const char *name, const BdrPatchParametric1D &param)
{
    os << name << "  degree=" << param.degree << "  ne=" << param.ne
       << "  extractors=" << param.extractors.size() << "\n  knots=[";
    for (std::size_t i = 0; i < param.knots.size(); ++i)
    {
        if (i > 0)
        {
            os << ", ";
        }
        os << param.knots[i];
    }
    os << "]\n";
    for (std::size_t e = 0; e < param.extractors.size(); ++e)
    {
        os << "  C" << e << ":\n";
        PrintMatrix(os, "    ", param.extractors[e].matrix);
    }
}

}  // namespace

int main(int argc, char *argv[])
{
    const char *mesh_path = (argc > 1) ? argv[1] : "meshes/iga/pipe-nurbs.mesh";
    const int bdr_patch = (argc > 2) ? std::atoi(argv[2]) : 14;

    mfem::Mesh mesh(mesh_path, 1, 1);

    try
    {
        const BdrPatchBlockData data = ExtractBdrPatchBlocks(mesh, bdr_patch);

        std::cout << "Mesh: " << mesh_path << "\n";
        std::cout << "NURBS boundary patch: " << bdr_patch << "\n";
        std::cout << "nu=" << data.nu << " nv=" << data.nv
                  << " ne_u=" << data.ne_u << " ne_v=" << data.ne_v
                  << "  blocks=" << data.elements.size() << "\n\n";

        std::cout << "--- parametric direction u ---\n";
        PrintKnots(std::cout, "param_u", data.param_u);
        std::cout << "\n--- parametric direction v ---\n";
        PrintKnots(std::cout, "param_v", data.param_v);
        std::cout << "\n";

        for (size_t e = 0; e < data.elements.size(); ++e)
        {
            std::cout << "=== block " << e << " ===\n";
            std::cout << "  span_u=" << data.elements[e].span_u
                      << "  span_v=" << data.elements[e].span_v
                      << "  (C_e = C_v[span_v] ⊗ C_u[span_u])\n";
            PrintMatrix(std::cout, "  C_e ", data.elements[e].C_e);
            std::cout << "\n";
            PrintMatrix(std::cout, "  P ", data.elements[e].P);
            std::cout << "\n";
            PrintMatrix(std::cout, "  Q_e ", data.elements[e].Q_e);
            std::cout << "\n";
            PrintSparseMatrix(std::cout, "  W ", data.elements[e].W);
            std::cout << "\n";
            PrintSparseMatrix(std::cout, "  W_b ", data.elements[e].W_b);
            std::cout << "\n";
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
