#include "cartesian_mesh_naming.hpp"

#include <sstream>
#include <string>

namespace
{
    template <typename T>
    std::string JoinWithX(const std::vector<T> &values, int dim)
    {
        std::ostringstream oss;
        for (int i = 0; i < dim; ++i)
        {
            if (i > 0) oss << "x";
            oss << values[i];
        }
        return oss.str();
    }

    bool OriginIsZero(const CartesianMeshSpec &spec)
    {
        for (double o : spec.origin)
        {
            if (o != 0.0)
            {
                return false;
            }
        }
        return true;
    }
}

std::string GenerateCartesianMeshName(const CartesianMeshSpec &spec)
{
    std::ostringstream mesh_filename;
    mesh_filename << "cartesian_mesh_"
    << spec.dim
    << "D_n"
    << JoinWithX(spec.n, spec.dim)
    << "_s"
    << JoinWithX(spec.s, spec.dim);

    if (!OriginIsZero(spec))
    {
        mesh_filename << "_o" << JoinWithX(spec.origin, spec.dim);
    }

    mesh_filename << ".mesh";

    return mesh_filename.str();
}