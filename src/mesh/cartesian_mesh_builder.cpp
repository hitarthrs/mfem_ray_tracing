#include "mfem_raytracing/mesh/cartesian_mesh_builder.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

CartesianMeshBuilder::CartesianMeshBuilder(const CartesianMeshSpec &spec)
   : spec_(spec)
{
   Validate();
}

void CartesianMeshBuilder::Validate() const
{
   if (spec_.dim < 1 || spec_.dim > 3)
   {
      throw std::invalid_argument("Dimension must be 1, 2, or 3");
   }

   // 1. Vector sezes must match dimension
   const std::size_t d = static_cast<std::size_t>(spec_.dim);
   if (spec_.n.size() != d || spec_.s.size() != d || spec_.origin.size() != d)
   {
      throw std::invalid_argument("Vector sizes must match dimension");
   }
   // 2. every cell count >= 1
   for (int count : spec_.n)
   {
      if (count < 1)
      {
         throw std::invalid_argument("each entry in n must be >= 1");
      }
   }
   // 3. every box length > 0
   for (double length : spec_.s)
   {
      if (length <= 0.0)
      {
         throw std::invalid_argument("each entry in s must be > 0");
      }
   }

}

mfem::Mesh CartesianMeshBuilder::BuildCartesianMesh() const
{
    using namespace mfem;

    const bool generate_edges = true;

    switch (spec_.dim)
    {
        case 1:
            return Mesh::MakeCartesian1D(spec_.n[0], spec_.s[0]);
        case 2:
            return Mesh::MakeCartesian2D(spec_.n[0], spec_.n[1], Element::QUADRILATERAL, generate_edges, spec_.s[0], spec_.s[1]);
        case 3:
            return Mesh::MakeCartesian3D(spec_.n[0], spec_.n[1], spec_.n[2], Element::HEXAHEDRON, spec_.s[0], spec_.s[1], spec_.s[2]);
        default:
            MFEM_ABORT("Invalid dimension");
    }
}

CartesianMeshSpec CartesianMeshBuilder::ParseCLI(int argc, char *argv[])
{
    if (argc < 2)
   {
      throw std::invalid_argument(
        "usage: <program> <dim> <n...> <s...> [origin...]\n"
         "  args after dim must be 2 or 3 groups of <dim> numbers\n"
         "  1D: program 1  20  1.0\n"
         "  2D: program 2  10 20  1 2\n"
         "  2D+origin: program 2  10 20  1 2  0.5 0\n"
         "  3D: program 3  4 4 4  1 1 1");
   }

   const int dim = std::stoi(argv[1]);

   if (dim < 1 || dim > 3)
   {
      throw std::invalid_argument("dim must be 1, 2, or 3");
   }

   const int tail = argc - 2; // arguments after dim

   if (tail % dim != 0)
   {
    throw std::invalid_argument(
        "Number of arguments after dimensions must be a multiple of dimensions."
    );
   }

   const int num_groups = tail / dim;

   if (num_groups > 3)
   {
    throw std::invalid_argument(
        "Too many groups of arguments after dimensions. The three groups in order are number of elements, length, and origin."
    );
   }

   CartesianMeshSpec spec;
   spec.SetDimension(dim);

   int idx = 2;

   for (int i = 0; i < dim; i++)
    {
        spec.n[i] = std::stoi(argv[idx]);
        spec.s[i] = std::stod(argv[dim+idx]);
        
        if (num_groups == 3)
        {
        spec.origin[i] = std::stod(argv[2*dim+idx]);
        }
        idx++;
    }

   return spec;

}
