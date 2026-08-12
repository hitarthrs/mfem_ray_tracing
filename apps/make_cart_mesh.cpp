#include "mfem_raytracing/mesh/cartesian_mesh_builder.hpp"
#include "mfem_raytracing/mesh/cartesian_mesh_naming.hpp"

#include <iostream>

int main(int argc, char *argv[])
{
   try
   {
      const CartesianMeshSpec spec = CartesianMeshBuilder::ParseCLI(argc, argv);
      const CartesianMeshBuilder builder(spec);

      std::cout << "Building " << spec.dim << "D mesh: " << spec.n[0];
      for (int i = 1; i < spec.dim; ++i)
      {
         std::cout << " x " << spec.n[i];
      }
      std::cout << std::endl;

      mfem::Mesh mesh = builder.BuildCartesianMesh();

      const std::string mesh_name = GenerateCartesianMeshName(spec);
      mesh.Save(mesh_name);
      std::cout << "Mesh saved to " << mesh_name << std::endl;

      std::cout << "Mesh dimension: " << mesh.Dimension() << std::endl;
      std::cout << "Elements:       " << mesh.GetNE() << std::endl;
      std::cout << "Vertices:       " << mesh.GetNV() << std::endl;
      

      return 0;
   }
   catch (const std::exception &e)
   {
      std::cerr << "Error: " << e.what() << std::endl;
      return 1;
   }
}