#ifndef CARTESIAN_MESH_SPEC_HPP
#define CARTESIAN_MESH_SPEC_HPP

#include <vector>

/// Input for building a uniform Cartesian mesh in 1D, 2D, or 3D.
struct CartesianMeshSpec
{
   int dim = 0; 
   std::vector<int> n;
   std::vector<double> s;
   std::vector<double> origin;

   /// set dimension, length, and origin vector

   void SetDimension(int dimension)
   {
      dim = dimension;
      n.assign(dim, 1);
      s.assign(dim, 1.0);
      origin.assign(dim, 0.0);
   }
};

#endif