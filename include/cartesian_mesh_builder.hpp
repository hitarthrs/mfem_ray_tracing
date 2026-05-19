#ifndef CARTESIAN_MESH_BUILDER_HPP
#define CARTESIAN_MESH_BUILDER_HPP

#include "cartesian_mesh_spec.hpp"
#include "mfem.hpp"

/// Buildes an mfem:Mesh from a CartestianMeshSpec

class CartesianMeshBuilder
{
public:
    explicit CartesianMeshBuilder(const CartesianMeshSpec &spec);

    /// Create and return a new MFEM mesh 
    mfem::Mesh BuildCartesianMesh() const;

    /// Parse CLI
    static CartesianMeshSpec ParseCLI(int argc, char *argv[]);

private:
     CartesianMeshSpec spec_;

     void Validate() const;
};

#endif