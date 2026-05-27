#include "cartesian_mesh_builder.hpp"
#include "cartesian_mesh_naming.hpp"
#include "cartesian_mesh_spec.hpp"
#include "test_helpers.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace
{

CartesianMeshSpec MakeSpec2D()
{
    CartesianMeshSpec spec;
    spec.SetDimension(2);
    spec.n = {10, 20};
    spec.s = {2.0, 1.0};
    spec.origin = {0.0, 0.0};
    return spec;
}

// SetDimension resizes n, s, and origin vectors and applies defaults.
void TestSetDimension()
{
    CartesianMeshSpec spec;
    spec.SetDimension(3);

    CHECK(spec.dim == 3);
    CHECK(spec.n.size() == 3u);
    CHECK(spec.s.size() == 3u);
    CHECK(spec.origin.size() == 3u);
    CHECK(spec.n[0] == 1);
    CHECK(spec.s[1] == 1.0);
    CHECK(spec.origin[2] == 0.0);
}

// Mesh filename omits origin when all components are zero.
void TestMeshNameWithoutOrigin()
{
    const CartesianMeshSpec spec = MakeSpec2D();
    const std::string name = GenerateCartesianMeshName(spec);

    CHECK(name == "cartesian_mesh_2D_n10x20_s2x1.mesh");
}

// Non-zero origin appears in the generated mesh filename.
void TestMeshNameWithOrigin()
{
    CartesianMeshSpec spec = MakeSpec2D();
    spec.origin = {0.5, 0.0};

    const std::string name = GenerateCartesianMeshName(spec);

    CHECK(name == "cartesian_mesh_2D_n10x20_s2x1_o0.5x0.mesh");
}

// Builder rejects dimension outside {1, 2, 3}.
void TestBuilderRejectsBadDimension()
{
    CartesianMeshSpec spec;
    spec.SetDimension(2);
    spec.dim = 4;

    bool threw = false;
    try
    {
        CartesianMeshBuilder builder(spec);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);
}

// Builder rejects n/s/origin vector lengths that do not match dim.
void TestBuilderRejectsMismatchedVectors()
{
    CartesianMeshSpec spec;
    spec.SetDimension(2);
    spec.n = {2, 2, 2};

    bool threw = false;
    try
    {
        CartesianMeshBuilder builder(spec);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);
}

// Builder rejects zero or negative cell spacing.
void TestBuilderRejectsNonPositiveSpacing()
{
    CartesianMeshSpec spec;
    spec.SetDimension(2);
    spec.s[1] = 0.0;

    bool threw = false;
    try
    {
        CartesianMeshBuilder builder(spec);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);
}

std::vector<char *> ArgvFromStrings(std::vector<std::string> *storage)
{
    std::vector<char *> argv;
    argv.reserve(storage->size());
    for (std::string &s : *storage)
    {
        argv.push_back(s.data());
    }
    return argv;
}

// ParseCLI reads dimension, counts, spacing, and default origin from argv.
void TestParseCLIBasic2D()
{
    std::vector<std::string> args = {"prog", "2", "10", "20", "1", "2"};
    std::vector<char *> argv = ArgvFromStrings(&args);

    const CartesianMeshSpec spec = CartesianMeshBuilder::ParseCLI(
        static_cast<int>(argv.size()), argv.data());

    CHECK(spec.dim == 2);
    CHECK(spec.n[0] == 10);
    CHECK(spec.n[1] == 20);
    CHECK_NEAR(spec.s[0], 1.0, 1e-12);
    CHECK_NEAR(spec.s[1], 2.0, 1e-12);
    CHECK_NEAR(spec.origin[0], 0.0, 1e-12);
    CHECK_NEAR(spec.origin[1], 0.0, 1e-12);
}

// ParseCLI reads optional origin coordinates after spacing arguments.
void TestParseCLIWithOrigin()
{
    std::vector<std::string> args = {"prog", "2", "4", "4", "1", "1", "0.5", "0"};
    std::vector<char *> argv = ArgvFromStrings(&args);

    const CartesianMeshSpec spec = CartesianMeshBuilder::ParseCLI(
        static_cast<int>(argv.size()), argv.data());

    CHECK(spec.dim == 2);
    CHECK_NEAR(spec.origin[0], 0.5, 1e-12);
    CHECK_NEAR(spec.origin[1], 0.0, 1e-12);
}

// Built MFEM meshes have expected element/vertex counts and geometry types in 1D/2D/3D.
void TestBuildCartesianMeshCounts()
{
    {
        CartesianMeshSpec spec;
        spec.SetDimension(1);
        spec.n[0] = 20;
        spec.s[0] = 1.0;

        const CartesianMeshBuilder builder(spec);
        const mfem::Mesh mesh = builder.BuildCartesianMesh();

        CHECK(mesh.Dimension() == 1);
        CHECK(mesh.SpaceDimension() == mesh.Dimension());
        CHECK(mesh.GetNE() == 20);
        CHECK(mesh.GetNV() == 21);
    }

    {
        const CartesianMeshSpec spec = MakeSpec2D();
        const CartesianMeshBuilder builder(spec);
        const mfem::Mesh mesh = builder.BuildCartesianMesh();

        CHECK(mesh.Dimension() == 2);
        CHECK(mesh.SpaceDimension() == mesh.Dimension());
        CHECK(mesh.GetNE() == 200);
        CHECK(mesh.GetNV() == 231);
        CHECK(mesh.GetElementBaseGeometry(0) == mfem::Geometry::SQUARE);
    }

    {
        CartesianMeshSpec spec;
        spec.SetDimension(3);
        spec.n = {4, 4, 4};
        spec.s = {1.0, 1.0, 1.0};

        const CartesianMeshBuilder builder(spec);
        const mfem::Mesh mesh = builder.BuildCartesianMesh();

        CHECK(mesh.Dimension() == 3);
        CHECK(mesh.SpaceDimension() == mesh.Dimension());
        CHECK(mesh.GetNE() == 64);
        CHECK(mesh.GetNV() == 125);
        CHECK(mesh.GetElementBaseGeometry(0) == mfem::Geometry::CUBE);
    }
}

// Ref-space center (0.5, 0.5) maps to the physical center of the first cell.
void TestBuildCartesianMeshElementTransform()
{
    const CartesianMeshSpec spec = MakeSpec2D();
    const CartesianMeshBuilder builder(spec);
    mfem::Mesh mesh = builder.BuildCartesianMesh();

    mfem::ElementTransformation *T = mesh.GetElementTransformation(0);
    mfem::IntegrationPoint ip;
    ip.x = 0.5;
    ip.y = 0.5;
    T->SetIntPoint(&ip);

    mfem::Vector point(3);
    T->Transform(ip, point);

    const double hx = spec.s[0] / spec.n[0];
    const double hy = spec.s[1] / spec.n[1];
    CHECK_NEAR(point(0), 0.5 * hx, 1e-10);
    CHECK_NEAR(point(1), 0.5 * hy, 1e-10);
}

}  // namespace

void TestCartesianMesh()
{
    TestSetDimension();
    TestMeshNameWithoutOrigin();
    TestMeshNameWithOrigin();
    TestBuilderRejectsBadDimension();
    TestBuilderRejectsMismatchedVectors();
    TestBuilderRejectsNonPositiveSpacing();
    TestParseCLIBasic2D();
    TestParseCLIWithOrigin();
    TestBuildCartesianMeshCounts();
    TestBuildCartesianMeshElementTransform();
}
