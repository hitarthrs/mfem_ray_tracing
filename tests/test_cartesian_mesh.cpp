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

void TestMeshNameWithoutOrigin()
{
    const CartesianMeshSpec spec = MakeSpec2D();
    const std::string name = GenerateCartesianMeshName(spec);

    CHECK(name == "cartesian_mesh_2D_n10x20_s2x1.mesh");
}

void TestMeshNameWithOrigin()
{
    CartesianMeshSpec spec = MakeSpec2D();
    spec.origin = {0.5, 0.0};

    const std::string name = GenerateCartesianMeshName(spec);

    CHECK(name == "cartesian_mesh_2D_n10x20_s2x1_o0.5x0.mesh");
}

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
        CHECK(mesh.GetNE() == 20);
        CHECK(mesh.GetNV() == 21);
    }

    {
        const CartesianMeshSpec spec = MakeSpec2D();
        const CartesianMeshBuilder builder(spec);
        const mfem::Mesh mesh = builder.BuildCartesianMesh();

        CHECK(mesh.Dimension() == 2);
        CHECK(mesh.GetNE() == 200);
        CHECK(mesh.GetNV() == 231);
    }

    {
        CartesianMeshSpec spec;
        spec.SetDimension(3);
        spec.n = {4, 4, 4};
        spec.s = {1.0, 1.0, 1.0};

        const CartesianMeshBuilder builder(spec);
        const mfem::Mesh mesh = builder.BuildCartesianMesh();

        CHECK(mesh.Dimension() == 3);
        CHECK(mesh.GetNE() == 64);
        CHECK(mesh.GetNV() == 125);
    }
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
}
