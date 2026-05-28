#include "id_generator.hpp"
#include "test_helpers.hpp"

namespace
{

// Five scalar points, no BCs: ID[A][0] = A + 1 (1-based), N_eq = 5.
void TestScalarAllFreeZeroBasedA()
{
    IDGenerator gen(5, 1);
    const auto id = gen.Generate2DIDArray();

    CHECK(static_cast<int>(id.size()) == 5);
    CHECK(gen.GetTotalActiveEquations() == 5);

    for (int A = 0; A < 5; ++A)
    {
        CHECK(static_cast<int>(id[A].size()) == 1);
        CHECK(id[A][0] == A + 1);
    }
}

// Fix control point A = 2 (0-based): active IDs compress, fixed gets 0.
void TestScalarOneFixedPoint()
{
    IDGenerator gen(5, 1);
    gen.SetFixedDOF(2);

    const auto id = gen.Generate2DIDArray();
    CHECK(gen.GetTotalActiveEquations() == 4);

    CHECK(id[0][0] == 1);
    CHECK(id[1][0] == 2);
    CHECK(id[2][0] == 0);
    CHECK(id[3][0] == 3);
    CHECK(id[4][0] == 4);
}

// Two points, two DOFs each (i = 0, 1); fix (A=0, i=1).
void TestVectorDofZeroBasedAI()
{
    IDGenerator gen(2, 2);
    gen.SetFixedDOF(0, 1);

    const auto id = gen.Generate2DIDArray();
    CHECK(gen.GetTotalActiveEquations() == 3);

    CHECK(id[0][0] == 1);
    CHECK(id[0][1] == 0);
    CHECK(id[1][0] == 2);
    CHECK(id[1][1] == 3);
}

}  // namespace

void TestIDGenerator()
{
    TestScalarAllFreeZeroBasedA();
    TestScalarOneFixedPoint();
    TestVectorDofZeroBasedAI();
}
