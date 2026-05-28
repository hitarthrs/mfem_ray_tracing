#include "id_generator.hpp"
#include "ien_generator.hpp"
#include "lm_generator.hpp"
#include "test_helpers.hpp"

namespace
{

// One bilinear element, four points, scalar: LM[0][a][0] = ID[IEN[0][a]][0] = a + 1.
void TestLMSingleElementScalar()
{
    const std::vector<double> knots = {0.0, 0.0, 1.0, 1.0};
    const IENGenerator ien_gen(knots, knots, 1, 1);
    const IDGenerator id_gen(4, 1);

    const auto lm = LMGenerator::Generate(ien_gen, id_gen);
    CHECK(static_cast<int>(lm.size()) == 1);
    CHECK(static_cast<int>(lm[0].size()) == 4);

    for (int a = 0; a < 4; ++a)
    {
        CHECK(lm[0][a][0] == a + 1);
    }
}

// LM passes through fixed DOF encoding from ID (P = 0).
void TestLMWithFixedDof()
{
    const std::vector<double> knots = {0.0, 0.0, 1.0, 1.0};
    const IENGenerator ien_gen(knots, knots, 1, 1);
    IDGenerator id_gen(4, 1);
    id_gen.SetFixedDOF(2);

    const auto ien = ien_gen.Generate2DIEN();
    const auto id = id_gen.Generate2DIDArray();
    const auto lm = LMGenerator::Generate(ien, id);

    CHECK(lm[0][2][0] == 0);
    CHECK(lm[0][0][0] == 1);
    CHECK(lm[0][3][0] == 3);
}

// 3×3 elements, vector DOFs: spot-check LM[e][a][i] = ID[IEN[e][a]][i].
void TestLMQuadraticThreeSpanPatch()
{
    const std::vector<double> knots = {
        0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0};
    const IENGenerator ien_gen(knots, knots, 2, 2);
    IDGenerator id_gen(25, 2);
    id_gen.SetFixedDOF(12, 0);

    const auto ien = ien_gen.Generate2DIEN();
    const auto id = id_gen.Generate2DIDArray();
    const auto lm = LMGenerator::Generate(ien, id);

    CHECK(static_cast<int>(lm.size()) == 9);
    CHECK(lm[0][0][0] == id[ien[0][0]][0]);
    CHECK(lm[4][4][1] == id[ien[4][4]][1]);
    CHECK(lm[4][4][0] == id[12][0]);
    CHECK(lm[4][4][0] == 0);
}

}  // namespace

void TestLMGenerator()
{
    TestLMSingleElementScalar();
    TestLMWithFixedDof();
    TestLMQuadraticThreeSpanPatch();
}
