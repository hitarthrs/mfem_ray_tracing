#include "id_generator.hpp"
#include "ien_generator.hpp"
#include "lm_generator.hpp"
#include "test_helpers.hpp"

#include <vector>

namespace
{

constexpr double kTol = 0.0;

// Reference uses element index e = ev * n_elem_u + eu (η slow, ξ fast).
// IENGenerator stores e = eu * n_elem_v + ev.
int OurElementIndex(int eu, int ev, int num_elems_v) { return eu * num_elems_v + ev; }

void SetReferenceDirichletBoundary(IDGenerator &id)
{
    for (int A : {0, 5, 10, 15, 20})
    {
        id.SetFixedDOF(A, 0);
        id.SetFixedDOF(A, 1);
    }
}

// Table I (1-based A): nine elements, nine local bases per element.
const int kIENReference1Based[9][9] = {
    {1, 2, 3, 6, 7, 8, 11, 12, 13},
    {2, 3, 4, 7, 8, 9, 12, 13, 14},
    {3, 4, 5, 8, 9, 10, 13, 14, 15},
    {6, 7, 8, 11, 12, 13, 16, 17, 18},
    {7, 8, 9, 12, 13, 14, 17, 18, 19},
    {8, 9, 10, 13, 14, 15, 18, 19, 20},
    {11, 12, 13, 16, 17, 18, 21, 22, 23},
    {12, 13, 14, 17, 18, 19, 22, 23, 24},
    {13, 14, 15, 18, 19, 20, 23, 24, 25},
};

// Table ID (1-based P) for A = 1..25, i = 1,2.
const int kIDReference1Based[25][2] = {
    {0, 0},   {1, 2},   {3, 4},   {5, 6},   {7, 8},   {0, 0},   {9, 10},  {11, 12},
    {13, 14}, {15, 16}, {0, 0},   {17, 18}, {19, 20}, {21, 22}, {23, 24}, {0, 0},
    {25, 26}, {27, 28}, {29, 30}, {31, 32}, {0, 0},   {33, 34}, {35, 36}, {37, 38},
    {39, 40},
};

void TestReferenceIENTable()
{
    const std::vector<double> knots = {
        0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0};
    const IENGenerator ien_gen(knots, knots, 2, 2);
    const auto ien = ien_gen.Generate2DIEN();

    CHECK(ien_gen.GetNumElementsU() == 3);
    CHECK(ien_gen.GetNumElementsV() == 3);
    CHECK(ien_gen.GetTotalElements() == 9);

    for (int eu = 0; eu < 3; ++eu)
    {
        for (int ev = 0; ev < 3; ++ev)
        {
            const int e_img = ev * 3 + eu;
            const int e_our = OurElementIndex(eu, ev, 3);
            for (int a = 0; a < 9; ++a)
            {
                CHECK(ien[e_our][a] + 1 == kIENReference1Based[e_img][a]);
            }
        }
    }
}

void TestReferenceIDTable()
{
    IDGenerator id_gen(25, 2);
    SetReferenceDirichletBoundary(id_gen);

    const auto id = id_gen.Generate2DIDArray();
    CHECK(id_gen.GetTotalActiveEquations() == 40);

    for (int A = 0; A < 25; ++A)
    {
        for (int i = 0; i < 2; ++i)
        {
            CHECK_NEAR(id[A][i], kIDReference1Based[A][i], kTol);
        }
    }
}

void TestReferenceLMTable()
{
    const std::vector<double> knots = {
        0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0};
    const IENGenerator ien_gen(knots, knots, 2, 2);
    IDGenerator id_gen(25, 2);
    SetReferenceDirichletBoundary(id_gen);

    const auto ien = ien_gen.Generate2DIEN();
    const auto id = id_gen.Generate2DIDArray();
    const auto lm = LMGenerator::Generate(ien, id);

    for (int eu = 0; eu < 3; ++eu)
    {
        for (int ev = 0; ev < 3; ++ev)
        {
            const int e_our = OurElementIndex(eu, ev, 3);
            for (int a = 0; a < 9; ++a)
            {
                for (int i = 0; i < 2; ++i)
                {
                    CHECK(lm[e_our][a][i] == id[ien[e_our][a]][i]);
                }
            }
        }
    }

    // Spot-check Table II: element 1 (e_img = 0), local a = 0,1.
    const int e0 = OurElementIndex(0, 0, 3);
    CHECK(lm[e0][0][0] == 0);
    CHECK(lm[e0][0][1] == 0);
    CHECK(lm[e0][1][0] == 1);
    CHECK(lm[e0][1][1] == 2);

    // Element 2 in the tables (e_img = 1): LM(a=0) uses A=2 → P=1,2.
    const int e1 = OurElementIndex(1, 0, 3);
    CHECK(lm[e1][0][0] == 1);
    CHECK(lm[e1][0][1] == 2);
}

}  // namespace

void TestIGAReferenceConnectivity()
{
    TestReferenceIENTable();
    TestReferenceIDTable();
    TestReferenceLMTable();
}
