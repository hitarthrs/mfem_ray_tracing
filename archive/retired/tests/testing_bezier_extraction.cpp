/**
 * Construct a 1D B-spline patch in R^3 (non-rational: all weights w_i = 1).
 *
 * MFEM stores geometry in a NURBSPatch; with unit weights the curve is a
 * polynomial B-spline, not a rational NURBS.
 *
 * Knot vector (14 values, degree p=4, n=9 CPs):
 *   0,0,0,0,0, 0.33,0.33, 0.66,0.66, 1,1,1,1,1
 *
 * Build:
 *   cmake --build build --target testing_bezier_extraction
 * Run (from repo root):
 *   ./build/testing_bezier_extraction
 */

#include "mfem.hpp"

#include <iostream>
#include <sstream>
#include <vector>

namespace
{

constexpr double kUnitWeight = 1.0;

mfem::KnotVector MakeKnotVector()
{
    std::istringstream input(
        "4 9 0 0 0 0 0 0.33 0.33 0.66 0.66 1 1 1 1 1");
    return mfem::KnotVector(input);
}

/// 9 control points in R^3 (example geometry — edit as needed).
std::vector<mfem::Vector> MakeControlPoints3D()
{
    return {
        mfem::Vector({0.0, 0.0, 0.0}),
        mfem::Vector({1.0, 0.0, 0.5}),
        mfem::Vector({2.0, 0.5, 1.0}),
        mfem::Vector({3.0, 1.0, 1.0}),
        mfem::Vector({4.0, 1.5, 0.5}),
        mfem::Vector({5.0, 1.0, 0.0}),
        mfem::Vector({6.0, 0.5, -0.5}),
        mfem::Vector({7.0, 0.0, -1.0}),
        mfem::Vector({8.0, -0.5, 0.0}),
    };
}

/// Pack homogeneous CP data (wx, wy, wz, w). With w=1 this is just (x, y, z, 1).
std::vector<double> ToHomogeneousControlData(const std::vector<mfem::Vector> &cp)
{
    const int n = static_cast<int>(cp.size());
    std::vector<double> data(4 * n);
    for (int i = 0; i < n; ++i)
    {
        data[4 * i + 0] = cp[static_cast<std::size_t>(i)](0);
        data[4 * i + 1] = cp[static_cast<std::size_t>(i)](1);
        data[4 * i + 2] = cp[static_cast<std::size_t>(i)](2);
        data[4 * i + 3] = kUnitWeight;
    }
    return data;
}

void PrintControlPoints(const mfem::NURBSPatch &patch, int ncp)
{
    std::cout << "Control points read back from patch (B-spline, w=1, R^3):\n";
    for (int i = 0; i < ncp; ++i)
    {
        const double w = patch(i, 3);
        const double x = patch(i, 0) / w;
        const double y = patch(i, 1) / w;
        const double z = patch(i, 2) / w;
        std::cout << "  Q" << i << " = (" << x << ", " << y << ", " << z
                  << ")  w=" << w << "\n";
    }
}

void PrintKnotVector(const mfem::KnotVector &kv, const char *label)
{
    std::cout << label << "\n";
    std::cout << "  degree p  = " << kv.GetOrder() << "\n";
    std::cout << "  n CPs     = " << kv.GetNCP() << "\n";
    std::cout << "  n spans   = " << kv.GetNE() << "\n";
    std::cout << "  knots     =";
    for (int i = 0; i < kv.Size(); ++i)
    {
        std::cout << " " << kv[i];
    }
    std::cout << "\n\n";
}

}  // namespace

int main()
{
    const mfem::KnotVector kv = MakeKnotVector();

    std::cout << "1D B-spline patch (all weights w_i = " << kUnitWeight << ")\n";
    PrintKnotVector(kv, "Before knot insertion:");

    const std::vector<mfem::Vector> cp_in = MakeControlPoints3D();
    if (static_cast<int>(cp_in.size()) != kv.GetNCP())
    {
        std::cerr << "Error: expected " << kv.GetNCP() << " control points\n";
        return 1;
    }

    const std::vector<double> homog = ToHomogeneousControlData(cp_in);

    mfem::Array<const mfem::KnotVector *> kv_array(1);
    kv_array[0] = &kv;

    mfem::NURBSPatch patch(kv_array, 4, homog.data());

    std::cout << "Input control points (R^3):\n";
    for (int i = 0; i < kv.GetNCP(); ++i)
    {
        std::cout << "  Q" << i << " = (" << cp_in[static_cast<std::size_t>(i)](0) << ", "
                  << cp_in[static_cast<std::size_t>(i)](1) << ", "
                  << cp_in[static_cast<std::size_t>(i)](2) << ")\n";
    }
    std::cout << "\n";

    PrintControlPoints(patch, kv.GetNCP());

    // Insert u = 0.33 twice (increases multiplicity at that knot).
    mfem::Vector knots_to_insert(2);
    knots_to_insert(0) = 0.33;
    knots_to_insert(1) = 0.33;
    patch.KnotInsert(0, knots_to_insert);

    const mfem::KnotVector &kv_after = *patch.GetKV(0);
    PrintKnotVector(kv_after, "After KnotInsert(0.33, 0.33):");
    PrintControlPoints(patch, kv_after.GetNCP());

    return 0;
}
