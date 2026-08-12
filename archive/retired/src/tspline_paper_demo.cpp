#include "tspline.hpp"

#include <iomanip>
#include <iostream>

using namespace mfem_raytracing::tspline;

int main()
{
    // Fig. 8-style regular cubic control grid.  The center has local knot
    // coordinates [0, 1, 2, 3, 4] in both directions.
    TMesh mesh;
    for (int t = 0; t < 5; ++t)
    {
        for (int s = 0; s < 5; ++s)
        {
            mesh.AddControlPoint({double(s), double(t)}, {double(s), double(t), 0.15 * s * t});
        }
    }
    const auto id = [](int s, int t) { return std::size_t(t * 5 + s); };
    for (int t = 0; t < 5; ++t)
    {
        for (int s = 0; s < 5; ++s)
        {
            if (s + 1 < 5) { mesh.AddEdge(id(s, t), id(s + 1, t)); }
            if (t + 1 < 5) { mesh.AddEdge(id(s, t), id(s, t + 1)); }
        }
    }

    const LocalKnotVectors knots = mesh.LocalKnots(id(2, 2));
    std::cout << "Fig. 8 local s knots:";
    for (double knot : knots.s) { std::cout << ' ' << knot; }
    std::cout << "\nFig. 8 local t knots:";
    for (double knot : knots.t) { std::cout << ' ' << knot; }
    std::cout << "\nCubic B(2) = " << BSplineBasis(knots.s, 2.0) << '\n';

    TSplineSurface surface(mesh);
    const Point3 point = surface.Evaluate(2.0, 2.0);
    std::cout << std::fixed << std::setprecision(6)
              << "T-spline surface at (2, 2): (" << point[0] << ", "
              << point[1] << ", " << point[2] << ")\n";
    return 0;
}
