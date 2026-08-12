#include "test_helpers.hpp"
#include "mfem_raytracing/tspline/tspline.hpp"

#include <cmath>

namespace
{

mfem_raytracing::tspline::TMesh MakeGrid(int n)
{
    using namespace mfem_raytracing::tspline;
    TMesh mesh;
    for (int t = 0; t < n; ++t)
    {
        for (int s = 0; s < n; ++s)
        {
            mesh.AddControlPoint({static_cast<double>(s), static_cast<double>(t)},
                                 {static_cast<double>(s), static_cast<double>(t), 0.0});
        }
    }
    const auto id = [n](int s, int t) { return static_cast<std::size_t>(t * n + s); };
    for (int t = 0; t < n; ++t)
    {
        for (int s = 0; s < n; ++s)
        {
            if (s + 1 < n) { mesh.AddEdge(id(s, t), id(s + 1, t)); }
            if (t + 1 < n) { mesh.AddEdge(id(s, t), id(s, t + 1)); }
        }
    }
    return mesh;
}

} // namespace

void TestTSpline()
{
    using namespace mfem_raytracing::tspline;

    const std::vector<double> cubic = {0.0, 1.0, 2.0, 3.0, 4.0};
    CHECK_NEAR(BSplineBasis(cubic, 2.0), 2.0 / 3.0, 1e-12);
    CHECK_NEAR(BSplineBasis(cubic, -0.01), 0.0, 1e-12);
    // Degree-one open-boundary hats have a repeated knot on one side.  Their
    // peak must remain one at the non-repeated face corner.
    CHECK_NEAR(BSplineBasis({0.0, 0.0, 1.0}, 0.0), 1.0, 1e-12);
    CHECK_NEAR(BSplineBasis({0.0, 1.0, 1.0}, 1.0), 1.0, 1e-12);

    TMesh grid = MakeGrid(5);
    grid.Validate();
    const LocalKnotVectors middle = grid.LocalKnots(12);
    CHECK(middle.s == cubic);
    CHECK(middle.t == cubic);
    CHECK(grid.TJunctions().empty());

    // Degree-one local knots use the paper's ray cast.  At this T-junction,
    // the missing direct +s edge must cast across the empty bay to the
    // vertical segment at s=3 rather than extrapolating an arbitrary boundary
    // interval from s=1.
    TMesh ray_cast;
    const auto add = [&ray_cast](double s, double t) {
        return ray_cast.AddControlPoint({s, t}, {s, t, 0.0});
    };
    const std::size_t p00 = add(0.0, 0.0), p10 = add(1.0, 0.0), p30 = add(3.0, 0.0);
    const std::size_t p01 = add(0.0, 1.0), p11 = add(1.0, 1.0), p31 = add(3.0, 1.0);
    const std::size_t p02 = add(0.0, 2.0), p12 = add(1.0, 2.0), p32 = add(3.0, 2.0);
    for (const auto [a, b] : std::vector<Edge>{{p00, p10}, {p10, p30},
                                                 {p01, p11},
                                                 {p02, p12}, {p12, p32},
                                                 {p00, p01}, {p01, p02},
                                                 {p10, p11}, {p11, p12},
                                                 {p30, p31}, {p31, p32}})
    {
        ray_cast.AddEdge(a, b);
    }
    const LocalKnotVectors ray_knots = ray_cast.LocalKnots(p11, 1);
    CHECK(ray_knots.s == std::vector<double>({0.0, 1.0, 3.0}));

    // A degree-1 T-spline has an exact rational bilinear restriction on every
    // knot cell, which is the representation consumed by the Embree leaf path.
    const std::vector<TSplineLeaf> extracted = ExtractDegreeOneLeaves(grid);
    CHECK(extracted.size() == 16);
    CHECK_NEAR(extracted.front().s_domain[0], 0.0, 1e-12);
    CHECK_NEAR(extracted.front().t_domain[1], 1.0, 1e-12);
    CHECK_NEAR(extracted.front().control_points[0][0][0], 0.0, 1e-12);
    CHECK_NEAR(extracted.front().control_points[1][1][1], 1.0, 1e-12);

    // The normalized form in paper Eq. (1) reproduces a weighted affine blend.
    const LocalKnotVectors knots = {cubic, cubic};
    PBSplineSurface rational({
        {{0.0, 0.0, 0.0}, 1.0, knots},
        {{2.0, 4.0, 6.0}, 3.0, knots},
    });
    const Point3 result = rational.Evaluate(2.0, 2.0);
    CHECK_NEAR(result[0], 1.5, 1e-12);
    CHECK_NEAR(result[1], 3.0, 1e-12);
    CHECK_NEAR(result[2], 4.5, 1e-12);

    // Remove one arm of the central cross: it becomes a T-junction.  The
    // two-bay extension is the cubic Bezier-domain construction in Sec. 5.2.
    TMesh tmesh = MakeGrid(5);
    const auto &old_edges = tmesh.Edges();
    TMesh with_t;
    for (const ControlPoint &p : tmesh.ControlPoints())
    {
        with_t.AddControlPoint(p.parameter, p.position, p.weight);
    }
    for (const Edge &edge : old_edges)
    {
        if (!(edge.first == 12 && edge.second == 13)) { with_t.AddEdge(edge.first, edge.second); }
    }
    const std::vector<std::size_t> junctions = with_t.TJunctions();
    CHECK(junctions.size() == 2);
    const std::vector<TJunctionExtension> extensions = with_t.TJunctionExtensions(2);
    CHECK(extensions.size() == 2);
    CHECK_NEAR(extensions[0].end[0] - extensions[0].start[0], 2.0, 1e-12);

    bool rejected = false;
    try
    {
        TMesh invalid;
        invalid.AddControlPoint({0.0, 0.0}, {0.0, 0.0, 0.0});
        invalid.AddControlPoint({1.0, 1.0}, {1.0, 1.0, 0.0});
        invalid.AddEdge(0, 1);
        invalid.Validate();
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    CHECK(rejected);
}
