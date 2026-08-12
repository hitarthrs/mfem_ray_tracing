#ifndef MFEM_RAYTRACING_TSPLINE_HPP
#define MFEM_RAYTRACING_TSPLINE_HPP

/**
 * Small, dependency-free T-spline kernel based on Sederberg et al.,
 * "T-splines and T-NURCCs" (2003).
 *
 * This is deliberately a representation/evaluation library, rather than a
 * complete modelling system.  It implements the paper's knot-interval T-mesh,
 * cubic local knot vectors, PB-spline blending functions, and their rational
 * generalisation.  Extraordinary-point T-NURCC refinement is out of scope.
 */

#include <array>
#include <cstddef>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

using Point2 = std::array<double, 2>;
using Point3 = std::array<double, 3>;
using Point4 = std::array<double, 4>;

struct ControlPoint
{
    Point2 parameter = {0.0, 0.0};
    Point3 position = {0.0, 0.0, 0.0};
    double weight = 1.0;
};

struct Edge
{
    std::size_t first = 0;
    std::size_t second = 0;
};

struct LocalKnotVectors
{
    std::vector<double> s;
    std::vector<double> t;
};

struct TJunctionExtension
{
    std::size_t junction = 0;
    Point2 start = {0.0, 0.0};
    Point2 end = {0.0, 0.0};
};

/** Exact rational bilinear restriction of a degree-1 T-spline knot cell. */
struct TSplineLeaf
{
    std::array<std::array<Point3, 2>, 2> control_points = {};
    std::array<std::array<double, 2>, 2> weights = {};
    std::array<double, 2> s_domain = {0.0, 1.0};
    std::array<double, 2> t_domain = {0.0, 1.0};
};

/** A rectilinear control grid with optional T-junctions. */
class TMesh
{
public:
    std::size_t AddControlPoint(const Point2 &parameter,
                                const Point3 &position,
                                double weight = 1.0);
    void AddEdge(std::size_t first, std::size_t second);

    const std::vector<ControlPoint> &ControlPoints() const { return points_; }
    const std::vector<Edge> &Edges() const { return edges_; }

    /// Throws std::invalid_argument when the control grid is not rectilinear.
    void Validate(double tolerance = 1e-12) const;

    /// Vertices with exactly three cardinal incident directions.
    std::vector<std::size_t> TJunctions(double tolerance = 1e-12) const;

    /**
     * Derive the p+2 local knots in both directions by tracing the T-mesh.
     * At an open boundary the last knot interval is extrapolated; applications
     * with special boundary conditions can instead provide PBControlPoints
     * with explicit local vectors to PBSplineSurface.
     */
    LocalKnotVectors LocalKnots(std::size_t vertex,
                                int degree = 3,
                                double tolerance = 1e-12) const;

    /// The missing arm of each T-junction, extended across `bays` knot spans.
    std::vector<TJunctionExtension> TJunctionExtensions(
        std::size_t bays = 2, double tolerance = 1e-12) const;

private:
    std::vector<ControlPoint> points_;
    std::vector<Edge> edges_;
};

/** One independently specified PB-spline control function (paper Eq. 1-4). */
struct PBControlPoint
{
    Point3 position = {0.0, 0.0, 0.0};
    double weight = 1.0;
    LocalKnotVectors knots;
};

/** Evaluate PB-splines and rational T-spline surfaces. */
class PBSplineSurface
{
public:
    explicit PBSplineSurface(std::vector<PBControlPoint> controls);

    /// Tensor-product basis B_i(s,t) from paper Eq. 2.
    static double Basis(const LocalKnotVectors &knots, double s, double t);

    /// Paper Eq. 1 when all weights are one; rational generalisation otherwise.
    Point3 Evaluate(double s, double t) const;

    /// Homogeneous numerator and denominator of the rational PB-spline.
    Point4 EvaluateHomogeneous(double s, double t) const;

private:
    std::vector<PBControlPoint> controls_;
};

/** Convenience adapter that derives each control point's local vectors from a T-mesh. */
class TSplineSurface
{
public:
    explicit TSplineSurface(const TMesh &mesh, int degree = 3);
    Point3 Evaluate(double s, double t) const;
    Point4 EvaluateHomogeneous(double s, double t) const;

private:
    PBSplineSurface surface_;
};

/** Cox-de Boor evaluation of the sole degree-p basis defined by p+2 knots. */
double BSplineBasis(const std::vector<double> &knots, double parameter);

/**
 * Extract every elementary knot cell as an exact rational bilinear patch.
 *
 * This is intentionally restricted to degree 1: on such a T-spline the
 * homogeneous surface is bilinear inside every knot cell, so the returned
 * leaves can be handed directly to the existing Embree bilinear primitive.
 */
std::vector<TSplineLeaf> ExtractDegreeOneLeaves(const TMesh &mesh);

} // namespace tspline
} // namespace mfem_raytracing

#endif
