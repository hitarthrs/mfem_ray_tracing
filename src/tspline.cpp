#include "tspline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

constexpr int kS = 0;
constexpr int kT = 1;

bool Near(double a, double b, double tolerance)
{
    return std::abs(a - b) <= tolerance;
}

bool Finite(const Point2 &point)
{
    return std::isfinite(point[kS]) && std::isfinite(point[kT]);
}

int Direction(const Point2 &from, const Point2 &to, int axis, double tolerance)
{
    const int other = 1 - axis;
    if (!Near(from[other], to[other], tolerance))
    {
        return 0;
    }
    if (to[axis] > from[axis] + tolerance) { return 1; }
    if (to[axis] < from[axis] - tolerance) { return -1; }
    return 0;
}

std::vector<int> IncidentDirections(const TMesh &mesh, std::size_t vertex,
                                    double tolerance)
{
    std::vector<int> directions;
    const auto &points = mesh.ControlPoints();
    for (const Edge &edge : mesh.Edges())
    {
        if (edge.first != vertex && edge.second != vertex) { continue; }
        const std::size_t other = edge.first == vertex ? edge.second : edge.first;
        const Point2 &origin = points[vertex].parameter;
        const Point2 &target = points[other].parameter;
        const int axis = Near(origin[kS], target[kS], tolerance) ? kT : kS;
        const int sign = Direction(origin, target, axis, tolerance);
        directions.push_back(2 * axis + (sign > 0 ? 1 : 0));
    }
    return directions;
}

std::size_t NextVertex(const TMesh &mesh, std::size_t vertex, int axis,
                       int sign, double tolerance)
{
    const auto &points = mesh.ControlPoints();
    std::size_t next = points.size();
    for (const Edge &edge : mesh.Edges())
    {
        if (edge.first != vertex && edge.second != vertex) { continue; }
        const std::size_t other = edge.first == vertex ? edge.second : edge.first;
        if (Direction(points[vertex].parameter, points[other].parameter, axis,
                      tolerance) == sign)
        {
            if (next != points.size())
            {
                throw std::invalid_argument("multiple T-mesh edges leave a vertex in one direction");
            }
            next = other;
        }
    }
    return next;
}

double BoundaryInterval(const TMesh &mesh, int axis, double tolerance)
{
    std::vector<double> values;
    values.reserve(mesh.ControlPoints().size());
    for (const ControlPoint &point : mesh.ControlPoints())
    {
        values.push_back(point.parameter[axis]);
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end(),
                             [tolerance](double a, double b) { return Near(a, b, tolerance); }),
                 values.end());
    double interval = std::numeric_limits<double>::infinity();
    for (std::size_t i = 1; i < values.size(); ++i)
    {
        const double delta = values[i] - values[i - 1];
        if (delta > tolerance) { interval = std::min(interval, delta); }
    }
    if (!std::isfinite(interval))
    {
        throw std::invalid_argument("cannot infer a boundary knot interval from one coordinate");
    }
    return interval;
}

bool IsInteriorVertex(const TMesh &mesh, std::size_t vertex, double tolerance)
{
    const Point2 &parameter = mesh.ControlPoints()[vertex].parameter;
    for (int axis = 0; axis < 2; ++axis)
    {
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        for (const ControlPoint &point : mesh.ControlPoints())
        {
            minimum = std::min(minimum, point.parameter[axis]);
            maximum = std::max(maximum, point.parameter[axis]);
        }
        if (parameter[axis] <= minimum + tolerance ||
            parameter[axis] >= maximum - tolerance)
        {
            return false;
        }
    }
    return true;
}

std::vector<double> KnotsAlong(const TMesh &mesh, std::size_t vertex, int axis,
                               int degree, double tolerance)
{
    const int left_count = (degree + 1) / 2;
    const int right_count = degree + 1 - left_count;
    const auto &points = mesh.ControlPoints();
    const double interval = BoundaryInterval(mesh, axis, tolerance);

    std::vector<double> left;
    left.reserve(left_count);
    std::size_t current = vertex;
    double coordinate = points[current].parameter[axis];
    double last_interval = interval;
    for (int i = 0; i < left_count; ++i)
    {
        const std::size_t next = NextVertex(mesh, current, axis, -1, tolerance);
        if (next == points.size())
        {
            coordinate -= last_interval;
        }
        else
        {
            const double next_coordinate = points[next].parameter[axis];
            last_interval = coordinate - next_coordinate;
            coordinate = next_coordinate;
            current = next;
        }
        left.push_back(coordinate);
    }

    std::vector<double> result(left.rbegin(), left.rend());
    result.push_back(points[vertex].parameter[axis]);
    current = vertex;
    coordinate = points[current].parameter[axis];
    last_interval = interval;
    for (int i = 0; i < right_count; ++i)
    {
        const std::size_t next = NextVertex(mesh, current, axis, 1, tolerance);
        if (next == points.size())
        {
            coordinate += last_interval;
        }
        else
        {
            const double next_coordinate = points[next].parameter[axis];
            last_interval = next_coordinate - coordinate;
            coordinate = next_coordinate;
            current = next;
        }
        result.push_back(coordinate);
    }
    return result;
}

// Section 4 of the degree-one construction uses a ray cast through the
// T-mesh, not merely the next graph edge incident to the control point.  This
// distinction matters at a T-junction: a ray may cross an empty bay before it
// reaches the next perpendicular segment.  The Python hard-seam pipeline
// uses precisely this rule when it writes each control's three local knots.
double RayCastDegreeOneKnot(const TMesh &mesh, std::size_t vertex, int axis,
                            int direction, double tolerance)
{
    const auto &points = mesh.ControlPoints();
    const Point2 &origin = points[vertex].parameter;
    const int perpendicular = 1 - axis;
    const double coordinate = origin[axis];
    const double cross_coordinate = origin[perpendicular];
    double found = coordinate;
    bool have_hit = false;

    for (const Edge &edge : mesh.Edges())
    {
        const Point2 &a = points[edge.first].parameter;
        const Point2 &b = points[edge.second].parameter;
        // A ray travelling in s meets vertical T-mesh segments; a ray in t
        // meets horizontal ones.
        if (!Near(a[axis], b[axis], tolerance) ||
            Near(a[perpendicular], b[perpendicular], tolerance))
        {
            continue;
        }
        const double candidate = 0.5 * (a[axis] + b[axis]);
        if ((candidate - coordinate) * static_cast<double>(direction) <= tolerance)
        {
            continue;
        }
        const double lo = std::min(a[perpendicular], b[perpendicular]);
        const double hi = std::max(a[perpendicular], b[perpendicular]);
        if (cross_coordinate < lo - tolerance || cross_coordinate > hi + tolerance)
        {
            continue;
        }
        if (!have_hit || (candidate - found) * static_cast<double>(direction) < 0.0)
        {
            found = candidate;
            have_hit = true;
        }
    }
    // A missing ray terminates at the current knot, producing the repeated
    // end knot expected for an open degree-one T-spline boundary.
    return found;
}

LocalKnotVectors DegreeOneRayCastKnots(const TMesh &mesh, std::size_t vertex,
                                       double tolerance)
{
    const Point2 &parameter = mesh.ControlPoints()[vertex].parameter;
    return {{RayCastDegreeOneKnot(mesh, vertex, kS, -1, tolerance), parameter[kS],
             RayCastDegreeOneKnot(mesh, vertex, kS, +1, tolerance)},
            {RayCastDegreeOneKnot(mesh, vertex, kT, -1, tolerance), parameter[kT],
             RayCastDegreeOneKnot(mesh, vertex, kT, +1, tolerance)}};
}

Point3 AddScaled(const Point3 &a, const Point3 &b, double scale)
{
    return {a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2]};
}

} // namespace

std::size_t TMesh::AddControlPoint(const Point2 &parameter, const Point3 &position,
                                   double weight)
{
    if (!Finite(parameter) || !std::isfinite(weight) || weight <= 0.0)
    {
        throw std::invalid_argument("T-spline control points need finite coordinates and positive weights");
    }
    points_.push_back({parameter, position, weight});
    return points_.size() - 1;
}

void TMesh::AddEdge(std::size_t first, std::size_t second)
{
    if (first >= points_.size() || second >= points_.size() || first == second)
    {
        throw std::invalid_argument("T-mesh edge has invalid endpoints");
    }
    edges_.push_back({first, second});
}

void TMesh::Validate(double tolerance) const
{
    if (tolerance <= 0.0) { throw std::invalid_argument("T-mesh tolerance must be positive"); }
    std::set<std::pair<std::size_t, std::size_t>> seen;
    for (const Edge &edge : edges_)
    {
        if (edge.first >= points_.size() || edge.second >= points_.size() ||
            edge.first == edge.second)
        {
            throw std::invalid_argument("T-mesh edge has invalid endpoints");
        }
        const auto ordered = std::minmax(edge.first, edge.second);
        if (!seen.insert(ordered).second)
        {
            throw std::invalid_argument("T-mesh contains a duplicate edge");
        }
        const Point2 &a = points_[edge.first].parameter;
        const Point2 &b = points_[edge.second].parameter;
        const bool horizontal = Near(a[kT], b[kT], tolerance) && !Near(a[kS], b[kS], tolerance);
        const bool vertical = Near(a[kS], b[kS], tolerance) && !Near(a[kT], b[kT], tolerance);
        if (!horizontal && !vertical)
        {
            throw std::invalid_argument("T-mesh edges must be nonzero and axis-aligned");
        }
    }
    for (std::size_t i = 0; i < points_.size(); ++i)
    {
        const std::vector<int> directions = IncidentDirections(*this, i, tolerance);
        std::set<int> unique(directions.begin(), directions.end());
        if (unique.size() != directions.size())
        {
            throw std::invalid_argument("T-mesh has two edges leaving a vertex in one direction");
        }
    }
}

std::vector<std::size_t> TMesh::TJunctions(double tolerance) const
{
    Validate(tolerance);
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < points_.size(); ++i)
    {
        if (IsInteriorVertex(*this, i, tolerance) &&
            IncidentDirections(*this, i, tolerance).size() == 3)
        {
            result.push_back(i);
        }
    }
    return result;
}

LocalKnotVectors TMesh::LocalKnots(std::size_t vertex, int degree,
                                   double tolerance) const
{
    Validate(tolerance);
    if (vertex >= points_.size()) { throw std::out_of_range("T-mesh control point index"); }
    if (degree < 1) { throw std::invalid_argument("T-spline degree must be positive"); }
    if (degree == 1) { return DegreeOneRayCastKnots(*this, vertex, tolerance); }
    return {KnotsAlong(*this, vertex, kS, degree, tolerance),
            KnotsAlong(*this, vertex, kT, degree, tolerance)};
}

std::vector<TJunctionExtension> TMesh::TJunctionExtensions(std::size_t bays,
                                                            double tolerance) const
{
    Validate(tolerance);
    std::vector<TJunctionExtension> result;
    for (const std::size_t junction : TJunctions(tolerance))
    {
        const Point2 start = points_[junction].parameter;
        const std::vector<int> directions = IncidentDirections(*this, junction, tolerance);
        bool occupied[2][2] = {{false, false}, {false, false}};
        for (const int direction : directions)
        {
            occupied[direction / 2][direction % 2] = true;
        }
        for (int axis = 0; axis < 2; ++axis)
        {
            for (int sign_index = 0; sign_index < 2; ++sign_index)
            {
                if (occupied[axis][sign_index]) { continue; }
                const int sign = sign_index == 0 ? -1 : 1;
                Point2 end = start;
                end[axis] += sign * static_cast<double>(bays) *
                             BoundaryInterval(*this, axis, tolerance);
                result.push_back({junction, start, end});
            }
        }
    }
    return result;
}

double BSplineBasis(const std::vector<double> &knots, double parameter)
{
    if (knots.size() < 3)
    {
        throw std::invalid_argument("a B-spline basis needs degree + 2 knots");
    }
    if (!std::is_sorted(knots.begin(), knots.end()))
    {
        throw std::invalid_argument("B-spline knots must be nondecreasing");
    }
    const int degree = static_cast<int>(knots.size()) - 2;
    if (parameter < knots.front() || parameter > knots.back()) { return 0.0; }

    // The hard-seam T-mesh uses degree-one hat functions with repeated knots
    // on open boundaries, e.g. [a, a, b] and [a, b, b].  The generic
    // Cox-de Boor endpoint convention below leaves the latter's value at b
    // zero, while the Section-4 construction (and the Python reference)
    // assigns the control's peak value there.  Use the compact degree-one
    // definition directly so every face corner has a valid partition of unity.
    if (degree == 1)
    {
        const double a = knots[0];
        const double b = knots[1];
        const double c = knots[2];
        if (parameter == b) { return 1.0; }
        if (a <= parameter && parameter < b && b > a)
        {
            return (parameter - a) / (b - a);
        }
        if (b < parameter && parameter <= c && c > b)
        {
            return (c - parameter) / (c - b);
        }
        return 0.0;
    }

    std::vector<double> values(knots.size() - 1, 0.0);
    for (std::size_t i = 0; i + 1 < knots.size(); ++i)
    {
        values[i] = (knots[i] <= parameter && parameter < knots[i + 1]) ? 1.0 : 0.0;
    }
    // Adopt the usual right-end convention for a clamped final span.
    if (parameter == knots.back() && knots[knots.size() - 2] < parameter)
    {
        values[values.size() - 1] = 1.0;
    }
    for (int p = 1; p <= degree; ++p)
    {
        for (std::size_t i = 0; i + static_cast<std::size_t>(p) + 1 < knots.size(); ++i)
        {
            const double left_denominator = knots[i + p] - knots[i];
            const double right_denominator = knots[i + p + 1] - knots[i + 1];
            const double left = left_denominator == 0.0 ? 0.0 :
                (parameter - knots[i]) * values[i] / left_denominator;
            const double right = right_denominator == 0.0 ? 0.0 :
                (knots[i + p + 1] - parameter) * values[i + 1] / right_denominator;
            values[i] = left + right;
        }
    }
    return values.front();
}

PBSplineSurface::PBSplineSurface(std::vector<PBControlPoint> controls)
    : controls_(std::move(controls))
{
    if (controls_.empty()) { throw std::invalid_argument("a PB-spline surface needs controls"); }
    for (const PBControlPoint &control : controls_)
    {
        if (control.knots.s.size() < 3 || control.knots.t.size() < 3 ||
            !std::isfinite(control.weight) || control.weight <= 0.0)
        {
            throw std::invalid_argument("PB-spline controls need local knots and positive weights");
        }
    }
}

double PBSplineSurface::Basis(const LocalKnotVectors &knots, double s, double t)
{
    return BSplineBasis(knots.s, s) * BSplineBasis(knots.t, t);
}

Point3 PBSplineSurface::Evaluate(double s, double t) const
{
    const Point4 homogeneous = EvaluateHomogeneous(s, t);
    if (std::abs(homogeneous[3]) <= std::numeric_limits<double>::epsilon())
    {
        throw std::out_of_range("parameter lies outside the PB-spline domain");
    }
    return {homogeneous[0] / homogeneous[3], homogeneous[1] / homogeneous[3],
            homogeneous[2] / homogeneous[3]};
}

Point4 PBSplineSurface::EvaluateHomogeneous(double s, double t) const
{
    Point4 numerator = {0.0, 0.0, 0.0, 0.0};
    for (const PBControlPoint &control : controls_)
    {
        const double blending = control.weight * Basis(control.knots, s, t);
        numerator[0] += blending * control.position[0];
        numerator[1] += blending * control.position[1];
        numerator[2] += blending * control.position[2];
        numerator[3] += blending;
    }
    return numerator;
}

TSplineSurface::TSplineSurface(const TMesh &mesh, int degree)
    : surface_([&mesh, degree]() {
          mesh.Validate();
          std::vector<PBControlPoint> controls;
          controls.reserve(mesh.ControlPoints().size());
          for (std::size_t i = 0; i < mesh.ControlPoints().size(); ++i)
          {
              const ControlPoint &point = mesh.ControlPoints()[i];
              controls.push_back({point.position, point.weight, mesh.LocalKnots(i, degree)});
          }
          return controls;
      }())
{
}

Point3 TSplineSurface::Evaluate(double s, double t) const
{
    return surface_.Evaluate(s, t);
}

Point4 TSplineSurface::EvaluateHomogeneous(double s, double t) const
{
    return surface_.EvaluateHomogeneous(s, t);
}

std::vector<TSplineLeaf> ExtractDegreeOneLeaves(const TMesh &mesh)
{
    mesh.Validate();
    std::vector<double> s_knots;
    std::vector<double> t_knots;
    s_knots.reserve(mesh.ControlPoints().size());
    t_knots.reserve(mesh.ControlPoints().size());
    for (const ControlPoint &control : mesh.ControlPoints())
    {
        s_knots.push_back(control.parameter[kS]);
        t_knots.push_back(control.parameter[kT]);
    }
    std::sort(s_knots.begin(), s_knots.end());
    std::sort(t_knots.begin(), t_knots.end());
    s_knots.erase(std::unique(s_knots.begin(), s_knots.end()), s_knots.end());
    t_knots.erase(std::unique(t_knots.begin(), t_knots.end()), t_knots.end());
    if (s_knots.size() < 2 || t_knots.size() < 2)
    {
        throw std::invalid_argument("degree-1 T-spline needs at least one knot cell");
    }
    const TSplineSurface surface(mesh, 1);
    std::vector<TSplineLeaf> leaves;
    leaves.reserve((s_knots.size() - 1) * (t_knots.size() - 1));
    for (std::size_t i = 0; i + 1 < s_knots.size(); ++i)
    {
        for (std::size_t j = 0; j + 1 < t_knots.size(); ++j)
        {
            TSplineLeaf leaf;
            leaf.s_domain = {s_knots[i], s_knots[i + 1]};
            leaf.t_domain = {t_knots[j], t_knots[j + 1]};
            for (int iu = 0; iu < 2; ++iu)
            {
                for (int iv = 0; iv < 2; ++iv)
                {
                    const Point4 h = surface.EvaluateHomogeneous(leaf.s_domain[iu],
                                                                   leaf.t_domain[iv]);
                    if (std::abs(h[3]) <= std::numeric_limits<double>::epsilon())
                    {
                        throw std::out_of_range("T-spline leaf corner lies outside the PB domain");
                    }
                    leaf.weights[iu][iv] = h[3];
                    leaf.control_points[iu][iv] = {h[0] / h[3], h[1] / h[3], h[2] / h[3]};
                }
            }
            leaves.push_back(leaf);
        }
    }
    return leaves;
}

} // namespace tspline
} // namespace mfem_raytracing
