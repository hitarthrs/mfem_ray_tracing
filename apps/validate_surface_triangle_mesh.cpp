// Sampled two-sided distance validation between one tensor-product NURBS
// surface and an OBJ triangle mesh.
//
// The tool deliberately keeps OCCT out of the C++ dependency graph: OCCT makes
// the mesh, while this native executable performs the calibration's expensive
// nearest-point work using the repository's SurfaceData loader.  Results are
// sampled estimates, not a formal Hausdorff certificate.
//
// Usage:
//   validate_surface_triangle_mesh --input surface.json --obj mesh.obj \
//       [--surface-grid 101] [--seed-grid 129] [--json result.json]

#include "mfem_raytracing/embree/leaf_patch_loader.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using mfem_raytracing::SurfaceData;

struct Vec3
{
    double x = 0.0, y = 0.0, z = 0.0;
    Vec3 &operator+=(const Vec3 &b) { x += b.x; y += b.y; z += b.z; return *this; }
};

Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(double s, const Vec3 &a) { return {s * a.x, s * a.y, s * a.z}; }
Vec3 operator*(const Vec3 &a, double s) { return s * a; }
double Dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 Cross(const Vec3 &a, const Vec3 &b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double NormSq(const Vec3 &a) { return Dot(a, a); }

struct Bounds
{
    Vec3 lo{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
    Vec3 hi{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()};

    void Extend(const Vec3 &p)
    {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    void Extend(const Bounds &b) { Extend(b.lo); Extend(b.hi); }
};

double DistanceSq(const Bounds &b, const Vec3 &p)
{
    const auto axis = [](double value, double lo, double hi) {
        return value < lo ? lo - value : (value > hi ? value - hi : 0.0);
    };
    const double x = axis(p.x, b.lo.x, b.hi.x);
    const double y = axis(p.y, b.lo.y, b.hi.y);
    const double z = axis(p.z, b.lo.z, b.hi.z);
    return x * x + y * y + z * z;
}

struct Triangle
{
    Vec3 a, b, c;
    Bounds bounds;
    Vec3 centroid;
};

double PointTriangleDistanceSq(const Vec3 &p, const Triangle &t)
{
    // Real-Time Collision Detection, Christer Ericson, section 5.1.5.
    const Vec3 ab = t.b - t.a;
    const Vec3 ac = t.c - t.a;
    const Vec3 ap = p - t.a;
    const double d1 = Dot(ab, ap), d2 = Dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return NormSq(ap); }
    const Vec3 bp = p - t.b;
    const double d3 = Dot(ab, bp), d4 = Dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return NormSq(bp); }
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        const double v = d1 / (d1 - d3);
        return NormSq(p - (t.a + v * ab));
    }
    const Vec3 cp = p - t.c;
    const double d5 = Dot(ab, cp), d6 = Dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return NormSq(cp); }
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        const double w = d2 / (d2 - d6);
        return NormSq(p - (t.a + w * ac));
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        const Vec3 bc = t.c - t.b;
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return NormSq(p - (t.b + w * bc));
    }
    const Vec3 normal = Cross(ab, ac);
    const double plane_distance = Dot(ap, normal);
    return plane_distance * plane_distance / std::max(NormSq(normal), 1e-300);
}

class TriangleBvh
{
public:
    explicit TriangleBvh(std::vector<Triangle> triangles) : triangles_(std::move(triangles))
    {
        if (triangles_.empty()) { throw std::runtime_error("OBJ contains no nondegenerate triangles"); }
        Build(0, static_cast<int>(triangles_.size()));
    }

    double NearestDistance(const Vec3 &point) const
    {
        double best = std::numeric_limits<double>::infinity();
        Query(0, point, best);
        return std::sqrt(best);
    }
    std::size_t TriangleCount() const { return triangles_.size(); }
    const std::vector<Triangle> &Triangles() const { return triangles_; }

private:
    struct Node { Bounds bounds; int left = -1, right = -1, first = 0, count = 0; };
    std::vector<Triangle> triangles_;
    std::vector<Node> nodes_;

    int Build(int first, int end)
    {
        const int node_id = static_cast<int>(nodes_.size());
        nodes_.push_back({});
        Bounds bounds, centroid_bounds;
        for (int i = first; i < end; ++i)
        {
            bounds.Extend(triangles_[static_cast<std::size_t>(i)].bounds);
            centroid_bounds.Extend(triangles_[static_cast<std::size_t>(i)].centroid);
        }
        Node node; node.bounds = bounds; node.first = first; node.count = end - first;
        if (node.count > 8)
        {
            const Vec3 extent = centroid_bounds.hi - centroid_bounds.lo;
            const int axis = extent.y > extent.x ? (extent.z > extent.y ? 2 : 1)
                                                : (extent.z > extent.x ? 2 : 0);
            const int mid = first + node.count / 2;
            auto coordinate = [axis](const Triangle &t) {
                return axis == 0 ? t.centroid.x : (axis == 1 ? t.centroid.y : t.centroid.z);
            };
            std::nth_element(triangles_.begin() + first, triangles_.begin() + mid,
                             triangles_.begin() + end,
                             [&](const Triangle &a, const Triangle &b) { return coordinate(a) < coordinate(b); });
            node.left = Build(first, mid);
            node.right = Build(mid, end);
            node.count = 0;
        }
        nodes_[static_cast<std::size_t>(node_id)] = node;
        return node_id;
    }

    void Query(int node_id, const Vec3 &point, double &best) const
    {
        const Node &node = nodes_[static_cast<std::size_t>(node_id)];
        if (DistanceSq(node.bounds, point) >= best) { return; }
        if (node.count)
        {
            for (int i = node.first; i < node.first + node.count; ++i)
            {
                best = std::min(best, PointTriangleDistanceSq(point, triangles_[static_cast<std::size_t>(i)]));
            }
            return;
        }
        const Node &left = nodes_[static_cast<std::size_t>(node.left)];
        const Node &right = nodes_[static_cast<std::size_t>(node.right)];
        if (DistanceSq(left.bounds, point) < DistanceSq(right.bounds, point))
        {
            Query(node.left, point, best); Query(node.right, point, best);
        }
        else
        {
            Query(node.right, point, best); Query(node.left, point, best);
        }
    }
};

int FindSpan(const std::vector<double> &knots, int degree, int n, double parameter)
{
    if (parameter >= knots[static_cast<std::size_t>(n + 1)]) { return n; }
    if (parameter <= knots[static_cast<std::size_t>(degree)]) { return degree; }
    int low = degree, high = n + 1, mid = (low + high) / 2;
    while (parameter < knots[static_cast<std::size_t>(mid)] ||
           parameter >= knots[static_cast<std::size_t>(mid + 1)])
    {
        if (parameter < knots[static_cast<std::size_t>(mid)]) { high = mid; }
        else { low = mid; }
        mid = (low + high) / 2;
    }
    return mid;
}

std::vector<double> BasisFuns(const std::vector<double> &knots, int degree, int span, double u)
{
    std::vector<double> basis(static_cast<std::size_t>(degree + 1), 0.0);
    std::vector<double> left(static_cast<std::size_t>(degree + 1), 0.0);
    std::vector<double> right(static_cast<std::size_t>(degree + 1), 0.0);
    basis[0] = 1.0;
    for (int j = 1; j <= degree; ++j)
    {
        left[static_cast<std::size_t>(j)] = u - knots[static_cast<std::size_t>(span + 1 - j)];
        right[static_cast<std::size_t>(j)] = knots[static_cast<std::size_t>(span + j)] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r)
        {
            const double denominator = right[static_cast<std::size_t>(r + 1)] + left[static_cast<std::size_t>(j - r)];
            const double temporary = denominator == 0.0 ? 0.0 : basis[static_cast<std::size_t>(r)] / denominator;
            basis[static_cast<std::size_t>(r)] = saved + right[static_cast<std::size_t>(r + 1)] * temporary;
            saved = left[static_cast<std::size_t>(j - r)] * temporary;
        }
        basis[static_cast<std::size_t>(j)] = saved;
    }
    return basis;
}

class NurbsEvaluator
{
public:
    explicit NurbsEvaluator(const SurfaceData &surface) : surface_(surface)
    {
        nu_ = surface_.NumControlPointsU();
        nv_ = surface_.NumControlPointsV();
        if (nu_ <= surface_.degree_u || nv_ <= surface_.degree_v)
        {
            throw std::runtime_error("surface has too few control points for its degree");
        }
    }

    Vec3 Evaluate(double u, double v) const
    {
        u = std::clamp(u, surface_.u_domain.first, surface_.u_domain.second);
        v = std::clamp(v, surface_.v_domain.first, surface_.v_domain.second);
        const int uspan = FindSpan(surface_.knotvector_u, surface_.degree_u, nu_ - 1, u);
        const int vspan = FindSpan(surface_.knotvector_v, surface_.degree_v, nv_ - 1, v);
        const std::vector<double> ubasis = BasisFuns(surface_.knotvector_u, surface_.degree_u, uspan, u);
        const std::vector<double> vbasis = BasisFuns(surface_.knotvector_v, surface_.degree_v, vspan, v);
        Vec3 numerator; double denominator = 0.0;
        for (int i = 0; i <= surface_.degree_u; ++i)
        {
            const int ci = uspan - surface_.degree_u + i;
            for (int j = 0; j <= surface_.degree_v; ++j)
            {
                const int cj = vspan - surface_.degree_v + j;
                const double weight = surface_.weights.empty() ? 1.0 : surface_.weights[static_cast<std::size_t>(ci)][static_cast<std::size_t>(cj)];
                const double coefficient = ubasis[static_cast<std::size_t>(i)] * vbasis[static_cast<std::size_t>(j)] * weight;
                const auto &p = surface_.control_points[static_cast<std::size_t>(ci)][static_cast<std::size_t>(cj)];
                numerator += coefficient * Vec3{p[0], p[1], p[2]};
                denominator += coefficient;
            }
        }
        return (1.0 / denominator) * numerator;
    }

    Vec3 Project(const Vec3 &point, double u, double v) const
    {
        const double hu = std::max((surface_.u_domain.second - surface_.u_domain.first) * 1e-5, 1e-8);
        const double hv = std::max((surface_.v_domain.second - surface_.v_domain.first) * 1e-5, 1e-8);
        for (int iteration = 0; iteration < 16; ++iteration)
        {
            const Vec3 value = Evaluate(u, v);
            const double ulo = std::max(surface_.u_domain.first, u - hu);
            const double uhi = std::min(surface_.u_domain.second, u + hu);
            const double vlo = std::max(surface_.v_domain.first, v - hv);
            const double vhi = std::min(surface_.v_domain.second, v + hv);
            const Vec3 du = (1.0 / (uhi - ulo)) * (Evaluate(uhi, v) - Evaluate(ulo, v));
            const Vec3 dv = (1.0 / (vhi - vlo)) * (Evaluate(u, vhi) - Evaluate(u, vlo));
            const Vec3 residual = value - point;
            const double a = Dot(du, du), b = Dot(du, dv), c = Dot(dv, dv);
            const double rhs_u = Dot(du, residual), rhs_v = Dot(dv, residual);
            const double determinant = a * c - b * b;
            if (determinant < 1e-24) { break; }
            const double next_u = std::clamp(u + (-c * rhs_u + b * rhs_v) / determinant,
                                             surface_.u_domain.first, surface_.u_domain.second);
            const double next_v = std::clamp(v + (b * rhs_u - a * rhs_v) / determinant,
                                             surface_.v_domain.first, surface_.v_domain.second);
            if (std::abs(next_u - u) + std::abs(next_v - v) < 1e-12) { break; }
            u = next_u; v = next_v;
        }
        return Evaluate(u, v);
    }

    const SurfaceData &surface() const { return surface_; }

private:
    const SurfaceData &surface_;
    int nu_ = 0, nv_ = 0;
};

class PointKdTree
{
public:
    explicit PointKdTree(std::vector<Vec3> points) : points_(std::move(points))
    {
        indices_.resize(points_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
        if (points_.empty()) { throw std::runtime_error("empty NURBS seed grid"); }
        Build(0, static_cast<int>(indices_.size()));
    }

    int Nearest(const Vec3 &point) const
    {
        int best = 0; double best_sq = std::numeric_limits<double>::infinity();
        Query(0, point, best, best_sq);
        return best;
    }

private:
    struct Node { int point = -1, axis = 0, left = -1, right = -1; };
    std::vector<Vec3> points_;
    std::vector<int> indices_;
    std::vector<Node> nodes_;

    int Build(int first, int end)
    {
        Bounds bounds;
        for (int i = first; i < end; ++i) { bounds.Extend(points_[static_cast<std::size_t>(indices_[static_cast<std::size_t>(i)])]); }
        const Vec3 extent = bounds.hi - bounds.lo;
        const int axis = extent.y > extent.x ? (extent.z > extent.y ? 2 : 1)
                                            : (extent.z > extent.x ? 2 : 0);
        const int mid = first + (end - first) / 2;
        auto coordinate = [&](int index) {
            const Vec3 &p = points_[static_cast<std::size_t>(index)];
            return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
        };
        std::nth_element(indices_.begin() + first, indices_.begin() + mid, indices_.begin() + end,
                         [&](int a, int b) { return coordinate(a) < coordinate(b); });
        const int id = static_cast<int>(nodes_.size());
        nodes_.push_back({indices_[static_cast<std::size_t>(mid)], axis, -1, -1});
        if (first < mid) { nodes_[static_cast<std::size_t>(id)].left = Build(first, mid); }
        if (mid + 1 < end) { nodes_[static_cast<std::size_t>(id)].right = Build(mid + 1, end); }
        return id;
    }

    void Query(int node_id, const Vec3 &point, int &best, double &best_sq) const
    {
        const Node &node = nodes_[static_cast<std::size_t>(node_id)];
        const Vec3 &candidate = points_[static_cast<std::size_t>(node.point)];
        const double candidate_sq = NormSq(point - candidate);
        if (candidate_sq < best_sq)
        {
            best_sq = candidate_sq;
            best = node.point;
        }
        const double delta = node.axis == 0 ? point.x - candidate.x : (node.axis == 1 ? point.y - candidate.y : point.z - candidate.z);
        const int first = delta < 0.0 ? node.left : node.right;
        const int second = delta < 0.0 ? node.right : node.left;
        if (first >= 0) { Query(first, point, best, best_sq); }
        if (second >= 0 && delta * delta < best_sq) { Query(second, point, best, best_sq); }
    }
};

std::vector<Triangle> LoadObj(const std::string &path)
{
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("cannot open OBJ '" + path + "'"); }
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
    std::string line;
    while (std::getline(input, line))
    {
        std::istringstream stream(line); std::string kind; stream >> kind;
        if (kind == "v")
        {
            Vec3 p; if (!(stream >> p.x >> p.y >> p.z)) { throw std::runtime_error("malformed OBJ vertex"); }
            vertices.push_back(p);
        }
        else if (kind == "f")
        {
            std::vector<int> indices; std::string token;
            while (stream >> token)
            {
                const std::size_t slash = token.find('/');
                const int raw = std::stoi(token.substr(0, slash));
                const int index = raw < 0 ? static_cast<int>(vertices.size()) + raw : raw - 1;
                if (index < 0 || index >= static_cast<int>(vertices.size())) { throw std::runtime_error("OBJ face index out of range"); }
                indices.push_back(index);
            }
            if (indices.size() < 3) { throw std::runtime_error("OBJ face has fewer than three vertices"); }
            for (std::size_t i = 1; i + 1 < indices.size(); ++i)
            {
                Triangle triangle{vertices[static_cast<std::size_t>(indices[0])], vertices[static_cast<std::size_t>(indices[i])], vertices[static_cast<std::size_t>(indices[i + 1])]};
                triangle.bounds.Extend(triangle.a); triangle.bounds.Extend(triangle.b); triangle.bounds.Extend(triangle.c);
                triangle.centroid = (1.0 / 3.0) * (triangle.a + triangle.b + triangle.c);
                if (NormSq(Cross(triangle.b - triangle.a, triangle.c - triangle.a)) > 1e-28) { triangles.push_back(triangle); }
            }
        }
    }
    return triangles;
}

double Lerp(double lo, double hi, int index, int count)
{
    return lo + (hi - lo) * static_cast<double>(index) / static_cast<double>(count - 1);
}

struct Result
{
    std::size_t triangles = 0;
    double surface_to_mesh = 0.0;
    double mesh_to_surface = 0.0;
    double surface_to_mesh_seconds = 0.0;
    double mesh_to_surface_seconds = 0.0;
};

Result Validate(const NurbsEvaluator &surface, TriangleBvh &mesh, int surface_grid, int seed_grid)
{
    const auto &data = surface.surface();
    Result result; result.triangles = mesh.TriangleCount();
    const auto surface_start = std::chrono::steady_clock::now();
    for (int i = 0; i < surface_grid; ++i)
    {
        const double u = Lerp(data.u_domain.first, data.u_domain.second, i, surface_grid);
        for (int j = 0; j < surface_grid; ++j)
        {
            const double v = Lerp(data.v_domain.first, data.v_domain.second, j, surface_grid);
            result.surface_to_mesh = std::max(result.surface_to_mesh, mesh.NearestDistance(surface.Evaluate(u, v)));
        }
    }
    result.surface_to_mesh_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - surface_start).count();

    std::vector<Vec3> seed_points; std::vector<std::array<double, 2>> seed_uv;
    seed_points.reserve(static_cast<std::size_t>(seed_grid) * seed_grid);
    seed_uv.reserve(static_cast<std::size_t>(seed_grid) * seed_grid);
    for (int i = 0; i < seed_grid; ++i)
    {
        const double u = Lerp(data.u_domain.first, data.u_domain.second, i, seed_grid);
        for (int j = 0; j < seed_grid; ++j)
        {
            const double v = Lerp(data.v_domain.first, data.v_domain.second, j, seed_grid);
            seed_uv.push_back({u, v}); seed_points.push_back(surface.Evaluate(u, v));
        }
    }
    const PointKdTree seeds(std::move(seed_points));
    const auto mesh_start = std::chrono::steady_clock::now();
    // Four interior samples are sufficient to catch the usual chord sag; use a
    // denser barycentric stencil if this becomes a published accuracy claim.
    const std::array<std::array<double, 3>, 4> bary = {{{0.5, 0.5, 0.0}, {0.5, 0.0, 0.5},
                                                         {0.0, 0.5, 0.5}, {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}}};
    for (const Triangle &triangle : mesh.Triangles())
    {
        for (const auto &w : bary)
        {
            const Vec3 point = w[0] * triangle.a + w[1] * triangle.b + w[2] * triangle.c;
            const int seed = seeds.Nearest(point);
            const auto &uv = seed_uv[static_cast<std::size_t>(seed)];
            result.mesh_to_surface = std::max(result.mesh_to_surface,
                                              std::sqrt(NormSq(point - surface.Project(point, uv[0], uv[1]))));
        }
    }
    result.mesh_to_surface_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - mesh_start).count();
    return result;
}

void PrintUsage()
{
    std::cerr << "usage: validate_surface_triangle_mesh --input surface.json --obj mesh.obj [--surface-grid N] [--seed-grid N] [--json result.json]\n";
}

} // namespace

int main(int argc, char **argv)
{
    std::string input_path, obj_path, json_path;
    int surface_grid = 101, seed_grid = 129;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&](const char *flag) -> std::string {
            if (i + 1 == argc) { throw std::runtime_error(std::string("missing value for ") + flag); }
            return argv[++i];
        };
        if (arg == "--input") { input_path = next("--input"); }
        else if (arg == "--obj") { obj_path = next("--obj"); }
        else if (arg == "--surface-grid") { surface_grid = std::stoi(next("--surface-grid")); }
        else if (arg == "--seed-grid") { seed_grid = std::stoi(next("--seed-grid")); }
        else if (arg == "--json") { json_path = next("--json"); }
        else if (arg == "-h" || arg == "--help") { PrintUsage(); return 0; }
        else { throw std::runtime_error("unknown argument '" + arg + "'"); }
    }
    try
    {
        if (input_path.empty() || obj_path.empty() || surface_grid < 2 || seed_grid < 2)
        {
            PrintUsage(); return 1;
        }
        const SurfaceData source_surface = mfem_raytracing::LoadSurfaceDataJson(input_path);
        const NurbsEvaluator surface(source_surface);
        TriangleBvh mesh(LoadObj(obj_path));
        const Result result = Validate(surface, mesh, surface_grid, seed_grid);
        std::cout << std::setprecision(12)
                  << "triangles: " << result.triangles << "\n"
                  << "surface_to_mesh_sample_max_error: " << result.surface_to_mesh << "\n"
                  << "mesh_to_surface_triangle_sample_max_error: " << result.mesh_to_surface << "\n"
                  << "surface_to_mesh_seconds: " << result.surface_to_mesh_seconds << "\n"
                  << "mesh_to_surface_seconds: " << result.mesh_to_surface_seconds << "\n";
        if (!json_path.empty())
        {
            std::ofstream out(json_path);
            if (!out) { throw std::runtime_error("cannot write '" + json_path + "'"); }
            out << std::setprecision(17)
                << "{\n  \"triangles\": " << result.triangles
                << ",\n  \"surface_to_mesh_sample_max_error\": " << result.surface_to_mesh
                << ",\n  \"surface_grid_samples_per_axis\": " << surface_grid
                << ",\n  \"mesh_to_surface_triangle_sample_max_error\": " << result.mesh_to_surface
                << ",\n  \"mesh_to_surface_samples_per_triangle\": 4"
                << ",\n  \"seed_grid_samples_per_axis\": " << seed_grid
                << ",\n  \"surface_to_mesh_seconds\": " << result.surface_to_mesh_seconds
                << ",\n  \"mesh_to_surface_seconds\": " << result.mesh_to_surface_seconds << "\n}\n";
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
