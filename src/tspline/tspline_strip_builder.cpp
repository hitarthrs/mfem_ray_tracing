#include "mfem_raytracing/tspline/tspline_strip_builder.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

struct SideFrame
{
    int normal_axis = 0;
    int tangent_axis = 1;
    double seam_value = 0.0;
    double normal_scale = 1.0;
    double tangent_lo = 0.0;
    double tangent_span = 1.0;
    bool reversed_tangent = false;

    std::array<double, 2> ToST(double u, double v) const
    {
        const double values[2] = {u, v};
        const double s = normal_scale * (values[normal_axis] - seam_value);
        double t = (values[tangent_axis] - tangent_lo) / tangent_span;
        if (reversed_tangent) { t = 1.0 - t; }
        return {s, t};
    }
};

ParameterRange RegionRange(const SeamRegion &region, int axis)
{
    return axis == 0 ? region.u_range : region.v_range;
}

SideFrame MakeFrame(const SeamRegion &region, bool left, bool reversed_tangent)
{
    const int normal_axis = BoundaryNormalAxis(region.side);
    const int tangent_axis = BoundaryTangentAxis(region.side);
    const ParameterRange normal = RegionRange(region, normal_axis);
    const ParameterRange tangent = RegionRange(region, tangent_axis);
    const bool high = BoundaryIsMaximum(region.side);
    const double seam = high ? normal.hi : normal.lo;
    const double far = high ? normal.lo : normal.hi;
    if (!(tangent.hi > tangent.lo) || std::abs(far - seam) <= 1e-14)
    {
        throw std::invalid_argument("cannot unfold a degenerate seam region");
    }
    SideFrame frame;
    frame.normal_axis = normal_axis;
    frame.tangent_axis = tangent_axis;
    frame.seam_value = seam;
    frame.normal_scale = (left ? -1.0 : 1.0) / (far - seam);
    frame.tangent_lo = tangent.lo;
    frame.tangent_span = tangent.hi - tangent.lo;
    frame.reversed_tangent = reversed_tangent;
    return frame;
}

void SortAndUnique(std::vector<double> &values, double tolerance)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end(), [tolerance](double a, double b) {
        return std::abs(a - b) <= tolerance;
    }), values.end());
}

std::size_t CanonicalIndex(const std::vector<double> &values, double value, double tolerance)
{
    const auto it = std::lower_bound(values.begin(), values.end(), value);
    std::size_t best = values.size();
    double distance = 2.0 * tolerance;
    for (const auto candidate : {it, it == values.begin() ? values.end() : std::prev(it)})
    {
        if (candidate == values.end()) { continue; }
        const double d = std::abs(*candidate - value);
        if (d < distance)
        {
            best = static_cast<std::size_t>(candidate - values.begin());
            distance = d;
        }
    }
    if (best == values.size())
    {
        throw std::runtime_error("could not locate a local T-mesh parameter breakpoint");
    }
    return best;
}

std::pair<std::size_t, std::size_t> OrderedEdge(std::size_t a, std::size_t b)
{
    return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
}

} // namespace

LocalDegreeOneStrip BuildLocalDegreeOneStrip(const AverageMergedSeam &merged,
                                             const LocalStripBuildOptions &options)
{
    if (options.parameter_tolerance <= 0.0 || options.geometry_tolerance < 0.0)
    {
        throw std::invalid_argument("local-strip tolerances must be positive/non-negative");
    }
    const SideFrame frame_a = MakeFrame(merged.region_a, true, false);
    const SideFrame frame_b = MakeFrame(merged.region_b, false, merged.b_reversed);

    struct RegionInput
    {
        StripRegion label;
        const SeamRegion *region;
        const SideFrame *frame;
    };
    const std::array<RegionInput, 2> regions = {{
        {StripRegion::A, &merged.region_a, &frame_a},
        {StripRegion::B, &merged.region_b, &frame_b},
    }};

    std::vector<double> s_raw = {0.0};
    std::vector<double> t_raw = {0.0, 1.0};
    for (const RegionInput &input : regions)
    {
        for (const SeamRegionLeaf &entry : input.region->leaves)
        {
            const LeafPatch &leaf = entry.leaf;
            for (const double u : {leaf.u_domain_global[0], leaf.u_domain_global[1]})
            {
                for (const double v : {leaf.v_domain_global[0], leaf.v_domain_global[1]})
                {
                    const auto st = input.frame->ToST(u, v);
                    s_raw.push_back(st[0]);
                    t_raw.push_back(st[1]);
                }
            }
        }
    }
    SortAndUnique(s_raw, options.parameter_tolerance);
    SortAndUnique(t_raw, options.parameter_tolerance);
    if (s_raw.size() < 2 || t_raw.size() < 2)
    {
        throw std::runtime_error("local T-mesh needs at least one cell in each direction");
    }

    std::map<std::pair<std::size_t, std::size_t>, std::size_t> vertex_by_key;
    std::vector<HomogeneousPoint> vertex_homogeneous;
    std::map<std::size_t, std::set<std::size_t>> vertices_by_s;
    std::map<std::size_t, std::set<std::size_t>> vertices_by_t;
    std::map<std::size_t, std::vector<std::pair<std::size_t, std::size_t>>> horizontal_segments;
    std::map<std::size_t, std::vector<std::pair<std::size_t, std::size_t>>> vertical_segments;
    struct FaceIndexData
    {
        LocalTSplineFace face;
        std::array<std::size_t, 4> rect_indices;
    };
    std::vector<FaceIndexData> faces;
    double max_point_delta = 0.0;
    double max_weight_delta = 0.0;

    const auto add_vertex = [&](std::size_t s, std::size_t t, const HomogeneousPoint &point) {
        const auto key = std::make_pair(s, t);
        const auto found = vertex_by_key.find(key);
        if (found != vertex_by_key.end())
        {
            const HomogeneousPoint &old = vertex_homogeneous[found->second];
            max_point_delta = std::max(max_point_delta, HomogeneousCartesianDistance(old, point));
            max_weight_delta = std::max(max_weight_delta, std::abs(old[3] - point[3]));
            return found->second;
        }
        const std::size_t id = vertex_homogeneous.size();
        vertex_by_key.emplace(key, id);
        vertex_homogeneous.push_back(point);
        vertices_by_s[s].insert(t);
        vertices_by_t[t].insert(s);
        return id;
    };

    for (const RegionInput &input : regions)
    {
        for (const SeamRegionLeaf &entry : input.region->leaves)
        {
            const LeafPatch &leaf = entry.leaf;
            const HomogeneousBilinearNet net = HomogenizeBilinearLeaf(leaf);
            std::array<std::size_t, 4> ids{};
            std::set<std::size_t> s_indices;
            std::set<std::size_t> t_indices;
            for (int corner = 0; corner < 4; ++corner)
            {
                const int iu = corner >= 2 ? 1 : 0;
                const int iv = corner % 2;
                const auto st = input.frame->ToST(leaf.u_domain_global[iu], leaf.v_domain_global[iv]);
                const std::size_t s = CanonicalIndex(s_raw, st[0], options.parameter_tolerance);
                const std::size_t t = CanonicalIndex(t_raw, st[1], options.parameter_tolerance);
                ids[corner] = add_vertex(s, t, net[corner]);
                s_indices.insert(s);
                t_indices.insert(t);
            }
            if (s_indices.size() != 2 || t_indices.size() != 2)
            {
                throw std::runtime_error("source leaf does not map to a non-degenerate local rectangle");
            }
            const std::size_t s0 = *s_indices.begin();
            const std::size_t s1 = *s_indices.rbegin();
            const std::size_t t0 = *t_indices.begin();
            const std::size_t t1 = *t_indices.rbegin();
            horizontal_segments[t0].push_back({s0, s1});
            horizontal_segments[t1].push_back({s0, s1});
            vertical_segments[s0].push_back({t0, t1});
            vertical_segments[s1].push_back({t0, t1});
            faces.push_back({{input.label, input.region->patch_id, entry.source_leaf_index,
                              leaf.total_error,
                              {vertex_by_key[{s0, t0}], vertex_by_key[{s1, t0}],
                               vertex_by_key[{s1, t1}], vertex_by_key[{s0, t1}]},
                              {s_raw[s0], s_raw[s1], t_raw[t0], t_raw[t1]}},
                             {s0, s1, t0, t1}});
        }
    }
    if (max_point_delta > options.geometry_tolerance ||
        max_weight_delta > options.geometry_tolerance)
    {
        throw std::runtime_error("coincident local T-mesh control vertices disagree");
    }

    std::size_t rule2_added = 0;
    for (const FaceIndexData &data : faces)
    {
        const auto [s0, s1, t0, t1] = data.rect_indices;
        for (const auto &[s, values] : vertices_by_s)
        {
            if (s0 < s && s < s1 && values.count(t0) != 0 && values.count(t1) != 0)
            {
                vertical_segments[s].push_back({t0, t1});
                ++rule2_added;
            }
        }
        for (const auto &[t, values] : vertices_by_t)
        {
            if (t0 < t && t < t1 && values.count(s0) != 0 && values.count(s1) != 0)
            {
                horizontal_segments[t].push_back({s0, s1});
                ++rule2_added;
            }
        }
    }

    std::set<std::pair<std::size_t, std::size_t>> edges;
    for (const auto &[t, segments] : horizontal_segments)
    {
        const std::set<std::size_t> &vertices = vertices_by_t.at(t);
        for (const auto &[lo, hi] : segments)
        {
            std::vector<std::size_t> inside;
            for (const std::size_t s : vertices)
            {
                if (lo <= s && s <= hi) { inside.push_back(s); }
            }
            for (std::size_t i = 1; i < inside.size(); ++i)
            {
                edges.insert(OrderedEdge(vertex_by_key.at({inside[i - 1], t}),
                                         vertex_by_key.at({inside[i], t})));
            }
        }
    }
    for (const auto &[s, segments] : vertical_segments)
    {
        const std::set<std::size_t> &vertices = vertices_by_s.at(s);
        for (const auto &[lo, hi] : segments)
        {
            std::vector<std::size_t> inside;
            for (const std::size_t t : vertices)
            {
                if (lo <= t && t <= hi) { inside.push_back(t); }
            }
            for (std::size_t i = 1; i < inside.size(); ++i)
            {
                edges.insert(OrderedEdge(vertex_by_key.at({s, inside[i - 1]}),
                                         vertex_by_key.at({s, inside[i]})));
            }
        }
    }

    LocalDegreeOneStrip result;
    result.s_breaks = std::move(s_raw);
    result.t_breaks = std::move(t_raw);
    result.rule2_added_edges = rule2_added;
    result.max_vertex_disagreement = max_point_delta;
    result.max_vertex_weight_disagreement = max_weight_delta;
    TMesh mesh;
    std::vector<std::pair<std::size_t, std::size_t>> key_by_id(vertex_homogeneous.size());
    for (const auto &[key, id] : vertex_by_key) { key_by_id[id] = key; }
    for (std::size_t id = 0; id < vertex_homogeneous.size(); ++id)
    {
        const auto xyz = ProjectHomogeneous(vertex_homogeneous[id]);
        const auto [s, t] = key_by_id[id];
        mesh.AddControlPoint({result.s_breaks[s], result.t_breaks[t]}, xyz,
                             vertex_homogeneous[id][3]);
    }
    for (const auto &[first, second] : edges) { mesh.AddEdge(first, second); }
    mesh.Validate(options.parameter_tolerance);
    result.mesh = std::move(mesh);
    result.t_junctions = result.mesh.TJunctions(options.parameter_tolerance);
    result.faces.reserve(faces.size());
    for (const FaceIndexData &data : faces) { result.faces.push_back(data.face); }
    return result;
}

} // namespace tspline
} // namespace mfem_raytracing
