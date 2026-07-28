#include "tspline_shell_watertightness.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

struct Point
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Point PointAt(const LeafPatch &leaf, int corner)
{
    return {leaf.patch.control_points[corner][0], leaf.patch.control_points[corner][1],
            leaf.patch.control_points[corner][2]};
}

Point Subtract(const Point &a, const Point &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Point Add(const Point &a, const Point &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Point Scale(const Point &a, double s)
{
    return {a.x * s, a.y * s, a.z * s};
}

double Dot(const Point &a, const Point &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double Norm(const Point &a)
{
    return std::sqrt(Dot(a, a));
}

struct LineKey
{
    std::array<long long, 6> values = {};
    bool operator<(const LineKey &other) const { return values < other.values; }
};

struct EdgeRecord
{
    Point start;
    Point direction;
    Point moment;
    double lo = 0.0;
    double hi = 0.0;
    std::size_t final_leaf_index = 0;
    SourceLeafRef source;
    std::string owner_id;
};

long long Quantize(double value, double step)
{
    if (!(step > 0.0) || !std::isfinite(value))
    {
        throw std::invalid_argument("invalid watertightness quantization input");
    }
    return static_cast<long long>(std::llround(value / step));
}

bool CanonicalSign(const Point &direction, double tolerance)
{
    if (std::abs(direction.x) > tolerance) { return direction.x > 0.0; }
    if (std::abs(direction.y) > tolerance) { return direction.y > 0.0; }
    return direction.z >= 0.0;
}

void AddIssue(WatertightnessReport &report, const WatertightnessOptions &options,
              WatertightnessIssue issue)
{
    if (report.issues.size() < options.max_reported_issues)
    {
        report.issues.push_back(std::move(issue));
    }
}

} // namespace

const char *WatertightnessIssueKindName(WatertightnessIssueKind kind)
{
    switch (kind)
    {
        case WatertightnessIssueKind::MissingSourceOwner: return "missing-source-owner";
        case WatertightnessIssueKind::MultipleSourceOwners: return "multiple-source-owners";
        case WatertightnessIssueKind::DegenerateEdge: return "degenerate-edge";
        case WatertightnessIssueKind::OpenEdgeSpan: return "open-edge-span";
        case WatertightnessIssueKind::NonManifoldEdgeSpan: return "nonmanifold-edge-span";
        case WatertightnessIssueKind::GeometricMismatch: return "geometric-mismatch";
    }
    return "unknown";
}

WatertightnessReport CheckShellWatertightness(
    const std::vector<WatertightLeaf> &leaves,
    const std::vector<SourceLeafRef> &expected_sources,
    const WatertightnessOptions &options)
{
    if (options.relative_tolerance < 0.0 || options.absolute_tolerance < 0.0)
    {
        throw std::invalid_argument("watertightness tolerances must be non-negative");
    }
    WatertightnessReport report;
    report.expected_source_leaf_count = expected_sources.size();
    if (leaves.empty())
    {
        // Preserve useful provenance diagnostics even when the construction
        // failed before it emitted a single RT leaf.
        for (const SourceLeafRef &source : expected_sources)
        {
            ++report.unowned_source_leaf_count;
            AddIssue(report, options,
                     {WatertightnessIssueKind::MissingSourceOwner, source, {}, 0, 0.0});
        }
        report.watertight = expected_sources.empty();
        return report;
    }

    Point minimum = PointAt(leaves.front().leaf, 0);
    Point maximum = minimum;
    double max_abs_coordinate = std::max({std::abs(minimum.x), std::abs(minimum.y), std::abs(minimum.z)});
    for (const WatertightLeaf &entry : leaves)
    {
        for (int corner = 0; corner < 4; ++corner)
        {
            const Point point = PointAt(entry.leaf, corner);
            minimum.x = std::min(minimum.x, point.x); minimum.y = std::min(minimum.y, point.y); minimum.z = std::min(minimum.z, point.z);
            maximum.x = std::max(maximum.x, point.x); maximum.y = std::max(maximum.y, point.y); maximum.z = std::max(maximum.z, point.z);
            max_abs_coordinate = std::max(max_abs_coordinate,
                                          std::max({std::abs(point.x), std::abs(point.y), std::abs(point.z)}));
        }
    }
    const double diameter = Norm(Subtract(maximum, minimum));
    report.geometry_tolerance = std::max(
        {options.absolute_tolerance,
         options.relative_tolerance * std::max(1.0, diameter),
         128.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, max_abs_coordinate)});
    const double direction_tolerance = report.geometry_tolerance / std::max(1.0, diameter);

    std::map<SourceLeafRef, std::set<std::string>> owners;
    for (const WatertightLeaf &entry : leaves)
    {
        if (!entry.owner_id.empty()) { owners[entry.source].insert(entry.owner_id); }
    }
    for (const SourceLeafRef &source : expected_sources)
    {
        const auto found = owners.find(source);
        if (found == owners.end() || found->second.empty())
        {
            ++report.unowned_source_leaf_count;
            AddIssue(report, options, {WatertightnessIssueKind::MissingSourceOwner, source, {}, 0, 0.0});
        }
        else if (found->second.size() != 1)
        {
            ++report.multiply_owned_source_leaf_count;
            AddIssue(report, options, {WatertightnessIssueKind::MultipleSourceOwners, source,
                                       *found->second.begin(), 0, 0.0});
        }
    }

    std::map<LineKey, std::vector<EdgeRecord>> lines;
    constexpr std::array<std::array<int, 2>, 4> kEdges = {{{{0, 2}}, {{2, 3}}, {{3, 1}}, {{1, 0}}}};
    for (const WatertightLeaf &entry : leaves)
    {
        for (const auto &edge : kEdges)
        {
            Point start = PointAt(entry.leaf, edge[0]);
            Point end = PointAt(entry.leaf, edge[1]);
            Point direction = Subtract(end, start);
            const double length = Norm(direction);
            if (length <= report.geometry_tolerance)
            {
                ++report.degenerate_edge_count;
                AddIssue(report, options, {WatertightnessIssueKind::DegenerateEdge, entry.source,
                                           entry.owner_id, entry.final_leaf_index, length});
                continue;
            }
            direction = Scale(direction, 1.0 / length);
            if (!CanonicalSign(direction, direction_tolerance)) { direction = Scale(direction, -1.0); }
            const Point moment = Subtract(start, Scale(direction, Dot(start, direction)));
            LineKey key;
            key.values = {Quantize(direction.x, direction_tolerance),
                          Quantize(direction.y, direction_tolerance),
                          Quantize(direction.z, direction_tolerance),
                          Quantize(moment.x, report.geometry_tolerance),
                          Quantize(moment.y, report.geometry_tolerance),
                          Quantize(moment.z, report.geometry_tolerance)};
            const double first = Dot(start, direction);
            const double second = Dot(end, direction);
            lines[key].push_back({start, direction, moment, std::min(first, second), std::max(first, second),
                                  entry.final_leaf_index, entry.source, entry.owner_id});
        }
    }

    for (const auto &[key, edges] : lines)
    {
        (void)key;
        std::vector<double> breaks;
        for (const EdgeRecord &edge : edges)
        {
            breaks.push_back(edge.lo);
            breaks.push_back(edge.hi);
        }
        std::sort(breaks.begin(), breaks.end());
        breaks.erase(std::unique(breaks.begin(), breaks.end(), [&report](double a, double b) {
            return std::abs(a - b) <= report.geometry_tolerance;
        }), breaks.end());
        for (std::size_t i = 0; i + 1 < breaks.size(); ++i)
        {
            const double lo = breaks[i];
            const double hi = breaks[i + 1];
            if (hi - lo <= report.geometry_tolerance) { continue; }
            const double mid = 0.5 * (lo + hi);
            std::vector<const EdgeRecord *> covering;
            for (const EdgeRecord &edge : edges)
            {
                if (edge.lo <= mid + report.geometry_tolerance &&
                    mid <= edge.hi + report.geometry_tolerance)
                {
                    covering.push_back(&edge);
                }
            }
            ++report.atomic_edge_span_count;
            if (covering.size() == 1)
            {
                ++report.open_edge_span_count;
                if (options.require_closed)
                {
                    AddIssue(report, options, {WatertightnessIssueKind::OpenEdgeSpan,
                                               covering.front()->source, covering.front()->owner_id,
                                               covering.front()->final_leaf_index, 0.0});
                }
            }
            else if (covering.size() > 2)
            {
                ++report.nonmanifold_edge_span_count;
                AddIssue(report, options, {WatertightnessIssueKind::NonManifoldEdgeSpan,
                                           covering.front()->source, covering.front()->owner_id,
                                           covering.front()->final_leaf_index, 0.0});
            }
            if (covering.size() >= 2)
            {
                for (std::size_t a = 0; a < covering.size(); ++a)
                {
                    for (std::size_t b = a + 1; b < covering.size(); ++b)
                    {
                        const double gap = Norm(Subtract(covering[a]->moment, covering[b]->moment));
                        report.max_edge_gap = std::max(report.max_edge_gap, gap);
                        if (gap > report.geometry_tolerance)
                        {
                            ++report.geometric_mismatch_count;
                            AddIssue(report, options, {WatertightnessIssueKind::GeometricMismatch,
                                                       covering[a]->source, covering[a]->owner_id,
                                                       covering[a]->final_leaf_index, gap});
                        }
                    }
                }
            }
        }
    }

    const bool ownership_ok = !options.require_single_source_owner ||
        (report.unowned_source_leaf_count == 0 && report.multiply_owned_source_leaf_count == 0);
    const bool edge_ok = (!options.require_closed || report.open_edge_span_count == 0) &&
        report.nonmanifold_edge_span_count == 0 && report.geometric_mismatch_count == 0 &&
        report.degenerate_edge_count == 0;
    report.watertight = ownership_ok && edge_ok;
    return report;
}

void RequireWatertightForRayTracing(const WatertightnessReport &report)
{
    if (report.watertight) { return; }
    std::ostringstream message;
    message << "T-spline shell is not safe for ray tracing: "
            << report.unowned_source_leaf_count << " unowned source leaves, "
            << report.multiply_owned_source_leaf_count << " multiply-owned source leaves, "
            << report.open_edge_span_count << " open edge spans, "
            << report.nonmanifold_edge_span_count << " non-manifold edge spans";
    throw std::runtime_error(message.str());
}

} // namespace tspline
} // namespace mfem_raytracing
