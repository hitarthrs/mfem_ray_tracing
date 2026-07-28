#include "tspline_shell_json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

constexpr int kCornerFromUv[2][2] = {{0, 1}, {2, 3}};

void WriteString(std::ostream &os, const std::string &value)
{
    os << '"';
    for (const char c : value)
    {
        switch (c)
        {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << c; break;
        }
    }
    os << '"';
}

void WriteBool(std::ostream &os, bool value)
{
    os << (value ? "true" : "false");
}

void WriteVec3(std::ostream &os, const double values[3])
{
    os << '[' << values[0] << ", " << values[1] << ", " << values[2] << ']';
}

void WriteDomain(std::ostream &os, const double values[2])
{
    os << '[' << values[0] << ", " << values[1] << ']';
}

void WriteRect(std::ostream &os, const std::array<double, 4> &rect)
{
    os << '[' << rect[0] << ", " << rect[1] << ", " << rect[2] << ", " << rect[3] << ']';
}

const char *CornerPolicyName(CornerOwnershipPolicy policy)
{
    switch (policy)
    {
        case CornerOwnershipPolicy::ExactBoundaryCornerCollar: return "exact-boundary-corner-collar";
        case CornerOwnershipPolicy::RejectAmbiguous: return "reject-ambiguous";
        case CornerOwnershipPolicy::CompatibilityOverlap: return "compatibility-overlap";
    }
    return "unknown";
}

void WriteSource(std::ostream &os, const SourceLeafRef &source)
{
    os << "{\"patch_id\": " << source.patch_id << ", \"leaf_index\": "
       << source.leaf_index << '}';
}

AxisAlignedBounds SceneBounds(const BakedTsplineShell &shell)
{
    AxisAlignedBounds result{};
    if (shell.leaves.empty()) { return result; }
    result = shell.leaves.front().leaf.bbox;
    for (std::size_t i = 1; i < shell.leaves.size(); ++i)
    {
        const AxisAlignedBounds &bounds = shell.leaves[i].leaf.bbox;
        for (int axis = 0; axis < 3; ++axis)
        {
            result.min[axis] = std::min(result.min[axis], bounds.min[axis]);
            result.max[axis] = std::max(result.max[axis], bounds.max[axis]);
        }
    }
    return result;
}

void WriteError(std::ostream &os, const LeafErrorBreakdown &error)
{
    os << "{\"source_reduction\": " << error.source_reduction_error
       << ", \"seam_displacement\": " << error.seam_displacement
       << ", \"bake_decomposition\": " << error.bake_decomposition_error
       << ", \"conservative_bound\": " << error.ConservativeBound() << '}';
}

void WriteLeaf(std::ostream &os, const BakedShellLeaf &entry)
{
    const LeafPatch &leaf = entry.leaf;
    os << "    {\n";
    os << "      \"index\": " << leaf.index << ",\n";
    os << "      \"patch_id\": " << leaf.patch_id << ",\n";
    os << "      \"role\": "; WriteString(os, leaf.role); os << ",\n";
    os << "      \"kind\": "; WriteString(os, leaf.kind); os << ",\n";
    os << "      \"interface\": "; WriteString(os, leaf.interface_id); os << ",\n";
    os << "      \"source\": "; WriteSource(os, entry.source); os << ",\n";
    os << "      \"owner_id\": "; WriteString(os, entry.owner_id); os << ",\n";
    os << "      \"strip_face_index\": " << entry.strip_face_index << ",\n";
    os << "      \"local_subrect\": "; WriteRect(os, entry.local_subrect); os << ",\n";
    os << "      \"error\": "; WriteError(os, entry.error); os << ",\n";
    os << "      \"degree_u\": 1,\n";
    os << "      \"degree_v\": 1,\n";
    os << "      \"bbox_min\": "; WriteVec3(os, leaf.bbox.min); os << ",\n";
    os << "      \"bbox_max\": "; WriteVec3(os, leaf.bbox.max); os << ",\n";
    os << "      \"u_domain_global\": "; WriteDomain(os, leaf.u_domain_global); os << ",\n";
    os << "      \"v_domain_global\": "; WriteDomain(os, leaf.v_domain_global); os << ",\n";
    os << "      \"total_error\": " << leaf.total_error << ",\n";
    os << "      \"weights_non_negative\": true,\n";
    os << "      \"control_points\": [";
    for (int u = 0; u < 2; ++u)
    {
        os << (u == 0 ? "[" : ", [");
        for (int v = 0; v < 2; ++v)
        {
            if (v != 0) { os << ", "; }
            WriteVec3(os, leaf.patch.control_points[kCornerFromUv[u][v]]);
        }
        os << ']';
    }
    os << "],\n";
    if (!leaf.patch.rational)
    {
        os << "      \"weights\": null\n";
    }
    else
    {
        os << "      \"weights\": [";
        for (int u = 0; u < 2; ++u)
        {
            os << (u == 0 ? "[" : ", [");
            for (int v = 0; v < 2; ++v)
            {
                if (v != 0) { os << ", "; }
                os << leaf.patch.weights[kCornerFromUv[u][v]];
            }
            os << ']';
        }
        os << "]\n";
    }
    os << "    }";
}

void WriteWatertightnessIssue(std::ostream &os, const WatertightnessIssue &issue)
{
    os << "      {\"kind\": ";
    WriteString(os, WatertightnessIssueKindName(issue.kind));
    os << ", \"source\": "; WriteSource(os, issue.source);
    os << ", \"owner_id\": "; WriteString(os, issue.owner_id);
    os << ", \"final_leaf_index\": " << issue.final_leaf_index;
    os << ", \"gap\": " << issue.gap << '}';
}

void WriteWatertightness(std::ostream &os, const WatertightnessReport &report)
{
    os << "  \"watertightness\": {\n";
    os << "    \"watertight\": "; WriteBool(os, report.watertight); os << ",\n";
    os << "    \"geometry_tolerance\": " << report.geometry_tolerance << ",\n";
    os << "    \"max_edge_gap\": " << report.max_edge_gap << ",\n";
    os << "    \"expected_source_leaf_count\": " << report.expected_source_leaf_count << ",\n";
    os << "    \"unowned_source_leaf_count\": " << report.unowned_source_leaf_count << ",\n";
    os << "    \"multiply_owned_source_leaf_count\": "
       << report.multiply_owned_source_leaf_count << ",\n";
    os << "    \"atomic_edge_span_count\": " << report.atomic_edge_span_count << ",\n";
    os << "    \"open_edge_span_count\": " << report.open_edge_span_count << ",\n";
    os << "    \"nonmanifold_edge_span_count\": " << report.nonmanifold_edge_span_count << ",\n";
    os << "    \"geometric_mismatch_count\": " << report.geometric_mismatch_count << ",\n";
    os << "    \"degenerate_edge_count\": " << report.degenerate_edge_count << ",\n";
    os << "    \"issues\": [";
    for (std::size_t i = 0; i < report.issues.size(); ++i)
    {
        os << (i == 0 ? "\n" : ",\n");
        WriteWatertightnessIssue(os, report.issues[i]);
    }
    os << (report.issues.empty() ? "]\n" : "\n    ]\n");
    os << "  }";
}

void WriteOwnership(std::ostream &os, const SeamOwnershipPlan &ownership)
{
    os << "  \"ownership\": {\n";
    os << "    \"raw_claim_count\": " << ownership.raw_claim_count << ",\n";
    os << "    \"unique_claimed_source_leaf_count\": "
       << ownership.unique_claimed_source_leaf_count << ",\n";
    os << "    \"conflict_count\": " << ownership.conflicts.size() << ",\n";
    os << "    \"conflicts\": [";
    for (std::size_t i = 0; i < ownership.conflicts.size(); ++i)
    {
        const CornerConflict &conflict = ownership.conflicts[i];
        os << (i == 0 ? "\n      {\"source\": " : ",\n      {\"source\": ");
        WriteSource(os, conflict.source);
        os << ", \"claims\": [";
        for (std::size_t j = 0; j < conflict.claims.size(); ++j)
        {
            const SeamClaim &claim = conflict.claims[j];
            os << (j == 0 ? "{" : ", {");
            os << "\"interface\": "; WriteString(os, claim.interface_id);
            os << ", \"region\": "; WriteString(os, claim.region == StripRegion::A ? "a" : "b");
            os << '}';
        }
        os << "]}";
    }
    os << (ownership.conflicts.empty() ? "]\n" : "\n    ]\n");
    os << "  }";
}

void WriteStrips(std::ostream &os, const std::vector<ShellStripResult> &strips)
{
    os << "  \"seam_strips\": [";
    for (std::size_t i = 0; i < strips.size(); ++i)
    {
        const ShellStripResult &strip = strips[i];
        os << (i == 0 ? "\n    {\n" : ",\n    {\n");
        os << "      \"interface\": "; WriteString(os, strip.interface_id); os << ",\n";
        os << "      \"patch_a\": " << strip.interface.patch_a << ",\n";
        os << "      \"side_a\": "; WriteString(os, BoundarySideName(strip.interface.side_a)); os << ",\n";
        os << "      \"patch_b\": " << strip.interface.patch_b << ",\n";
        os << "      \"side_b\": "; WriteString(os, BoundarySideName(strip.interface.side_b)); os << ",\n";
        os << "      \"b_reversed\": "; WriteBool(os, strip.interface.b_reversed); os << ",\n";
        os << "      \"max_pre_merge_disagreement\": " << strip.merge.max_pre_merge_disagreement << ",\n";
        os << "      \"max_displacement_a\": " << strip.merge.max_displacement_a << ",\n";
        os << "      \"max_displacement_b\": " << strip.merge.max_displacement_b << ",\n";
        os << "      \"n_source_faces\": " << strip.strip.faces.size() << ",\n";
        os << "      \"n_t_junctions\": " << strip.strip.t_junctions.size() << ",\n";
        os << "      \"n_rule2_added_edges\": " << strip.strip.rule2_added_edges << ",\n";
        os << "      \"n_exact_primitives\": " << strip.baked.n_exact_primitives << ",\n";
        os << "      \"n_junction_primitives\": " << strip.baked.n_junction_primitives << ",\n";
        os << "      \"max_decomposition_error\": " << strip.verification.max_decomposition_error << ",\n";
        os << "      \"decomposition_verified\": "; WriteBool(os, strip.verification.exact); os << "\n";
        os << "    }";
    }
    os << (strips.empty() ? "]\n" : "\n  ]\n");
}

} // namespace

void WriteBakedTsplineShellJson(std::ostream &os, const BakedTsplineShell &shell)
{
    // The C++ JSON reader consumes the same leaf fields emitted here.  Reject
    // non-finite metrics early instead of emitting invalid JSON tokens.
    const ErrorAccountingReport &accounting = shell.errors.accounting;
    if (!std::isfinite(accounting.max_conservative_error))
    {
        throw std::invalid_argument("cannot serialize a T-spline shell with a non-finite error bound");
    }
    const AxisAlignedBounds bounds = SceneBounds(shell);
    const bool rt_certified = shell.ReadyForRayTracing();

    os << std::setprecision(17);
    os << "{\n";
    os << "  \"surface\": "; WriteString(os, shell.surface_name); os << ",\n";
    os << "  \"backend\": \"degree_one_tspline_baked_shell\",\n";
    os << "  \"max_error\": " << accounting.max_conservative_error << ",\n";
    os << "  \"n_leaves\": " << shell.leaves.size() << ",\n";
    os << "  \"scene_bbox_min\": "; WriteVec3(os, bounds.min); os << ",\n";
    os << "  \"scene_bbox_max\": "; WriteVec3(os, bounds.max); os << ",\n";
    os << "  \"certification\": {\n";
    os << "    \"rt_certified\": "; WriteBool(os, rt_certified); os << ",\n";
    os << "    \"corner_ownership_policy\": "; WriteString(os, CornerPolicyName(shell.corner_policy)); os << ",\n";
    os << "    \"status\": "; WriteString(os, rt_certified ? "ready-for-ray-tracing" : "diagnostic-only"); os << "\n";
    os << "  },\n";
    os << "  \"error_accounting\": {\n";
    os << "    \"leaf_count\": " << accounting.leaf_count << ",\n";
    os << "    \"max_source_reduction_error\": " << accounting.max_source_reduction_error << ",\n";
    os << "    \"max_seam_displacement\": " << accounting.max_seam_displacement << ",\n";
    os << "    \"max_bake_decomposition_error\": " << accounting.max_bake_decomposition_error << ",\n";
    os << "    \"max_conservative_error\": " << accounting.max_conservative_error << ",\n";
    os << "    \"within_requested_limit\": "; WriteBool(os, shell.errors.within_requested_limit); os << "\n";
    os << "  },\n";
    WriteOwnership(os, shell.ownership); os << ",\n";
    WriteWatertightness(os, shell.watertightness); os << ",\n";
    WriteStrips(os, shell.strips); os << ",\n";
    os << "  \"leaves\": [";
    for (std::size_t i = 0; i < shell.leaves.size(); ++i)
    {
        os << (i == 0 ? "\n" : ",\n");
        WriteLeaf(os, shell.leaves[i]);
    }
    os << (shell.leaves.empty() ? "]\n" : "\n  ]\n");
    os << "}\n";
}

} // namespace tspline
} // namespace mfem_raytracing
