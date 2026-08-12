#include "mfem_raytracing/reduction/hard_seam_bilinearization.hpp"

#include "mfem_raytracing/pipeline/reduce.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace mfem_raytracing
{
namespace
{

constexpr double kHardSeamTolerance = 1e-12;

std::vector<double> UniqueKnots(const std::vector<double> &knots)
{
    std::vector<double> unique = knots;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end(), [](double a, double b) {
        return std::fabs(a - b) <= kHardSeamTolerance;
    }), unique.end());
    return unique;
}

bool CrossesHardSeam(double lo, double hi, const std::vector<double> &knots)
{
    for (const double knot : knots)
    {
        if (lo + kHardSeamTolerance < knot && knot < hi - kHardSeamTolerance)
        {
            return true;
        }
    }
    return false;
}

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

void WriteVec3(std::ostream &os, const double values[3])
{
    os << '[' << values[0] << ", " << values[1] << ", " << values[2] << ']';
}

void WriteDomain(std::ostream &os, const std::pair<double, double> &domain)
{
    os << '[' << domain.first << ", " << domain.second << ']';
}

void WriteKnotVector(std::ostream &os, const std::vector<double> &knots)
{
    os << '[';
    for (std::size_t i = 0; i < knots.size(); ++i)
    {
        if (i != 0) { os << ", "; }
        os << knots[i];
    }
    os << ']';
}

void WriteLeaf(std::ostream &os, const BilinearLeafPatch &leaf,
               const SurfacePatchDescriptor &source)
{
    const bool rational = leaf.surface.IsRational();
    os << "    {\n";
    os << "      \"index\": " << leaf.index << ",\n";
    os << "      \"patch_id\": " << source.id << ",\n";
    os << "      \"role\": "; WriteString(os, source.role); os << ",\n";
    os << "      \"degree_u\": " << leaf.surface.degree_u << ",\n";
    os << "      \"degree_v\": " << leaf.surface.degree_v << ",\n";
    os << "      \"bbox_min\": "; WriteVec3(os, leaf.bbox.min); os << ",\n";
    os << "      \"bbox_max\": "; WriteVec3(os, leaf.bbox.max); os << ",\n";
    os << "      \"u_domain_local\": "; WriteDomain(os, leaf.u_domain_local); os << ",\n";
    os << "      \"v_domain_local\": "; WriteDomain(os, leaf.v_domain_local); os << ",\n";
    os << "      \"u_domain_global\": "; WriteDomain(os, leaf.u_domain_global); os << ",\n";
    os << "      \"v_domain_global\": "; WriteDomain(os, leaf.v_domain_global); os << ",\n";
    os << "      \"total_error\": " << leaf.total_error << ",\n";
    os << "      \"weights_non_negative\": true,\n";
    os << "      \"weights_were_clamped\": " << (leaf.weights_were_clamped ? "true" : "false") << ",\n";
    os << "      \"control_points\": [";
    for (int i = 0; i < 2; ++i)
    {
        os << (i == 0 ? "[" : ", [");
        for (int j = 0; j < 2; ++j)
        {
            if (j != 0) { os << ", "; }
            WriteVec3(os, leaf.surface.control_points[static_cast<std::size_t>(i)]
                                                [static_cast<std::size_t>(j)].data());
        }
        os << ']';
    }
    os << "],\n";
    if (!rational)
    {
        os << "      \"weights\": null\n";
    }
    else
    {
        os << "      \"weights\": [";
        for (int i = 0; i < 2; ++i)
        {
            os << (i == 0 ? "[" : ", [");
            for (int j = 0; j < 2; ++j)
            {
                if (j != 0) { os << ", "; }
                os << leaf.surface.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            }
            os << ']';
        }
        os << "]\n";
    }
    os << "    }";
}

} // namespace

std::size_t HardSeamBilinearizationResult::LeafCount() const
{
    std::size_t count = 0;
    for (const HardSeamPatchBilinearization &patch : patches)
    {
        count += patch.leaves.size();
    }
    return count;
}

std::size_t HardSeamBilinearizationResult::CompetingCount() const
{
    std::size_t count = 0;
    for (const HardSeamPatchBilinearization &patch : patches)
    {
        count += patch.n_competing;
    }
    return count;
}

AxisAlignedBounds HardSeamBilinearizationResult::SceneBBox() const
{
    AxisAlignedBounds result{};
    bool initialized = false;
    for (const HardSeamPatchBilinearization &patch : patches)
    {
        for (const BilinearLeafPatch &leaf : patch.leaves)
        {
            if (!initialized)
            {
                result = leaf.bbox;
                initialized = true;
                continue;
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                result.min[axis] = std::min(result.min[axis], leaf.bbox.min[axis]);
                result.max[axis] = std::max(result.max[axis], leaf.bbox.max[axis]);
            }
        }
    }
    return result;
}

HardSeamBilinearizationResult BilinearizePatchesWithHardSeams(
    const SurfacePatchCatalog &catalog, const std::vector<int> &patch_ids,
    const HardSeamBilinearizationOptions &options)
{
    if (options.max_error < 0.0)
    {
        throw std::invalid_argument("hard-seam bilinearization: max_error must be non-negative");
    }
    if (patch_ids.empty())
    {
        throw std::invalid_argument("hard-seam bilinearization: no patch ids were requested");
    }

    HardSeamBilinearizationResult result;
    result.mesh = catalog.mesh;
    result.max_error = options.max_error;
    int global_index = 0;
    for (const int patch_id : patch_ids)
    {
        HardSeamPatchBilinearization output;
        output.source = FindSurfacePatch(catalog, patch_id);
        output.n_steps = std::max(output.source.surface.degree_u - 1,
                                  output.source.surface.degree_v - 1);
        if (output.n_steps < 1)
        {
            throw std::invalid_argument("hard-seam bilinearization: patch '" +
                                        output.source.name + "' is already degree (1,1)");
        }
        output.hard_knots_u = UniqueKnots(output.source.surface.knotvector_u);
        output.hard_knots_v = UniqueKnots(output.source.surface.knotvector_v);

        // Single compete+coalesce pass (hard seams ON). Competing grid size and
        // phase timings come from the same run — do not re-run the grid build.
        PatchLeafReductionOptions reduce_opts;
        reduce_opts.threads = options.threads;
        const BilinearLeafCollection collection =
            ReducePatchToBilinearLeaves(output.source.surface, options.max_error, reduce_opts);
        output.n_competing = collection.n_competing;
        output.seconds_compete = collection.seconds_compete;
        output.seconds_coalesce = collection.seconds_coalesce;
        result.seconds_compete += collection.seconds_compete;
        result.seconds_coalesce += collection.seconds_coalesce;
        output.leaves.reserve(collection.leaves.size());
        for (BilinearLeafPatch leaf : collection.leaves)
        {
            if (CrossesHardSeam(leaf.u_domain_global.first, leaf.u_domain_global.second,
                                output.hard_knots_u) ||
                CrossesHardSeam(leaf.v_domain_global.first, leaf.v_domain_global.second,
                                output.hard_knots_v))
            {
                throw std::runtime_error("hard-seam bilinearization: invalid coalesced leaf for patch '" +
                                         output.source.name + "'");
            }
            leaf.index = global_index++;
            output.leaves.push_back(std::move(leaf));
        }
        result.patches.push_back(std::move(output));
    }
    return result;
}

void WriteHardSeamBilinearizationJson(std::ostream &os,
                                      const HardSeamBilinearizationResult &result,
                                      const std::string &surface_name)
{
    os << std::setprecision(17);
    os << "{\n  \"mesh\": "; WriteString(os, result.mesh); os << ",\n";
    os << "  \"surface\": "; WriteString(os, surface_name); os << ",\n";
    os << "  \"backend\": \"approach_4_competing_coalesced_hard_seams\",\n";
    os << "  \"note\": \"full multi-span world-coordinate hard-seam reduction; no cage centering; no inter-patch sewing\",\n";
    os << "  \"max_error\": " << result.max_error << ",\n";
    os << "  \"patch_ids\": [";
    for (std::size_t i = 0; i < result.patches.size(); ++i)
    {
        if (i != 0) { os << ", "; }
        os << result.patches[i].source.id;
    }
    os << "],\n  \"per_patch\": [";
    for (std::size_t i = 0; i < result.patches.size(); ++i)
    {
        const HardSeamPatchBilinearization &patch = result.patches[i];
        os << (i == 0 ? "\n    {\n" : ",\n    {\n");
        os << "      \"patch_id\": " << patch.source.id << ",\n      \"name\": ";
        WriteString(os, patch.source.name); os << ",\n      \"role\": ";
        WriteString(os, patch.source.role); os << ",\n      \"quarter\": " << patch.source.quarter;
        os << ",\n      \"n_competing\": " << patch.n_competing;
        os << ",\n      \"n_leaves\": " << patch.leaves.size();
        os << ",\n      \"coalesce_delta\": "
           << (static_cast<long long>(patch.n_competing) -
               static_cast<long long>(patch.leaves.size()));
        os << ",\n      \"seconds_compete\": " << patch.seconds_compete;
        os << ",\n      \"seconds_coalesce\": " << patch.seconds_coalesce;
        os << ",\n      \"leaves_crossing_hard_seam\": 0";
        os << ",\n      \"hard_knots_u\": "; WriteKnotVector(os, patch.hard_knots_u);
        os << ",\n      \"hard_knots_v\": "; WriteKnotVector(os, patch.hard_knots_v);
        os << ",\n      \"leaf_shapes\": {\"(1, 1, 2, 2)\": " << patch.leaves.size() << '}';
        os << "\n    }";
    }
    os << "\n  ],\n  \"n_leaves\": " << result.LeafCount() << ",\n";
    os << "  \"n_competing\": " << result.CompetingCount() << ",\n";
    os << "  \"coalesce_delta\": "
       << (static_cast<long long>(result.CompetingCount()) -
           static_cast<long long>(result.LeafCount()))
       << ",\n";
    os << "  \"seconds_compete\": " << result.seconds_compete << ",\n";
    os << "  \"seconds_coalesce\": " << result.seconds_coalesce << ",\n";
    const AxisAlignedBounds bbox = result.SceneBBox();
    os << "  \"scene_bbox_min\": "; WriteVec3(os, bbox.min); os << ",\n";
    os << "  \"scene_bbox_max\": "; WriteVec3(os, bbox.max); os << ",\n";
    os << "  \"leaves\": [";
    bool first_leaf = true;
    for (const HardSeamPatchBilinearization &patch : result.patches)
    {
        for (const BilinearLeafPatch &leaf : patch.leaves)
        {
            os << (first_leaf ? "\n" : ",\n");
            WriteLeaf(os, leaf, patch.source);
            first_leaf = false;
        }
    }
    os << "\n  ]\n}\n";
}

} // namespace mfem_raytracing
