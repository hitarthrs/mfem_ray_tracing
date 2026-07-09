// Bilinear leaf extraction + JSON export for multi-step surface reduction.
//
// Port of python_experiments/multiple_step_degree_reduction_surfaces/leaf_bvh.py
// and the JSON schema of demo_leaf_bboxes.py (loadable by LoadLeafPatchScene and
// the bilinear_ray_tracer.html viewer).

#include "bilinear_leaf_extraction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace mfem_raytracing
{
namespace
{

// Shortest decimal representation that round-trips to the same double, with a
// Python-style trailing ".0" for integral values (diffable against json.dumps).
std::string FormatDouble(double value)
{
    std::string text;
    for (int precision = 15; precision <= 17; ++precision)
    {
        std::ostringstream oss;
        oss.precision(precision);
        oss << value;
        text = oss.str();
        if (std::strtod(text.c_str(), nullptr) == value)
        {
            break;
        }
    }
    if (text.find('.') == std::string::npos && text.find('e') == std::string::npos &&
        text.find("inf") == std::string::npos && text.find("nan") == std::string::npos)
    {
        text += ".0";
    }
    return text;
}

void WriteVec3(std::ostream &os, const double v[3])
{
    os << '[' << FormatDouble(v[0]) << ", " << FormatDouble(v[1]) << ", " << FormatDouble(v[2])
       << ']';
}

void WriteDomain(std::ostream &os, const std::pair<double, double> &domain)
{
    os << '[' << FormatDouble(domain.first) << ", " << FormatDouble(domain.second) << ']';
}

} // namespace

AxisAlignedBounds BilinearLeafCollection::SceneBBox() const
{
    AxisAlignedBounds bbox;
    if (leaves.empty())
    {
        return bbox;
    }
    bbox = leaves.front().bbox;
    for (std::size_t i = 1; i < leaves.size(); ++i)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            bbox.min[axis] = std::min(bbox.min[axis], leaves[i].bbox.min[axis]);
            bbox.max[axis] = std::max(bbox.max[axis], leaves[i].bbox.max[axis]);
        }
    }
    return bbox;
}

ErrorBudgetPolicy BudgetPolicyFromBackendName(const std::string &backend_name)
{
    if (backend_name == "approach_1")
    {
        return ErrorBudgetPolicy::Cumulative;
    }
    if (backend_name == "approach_3")
    {
        return ErrorBudgetPolicy::EqualPerStep;
    }
    if (backend_name == "approach_4")
    {
        return ErrorBudgetPolicy::WeightedLate;
    }
    if (backend_name == "approach_5")
    {
        return ErrorBudgetPolicy::Geometric;
    }
    throw std::invalid_argument("unknown backend '" + backend_name +
                                "'; choose from approach_1, approach_3, approach_4, approach_5");
}

bool WeightsAreNonNegative(const std::vector<std::vector<double>> &weights, double tol)
{
    for (const auto &row : weights)
    {
        for (double w : row)
        {
            if (w < -tol)
            {
                return false;
            }
        }
    }
    return true;
}

bool ClampWeightsNonNegative(std::vector<std::vector<double>> &weights, double floor, double tol)
{
    if (WeightsAreNonNegative(weights, tol))
    {
        return false;
    }
    for (auto &row : weights)
    {
        for (double &w : row)
        {
            w = std::max(w, floor);
        }
    }
    return true;
}

AxisAlignedBounds BilinearControlNetAABB(const SurfaceData &surface)
{
    if (surface.degree_u != 1 || surface.degree_v != 1)
    {
        std::ostringstream oss;
        oss << "BilinearControlNetAABB: expected bilinear leaf (1, 1), got ("
            << surface.degree_u << ", " << surface.degree_v << ")";
        throw std::invalid_argument(oss.str());
    }
    if (surface.NumControlPointsU() != 2 || surface.NumControlPointsV() != 2)
    {
        std::ostringstream oss;
        oss << "BilinearControlNetAABB: expected a 2x2 control net, got "
            << surface.NumControlPointsU() << "x" << surface.NumControlPointsV();
        throw std::invalid_argument(oss.str());
    }
    if (surface.IsRational() && !WeightsAreNonNegative(surface.weights))
    {
        throw std::invalid_argument(
            "BilinearControlNetAABB: exact AABB requires non-negative weights; "
            "clamp them first");
    }

    AxisAlignedBounds bbox;
    for (int axis = 0; axis < 3; ++axis)
    {
        bbox.min[axis] = surface.control_points[0][0][static_cast<std::size_t>(axis)];
        bbox.max[axis] = bbox.min[axis];
    }
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            const auto &point = surface.control_points[static_cast<std::size_t>(i)]
                                                      [static_cast<std::size_t>(j)];
            if (static_cast<int>(point.size()) != 3)
            {
                throw std::invalid_argument(
                    "BilinearControlNetAABB: expected 3D control points");
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                const double value = point[static_cast<std::size_t>(axis)];
                bbox.min[axis] = std::min(bbox.min[axis], value);
                bbox.max[axis] = std::max(bbox.max[axis], value);
            }
        }
    }
    return bbox;
}

BilinearLeafPatch ExtractBilinearLeafPatch(const ReducedSurfaceLeaf &leaf,
                                           int index,
                                           bool enforce_nonneg_weights,
                                           double weight_tol)
{
    BilinearLeafPatch patch;
    patch.index = index;
    patch.surface = leaf.surface;
    patch.weights_were_clamped = false;

    if (patch.surface.IsRational())
    {
        if (enforce_nonneg_weights)
        {
            patch.weights_were_clamped =
                ClampWeightsNonNegative(patch.surface.weights, 0.0, weight_tol);
        }
        else if (!WeightsAreNonNegative(patch.surface.weights, weight_tol))
        {
            std::ostringstream oss;
            oss << "leaf " << index
                << " has negative weights; set enforce_nonneg_weights=true to clamp";
            throw std::invalid_argument(oss.str());
        }
    }

    patch.bbox = BilinearControlNetAABB(patch.surface);
    patch.u_domain_local = patch.surface.u_domain;
    patch.v_domain_local = patch.surface.v_domain;
    patch.u_domain_global = leaf.u_domain_global;
    patch.v_domain_global = leaf.v_domain_global;
    patch.total_error = leaf.total_error;
    return patch;
}

BilinearLeafCollection ReduceSurfaceToBilinearLeaves(const SurfaceData &initial_surface,
                                                     int n_steps,
                                                     double max_error,
                                                     const BilinearLeafReductionOptions &options)
{
    MultiStepSurfaceReductionOptions multi_options;
    multi_options.budget_policy = BudgetPolicyFromBackendName(options.backend_name);
    multi_options.single_step = options.single_step;

    const MultipleStepSurfaceReductionResult result =
        DegreeReduceMultipleSteps(initial_surface, n_steps, max_error, multi_options);

    BilinearLeafCollection collection;
    collection.max_error = max_error;
    collection.n_steps = n_steps;
    collection.backend_name = options.backend_name;
    collection.leaves.reserve(result.segments.size());
    for (std::size_t i = 0; i < result.segments.size(); ++i)
    {
        collection.leaves.push_back(ExtractBilinearLeafPatch(result.segments[i],
                                                             static_cast<int>(i),
                                                             options.enforce_nonneg_weights));
    }
    return collection;
}

void WriteLeafBBoxJson(std::ostream &os,
                       const BilinearLeafCollection &collection,
                       const std::string &surface_name)
{
    os << "{\n";
    os << "  \"surface\": \"" << surface_name << "\",\n";
    os << "  \"backend\": \"" << collection.backend_name << "\",\n";
    os << "  \"max_error\": " << FormatDouble(collection.max_error) << ",\n";
    os << "  \"n_steps\": " << collection.n_steps << ",\n";
    os << "  \"n_leaves\": " << collection.leaves.size() << ",\n";

    if (collection.leaves.empty())
    {
        os << "  \"scene_bbox_min\": null,\n";
        os << "  \"scene_bbox_max\": null,\n";
    }
    else
    {
        const AxisAlignedBounds scene = collection.SceneBBox();
        os << "  \"scene_bbox_min\": ";
        WriteVec3(os, scene.min);
        os << ",\n  \"scene_bbox_max\": ";
        WriteVec3(os, scene.max);
        os << ",\n";
    }

    os << "  \"leaves\": [";
    for (std::size_t k = 0; k < collection.leaves.size(); ++k)
    {
        const BilinearLeafPatch &leaf = collection.leaves[k];
        os << (k == 0 ? "\n" : ",\n");
        os << "    {\n";
        os << "      \"index\": " << leaf.index << ",\n";
        os << "      \"degree_u\": " << leaf.surface.degree_u << ",\n";
        os << "      \"degree_v\": " << leaf.surface.degree_v << ",\n";
        os << "      \"bbox_min\": ";
        WriteVec3(os, leaf.bbox.min);
        os << ",\n      \"bbox_max\": ";
        WriteVec3(os, leaf.bbox.max);
        os << ",\n      \"u_domain_local\": ";
        WriteDomain(os, leaf.u_domain_local);
        os << ",\n      \"v_domain_local\": ";
        WriteDomain(os, leaf.v_domain_local);
        os << ",\n      \"u_domain_global\": ";
        WriteDomain(os, leaf.u_domain_global);
        os << ",\n      \"v_domain_global\": ";
        WriteDomain(os, leaf.v_domain_global);
        os << ",\n      \"total_error\": " << FormatDouble(leaf.total_error) << ",\n";

        const bool rational = leaf.surface.IsRational();
        const bool weights_non_negative =
            !rational || WeightsAreNonNegative(leaf.surface.weights, 0.0);
        os << "      \"weights_non_negative\": " << (weights_non_negative ? "true" : "false")
           << ",\n";
        os << "      \"weights_were_clamped\": " << (leaf.weights_were_clamped ? "true" : "false")
           << ",\n";

        os << "      \"control_points\": [";
        for (int i = 0; i < 2; ++i)
        {
            os << (i == 0 ? "[" : ", [");
            for (int j = 0; j < 2; ++j)
            {
                if (j != 0)
                {
                    os << ", ";
                }
                WriteVec3(os,
                          leaf.surface.control_points[static_cast<std::size_t>(i)]
                                                     [static_cast<std::size_t>(j)]
                              .data());
            }
            os << "]";
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
                    if (j != 0)
                    {
                        os << ", ";
                    }
                    os << FormatDouble(leaf.surface.weights[static_cast<std::size_t>(i)]
                                                           [static_cast<std::size_t>(j)]);
                }
                os << "]";
            }
            os << "]\n";
        }
        os << "    }";
    }
    os << "\n  ]\n}\n";
}

} // namespace mfem_raytracing
