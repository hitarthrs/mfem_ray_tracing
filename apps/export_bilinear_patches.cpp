// Watertight bilinear-patch export for any (p, q) NURBS/B-spline surface.
//
// Runs the conforming (competing + coalesce) multi-step reduction to degrees
// (1, 1) and writes the leaf-bbox JSON consumed by LoadLeafPatchScene,
// render_leaf_patches and bilinear_ray_tracer.html.
//
// Input surfaces come from input-surface JSON files (see
// python_experiments/export_surface_inputs.py, which exports every degree-4
// example to python_experiments/nurbs_surface_examples/exports/<name>.json).
//
// Usage:
//   export_bilinear_patches -s d4_semicircle_plateau_shell --max-error 0.1 \
//       --json out.json
//   export_bilinear_patches --input my_surface.json --max-error 0.05 --json out.json

#include "mfem_raytracing/reduction/bilinear_leaf_extraction.hpp"
#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "mfem_raytracing/reduction/surface_conforming_reduction.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace
{

constexpr const char *kExportsDir = "python_experiments/nurbs_surface_examples/exports";

void PrintUsage()
{
    std::cerr
        << "usage: export_bilinear_patches [options]\n"
        << "  -s, --surface NAME     degree-4 example (reads " << kExportsDir
        << "/NAME.json)\n"
        << "      --input PATH       input-surface JSON (any (p, q) surface)\n"
        << "      --max-error X      global error budget (required)\n"
        << "      --backend NAME     budget policy approach_1|3|4|5 (default: approach_4)\n"
        << "      --error-combination sum|max   leaf error accounting (default: sum)\n"
        << "      --no-rational-tol-correction  disable Piegl & Tiller Eq. 5.30\n"
        << "      --no-coalesce      keep the raw conforming grid (skip A+C pass)\n"
        << "      --non-conforming   legacy non-watertight driver (not recommended)\n"
        << "      --json PATH        output leaf-bbox JSON (required)\n";
}

} // namespace

int main(int argc, char **argv)
{
    std::string surface_name;
    std::string input_path;
    std::string json_path;
    double max_error = -1.0;
    mfem_raytracing::BilinearLeafReductionOptions options;
    options.conforming = true;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&](const char *flag) -> std::string {
            if (i + 1 >= argc)
            {
                std::cerr << "error: missing value for " << flag << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "-s" || arg == "--surface")
        {
            surface_name = next(arg.c_str());
        }
        else if (arg == "--input")
        {
            input_path = next(arg.c_str());
        }
        else if (arg == "--max-error")
        {
            max_error = std::stod(next(arg.c_str()));
        }
        else if (arg == "--backend")
        {
            options.backend_name = next(arg.c_str());
        }
        else if (arg == "--error-combination")
        {
            const std::string mode = next(arg.c_str());
            if (mode == "sum")
            {
                options.single_step.error_combination =
                    mfem_raytracing::SurfaceErrorCombination::Sum;
            }
            else if (mode == "max")
            {
                options.single_step.error_combination =
                    mfem_raytracing::SurfaceErrorCombination::Max;
            }
            else
            {
                std::cerr << "error: --error-combination must be 'sum' or 'max'\n";
                return 1;
            }
        }
        else if (arg == "--no-rational-tol-correction")
        {
            options.single_step.rational_tol_correction = false;
        }
        else if (arg == "--no-coalesce")
        {
            options.coalesce = false;
        }
        else if (arg == "--non-conforming")
        {
            options.conforming = false;
        }
        else if (arg == "--json")
        {
            json_path = next(arg.c_str());
        }
        else if (arg == "-h" || arg == "--help")
        {
            PrintUsage();
            return 0;
        }
        else
        {
            std::cerr << "error: unknown argument '" << arg << "'\n";
            PrintUsage();
            return 1;
        }
    }

    if ((surface_name.empty() && input_path.empty()) || max_error < 0.0 || json_path.empty())
    {
        PrintUsage();
        return 1;
    }
    if (input_path.empty())
    {
        input_path = std::string(kExportsDir) + "/" + surface_name + ".json";
    }
    if (surface_name.empty())
    {
        surface_name = input_path;
    }

    mfem_raytracing::SurfaceData surface;
    try
    {
        surface = mfem_raytracing::LoadSurfaceDataJson(input_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    const int n_steps = std::max(surface.degree_u - 1, surface.degree_v - 1);
    std::cout << "=== " << surface_name << " ===\n"
              << "reduction: (" << surface.degree_u << "," << surface.degree_v
              << ") -> (1,1), max_error=" << max_error << ", "
              << (options.conforming
                      ? (options.coalesce ? "conforming + coalesce (watertight)"
                                          : "conforming grid (watertight)")
                      : "non-conforming (NOT watertight)")
              << "\n";

    mfem_raytracing::BilinearLeafCollection collection;
    try
    {
        collection = mfem_raytracing::ReduceSurfaceToBilinearLeaves(surface, n_steps,
                                                                    max_error, options);
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "leaves: " << collection.leaves.size() << "\n";
    if (!collection.leaves.empty())
    {
        const auto scene = collection.SceneBBox();
        std::cout << "scene AABB: min=[" << scene.min[0] << ", " << scene.min[1] << ", "
                  << scene.min[2] << "] max=[" << scene.max[0] << ", " << scene.max[1]
                  << ", " << scene.max[2] << "]\n";
    }

    std::ofstream out(json_path);
    if (!out)
    {
        std::cerr << "error: cannot write '" << json_path << "'\n";
        return 1;
    }
    mfem_raytracing::WriteLeafBBoxJson(out, collection, surface_name);
    std::cout << "wrote " << json_path << "\n";
    return 0;
}
