// Extract bilinear leaf AABBs after multi-step surface degree reduction and
// write them in the demo_leaf_bboxes.py JSON schema (loadable by
// LoadLeafPatchScene and the bilinear_ray_tracer.html viewer).
//
// Input surfaces are the degree-4 examples embedded in the generated golden
// fixtures (tests/generated_surface_degree_reduction_cases.inc).
//
// Usage:
//   export_leaf_bboxes [-s d4_s_shaped_peak_saddle] [--backend approach_4]
//                      [--max-error 0.5] [--degree-out 1] [--json out.json]

#include "bilinear_leaf_extraction.hpp"

#include "surface_golden_cases.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace
{

#include "generated_surface_degree_reduction_cases.inc"

const std::map<std::string, SingleStepSurfaceGoldenCase (*)()> kExampleSurfaces = {
    {"d4_s_shaped_peak_saddle", &SurfaceApproach1SShapedPeakSaddle},
    {"d4_multiple_peak_uniform_shell", &SurfaceApproach1MultiplePeakUniformShell},
    {"d4_semicircle_plateau_shell", &SurfaceApproach1SemicirclePlateauShell},
    {"d4_semicircle_s_shaped_crown", &SurfaceApproach1SemicircleSShapedCrown},
};

void PrintUsage()
{
    std::cerr << "usage: export_leaf_bboxes [options]\n"
              << "  -s, --surface NAME     degree-4 example surface (default: "
                 "d4_s_shaped_peak_saddle)\n"
              << "      --backend NAME     approach_1|approach_3|approach_4|approach_5 "
                 "(default: approach_4)\n"
              << "      --max-error X      global error budget per branch (default: 0.5)\n"
              << "      --degree-out N     target degree (default: 1)\n"
              << "      --max-depth N      peak-error splits per step (default: 25)\n"
              << "      --min-span-width X minimum parameter span (default: 1e-8)\n"
              << "      --error-combination sum|max   leaf error accounting (default: sum)\n"
              << "      --no-rational-tol-correction  disable Piegl & Tiller Eq. 5.30\n"
              << "      --json PATH        write leaf bbox JSON to PATH\n"
              << "surfaces:\n";
    for (const auto &entry : kExampleSurfaces)
    {
        std::cerr << "  " << entry.first << "\n";
    }
}

} // namespace

int main(int argc, char **argv)
{
    std::string surface_name = "d4_s_shaped_peak_saddle";
    std::string backend = "approach_4";
    double max_error = 0.5;
    int degree_out = 1;
    std::string json_path;
    mfem_raytracing::BilinearLeafReductionOptions options;

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
        else if (arg == "--backend")
        {
            backend = next(arg.c_str());
        }
        else if (arg == "--max-error")
        {
            max_error = std::stod(next(arg.c_str()));
        }
        else if (arg == "--degree-out")
        {
            degree_out = std::stoi(next(arg.c_str()));
        }
        else if (arg == "--max-depth")
        {
            options.single_step.max_depth = std::stoi(next(arg.c_str()));
        }
        else if (arg == "--min-span-width")
        {
            options.single_step.min_span_width = std::stod(next(arg.c_str()));
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

    const auto example = kExampleSurfaces.find(surface_name);
    if (example == kExampleSurfaces.end())
    {
        std::cerr << "error: unknown surface '" << surface_name << "'\n";
        PrintUsage();
        return 1;
    }
    options.backend_name = backend;

    const mfem_raytracing::SurfaceData input = MakeSurfaceData(example->second().input);
    const int n_steps = input.degree_u - degree_out;

    std::cout << "=== " << surface_name << " (" << backend << ") — leaf AABBs ===\n";
    std::cout << "reduction: (" << input.degree_u << "," << input.degree_v << ") -> ("
              << degree_out << "," << degree_out << ") in " << n_steps
              << " steps, max_error=" << max_error << "\n";

    mfem_raytracing::BilinearLeafCollection collection;
    try
    {
        collection = mfem_raytracing::ReduceSurfaceToBilinearLeaves(input,
                                                                    n_steps,
                                                                    max_error,
                                                                    options);
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    int clamped_count = 0;
    for (const auto &leaf : collection.leaves)
    {
        clamped_count += leaf.weights_were_clamped ? 1 : 0;
    }
    std::cout << "leaves: " << collection.leaves.size() << "\n";
    std::cout << "weight-clamped leaves: " << clamped_count << "\n";
    if (!collection.leaves.empty())
    {
        const auto scene = collection.SceneBBox();
        std::cout << "scene AABB: min=[" << scene.min[0] << ", " << scene.min[1] << ", "
                  << scene.min[2] << "] max=[" << scene.max[0] << ", " << scene.max[1] << ", "
                  << scene.max[2] << "]\n";
    }

    if (!json_path.empty())
    {
        std::ofstream out(json_path);
        if (!out)
        {
            std::cerr << "error: cannot write '" << json_path << "'\n";
            return 1;
        }
        mfem_raytracing::WriteLeafBBoxJson(out, collection, surface_name);
        std::cout << "wrote " << json_path << "\n";
    }
    return 0;
}
