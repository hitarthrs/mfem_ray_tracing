// Compose the degree-one T-spline seam strips, bake them back to bilinear
// leaves, certify the whole shell, then write the runtime JSON artifact.
//
// A strict export is deliberately the default: the output is intended to be
// handed directly to Embree, so unresolved source ownership or non-watertight
// coverage aborts before a JSON file is created.  --allow-non-watertight is a
// regression/debug escape hatch and labels its artifact diagnostic-only.

#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "mfem_raytracing/reduction/hard_seam_bilinearization.hpp"
#include "mfem_raytracing/tspline/tspline_shell_composer.hpp"
#include "mfem_raytracing/tspline/tspline_shell_json.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

void PrintUsage()
{
    std::cerr
        << "usage: export_tspline_bilinear_shell --catalog PATH --leaves PATH --json PATH [options]\n"
        << "  --catalog PATH              source NURBS patch catalog\n"
        << "  --leaves PATH               independent hard-seam bilinear leaf JSON\n"
        << "  --json PATH                 composed baked-shell output JSON\n"
        << "  --max-error X               conservative final error limit (default: input max_error)\n"
        << "  --seam-band-rows N          source leaf rows retained on each seam side (default: 1)\n"
        << "  --allow-non-watertight      write a diagnostic-only artifact instead of failing the RT gate\n"
        << "  --compatibility-overlap     use legacy overlapping T-spline strips for comparison only\n";
}

std::string NextValue(int &i, int argc, char **argv, const char *flag)
{
    if (i + 1 >= argc)
    {
        throw std::invalid_argument(std::string("missing value for ") + flag);
    }
    return argv[++i];
}

void PrintReport(const mfem_raytracing::tspline::BakedTsplineShell &shell)
{
    const auto &errors = shell.errors.accounting;
    const auto &water = shell.watertightness;
    std::cout << "leaves: " << shell.leaves.size() << " (" << shell.strips.size()
              << " seam strips)\n"
              << "errors: source=" << errors.max_source_reduction_error
              << ", seam=" << errors.max_seam_displacement
              << ", bake=" << errors.max_bake_decomposition_error
              << ", conservative=" << errors.max_conservative_error << "\n"
              << "ownership: " << shell.ownership.raw_claim_count << " raw claims, "
              << shell.ownership.unique_claimed_source_leaf_count << " unique source leaves, "
              << shell.ownership.conflicts.size() << " conflicts\n"
              << "watertightness: " << (water.watertight ? "PASS" : "FAIL")
              << " (unowned=" << water.unowned_source_leaf_count
              << ", multiply-owned=" << water.multiply_owned_source_leaf_count
              << ", open-spans=" << water.open_edge_span_count
              << ", nonmanifold-spans=" << water.nonmanifold_edge_span_count << ")\n";
}

} // namespace

int main(int argc, char **argv)
{
    std::string catalog_path;
    std::string leaves_path;
    std::string json_path;
    double maximum_error = std::numeric_limits<double>::quiet_NaN();
    std::size_t seam_band_rows = 1;
    bool allow_non_watertight = false;
    bool compatibility_overlap = false;

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--catalog") { catalog_path = NextValue(i, argc, argv, "--catalog"); }
            else if (arg == "--leaves") { leaves_path = NextValue(i, argc, argv, "--leaves"); }
            else if (arg == "--json") { json_path = NextValue(i, argc, argv, "--json"); }
            else if (arg == "--max-error")
            {
                maximum_error = std::stod(NextValue(i, argc, argv, "--max-error"));
            }
            else if (arg == "--seam-band-rows")
            {
                seam_band_rows = static_cast<std::size_t>(
                    std::stoull(NextValue(i, argc, argv, "--seam-band-rows")));
            }
            else if (arg == "--allow-non-watertight") { allow_non_watertight = true; }
            else if (arg == "--compatibility-overlap") { compatibility_overlap = true; }
            else if (arg == "--help" || arg == "-h") { PrintUsage(); return 0; }
            else
            {
                throw std::invalid_argument("unknown argument '" + arg + "'");
            }
        }
        if (catalog_path.empty() || leaves_path.empty() || json_path.empty())
        {
            PrintUsage();
            return 1;
        }
        if (seam_band_rows == 0)
        {
            throw std::invalid_argument("--seam-band-rows must be positive");
        }

        const mfem_raytracing::SurfacePatchCatalog catalog =
            mfem_raytracing::LoadSurfacePatchCatalogJson(catalog_path);
        const mfem_raytracing::LeafPatchScene input =
            mfem_raytracing::LoadLeafPatchScene(leaves_path);
        if (!std::isfinite(maximum_error)) { maximum_error = input.max_error; }
        if (!(maximum_error >= 0.0) || !std::isfinite(maximum_error))
        {
            throw std::invalid_argument("--max-error must be finite and non-negative, or input must provide it");
        }

        mfem_raytracing::tspline::ShellBuildOptions options;
        options.seam_band_rows = seam_band_rows;
        options.error_validation.maximum_conservative_error = maximum_error;
        // Compatibility is an explicit construction choice.  Merely allowing
        // a diagnostic write must not silently replace the exact partition.
        options.corner_policy = compatibility_overlap ?
            mfem_raytracing::tspline::CornerOwnershipPolicy::CompatibilityOverlap :
            mfem_raytracing::tspline::CornerOwnershipPolicy::ExactBoundaryCornerCollar;
        const mfem_raytracing::tspline::BakedTsplineShell shell =
            mfem_raytracing::tspline::ComposeBakedTsplineShell(input, catalog, options);
        PrintReport(shell);

        if (!allow_non_watertight)
        {
            // This is deliberately immediately before the artifact becomes an
            // RT input.  It gives all downstream render paths one certified
            // invariant instead of duplicating partial checks.
            mfem_raytracing::tspline::RequireShellReadyForRayTracing(shell);
        }

        std::ofstream output(json_path);
        if (!output)
        {
            throw std::runtime_error("cannot write '" + json_path + "'");
        }
        mfem_raytracing::tspline::WriteBakedTsplineShellJson(output, shell);
        std::cout << "wrote " << json_path << " ("
                  << (shell.ReadyForRayTracing() ? "RT-certified" : "diagnostic-only") << ")\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
