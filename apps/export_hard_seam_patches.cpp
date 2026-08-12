// Export selected patches from a NURBS patch catalog through the reusable
// world-coordinate hard-seam bilinearization pipeline.

#include "mfem_raytracing/reduction/hard_seam_bilinearization.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::vector<int> ParsePatchIds(const std::string &spec)
{
    std::vector<int> ids;
    std::size_t begin = 0;
    while (begin < spec.size())
    {
        const std::size_t end = spec.find(',', begin);
        const std::string part = spec.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        const std::size_t dash = part.find('-');
        if (dash == std::string::npos)
        {
            ids.push_back(std::stoi(part));
        }
        else
        {
            const int lo = std::stoi(part.substr(0, dash));
            const int hi = std::stoi(part.substr(dash + 1));
            if (hi < lo) { throw std::invalid_argument("patch range must be ascending"); }
            for (int id = lo; id <= hi; ++id) { ids.push_back(id); }
        }
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (ids.empty()) { throw std::invalid_argument("no patch ids were parsed"); }
    return ids;
}

void PrintUsage()
{
    std::cerr
        << "usage: export_hard_seam_patches --catalog PATH --patches IDS --max-error X --json PATH\n"
        << "  IDS accepts 0-7, 0,1,4, or 0-3,6.\n"
        << "  --surface NAME is optional metadata for the output JSON.\n"
        << "  Uses world-coordinate, approach_4 hard-seam reduction; there is no cage-centering mode.\n";
}

} // namespace

int main(int argc, char **argv)
{
    std::string catalog_path;
    std::string patch_spec = "1";
    std::string json_path;
    std::string surface_name = "hard_seam_bilinearization";
    double max_error = -1.0;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto value = [&](const char *flag) -> std::string {
            if (++i >= argc)
            {
                throw std::invalid_argument(std::string("missing value for ") + flag);
            }
            return argv[i];
        };
        if (arg == "--catalog") { catalog_path = value("--catalog"); }
        else if (arg == "--patches") { patch_spec = value("--patches"); }
        else if (arg == "--max-error") { max_error = std::stod(value("--max-error")); }
        else if (arg == "--json") { json_path = value("--json"); }
        else if (arg == "--surface") { surface_name = value("--surface"); }
        else if (arg == "--help" || arg == "-h") { PrintUsage(); return 0; }
        else { std::cerr << "error: unknown argument '" << arg << "'\n"; PrintUsage(); return 1; }
    }
    if (catalog_path.empty() || json_path.empty() || max_error < 0.0)
    {
        PrintUsage();
        return 1;
    }

    try
    {
        const mfem_raytracing::SurfacePatchCatalog catalog =
            mfem_raytracing::LoadSurfacePatchCatalogJson(catalog_path);
        const std::vector<int> patch_ids = ParsePatchIds(patch_spec);
        mfem_raytracing::HardSeamBilinearizationOptions options;
        options.max_error = max_error;
        const mfem_raytracing::HardSeamBilinearizationResult result =
            mfem_raytracing::BilinearizePatchesWithHardSeams(catalog, patch_ids, options);

        std::ofstream out(json_path);
        if (!out)
        {
            throw std::runtime_error("cannot write '" + json_path + "'");
        }
        mfem_raytracing::WriteHardSeamBilinearizationJson(out, result, surface_name);

        const std::size_t n_competing = result.CompetingCount();
        const std::size_t n_leaves = result.LeafCount();
        const long long delta =
            static_cast<long long>(n_competing) - static_cast<long long>(n_leaves);
        const double pct =
            n_competing > 0 ? 100.0 * static_cast<double>(delta) / static_cast<double>(n_competing)
                            : 0.0;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "wrote " << json_path << " (" << n_leaves << " leaves)\n";
        std::cout << "timing: compete " << result.seconds_compete << "s  coalesce "
                  << result.seconds_coalesce << "s  total "
                  << (result.seconds_compete + result.seconds_coalesce) << "s\n";
        std::cout << "coalesce: " << n_competing << " competing -> " << n_leaves
                  << " leaves  delta " << delta << " (" << std::setprecision(1) << pct
                  << "%); positive=saved by merge, negative=extra 2x2 from hard seams\n";
        std::cout << std::setprecision(3);
        for (const auto &patch : result.patches)
        {
            const long long saved =
                static_cast<long long>(patch.n_competing) -
                static_cast<long long>(patch.leaves.size());
            std::cout << "  p" << patch.source.id << ": " << patch.n_competing
                      << " competing -> " << patch.leaves.size() << " leaves  delta " << saved
                      << "  compete " << patch.seconds_compete << "s  coalesce "
                      << patch.seconds_coalesce << "s  " << patch.source.role << "  "
                      << patch.source.name << "\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
