// Sample cubic T-spline surfaces from the Sederberg-style kernel and write JSON
// for Python visualization (control net + dense surface grid).
//
// Usage:
//   ./export_tspline_surface [--resolution 48] [--outdir PATH]
//
// Writes:
//   <outdir>/tspline_tensor_surface.json
//   <outdir>/tspline_tjunction_surface.json

#include "tspline.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mfem_raytracing::tspline;

namespace
{

double Hill(double s, double t)
{
    const double ds = s - 2.0;
    const double dt = t - 2.0;
    return 1.15 * std::exp(-0.35 * (ds * ds + dt * dt));
}

TMesh MakeTensorGrid(int n)
{
    TMesh mesh;
    for (int t = 0; t < n; ++t)
    {
        for (int s = 0; s < n; ++s)
        {
            const double ps = static_cast<double>(s);
            const double pt = static_cast<double>(t);
            mesh.AddControlPoint({ps, pt}, {ps, pt, Hill(ps, pt)});
        }
    }
    const auto id = [n](int s, int t) {
        return static_cast<std::size_t>(t * n + s);
    };
    for (int t = 0; t < n; ++t)
    {
        for (int s = 0; s < n; ++s)
        {
            if (s + 1 < n) { mesh.AddEdge(id(s, t), id(s + 1, t)); }
            if (t + 1 < n) { mesh.AddEdge(id(s, t), id(s, t + 1)); }
        }
    }
    return mesh;
}

/** 5×5 tensor product with the right arm of the center cross removed (two T-junctions). */
TMesh MakeTJunctionMesh()
{
    constexpr int n = 5;
    TMesh full = MakeTensorGrid(n);
    TMesh mesh;
    for (const ControlPoint &p : full.ControlPoints())
    {
        mesh.AddControlPoint(p.parameter, p.position, p.weight);
    }
    // Center vertex index on a 5×5 row-major grid: t=2,s=2 → 12; neighbour s+1 → 13.
    constexpr std::size_t center = 12;
    constexpr std::size_t right = 13;
    for (const Edge &edge : full.Edges())
    {
        const bool drop = (edge.first == center && edge.second == right) ||
                          (edge.first == right && edge.second == center);
        if (!drop) { mesh.AddEdge(edge.first, edge.second); }
    }
    return mesh;
}

std::string Escape(const std::string &s)
{
    return s; // names are plain identifiers
}

void WriteJson(const std::string &path, const std::string &name, const TMesh &mesh,
               int degree, int resolution)
{
    mesh.Validate();
    const TSplineSurface surface(mesh, degree);
    const auto &points = mesh.ControlPoints();
    const auto junctions = mesh.TJunctions();
    const auto extensions = mesh.TJunctionExtensions(2);

    // Interior of the knot domain [0, n-1]^2 — avoid boundary where denom → 0.
    const double s_lo = 0.5;
    const double t_lo = 0.5;
    const double s_hi = 3.5;
    const double t_hi = 3.5;

    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"name\": \"" << Escape(name) << "\",\n";
    out << "  \"degree\": " << degree << ",\n";
    out << "  \"controls\": [\n";
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const ControlPoint &p = points[i];
        out << "    {\"s\": " << p.parameter[0] << ", \"t\": " << p.parameter[1]
            << ", \"x\": " << p.position[0] << ", \"y\": " << p.position[1]
            << ", \"z\": " << p.position[2] << ", \"w\": " << p.weight << "}";
        out << (i + 1 < points.size() ? ",\n" : "\n");
    }
    out << "  ],\n";
    out << "  \"edges\": [\n";
    for (std::size_t i = 0; i < mesh.Edges().size(); ++i)
    {
        const Edge &e = mesh.Edges()[i];
        out << "    [" << e.first << ", " << e.second << "]";
        out << (i + 1 < mesh.Edges().size() ? ",\n" : "\n");
    }
    out << "  ],\n";
    out << "  \"t_junctions\": [";
    for (std::size_t i = 0; i < junctions.size(); ++i)
    {
        out << junctions[i] << (i + 1 < junctions.size() ? ", " : "");
    }
    out << "],\n";
    out << "  \"extensions\": [\n";
    for (std::size_t i = 0; i < extensions.size(); ++i)
    {
        const TJunctionExtension &ext = extensions[i];
        out << "    {\"junction\": " << ext.junction
            << ", \"start\": [" << ext.start[0] << ", " << ext.start[1] << "]"
            << ", \"end\": [" << ext.end[0] << ", " << ext.end[1] << "]}";
        out << (i + 1 < extensions.size() ? ",\n" : "\n");
    }
    out << "  ],\n";

    out << "  \"surface\": {\n";
    out << "    \"s_range\": [" << s_lo << ", " << s_hi << "],\n";
    out << "    \"t_range\": [" << t_lo << ", " << t_hi << "],\n";
    out << "    \"resolution\": " << resolution << ",\n";
    out << "    \"xyz\": [\n";
    for (int j = 0; j < resolution; ++j)
    {
        const double t = t_lo + (t_hi - t_lo) * static_cast<double>(j) /
                                    static_cast<double>(resolution - 1);
        out << "      [";
        for (int i = 0; i < resolution; ++i)
        {
            const double s = s_lo + (s_hi - s_lo) * static_cast<double>(i) /
                                        static_cast<double>(resolution - 1);
            Point3 p = {0.0, 0.0, 0.0};
            try
            {
                p = surface.Evaluate(s, t);
            }
            catch (const std::out_of_range &)
            {
                p = {s, t, 0.0}; // rare hole; keep grid shape
            }
            out << "[" << p[0] << ", " << p[1] << ", " << p[2] << "]";
            if (i + 1 < resolution) { out << ", "; }
        }
        out << "]";
        out << (j + 1 < resolution ? ",\n" : "\n");
    }
    out << "    ]\n";
    out << "  }\n";
    out << "}\n";

    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error("cannot write " + path);
    }
    file << out.str();
}

void PrintUsage(const char *argv0)
{
    std::cerr << "Usage: " << argv0
              << " [--resolution N] [--outdir PATH] [--degree P]\n";
}

} // namespace

int main(int argc, char **argv)
{
    int resolution = 48;
    int degree = 3;
    std::string outdir = "python_experiments/tsplines/outputs";

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto need = [&](const char *flag) -> std::string {
            if (i + 1 >= argc)
            {
                std::cerr << "error: " << flag << " needs a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--resolution") { resolution = std::stoi(need("--resolution")); }
        else if (arg == "--degree") { degree = std::stoi(need("--degree")); }
        else if (arg == "--outdir") { outdir = need("--outdir"); }
        else if (arg == "-h" || arg == "--help")
        {
            PrintUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "error: unknown argument '" << arg << "'\n";
            PrintUsage(argv[0]);
            return 2;
        }
    }
    if (resolution < 2)
    {
        std::cerr << "error: --resolution must be >= 2\n";
        return 2;
    }

    try
    {
        const std::string tensor_path = outdir + "/tspline_tensor_surface.json";
        const std::string tj_path = outdir + "/tspline_tjunction_surface.json";
        WriteJson(tensor_path, "tensor", MakeTensorGrid(5), degree, resolution);
        std::cout << "wrote " << tensor_path << '\n';
        WriteJson(tj_path, "tjunction", MakeTJunctionMesh(), degree, resolution);
        std::cout << "wrote " << tj_path << '\n';
    }
    catch (const std::exception &ex)
    {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
    return 0;
}
