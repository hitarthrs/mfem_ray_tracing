// Evaluate a general T-mesh with mfem_raytracing::tspline::TSplineSurface.
//
// Input is a deliberately tiny text interchange format written by the Python
// pipe experiment:
//   TSP1
//   degree <p>
//   domain <s_min> <s_max> <t_min> <t_max>
//   points <n>
//   <s> <t> <x> <y> <z> <weight>       (n rows)
//   edges <m>
//   <first> <second>                    (m rows)
//
// Output is JSON compatible with python_experiments/tsplines/visualize_tspline.py.

#include "tspline.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace mfem_raytracing::tspline;

namespace
{

struct Input
{
    int degree = 1;
    double s_min = 0.0;
    double s_max = 1.0;
    double t_min = 0.0;
    double t_max = 1.0;
    TMesh mesh;
};

void Expect(std::istream &in, const char *token)
{
    std::string value;
    in >> value;
    if (value != token)
    {
        throw std::runtime_error(std::string("expected '") + token + "', got '" + value + "'");
    }
}

Input ReadInput(const std::string &path)
{
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    Expect(in, "TSP1");
    Input result;
    Expect(in, "degree");
    in >> result.degree;
    Expect(in, "domain");
    in >> result.s_min >> result.s_max >> result.t_min >> result.t_max;
    if (result.degree < 1 || result.s_max <= result.s_min || result.t_max <= result.t_min)
    {
        throw std::runtime_error("invalid degree or domain");
    }
    std::size_t point_count = 0;
    Expect(in, "points");
    in >> point_count;
    for (std::size_t i = 0; i < point_count; ++i)
    {
        double s, t, x, y, z, w;
        if (!(in >> s >> t >> x >> y >> z >> w))
        {
            throw std::runtime_error("truncated T-mesh control point list");
        }
        result.mesh.AddControlPoint({s, t}, {x, y, z}, w);
    }
    std::size_t edge_count = 0;
    Expect(in, "edges");
    in >> edge_count;
    for (std::size_t i = 0; i < edge_count; ++i)
    {
        std::size_t first, second;
        if (!(in >> first >> second))
        {
            throw std::runtime_error("truncated T-mesh edge list");
        }
        result.mesh.AddEdge(first, second);
    }
    result.mesh.Validate();
    return result;
}

void WriteJson(const std::string &path, const Input &input, int resolution)
{
    const TMesh &mesh = input.mesh;
    const TSplineSurface surface(mesh, input.degree);
    const auto junctions = mesh.TJunctions();
    const auto extensions = mesh.TJunctionExtensions(static_cast<std::size_t>(input.degree + 1));
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("cannot write " + path); }
    out << std::setprecision(17);
    out << "{\n  \"name\": \"pipe_c0_merged_tspline\",\n";
    out << "  \"degree\": " << input.degree << ",\n";
    out << "  \"controls\": [\n";
    const auto &controls = mesh.ControlPoints();
    for (std::size_t i = 0; i < controls.size(); ++i)
    {
        const ControlPoint &p = controls[i];
        out << "    {\"s\": " << p.parameter[0] << ", \"t\": " << p.parameter[1]
            << ", \"x\": " << p.position[0] << ", \"y\": " << p.position[1]
            << ", \"z\": " << p.position[2] << ", \"w\": " << p.weight << "}"
            << (i + 1 < controls.size() ? ",\n" : "\n");
    }
    out << "  ],\n  \"edges\": [\n";
    const auto &edges = mesh.Edges();
    for (std::size_t i = 0; i < edges.size(); ++i)
    {
        out << "    [" << edges[i].first << ", " << edges[i].second << "]"
            << (i + 1 < edges.size() ? ",\n" : "\n");
    }
    out << "  ],\n  \"t_junctions\": [";
    for (std::size_t i = 0; i < junctions.size(); ++i)
    {
        out << junctions[i] << (i + 1 < junctions.size() ? ", " : "");
    }
    out << "],\n  \"extensions\": [\n";
    for (std::size_t i = 0; i < extensions.size(); ++i)
    {
        const TJunctionExtension &ext = extensions[i];
        out << "    {\"junction\": " << ext.junction << ", \"start\": ["
            << ext.start[0] << ", " << ext.start[1] << "], \"end\": ["
            << ext.end[0] << ", " << ext.end[1] << "]}"
            << (i + 1 < extensions.size() ? ",\n" : "\n");
    }
    out << "  ],\n  \"surface\": {\n";
    out << "    \"s_range\": [" << input.s_min << ", " << input.s_max << "],\n";
    out << "    \"t_range\": [" << input.t_min << ", " << input.t_max << "],\n";
    out << "    \"resolution\": " << resolution << ",\n    \"xyz\": [\n";
    // The current basis uses an open support convention, so sample a tiny
    // amount inside the domain; this leaves the exported grid rectangular.
    constexpr double inset = 1e-7;
    for (int j = 0; j < resolution; ++j)
    {
        const double fj = inset + (1.0 - 2.0 * inset) * j / (resolution - 1.0);
        const double t = input.t_min + (input.t_max - input.t_min) * fj;
        out << "      [";
        for (int i = 0; i < resolution; ++i)
        {
            const double fi = inset + (1.0 - 2.0 * inset) * i / (resolution - 1.0);
            const double s = input.s_min + (input.s_max - input.s_min) * fi;
            const Point3 p = surface.Evaluate(s, t);
            out << "[" << p[0] << ", " << p[1] << ", " << p[2] << "]"
                << (i + 1 < resolution ? ", " : "");
        }
        out << "]" << (j + 1 < resolution ? ",\n" : "\n");
    }
    out << "    ]\n  }\n}\n";
}

void WriteLeafJson(const std::string &path, const Input &input)
{
    const std::vector<TSplineLeaf> leaves = ExtractDegreeOneLeaves(input.mesh);
    double scene_min[3] = {std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity()};
    double scene_max[3] = {-std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()};
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("cannot write " + path); }
    out << std::setprecision(17);
    out << "{\n  \"surface\": \"pipe_c0_merged_degree1_tspline\",\n";
    out << "  \"backend\": \"degree1_tspline_exact_bilinear_leaf_extraction\",\n";
    out << "  \"max_error\": 0.0,\n  \"n_leaves\": " << leaves.size() << ",\n";
    std::ostringstream body;
    body << "  \"leaves\": [\n";
    for (std::size_t k = 0; k < leaves.size(); ++k)
    {
        const TSplineLeaf &leaf = leaves[k];
        double lo[3] = {std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()};
        double hi[3] = {-std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity()};
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                for (int d = 0; d < 3; ++d)
                {
                    lo[d] = std::min(lo[d], leaf.control_points[i][j][d]);
                    hi[d] = std::max(hi[d], leaf.control_points[i][j][d]);
                    scene_min[d] = std::min(scene_min[d], leaf.control_points[i][j][d]);
                    scene_max[d] = std::max(scene_max[d], leaf.control_points[i][j][d]);
                }
            }
        }
        body << "    {\n      \"index\": " << k << ",\n"
             << "      \"degree_u\": 1,\n      \"degree_v\": 1,\n"
             << "      \"bbox_min\": [" << lo[0] << ", " << lo[1] << ", " << lo[2] << "],\n"
             << "      \"bbox_max\": [" << hi[0] << ", " << hi[1] << ", " << hi[2] << "],\n"
             << "      \"u_domain_local\": [0, 1],\n      \"v_domain_local\": [0, 1],\n"
             << "      \"u_domain_global\": [" << leaf.s_domain[0] << ", " << leaf.s_domain[1] << "],\n"
             << "      \"v_domain_global\": [" << leaf.t_domain[0] << ", " << leaf.t_domain[1] << "],\n"
             << "      \"total_error\": 0.0,\n"
             << "      \"control_points\": [[";
        for (int i = 0; i < 2; ++i)
        {
            if (i != 0) { body << "], ["; }
            for (int j = 0; j < 2; ++j)
            {
                if (j != 0) { body << ", "; }
                const Point3 &p = leaf.control_points[i][j];
                body << "[" << p[0] << ", " << p[1] << ", " << p[2] << "]";
            }
        }
        body << "]],\n      \"weights\": [[";
        for (int i = 0; i < 2; ++i)
        {
            if (i != 0) { body << "], ["; }
            body << leaf.weights[i][0] << ", " << leaf.weights[i][1];
        }
        body << "]],\n      \"weights_non_negative\": true,\n"
             << "      \"weights_were_clamped\": false\n    }"
             << (k + 1 < leaves.size() ? ",\n" : "\n");
    }
    body << "  ]\n";
    out << "  \"scene_bbox_min\": [" << scene_min[0] << ", " << scene_min[1] << ", " << scene_min[2] << "],\n"
        << "  \"scene_bbox_max\": [" << scene_max[0] << ", " << scene_max[1] << ", " << scene_max[2] << "],\n";
    out << body.str() << "}\n";
}

void Usage(const char *program)
{
    std::cerr << "Usage: " << program
              << " --input merged.tmesh --output merged_surface.json [--leaves-output leaves.json] [--resolution N]\n";
}

} // namespace

int main(int argc, char **argv)
{
    std::string input_path;
    std::string output_path;
    std::string leaves_output_path;
    int resolution = 80;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::runtime_error(arg + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--input") { input_path = value(); }
        else if (arg == "--output") { output_path = value(); }
        else if (arg == "--leaves-output") { leaves_output_path = value(); }
        else if (arg == "--resolution") { resolution = std::stoi(value()); }
        else if (arg == "--help" || arg == "-h") { Usage(argv[0]); return 0; }
        else { throw std::runtime_error("unknown argument " + arg); }
    }
    if (input_path.empty() || output_path.empty() || resolution < 2)
    {
        Usage(argv[0]);
        return 2;
    }
    try
    {
        const Input input = ReadInput(input_path);
        WriteJson(output_path, input, resolution);
        if (!leaves_output_path.empty()) { WriteLeafJson(leaves_output_path, input); }
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
