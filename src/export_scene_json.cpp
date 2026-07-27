// Exports a bilinear leaf-patch scene plus a grid of traced rays to JSON, for
// the interactive web viewer. Each ray records its origin, direction, every hit
// along the path (multi-patch pierce via IntersectAll), and a far endpoint so
// the viewer can draw the beam continuing through successive surfaces.
//
// Usage: export_scene_json <leaf_bboxes.json> <out.json> [grid_n]

#include "embree/leaf_patch_loader.hpp"
#include "embree/raytracer.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace mfem_raytracing;

namespace
{

void WriteVec(std::ostream &os, const double v[3])
{
    os << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: export_scene_json <leaf_bboxes.json> <out.json> [grid_n]\n";
        return 1;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argv[2];
    const int grid_n = (argc > 3) ? std::max(2, std::atoi(argv[3])) : 16;

    LeafPatchScene scene;
    try
    {
        scene = LoadLeafPatchScene(in_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    EmbreeRayTracer tracer;
    tracer.RegisterPatches(scene.Patches());
    tracer.CommitScene();

    const double *lo = scene.scene_bbox.min;
    const double *hi = scene.scene_bbox.max;

    // Oblique parallel rays entering from above, so the viewer sees them enter,
    // hit the surface, and keep going underneath.
    double dir[3] = {0.28, 0.14, -1.0};
    const double dlen = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (double &c : dir)
    {
        c /= dlen;
    }
    const double span_z = (hi[2] - lo[2]);
    const double span_x = (hi[0] - lo[0]);
    const double span_y = (hi[1] - lo[1]);
    const double scene_diag = std::sqrt(span_x * span_x + span_y * span_y + span_z * span_z);
    const double start_z = hi[2] + 0.35 * span_z + 1.0;
    // Run well past the box so oblique rays have room to traverse multi-shell
    // scenes and keep visibly exiting after the final hit.
    const double t_end = (start_z - lo[2]) / std::fabs(dir[2]) + 4.0 * std::max(1.0, scene_diag);

    std::ofstream os(out_path);
    if (!os)
    {
        std::cerr << "error: cannot write '" << out_path << "'\n";
        return 1;
    }
    os << std::setprecision(7);

    os << "{\n  \"surface\": \"" << scene.surface_name << "\",\n";
    os << "  \"bbox_min\": ";
    WriteVec(os, lo);
    os << ",\n  \"bbox_max\": ";
    WriteVec(os, hi);
    os << ",\n";

    // Patches: corners wound P00 -> P10 -> P11 -> P01 for quad rendering.
    os << "  \"patches\": [";
    for (std::size_t i = 0; i < scene.leaves.size(); ++i)
    {
        const auto &cp = scene.leaves[i].patch.control_points;
        const int order[4] = {static_cast<int>(BilinearCorner::P00),
                              static_cast<int>(BilinearCorner::P10),
                              static_cast<int>(BilinearCorner::P11),
                              static_cast<int>(BilinearCorner::P01)};
        os << (i ? ",\n    " : "\n    ") << "[";
        for (int k = 0; k < 4; ++k)
        {
            if (k)
            {
                os << ',';
            }
            WriteVec(os, cp[order[k]]);
        }
        os << "]";
    }
    os << "\n  ],\n";

    // Rays over a grid in the xy footprint (all hits along each ray).
    os << "  \"rays\": [";
    long ray_hit_count = 0;   // rays with ≥1 hit
    long total_hit_count = 0; // sum of intersections across all rays
    bool first_ray = true;
    for (int iy = 0; iy < grid_n; ++iy)
    {
        for (int ix = 0; ix < grid_n; ++ix)
        {
            const double fx = (ix + 0.5) / grid_n;
            const double fy = (iy + 0.5) / grid_n;
            // Back the origin up in xy so the oblique ray lands inside the footprint.
            const double x = lo[0] + fx * (hi[0] - lo[0]) - dir[0] * (start_z - hi[2]);
            const double y = lo[1] + fy * (hi[1] - lo[1]) - dir[1] * (start_z - hi[2]);
            const double origin[3] = {x, y, start_z};

            const std::vector<RayHitRecord> hits = tracer.IntersectAll(origin, dir, 0.0, t_end);

            os << (first_ray ? "\n    " : ",\n    ") << "{\"o\":";
            WriteVec(os, origin);
            os << ",\"d\":[" << dir[0] << ',' << dir[1] << ',' << dir[2] << "],\"tEnd\":" << t_end;
            os << ",\"hits\":[";
            for (std::size_t hi = 0; hi < hits.size(); ++hi)
            {
                if (hi)
                {
                    os << ',';
                }
                os << "{\"t\":" << hits[hi].t << ",\"prim\":" << hits[hi].prim_id << ",\"n\":";
                WriteVec(os, hits[hi].Ng);
                os << '}';
            }
            os << ']';
            // Backward-compatible first-hit fields for older viewers.
            if (!hits.empty())
            {
                ++ray_hit_count;
                total_hit_count += static_cast<long>(hits.size());
                os << ",\"tHit\":" << hits[0].t << ",\"prim\":" << hits[0].prim_id << ",\"n\":";
                WriteVec(os, hits[0].Ng);
            }
            else
            {
                os << ",\"tHit\":null";
            }
            os << "}";
            first_ray = false;
        }
    }
    os << "\n  ]\n}\n";

    std::cout << "exported " << scene.leaves.size() << " patches, " << (grid_n * grid_n)
              << " rays (" << ray_hit_count << " with hits, " << total_hit_count
              << " total intersections) -> " << out_path << "\n";
    return 0;
}
