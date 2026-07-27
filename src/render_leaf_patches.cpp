// Renders a bilinear leaf-patch scene (d4_leaf_bboxes.json export) with the
// Embree ray tracer and writes PPM images:
//   <prefix>_shaded.ppm  — perspective view, patch-id color × Lambertian
//   <prefix>_surface.ppm — perspective view, solid albedo × Lambertian
//                          (shows the surface shape from ray hits only)
//   <prefix>_uv.ppm      — perspective view, colored by global (u, v) on the
//                          original surface (continuity across leaf seams is a
//                          visual check of the uv mapping)
//   <prefix>_top.ppm     — top-down orthographic height map
//
// Usage: render_leaf_patches <leaf_bboxes.json> <output_prefix> [image_size]
//                           [--diagnose] [--rt-diagnostics] [--bruteforce-grid N]
//                           [--multihit-grid N]

#include "embree/leaf_patch_loader.hpp"
#include "embree/raytracer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace mfem_raytracing;

namespace
{

struct Vec3
{
    double x = 0.0, y = 0.0, z = 0.0;
};

Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(double s, const Vec3 &a) { return {s * a.x, s * a.y, s * a.z}; }
double Dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 Cross(const Vec3 &a, const Vec3 &b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
Vec3 Normalized(const Vec3 &a)
{
    const double len = std::sqrt(Dot(a, a));
    return (len > 0.0) ? (1.0 / len) * a : a;
}

struct Image
{
    int width = 0, height = 0;
    std::vector<unsigned char> rgb; // 3 bytes per pixel, row-major from top

    Image(int w, int h) : width(w), height(h), rgb(static_cast<std::size_t>(w) * h * 3, 0) {}

    void Set(int x, int y, double r, double g, double b)
    {
        auto to_byte = [](double c) {
            return static_cast<unsigned char>(std::lround(255.0 * std::clamp(c, 0.0, 1.0)));
        };
        const std::size_t k = 3 * (static_cast<std::size_t>(y) * width + x);
        rgb[k + 0] = to_byte(r);
        rgb[k + 1] = to_byte(g);
        rgb[k + 2] = to_byte(b);
    }

    bool WritePPM(const std::string &path) const
    {
        std::ofstream out(path, std::ios::binary);
        if (!out)
        {
            return false;
        }
        out << "P6\n" << width << " " << height << "\n255\n";
        out.write(reinterpret_cast<const char *>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
        return static_cast<bool>(out);
    }
};

// Distinct-ish color per patch id (golden-ratio hue walk).
void PatchIdColor(unsigned int id, double &r, double &g, double &b)
{
    const double hue = std::fmod(0.618033988749895 * (id + 1), 1.0);
    const double h6 = 6.0 * hue;
    const int sector = static_cast<int>(h6) % 6;
    const double f = h6 - std::floor(h6);
    const double p = 0.25, q = 1.0 - 0.75 * f, t = 0.25 + 0.75 * f;
    switch (sector)
    {
        case 0: r = 1.0; g = t;   b = p;   break;
        case 1: r = q;   g = 1.0; b = p;   break;
        case 2: r = p;   g = 1.0; b = t;   break;
        case 3: r = p;   g = q;   b = 1.0; break;
        case 4: r = t;   g = p;   b = 1.0; break;
        default: r = 1.0; g = p;  b = q;   break;
    }
}

void RoleColor(const std::string &role, double &r, double &g, double &b)
{
    if (role == "interior")
    {
        r = 0.25; g = 0.65; b = 1.0;
    }
    else if (role == "seam-exact")
    {
        r = 1.0; g = 0.82; b = 0.10;
    }
    else if (role == "seam-phantom")
    {
        r = 1.0; g = 0.15; b = 0.85;
    }
    else
    {
        r = 0.85; g = 0.85; b = 0.85;
    }
}

void AddDiagnostics(RayQueryDiagnostics &total, const RayQueryDiagnostics &value)
{
    total.kernel_invocations += value.kernel_invocations;
    total.kernel_rejections += value.kernel_rejections;
    total.reported_hits += value.reported_hits;
    total.reject_invalid_ray += value.reject_invalid_ray;
    total.reject_no_root += value.reject_no_root;
    total.reject_denominator += value.reject_denominator;
    total.reject_residual += value.reject_residual;
    total.reject_domain += value.reject_domain;
    total.reject_weight += value.reject_weight;
    total.reject_t_range += value.reject_t_range;
}

double NormalizedLogCount(std::uint64_t count, std::uint64_t maximum)
{
    return (count == 0 || maximum == 0) ? 0.0 :
        std::log1p(static_cast<double>(count)) / std::log1p(static_cast<double>(maximum));
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: render_leaf_patches <leaf_bboxes.json> <output_prefix> [image_size]"
                  << " [--diagnose] [--rt-diagnostics] [--bruteforce-grid N]"
                  << " [--multihit-grid N]\n";
        return 1;
    }
    const std::string json_path = argv[1];
    const std::string prefix = argv[2];
    int size = 800;
    bool write_rt_diagnostics = false;
    bool diagnose = false;
    int brute_force_grid = 0;
    int multihit_grid = 0;
    bool size_seen = false;
    for (int i = 3; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--rt-diagnostics")
        {
            write_rt_diagnostics = true;
        }
        else if (arg == "--diagnose")
        {
            // The default sampled grids keep direct all-leaf validation fast
            // even for large patch scenes. Explicit grid flags may override.
            diagnose = true;
            write_rt_diagnostics = true;
        }
        else if (arg == "--bruteforce-grid")
        {
            if (++i >= argc)
            {
                std::cerr << "error: --bruteforce-grid requires a positive integer\n";
                return 1;
            }
            brute_force_grid = std::atoi(argv[i]);
            if (brute_force_grid < 1)
            {
                std::cerr << "error: --bruteforce-grid requires a positive integer\n";
                return 1;
            }
            write_rt_diagnostics = true;
        }
        else if (arg == "--multihit-grid")
        {
            if (++i >= argc)
            {
                std::cerr << "error: --multihit-grid requires a positive integer\n";
                return 1;
            }
            multihit_grid = std::atoi(argv[i]);
            if (multihit_grid < 1)
            {
                std::cerr << "error: --multihit-grid requires a positive integer\n";
                return 1;
            }
        }
        else if (!size_seen)
        {
            size = std::max(16, std::atoi(argv[i]));
            size_seen = true;
        }
        else
        {
            std::cerr << "error: unknown argument '" << arg << "'\n";
            return 1;
        }
    }
    if (diagnose)
    {
        if (brute_force_grid == 0)
        {
            brute_force_grid = 32;
        }
        if (multihit_grid == 0)
        {
            multihit_grid = 32;
        }
    }

    LeafPatchScene scene;
    try
    {
        scene = LoadLeafPatchScene(json_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    EmbreeRayTracer tracer;
    tracer.RegisterPatches(scene.Patches());
    tracer.CommitScene();

    const Vec3 lo = {scene.scene_bbox.min[0], scene.scene_bbox.min[1], scene.scene_bbox.min[2]};
    const Vec3 hi = {scene.scene_bbox.max[0], scene.scene_bbox.max[1], scene.scene_bbox.max[2]};
    const Vec3 center = 0.5 * (lo + hi);
    const Vec3 extent = hi - lo;
    const double radius = 0.5 * std::sqrt(Dot(extent, extent));

    std::cout << "scene '" << scene.surface_name << "': " << scene.leaves.size()
              << " leaves, bbox [" << lo.x << ", " << lo.y << ", " << lo.z << "] – [" << hi.x
              << ", " << hi.y << ", " << hi.z << "]\n";

    // ---------------- Perspective views (shaded + uv) ----------------
    // Camera on a 3/4 diagonal above the scene, looking at the center.
    const Vec3 eye = center + 2.2 * radius * Normalized(Vec3{-0.55, -1.0, 0.85});
    const Vec3 forward = Normalized(center - eye);
    const Vec3 right = Normalized(Cross(forward, Vec3{0.0, 0.0, 1.0}));
    const Vec3 up = Cross(right, forward);
    const double fov_scale = 0.62; // ~2*atan(0.62) ≈ 63° vertical FOV
    const Vec3 light_dir = Normalized(Vec3{0.35, -0.5, 0.85});

    Image shaded(size, size);
    Image surface(size, size);
    Image uv_view(size, size);
    Image roles(size, size);
    std::vector<RayQueryDiagnostics> diagnostics_by_pixel(
        static_cast<std::size_t>(size) * size);
    Image brute_force_disagreements(size, size);
    long hit_count = 0;
    RayQueryDiagnostics total_diagnostics;
    long zero_callback_rays = 0;
    std::uint64_t brute_force_samples = 0;
    std::uint64_t brute_force_bvh_misses = 0;
    std::uint64_t brute_force_embree_only_hits = 0;
    std::uint64_t brute_force_t_disagreements = 0;
    const int brute_force_stride = (brute_force_grid == 0) ? 0 :
        std::max(1, size / brute_force_grid);
    std::uint64_t multihit_samples = 0;
    std::uint64_t multihit_raw_hits = 0;
    std::uint64_t multihit_adjacent_pairs = 0;
    std::uint64_t multihit_pairs_le_1e5 = 0;
    std::uint64_t multihit_pairs_le_1e4 = 0;
    std::uint64_t multihit_pairs_le_1e3 = 0;
    std::uint64_t multihit_pairs_with_seam = 0;
    std::uint64_t multihit_pairs_with_phantom = 0;
    std::uint64_t multihit_expected_clustered_hits = 0;
    std::uint64_t multihit_intersect_all_hits = 0;
    std::uint64_t multihit_count_mismatches = 0;
    std::size_t multihit_max_raw_hits = 0;
    double multihit_min_gap = std::numeric_limits<double>::infinity();
    const int multihit_stride = (multihit_grid == 0) ? 0 :
        std::max(1, size / multihit_grid);
    // Neutral warm albedo so shading (normals from Embree hits) carries the shape.
    const double albedo_r = 0.82, albedo_g = 0.78, albedo_b = 0.70;

    for (int py = 0; py < size; ++py)
    {
        for (int px = 0; px < size; ++px)
        {
            const double sx = (2.0 * (px + 0.5) / size - 1.0) * fov_scale;
            const double sy = (1.0 - 2.0 * (py + 0.5) / size) * fov_scale;
            const Vec3 dir = Normalized(forward + sx * right + sy * up);

            const double org_arr[3] = {eye.x, eye.y, eye.z};
            const double dir_arr[3] = {dir.x, dir.y, dir.z};
            RayQueryDiagnostics diagnostics;
            const RayHitRecord hit = tracer.Intersect(org_arr, dir_arr, 0.0,
                                                       std::numeric_limits<double>::infinity(),
                                                       write_rt_diagnostics ? &diagnostics : nullptr);
            if (write_rt_diagnostics)
            {
                const std::size_t pixel = static_cast<std::size_t>(py) * size + px;
                diagnostics_by_pixel[pixel] = diagnostics;
                AddDiagnostics(total_diagnostics, diagnostics);
                zero_callback_rays += diagnostics.kernel_invocations == 0;
            }

            const bool brute_force_sample = brute_force_stride != 0 &&
                px % brute_force_stride == brute_force_stride / 2 &&
                py % brute_force_stride == brute_force_stride / 2;
            if (brute_force_sample)
            {
                ++brute_force_samples;
                const RayHitRecord brute_hit = tracer.IntersectBruteForce(org_arr, dir_arr);
                if (!hit.hit && brute_hit.hit)
                {
                    ++brute_force_bvh_misses;
                    brute_force_disagreements.Set(px, py, 1.0, 0.0, 0.0); // red: Embree missed
                }
                else if (hit.hit && !brute_hit.hit)
                {
                    ++brute_force_embree_only_hits;
                    brute_force_disagreements.Set(px, py, 1.0, 0.85, 0.0); // yellow: direct missed
                }
                else if (hit.hit && brute_hit.hit)
                {
                    const double t_tol = std::max(1e-4, 1e-5 * std::max(std::fabs(hit.t), std::fabs(brute_hit.t)));
                    if (std::fabs(hit.t - brute_hit.t) > t_tol)
                    {
                        ++brute_force_t_disagreements;
                        brute_force_disagreements.Set(px, py, 1.0, 0.0, 1.0); // magenta: different closest t
                    }
                }
            }

            const bool multihit_sample = multihit_stride != 0 &&
                px % multihit_stride == multihit_stride / 2 &&
                py % multihit_stride == multihit_stride / 2;
            if (multihit_sample)
            {
                ++multihit_samples;
                const std::vector<RayHitRecord> raw_hits = tracer.IntersectAllBruteForce(org_arr, dir_arr);
                const std::vector<RayHitRecord> all_hits = tracer.IntersectAll(org_arr, dir_arr);
                multihit_raw_hits += raw_hits.size();
                multihit_intersect_all_hits += all_hits.size();
                multihit_max_raw_hits = std::max(multihit_max_raw_hits, raw_hits.size());
                std::size_t expected_cluster_count = 0;
                double cluster_t = 0.0;
                for (std::size_t i = 1; i < raw_hits.size(); ++i)
                {
                    const double gap = raw_hits[i].t - raw_hits[i - 1].t;
                    ++multihit_adjacent_pairs;
                    multihit_min_gap = std::min(multihit_min_gap, gap);
                    multihit_pairs_le_1e5 += gap <= 1e-5;
                    multihit_pairs_le_1e4 += gap <= 1e-4;
                    multihit_pairs_le_1e3 += gap <= 1e-3;
                    const std::string &left_role = scene.leaves[raw_hits[i - 1].prim_id].role;
                    const std::string &right_role = scene.leaves[raw_hits[i].prim_id].role;
                    const bool has_phantom = left_role == "seam-phantom" || right_role == "seam-phantom";
                    const bool has_seam = has_phantom || left_role == "seam-exact" || right_role == "seam-exact";
                    multihit_pairs_with_phantom += has_phantom;
                    multihit_pairs_with_seam += has_seam;
                }
                for (const RayHitRecord &raw_hit : raw_hits)
                {
                    if (expected_cluster_count == 0)
                    {
                        ++expected_cluster_count;
                        cluster_t = raw_hit.t;
                        continue;
                    }
                    const double tolerance = std::max(1e-5, 1e-6 * std::max(std::fabs(raw_hit.t), std::fabs(cluster_t)));
                    if (raw_hit.t - cluster_t > tolerance)
                    {
                        ++expected_cluster_count;
                        cluster_t = raw_hit.t;
                    }
                }
                multihit_expected_clustered_hits += expected_cluster_count;
                multihit_count_mismatches += all_hits.size() != expected_cluster_count;
            }

            if (!hit.hit)
            {
                const double sky = 0.12 + 0.10 * (1.0 - (py + 0.5) / size);
                shaded.Set(px, py, sky, sky, sky + 0.03);
                surface.Set(px, py, sky, sky, sky + 0.03);
                uv_view.Set(px, py, sky, sky, sky + 0.03);
                continue;
            }
            ++hit_count;

            // Lambertian, double-sided, headlight-mixed.
            Vec3 n = Normalized(Vec3{hit.Ng[0], hit.Ng[1], hit.Ng[2]});
            if (Dot(n, dir) > 0.0)
            {
                n = -1.0 * n;
            }
            const double diffuse = std::max(0.0, Dot(n, light_dir));
            const double head = std::max(0.0, -Dot(n, dir));
            const double shade = 0.12 + 0.65 * diffuse + 0.23 * head;

            double pr, pg, pb;
            PatchIdColor(hit.prim_id, pr, pg, pb);
            shaded.Set(px, py, shade * pr, shade * pg, shade * pb);
            surface.Set(px, py, shade * albedo_r, shade * albedo_g, shade * albedo_b);

            // Global (u, v) on the original surface via the leaf's domain.
            const LeafPatch &leaf = scene.leaves[hit.prim_id];
            const double gu = leaf.u_domain_global[0] +
                              hit.u * (leaf.u_domain_global[1] - leaf.u_domain_global[0]);
            const double gv = leaf.v_domain_global[0] +
                              hit.v * (leaf.v_domain_global[1] - leaf.v_domain_global[0]);
            uv_view.Set(px, py, shade * gu, shade * gv, shade * (1.0 - 0.5 * (gu + gv)));

            if (write_rt_diagnostics)
            {
                double rr, rg, rb;
                RoleColor(leaf.role, rr, rg, rb);
                roles.Set(px, py, rr, rg, rb);
            }
        }
    }

    // ---------------- Top-down orthographic height map ----------------
    Image top(size, size);
    const double z_top = hi.z + 1.0;
    const double z_range = std::max(1e-12, hi.z - lo.z);
    const double dir_down[3] = {0.0, 0.0, -1.0};

    for (int py = 0; py < size; ++py)
    {
        for (int px = 0; px < size; ++px)
        {
            const double fx = (px + 0.5) / size;
            const double fy = 1.0 - (py + 0.5) / size;
            const double org_arr[3] = {lo.x + fx * (hi.x - lo.x), lo.y + fy * (hi.y - lo.y), z_top};

            const RayHitRecord hit = tracer.Intersect(org_arr, dir_down);
            if (!hit.hit)
            {
                top.Set(px, py, 0.10, 0.10, 0.12);
                continue;
            }
            const double z_hit = z_top - hit.t;
            const double h = std::clamp((z_hit - lo.z) / z_range, 0.0, 1.0);
            // Blue (low) -> teal -> yellow (high).
            top.Set(px, py, h, 0.35 + 0.55 * h, 0.9 - 0.65 * h);
        }
    }

    const std::string shaded_path = prefix + "_shaded.ppm";
    const std::string surface_path = prefix + "_surface.ppm";
    const std::string uv_path = prefix + "_uv.ppm";
    const std::string top_path = prefix + "_top.ppm";
    std::string callbacks_path;
    std::string rejections_path;
    std::string roles_path;
    std::string no_root_path;
    std::string denominator_path;
    std::string residual_path;
    std::string domain_path;
    std::string weight_path;
    std::string t_range_path;
    std::string brute_force_path;
    Image callbacks(size, size);
    Image rejections(size, size);
    Image no_root(size, size);
    Image denominator(size, size);
    Image residual(size, size);
    Image domain(size, size);
    Image weight(size, size);
    Image t_range(size, size);
    if (write_rt_diagnostics)
    {
        const auto max_count = [&](std::uint64_t RayQueryDiagnostics::*member) {
            std::uint64_t maximum = 0;
            for (const RayQueryDiagnostics &diagnostics : diagnostics_by_pixel)
            {
                maximum = std::max(maximum, diagnostics.*member);
            }
            return maximum;
        };
        const auto paint_count_map = [&](Image &image, std::uint64_t RayQueryDiagnostics::*member,
                                         double r, double g, double b) {
            const std::uint64_t maximum = max_count(member);
            for (int py = 0; py < size; ++py)
            {
                for (int px = 0; px < size; ++px)
                {
                    const std::size_t pixel = static_cast<std::size_t>(py) * size + px;
                    const double value = NormalizedLogCount(diagnostics_by_pixel[pixel].*member, maximum);
                    image.Set(px, py, value * r, value * g, value * b);
                }
            }
        };
        paint_count_map(callbacks, &RayQueryDiagnostics::kernel_invocations, 1.0, 0.45, 0.0);
        paint_count_map(rejections, &RayQueryDiagnostics::kernel_rejections, 1.0, 0.0, 0.0);
        paint_count_map(no_root, &RayQueryDiagnostics::reject_no_root, 0.9, 0.2, 0.0);
        paint_count_map(denominator, &RayQueryDiagnostics::reject_denominator, 0.7, 0.0, 1.0);
        paint_count_map(residual, &RayQueryDiagnostics::reject_residual, 0.0, 0.85, 1.0);
        paint_count_map(domain, &RayQueryDiagnostics::reject_domain, 0.0, 1.0, 0.25);
        paint_count_map(weight, &RayQueryDiagnostics::reject_weight, 1.0, 1.0, 1.0);
        paint_count_map(t_range, &RayQueryDiagnostics::reject_t_range, 1.0, 0.8, 0.0);
        callbacks_path = prefix + "_rt_callbacks.ppm";
        rejections_path = prefix + "_rt_rejections.ppm";
        roles_path = prefix + "_rt_roles.ppm";
        no_root_path = prefix + "_rt_reject_no_root.ppm";
        denominator_path = prefix + "_rt_reject_denominator.ppm";
        residual_path = prefix + "_rt_reject_residual.ppm";
        domain_path = prefix + "_rt_reject_domain.ppm";
        weight_path = prefix + "_rt_reject_weight.ppm";
        t_range_path = prefix + "_rt_reject_t_range.ppm";
        brute_force_path = prefix + "_rt_bruteforce_disagreements.ppm";
    }
    if (!shaded.WritePPM(shaded_path) || !surface.WritePPM(surface_path) ||
        !uv_view.WritePPM(uv_path) || !top.WritePPM(top_path) ||
        (write_rt_diagnostics &&
         (!callbacks.WritePPM(callbacks_path) || !rejections.WritePPM(rejections_path) ||
          !roles.WritePPM(roles_path) || !no_root.WritePPM(no_root_path) ||
          !denominator.WritePPM(denominator_path) || !residual.WritePPM(residual_path) ||
          !domain.WritePPM(domain_path) || !weight.WritePPM(weight_path) ||
          !t_range.WritePPM(t_range_path) ||
          (brute_force_grid != 0 && !brute_force_disagreements.WritePPM(brute_force_path)))))
    {
        std::cerr << "error: failed to write output images\n";
        return 1;
    }

    std::cout << "perspective rays hitting surface: " << hit_count << " / " << (size * size)
              << "\nwrote " << shaded_path << ", " << surface_path << ", " << uv_path << ", "
              << top_path << "\n";
    if (write_rt_diagnostics)
    {
        std::cout << "RT diagnostics: " << total_diagnostics.kernel_invocations << " callback invocations, "
                  << total_diagnostics.kernel_rejections << " solver rejections, " << zero_callback_rays
                  << " rays with no callback\n"
                  << "  rejection reasons (not mutually exclusive): no-root="
                  << total_diagnostics.reject_no_root << ", denominator="
                  << total_diagnostics.reject_denominator << ", residual="
                  << total_diagnostics.reject_residual << ", domain="
                  << total_diagnostics.reject_domain << ", weight="
                  << total_diagnostics.reject_weight << ", t-range="
                  << total_diagnostics.reject_t_range << "\n"
                  << "wrote role and per-reason RT diagnostic maps with prefix " << prefix << "_rt_*.ppm\n";
        if (brute_force_grid != 0)
        {
            std::cout << "brute-force comparison: " << brute_force_samples << " sampled rays, "
                      << brute_force_bvh_misses << " Embree misses, "
                      << brute_force_embree_only_hits << " Embree-only hits, "
                      << brute_force_t_disagreements << " t disagreements\n"
                      << "wrote " << brute_force_path
                      << " (red: Embree miss, yellow: direct miss, magenta: t mismatch)\n";
        }
    }
    if (multihit_grid != 0)
    {
        std::cout << "raw multi-hit diagnostic: " << multihit_samples << " sampled rays, "
                  << multihit_raw_hits << " raw leaf hits, max " << multihit_max_raw_hits
                  << " hits/ray, " << multihit_adjacent_pairs << " adjacent pairs\n"
                  << "  gaps <= 1e-5: " << multihit_pairs_le_1e5
                  << ", <= 1e-4: " << multihit_pairs_le_1e4
                  << ", <= 1e-3: " << multihit_pairs_le_1e3
                  << ", minimum: "
                  << (std::isfinite(multihit_min_gap) ? multihit_min_gap : 0.0) << "\n"
                  << "  adjacent pairs involving seam leaves: " << multihit_pairs_with_seam
                  << " (phantom: " << multihit_pairs_with_phantom << ")\n";
        std::cout << "  clustered direct hits: " << multihit_expected_clustered_hits
                  << ", IntersectAll hits: " << multihit_intersect_all_hits
                  << ", count mismatches: " << multihit_count_mismatches << "\n";
    }
    return 0;
}
