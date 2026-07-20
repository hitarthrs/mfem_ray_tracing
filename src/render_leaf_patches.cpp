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

#include "embree/leaf_patch_loader.hpp"
#include "embree/raytracer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
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

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: render_leaf_patches <leaf_bboxes.json> <output_prefix> [image_size]\n";
        return 1;
    }
    const std::string json_path = argv[1];
    const std::string prefix = argv[2];
    const int size = (argc > 3) ? std::max(16, std::atoi(argv[3])) : 800;

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
    long hit_count = 0;
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
            const RayHitRecord hit = tracer.Intersect(org_arr, dir_arr);

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
    if (!shaded.WritePPM(shaded_path) || !surface.WritePPM(surface_path) ||
        !uv_view.WritePPM(uv_path) || !top.WritePPM(top_path))
    {
        std::cerr << "error: failed to write output images\n";
        return 1;
    }

    std::cout << "perspective rays hitting surface: " << hit_count << " / " << (size * size)
              << "\nwrote " << shaded_path << ", " << surface_path << ", " << uv_path << ", "
              << top_path << "\n";
    return 0;
}
