// Interactive, fully Embree-rendered viewer for degree-reduced bilinear leaves.
//
// Unlike scene_viewer.cpp, this does not project and painter-sort patch quads.
// Every visible surface pixel is a first-hit query through Embree's bilinear
// user geometry, so visibility, selection, UVs, normals, and diagnostics all
// exercise the same backend used by the ray-tracing pipeline.
//
// Usage:
//   embree_surface_lab leaves.json [--allow-diagnostic-shell]
//   embree_surface_lab leaves.json --check [--allow-diagnostic-shell]
//   embree_surface_lab leaves.json --snapshot image.ppm [--size 900]
//
// Controls:
//   left drag              orbit
//   shift+left / middle    pan
//   wheel                  dolly
//   click                  inspect/select exact Embree hit
//   right click            fire and retain an exact world-space Embree ray
//                          (line continues through crossings so the exit side
//                          and leaving ray stay visible; markers on every hit)
//   space                  toggle left-click orbit / shooter mode
//   r / c                  fire a parallel ray grid / clear retained rays
//   b                      toggle dataset A/B without moving retained rays
//   1..8 / tab             shaded, error, source, leaf, normals, UV,
//                          callbacks, crossing-count views
//   w / s                  leaf wires / hard shadows (camera-linked key light)
//   i                      isolate selected source patch (or leaf)
//   a                      show all leaves
//   left/right             isolate previous/next source patch
//   f                      fit view
//   p                      save current full-resolution PPM
//   drag a JSON file       reload in place
//   q / escape             quit

#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "mfem_raytracing/embree/raytracer.hpp"

#include <SDL.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace mfem_raytracing;

namespace
{

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
Vec3 operator*(double s, Vec3 a) { return a * s; }
Vec3 operator/(Vec3 a, double s) { return {a.x / s, a.y / s, a.z / s}; }
double Dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 Cross(Vec3 a, Vec3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double Length(Vec3 a) { return std::sqrt(Dot(a, a)); }
Vec3 Normalize(Vec3 a)
{
    const double length = Length(a);
    return length > 0.0 ? a / length : Vec3{};
}
Vec3 FromArray(const double value[3]) { return {value[0], value[1], value[2]}; }

struct Color
{
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
};

Color operator+(Color a, Color b) { return {a.r + b.r, a.g + b.g, a.b + b.b}; }
Color operator*(Color a, double s) { return {a.r * s, a.g * s, a.b * s}; }
Color Mix(Color a, Color b, double t)
{
    t = std::clamp(t, 0.0, 1.0);
    return a * (1.0 - t) + b * t;
}

std::uint32_t Pack(Color c)
{
    auto byte = [](double x) {
        x = std::pow(std::clamp(x, 0.0, 1.0), 1.0 / 2.2);
        return static_cast<std::uint32_t>(std::lround(255.0 * x));
    };
    return 0xff000000u | (byte(c.r) << 16u) | (byte(c.g) << 8u) | byte(c.b);
}

Color HashColor(std::uint32_t value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    const double hue = (value & 0xffffu) / 65535.0;
    const double h = hue * 6.0;
    const double x = 1.0 - std::fabs(std::fmod(h, 2.0) - 1.0);
    Color rgb;
    if (h < 1.0) rgb = {1.0, x, 0.0};
    else if (h < 2.0) rgb = {x, 1.0, 0.0};
    else if (h < 3.0) rgb = {0.0, 1.0, x};
    else if (h < 4.0) rgb = {0.0, x, 1.0};
    else if (h < 5.0) rgb = {x, 0.0, 1.0};
    else rgb = {1.0, 0.0, x};
    return Mix({0.18, 0.23, 0.27}, rgb, 0.62);
}

Color ErrorColor(double value)
{
    value = std::clamp(value, 0.0, 1.0);
    if (value < 0.5)
    {
        return Mix({0.04, 0.22, 0.55}, {0.03, 0.72, 0.62}, value * 2.0);
    }
    return Mix({0.03, 0.72, 0.62}, {0.95, 0.20, 0.04}, (value - 0.5) * 2.0);
}

struct Camera
{
    enum class UpAxis { Y, Z };
    Vec3 center;
    double scene_radius = 1.0;
    double distance = 3.0;
    double yaw = -0.78;
    double pitch = 0.32;
    double fov_y = 42.0 * 3.14159265358979323846 / 180.0;
    UpAxis up_axis = UpAxis::Z;
};

struct CameraFrame
{
    Vec3 eye;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
};

CameraFrame Frame(const Camera &camera)
{
    const double cp = std::cos(camera.pitch);
    const Vec3 outward = camera.up_axis == Camera::UpAxis::Y
        ? Vec3{cp * std::cos(camera.yaw), std::sin(camera.pitch),
               cp * std::sin(camera.yaw)}
        : Vec3{cp * std::cos(camera.yaw), cp * std::sin(camera.yaw),
               std::sin(camera.pitch)};
    const Vec3 world_up = camera.up_axis == Camera::UpAxis::Y
        ? Vec3{0.0, 1.0, 0.0} : Vec3{0.0, 0.0, 1.0};
    const Vec3 eye = camera.center + outward * camera.distance;
    const Vec3 forward = Normalize(camera.center - eye);
    Vec3 right = Normalize(Cross(forward, world_up));
    if (Length(right) < 1e-12) right = {1.0, 0.0, 0.0};
    const Vec3 up = Normalize(Cross(right, forward));
    return {eye, forward, right, up};
}

void FitCamera(Camera &camera, const LeafPatchScene &scene)
{
    const Vec3 lo = FromArray(scene.scene_bbox.min);
    const Vec3 hi = FromArray(scene.scene_bbox.max);
    camera.center = (lo + hi) * 0.5;
    camera.scene_radius = std::max(1e-9, 0.5 * Length(hi - lo));
    camera.distance = camera.scene_radius * 3.0;
    camera.yaw = camera.up_axis == Camera::UpAxis::Y ? -1.45 : -0.78;
    camera.pitch = 0.32;
}

Camera::UpAxis InferUpAxis(const LeafPatchScene &scene)
{
    const double y_span = scene.scene_bbox.max[1] - scene.scene_bbox.min[1];
    const double z_span = scene.scene_bbox.max[2] - scene.scene_bbox.min[2];
    const bool y_starts_at_ground = std::fabs(scene.scene_bbox.min[1]) <
                                    1e-8 * std::max(1.0, y_span);
    const bool z_straddles_zero = scene.scene_bbox.min[2] < 0.0 && scene.scene_bbox.max[2] > 0.0;
    return y_starts_at_ground && z_straddles_zero ? Camera::UpAxis::Y : Camera::UpAxis::Z;
}

struct Ray
{
    Vec3 origin;
    Vec3 direction;
};

Ray CameraRay(const Camera &camera, double px, double py, int width, int height)
{
    const CameraFrame frame = Frame(camera);
    const double aspect = static_cast<double>(width) / std::max(1, height);
    const double tan_half = std::tan(0.5 * camera.fov_y);
    const double x = (2.0 * (px + 0.5) / width - 1.0) * aspect * tan_half;
    const double y = (1.0 - 2.0 * (py + 0.5) / height) * tan_half;
    return {frame.eye, Normalize(frame.forward + frame.right * x + frame.up * y)};
}

enum class ViewMode
{
    Shaded,
    Error,
    SourcePatch,
    Leaf,
    Normal,
    UV,
    Callbacks,
    Crossings,
};

const char *ModeName(ViewMode mode)
{
    switch (mode)
    {
    case ViewMode::Shaded: return "SHADED";
    case ViewMode::Error: return "ERROR";
    case ViewMode::SourcePatch: return "SOURCE PATCH";
    case ViewMode::Leaf: return "LEAF ID";
    case ViewMode::Normal: return "NORMAL";
    case ViewMode::UV: return "GLOBAL UV";
    case ViewMode::Callbacks: return "BVH CALLBACKS";
    case ViewMode::Crossings: return "CROSSINGS";
    }
    return "UNKNOWN";
}

struct ActiveScene
{
    std::unique_ptr<EmbreeRayTracer> tracer;
    std::vector<std::size_t> prim_to_leaf;
    int source_filter = -1;
};

ActiveScene BuildActiveScene(const LeafPatchScene &scene, int source_filter,
                             std::size_t leaf_filter, bool allow_diagnostic_shell)
{
    scene.RequireRayTracingCertified(allow_diagnostic_shell);
    std::vector<BilinearPatchPrimitive> patches;
    std::vector<std::size_t> mapping;
    for (std::size_t i = 0; i < scene.leaves.size(); ++i)
    {
        const LeafPatch &leaf = scene.leaves[i];
        const bool source_ok = source_filter < 0 || leaf.patch_id == source_filter;
        const bool leaf_ok = leaf_filter == std::numeric_limits<std::size_t>::max() || i == leaf_filter;
        if (source_ok && leaf_ok)
        {
            patches.push_back(leaf.patch);
            mapping.push_back(i);
        }
    }
    if (patches.empty()) throw std::runtime_error("active filter selected no leaves");
    auto tracer = std::make_unique<EmbreeRayTracer>();
    tracer->RegisterPatches(std::move(patches));
    tracer->CommitScene();
    return {std::move(tracer), std::move(mapping), source_filter};
}

struct Framebuffer
{
    int width = 0;
    int height = 0;
    std::vector<std::uint32_t> pixels;
    std::vector<int> leaf_ids;
    double milliseconds = 0.0;
    std::size_t hit_pixels = 0;
};

/// Persistent row scheduler shared by interactive redraws, full-resolution
/// snapshots, and comparison renders. The main thread submits one job at a
/// time and waits for all workers, so job captures remain valid until return.
class RenderWorkerPool
{
public:
    RenderWorkerPool()
    {
        const unsigned int count = std::max(1u, std::thread::hardware_concurrency());
        workers_.reserve(count);
        for (unsigned int id = 0; id < count; ++id)
        {
            workers_.emplace_back([this, id]() { WorkerLoop(id); });
        }
    }

    ~RenderWorkerPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        work_ready_.notify_all();
        for (auto &worker : workers_) worker.join();
    }

    RenderWorkerPool(const RenderWorkerPool &) = delete;
    RenderWorkerPool &operator=(const RenderWorkerPool &) = delete;

    std::size_t Size() const { return workers_.size(); }

    void ParallelRows(std::size_t rows,
                      std::function<void(std::size_t, unsigned int)> job)
    {
        if (rows == 0) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            rows_ = rows;
            next_row_.store(0, std::memory_order_relaxed);
            job_ = std::move(job);
            exception_ = nullptr;
            remaining_ = workers_.size();
            ++generation_;
        }
        work_ready_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, [&]() { return remaining_ == 0; });
        job_ = {};
        if (exception_) std::rethrow_exception(exception_);
    }

private:
    void WorkerLoop(unsigned int id)
    {
        std::size_t observed_generation = 0;
        while (true)
        {
            std::function<void(std::size_t, unsigned int)> job;
            std::size_t rows = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_ready_.wait(lock, [&]() {
                    return stopping_ || generation_ != observed_generation;
                });
                if (stopping_) return;
                observed_generation = generation_;
                job = job_;
                rows = rows_;
            }
            try
            {
                while (true)
                {
                    const std::size_t row = next_row_.fetch_add(1, std::memory_order_relaxed);
                    if (row >= rows) break;
                    job(row, id);
                }
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!exception_) exception_ = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--remaining_ == 0) finished_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable finished_;
    std::function<void(std::size_t, unsigned int)> job_;
    std::atomic<std::size_t> next_row_{0};
    std::size_t rows_ = 0;
    std::size_t remaining_ = 0;
    std::size_t generation_ = 0;
    std::exception_ptr exception_;
    bool stopping_ = false;
};

struct alignas(64) WorkerCounter
{
    std::size_t value = 0;
};

struct RenderOptions
{
    ViewMode mode = ViewMode::Shaded;
    bool wire = true;
    bool shadows = true;
    int selected_leaf = -1;
};

Color Background(int y, int height)
{
    (void)y;
    (void)height;
    return {1.0, 1.0, 1.0};
}

void Render(const LeafPatchScene &scene, const ActiveScene &active, const Camera &camera,
            int width, int height, const RenderOptions &options,
            RenderWorkerPool &worker_pool, Framebuffer &frame)
{
    frame.width = std::max(1, width);
    frame.height = std::max(1, height);
    const std::size_t count = static_cast<std::size_t>(frame.width) * frame.height;
    frame.pixels.resize(count);
    frame.leaf_ids.assign(count, -1);
    std::vector<WorkerCounter> worker_hits(worker_pool.Size());
    const auto started = std::chrono::steady_clock::now();
    // Key light rides with the camera so orbiting updates shading and shadows.
    const CameraFrame cam_frame = Frame(camera);
    const Vec3 key_light = Normalize(cam_frame.forward * 0.28 + cam_frame.up * 0.78 +
                                     cam_frame.right * 0.42);

    worker_pool.ParallelRows(static_cast<std::size_t>(frame.height),
        [&](std::size_t y, unsigned int worker) {
            std::size_t row_hits = 0;
            for (int x = 0; x < frame.width; ++x)
            {
                const std::size_t pixel = y * frame.width + x;
                const Ray ray = CameraRay(camera, x, static_cast<double>(y), frame.width, frame.height);
                const double origin[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
                const double direction[3] = {ray.direction.x, ray.direction.y, ray.direction.z};
                RayQueryDiagnostics diagnostics;
                RayHitRecord hit;
                std::size_t crossings = 0;
                if (options.mode == ViewMode::Crossings)
                {
                    const auto all = active.tracer->IntersectAll(origin, direction, 0.0,
                                                                std::numeric_limits<double>::infinity(), 8);
                    crossings = all.size();
                    if (!all.empty()) hit = all.front();
                }
                else
                {
                    hit = active.tracer->Intersect(
                        origin, direction, 0.0, std::numeric_limits<double>::infinity(),
                        options.mode == ViewMode::Callbacks ? &diagnostics : nullptr);
                }
                if (!hit.hit || hit.prim_id >= active.prim_to_leaf.size())
                {
                    frame.pixels[pixel] = Pack(Background(static_cast<int>(y), frame.height));
                    continue;
                }
                ++row_hits;
                const std::size_t leaf_index = active.prim_to_leaf[hit.prim_id];
                const LeafPatch &leaf = scene.leaves[leaf_index];
                frame.leaf_ids[pixel] = static_cast<int>(leaf_index);

                Color color;
                // Epsilon for the shadow ray is tied to the model scale and
                // supplied here instead of relying on pixel size.
                Vec3 normal = Normalize({hit.Ng[0], hit.Ng[1], hit.Ng[2]});
                if (Dot(normal, ray.direction) > 0.0) normal = -normal;
                if (options.mode == ViewMode::Normal)
                {
                    color = {0.5 * (normal.x + 1.0), 0.5 * (normal.y + 1.0),
                             0.5 * (normal.z + 1.0)};
                }
                else
                {
                    // Match the original scene_viewer palette: pale sage
                    // surfaces whose value tracks view-facing, while the
                    // visibility itself still comes from Embree.
                    const double view_facing = std::fabs(Dot(normal, ray.direction));
                    const double sage_g = (145.0 + 70.0 * view_facing) / 255.0;
                    Color base{std::max(0.0, sage_g - 12.0 / 255.0), sage_g,
                               std::min(1.0, sage_g + 6.0 / 255.0)};
                    if (options.mode == ViewMode::Error)
                    {
                        double denom = scene.max_error;
                        if (!(denom > 0.0))
                            for (const LeafPatch &item : scene.leaves)
                                denom = std::max(denom, item.total_error);
                        base = ErrorColor(denom > 0.0 ? leaf.total_error / denom : 0.0);
                    }
                    else if (options.mode == ViewMode::SourcePatch)
                        base = HashColor(static_cast<std::uint32_t>(std::max(0, leaf.patch_id) + 41));
                    else if (options.mode == ViewMode::Leaf)
                        base = HashColor(static_cast<std::uint32_t>(leaf.index + 173));
                    else if (options.mode == ViewMode::UV)
                    {
                        const double u = leaf.u_domain_global[0] + hit.u *
                            (leaf.u_domain_global[1] - leaf.u_domain_global[0]);
                        const double v = leaf.v_domain_global[0] + hit.v *
                            (leaf.v_domain_global[1] - leaf.v_domain_global[0]);
                        base = {std::clamp(u, 0.0, 1.0), std::clamp(v, 0.0, 1.0),
                                std::clamp(1.0 - 0.5 * (u + v), 0.0, 1.0)};
                    }
                    else if (options.mode == ViewMode::Callbacks)
                    {
                        const double calls = std::log2(1.0 + diagnostics.kernel_invocations) / 5.0;
                        const double rejects = std::log2(1.0 + diagnostics.kernel_rejections) / 5.0;
                        base = {std::clamp(rejects, 0.0, 1.0), std::clamp(calls, 0.0, 1.0), 0.06};
                    }
                    else if (options.mode == ViewMode::Crossings)
                    {
                        static const Color crossing_colors[] = {
                            {0.10, 0.10, 0.12}, {0.12, 0.58, 0.82}, {0.12, 0.76, 0.36},
                            {0.95, 0.68, 0.08}, {0.94, 0.20, 0.12}, {0.72, 0.18, 0.82},
                        };
                        constexpr std::size_t n_crossing_colors =
                            sizeof(crossing_colors) / sizeof(crossing_colors[0]);
                        base = crossing_colors[std::min<std::size_t>(
                            crossings, n_crossing_colors - 1)];
                    }

                    const double diffuse = std::max(0.0, Dot(normal, key_light));
                    const double facing = std::max(0.0, -Dot(normal, ray.direction));
                    double visibility = 1.0;
                    if (options.shadows &&
                        (options.mode == ViewMode::Shaded || options.mode == ViewMode::Error ||
                         options.mode == ViewMode::SourcePatch || options.mode == ViewMode::Leaf ||
                         options.mode == ViewMode::UV))
                    {
                        const double epsilon = std::max(1e-7, camera.scene_radius * 1e-6);
                        const Vec3 point = ray.origin + ray.direction * hit.t;
                        const Vec3 shadow_origin = point + normal * epsilon;
                        const double so[3] = {shadow_origin.x, shadow_origin.y, shadow_origin.z};
                        const double sd[3] = {key_light.x, key_light.y, key_light.z};
                        visibility = active.tracer->Occluded(so, sd, epsilon) ? 0.72 : 1.0;
                    }
                    const bool unlit = options.mode == ViewMode::Callbacks ||
                                       options.mode == ViewMode::Crossings;
                    // Ambient + camera-linked diffuse; soft shadow attenuation (not crushed blacks).
                    const double lighting = unlit ? 1.0 :
                        (0.34 + visibility * (0.52 * diffuse + 0.18 * facing));
                    color = base * lighting;
                }

                if (options.wire && std::min({hit.u, 1.0 - hit.u, hit.v, 1.0 - hit.v}) < 0.018)
                    color = Mix(color, {40.0 / 255.0, 62.0 / 255.0, 52.0 / 255.0}, 0.62);
                frame.pixels[pixel] = Pack(color);
            }
            worker_hits[worker].value += row_hits;
        });
    frame.hit_pixels = 0;
    for (const auto &counter : worker_hits) frame.hit_pixels += counter.value;

    if (options.selected_leaf >= 0)
    {
        const std::uint32_t highlight = Pack({1.0, 0.55, 0.04});
        const std::vector<int> ids = frame.leaf_ids;
        for (int y = 1; y + 1 < frame.height; ++y)
        {
            for (int x = 1; x + 1 < frame.width; ++x)
            {
                const std::size_t p = static_cast<std::size_t>(y) * frame.width + x;
                const bool selected = ids[p] == options.selected_leaf;
                if (!selected) continue;
                if (ids[p - 1] != options.selected_leaf || ids[p + 1] != options.selected_leaf ||
                    ids[p - frame.width] != options.selected_leaf ||
                    ids[p + frame.width] != options.selected_leaf)
                    frame.pixels[p] = highlight;
            }
        }
    }

    frame.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

bool WritePPM(const std::string &path, const Framebuffer &frame)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P6\n" << frame.width << " " << frame.height << "\n255\n";
    for (std::uint32_t pixel : frame.pixels)
    {
        const char rgb[3] = {static_cast<char>((pixel >> 16u) & 0xffu),
                             static_cast<char>((pixel >> 8u) & 0xffu),
                             static_cast<char>(pixel & 0xffu)};
        out.write(rgb, 3);
    }
    return static_cast<bool>(out);
}

struct ComparisonCounts
{
    std::size_t same_hit = 0;
    std::size_t same_miss = 0;
    std::size_t sealed = 0;
    std::size_t opened = 0;
};

std::pair<Framebuffer, ComparisonCounts> RenderComparison(
    const ActiveScene &a, const ActiveScene &b, const Camera &camera, int width, int height,
    RenderWorkerPool &worker_pool)
{
    Framebuffer frame;
    frame.width = std::max(1, width);
    frame.height = std::max(1, height);
    frame.pixels.resize(static_cast<std::size_t>(frame.width) * frame.height);
    struct alignas(64) WorkerComparison { ComparisonCounts counts; };
    std::vector<WorkerComparison> worker_counts(worker_pool.Size());
    const auto started = std::chrono::steady_clock::now();
    worker_pool.ParallelRows(static_cast<std::size_t>(frame.height),
        [&](std::size_t y, unsigned int worker) {
            ComparisonCounts row_counts;
            for (int x = 0; x < frame.width; ++x)
            {
                const Ray ray = CameraRay(camera, x, static_cast<double>(y),
                                          frame.width, frame.height);
                const double origin[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
                const double direction[3] = {
                    ray.direction.x, ray.direction.y, ray.direction.z};
                const RayHitRecord ah = a.tracer->Intersect(origin, direction);
                const RayHitRecord bh = b.tracer->Intersect(origin, direction);
                const double tolerance = std::max(1e-9, camera.scene_radius * 1e-6);
                Color color;
                if (bh.hit && (!ah.hit || bh.t + tolerance < ah.t))
                {
                    ++row_counts.sealed;
                    color = {0.0, 0.72, 0.28};
                }
                else if (ah.hit && (!bh.hit || ah.t + tolerance < bh.t))
                {
                    ++row_counts.opened;
                    color = {0.82, 0.04, 0.38};
                }
                else if (ah.hit && bh.hit)
                {
                    ++row_counts.same_hit;
                    color = {0.48, 0.55, 0.59};
                }
                else
                {
                    ++row_counts.same_miss;
                    color = Background(static_cast<int>(y), frame.height);
                }
                frame.pixels[y * frame.width + x] = Pack(color);
            }
            auto &total = worker_counts[worker].counts;
            total.same_hit += row_counts.same_hit;
            total.same_miss += row_counts.same_miss;
            total.sealed += row_counts.sealed;
            total.opened += row_counts.opened;
        });
    ComparisonCounts counts;
    for (const auto &worker : worker_counts)
    {
        counts.same_hit += worker.counts.same_hit;
        counts.same_miss += worker.counts.same_miss;
        counts.sealed += worker.counts.sealed;
        counts.opened += worker.counts.opened;
    }
    frame.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    frame.hit_pixels = counts.same_hit + counts.sealed + counts.opened;
    return {std::move(frame), counts};
}

std::string Fixed(double value, int precision = 4)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

struct Shot
{
    Ray ray;
    double extent = 0.0;
    std::array<std::vector<RayHitRecord>, 2> hits;
};

void TraceShot(Shot &shot, const ActiveScene &active, int dataset)
{
    const double origin[3] = {shot.ray.origin.x, shot.ray.origin.y, shot.ray.origin.z};
    const double direction[3] = {
        shot.ray.direction.x, shot.ray.direction.y, shot.ray.direction.z};
    shot.hits[static_cast<std::size_t>(dataset)] = active.tracer->IntersectAll(
        origin, direction, 0.0, shot.extent, 64);
}

void TraceShotPair(Shot &shot, const ActiveScene &a, const ActiveScene *b)
{
    TraceShot(shot, a, 0);
    shot.hits[1].clear();
    if (b) TraceShot(shot, *b, 1);
}

bool ProjectPoint(const Camera &camera, Vec3 point, int width, int height,
                  float &screen_x, float &screen_y)
{
    const CameraFrame frame = Frame(camera);
    const Vec3 relative = point - frame.eye;
    const double depth = Dot(relative, frame.forward);
    if (depth <= 1e-7) return false;
    const double tan_half = std::tan(0.5 * camera.fov_y);
    const double aspect = static_cast<double>(width) / std::max(1, height);
    const double nx = Dot(relative, frame.right) / (depth * tan_half * aspect);
    const double ny = Dot(relative, frame.up) / (depth * tan_half);
    screen_x = static_cast<float>((0.5 + 0.5 * nx) * width);
    screen_y = static_cast<float>((0.5 - 0.5 * ny) * height);
    return std::isfinite(screen_x) && std::isfinite(screen_y);
}

struct ShotStats
{
    std::size_t hit = 0;
    std::size_t miss = 0;
    std::size_t sealed = 0;
    std::size_t opened = 0;
};

ShotStats GetShotStats(const std::vector<Shot> &shots, int dataset, bool has_compare)
{
    ShotStats stats;
    for (const Shot &shot : shots)
    {
        const bool a_hit = !shot.hits[0].empty();
        const bool b_hit = !shot.hits[1].empty();
        const bool current_hit = !shot.hits[static_cast<std::size_t>(dataset)].empty();
        current_hit ? ++stats.hit : ++stats.miss;
        const double tolerance = std::max(1e-9, shot.extent * 1e-6);
        if (has_compare && b_hit &&
            (!a_hit || shot.hits[1].front().t + tolerance < shot.hits[0].front().t))
            ++stats.sealed;
        if (has_compare && a_hit &&
            (!b_hit || shot.hits[0].front().t + tolerance < shot.hits[1].front().t))
            ++stats.opened;
    }
    return stats;
}

void DrawThickLine(SDL_Renderer *renderer, float x0, float y0, float x1, float y1)
{
    SDL_RenderDrawLineF(renderer, x0, y0, x1, y1);
    SDL_RenderDrawLineF(renderer, x0 + 1.0f, y0, x1 + 1.0f, y1);
}

void DrawProjectedSegment(SDL_Renderer *renderer, const Camera &camera, int width, int height,
                          Vec3 a, Vec3 b)
{
    float x0 = 0.0f, y0 = 0.0f, x1 = 0.0f, y1 = 0.0f;
    if (ProjectPoint(camera, a, width, height, x0, y0) &&
        ProjectPoint(camera, b, width, height, x1, y1))
        DrawThickLine(renderer, x0, y0, x1, y1);
}

void DrawShotOverlay(SDL_Renderer *renderer, const std::vector<Shot> &shots,
                     const Camera &camera, int width, int height, int dataset,
                     bool has_compare)
{
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    for (const Shot &shot : shots)
    {
        const bool a_hit = !shot.hits[0].empty();
        const bool b_hit = !shot.hits[1].empty();
        const auto &hits = shot.hits[static_cast<std::size_t>(dataset)];
        const bool hit = !hits.empty();
        const double first_t = hit ? hits.front().t : shot.extent;
        const Vec3 start = shot.ray.origin + shot.ray.direction * (shot.extent * 0.002);
        const Vec3 entry = shot.ray.origin + shot.ray.direction * first_t;
        const Vec3 far_end = shot.ray.origin + shot.ray.direction * shot.extent;
        const double tolerance = std::max(1e-9, shot.extent * 1e-6);
        const bool sealed = has_compare && b_hit &&
            (!a_hit || shot.hits[1].front().t + tolerance < shot.hits[0].front().t);
        const bool opened = has_compare && a_hit &&
            (!b_hit || shot.hits[0].front().t + tolerance < shot.hits[1].front().t);

        // Incoming segment (origin → first hit / miss extent).
        if (sealed) SDL_SetRenderDrawColor(renderer, 0, 166, 81, 235);
        else if (opened) SDL_SetRenderDrawColor(renderer, 205, 35, 125, 235);
        else if (hit) SDL_SetRenderDrawColor(renderer, 0, 112, 190, 210);
        else SDL_SetRenderDrawColor(renderer, 226, 74, 51, 225);
        DrawProjectedSegment(renderer, camera, width, height, start, entry);

        // Continue past the entry so the exit side and leaving ray stay visible.
        // Dimmer so entry vs through/exit read clearly; markers still mark every crossing.
        if (hit)
        {
            if (sealed) SDL_SetRenderDrawColor(renderer, 0, 166, 81, 120);
            else if (opened) SDL_SetRenderDrawColor(renderer, 205, 35, 125, 120);
            else SDL_SetRenderDrawColor(renderer, 0, 112, 190, 110);
            DrawProjectedSegment(renderer, camera, width, height, entry, far_end);
        }

        for (std::size_t i = 0; i < hits.size(); ++i)
        {
            const Vec3 point = shot.ray.origin + shot.ray.direction * hits[i].t;
            float x = 0.0f, y = 0.0f;
            if (ProjectPoint(camera, point, width, height, x, y))
            {
                // Slightly larger markers on first (entry) and last (exit) crossings.
                const float half = (i == 0 || i + 1 == hits.size()) ? 4.5f : 3.0f;
                SDL_FRect marker{x - half, y - half, 2.0f * half + 1.0f, 2.0f * half + 1.0f};
                SDL_RenderFillRectF(renderer, &marker);
            }
        }
    }
}

void SetWindowCaption(SDL_Window *window, const LeafPatchScene &scene, const ActiveScene &active,
                      const RenderOptions &options, const Framebuffer &frame,
                      const std::string &dataset_label, bool shoot_mode)
{
    std::ostringstream title;
    title << "Embree Surface Lab · " << dataset_label << " · " << scene.surface_name
          << " · " << (shoot_mode ? "SHOOT" : "ORBIT") << " · " << ModeName(options.mode)
          << " · " << active.prim_to_leaf.size() << " leaves · "
          << std::fixed << std::setprecision(1) << frame.milliseconds << " ms";
    if (active.source_filter >= 0) title << " · source " << active.source_filter;
    SDL_SetWindowTitle(window, title.str().c_str());
}

void DrawText(SDL_Renderer *renderer, int x, int y, const std::string &text, int scale = 2)
{
    static const std::map<char, std::array<unsigned char, 7>> font = {
        {' ', {0, 0, 0, 0, 0, 0, 0}}, {'-', {0, 0, 0, 31, 0, 0, 0}},
        {'.', {0, 0, 0, 0, 0, 12, 12}}, {'/', {1, 2, 4, 8, 16, 0, 0}},
        {':', {0, 12, 12, 0, 12, 12, 0}}, {'=', {0, 31, 0, 31, 0, 0, 0}},
        {'(', {2, 4, 8, 8, 8, 4, 2}}, {')', {8, 4, 2, 2, 2, 4, 8}},
        {'0', {14, 17, 19, 21, 25, 17, 14}}, {'1', {4, 12, 4, 4, 4, 4, 14}},
        {'2', {14, 17, 1, 2, 4, 8, 31}}, {'3', {30, 1, 1, 14, 1, 1, 30}},
        {'4', {2, 6, 10, 18, 31, 2, 2}}, {'5', {31, 16, 30, 1, 1, 17, 14}},
        {'6', {6, 8, 16, 30, 17, 17, 14}}, {'7', {31, 1, 2, 4, 8, 8, 8}},
        {'8', {14, 17, 17, 14, 17, 17, 14}}, {'9', {14, 17, 17, 15, 1, 2, 12}},
        {'A', {14, 17, 17, 31, 17, 17, 17}}, {'B', {30, 17, 17, 30, 17, 17, 30}},
        {'C', {14, 17, 16, 16, 16, 17, 14}}, {'D', {30, 17, 17, 17, 17, 17, 30}},
        {'E', {31, 16, 16, 30, 16, 16, 31}}, {'F', {31, 16, 16, 30, 16, 16, 16}},
        {'G', {14, 17, 16, 23, 17, 17, 15}}, {'H', {17, 17, 17, 31, 17, 17, 17}},
        {'I', {14, 4, 4, 4, 4, 4, 14}}, {'J', {7, 2, 2, 2, 2, 18, 12}},
        {'K', {17, 18, 20, 24, 20, 18, 17}}, {'L', {16, 16, 16, 16, 16, 16, 31}},
        {'M', {17, 27, 21, 21, 17, 17, 17}}, {'N', {17, 25, 21, 19, 17, 17, 17}},
        {'O', {14, 17, 17, 17, 17, 17, 14}}, {'P', {30, 17, 17, 30, 16, 16, 16}},
        {'Q', {14, 17, 17, 17, 21, 18, 13}}, {'R', {30, 17, 17, 30, 20, 18, 17}},
        {'S', {15, 16, 16, 14, 1, 1, 30}}, {'T', {31, 4, 4, 4, 4, 4, 4}},
        {'U', {17, 17, 17, 17, 17, 17, 14}}, {'V', {17, 17, 17, 17, 10, 10, 4}},
        {'W', {17, 17, 17, 21, 21, 27, 17}}, {'X', {17, 17, 10, 4, 10, 17, 17}},
        {'Y', {17, 17, 10, 4, 4, 4, 4}}, {'Z', {31, 1, 2, 4, 8, 16, 31}},
    };
    int cursor = x;
    for (char raw : text)
    {
        const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(raw)));
        const auto found = font.find(c);
        const auto &glyph = found == font.end() ? font.at(' ') : found->second;
        for (int row = 0; row < 7; ++row)
            for (int column = 0; column < 5; ++column)
                if (glyph[row] & (1u << (4 - column)))
                {
                    SDL_Rect pixel{cursor + column * scale, y + row * scale, scale, scale};
                    SDL_RenderFillRect(renderer, &pixel);
                }
        cursor += 6 * scale;
    }
}

void DrawHud(SDL_Renderer *renderer, const LeafPatchScene &scene, const ActiveScene &active,
             const RenderOptions &options, const Framebuffer &framebuffer, bool draft,
             const std::string &dataset_label, bool shoot_mode,
             const std::vector<Shot> &shots, int dataset, bool has_compare)
{
    const ShotStats stats = GetShotStats(shots, dataset, has_compare);
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_Rect panel{14, 14, 390, options.selected_leaf >= 0 ? 126 : 108};
    SDL_SetRenderDrawColor(renderer, 243, 248, 245, 235);
    SDL_RenderFillRect(renderer, &panel);
    SDL_SetRenderDrawColor(renderer, 30, 54, 42, 90);
    SDL_RenderDrawRect(renderer, &panel);
    SDL_SetRenderDrawColor(renderer, 11, 122, 110, 235);
    SDL_Rect accent{panel.x, panel.y, 4, panel.h};
    SDL_RenderFillRect(renderer, &accent);
    SDL_SetRenderDrawColor(renderer, 11, 122, 110, 255);
    DrawText(renderer, 28, 27, "EMBREE SURFACE LAB", 2);
    SDL_SetRenderDrawColor(renderer, 29, 42, 50, 255);
    DrawText(renderer, 28, 51,
             dataset_label + "  " + (shoot_mode ? "SHOOT" : "ORBIT") + "  " +
                 ModeName(options.mode), 1);
    DrawText(renderer, 28, 65,
             std::string("LEAVES ") +
                 std::to_string(active.prim_to_leaf.size()), 1);
    DrawText(renderer, 132, 65,
             std::string(draft ? "DRAFT " : "FULL  ") + Fixed(framebuffer.milliseconds, 1) +
                 " MS  HITS " + std::to_string(framebuffer.hit_pixels), 1);
    DrawText(renderer, 28, 79,
             "RAYS " + std::to_string(shots.size()) + "  HIT " + std::to_string(stats.hit) +
                 "  MISS " + std::to_string(stats.miss), 1);
    if (has_compare)
    {
        SDL_SetRenderDrawColor(renderer, 0, 140, 72, 255);
        DrawText(renderer, 216, 79, "SEALED " + std::to_string(stats.sealed), 1);
    }
    if (options.selected_leaf >= 0 &&
        static_cast<std::size_t>(options.selected_leaf) < scene.leaves.size())
    {
        const LeafPatch &leaf = scene.leaves[static_cast<std::size_t>(options.selected_leaf)];
        SDL_SetRenderDrawColor(renderer, 210, 91, 0, 255);
        DrawText(renderer, 28, 99,
                 "SELECT LEAF " + std::to_string(leaf.index) + "  SOURCE " +
                     std::to_string(leaf.patch_id) + "  ERR " + Fixed(leaf.total_error, 5), 1);
    }
}

void PrintProbe(const LeafPatchScene &scene, const ActiveScene &active, const Camera &camera,
                int mouse_x, int mouse_y, int width, int height, int &selected_leaf)
{
    const Ray ray = CameraRay(camera, mouse_x, mouse_y, width, height);
    const double origin[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    const double direction[3] = {ray.direction.x, ray.direction.y, ray.direction.z};
    RayQueryDiagnostics diagnostics;
    const RayHitRecord hit = active.tracer->Intersect(origin, direction, 0.0,
        std::numeric_limits<double>::infinity(), &diagnostics);
    if (!hit.hit || hit.prim_id >= active.prim_to_leaf.size())
    {
        selected_leaf = -1;
        std::cout << "probe: miss; callbacks=" << diagnostics.kernel_invocations
                  << ", rejections=" << diagnostics.kernel_rejections << "\n";
        return;
    }
    const std::size_t leaf_index = active.prim_to_leaf[hit.prim_id];
    const LeafPatch &leaf = scene.leaves[leaf_index];
    selected_leaf = static_cast<int>(leaf_index);
    const double gu = leaf.u_domain_global[0] +
                      hit.u * (leaf.u_domain_global[1] - leaf.u_domain_global[0]);
    const double gv = leaf.v_domain_global[0] +
                      hit.v * (leaf.v_domain_global[1] - leaf.v_domain_global[0]);
    const auto all_hits = active.tracer->IntersectAll(origin, direction, 0.0,
        std::numeric_limits<double>::infinity(), 64);
    std::cout << "probe: leaf=" << leaf.index << ", source_patch=" << leaf.patch_id
              << ", local_uv=(" << Fixed(hit.u) << ", " << Fixed(hit.v) << ")"
              << ", global_uv=(" << Fixed(gu) << ", " << Fixed(gv) << ")"
              << ", t=" << Fixed(hit.t, 6) << ", error=" << Fixed(leaf.total_error, 8)
              << ", crossings=" << all_hits.size() << ", callbacks="
              << diagnostics.kernel_invocations << ", rejections="
              << diagnostics.kernel_rejections << ", role=" << leaf.role << "\n";
}

std::vector<int> SourcePatchIds(const LeafPatchScene &scene)
{
    std::set<int> ids;
    for (const LeafPatch &leaf : scene.leaves)
        if (leaf.patch_id >= 0) ids.insert(leaf.patch_id);
    return {ids.begin(), ids.end()};
}

void PrintControls()
{
    std::cout
        << "controls: space toggles ORBIT/SHOOT; right click always fires; wheel dolly\n"
        << "          left drag orbit; shift+left/middle pan; left click inspect/shoot\n"
        << "          R parallel ray grid; C clear rays; B toggle exact A/B geometry\n"
        << "          1-8/tab views; W wires; S shadows; I isolate; A all; F fit\n"
        << "          left/right source patch; P clean PPM; drag/drop JSON; Q quit\n";
}

struct Cli
{
    std::string input;
    std::string compare;
    std::string compare_report;
    std::string snapshot;
    bool check = false;
    bool allow_diagnostic_shell = false;
    int size = 900;
    std::string up_axis = "auto";
};

Cli ParseCli(int argc, char **argv)
{
    if (argc < 2)
        throw std::runtime_error(
            "usage: embree_surface_lab leaves.json [--check] [--snapshot image.ppm] "
            "[--compare other.json] [--compare-report mask.ppm] "
            "[--size N] [--allow-diagnostic-shell]");
    Cli cli;
    cli.input = argv[1];
    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--check") cli.check = true;
        else if (arg == "--allow-diagnostic-shell") cli.allow_diagnostic_shell = true;
        else if (arg == "--compare" && i + 1 < argc) cli.compare = argv[++i];
        else if (arg == "--compare-report" && i + 1 < argc) cli.compare_report = argv[++i];
        else if (arg == "--snapshot" && i + 1 < argc) cli.snapshot = argv[++i];
        else if (arg == "--size" && i + 1 < argc) cli.size = std::max(64, std::stoi(argv[++i]));
        else if (arg == "--up" && i + 1 < argc)
        {
            cli.up_axis = argv[++i];
            if (cli.up_axis != "auto" && cli.up_axis != "y" && cli.up_axis != "z")
                throw std::runtime_error("--up must be auto, y, or z");
        }
        else throw std::runtime_error("unknown/incomplete argument '" + arg + "'");
    }
    return cli;
}

} // namespace

int main(int argc, char **argv)
{
    Cli cli;
    LeafPatchScene scene;
    ActiveScene active;
    LeafPatchScene compare_scene;
    ActiveScene compare_active;
    bool has_compare = false;
    Camera camera;
    try
    {
        cli = ParseCli(argc, argv);
        scene = LoadLeafPatchScene(cli.input);
        active = BuildActiveScene(scene, -1, std::numeric_limits<std::size_t>::max(),
                                  cli.allow_diagnostic_shell);
        if (!cli.compare.empty())
        {
            compare_scene = LoadLeafPatchScene(cli.compare);
            compare_active = BuildActiveScene(compare_scene, -1,
                std::numeric_limits<std::size_t>::max(), cli.allow_diagnostic_shell);
            has_compare = true;
        }
        camera.up_axis = cli.up_axis == "y" ? Camera::UpAxis::Y :
                         cli.up_axis == "z" ? Camera::UpAxis::Z : InferUpAxis(scene);
        FitCamera(camera, scene);
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    if (cli.check)
    {
        std::cout << "embree_surface_lab check ok\n"
                  << "  surface: " << scene.surface_name << "\n"
                  << "  leaves:  " << scene.leaves.size() << "\n"
                  << "  error:   " << scene.max_error << "\n";
        if (has_compare)
            std::cout << "  compare: " << compare_scene.surface_name << " ("
                      << compare_scene.leaves.size() << " leaves)\n";
        return 0;
    }

    RenderWorkerPool render_workers;

    if (!cli.compare_report.empty())
    {
        if (!has_compare)
        {
            std::cerr << "error: --compare-report requires --compare other.json\n";
            return 1;
        }
        auto report = RenderComparison(active, compare_active, camera, cli.size, cli.size,
                                       render_workers);
        if (!WritePPM(cli.compare_report, report.first))
        {
            std::cerr << "error: cannot write '" << cli.compare_report << "'\n";
            return 1;
        }
        const std::size_t total = static_cast<std::size_t>(cli.size) * cli.size;
        std::cout << "comparison rays: " << total << "\n"
                  << "  unchanged hit: " << report.second.same_hit << "\n"
                  << "  unchanged miss: " << report.second.same_miss << "\n"
                  << "  stopped by B:   " << report.second.sealed << "\n"
                  << "  earlier in A:   " << report.second.opened << "\n"
                  << "  mask: " << cli.compare_report << "\n";
        return 0;
    }

    RenderOptions options;
    if (!cli.snapshot.empty())
    {
        Framebuffer frame;
        Render(scene, active, camera, cli.size, cli.size, options, render_workers, frame);
        if (!WritePPM(cli.snapshot, frame))
        {
            std::cerr << "error: cannot write '" << cli.snapshot << "'\n";
            return 1;
        }
        std::cout << "wrote " << cli.snapshot << " using " << frame.hit_pixels << " Embree-hit pixels in "
                  << Fixed(frame.milliseconds, 1) << " ms\n";
        return 0;
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
        return 1;
    }
    SDL_Window *window = SDL_CreateWindow("Embree Surface Lab", SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, 1280, 860,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window)
    {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }
    SDL_Renderer *renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    if (!renderer)
    {
        std::cerr << "SDL_CreateRenderer failed: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_EventState(SDL_DROPFILE, SDL_ENABLE);
    SDL_Texture *texture = nullptr;
    int texture_width = 0;
    int texture_height = 0;
    bool running = true;
    bool dirty = true;
    bool draft = false;
    bool dragging = false;
    bool panning = false;
    bool moved = false;
    bool shoot_mode = false;
    bool showing_b = false;
    int mouse_down_x = 0;
    int mouse_down_y = 0;
    int last_x = 0;
    int last_y = 0;
    int selected_leaf = -1;
    std::size_t isolated_leaf = std::numeric_limits<std::size_t>::max();
    std::string current_path = cli.input;
    std::string compare_path = cli.compare;
    Framebuffer framebuffer;
    std::vector<Shot> shots;
    std::vector<int> source_ids = SourcePatchIds(scene);
    int snapshot_index = 0;
    PrintControls();

    auto trace_pair = [&](Shot &shot) {
        if (!showing_b)
            TraceShotPair(shot, active, has_compare ? &compare_active : nullptr);
        else
        {
            TraceShot(shot, compare_active, 0);
            TraceShot(shot, active, 1);
        }
    };
    auto fire_screen_ray = [&](int window_x, int window_y) {
        int window_width = 1, window_height = 1, output_width = 1, output_height = 1;
        SDL_GetWindowSize(window, &window_width, &window_height);
        SDL_GetRendererOutputSize(renderer, &output_width, &output_height);
        const double x = window_x * (static_cast<double>(output_width) / window_width);
        const double y = window_y * (static_cast<double>(output_height) / window_height);
        Shot shot;
        shot.ray = CameraRay(camera, x, y, output_width, output_height);
        shot.extent = std::max(camera.distance + 4.0 * camera.scene_radius,
                               8.0 * camera.scene_radius);
        trace_pair(shot);
        const bool a_hit = !shot.hits[0].empty();
        const bool b_hit = !shot.hits[1].empty();
        shots.push_back(std::move(shot));
        std::cout << "shot " << shots.size() << ": A=" << (a_hit ? "hit" : "miss");
        if (has_compare)
            std::cout << ", B=" << (b_hit ? "hit" : "miss")
                      << (!a_hit && b_hit ? " [SEALED BY B]" :
                          (a_hit && !b_hit ? " [OPENED IN B]" : ""));
        std::cout << "\n";
        dirty = true;
    };
    auto fire_grid = [&]() {
        constexpr int grid_size = 19;
        const CameraFrame cf = Frame(camera);
        const double span = camera.scene_radius * 1.32;
        const double aspect = [&]() {
            int width = 1, height = 1;
            SDL_GetRendererOutputSize(renderer, &width, &height);
            return static_cast<double>(width) / std::max(1, height);
        }();
        shots.clear();
        shots.reserve(grid_size * grid_size);
        for (int row = 0; row < grid_size; ++row)
            for (int column = 0; column < grid_size; ++column)
            {
                const double u = 2.0 * column / (grid_size - 1) - 1.0;
                const double v = 1.0 - 2.0 * row / (grid_size - 1);
                Shot shot;
                shot.ray.origin = camera.center - cf.forward * (3.0 * camera.scene_radius) +
                                  cf.right * (u * span * aspect) + cf.up * (v * span);
                shot.ray.direction = cf.forward;
                shot.extent = 6.0 * camera.scene_radius;
                trace_pair(shot);
                shots.push_back(std::move(shot));
            }
        const ShotStats stats = GetShotStats(shots, showing_b ? 1 : 0, has_compare);
        std::cout << "parallel grid: " << shots.size() << " rays, A->B sealed="
                  << stats.sealed << ", A->B opened=" << stats.opened << "\n";
        dirty = true;
    };

    auto rebuild = [&](int source_filter, std::size_t leaf_filter) {
        active = BuildActiveScene(scene, source_filter, leaf_filter, cli.allow_diagnostic_shell);
        isolated_leaf = leaf_filter;
        dirty = true;
        draft = false;
    };
    auto show_all = [&]() {
        selected_leaf = -1;
        rebuild(-1, std::numeric_limits<std::size_t>::max());
    };
    auto load = [&](const std::string &path) {
        try
        {
            LeafPatchScene next = LoadLeafPatchScene(path);
            ActiveScene next_active = BuildActiveScene(
                next, -1, std::numeric_limits<std::size_t>::max(), cli.allow_diagnostic_shell);
            scene = std::move(next);
            active = std::move(next_active);
            source_ids = SourcePatchIds(scene);
            if (cli.up_axis == "auto") camera.up_axis = InferUpAxis(scene);
            FitCamera(camera, scene);
            current_path = path;
            selected_leaf = -1;
            isolated_leaf = std::numeric_limits<std::size_t>::max();
            dirty = true;
            draft = false;
            std::cout << "loaded " << path << ": " << scene.leaves.size() << " leaves\n";
        }
        catch (const std::exception &error)
        {
            std::cerr << "reload failed: " << error.what() << "\n";
        }
    };

    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT) running = false;
            else if (event.type == SDL_DROPFILE)
            {
                if (event.drop.file)
                {
                    load(event.drop.file);
                    SDL_free(event.drop.file);
                }
            }
            else if (event.type == SDL_WINDOWEVENT &&
                     (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED ||
                      event.window.event == SDL_WINDOWEVENT_EXPOSED))
            {
                dirty = true;
                draft = false;
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN &&
                     (event.button.button == SDL_BUTTON_RIGHT ||
                      (event.button.button == SDL_BUTTON_LEFT && shoot_mode)))
            {
                fire_screen_ray(event.button.x, event.button.y);
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN &&
                     (event.button.button == SDL_BUTTON_LEFT ||
                      event.button.button == SDL_BUTTON_MIDDLE))
            {
                dragging = true;
                panning = event.button.button == SDL_BUTTON_MIDDLE ||
                    (SDL_GetModState() & (KMOD_LSHIFT | KMOD_RSHIFT));
                moved = false;
                mouse_down_x = last_x = event.button.x;
                mouse_down_y = last_y = event.button.y;
            }
            else if (event.type == SDL_MOUSEMOTION && dragging)
            {
                const int dx = event.motion.x - last_x;
                const int dy = event.motion.y - last_y;
                moved = moved || std::abs(event.motion.x - mouse_down_x) > 3 ||
                                  std::abs(event.motion.y - mouse_down_y) > 3;
                if (panning)
                {
                    const CameraFrame cf = Frame(camera);
                    const double units = 2.0 * camera.distance * std::tan(0.5 * camera.fov_y) /
                                         std::max(1, framebuffer.height);
                    camera.center = camera.center - cf.right * (dx * units) + cf.up * (dy * units);
                }
                else
                {
                    camera.yaw -= dx * 0.008;
                    camera.pitch = std::clamp(camera.pitch + dy * 0.008, -1.52, 1.52);
                }
                last_x = event.motion.x;
                last_y = event.motion.y;
                dirty = true;
                draft = true;
            }
            else if (event.type == SDL_MOUSEBUTTONUP &&
                     (event.button.button == SDL_BUTTON_LEFT ||
                      event.button.button == SDL_BUTTON_MIDDLE))
            {
                if (event.button.button == SDL_BUTTON_LEFT && !moved && !shoot_mode)
                {
                    int window_width = 1, window_height = 1, output_width = 1, output_height = 1;
                    SDL_GetWindowSize(window, &window_width, &window_height);
                    SDL_GetRendererOutputSize(renderer, &output_width, &output_height);
                    const int x = static_cast<int>(event.button.x *
                        (static_cast<double>(output_width) / window_width));
                    const int y = static_cast<int>(event.button.y *
                        (static_cast<double>(output_height) / window_height));
                    PrintProbe(scene, active, camera, x, y, output_width, output_height,
                               selected_leaf);
                }
                dragging = false;
                dirty = true;
                draft = false;
            }
            else if (event.type == SDL_MOUSEWHEEL)
            {
                camera.distance *= std::pow(0.88, event.wheel.y);
                camera.distance = std::clamp(camera.distance,
                    camera.scene_radius * 0.03, camera.scene_radius * 100.0);
                dirty = true;
                draft = false;
            }
            else if (event.type == SDL_KEYDOWN)
            {
                const SDL_Keycode key = event.key.keysym.sym;
                if (key == SDLK_ESCAPE || key == SDLK_q) running = false;
                else if (key >= SDLK_1 && key <= SDLK_8)
                {
                    options.mode = static_cast<ViewMode>(key - SDLK_1);
                    dirty = true;
                }
                else if (key == SDLK_TAB)
                {
                    options.mode = static_cast<ViewMode>((static_cast<int>(options.mode) + 1) % 8);
                    dirty = true;
                }
                else if (key == SDLK_w) { options.wire = !options.wire; dirty = true; }
                else if (key == SDLK_s) { options.shadows = !options.shadows; dirty = true; }
                else if (key == SDLK_f) { FitCamera(camera, scene); dirty = true; }
                else if (key == SDLK_SPACE)
                {
                    shoot_mode = !shoot_mode;
                    std::cout << (shoot_mode ? "SHOOT mode\n" : "ORBIT mode\n");
                    dirty = true;
                }
                else if (key == SDLK_r) fire_grid();
                else if (key == SDLK_c) { shots.clear(); dirty = true; }
                else if (key == SDLK_b && has_compare)
                {
                    std::swap(scene, compare_scene);
                    std::swap(active, compare_active);
                    std::swap(current_path, compare_path);
                    showing_b = !showing_b;
                    source_ids = SourcePatchIds(scene);
                    selected_leaf = -1;
                    isolated_leaf = std::numeric_limits<std::size_t>::max();
                    std::cout << "showing dataset " << (showing_b ? "B" : "A") << ": "
                              << current_path << " (same retained world-space rays)\n";
                    dirty = true;
                    draft = false;
                }
                else if (key == SDLK_a) show_all();
                else if (key == SDLK_i && selected_leaf >= 0)
                {
                    const LeafPatch &leaf = scene.leaves[static_cast<std::size_t>(selected_leaf)];
                    if (leaf.patch_id >= 0)
                        rebuild(leaf.patch_id, std::numeric_limits<std::size_t>::max());
                    else
                        rebuild(-1, static_cast<std::size_t>(selected_leaf));
                }
                else if ((key == SDLK_LEFT || key == SDLK_RIGHT) && !source_ids.empty())
                {
                    auto it = std::find(source_ids.begin(), source_ids.end(), active.source_filter);
                    int position = it == source_ids.end() ? 0 : static_cast<int>(it - source_ids.begin());
                    position += key == SDLK_RIGHT ? 1 : -1;
                    position = (position + static_cast<int>(source_ids.size())) %
                               static_cast<int>(source_ids.size());
                    rebuild(source_ids[static_cast<std::size_t>(position)],
                            std::numeric_limits<std::size_t>::max());
                }
                else if (key == SDLK_p)
                {
                    int width = 1, height = 1;
                    SDL_GetRendererOutputSize(renderer, &width, &height);
                    Framebuffer full;
                    Render(scene, active, camera, width, height, options, render_workers, full);
                    const std::string path = "embree_surface_lab_" +
                        std::to_string(++snapshot_index) + ".ppm";
                    if (WritePPM(path, full)) std::cout << "wrote " << path << "\n";
                    else std::cerr << "failed to write " << path << "\n";
                }
            }
        }

        if (dirty)
        {
            int output_width = 1, output_height = 1;
            SDL_GetRendererOutputSize(renderer, &output_width, &output_height);
            // Orbit/pan uses a lower draft res for snappy interaction; settled
            // frames render at the native drawable size (HiDPI-aware).
            constexpr int kDraftLongEdge = 720;
            const int long_edge = std::max(output_width, output_height);
            const double scale = draft
                ? std::min(1.0, static_cast<double>(kDraftLongEdge) / long_edge)
                : 1.0;
            const int render_width = std::max(1, static_cast<int>(std::lround(output_width * scale)));
            const int render_height = std::max(1, static_cast<int>(std::lround(output_height * scale)));
            options.selected_leaf = selected_leaf;
            Render(scene, active, camera, render_width, render_height, options,
                   render_workers, framebuffer);
            if (!texture || texture_width != render_width || texture_height != render_height)
            {
                if (texture) SDL_DestroyTexture(texture);
                texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                    SDL_TEXTUREACCESS_STREAMING, render_width, render_height);
                texture_width = render_width;
                texture_height = render_height;
            }
            // Nearest keeps settled native frames crisp; linear softens draft upscales.
            SDL_SetTextureScaleMode(texture,
                draft ? SDL_ScaleModeLinear : SDL_ScaleModeNearest);
            SDL_UpdateTexture(texture, nullptr, framebuffer.pixels.data(),
                              render_width * static_cast<int>(sizeof(std::uint32_t)));
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);
            DrawShotOverlay(renderer, shots, camera, output_width, output_height,
                            showing_b ? 1 : 0, has_compare);
            DrawHud(renderer, scene, active, options, framebuffer, draft,
                    showing_b ? "B CONNECTED" : "A INDEPENDENT", shoot_mode,
                    shots, showing_b ? 1 : 0, has_compare);
            SDL_RenderPresent(renderer);
            SetWindowCaption(window, scene, active, options, framebuffer,
                             showing_b ? "B CONNECTED" : "A INDEPENDENT", shoot_mode);
            dirty = false;
        }
        else
        {
            SDL_Delay(8);
        }
    }

    if (texture) SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
