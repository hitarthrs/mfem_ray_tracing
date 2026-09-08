// Reproducible single-thread query/BVH comparison; timings exclude validation
// and diagnostics. Run a Release build, without other benchmarks in parallel.
#include "mfem_raytracing/embree/raytracer.hpp"
#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "mfem_raytracing/embree/device_memory_monitor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mfem_raytracing;
namespace
{
using Clock = std::chrono::steady_clock;
struct Ray { double origin[3]{}; double direction[3]{}; };
struct Configuration
{
    const char *name;
    PatchStoragePolicy storage;
    BvhBuildOptions build;
};
struct Run
{
    // Monitor outlives the tracer, including its scene/device destruction.
    DeviceMemoryMonitor monitor;
    std::unique_ptr<EmbreeRayTracer> tracer;
    const char *name;
    double register_ms = 0.0, commit_ms = 0.0, callbacks_per_ray = 0.0;
    std::vector<double> milliseconds;
    explicit Run(const Configuration &config) :
        tracer(std::make_unique<EmbreeRayTracer>(config.storage, config.build)), name(config.name)
    { monitor.AttachTo(tracer->Device()); }
};

double Milliseconds(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::vector<Ray> MakeRays(const AxisAlignedBounds &box, int grid, const std::string &order)
{
    std::vector<Ray> rays;
    std::mt19937 rng(20260908);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double span = std::max({box.max[0] - box.min[0], box.max[1] - box.min[1],
                                  box.max[2] - box.min[2], 1e-6});
    for (int axis = 0; axis < 3; ++axis)
    {
        for (int y = 0; y < grid; ++y)
        {
            for (int x = 0; x < grid; ++x)
            {
                Ray ray;
                const int a = (axis + 1) % 3, b = (axis + 2) % 3;
                ray.origin[axis] = box.min[axis] - span;
                ray.direction[axis] = 1.0;
                ray.origin[a] = box.min[a] + (x + 0.5) / grid * (box.max[a] - box.min[a]);
                ray.origin[b] = box.min[b] + (y + 0.5) / grid * (box.max[b] - box.min[b]);
                if (order == "divergent")
                {
                    // Varied origins and directions, including outward-facing misses.
                    for (int k = 0; k < 3; ++k)
                    {
                        ray.origin[k] = box.min[k] + (1.4 * unit(rng) - 0.2) * span;
                        ray.direction[k] = 2.0 * unit(rng) - 1.0;
                    }
                }
                rays.push_back(ray);
            }
        }
    }
    if (order == "shuffled") { std::shuffle(rays.begin(), rays.end(), rng); }
    return rays;
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc < 2 || argc > 5)
        {
            std::cerr << "usage: bench_ray_queries leaves.json [grid=128] [trials=7] [coherent|shuffled|divergent]\n";
            return 2;
        }
        const int grid = argc > 2 ? std::stoi(argv[2]) : 128;
        const int trials = argc > 3 ? std::stoi(argv[3]) : 7;
        const std::string order = argc > 4 ? argv[4] : "coherent";
        if (grid < 1 || grid > 2048 || trials < 1 || trials > 101 ||
            (order != "coherent" && order != "shuffled" && order != "divergent"))
        { throw std::invalid_argument("invalid grid, trials or ray order"); }
        const auto scene = LoadLeafPatchScene(argv[1]);
        const auto rays = MakeRays(scene.scene_bbox, grid, order);
        const Configuration configurations[] = {
            {"dense_high", PatchStoragePolicy::Dense, {RTC_BUILD_QUALITY_HIGH, false}},
            {"indexed_high", PatchStoragePolicy::AutoIndexed, {RTC_BUILD_QUALITY_HIGH, false}},
            {"indexed_medium", PatchStoragePolicy::AutoIndexed, {RTC_BUILD_QUALITY_MEDIUM, false}},
            {"indexed_low", PatchStoragePolicy::AutoIndexed, {RTC_BUILD_QUALITY_LOW, false}},
            {"indexed_high_compact", PatchStoragePolicy::AutoIndexed, {RTC_BUILD_QUALITY_HIGH, true}},
            {"indexed_medium_compact", PatchStoragePolicy::AutoIndexed, {RTC_BUILD_QUALITY_MEDIUM, true}},
        };
        std::vector<std::unique_ptr<Run>> runs;
        std::vector<RayHitRecord> reference;
        for (const auto &config : configurations)
        {
            auto run = std::make_unique<Run>(config);
            auto start = Clock::now();
            // Benchmark tooling may inspect diagnostic scenes; does not change admission defaults.
            run->tracer->RegisterLeafPatchScene(scene, true);
            run->register_ms = Milliseconds(start);
            start = Clock::now();
            run->tracer->CommitScene();
            run->commit_ms = Milliseconds(start);
            std::size_t callbacks = 0;
            for (std::size_t i = 0; i < rays.size(); ++i)
            {
                RayQueryDiagnostics diagnostics;
                const auto &ray = rays[i];
                const auto hit = run->tracer->Intersect(ray.origin, ray.direction, 0.0,
                    std::numeric_limits<double>::infinity(), &diagnostics);
                callbacks += diagnostics.kernel_invocations;
                if (runs.empty()) { reference.push_back(hit); }
                else
                {
                    const auto &expected = reference[i];
                    // Alternative trees can select different primitives at coincident
                    // boundaries. Require the same first crossing, not the tie winner.
                    const double tolerance = 1e-5 * std::max(1.0, std::fabs(expected.t));
                    if (hit.hit != expected.hit || (hit.hit &&
                        (!std::isfinite(hit.t) || std::fabs(hit.t - expected.t) > tolerance)))
                    { throw std::runtime_error(std::string(config.name) + ": first crossing mismatch"); }
                }
            }
            run->callbacks_per_ray = static_cast<double>(callbacks) / rays.size();
            runs.push_back(std::move(run));
        }
        std::size_t checksum = 0;
        constexpr int repeats = 4;
        // Rotate ordering each trial to reduce thermal/order bias.
        for (int trial = 0; trial < trials; ++trial)
        {
            for (std::size_t step = 0; step < runs.size(); ++step)
            {
                auto &run = *runs[(step + trial) % runs.size()];
                const auto start = Clock::now();
                for (int repeat = 0; repeat < repeats; ++repeat)
                {
                    for (const auto &ray : rays) { checksum += run.tracer->Intersect(ray.origin, ray.direction).hit; }
                }
                run.milliseconds.push_back(Milliseconds(start));
            }
        }
        std::cerr << "embree=" << RTC_VERSION_STRING << " compiler=" << __VERSION__ << '\n';
        std::cerr << "patches=" << scene.leaves.size() << " rays_per_trial=" << rays.size() * repeats
                  << " trials=" << trials << " order=" << order << " checksum=" << checksum << '\n';
        std::cout << "configuration,median_ms,million_rays_per_second,callbacks_per_ray,register_ms,commit_ms,bvh_bytes,payload_bytes\n";
        for (auto &run : runs)
        {
            auto &times = run->milliseconds;
            std::sort(times.begin(), times.end());
            const double ms = times[times.size() / 2];
            std::cout << run->name << ',' << ms << ',' << rays.size() * repeats / (ms * 1000.0)
                      << ',' << run->callbacks_per_ray << ',' << run->register_ms << ',' << run->commit_ms
                      << ',' << run->monitor.live_bytes() << ',' << run->tracer->StorageStatistics().buffer_bytes << '\n';
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
