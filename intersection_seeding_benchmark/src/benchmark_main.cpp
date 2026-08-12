// Bilinear-seeded vs. naive Newton-Raphson seeding benchmark on the pipe elbow.
//
// For every boundary patch of the thick-walled 90 deg elbow (flat annular
// end-caps vs. curved elbow walls), generate a reproducible stratified ray set
// that passes through known surface points at controlled incidence angles, then
// solve the ray/NURBS intersection with two initial guesses:
//
//   naive     : the patch's UV-domain midpoint (the zero-knowledge default).
//   bilinear  : the (u, v) implied by the first-hit leaf of the patch's
//               tolerance-controlled bilinear approximation (Embree BVH lookup).
//
// Both run the identical Newton solver on the identical analytic surface; only
// the seed differs. One CSV row per (ray x strategy x tolerance) is written for
// later analysis, and an aggregate summary is printed.
//
// Run from the repository root (surface paths in the manifest are repo-relative):
//   ./intersection_seeding_benchmark/seeding_benchmark \
//       --manifest intersection_seeding_benchmark/surfaces/manifest.csv \
//       --out intersection_seeding_benchmark/results.csv

#include "bilinear_seeder.hpp"
#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "newton_intersect.hpp"
#include "nurbs_surface.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{

using seeding_benchmark::NewtonConfig;
using seeding_benchmark::NewtonIntersect;
using seeding_benchmark::NewtonResult;
using seeding_benchmark::NewtonStatus;
using seeding_benchmark::NurbsSurface;
using seeding_benchmark::Ray;
using seeding_benchmark::SurfaceSample;
using seeding_benchmark::Vec3;

struct PatchEntry
{
    int patch_id = 0;
    std::string role;
    std::string group;  // "flat" | "bend"
    std::string name;
    std::string json;
};

double Dot(const Vec3 &a, const Vec3 &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 Cross(const Vec3 &a, const Vec3 &b)
{
    return {{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

Vec3 Normalize(const Vec3 &a)
{
    const double n = std::sqrt(Dot(a, a));
    if (n <= 1e-300)
    {
        return {{0.0, 0.0, 0.0}};
    }
    return {{a[0] / n, a[1] / n, a[2] / n}};
}

std::vector<PatchEntry> ReadManifest(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("cannot open manifest: " + path);
    }
    std::vector<PatchEntry> entries;
    std::string line;
    bool header = true;
    while (std::getline(in, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();  // Python csv writes CRLF line endings
        }
        if (line.empty())
        {
            continue;
        }
        if (header)
        {
            header = false;  // skip "patch_id,role,group,name,json"
            continue;
        }
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;
        while (std::getline(ss, field, ','))
        {
            fields.push_back(field);
        }
        if (fields.size() < 5)
        {
            continue;
        }
        PatchEntry e;
        e.patch_id = std::stoi(fields[0]);
        e.role = fields[1];
        e.group = fields[2];
        e.name = fields[3];
        e.json = fields[4];
        entries.push_back(e);
    }
    return entries;
}

// Characteristic length of a patch from its control-net bounding box diagonal.
double CharacteristicLength(const NurbsSurface &surface)
{
    const auto &cps = surface.Data().control_points;
    Vec3 lo{{1e300, 1e300, 1e300}};
    Vec3 hi{{-1e300, -1e300, -1e300}};
    for (const auto &row : cps)
    {
        for (const auto &p : row)
        {
            for (int c = 0; c < 3; ++c)
            {
                lo[c] = std::min(lo[c], p[c]);
                hi[c] = std::max(hi[c], p[c]);
            }
        }
    }
    const Vec3 d{{hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]}};
    const double diag = std::sqrt(Dot(d, d));
    return diag > 1e-12 ? diag : 1.0;
}

struct GeneratedRay
{
    Ray ray;
    double u_star = 0.0;
    double v_star = 0.0;
    Vec3 P{{0.0, 0.0, 0.0}};  // true hit point
    double incidence_deg = 0.0;
    int ray_id = 0;
};

// Build a stratified ray set that passes exactly through interior surface
// points at prescribed incidence angles (measured from the surface normal).
std::vector<GeneratedRay> GenerateRays(const NurbsSurface &surface,
                                       int grid,
                                       const std::vector<double> &angles_deg,
                                       double standoff_scale,
                                       std::mt19937 &rng)
{
    std::vector<GeneratedRay> rays;
    const auto ud = surface.UDomain();
    const auto vd = surface.VDomain();
    const double u0 = ud.first, u1 = ud.second;
    const double v0 = vd.first, v1 = vd.second;
    const double L = standoff_scale * CharacteristicLength(surface);

    std::uniform_real_distribution<double> azimuth(0.0, 2.0 * M_PI);

    int ray_id = 0;
    // Interior grid, avoiding exact edges where a normal can degenerate.
    for (int iu = 0; iu < grid; ++iu)
    {
        const double fu = (iu + 0.5) / grid;
        const double us = u0 + fu * (u1 - u0);
        for (int iv = 0; iv < grid; ++iv)
        {
            const double fv = (iv + 0.5) / grid;
            const double vs = v0 + fv * (v1 - v0);

            const SurfaceSample s = surface.Evaluate(us, vs);
            const Vec3 N = surface.Normal(us, vs);
            if (Dot(N, N) < 0.5)
            {
                continue;  // degenerate normal; skip
            }
            // Tangent basis in the plane perpendicular to N.
            Vec3 t1 = Normalize(s.dSdu);
            if (Dot(t1, t1) < 0.5)
            {
                t1 = Normalize(s.dSdv);
            }
            const Vec3 t2 = Normalize(Cross(N, t1));

            for (double theta_deg : angles_deg)
            {
                const double theta = theta_deg * M_PI / 180.0;
                const double phi = azimuth(rng);
                const Vec3 T{{std::cos(phi) * t1[0] + std::sin(phi) * t2[0],
                              std::cos(phi) * t1[1] + std::sin(phi) * t2[1],
                              std::cos(phi) * t1[2] + std::sin(phi) * t2[2]}};
                // Origin offset from P: cos(theta) along N, sin(theta) along T.
                const Vec3 off{{std::cos(theta) * N[0] + std::sin(theta) * T[0],
                                std::cos(theta) * N[1] + std::sin(theta) * T[1],
                                std::cos(theta) * N[2] + std::sin(theta) * T[2]}};
                GeneratedRay g;
                g.P = s.S;
                g.u_star = us;
                g.v_star = vs;
                g.incidence_deg = theta_deg;
                g.ray_id = ray_id++;
                g.ray.origin = {{s.S[0] + L * off[0], s.S[1] + L * off[1],
                                 s.S[2] + L * off[2]}};
                g.ray.dir = Normalize({{s.S[0] - g.ray.origin[0], s.S[1] - g.ray.origin[1],
                                        s.S[2] - g.ray.origin[2]}});
                g.ray.tmin = 1e-6 * L;
                g.ray.tmax = 4.0 * L;
                rays.push_back(g);
            }
        }
    }
    return rays;
}

const char *StatusString(NewtonStatus s)
{
    switch (s)
    {
    case NewtonStatus::Converged:
        return "converged";
    case NewtonStatus::MaxIter:
        return "maxiter";
    case NewtonStatus::Diverged:
        return "diverged";
    }
    return "unknown";
}

// Aggregate accumulator per (tolerance, group, strategy).
struct Agg
{
    long n = 0;
    long success = 0;   // valid_hit and converged to the right point
    long seed_miss = 0; // bilinear seed found no leaf
    std::vector<int> iters;
    double newton_ns_sum = 0.0;
    double seed_ns_sum = 0.0;

    void Add(bool ok, int it, double nns, double sns, bool smiss)
    {
        ++n;
        if (ok)
        {
            ++success;
        }
        if (smiss)
        {
            ++seed_miss;
        }
        iters.push_back(it);
        newton_ns_sum += nns;
        seed_ns_sum += sns;
    }
    double MeanIters() const
    {
        if (iters.empty())
            return 0.0;
        double s = 0.0;
        for (int x : iters)
            s += x;
        return s / iters.size();
    }
    int MedianIters()
    {
        if (iters.empty())
            return 0;
        std::sort(iters.begin(), iters.end());
        return iters[iters.size() / 2];
    }
    int MaxIters() const
    {
        int m = 0;
        for (int x : iters)
            m = std::max(m, x);
        return m;
    }
    double FailRate() const
    {
        return n ? static_cast<double>(n - success) / n : 0.0;
    }
};

}  // namespace

int main(int argc, char **argv)
{
    std::string manifest_path = "intersection_seeding_benchmark/surfaces/manifest.csv";
    std::string out_path = "intersection_seeding_benchmark/results.csv";
    std::vector<double> tols = {0.5, 0.1, 0.02};
    std::vector<double> angles = {0.0, 40.0, 65.0, 82.0};
    int grid = 12;
    double standoff = 3.0;
    double box_bump = 0.0;  // absolute AABB padding on leaf boxes
    NewtonConfig cfg;  // max_iter=50, residual_tol=1e-10
    unsigned int seed = 20240723u;
    const double success_param_tol = 1e-3;  // normalized-domain closeness to truth

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
            {
                std::cerr << "missing value for " << a << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--manifest")
            manifest_path = next();
        else if (a == "--out")
            out_path = next();
        else if (a == "--grid")
            grid = std::stoi(next());
        else if (a == "--standoff")
            standoff = std::stod(next());
        else if (a == "--box-bump")
            box_bump = std::stod(next());
        else if (a == "--max-iter")
            cfg.max_iter = std::stoi(next());
        else if (a == "--residual-tol")
            cfg.residual_tol = std::stod(next());
        else if (a == "--seed")
            seed = static_cast<unsigned int>(std::stoul(next()));
        else if (a == "--tols")
        {
            tols.clear();
            std::stringstream ss(next());
            std::string tok;
            while (std::getline(ss, tok, ','))
                tols.push_back(std::stod(tok));
        }
        else if (a == "--angles")
        {
            angles.clear();
            std::stringstream ss(next());
            std::string tok;
            while (std::getline(ss, tok, ','))
                angles.push_back(std::stod(tok));
        }
        else if (a == "-h" || a == "--help")
        {
            std::cout
                << "usage: seeding_benchmark [--manifest CSV] [--out CSV] "
                   "[--tols a,b,c]\n"
                   "  [--angles d,d,..] [--grid N] [--standoff X] [--max-iter N] "
                   "[--residual-tol X] [--seed N]\n";
            return 0;
        }
        else
        {
            std::cerr << "unknown argument: " << a << "\n";
            return 1;
        }
    }

    std::vector<PatchEntry> patches;
    try
    {
        patches = ReadManifest(manifest_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "loaded " << patches.size() << " patches from " << manifest_path
              << "\n";

    std::ofstream csv(out_path);
    if (!csv)
    {
        std::cerr << "cannot write " << out_path << "\n";
        return 1;
    }
    csv << "tol,patch_id,role,group,ray_id,incidence_deg,strategy,seed_hit,"
           "seed_u,seed_v,iters,status,valid_hit,success,residual,param_err,"
           "phys_err,seed_ns,newton_ns,u_star,v_star,u_final,v_final\n";
    csv << std::setprecision(10);

    // key: (tol_index, group, strategy) -> Agg
    std::map<std::tuple<int, std::string, std::string>, Agg> agg;

    for (std::size_t ti = 0; ti < tols.size(); ++ti)
    {
        const double tol = tols[ti];
        std::cout << "=== tolerance " << tol << " ===\n";

        for (const PatchEntry &pe : patches)
        {
            mfem_raytracing::SurfaceData sd;
            try
            {
                sd = mfem_raytracing::LoadSurfaceDataJson(pe.json);
            }
            catch (const std::exception &e)
            {
                std::cerr << "  skip patch " << pe.patch_id << ": " << e.what()
                          << "\n";
                continue;
            }
            NurbsSurface surface(sd);

            // Same ray set for every tolerance (seed RNG reset per patch).
            std::mt19937 rng(seed + static_cast<unsigned int>(pe.patch_id));
            const std::vector<GeneratedRay> rays =
                GenerateRays(surface, grid, angles, standoff, rng);

            seeding_benchmark::BilinearSeeder seeder(sd, tol, box_bump);

            const double u_span = surface.UDomain().second - surface.UDomain().first;
            const double v_span = surface.VDomain().second - surface.VDomain().first;

            for (const GeneratedRay &g : rays)
            {
                // --- Naive strategy: UV-domain midpoint seed. ---
                {
                    const double su = surface.UMid();
                    const double sv = surface.VMid();
                    const auto t0 = std::chrono::steady_clock::now();
                    NewtonResult r = NewtonIntersect(surface, g.ray, su, sv, cfg);
                    const auto t1 = std::chrono::steady_clock::now();
                    const double nns =
                        std::chrono::duration<double, std::nano>(t1 - t0).count();

                    const double pu = (r.u - g.u_star) / u_span;
                    const double pv = (r.v - g.v_star) / v_span;
                    const double param_err = std::sqrt(pu * pu + pv * pv);
                    const Vec3 Sf = surface.Point(r.u, r.v);
                    const double phys_err = std::sqrt(
                        (Sf[0] - g.P[0]) * (Sf[0] - g.P[0]) +
                        (Sf[1] - g.P[1]) * (Sf[1] - g.P[1]) +
                        (Sf[2] - g.P[2]) * (Sf[2] - g.P[2]));
                    const bool success = r.valid_hit && param_err < success_param_tol;

                    csv << tol << ',' << pe.patch_id << ',' << pe.role << ','
                        << pe.group << ',' << g.ray_id << ',' << g.incidence_deg
                        << ",naive,1," << su << ',' << sv << ',' << r.iterations
                        << ',' << StatusString(r.status) << ',' << (r.valid_hit ? 1 : 0)
                        << ',' << (success ? 1 : 0) << ',' << r.residual << ','
                        << param_err << ',' << phys_err << ",0," << nns << ','
                        << g.u_star << ',' << g.v_star << ',' << r.u << ',' << r.v
                        << '\n';

                    agg[std::make_tuple((int)ti, pe.group, std::string("naive"))]
                        .Add(success, r.iterations, nns, 0.0, false);
                }

                // --- Bilinear-seeded strategy. ---
                {
                    const auto s0 = std::chrono::steady_clock::now();
                    const seeding_benchmark::SeedResult seed_res = seeder.Seed(g.ray);
                    const auto s1 = std::chrono::steady_clock::now();
                    const double sns =
                        std::chrono::duration<double, std::nano>(s1 - s0).count();

                    double su, sv;
                    if (seed_res.hit)
                    {
                        su = seed_res.u;
                        sv = seed_res.v;
                    }
                    else
                    {
                        su = surface.UMid();  // fall back to naive seed on a miss
                        sv = surface.VMid();
                    }

                    const auto t0 = std::chrono::steady_clock::now();
                    NewtonResult r = NewtonIntersect(surface, g.ray, su, sv, cfg);
                    const auto t1 = std::chrono::steady_clock::now();
                    const double nns =
                        std::chrono::duration<double, std::nano>(t1 - t0).count();

                    const double pu = (r.u - g.u_star) / u_span;
                    const double pv = (r.v - g.v_star) / v_span;
                    const double param_err = std::sqrt(pu * pu + pv * pv);
                    const Vec3 Sf = surface.Point(r.u, r.v);
                    const double phys_err = std::sqrt(
                        (Sf[0] - g.P[0]) * (Sf[0] - g.P[0]) +
                        (Sf[1] - g.P[1]) * (Sf[1] - g.P[1]) +
                        (Sf[2] - g.P[2]) * (Sf[2] - g.P[2]));
                    const bool success = r.valid_hit && param_err < success_param_tol;

                    csv << tol << ',' << pe.patch_id << ',' << pe.role << ','
                        << pe.group << ',' << g.ray_id << ',' << g.incidence_deg
                        << ",bilinear," << (seed_res.hit ? 1 : 0) << ',' << su << ','
                        << sv << ',' << r.iterations << ',' << StatusString(r.status)
                        << ',' << (r.valid_hit ? 1 : 0) << ',' << (success ? 1 : 0)
                        << ',' << r.residual << ',' << param_err << ',' << phys_err
                        << ',' << sns << ',' << nns << ',' << g.u_star << ','
                        << g.v_star << ',' << r.u << ',' << r.v << '\n';

                    agg[std::make_tuple((int)ti, pe.group, std::string("bilinear"))]
                        .Add(success, r.iterations, nns, sns, !seed_res.hit);
                }
            }
        }
    }
    csv.close();
    std::cout << "wrote " << out_path << "\n\n";

    // ---- Aggregate summary ----
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "SUMMARY (per tolerance x group x strategy)\n";
    std::cout << "tol      group  strategy  n      mean_it  med_it  max_it  "
                 "fail%   seedmiss%  newton_us  seed_us\n";
    for (std::size_t ti = 0; ti < tols.size(); ++ti)
    {
        for (const std::string &grp : {std::string("flat"), std::string("bend")})
        {
            for (const std::string &strat :
                 {std::string("naive"), std::string("bilinear")})
            {
                auto it = agg.find(std::make_tuple((int)ti, grp, strat));
                if (it == agg.end())
                    continue;
                Agg &a = it->second;
                std::cout << std::setw(7) << tols[ti] << "  " << std::setw(5) << grp
                          << "  " << std::setw(8) << strat << "  " << std::setw(6)
                          << a.n << "  " << std::setw(7) << a.MeanIters() << "  "
                          << std::setw(6) << a.MedianIters() << "  " << std::setw(6)
                          << a.MaxIters() << "  " << std::setw(6)
                          << 100.0 * a.FailRate() << "  " << std::setw(9)
                          << 100.0 * (a.n ? (double)a.seed_miss / a.n : 0.0) << "  "
                          << std::setw(9) << (a.n ? a.newton_ns_sum / a.n / 1000.0 : 0.0)
                          << "  " << std::setw(7)
                          << (a.n ? a.seed_ns_sum / a.n / 1000.0 : 0.0) << "\n";
            }
        }
    }
    return 0;
}
