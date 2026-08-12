# Bilinear-seeded vs. naive Newton-Raphson seeding benchmark

Does seeding the Newton-Raphson (N-R) ray/NURBS-surface solve with a hit from the
**bilinear patch approximation** — instead of a naive fixed initial guess — actually
help? This harness measures it on the thick-walled 90° pipe elbow
(`meshes/iga/pipe-nurbs.mesh`), separating flat regions from the high-curvature
elbow bend, and answers three questions:

1. Does bilinear seeding cut N-R iteration count?
2. Does it improve robustness (fewer divergences/failures) in high-curvature regions?
3. Does the bilinear lookup (BVH traversal) cost more wall-clock than it saves?

Both strategies run the **identical** N-R solver on the **identical** analytic
NURBS surface — only the initial guess differs.

## What it does

* **Truth surface.** Each of the 16 elbow boundary patches (degree-2 rational
  NURBS; the walls are multi-span in *v* with domain `[0, 2]`) is evaluated by a
  self-contained NURBS evaluator with first derivatives
  ([`nurbs_surface.hpp`](include/nurbs_surface.hpp)). This is independent of MFEM's
  per-element transformation on purpose: the bilinear leaves carry `(u, v)` in the
  **whole-patch** parameter domain, so both truth and seed must live there.
* **Bilinear approximation.** Per patch, `ReduceSurfaceToBilinearLeaves` (the
  existing watertight degree-reduction pipeline) builds a set of degree-(1,1) leaf
  patches at a chosen error budget; the leaves are loaded into an Embree BVH
  ([`bilinear_seeder.hpp`](include/bilinear_seeder.hpp)). A ray's first-hit leaf,
  mapped from its local `[0,1]²` back through `u/v_domain_global`, is the seed.
* **Two seeds.** `naive` = the patch's UV-domain midpoint (the zero-knowledge
  default). `bilinear` = the BVH leaf hit (falls back to the midpoint on a miss,
  flagged in the CSV).
* **Stratified rays.** Reproducible (fixed RNG). Each ray is constructed to pass
  *exactly* through a known interior surface point at a controlled incidence angle
  (0°/40°/65°/82° from the normal), giving ground truth for every ray. Patches are
  split into a **flat** group (planar annular end-caps) and a **bend** group
  (curved elbow walls) — see [`ray sampling`](src/benchmark_main.cpp).
* **Instrumentation.** Per ray × strategy × tolerance: iteration count, status
  (converged/maxiter/diverged), valid-hit flag, final residual, parametric error
  vs. truth, and separate wall-clock timings for the seed lookup and the N-R solve.

## Files

| Path | Role |
|------|------|
| `prepare_pipe_surfaces.py` | Split `pipe_nurbs_border_patches.json` → per-patch `SurfaceData` JSON + `manifest.csv` (patch id, role, flat/bend group). |
| `surfaces/` | Generated per-patch surface JSONs + manifest (checked in after running the prep script). |
| `include/`, `src/` | NURBS evaluator, instrumented Newton solver, bilinear-seed BVH, and the driver. |
| `analyze_results.py` | Aggregate metrics, iteration histograms (text + PNG), and a verdict, split by region. |
| `results.csv`, `summary.txt` | Benchmark output (one row per ray × strategy × tolerance) and the C++ summary table. |

## Build & run

The harness needs Embree (for the BVH). Configure the top-level project with
`-DMFEM_RAYTRACING_ENABLE_EMBREE=ON` (the `build-embree-test/` tree already is):

```bash
cmake -S . -B build-embree-test
cmake --build build-embree-test --target seeding_benchmark -j4
```

Then, **from the repository root** (surface paths in the manifest are repo-relative):

```bash
python3 intersection_seeding_benchmark/prepare_pipe_surfaces.py
./build-embree-test/intersection_seeding_benchmark/seeding_benchmark \
    --out intersection_seeding_benchmark/results.csv
python3 intersection_seeding_benchmark/analyze_results.py
```

### Key options

| Flag | Default | Meaning |
|------|---------|---------|
| `--tols a,b,c` | `0.5,0.1,0.02` | Bilinear error budgets to sweep (coarse → fine). Finer ⇒ more leaves, better seed, larger BVH. |
| `--angles d,d,..` | `0,40,65,82` | Incidence angles (deg from normal). Grazing angles stress the solver. |
| `--grid N` | `12` | N×N parameter samples per patch. Total rays/patch = N²·(#angles). |
| `--max-iter N` | `50` | N-R iteration cap. |
| `--residual-tol X` | `1e-10` | Convergence tolerance on the physical 2D residual. |
| `--standoff X` | `3.0` | Ray origin distance from the surface, in patch diagonals. |

## Interpreting the result

`analyze_results.py` prints a verdict per tolerance following the plan's criteria:

* **Negligible iteration savings + no failure-rate change in the bend** → the added
  complexity isn't justified; naive seeding is fine.
* **Meaningfully fewer failures/divergences in the bend** → justified as a
  robustness measure, even if average iteration savings are small.
* **Worse total wall-clock** (BVH lookup outweighs the N-R savings) → not worth it
  unless robustness is the priority.

The `seedmiss%` column matters: a bilinear seed only helps when the BVH actually
returns a leaf. Grazing rays can slip past a coarse approximation and fall back to
the naive midpoint — the tolerance sweep shows whether refining the leaves fixes it.
