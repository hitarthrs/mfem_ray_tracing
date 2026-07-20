# mfem_raytracing

**Enabling ray tracing on degree-reduced NURBS surfaces** — a C++ library that turns high-order CAD / IGA surfaces into watertight bilinear patches and traces rays against them with Embree.

Developed in the Computational Sciences (CPS) Division at Argonne National Laboratory.

---

## Motivation

Monte Carlo particle transport codes such as [OpenMC](https://docs.openmc.org/) rely on fast ray–surface intersection tests for transport, random-ray methods, homogenization, and visualization. Extending those workflows to **CAD-based geometry** means handling **NURBS**, and the usual options have clear drawbacks:

| Approach | Problem |
| --- | --- |
| Direct ray–NURBS intersection | Too expensive for production transport |
| Fine tessellation into triangles | Primitive counts explode at facility scale |

This library takes a middle path: **degree-reduce each NURBS surface to a watertight mesh of bilinear patches**, then ray-trace those instead. The goal is fewer primitives than a comparable tessellation, without giving up robustness or watertightness — geometries that are built once and ray-traced many times.

On fifth-order test surfaces, a competing + coalesced bilinear reduction has shown **up to ~23× fewer primitives** than curvature-adaptive triangulation at tight error tolerances (generation cost grows as the tolerance tightens).

---

## What this repo provides (C++)

### 1. Curve & surface degree reduction (Piegl–Tiller)

- One-step **B-spline / Bézier / NURBS** curve degree reduction with an error tolerance
- **Multi-step surface reduction** from degree \((p,q)\) down to bilinear \((1,1)\)
- Error-driven **global iso-line splitting** so large patches stay within budget
- **Competing** \(u\)/\(v\) steps against a shared error budget
- **Coalescing**: greedy UV line removal that keeps a conforming tensor grid
- **Hard seams**: original unique knots are never crossed during coalesce, so every leaf is a true \(2\times2\) bilinear (important for multi-span rational surfaces such as a torus)

### 2. Watertight bilinear leaf meshes

- Conforming reduction produces a tensor grid of leaves that share identical edge control data (**C⁰** after corner welding)
- Leaf AABBs and control nets can be exported as JSON for inspection or Embree loading
- Boundary-patch utilities for MFEM NURBS meshes (extract face patches, inspect connectivity)

### 3. Embree ray tracing (optional)

When built with Embree enabled:

- Bilinear patches registered as **custom user geometry** (bounds + intersect + occluded callbacks)
- First-hit and occlusion queries via a small `EmbreeRayTracer` wrapper
- **Multi-hit** continuation (`IntersectAll`): a ray that pierces one patch can keep going and hit others (e.g. front and back of a torus)
- Tools to **render** leaf scenes (PPM) and **export** ray grids for interactive viewers

### 4. Supporting mesh utilities

Cartesian mesh builders and NURBS mesh helpers used by examples and tests.

---

## Dependencies

| Dependency | Role |
| --- | --- |
| **CMake** ≥ 3.16 | Build |
| **C++17** compiler | Library & tools |
| **[MFEM](https://mfem.org/)** | NURBS meshes and geometry primitives |
| **[Embree](https://www.embree.org/)** 3.6.1+ or 4.x | Optional BVH ray tracing |

Point CMake at your MFEM build with `MFEM_DIR` (defaults to `$HOME/mesh_software/mfem-4.9/build` if that path exists).

---

## Build

```bash
cmake -S . -B build \
  -DMFEM_DIR=/path/to/mfem/build \
  -DMFEM_RAYTRACING_ENABLE_EMBREE=ON \
  -DEMBREE_DIR=/path/to/embree   # optional if Embree is on CMAKE_PREFIX_PATH

cmake --build build -j
ctest --test-dir build          # or: ./build/run_tests  (from the repo root)
```

Embree discovery lives in [`cmake/Embree.cmake`](cmake/Embree.cmake). Omit `-DMFEM_RAYTRACING_ENABLE_EMBREE=ON` for a reduction-only build.

---

## Library layout

| Path | Contents |
| --- | --- |
| `include/` · `src/` | Core library (`mfem_raytracing`) |
| `include/embree/` · `src/embree/` | Embree tracer & user-geometry callbacks |
| `cmake/` | Embree CMake helpers |
| `tests/` | Unit tests (`run_tests`) and small mesh dump utilities |
| `meshes/` | Example Cartesian and NURBS meshes |

Principal reduction / tracing headers:

- [`include/surface_conforming_reduction.hpp`](include/surface_conforming_reduction.hpp) — watertight multi-step surface reduction + coalesce (+ hard seams)
- [`include/bilinear_leaf_extraction.hpp`](include/bilinear_leaf_extraction.hpp) — bilinear leaf collection for BVH / export
- [`include/embree/raytracer.hpp`](include/embree/raytracer.hpp) — Embree scene, `Intersect` / `IntersectAll` / `Occluded`
- [`include/bilinear_intersect.hpp`](include/bilinear_intersect.hpp) — analytic bilinear patch intersection used by Embree callbacks

---

## Useful executables

| Target | Purpose |
| --- | --- |
| `run_tests` | Full unit suite (reduction, meshes, Embree when enabled) |
| `export_bilinear_patches` | Reduce a surface to watertight bilinears and write leaf JSON |
| `export_leaf_bboxes` | Multi-step reduction → leaf AABB JSON (golden-surface demos) |
| `render_leaf_patches` | Embree render of a leaf JSON → shaded / UV / top-down PPM |
| `export_scene_json` | Embree multi-hit ray grid over a leaf scene (for visualization) |
| `bench_*_degree_reduction` | Curve / surface reduction microbenchmarks |

Example (Embree build):

```bash
# From the repository root so relative mesh/JSON paths resolve
./build/export_bilinear_patches --help
./build/render_leaf_patches path/to/leaves.json out/prefix 512
./build/export_scene_json path/to/leaves.json out/scene.json 24
```

---

## Design notes

- **Robustness over remeshing speed.** Bilinear generation can be slower than tessellation at tight tolerances; the payoff is fewer ray-tracing primitives for repeated queries.
- **Watertightness first.** Conforming splits + coalesce + corner welding keep shared leaf edges C⁰. Hard seams prevent multi-span “fake” degree-1 nets on surfaces with interior knots.
- **Bridge to IGA.** The same leaf representation is a natural substrate for ray tracing on isogeometric meshes, not only CAD shells.

---

## References

1. Piegl, L., & Tiller, W. (1997). *The NURBS Book* (2nd ed.). Springer.
2. Intel Embree — [https://www.embree.org/](https://www.embree.org/)
3. MFEM — [https://mfem.org/](https://mfem.org/)

---

## License / status

Research / internship prototype for CAD ↔ Monte Carlo ray-tracing workflows. See repository metadata for license and citation details as they are published.
