# Unit tests

Simple tests for ray tracing utilities (no external test framework). MFEM types are used throughout to verify library compatibility.

## Build and run

From the `mfem_raytracing` project directory (required for mesh file paths):

```bash
cd mfem_raytracing
cmake -S . -B build
cmake --build build
./build/run_tests
```

On Apple Silicon, if MFEM was built for `x86_64`, configure with:

```bash
cmake -S . -B build -DCMAKE_OSX_ARCHITECTURES=x86_64
```

Or with CTest:

```bash
cd build && ctest
```

## What is covered

| Test group | MFEM objects exercised |
|------------|------------------------|
| **Ray** | `Vector` (evaluate, normalize, segment/weight) |
| **IntersectAABB** | `Mesh`, `Vector`, `GetBoundingBox`; 2D/3D Cartesian and loaded `.mesh` files |
| **Cartesian mesh** | `Mesh`, `GetElementBaseGeometry`, `ElementTransformation`, `IntegrationPoint` |
| **Ray trace** | `Mesh`, `FindPoints`, `DenseMatrix`, `Array<IntegrationPoint>`; 2D and 3D Cartesian meshes |
| **NURBS mesh** | `Mesh`, `NURBSExtension`, `KnotVector`, `ElementTransformation`, `RefinedGeometry` |
| **Element extractor** | `KnotVector` from NURBS meshes, `DenseMatrix`; hand-crafted knot regressions |
| **Bilinear intersection** | `Mesh`, `IsoparametricTransformation`, `IntegrationPoint` (maintained separately) |

Shared helpers in `test_helpers.hpp`: `KnotsFromKV`, `SplineDegreeFromKV`, `FindElemAndIP`.

## Mesh assets

Tests load files under `meshes/iga/` and `meshes/cartesian/`. Run the executable from `mfem_raytracing/` so relative paths resolve.
