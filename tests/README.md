# Unit tests

Simple tests for `Ray` and `IntersectAABB` (no external test framework).

## Build and run

From the project root:

```bash
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

- **Ray**: `Evaluate`, normalized direction, default `t` range and weight
- **IntersectAABB**: hit/miss on a unit square mesh, axis-parallel ray, clipping to `SetTMin`/`SetTMax`
- **Cartesian mesh**: `SetDimension`, `GenerateCartesianMeshName`, builder validation, `ParseCLI`, mesh element/vertex counts for 1D/2D/3D
- **Ray trace (`TraceFindPoints`)**: two-cell and single-cell crossings, miss outside mesh, distinct cells along ray
