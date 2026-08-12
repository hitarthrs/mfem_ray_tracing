#ifndef MFEM_RAYTRACING_MFEM_RAYTRACING_HPP
#define MFEM_RAYTRACING_MFEM_RAYTRACING_HPP

// Public umbrella for the Reduce → Connect → T-mesh → Bake pipeline.
// Prefer these headers from application code. Internal tspline helpers
// (strip builder, average-merge knobs, etc.) remain available under
// mfem_raytracing/tspline/ for library development and tests.

#include "mfem_raytracing/pipeline/reduce.hpp"
#include "mfem_raytracing/pipeline/connect.hpp"
#include "mfem_raytracing/pipeline/multi_patch_tmesh.hpp"
#include "mfem_raytracing/pipeline/bake.hpp"

#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "mfem_raytracing/reduction/hard_seam_bilinearization.hpp"

#endif
