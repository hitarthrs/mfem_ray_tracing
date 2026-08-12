#ifndef MFEM_RAYTRACING_PIPELINE_BAKE_HPP
#define MFEM_RAYTRACING_PIPELINE_BAKE_HPP

// Public Stage 4: bake a multi-patch T-mesh mid-product to an RT leaf scene.

#include "mfem_raytracing/pipeline/multi_patch_tmesh.hpp"
#include "mfem_raytracing/tspline/tspline_shell_composer.hpp"

namespace mfem_raytracing
{
namespace tspline
{

/// RT-certified (or diagnostic) bilinear leaf scene produced by baking.
using RtLeafScene = BakedTsplineShell;

/// Bake + certify from an always-materialized MultiPatchTMesh.
/// Uses the assembly's corner policy (exact collar by default) for behavior
/// parity with the historical ComposeBakedTsplineShell path.
RtLeafScene BakeForRayTracing(const MultiPatchTMesh &mesh);

} // namespace tspline
} // namespace mfem_raytracing

#endif
