#ifndef MFEM_RAYTRACING_TSPLINE_SHELL_JSON_HPP
#define MFEM_RAYTRACING_TSPLINE_SHELL_JSON_HPP

// JSON serialization for a composed degree-one T-spline shell.  This stays
// separate from composition so the runtime-facing artifact and its safety
// certificate can be audited without stepping through mesh construction.

#include "mfem_raytracing/tspline/tspline_shell_composer.hpp"

#include <iosfwd>

namespace mfem_raytracing
{
namespace tspline
{

/// Emit a self-describing leaf scene which remains readable by
/// LoadLeafPatchScene.  In addition to the standard bilinear-leaf fields, the
/// document records source ownership, every error term, seam-strip diagnostics
/// and the complete RT safety certificate.  This function intentionally also
/// writes diagnostic-only shells; callers that are about to ray trace must
/// call RequireShellReadyForRayTracing first.
void WriteBakedTsplineShellJson(std::ostream &os, const BakedTsplineShell &shell);

} // namespace tspline
} // namespace mfem_raytracing

#endif
