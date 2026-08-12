#ifndef MFEM_RAYTRACING_TSPLINE_STRIP_BUILDER_HPP
#define MFEM_RAYTRACING_TSPLINE_STRIP_BUILDER_HPP

#include "mfem_raytracing/tspline/tspline_average_merge.hpp"
#include "mfem_raytracing/tspline/tspline.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

enum class StripRegion
{
    A,
    B,
};

/// One original bilinear rectangle embedded in the local (s,t) T-mesh
/// pre-image.  Future baking operates from this explicit face list rather
/// than treating every Cartesian knot cell as an active face.
struct LocalTSplineFace
{
    StripRegion region = StripRegion::A;
    int source_patch_id = -1;
    std::size_t source_leaf_index = 0;
    double source_reduction_error = 0.0;
    std::array<std::size_t, 4> control_ids = {};
    /// [s0, s1, t0, t1]
    std::array<double, 4> rect = {};
};

struct LocalDegreeOneStrip
{
    TMesh mesh;
    std::vector<LocalTSplineFace> faces;
    std::vector<double> s_breaks;
    std::vector<double> t_breaks;
    std::vector<std::size_t> t_junctions;
    std::size_t rule2_added_edges = 0;
    double max_vertex_disagreement = 0.0;
    double max_vertex_weight_disagreement = 0.0;
};

struct LocalStripBuildOptions
{
    double parameter_tolerance = 1e-9;
    double geometry_tolerance = 1e-6;
};

/// Unfold two average-merged bands into one degree-1 T-mesh.  The returned
/// mesh uses a seam at s=0, A in s<=0, B in s>=0, and preserves every source
/// leaf as a face record for exact later RT baking.
LocalDegreeOneStrip BuildLocalDegreeOneStrip(
    const AverageMergedSeam &merged,
    const LocalStripBuildOptions &options = {});

} // namespace tspline
} // namespace mfem_raytracing

#endif
