#ifndef MFEM_RAYTRACING_TSPLINE_DEGREE_ONE_BAKE_HPP
#define MFEM_RAYTRACING_TSPLINE_DEGREE_ONE_BAKE_HPP

#include "mfem_raytracing/tspline/tspline_bilinear_ops.hpp"
#include "mfem_raytracing/tspline/tspline_strip_builder.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

enum class BakedPrimitiveKind
{
    ExactFace,
    JunctionPiece,
};

struct FaceBakeDiagnostic
{
    std::size_t face_index = 0;
    bool flat_face_is_exact = false;
    double flat_face_max_deviation = 0.0;
    std::size_t emitted_primitive_count = 0;
};

/// One exact rational-bilinear restriction of an active degree-1 T-spline
/// face.  Its parameter rectangle is local to the seam strip's (s,t) domain.
struct BakedTSplinePrimitive
{
    LeafPatch leaf;
    BakedPrimitiveKind kind = BakedPrimitiveKind::ExactFace;
    std::size_t source_face_index = 0;
    int source_patch_id = -1;
    std::size_t source_leaf_index = 0;
    double source_reduction_error = 0.0;
    /// [s0, s1, t0, t1]
    std::array<double, 4> rect = {};
};

struct BakedDegreeOneStrip
{
    std::vector<BakedTSplinePrimitive> primitives;
    std::vector<FaceBakeDiagnostic> faces;
    std::size_t n_exact_primitives = 0;
    std::size_t n_junction_primitives = 0;
};

struct DegreeOneBakeOptions
{
    double geometry_tolerance = 1e-9;
    std::array<double, 3> exactness_fractions = {{0.15, 0.5, 0.85}};
    std::array<double, 6> verification_fractions = {{0.05, 0.23, 0.41, 0.59, 0.77, 0.95}};
};

struct BakeVerification
{
    bool exact = false;
    double max_decomposition_error = 0.0;
    std::size_t sample_count = 0;
};

/// Bake only active faces from a local degree-1 strip.  Flat faces which
/// already equal the T-spline stay one primitive; junction faces are split at
/// hanging-vertex crease lines and evaluated in homogeneous coordinates.
BakedDegreeOneStrip BakeDegreeOneStrip(const LocalDegreeOneStrip &strip,
                                       const DegreeOneBakeOptions &options = {});

BakeVerification VerifyBakedDegreeOneStrip(const LocalDegreeOneStrip &strip,
                                           const BakedDegreeOneStrip &baked,
                                           const DegreeOneBakeOptions &options = {});

const char *BakedPrimitiveKindName(BakedPrimitiveKind kind);

} // namespace tspline
} // namespace mfem_raytracing

#endif
