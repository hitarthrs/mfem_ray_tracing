#include "mfem_raytracing/tspline/tspline_degree_one_bake.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

HomogeneousPoint ToHomogeneous(const ControlPoint &control)
{
    return {control.weight * control.position[0], control.weight * control.position[1],
            control.weight * control.position[2], control.weight};
}

HomogeneousBilinearNet FaceNet(const LocalDegreeOneStrip &strip, const LocalTSplineFace &face)
{
    HomogeneousBilinearNet result{};
    const auto &controls = strip.mesh.ControlPoints();
    // LocalTSplineFace follows the T-mesh cycle
    //   (s0,t0), (s1,t0), (s1,t1), (s0,t1),
    // whereas a LeafPatch/HomogeneousBilinearNet uses Embree's order
    //   (s0,t0), (s0,t1), (s1,t0), (s1,t1).
    // Keep this conversion here, at the topology-to-bilinear boundary, so
    // the rest of the bake path has one unambiguous corner convention.
    constexpr std::array<int, 4> kFaceToBilinear = {{0, 3, 1, 2}};
    for (int corner = 0; corner < 4; ++corner)
    {
        const std::size_t control_id = face.control_ids[kFaceToBilinear[corner]];
        if (control_id >= controls.size())
        {
            throw std::out_of_range("local T-spline face references an invalid control id");
        }
        result[corner] = ToHomogeneous(controls[control_id]);
    }
    return result;
}

double SceneScale(const LocalDegreeOneStrip &strip)
{
    double scale = 1.0;
    for (const ControlPoint &control : strip.mesh.ControlPoints())
    {
        for (const double value : control.position) { scale = std::max(scale, std::abs(value)); }
    }
    return scale;
}

LeafPatch MakeBakedLeaf(const HomogeneousBilinearNet &net, const LocalTSplineFace &face,
                         const std::array<double, 4> &rect, BakedPrimitiveKind kind)
{
    LeafPatch leaf;
    leaf.patch_id = face.source_patch_id;
    leaf.role = kind == BakedPrimitiveKind::ExactFace ? "seam-exact" : "seam-phantom";
    leaf.kind = BakedPrimitiveKindName(kind);
    leaf.u_domain_global[0] = rect[0];
    leaf.u_domain_global[1] = rect[1];
    leaf.v_domain_global[0] = rect[2];
    leaf.v_domain_global[1] = rect[3];
    SetHomogeneousBilinearNet(leaf, net);
    return leaf;
}

double FaceFlatDeviation(const TSplineSurface &surface, const HomogeneousBilinearNet &net,
                         const std::array<double, 4> &rect,
                         const std::array<double, 3> &fractions)
{
    double result = 0.0;
    for (const double fu : fractions)
    {
        for (const double fv : fractions)
        {
            const double s = rect[0] + fu * (rect[1] - rect[0]);
            const double t = rect[2] + fv * (rect[3] - rect[2]);
            const auto flat = ProjectHomogeneous(EvaluateHomogeneousBilinear(net, fu, fv));
            const Point3 exact = surface.Evaluate(s, t);
            const double dx = flat[0] - exact[0];
            const double dy = flat[1] - exact[1];
            const double dz = flat[2] - exact[2];
            result = std::max(result, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
    }
    return result;
}

void InsertStrict(std::vector<double> &values, double value, double lo, double hi,
                  double tolerance)
{
    if (lo + tolerance < value && value < hi - tolerance) { values.push_back(value); }
}

void SortAndUnique(std::vector<double> &values, double tolerance)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end(), [tolerance](double a, double b) {
        return std::abs(a - b) <= tolerance;
    }), values.end());
}

std::pair<std::vector<double>, std::vector<double>> FaceCreaseLines(
    const LocalDegreeOneStrip &strip, const LocalTSplineFace &face, double tolerance)
{
    std::vector<double> s = {face.rect[0], face.rect[1]};
    std::vector<double> t = {face.rect[2], face.rect[3]};
    for (const ControlPoint &control : strip.mesh.ControlPoints())
    {
        const double cs = control.parameter[0];
        const double ct = control.parameter[1];
        if (std::abs(cs - face.rect[0]) <= tolerance ||
            std::abs(cs - face.rect[1]) <= tolerance)
        {
            InsertStrict(t, ct, face.rect[2], face.rect[3], tolerance);
        }
        if (std::abs(ct - face.rect[2]) <= tolerance ||
            std::abs(ct - face.rect[3]) <= tolerance)
        {
            InsertStrict(s, cs, face.rect[0], face.rect[1], tolerance);
        }
    }
    SortAndUnique(s, tolerance);
    SortAndUnique(t, tolerance);
    return {std::move(s), std::move(t)};
}

BakedTSplinePrimitive MakePrimitive(const TSplineSurface &surface, const LocalTSplineFace &face,
                                    std::size_t face_index, const std::array<double, 4> &rect,
                                    BakedPrimitiveKind kind)
{
    HomogeneousBilinearNet net{};
    net[0] = surface.EvaluateHomogeneous(rect[0], rect[2]);
    net[1] = surface.EvaluateHomogeneous(rect[0], rect[3]);
    net[2] = surface.EvaluateHomogeneous(rect[1], rect[2]);
    net[3] = surface.EvaluateHomogeneous(rect[1], rect[3]);
    BakedTSplinePrimitive result;
    result.leaf = MakeBakedLeaf(net, face, rect, kind);
    result.kind = kind;
    result.source_face_index = face_index;
    result.source_patch_id = face.source_patch_id;
    result.source_leaf_index = face.source_leaf_index;
    result.source_reduction_error = face.source_reduction_error;
    result.rect = rect;
    return result;
}

} // namespace

const char *BakedPrimitiveKindName(BakedPrimitiveKind kind)
{
    return kind == BakedPrimitiveKind::ExactFace ? "exact-face" : "junction-piece";
}

BakedDegreeOneStrip BakeDegreeOneStrip(const LocalDegreeOneStrip &strip,
                                       const DegreeOneBakeOptions &options)
{
    if (options.geometry_tolerance < 0.0)
    {
        throw std::invalid_argument("degree-one bake tolerance must be non-negative");
    }
    const TSplineSurface surface(strip.mesh, 1);
    const double exact_tolerance = options.geometry_tolerance * SceneScale(strip);
    BakedDegreeOneStrip result;
    for (std::size_t face_index = 0; face_index < strip.faces.size(); ++face_index)
    {
        const LocalTSplineFace &face = strip.faces[face_index];
        const HomogeneousBilinearNet flat = FaceNet(strip, face);
        const double deviation = FaceFlatDeviation(surface, flat, face.rect,
                                                   options.exactness_fractions);
        FaceBakeDiagnostic diagnostic;
        diagnostic.face_index = face_index;
        diagnostic.flat_face_max_deviation = deviation;
        diagnostic.flat_face_is_exact = deviation <= exact_tolerance;
        if (diagnostic.flat_face_is_exact)
        {
            result.primitives.push_back(MakePrimitive(surface, face, face_index, face.rect,
                                                      BakedPrimitiveKind::ExactFace));
            diagnostic.emitted_primitive_count = 1;
            ++result.n_exact_primitives;
        }
        else
        {
            const auto [s_lines, t_lines] = FaceCreaseLines(strip, face, options.geometry_tolerance);
            for (std::size_t i = 0; i + 1 < s_lines.size(); ++i)
            {
                for (std::size_t j = 0; j + 1 < t_lines.size(); ++j)
                {
                    const std::array<double, 4> rect = {
                        s_lines[i], s_lines[i + 1], t_lines[j], t_lines[j + 1]};
                    result.primitives.push_back(MakePrimitive(surface, face, face_index, rect,
                                                              BakedPrimitiveKind::JunctionPiece));
                    ++diagnostic.emitted_primitive_count;
                    ++result.n_junction_primitives;
                }
            }
        }
        result.faces.push_back(diagnostic);
    }
    return result;
}

BakeVerification VerifyBakedDegreeOneStrip(const LocalDegreeOneStrip &strip,
                                           const BakedDegreeOneStrip &baked,
                                           const DegreeOneBakeOptions &options)
{
    const TSplineSurface surface(strip.mesh, 1);
    BakeVerification result;
    for (const BakedTSplinePrimitive &primitive : baked.primitives)
    {
        const HomogeneousBilinearNet net = HomogenizeBilinearLeaf(primitive.leaf);
        for (const double fu : options.verification_fractions)
        {
            for (const double fv : options.verification_fractions)
            {
                const auto got = ProjectHomogeneous(EvaluateHomogeneousBilinear(net, fu, fv));
                const double s = primitive.rect[0] + fu * (primitive.rect[1] - primitive.rect[0]);
                const double t = primitive.rect[2] + fv * (primitive.rect[3] - primitive.rect[2]);
                const Point3 expected = surface.Evaluate(s, t);
                const double dx = got[0] - expected[0];
                const double dy = got[1] - expected[1];
                const double dz = got[2] - expected[2];
                result.max_decomposition_error = std::max(
                    result.max_decomposition_error, std::sqrt(dx * dx + dy * dy + dz * dz));
                ++result.sample_count;
            }
        }
    }
    result.exact = result.max_decomposition_error <= options.geometry_tolerance * SceneScale(strip);
    return result;
}

} // namespace tspline
} // namespace mfem_raytracing
