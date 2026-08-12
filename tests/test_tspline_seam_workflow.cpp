#include "mfem_raytracing/embree/leaf_patch_loader.hpp"
#include "mfem_raytracing/pipeline/bake.hpp"
#include "mfem_raytracing/pipeline/connect.hpp"
#include "mfem_raytracing/pipeline/multi_patch_tmesh.hpp"
#include "mfem_raytracing/pipeline/reduce.hpp"
#include "mfem_raytracing/reduction/hard_seam_bilinearization.hpp"
#include "mfem_raytracing/tspline/tspline_average_merge.hpp"
#include "mfem_raytracing/tspline/tspline_bilinear_ops.hpp"
#include "mfem_raytracing/tspline/tspline_leaf_assembly.hpp"
#include "mfem_raytracing/tspline/tspline_patch_interfaces.hpp"
#include "mfem_raytracing/tspline/tspline_degree_one_bake.hpp"
#include "mfem_raytracing/tspline/tspline_shell_composer.hpp"
#include "mfem_raytracing/tspline/tspline_shell_json.hpp"
#include "mfem_raytracing/tspline/tspline_shell_watertightness.hpp"
#include "mfem_raytracing/tspline/tspline_strip_builder.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace
{

using namespace mfem_raytracing;
using namespace mfem_raytracing::tspline;

int failures = 0;

void Check(bool condition, const std::string &message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

LeafPatch MakeLeaf(int patch_id, double u0, double u1, double v0, double v1)
{
    LeafPatch leaf;
    leaf.patch_id = patch_id;
    leaf.role = "test";
    leaf.u_domain_global[0] = u0;
    leaf.u_domain_global[1] = u1;
    leaf.v_domain_global[0] = v0;
    leaf.v_domain_global[1] = v1;
    const double corners[4][3] = {
        {u0, v0, 0.0}, {u0, v1, 0.0}, {u1, v0, 0.0}, {u1, v1, 0.0},
    };
    for (int corner = 0; corner < 4; ++corner)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            leaf.patch.control_points[corner][axis] = corners[corner][axis];
        }
    }
    return leaf;
}

LeafPatch MakeQuadLeaf(int patch_id, const std::array<double, 3> &a,
                       const std::array<double, 3> &b, const std::array<double, 3> &c,
                       const std::array<double, 3> &d)
{
    LeafPatch leaf;
    leaf.patch_id = patch_id;
    const std::array<std::array<double, 3>, 4> corners = {{a, d, b, c}};
    for (int corner = 0; corner < 4; ++corner)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            leaf.patch.control_points[corner][axis] = corners[corner][axis];
        }
    }
    return leaf;
}

void TranslateLeaf(LeafPatch &leaf, double x, double y, double z)
{
    for (int corner = 0; corner < 4; ++corner)
    {
        leaf.patch.control_points[corner][0] += x;
        leaf.patch.control_points[corner][1] += y;
        leaf.patch.control_points[corner][2] += z;
    }
}

void TestSyntheticBandAndChain()
{
    LeafPatchScene scene;
    scene.leaves = {
        MakeLeaf(7, 0.0, 0.5, 0.0, 0.5), MakeLeaf(7, 0.0, 0.5, 0.5, 1.0),
        MakeLeaf(7, 0.5, 1.0, 0.0, 0.5), MakeLeaf(7, 0.5, 1.0, 0.5, 1.0),
    };
    const auto assemblies = BuildBilinearLeafAssemblies(scene);
    Check(assemblies.size() == 1, "synthetic leaves form one assembly");
    const BilinearLeafAssembly &assembly = FindBilinearLeafAssembly(assemblies, 7);
    const BoundaryBand band = SelectBoundaryBand(assembly, BoundarySide::UMax, 1);
    Check(band.leaf_indices.size() == 2, "one U-max row selects two leaves");
    const BoundaryChain chain = BuildBoundaryChain(assembly, band);
    Check(chain.segments.size() == 2, "U-max chain has two tangent segments");
    Check(std::abs(chain.boundary_value - 1.0) < 1e-12, "U-max boundary is cached");
    Check(std::abs(chain.tangent_range.lo) < 1e-12 &&
          std::abs(chain.tangent_range.hi - 1.0) < 1e-12,
          "chain spans the complete tangent range");
    Check(chain.segments[0].leaf_index == 2 && chain.segments[1].leaf_index == 3,
          "chain is ordered by tangent parameter rather than source order assumptions");
}

void TestReversedSourceCurveMatch()
{
    NurbsBoundaryCurve a;
    a.degree = 1;
    a.knots = {0.0, 0.0, 1.0, 1.0};
    a.controls = {{{0.0, 0.0, 0.0, 1.0}, {2.0, 0.0, 0.0, 2.0}}};
    NurbsBoundaryCurve b;
    b.degree = 1;
    b.knots = a.knots;
    b.controls = {{{2.0, 0.0, 0.0, 2.0}, {0.0, 0.0, 0.0, 1.0}}};
    const BoundaryCurveMatch match = MatchNurbsBoundaryCurves(a, b);
    Check(match.matches, "reversed source curve is recognized as a shared interface");
    Check(match.b_reversed, "reversed source curve records its traversal orientation");
    Check(match.max_point_disagreement < 1e-12 && match.max_weight_disagreement < 1e-12,
          "reversed source curve has exact homogeneous control agreement");
}

void TestRationalSplitAndAverageMerge()
{
    LeafPatch rational = MakeLeaf(99, 0.0, 1.0, 0.0, 1.0);
    rational.patch.rational = true;
    rational.patch.weights[0] = 1.0;
    rational.patch.weights[1] = 1.5;
    rational.patch.weights[2] = 2.0;
    rational.patch.weights[3] = 2.5;
    const HomogeneousBilinearNet original = HomogenizeBilinearLeaf(rational);
    const auto split = SplitRationalBilinearLeaf(rational, 0, 0.5);
    const auto before = ProjectHomogeneous(EvaluateHomogeneousBilinear(original, 0.25, 0.35));
    const auto after = ProjectHomogeneous(
        EvaluateHomogeneousBilinear(HomogenizeBilinearLeaf(split[0]), 0.5, 0.35));
    const double split_error = std::sqrt((before[0] - after[0]) * (before[0] - after[0]) +
                                         (before[1] - after[1]) * (before[1] - after[1]) +
                                         (before[2] - after[2]) * (before[2] - after[2]));
    Check(split_error < 1e-14, "rational u split preserves the represented surface exactly");

    LeafPatchScene scene;
    // A has two seam spans; B has four.  The B band begins at x=1 (A ends at
    // x=1) but is translated upward, so a symmetric average must move both.
    scene.leaves = {
        MakeLeaf(10, 0.0, 1.0, 0.0, 0.5), MakeLeaf(10, 0.0, 1.0, 0.5, 1.0),
        MakeLeaf(11, 0.0, 1.0, 0.0, 0.25), MakeLeaf(11, 0.0, 1.0, 0.25, 0.5),
        MakeLeaf(11, 0.0, 1.0, 0.5, 0.75), MakeLeaf(11, 0.0, 1.0, 0.75, 1.0),
    };
    for (std::size_t i = 2; i < scene.leaves.size(); ++i)
    {
        TranslateLeaf(scene.leaves[i], 1.0, 0.0, 0.2);
    }
    const auto assemblies = BuildBilinearLeafAssemblies(scene);
    const BilinearLeafAssembly &a = FindBilinearLeafAssembly(assemblies, 10);
    const BilinearLeafAssembly &b = FindBilinearLeafAssembly(assemblies, 11);
    const AverageMergedSeam merged = AverageMergeSeam(
        a, SelectBoundaryBand(a, BoundarySide::UMax),
        b, SelectBoundaryBand(b, BoundarySide::UMin));
    Check(merged.union_parameters.size() == 5, "average merge forms the union of two and four seam spans");
    Check(std::abs(merged.max_pre_merge_disagreement - 0.2) < 1e-12,
          "average merge measures the pre-merge offset");
    Check(std::abs(merged.max_displacement_a - 0.1) < 1e-12 &&
          std::abs(merged.max_displacement_b - 0.1) < 1e-12,
          "average merge moves both sides symmetrically in homogeneous space");
    Check(merged.region_a.leaves.size() == 2 && merged.region_b.leaves.size() == 4,
          "average merge preserves source leaf counts before T-mesh baking");

    const LocalDegreeOneStrip strip = BuildLocalDegreeOneStrip(merged);
    Check(strip.faces.size() == 6, "local strip retains one explicit T-mesh face per input leaf");
    Check(strip.t_junctions.size() == 2, "mismatched seam spans produce two local T-junctions");
    Check(strip.mesh.ControlPoints().size() > 0 && strip.mesh.Edges().size() > 0,
          "local strip builds a usable T-mesh topology");
    const BakedDegreeOneStrip baked = BakeDegreeOneStrip(strip);
    const BakeVerification bake_verification = VerifyBakedDegreeOneStrip(strip, baked);
    // This fixture is affine on both sides (the B side is only translated),
    // so its two topological T-junctions do not induce a geometric crease.
    // All six original faces are therefore already exact one-primitive faces.
    Check(baked.primitives.size() == 6 && baked.n_exact_primitives == 6 &&
          baked.n_junction_primitives == 0,
          "degree-one bake keeps affine T-junction faces whole "
          "(got total=" + std::to_string(baked.primitives.size()) +
          ", exact=" + std::to_string(baked.n_exact_primitives) +
          ", junction=" + std::to_string(baked.n_junction_primitives) + ")");
    Check(bake_verification.exact && bake_verification.max_decomposition_error < 1e-10,
          "baked bilinear primitives reproduce the synthetic degree-one T-spline");
}

void TestWatertightnessCertificate()
{
    const std::array<double, 3> p000 = {0.0, 0.0, 0.0};
    const std::array<double, 3> p100 = {1.0, 0.0, 0.0};
    const std::array<double, 3> p110 = {1.0, 1.0, 0.0};
    const std::array<double, 3> p010 = {0.0, 1.0, 0.0};
    const std::array<double, 3> p001 = {0.0, 0.0, 1.0};
    const std::array<double, 3> p101 = {1.0, 0.0, 1.0};
    const std::array<double, 3> p111 = {1.0, 1.0, 1.0};
    const std::array<double, 3> p011 = {0.0, 1.0, 1.0};
    std::vector<WatertightLeaf> leaves;
    const std::array<LeafPatch, 6> cube = {{
        MakeQuadLeaf(0, p000, p100, p110, p010), MakeQuadLeaf(1, p001, p101, p111, p011),
        MakeQuadLeaf(2, p000, p100, p101, p001), MakeQuadLeaf(3, p100, p110, p111, p101),
        MakeQuadLeaf(4, p110, p010, p011, p111), MakeQuadLeaf(5, p010, p000, p001, p011),
    }};
    std::vector<SourceLeafRef> sources;
    for (std::size_t i = 0; i < cube.size(); ++i)
    {
        const SourceLeafRef source = {static_cast<int>(i), 0};
        leaves.push_back({i, cube[i], source, "cube", 0});
        sources.push_back(source);
    }
    const WatertightnessReport good = CheckShellWatertightness(leaves, sources);
    Check(good.watertight && good.open_edge_span_count == 0 && good.nonmanifold_edge_span_count == 0,
          "watertightness certificate accepts a closed six-face bilinear shell");

    std::vector<WatertightLeaf> duplicate = leaves;
    duplicate.push_back({6, cube[0], {0, 0}, "second-owner", 0});
    const WatertightnessReport bad_owner = CheckShellWatertightness(duplicate, sources);
    Check(!bad_owner.watertight && bad_owner.multiply_owned_source_leaf_count == 1,
          "watertightness certificate rejects duplicate source ownership");

    const WatertightnessReport empty = CheckShellWatertightness({}, sources);
    Check(!empty.watertight && empty.unowned_source_leaf_count == sources.size(),
          "watertightness certificate reports every missing source when no leaves were emitted");

    BakedTsplineShell compatibility_only;
    compatibility_only.corner_policy = CornerOwnershipPolicy::CompatibilityOverlap;
    compatibility_only.ownership.conflicts.push_back({{0, 0}, {}});
    compatibility_only.errors.within_requested_limit = true;
    compatibility_only.watertightness.watertight = true;
    Check(!compatibility_only.ReadyForRayTracing(),
          "compatibility overlap can never certify unresolved source ownership for RT");
    bool rt_gate_rejected_overlap = false;
    try
    {
        RequireShellReadyForRayTracing(compatibility_only);
    }
    catch (const std::runtime_error &)
    {
        rt_gate_rejected_overlap = true;
    }
    Check(rt_gate_rejected_overlap,
          "pre-RT shell gate rejects compatibility overlap even with a relaxed lower-level certificate");
}

void TestPipeCase(const std::filesystem::path &root)
{
    const auto catalog_path = root / "python_experiments/multiple_step_degree_reduction_surfaces/pipe_nurbs_border_patches.json";
    const auto leaves_path = root / "python_experiments/multiple_step_degree_reduction_surfaces/outputs/pipe_patch_all_e_0_05_hard_seams.json";
    Check(std::filesystem::is_regular_file(catalog_path), "pipe NURBS catalog fixture exists");
    Check(std::filesystem::is_regular_file(leaves_path), "pipe hard-seam leaf fixture exists");
    if (!std::filesystem::is_regular_file(catalog_path) || !std::filesystem::is_regular_file(leaves_path)) { return; }

    const SurfacePatchCatalog catalog = LoadSurfacePatchCatalogJson(catalog_path.string());
    const PatchBoundaryIndex boundary_index = BuildPatchBoundaryIndex(catalog);
    Check(boundary_index.patches.size() == 16, "pipe catalog builds 16 cached boundary entries");
    const std::vector<PatchInterface> interfaces = DiscoverPatchInterfaces(boundary_index);
    Check(interfaces.size() == 32, "pipe catalog has 32 pairwise shared interfaces");
    std::set<std::pair<int, int>> side_keys;
    for (const PatchInterface &interface : interfaces)
    {
        Check(interface.max_point_disagreement <= 1e-5,
              "discovered interface control points agree within tolerance");
        side_keys.insert({interface.patch_a, static_cast<int>(interface.side_a)});
        side_keys.insert({interface.patch_b, static_cast<int>(interface.side_b)});
    }
    Check(side_keys.size() == 64, "each pipe interface side is uniquely paired");

    const LeafPatchScene scene = LoadLeafPatchScene(leaves_path.string());
    const std::vector<BilinearLeafAssembly> assemblies = BuildBilinearLeafAssemblies(scene);
    Check(assemblies.size() == 16, "pipe leaves group into 16 source assemblies");
    std::size_t leaf_count = 0;
    for (const BilinearLeafAssembly &assembly : assemblies)
    {
        Check(!assembly.leaves.empty(), "every pipe source patch owns leaves");
        leaf_count += assembly.leaves.size();
    }
    Check(leaf_count == scene.leaves.size(),
          "assembly grouping preserves every leaf declared by the pipe fixture");
    Check(scene.max_error <= 0.05 + 1e-12,
          "pipe fixture declares the expected 0.05 reduction budget");

    std::size_t built_pipe_strips = 0;
    for (const PatchInterface &interface : interfaces)
    {
        const BilinearLeafAssembly &a = FindBilinearLeafAssembly(assemblies, interface.patch_a);
        const BilinearLeafAssembly &b = FindBilinearLeafAssembly(assemblies, interface.patch_b);
        const BoundaryChain chain_a = BuildBoundaryChain(a, SelectBoundaryBand(a, interface.side_a));
        const BoundaryChain chain_b = BuildBoundaryChain(b, SelectBoundaryBand(b, interface.side_b));
        Check(!chain_a.segments.empty() && !chain_b.segments.empty(),
              "every discovered pipe interface produces two non-empty boundary chains");
        Check(chain_a.tangent_range.hi > chain_a.tangent_range.lo &&
              chain_b.tangent_range.hi > chain_b.tangent_range.lo,
              "every pipe boundary chain has positive tangent extent");
        const AverageMergedSeam merged = AverageMergeSeam(
            a, SelectBoundaryBand(a, interface.side_a),
            b, SelectBoundaryBand(b, interface.side_b));
        const LocalDegreeOneStrip strip = BuildLocalDegreeOneStrip(merged);
        Check(strip.faces.size() == merged.region_a.leaves.size() + merged.region_b.leaves.size(),
              "a real pipe seam keeps an explicit local face for every band leaf");
        Check(strip.max_vertex_disagreement <= 1e-6,
              "a real pipe seam has consistent shared local T-mesh controls");
        Check(!strip.mesh.ControlPoints().empty() && !strip.mesh.Edges().empty(),
              "a real pipe seam builds a validated local T-mesh");
        const TSplineSurface surface(strip.mesh, 1);
        for (const LocalTSplineFace &face : strip.faces)
        {
            const auto point = surface.Evaluate(0.5 * (face.rect[0] + face.rect[1]),
                                                0.5 * (face.rect[2] + face.rect[3]));
            Check(std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]),
                  "every local pipe T-mesh face evaluates through the existing C++ surface kernel");
        }
        ++built_pipe_strips;
    }
    Check(built_pipe_strips == 32, "all 32 pipe interfaces build connected local T-mesh strips");

    ShellBuildOptions shell_options;
    shell_options.corner_policy = CornerOwnershipPolicy::CompatibilityOverlap;
    shell_options.error_validation.maximum_conservative_error = 0.05;
    const BakedTsplineShell shell = ComposeBakedTsplineShell(scene, catalog, shell_options);
    Check(shell.strips.size() == 32, "pipe shell composer builds all 32 local seam strips");
    Check(shell.ownership.raw_claim_count == 1300 &&
          shell.ownership.unique_claimed_source_leaf_count == 1188 &&
          shell.ownership.conflicts.size() == 96,
          "pipe shell composer records the known multi-interface corner ownership conflicts");
    Check(shell.errors.accounting.max_bake_decomposition_error < 1e-9,
          "all pipe baked seam strips reproduce their local T-splines to numerical precision");
    std::size_t seam_exact = 0;
    std::size_t seam_phantom = 0;
    for (const BakedShellLeaf &leaf : shell.leaves)
    {
        seam_exact += leaf.leaf.role == "seam-exact";
        seam_phantom += leaf.leaf.role == "seam-phantom";
    }
    Check(shell.leaves.size() == 10902 && seam_exact == 1134 && seam_phantom == 588,
          "pipe integration bake matches the independent reference exact/phantom leaf counts");
    Check(shell.errors.accounting.max_source_reduction_error <= 0.05 + 1e-12 &&
          shell.errors.accounting.max_bake_decomposition_error < 1e-9,
          "pipe source leaves meet the 0.05 reduction limit and seam baking is exact");
    Check(shell.errors.accounting.max_conservative_error > 0.05 &&
          !shell.errors.within_requested_limit,
          "pipe integration reports that symmetric seam motion needs error-budget headroom");
    Check(!shell.watertightness.watertight &&
          shell.watertightness.multiply_owned_source_leaf_count == 96,
          "pre-RT shell certificate blocks the Python-compatible overlapping pipe shell");

    // The direct all-patches artifact is the production input: its boundary
    // cells are partitioned once and corner cells satisfy both incident
    // average-merge constraints.  This is intentionally not compared to the
    // legacy full-strip leaf count above; it is a different, non-overlapping
    // RT construction with no phantom duplicate coverage.
    const auto direct_leaves_path = root /
        "python_experiments/multiple_step_degree_reduction_surfaces/outputs/all_patches_0_05.json";
    Check(std::filesystem::is_regular_file(direct_leaves_path),
          "direct 0.05 all-patches fixture exists");
    if (std::filesystem::is_regular_file(direct_leaves_path))
    {
        const LeafPatchScene direct_scene = LoadLeafPatchScene(direct_leaves_path.string());
        ShellBuildOptions resolved_options;
        resolved_options.error_validation.maximum_conservative_error = 0.05;
        const BakedTsplineShell resolved = ComposeBakedTsplineShell(direct_scene, catalog,
                                                                     resolved_options);
        Check(resolved.corner_policy == CornerOwnershipPolicy::ExactBoundaryCornerCollar,
              "production shell defaults to the exact boundary/corner collar");
        Check(resolved.ownership.conflicts.empty(),
              "corner partition resolves every multi-interface source claim");
        Check(resolved.watertightness.watertight &&
              resolved.watertightness.unowned_source_leaf_count == 0 &&
              resolved.watertightness.multiply_owned_source_leaf_count == 0 &&
              resolved.watertightness.open_edge_span_count == 0 &&
              resolved.watertightness.nonmanifold_edge_span_count == 0,
              "direct pipe shell is one-owner, closed, and manifold before RT");
        Check(resolved.errors.within_requested_limit && resolved.ReadyForRayTracing(),
              "direct 0.05 pipe shell passes the error and RT certification gates");
        Check(resolved.leaves.size() == 2304,
              "direct pipe collar emits the expected exact refined leaf partition");

        // Staged public API must match the Compose wrapper.
        SeamAssembly assembly = ConnectPatchLeaves(direct_scene, catalog, resolved_options);
        MultiPatchTMesh tmesh = BuildMultiPatchTMesh(std::move(assembly));
        Check(tmesh.has_unified && !tmesh.components.empty(),
              "BuildMultiPatchTMesh always materializes strip/interior components + unified chart");
        Check(std::isfinite(tmesh.EvaluateComponent(0, 0.0, 0.0)[0]),
              "non-RT EvaluateComponent works on the materialized multi-patch T-mesh");
        const RtLeafScene staged = BakeForRayTracing(tmesh);
        Check(staged.leaves.size() == resolved.leaves.size() && staged.ReadyForRayTracing(),
              "Connect → Build → Bake matches ComposeBakedTsplineShell for the direct pipe");
    }

    // The compatibility artifact is useful for regression inspection, but its
    // explicit false certificate must stop a normal RT loader path.  The
    // override is deliberately required and covers the interactive tools'
    // --allow-diagnostic-shell behavior.
    const auto diagnostic_json = std::filesystem::temp_directory_path() /
        "mfem_raytracing_pipe_tspline_diagnostic.json";
    {
        std::ofstream output(diagnostic_json);
        WriteBakedTsplineShellJson(output, shell);
    }
    const LeafPatchScene reloaded_diagnostic = LoadLeafPatchScene(diagnostic_json.string());
    Check(reloaded_diagnostic.declares_rt_certification && !reloaded_diagnostic.rt_certified &&
          reloaded_diagnostic.leaves.size() == shell.leaves.size(),
          "baked shell JSON preserves its diagnostic-only certificate and leaf payload");
    bool rejected_diagnostic = false;
    try
    {
        reloaded_diagnostic.RequireRayTracingCertified();
    }
    catch (const std::runtime_error &)
    {
        rejected_diagnostic = true;
    }
    Check(rejected_diagnostic,
          "normal RT admission rejects a serialized non-certified baked shell");
    try
    {
        reloaded_diagnostic.RequireRayTracingCertified(true);
        Check(true, "explicit diagnostic override permits regression inspection");
    }
    catch (const std::exception &)
    {
        Check(false, "explicit diagnostic override permits regression inspection");
    }
    std::error_code remove_error;
    std::filesystem::remove(diagnostic_json, remove_error);
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        TestSyntheticBandAndChain();
        TestReversedSourceCurveMatch();
        TestRationalSplitAndAverageMerge();
        TestWatertightnessCertificate();
        if (argc != 2) { throw std::invalid_argument("expected source-root argument"); }
        TestPipeCase(argv[1]);
    }
    catch (const std::exception &error)
    {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0)
    {
        std::cerr << failures << " seam workflow test(s) failed\n";
        return 1;
    }
    std::cout << "T-spline seam workflow tests passed.\n";
    return 0;
}
