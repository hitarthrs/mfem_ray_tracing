#include "mfem_raytracing/tspline/tspline_shell_composer.hpp"

#include "mfem_raytracing/pipeline/bake.hpp"
#include "mfem_raytracing/pipeline/connect.hpp"
#include "mfem_raytracing/pipeline/multi_patch_tmesh.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

namespace mfem_raytracing
{
namespace tspline
{
namespace
{

void AddClaims(SeamOwnershipPlan &plan, const PatchInterface &interface, const BoundaryBand &band,
               StripRegion region)
{
    const std::string id = MakeInterfaceId(interface);
    for (const std::size_t leaf_index : band.leaf_indices)
    {
        const SourceLeafRef source = {band.patch_id, leaf_index};
        plan.claims_by_source[source].push_back({source, id, region});
        ++plan.raw_claim_count;
    }
}

double DisplacementForFace(const ShellStripResult &strip, const BakedTSplinePrimitive &primitive)
{
    if (primitive.source_face_index >= strip.strip.faces.size())
    {
        throw std::out_of_range("baked primitive references an invalid local T-spline face");
    }
    return strip.strip.faces[primitive.source_face_index].region == StripRegion::A ?
        strip.merge.max_displacement_a : strip.merge.max_displacement_b;
}

std::string SourceOwnerId(const SourceLeafRef &source)
{
    return "source:p" + std::to_string(source.patch_id) + ":l" +
           std::to_string(source.leaf_index);
}

BoundaryMergeConstraint MakeConstraint(const ShellStripResult &strip, bool side_a)
{
    BoundaryMergeConstraint constraint;
    constraint.interface_id = strip.interface_id;
    constraint.patch_id = side_a ? strip.interface.patch_a : strip.interface.patch_b;
    constraint.side = side_a ? strip.interface.side_a : strip.interface.side_b;
    constraint.chain = side_a ? strip.merge.chain_a : strip.merge.chain_b;
    constraint.reversed = side_a ? false : strip.merge.b_reversed;
    constraint.union_parameters = strip.merge.union_parameters;
    constraint.merged_controls = strip.merge.merged_controls;
    constraint.seam_displacement = side_a ? strip.merge.max_displacement_a :
                                            strip.merge.max_displacement_b;
    return constraint;
}

} // namespace

bool BakedTsplineShell::ReadyForRayTracing() const
{
    // CompatibilityOverlap is a diagnostic/regression construction mode.  It
    // must never turn unresolved corner overlap into a certifiable RT shell,
    // even if a caller relaxes the lower-level ownership checker.
    return ownership.conflicts.empty() && errors.within_requested_limit &&
           watertightness.watertight;
}

std::vector<LeafPatch> BakedTsplineShell::RuntimeLeaves() const
{
    std::vector<LeafPatch> result;
    result.reserve(leaves.size());
    for (const BakedShellLeaf &entry : leaves) { result.push_back(entry.leaf); }
    return result;
}

std::string MakeInterfaceId(const PatchInterface &interface)
{
    std::ostringstream result;
    result << "p" << interface.patch_a << ':' << BoundarySideName(interface.side_a)
           << "<->p" << interface.patch_b << ':' << BoundarySideName(interface.side_b);
    return result.str();
}

SeamOwnershipPlan MakeSeamOwnershipPlan(
    const std::vector<std::pair<PatchInterface, std::pair<BoundaryBand, BoundaryBand>>> &bands)
{
    SeamOwnershipPlan result;
    for (const auto &entry : bands)
    {
        AddClaims(result, entry.first, entry.second.first, StripRegion::A);
        AddClaims(result, entry.first, entry.second.second, StripRegion::B);
    }
    result.unique_claimed_source_leaf_count = result.claims_by_source.size();
    for (const auto &[source, claims] : result.claims_by_source)
    {
        std::set<std::string> owners;
        for (const SeamClaim &claim : claims) { owners.insert(claim.interface_id); }
        if (owners.size() > 1) { result.conflicts.push_back({source, claims}); }
    }
    return result;
}

SeamAssembly ConnectPatchLeaves(const LeafPatchScene &input,
                                const SurfacePatchCatalog &catalog,
                                const ShellBuildOptions &options)
{
    if (options.seam_band_rows == 0)
    {
        throw std::invalid_argument("T-spline shell needs at least one seam-band row");
    }

    SeamAssembly assembly;
    assembly.surface_name = input.surface_name;
    assembly.options = options;
    assembly.assemblies = BuildBilinearLeafAssemblies(input);
    assembly.interfaces = DiscoverPatchInterfaces(catalog, options.interface_discovery);

    for (const BilinearLeafAssembly &patch_assembly : assembly.assemblies)
    {
        for (std::size_t i = 0; i < patch_assembly.leaves.size(); ++i)
        {
            assembly.expected_sources.push_back({patch_assembly.patch_id, i});
        }
    }

    assembly.bands.reserve(assembly.interfaces.size());
    for (const PatchInterface &interface : assembly.interfaces)
    {
        const BilinearLeafAssembly &a =
            FindBilinearLeafAssembly(assembly.assemblies, interface.patch_a);
        const BilinearLeafAssembly &b =
            FindBilinearLeafAssembly(assembly.assemblies, interface.patch_b);
        assembly.bands.push_back(
            {interface,
             {SelectBoundaryBand(a, interface.side_a, options.seam_band_rows),
              SelectBoundaryBand(b, interface.side_b, options.seam_band_rows)}});
    }
    assembly.ownership = MakeSeamOwnershipPlan(assembly.bands);

    assembly.strips.reserve(assembly.bands.size());
    for (const auto &entry : assembly.bands)
    {
        const PatchInterface &interface = entry.first;
        const BoundaryBand &band_a = entry.second.first;
        const BoundaryBand &band_b = entry.second.second;
        const BilinearLeafAssembly &assembly_a =
            FindBilinearLeafAssembly(assembly.assemblies, interface.patch_a);
        const BilinearLeafAssembly &assembly_b =
            FindBilinearLeafAssembly(assembly.assemblies, interface.patch_b);
        ShellStripResult strip;
        strip.interface = interface;
        strip.interface_id = MakeInterfaceId(interface);
        strip.merge =
            AverageMergeSeam(assembly_a, band_a, assembly_b, band_b, options.average_merge);
        assembly.strips.push_back(std::move(strip));
    }
    return assembly;
}

RtLeafScene BakeForRayTracing(const MultiPatchTMesh &mesh)
{
    const SeamAssembly &assembly = mesh.assembly;
    const ShellBuildOptions &options = assembly.options;

    RtLeafScene result;
    result.surface_name = assembly.surface_name;
    result.corner_policy = options.corner_policy;
    result.ownership = assembly.ownership;
    result.strips = assembly.strips;

    // Ensure strip T-meshes / bakes exist for non-collar policies (Build fills
    // strip meshes; bake primitives may still be empty until here).
    for (ShellStripResult &strip : result.strips)
    {
        if (strip.strip.faces.empty())
        {
            strip.strip = BuildLocalDegreeOneStrip(strip.merge, options.strip_build);
        }
        if (options.corner_policy == CornerOwnershipPolicy::CompatibilityOverlap ||
            options.corner_policy == CornerOwnershipPolicy::RejectAmbiguous)
        {
            strip.baked = BakeDegreeOneStrip(strip.strip, options.bake);
            strip.verification = VerifyBakedDegreeOneStrip(strip.strip, strip.baked, options.bake);
        }
    }

    std::vector<LeafErrorBreakdown> error_terms;
    std::size_t final_index = 0;

    if (options.corner_policy == CornerOwnershipPolicy::ExactBoundaryCornerCollar)
    {
        std::vector<BoundaryMergeConstraint> constraints;
        constraints.reserve(2 * result.strips.size());
        for (const ShellStripResult &strip : result.strips)
        {
            constraints.push_back(MakeConstraint(strip, true));
            constraints.push_back(MakeConstraint(strip, false));
        }
        const CornerCollarOptions collar_options = {options.average_merge.parameter_tolerance,
                                                    options.average_merge.geometry_tolerance};
        const std::vector<CornerCollarLeaf> collar = BuildExactBoundaryCornerCollar(
            assembly.assemblies, constraints, collar_options);
        for (const CornerCollarLeaf &entry : collar)
        {
            BakedShellLeaf output;
            output.leaf = entry.leaf;
            output.leaf.index = static_cast<int>(final_index++);
            output.source = entry.source;
            output.owner_id = SourceOwnerId(entry.source);
            output.leaf.interface_id = entry.leaf.interface_id;
            output.error.source_reduction_error = entry.leaf.total_error;
            output.error.seam_displacement = entry.seam_displacement;
            output.leaf.total_error = output.error.ConservativeBound();
            error_terms.push_back(output.error);
            result.leaves.push_back(std::move(output));
        }
        result.ownership.conflicts.clear();
    }
    else
    {
        std::set<SourceLeafRef> claimed;
        for (const auto &[source, claims] : result.ownership.claims_by_source)
        {
            (void)claims;
            claimed.insert(source);
        }
        for (const BilinearLeafAssembly &patch_assembly : assembly.assemblies)
        {
            for (std::size_t leaf_index = 0; leaf_index < patch_assembly.leaves.size(); ++leaf_index)
            {
                const SourceLeafRef source = {patch_assembly.patch_id, leaf_index};
                if (claimed.count(source) != 0) { continue; }
                BakedShellLeaf output;
                output.leaf = patch_assembly.leaves[leaf_index];
                output.leaf.index = static_cast<int>(final_index++);
                output.leaf.role = "interior";
                output.leaf.kind = "interior";
                output.source = source;
                output.owner_id = "interior:p" + std::to_string(source.patch_id);
                output.error.source_reduction_error = output.leaf.total_error;
                output.leaf.total_error = output.error.ConservativeBound();
                error_terms.push_back(output.error);
                result.leaves.push_back(std::move(output));
            }
        }
        for (const ShellStripResult &strip : result.strips)
        {
            for (const BakedTSplinePrimitive &primitive : strip.baked.primitives)
            {
                BakedShellLeaf output;
                output.leaf = primitive.leaf;
                output.leaf.index = static_cast<int>(final_index++);
                output.leaf.interface_id = strip.interface_id;
                output.source = {primitive.source_patch_id, primitive.source_leaf_index};
                output.owner_id = strip.interface_id;
                output.strip_face_index = primitive.source_face_index;
                output.local_subrect = primitive.rect;
                output.error.source_reduction_error = primitive.source_reduction_error;
                output.error.seam_displacement = DisplacementForFace(strip, primitive);
                output.error.bake_decomposition_error = strip.verification.max_decomposition_error;
                output.leaf.total_error = output.error.ConservativeBound();
                error_terms.push_back(output.error);
                result.leaves.push_back(std::move(output));
            }
        }
    }

    result.errors = ValidateErrorBounds(error_terms, options.error_validation);
    std::vector<WatertightLeaf> watertight_leaves;
    watertight_leaves.reserve(result.leaves.size());
    for (const BakedShellLeaf &entry : result.leaves)
    {
        watertight_leaves.push_back({static_cast<std::size_t>(entry.leaf.index), entry.leaf,
                                     entry.source, entry.owner_id, entry.strip_face_index});
    }
    result.watertightness = CheckShellWatertightness(watertight_leaves, assembly.expected_sources,
                                                     options.watertightness);
    return result;
}

BakedTsplineShell ComposeBakedTsplineShell(const LeafPatchScene &input,
                                           const SurfacePatchCatalog &catalog,
                                           const ShellBuildOptions &options)
{
    // Deprecated thin wrapper: Connect → Build → Bake.
    SeamAssembly assembly = ConnectPatchLeaves(input, catalog, options);
    MultiPatchTMesh tmesh = BuildMultiPatchTMesh(std::move(assembly));
    return BakeForRayTracing(tmesh);
}

void RequireShellReadyForRayTracing(const BakedTsplineShell &shell)
{
    if (!shell.ownership.conflicts.empty())
    {
        throw std::runtime_error("T-spline shell has unresolved multi-interface corner ownership conflicts");
    }
    if (!shell.errors.within_requested_limit)
    {
        throw std::runtime_error("T-spline shell exceeds its requested conservative error bound");
    }
    RequireWatertightForRayTracing(shell.watertightness);
}

} // namespace tspline
} // namespace mfem_raytracing
