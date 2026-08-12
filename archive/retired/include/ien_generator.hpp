#ifndef IEN_GENERATOR_HPP
#define IEN_GENERATOR_HPP

#include "mfem.hpp"

#include <vector>

/**
 * @brief 2D tensor-product B-spline element-to-control-point connectivity (IEN).
 *
 * For a patch with n control points in the u (ξ) direction and m in the v (η)
 * direction, element e and local index a map to a global control-point index
 *   A = i + n * j
 * with (i, j) the global parametric indices of the local node.
 *
 * MFEM-oriented outputs use mfem::Table (element → DOF list), matching the
 * layout used by finite element spaces.
 */
class IENGenerator
{
public:
    /** @brief Build from MFEM knot vectors (degree = kv.GetOrder(), n = kv.GetNCP()). */
    IENGenerator(const mfem::KnotVector &kv_u, const mfem::KnotVector &kv_v);

    /** @brief Build from knot arrays and explicit polynomial degrees. */
    IENGenerator(const mfem::Array<double> &U_knots,
                 const mfem::Array<double> &V_knots,
                 int p_deg,
                 int q_deg);

    /** @brief Build from std::vector knots (copied into internal mfem::Array). */
    IENGenerator(const std::vector<double> &U_knots_in,
                 const std::vector<double> &V_knots_in,
                 int p_deg,
                 int q_deg);

    /**
     * @brief Element-to-DOF connectivity as an MFEM Table (Finalize() called).
     *
     * Row @a e lists global control-point indices for local nodes on element @a e.
     * Local node ordering: u (ξ) index varies fastest, v (η) slowest.
     */
    mfem::Table MakeElementToDofTable() const;

    /** @brief Fill an existing Table (same layout as MakeElementToDofTable). */
    void FillElementToDofTable(mfem::Table &el_to_dof) const;

    /** @brief Dense 2D IEN[e][a] for debugging and non-MFEM callers. */
    std::vector<std::vector<int>> Generate2DIEN() const;

    /** @brief Global control-point index for (element, local_node). */
    int GetGlobalDof(int elem, int local_node) const;

    int GetDegreeU() const { return p_; }
    int GetDegreeV() const { return q_; }
    int GetNumControlPointsU() const { return n_; }
    int GetNumControlPointsV() const { return m_; }
    int GetNumElementsU() const { return num_elems_u_; }
    int GetNumElementsV() const { return num_elems_v_; }
    int GetTotalElements() const { return num_elems_u_ * num_elems_v_; }
    int GetLocalNodesPerElement() const { return (p_ + 1) * (q_ + 1); }

private:
    mfem::Array<double> U_knots_;
    mfem::Array<double> V_knots_;
    int p_ = 0;
    int q_ = 0;
    int n_ = 0;
    int m_ = 0;
    int num_elems_u_ = 0;
    int num_elems_v_ = 0;

    /// Starting u/v control-point index for each active knot span.
    std::vector<int> active_u_spans_;
    std::vector<int> active_v_spans_;

    void Initialize();
    static void CopyKnots(const mfem::KnotVector &kv, mfem::Array<double> &out);
    static std::vector<int> ActiveControlStarts(const mfem::Array<double> &knots, int p, int ncp);
    static std::vector<int> ActiveControlStarts(const mfem::KnotVector &kv);
};

/**
 * @brief Build an IEN generator from NURBS patch parametric directions 0 and 1.
 *
 * Requires mesh.NURBSext and a 2D or 3D patch (uses ξ and η knot vectors only).
 */
IENGenerator IENGeneratorFromPatch(const mfem::Mesh &mesh, int patch);

#endif
