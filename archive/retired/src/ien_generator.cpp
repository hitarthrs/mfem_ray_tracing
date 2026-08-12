#include "ien_generator.hpp"

#include <stdexcept>

// Builds 2D tensor-product IEN (element-to-control-point connectivity) for
// B-spline patches. Global DOF index: A = i + n*j (u index i, v index j).

namespace
{

// m = knot count, degree p => need m >= 2p + 2 for a valid open knot vector.
void ValidateKnots(const mfem::Array<double> &knots, int degree)
{
    const int m = knots.Size();
    if (degree < 1)
    {
        throw std::invalid_argument("IENGenerator: degree must be >= 1");
    }
    if (m < 2 * degree + 2)
    {
        throw std::invalid_argument("IENGenerator: knot vector too short for the given degree");
    }
}

}  // namespace

// Copy MFEM knot values into a dense array (keeps a snapshot independent of mesh).
void IENGenerator::CopyKnots(const mfem::KnotVector &kv, mfem::Array<double> &out)
{
    out.SetSize(kv.Size());
    for (int i = 0; i < kv.Size(); ++i)
    {
        out[i] = kv[i];
    }
}

// MFEM isElement(i): non-empty knot span i uses control points i, ..., i+p.
std::vector<int> IENGenerator::ActiveControlStarts(const mfem::KnotVector &kv)
{
    std::vector<int> starts;
    const int nks = kv.GetNKS();
    for (int i = 0; i < nks; ++i)
    {
        if (kv.isElement(i))
        {
            starts.push_back(i);
        }
    }
    return starts;
}

// Same rule on a raw knot array: knot[p+i] != knot[p+i+1].
std::vector<int> IENGenerator::ActiveControlStarts(const mfem::Array<double> &knots,
                                                   int p,
                                                   int ncp)
{
    std::vector<int> starts;
    const int nks = ncp - p;
    for (int i = 0; i < nks; ++i)
    {
        if (knots[p + i] != knots[p + i + 1])
        {
            starts.push_back(i);
        }
    }
    return starts;
}

IENGenerator::IENGenerator(const mfem::KnotVector &kv_u, const mfem::KnotVector &kv_v)
{
    CopyKnots(kv_u, U_knots_);
    CopyKnots(kv_v, V_knots_);
    p_ = kv_u.GetOrder();
    q_ = kv_v.GetOrder();
    n_ = kv_u.GetNCP();
    m_ = kv_v.GetNCP();
    Initialize();
}

IENGenerator::IENGenerator(const mfem::Array<double> &U_knots,
                           const mfem::Array<double> &V_knots,
                           int p_deg,
                           int q_deg)
    : U_knots_(U_knots), V_knots_(V_knots), p_(p_deg), q_(q_deg)
{
    n_ = U_knots_.Size() - p_ - 1;
    m_ = V_knots_.Size() - q_ - 1;
    Initialize();
}

IENGenerator::IENGenerator(const std::vector<double> &U_knots_in,
                           const std::vector<double> &V_knots_in,
                           int p_deg,
                           int q_deg)
    : p_(p_deg), q_(q_deg)
{
    U_knots_.SetSize(static_cast<int>(U_knots_in.size()));
    for (int i = 0; i < U_knots_.Size(); ++i)
    {
        U_knots_[i] = U_knots_in[i];
    }
    V_knots_.SetSize(static_cast<int>(V_knots_in.size()));
    for (int i = 0; i < V_knots_.Size(); ++i)
    {
        V_knots_[i] = V_knots_in[i];
    }
    n_ = U_knots_.Size() - p_ - 1;
    m_ = V_knots_.Size() - q_ - 1;
    Initialize();
}

void IENGenerator::Initialize()
{
    ValidateKnots(U_knots_, p_);
    ValidateKnots(V_knots_, q_);

    if (n_ != U_knots_.Size() - p_ - 1 || m_ != V_knots_.Size() - q_ - 1)
    {
        throw std::invalid_argument(
            "IENGenerator: control-point counts do not match knot vector sizes");
    }

    active_u_spans_ = ActiveControlStarts(U_knots_, p_, n_);
    active_v_spans_ = ActiveControlStarts(V_knots_, q_, m_);
    num_elems_u_ = static_cast<int>(active_u_spans_.size());
    num_elems_v_ = static_cast<int>(active_v_spans_.size());
}

// Loop over tensor-product knot spans; element index e = eu * n_v + ev.
std::vector<std::vector<int>> IENGenerator::Generate2DIEN() const
{
    const int ne = GetTotalElements();
    const int nn = GetLocalNodesPerElement();
    std::vector<std::vector<int>> ien(ne, std::vector<int>(nn));

    int elem = 0;
    for (int eu = 0; eu < num_elems_u_; ++eu)
    {
        const int i0 = active_u_spans_[eu];
        for (int ev = 0; ev < num_elems_v_; ++ev)
        {
            const int j0 = active_v_spans_[ev];
            int local = 0;
            // Local ordering: u (ξ) fastest, v (η) slowest.
            for (int j_loc = 0; j_loc <= q_; ++j_loc)
            {
                for (int i_loc = 0; i_loc <= p_; ++i_loc)
                {
                    const int i_global = i0 + i_loc;
                    const int j_global = j0 + j_loc;
                    ien[elem][local++] = i_global + n_ * j_global;
                }
            }
            ++elem;
        }
    }

    return ien;
}

// MFEM Table: one row per element, fixed (p+1)(q+1) DOFs per row.
void IENGenerator::FillElementToDofTable(mfem::Table &el_to_dof) const
{
    const int ne = GetTotalElements();
    const int nn = GetLocalNodesPerElement();

    el_to_dof.Clear();
    el_to_dof.SetSize(ne, nn);

    const auto ien = Generate2DIEN();
    for (int e = 0; e < ne; ++e)
    {
        for (int a = 0; a < nn; ++a)
        {
            el_to_dof.Push(e, ien[e][a]);
        }
    }
    el_to_dof.Finalize();
}

mfem::Table IENGenerator::MakeElementToDofTable() const
{
    mfem::Table table;
    FillElementToDofTable(table);
    return table;
}

int IENGenerator::GetGlobalDof(int elem, int local_node) const
{
    const auto ien = Generate2DIEN();
    if (elem < 0 || elem >= GetTotalElements())
    {
        throw std::out_of_range("IENGenerator::GetGlobalDof: invalid element index");
    }
    if (local_node < 0 || local_node >= GetLocalNodesPerElement())
    {
        throw std::out_of_range("IENGenerator::GetGlobalDof: invalid local node index");
    }
    return ien[elem][local_node];
}

// Parametric directions 0 and 1 of a NURBS patch (ξ, η).
IENGenerator IENGeneratorFromPatch(const mfem::Mesh &mesh, int patch)
{
    if (mesh.NURBSext == nullptr)
    {
        throw std::invalid_argument("IENGeneratorFromPatch: mesh has no NURBS extension");
    }
    if (patch < 0 || patch >= mesh.NURBSext->GetNP())
    {
        throw std::out_of_range("IENGeneratorFromPatch: invalid patch index");
    }

    mfem::Array<const mfem::KnotVector *> kv;
    mesh.NURBSext->GetPatchKnotVectors(patch, kv);
    if (kv.Size() < 2)
    {
        throw std::invalid_argument(
            "IENGeneratorFromPatch: patch must have at least two parametric directions");
    }

    return IENGenerator(*kv[0], *kv[1]);
}
