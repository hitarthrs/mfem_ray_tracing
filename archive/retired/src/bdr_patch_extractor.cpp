#include "bdr_patch_extractor.hpp"
#include "kronecker_product.hpp"

#include <algorithm>
#include <stdexcept>

namespace
{

std::vector<double> KnotsFromKV(const mfem::KnotVector &kv)
{
    std::vector<double> knots(kv.Size());
    for (int i = 0; i < kv.Size(); ++i)
    {
        knots[i] = kv[i];
    }
    return knots;
}

void FillParametric1D(BdrPatchParametric1D &param, const mfem::KnotVector *kv)
{
    param.knots = KnotsFromKV(*kv);
    param.degree = kv->GetOrder();
    param.ne = kv->GetNE();
    param.extractors =
        ElementExtractorsPerSpan(param.knots, param.degree, param.ne);
}

/// Decode MFEM bel_to_IJK entry to knot-span start index.
int KnotSpanStartFromIJK(int ijk_entry)
{
    return (ijk_entry >= 0) ? ijk_entry : (-1 - ijk_entry);
}

/// Map knot-span start index to compact extractor index 0 .. ne-1.
int KnotSpanToCompactIndex(const mfem::KnotVector &kv, int knot_span_start)
{
    int compact = 0;
    const int nks = kv.GetNKS();
    for (int i = 0; i < nks; ++i)
    {
        if (!kv.isElement(i))
        {
            continue;
        }
        if (i == knot_span_start)
        {
            return compact;
        }
        ++compact;
    }
    throw std::invalid_argument(
        "BdrPatchExtractor: knot-span index not active in knot vector");
}

mfem::DenseMatrix BuildElementExtractionMatrix(const BdrPatchParametric1D &param_u,
                                               const BdrPatchParametric1D &param_v,
                                               int span_u,
                                               int span_v)
{
    if (span_u < 0 || span_u >= param_u.ne || span_v < 0 || span_v >= param_v.ne)
    {
        throw std::invalid_argument(
            "BdrPatchExtractor: span index out of range for extraction operator");
    }
    if (static_cast<int>(param_u.extractors.size()) != param_u.ne ||
        static_cast<int>(param_v.extractors.size()) != param_v.ne)
    {
        throw std::invalid_argument(
            "BdrPatchExtractor: expected one extractor per knot span in each direction");
    }

    const mfem::DenseMatrix &C_u = param_u.extractors[static_cast<std::size_t>(span_u)].matrix;
    const mfem::DenseMatrix &C_v = param_v.extractors[static_cast<std::size_t>(span_v)].matrix;

    // DOF layout ii + nu*jj (u fastest) matches KroneckerProduct(C_v, C_u).
    return KroneckerProduct(C_v, C_u);
}

struct BdrElementRef
{
    int be = -1;
    int ijk[2] = {0, 0};
    int span_u = 0;
    int span_v = 0;
};

bool IJKLess(const BdrElementRef &a, const BdrElementRef &b)
{
    if (a.ijk[0] != b.ijk[0])
    {
        return a.ijk[0] < b.ijk[0];
    }
    return a.ijk[1] < b.ijk[1];
}

void FillElementBlock(BdrPatchElementBlock &block,
                      const mfem::Array<int> &dofs,
                      const mfem::Vector &nodes,
                      const mfem::Vector &weights,
                      int nu,
                      int nv,
                      int sdim)
{
    const int ncp = nu * nv;
    MFEM_VERIFY(dofs.Size() == ncp,
                "BdrPatchExtractor: DOF count does not match tensor-product size");

    block.P.SetSize(ncp, sdim);

    mfem::Vector w_spline(ncp);
    w_spline = 0.0;
    for (int jj = 0; jj < nv; ++jj)
    {
        for (int ii = 0; ii < nu; ++ii)
        {
            const int a = ii + nu * jj;
            const int g = dofs[a];
            for (int d = 0; d < sdim; ++d)
            {
                block.P(a, d) = nodes(g * sdim + d);
            }
            w_spline(a) = weights(g);
        }
    }

    block.W = mfem::SparseMatrix(w_spline);
}

/// w_b = C_e^T w with w the per-DOF NURBS weights (diagonal of W).
void FillBezierWeights(BdrPatchElementBlock &block)
{
    const int n = block.W.Height();
    MFEM_VERIFY(block.C_e.Height() == n && block.C_e.Width() == n,
                "BdrPatchExtractor: C_e and weight vector size mismatch");

    mfem::Vector w_spline(n);
    w_spline = 0.0;
    for (int a = 0; a < n; ++a)
    {
        w_spline(a) = block.W(a, a);
    }

    block.w_b_e.SetSize(n);
    block.w_b_e = 0.0;
    block.C_e.MultTranspose(w_spline, block.w_b_e);

    block.W_b = mfem::SparseMatrix(block.w_b_e);
}

/// Q_e = (W_b)^{-1} C_e^T W P  (localize rational NURBS CPs to Bézier element).
void FillBezierControlPoints(BdrPatchElementBlock &block)
{
    const int n = block.P.Height();
    const int sdim = block.P.Width();
    MFEM_VERIFY(block.C_e.Height() == n && block.C_e.Width() == n,
                "BdrPatchExtractor: C_e and P row count mismatch");
    MFEM_VERIFY(block.W.Height() == n && block.w_b_e.Size() == n,
                "BdrPatchExtractor: weight size mismatch");

    mfem::DenseMatrix wp(n, sdim);
    for (int a = 0; a < n; ++a)
    {
        const double wa = block.W(a, a);
        for (int d = 0; d < sdim; ++d)
        {
            wp(a, d) = wa * block.P(a, d);
        }
    }

    mfem::DenseMatrix ct_wp(n, sdim);
    mfem::MultAtB(block.C_e, wp, ct_wp);

    block.Q_e.SetSize(n, sdim);
    for (int a = 0; a < n; ++a)
    {
        const double wb = block.w_b_e(a);
        MFEM_VERIFY(wb != 0.0, "BdrPatchExtractor: zero Bézier weight in W_b");
        const double inv_wb = 1.0 / wb;
        for (int d = 0; d < sdim; ++d)
        {
            block.Q_e(a, d) = ct_wp(a, d) * inv_wb;
        }
    }
}

}  // namespace

BdrPatchBlockData ExtractBdrPatchBlocks(const mfem::Mesh &mesh, int nurbs_bdr_patch)
{
    if (mesh.NURBSext == nullptr)
    {
        throw std::invalid_argument(
            "ExtractBdrPatchBlocks: mesh has no NURBSExtension");
    }
    if (mesh.GetNodes() == nullptr)
    {
        throw std::invalid_argument(
            "ExtractBdrPatchBlocks: mesh has no nodal GridFunction");
    }

    mfem::Table *bel_dof = mesh.NURBSext->GetBdrElementDofTable();
    if (bel_dof == nullptr)
    {
        throw std::invalid_argument(
            "ExtractBdrPatchBlocks: no boundary element DOF table");
    }

    const mfem::FiniteElementSpace &fes = *mesh.GetNodes()->FESpace();
    std::vector<BdrElementRef> refs;

    for (int be = 0; be < mesh.GetNBE(); ++be)
    {
        const mfem::FiniteElement *fe = fes.GetBE(be);
        const auto *nfe = dynamic_cast<const mfem::NURBSFiniteElement *>(fe);
        if (nfe == nullptr || nfe->GetPatch() != nurbs_bdr_patch)
        {
            continue;
        }

        BdrElementRef ref;
        ref.be = be;
        if (const int *ijk = nfe->GetIJK())
        {
            ref.ijk[0] = ijk[0];
            ref.ijk[1] = ijk[1];
        }
        refs.push_back(ref);
    }

    if (refs.empty())
    {
        throw std::invalid_argument(
            "ExtractBdrPatchBlocks: no boundary elements for patch");
    }

    std::sort(refs.begin(), refs.end(), IJKLess);

    const mfem::FiniteElement *fe0 = fes.GetBE(refs.front().be);
    const auto *nfe0 = dynamic_cast<const mfem::NURBSFiniteElement *>(fe0);
    MFEM_VERIFY(nfe0 != nullptr, "");

    const mfem::Array<const mfem::KnotVector *> &kv = nfe0->KnotVectors();
    MFEM_VERIFY(kv.Size() == 2, "BdrPatchExtractor: expected 2D boundary patch");

    BdrPatchBlockData data;
    data.nu = kv[0]->GetOrder() + 1;
    data.nv = kv[1]->GetOrder() + 1;
    data.ne_u = kv[0]->GetNE();
    data.ne_v = kv[1]->GetNE();

    FillParametric1D(data.param_u, kv[0]);
    FillParametric1D(data.param_v, kv[1]);

    for (BdrElementRef &ref : refs)
    {
        const int start_u = KnotSpanStartFromIJK(ref.ijk[0]);
        const int start_v = KnotSpanStartFromIJK(ref.ijk[1]);
        ref.span_u = KnotSpanToCompactIndex(*kv[0], start_u);
        ref.span_v = KnotSpanToCompactIndex(*kv[1], start_v);
    }

    const int expected_ne = data.ne_u * data.ne_v;
    if (static_cast<int>(refs.size()) != expected_ne)
    {
        throw std::invalid_argument(
            "ExtractBdrPatchBlocks: patch element count does not match ne_u*ne_v");
    }

    data.elements.reserve(static_cast<size_t>(expected_ne));

    mfem::Vector nodes;
    mesh.GetNodes(nodes);
    const int sdim = mesh.SpaceDimension();
    const mfem::Vector &w = mesh.NURBSext->GetWeights();

    for (const BdrElementRef &ref : refs)
    {
        mfem::Array<int> dofs;
        bel_dof->GetRow(ref.be, dofs);

        BdrPatchElementBlock block;
        block.span_u = ref.span_u;
        block.span_v = ref.span_v;
        block.C_e = BuildElementExtractionMatrix(
            data.param_u, data.param_v, ref.span_u, ref.span_v);
        FillElementBlock(block, dofs, nodes, w, data.nu, data.nv, sdim);
        FillBezierWeights(block);
        FillBezierControlPoints(block);
        data.elements.push_back(block);
    }

    return data;
}
