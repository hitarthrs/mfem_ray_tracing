#ifndef BDR_PATCH_EXTRACTOR_HPP
#define BDR_PATCH_EXTRACTOR_HPP

#include "element_extractor.hpp"

#include "mfem.hpp"

#include <vector>

/**
 * @brief Per-boundary-element NURBS data on one patch face.
 *
 * P is (nu*nv) x sdim with rows in tensor order (u fastest, v slowest).
 * W is (nu*nv) x (nu*nv) sparse diagonal Diag(w) with NURBS weights w.
 * w_b_e is (nu*nv): Bézier weights w_b = C_e^T w.
 * W_b is (nu*nv) x (nu*nv) sparse diagonal Diag(w_b_e).
 * Q_e is (nu*nv) x sdim Bézier control points: Q_e = (W_b)^{-1} C_e^T W P (eq. 79).
 */
struct BdrPatchElementBlock
{
    /// Compact knot-span indices (0 .. ne_u-1, 0 .. ne_v-1) for extractor lookup.
    int span_u = 0;
    int span_v = 0;
    /// Bézier extraction on this face element: C_e = C_v ⊗ C_u (v slow, u fast).
    mfem::DenseMatrix C_e;
    /// Bézier weights: w_b_e = C_e^T w (same length and ordering as diagonal of W).
    mfem::Vector w_b_e;
    /// NURBS control points (homogeneous numerator layout before rational divide).
    mfem::DenseMatrix P;
    /// Bézier control points Q_e = (W_b)^{-1} C_e^T W P.
    mfem::DenseMatrix Q_e;
    /// W = Diag(w), W_b = Diag(w_b_e) via mfem::SparseMatrix(const Vector&).
    mfem::SparseMatrix W;
    mfem::SparseMatrix W_b;
};

/**
 * @brief One parametric direction: knot vector and 1D Bézier extraction operators.
 *
 * @a ne is MFEM KnotVector::GetNE() (active knot spans).
 * @a extractors.size() == @a ne (one C_e per knot span via ElementExtractorsPerSpan).
 */
struct BdrPatchParametric1D
{
    std::vector<double> knots;
    int degree = 0;
    int ne = 0;
    std::vector<ElementExtractor> extractors;
};

/**
 * @brief Control points and weights for all knot-span blocks on a NURBS boundary patch.
 *
 * elements.size() == ne_u * ne_v, sorted by IJK (first parametric direction, then second).
 * param_u / param_v hold surface knot vectors and univariate extractors (one per knot span).
 * elements[e].C_e = KroneckerProduct(C_v[span_v], C_u[span_u]) with DOF order u fastest, v slowest.
 */
struct BdrPatchBlockData
{
    int nu = 0;
    int nv = 0;
    int ne_u = 0;
    int ne_v = 0;
    BdrPatchParametric1D param_u;
    BdrPatchParametric1D param_v;
    std::vector<BdrPatchElementBlock> elements;
};

/**
 * @brief Extract per-element control-point and weight matrices for a NURBS boundary patch.
 *
 * @param mesh NURBS mesh with nodal GridFunction.
 * @param nurbs_bdr_patch Index from NURBSFiniteElement::GetPatch() (not volume patch).
 * @return Per element: P, Q_e (ncp x sdim); W, W_b diagonal sparse (ncp x ncp).
 * @throws std::invalid_argument if the mesh is not NURBS or the patch has no elements.
 */
BdrPatchBlockData ExtractBdrPatchBlocks(const mfem::Mesh &mesh, int nurbs_bdr_patch);

#endif
