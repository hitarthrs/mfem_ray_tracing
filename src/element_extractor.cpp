#include "element_extractor.hpp"

#include <stdexcept>
#include <vector>

namespace
{

void EnsureOperator(std::vector<ElementExtractor> &ops, int index, int p)
{
    if (static_cast<int>(ops.size()) <= index)
    {
        ops.resize(index + 1);
    }

    mfem::IdentityMatrixCoefficient identity(p + 1);
    mfem::IsoparametricTransformation transform;
    mfem::IntegrationPoint ip;
    identity.Eval(ops[index].matrix, transform, ip);
}

// C(:, k) = alpha * C(:, k) + (1 - alpha) * C(:, k - 1)
void UpdateColumn(mfem::DenseMatrix &C, int k, double alpha, int p)
{
    for (int row = 0; row <= p; ++row)
    {
        const double val_k = C(row, k);
        const double val_km1 = C(row, k - 1);
        C(row, k) = alpha * val_k + (1.0 - alpha) * val_km1;
    }
}

// C_next(save : save + j, save) = C_curr(p - j : p, p)
void CopyColumnSegment(mfem::DenseMatrix &C_next,
                       const mfem::DenseMatrix &C_curr,
                       int save,
                       int j,
                       int p)
{
    for (int row = 0; row <= j; ++row)
    {
        C_next(save + row, save) = C_curr(p - j + row, p);
    }
}

}  // namespace

std::vector<ElementExtractor> ElementExtractors(const std::vector<double> &knot_vector, int degree)
{
    if (knot_vector.size() < 2)
    {
        throw std::invalid_argument("Knot vector must have at least 2 knots");
    }
    if (degree < 1)
    {
        throw std::invalid_argument("Degree must be at least 1");
    }

    const int p = degree;
    const int m = static_cast<int>(knot_vector.size());
    if (m < 2 * p + 2)
    {
        throw std::invalid_argument("Knot vector too short for the given degree");
    }

    int a = p;
    int b = a + 1;
    int nb = 1;

    std::vector<ElementExtractor> element_extractors;
    EnsureOperator(element_extractors, 0, p);

    while (b < m)
    {
        EnsureOperator(element_extractors, nb, p);

        const int i = b;

        // Count multiplicity of knot at location b
        while (b + 1 < m && knot_vector[b + 1] == knot_vector[b])
        {
            ++b;
        }
        const int mult = b - i + 1;

        if (mult < p)
        {
            const double numer = knot_vector[b] - knot_vector[a];
            std::vector<double> alphas(p - mult + 1, 0.0);
            for (int j = p; j >= mult + 1; --j)
            {
                alphas[j - mult] = numer / (knot_vector[a + j] - knot_vector[a]);
            }

            const int r = p - mult;
            mfem::DenseMatrix &C_curr = element_extractors[nb - 1].matrix;
            mfem::DenseMatrix &C_next = element_extractors[nb].matrix;

            for (int j = 1; j <= r; ++j)
            {
                const int save = r - j;
                const int s = mult + j;
                for (int k = p; k >= s; --k)
                {
                    const double alpha = alphas[k + 1 - s];
                    UpdateColumn(C_curr, k, alpha, p);
                }
                if (b + 1 < m)
                {
                    CopyColumnSegment(C_next, C_curr, save, j, p);
                }
            }
        }

        ++nb;
        if (b + 1 < m)
        {
            a = b;
            b = b + 1;
        }
        else
        {
            b = m;
        }
    }

    element_extractors.resize(nb);
    return element_extractors;
}
