#include "tspline_error_accounting.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mfem_raytracing
{
namespace tspline
{

double LeafErrorBreakdown::ConservativeBound() const
{
    if (source_reduction_error < 0.0 || seam_displacement < 0.0 ||
        bake_decomposition_error < 0.0)
    {
        throw std::invalid_argument("T-spline error terms must be non-negative");
    }
    return source_reduction_error + seam_displacement + bake_decomposition_error;
}

ErrorAccountingReport SummarizeErrors(const std::vector<LeafErrorBreakdown> &errors)
{
    ErrorAccountingReport result;
    result.leaf_count = errors.size();
    for (const LeafErrorBreakdown &error : errors)
    {
        result.max_source_reduction_error = std::max(result.max_source_reduction_error,
                                                     error.source_reduction_error);
        result.max_seam_displacement = std::max(result.max_seam_displacement,
                                                error.seam_displacement);
        result.max_bake_decomposition_error = std::max(result.max_bake_decomposition_error,
                                                       error.bake_decomposition_error);
        result.max_conservative_error = std::max(result.max_conservative_error,
                                                 error.ConservativeBound());
    }
    return result;
}

ErrorValidationReport ValidateErrorBounds(const std::vector<LeafErrorBreakdown> &errors,
                                          const ErrorValidationOptions &options)
{
    if (options.maximum_conservative_error < 0.0)
    {
        throw std::invalid_argument("requested T-spline error limit must be non-negative");
    }
    ErrorValidationReport result;
    result.accounting = SummarizeErrors(errors);
    result.within_requested_limit =
        result.accounting.max_conservative_error <= options.maximum_conservative_error;
    return result;
}

} // namespace tspline
} // namespace mfem_raytracing
