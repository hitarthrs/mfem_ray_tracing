#ifndef MFEM_RAYTRACING_TSPLINE_ERROR_ACCOUNTING_HPP
#define MFEM_RAYTRACING_TSPLINE_ERROR_ACCOUNTING_HPP

#include <cstddef>
#include <limits>
#include <vector>

namespace mfem_raytracing
{
namespace tspline
{

/// Error terms remain separate so a near-zero T-spline bake residual cannot
/// hide either the original reduction error or the geometry moved by a seam
/// average merge.
struct LeafErrorBreakdown
{
    double source_reduction_error = 0.0;
    double seam_displacement = 0.0;
    double bake_decomposition_error = 0.0;

    double ConservativeBound() const;
};

struct ErrorAccountingReport
{
    std::size_t leaf_count = 0;
    double max_source_reduction_error = 0.0;
    double max_seam_displacement = 0.0;
    double max_bake_decomposition_error = 0.0;
    double max_conservative_error = 0.0;
};

struct ErrorValidationOptions
{
    double maximum_conservative_error = std::numeric_limits<double>::infinity();
};

struct ErrorValidationReport
{
    ErrorAccountingReport accounting;
    bool within_requested_limit = true;
};

ErrorAccountingReport SummarizeErrors(const std::vector<LeafErrorBreakdown> &errors);
ErrorValidationReport ValidateErrorBounds(const std::vector<LeafErrorBreakdown> &errors,
                                          const ErrorValidationOptions &options = {});

} // namespace tspline
} // namespace mfem_raytracing

#endif
