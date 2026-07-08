#ifndef CURVE_REDUCTION_ERROR_BUDGET_HPP
#define CURVE_REDUCTION_ERROR_BUDGET_HPP

namespace mfem_raytracing
{

enum class ErrorBudgetPolicy
{
    Cumulative,
    EqualPerStep,
    WeightedLate,
    Geometric,
};

double StepErrorBudget(double max_error,
                       int n_steps,
                       int step_index,
                       double remaining_budget,
                       ErrorBudgetPolicy policy);

} // namespace mfem_raytracing

#endif
