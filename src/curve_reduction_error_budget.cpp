#include "curve_reduction_error_budget.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mfem_raytracing
{

double StepErrorBudget(double max_error,
                       int n_steps,
                       int step_index,
                       double remaining_budget,
                       ErrorBudgetPolicy policy)
{
    if (!std::isfinite(max_error))
    {
        return std::numeric_limits<double>::infinity();
    }

    switch (policy)
    {
    case ErrorBudgetPolicy::Cumulative:
        return remaining_budget;
    case ErrorBudgetPolicy::EqualPerStep:
        return std::min(remaining_budget, max_error / static_cast<double>(n_steps));
    case ErrorBudgetPolicy::WeightedLate:
    {
        double weight_sum = 0.0;
        for (int k = step_index; k <= n_steps; ++k)
        {
            weight_sum += static_cast<double>(k);
        }
        return remaining_budget * static_cast<double>(step_index) / weight_sum;
    }
    case ErrorBudgetPolicy::Geometric:
    {
        const double weight = std::pow(2.0, -(n_steps - step_index + 1));
        const double weight_sum = 1.0 - std::pow(2.0, -n_steps);
        return max_error * weight / weight_sum;
    }
    }

    throw std::invalid_argument("StepErrorBudget: unknown policy");
}

} // namespace mfem_raytracing
