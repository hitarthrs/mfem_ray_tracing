#include "id_generator.hpp"

#include <stdexcept>

namespace
{

void ValidatePointCount(int num_global_points)
{
    if (num_global_points <= 0)
    {
        throw std::invalid_argument("IDGenerator: num_global_points must be positive");
    }
}

void ValidateDofsPerPoint(int dofs_per_point)
{
    if (dofs_per_point <= 0)
    {
        throw std::invalid_argument("IDGenerator: dofs_per_point must be positive");
    }
}

}  // namespace

IDGenerator::IDGenerator(int num_global_points, int dofs_per_point)
    : num_global_points_(num_global_points), dofs_per_point_(dofs_per_point)
{
    ValidatePointCount(num_global_points_);
    ValidateDofsPerPoint(dofs_per_point_);
    fixed_dofs_.resize(num_global_points_);
}

void IDGenerator::SetFixedDOF(int global_point_idx, int local_dof_idx)
{
    if (global_point_idx < 0 || global_point_idx >= num_global_points_)
    {
        throw std::out_of_range("IDGenerator::SetFixedDOF: global_point_idx out of range");
    }
    if (local_dof_idx < 0 || local_dof_idx >= dofs_per_point_)
    {
        throw std::out_of_range("IDGenerator::SetFixedDOF: local_dof_idx out of range");
    }
    fixed_dofs_[global_point_idx].insert(local_dof_idx);
}

std::vector<std::vector<int>> IDGenerator::Generate2DIDArray() const
{
    std::vector<std::vector<int>> id(
        num_global_points_, std::vector<int>(dofs_per_point_, 0));

    int next_eq = 1;  // 1-based active equation numbers; 0 reserved for fixed DOFs.
    for (int A = 0; A < num_global_points_; ++A)
    {
        for (int i = 0; i < dofs_per_point_; ++i)
        {
            if (fixed_dofs_[A].count(i) > 0)
            {
                id[A][i] = 0;
            }
            else
            {
                id[A][i] = next_eq++;
            }
        }
    }

    return id;
}

int IDGenerator::GetTotalActiveEquations() const
{
    const int total_dofs = num_global_points_ * dofs_per_point_;
    int fixed_count = 0;
    for (int A = 0; A < num_global_points_; ++A)
    {
        fixed_count += static_cast<int>(fixed_dofs_[A].size());
    }
    return total_dofs - fixed_count;
}
