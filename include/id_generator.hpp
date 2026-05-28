#ifndef ID_GENERATOR_HPP
#define ID_GENERATOR_HPP

#include <set>
#include <vector>

/**
 * @brief Equation numbering (ID array) for isogeometric analysis.
 *
 * Maps global control-point index A and local DOF index i (both 0-based) to a
 * global equation number P. Fixed (Dirichlet) DOFs are assigned P = 0; active
 * DOFs use 1-based equation numbers (1, 2, ..., N_eq).
 */
class IDGenerator
{
public:
    /**
     * @param num_global_points Total number of global control points.
     * @param dofs_per_point Physical DOFs per point (default 1 for scalar / ray tracing).
     */
    IDGenerator(int num_global_points, int dofs_per_point = 1);

    /** @brief Mark DOF @a local_dof_idx at control point @a global_point_idx as fixed. Might be useful if specific boundary conditions are required. */
    void SetFixedDOF(int global_point_idx, int local_dof_idx = 0);

    /**
     * @brief Build the full ID array.
     * @return ID[A][i] = P (1-based equation number, or 0 if fixed).
     */
    std::vector<std::vector<int>> Generate2DIDArray() const;

    /** @brief Number of active (unconstrained) equations N_eq. */
    int GetTotalActiveEquations() const;

private:
    int num_global_points_ = 0;
    int dofs_per_point_ = 1;

    // fixed_dofs_[A] = local DOF indices i that are constrained at point A.
    std::vector<std::set<int>> fixed_dofs_;
};

#endif
