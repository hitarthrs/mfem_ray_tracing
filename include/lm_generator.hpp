#ifndef LM_GENERATOR_HPP
#define LM_GENERATOR_HPP

#include "id_generator.hpp"
#include "ien_generator.hpp"

#include <vector>

/**
 * @brief Location matrix (LM) for isogeometric assembly.
 *
 * Combines IEN and ID so that for element @a e, local basis index @a a (0-based),
 * and DOF component @a i (0-based):
 *   LM[e][a][i] = ID[A][i]  where A = IEN[e][a].
 *
 * Active equations are 1-based in ID; fixed DOFs remain 0.
 */
class LMGenerator
{
public:
    /**
     * @brief Build LM from precomputed IEN and ID arrays.
     * @param ien IEN[e][a] = global control-point index A (0-based).
     * @param id  ID[A][i] = equation number P (0 if fixed, else 1-based).
     */
    static std::vector<std::vector<std::vector<int>>> Generate(
        const std::vector<std::vector<int>> &ien,
        const std::vector<std::vector<int>> &id);

    /** @brief Convenience: build IEN and ID internally, then form LM. */
    static std::vector<std::vector<std::vector<int>>> Generate(const IENGenerator &ien_gen,
                                                               const IDGenerator &id_gen);
};

#endif
