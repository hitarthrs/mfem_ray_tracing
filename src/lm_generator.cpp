#include "lm_generator.hpp"

#include <stdexcept>

namespace
{

void ValidateIENAndID(const std::vector<std::vector<int>> &ien,
                      const std::vector<std::vector<int>> &id)
{
    if (ien.empty())
    {
        throw std::invalid_argument("LMGenerator: IEN must have at least one element");
    }
    if (id.empty())
    {
        throw std::invalid_argument("LMGenerator: ID must have at least one control point");
    }

    const int nn = static_cast<int>(ien[0].size());
    if (nn <= 0)
    {
        throw std::invalid_argument("LMGenerator: IEN element must have local nodes");
    }

    const int dofs_per_point = static_cast<int>(id[0].size());
    if (dofs_per_point <= 0)
    {
        throw std::invalid_argument("LMGenerator: ID must have at least one DOF per point");
    }

    for (const auto &elem : ien)
    {
        if (static_cast<int>(elem.size()) != nn)
        {
            throw std::invalid_argument("LMGenerator: inconsistent local node count in IEN");
        }
    }

    for (const auto &point : id)
    {
        if (static_cast<int>(point.size()) != dofs_per_point)
        {
            throw std::invalid_argument("LMGenerator: inconsistent DOF count in ID");
        }
    }
}

}  // namespace

std::vector<std::vector<std::vector<int>>> LMGenerator::Generate(
    const std::vector<std::vector<int>> &ien,
    const std::vector<std::vector<int>> &id)
{
    ValidateIENAndID(ien, id);

    const int ne = static_cast<int>(ien.size());
    const int nn = static_cast<int>(ien[0].size());
    const int ncp = static_cast<int>(id.size());
    const int dofs_per_point = static_cast<int>(id[0].size());

    std::vector<std::vector<std::vector<int>>> lm(
        ne, std::vector<std::vector<int>>(nn, std::vector<int>(dofs_per_point)));

    for (int e = 0; e < ne; ++e)
    {
        for (int a = 0; a < nn; ++a)
        {
            const int A = ien[e][a];
            if (A < 0 || A >= ncp)
            {
                throw std::out_of_range("LMGenerator: IEN entry out of ID range");
            }
            for (int i = 0; i < dofs_per_point; ++i)
            {
                lm[e][a][i] = id[A][i];
            }
        }
    }

    return lm;
}

std::vector<std::vector<std::vector<int>>> LMGenerator::Generate(const IENGenerator &ien_gen,
                                                                 const IDGenerator &id_gen)
{
    const auto ien = ien_gen.Generate2DIEN();
    const auto id = id_gen.Generate2DIDArray();
    return Generate(ien, id);
}
