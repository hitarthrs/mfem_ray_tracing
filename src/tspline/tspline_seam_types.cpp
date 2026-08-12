#include "mfem_raytracing/tspline/tspline_seam_types.hpp"

#include <stdexcept>

namespace mfem_raytracing
{
namespace tspline
{

const char *BoundarySideName(BoundarySide side)
{
    switch (side)
    {
        case BoundarySide::UMin: return "u_min";
        case BoundarySide::UMax: return "u_max";
        case BoundarySide::VMin: return "v_min";
        case BoundarySide::VMax: return "v_max";
    }
    throw std::invalid_argument("unknown boundary side");
}

int BoundaryNormalAxis(BoundarySide side)
{
    switch (side)
    {
        case BoundarySide::UMin:
        case BoundarySide::UMax: return 0;
        case BoundarySide::VMin:
        case BoundarySide::VMax: return 1;
    }
    throw std::invalid_argument("unknown boundary side");
}

int BoundaryTangentAxis(BoundarySide side)
{
    return 1 - BoundaryNormalAxis(side);
}

bool BoundaryIsMaximum(BoundarySide side)
{
    return side == BoundarySide::UMax || side == BoundarySide::VMax;
}

std::array<int, 2> BoundaryCornerIndices(BoundarySide side)
{
    switch (side)
    {
        case BoundarySide::UMin: return {0, 1};
        case BoundarySide::UMax: return {2, 3};
        case BoundarySide::VMin: return {0, 2};
        case BoundarySide::VMax: return {1, 3};
    }
    throw std::invalid_argument("unknown boundary side");
}

} // namespace tspline
} // namespace mfem_raytracing
