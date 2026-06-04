"""geomdl NURBS curve for the S-shaped demo from ``run_p4_to_p3.py``."""

from __future__ import annotations

import numpy as np
from geomdl import NURBS, operations

from visualize_bspline_curve import s_shaped_control_points

# Same as run_p4_to_p3.py: Qw, U, p = 4.
S_SHAPED_DEGREE = 4
S_SHAPED_KNOTS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=float,
)


def build_s_shaped_nurbs_curve(*, delta: float = 0.01) -> NURBS.Curve:
    """Unit-weight NURBS curve matching ``run_p4_to_p3`` (degree 4, multi-span knots)."""
    curve = NURBS.Curve()
    curve.degree = S_SHAPED_DEGREE
    curve.ctrlpts = [[float(x), float(y), 0.0] for x, y in s_shaped_control_points()]
    curve.knotvector = S_SHAPED_KNOTS.tolist()
    curve.delta = delta
    return curve


# Knots to insert (each value once; pair => multiplicity 2 after two insertions).
S_SHAPED_KNOTS_TO_INSERT = (0.15, 0.15, 0.2, 0.2,0.5, 0.5, 0.6, 0.6, 0.9, 0.9)


def refine_s_shaped_knots(curve: NURBS.Curve | None = None) -> NURBS.Curve:
    """Insert ``S_SHAPED_KNOTS_TO_INSERT``; curve geometry unchanged, 9 CPs -> 15."""
    curve = build_s_shaped_nurbs_curve() if curve is None else curve
    for u in S_SHAPED_KNOTS_TO_INSERT:
        operations.insert_knot(curve, [u], [1])
    return curve


if __name__ == "__main__":
    curve = refine_s_shaped_knots()
    print("degree:", curve.degree)
    print("knots:", curve.knotvector)
    print("ctrlpts (x, y):")
    for pt in curve.ctrlpts:
        print(f"  [{pt[0]:.10f}, {pt[1]:.10f}]")
