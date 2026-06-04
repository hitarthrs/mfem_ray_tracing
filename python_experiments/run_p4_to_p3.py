import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from b_spline_curve_reduction import DegreeReduceCurve
from visualize_bspline_curve import evaluate_bspline, horseshoe_control_points, s_shaped_control_points, s_shaped_20_control_points, s_shaped_ultra_refined_control_points

U = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.5, 0.5, 0.6, 0.6, 0.75, 0.75, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
Qw = s_shaped_ultra_refined_control_points()
p, ph = 4, 3

out = DegreeReduceCurve(Qw.shape[0], p, U, Qw)
if out == 1:
    raise SystemExit("tolerance exceeded")
Pw, Uh, err = out
last = int(np.max(np.where(np.linalg.norm(Pw, axis=1) > 1e-12)[0], initial=0))
Pw = Pw[: last + 1]
Uh = Uh[: Pw.shape[0] + ph + 1]

print("Pw:\n", Pw)
print("Uh:", Uh)
print("error_array:", err)
if np.any(err > 0):
    print("error_array (nonzero):", err[err > 0])

_, curve_q = evaluate_bspline(Qw, U, p)
_, curve_p = evaluate_bspline(Pw, Uh, ph)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(curve_q[:, 0], curve_q[:, 1], "b-", lw=2, label=f"Qw, U (p={p})")
ax.plot(curve_p[:, 0], curve_p[:, 1], "r-", lw=2, label=f"Pw, Uh (p={ph})")
ax.plot(Qw[:, 0], Qw[:, 1], "ko--", ms=4, label="Qw")
ax.plot(Pw[:, 0], Pw[:, 1], "rs--", ms=4, label="Pw")
ax.set_aspect("equal")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("outputs/p4_to_p3_knot_ultra_refined.png", dpi=150)
print("Saved outputs/p4_to_p3_knot_ultra_refined.png")
