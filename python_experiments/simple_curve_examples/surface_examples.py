"""Tensor-product B-spline / NURBS surfaces from simple curve examples.

Each surface is built from two registered curves in :mod:`examples` by forming a
separable 3D control net

    P[i, j] = (x_u(i), y_v(j), z_u(i) + z_v(j))

and using the u-curve / v-curve knot vectors and degrees along the two parametric
directions.  Rational curves contribute tensor-product weights w[i, j] = w_u(i) w_v(j).

Plotting uses geomdl VisMPL (interactive 3D) or the matplotlib breakdown helpers
from :mod:`bspline_surface_visualizer`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from geomdl import BSpline, NURBS

_PYTHON_EXPERIMENTS = Path(__file__).resolve().parent.parent
if str(_PYTHON_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_PYTHON_EXPERIMENTS))

from bspline_surface_visualizer import (
    build_bspline_surface,
    render_surface_vismpl,
    visualize_bspline_surface,
)
from simple_curve_examples.examples import SimpleCurveExample, load_simple_example

SurfaceType = Literal["bspline", "nurbs"]
PlotBackend = Literal["vismpl", "matplotlib", "breakdown"]


def tensor_product_control_net(
    u_curve: SimpleCurveExample,
    v_curve: SimpleCurveExample,
    *,
    y_scale: float = 1.0,
    z_scale: float = 1.0,
) -> np.ndarray:
    """
    Build a 3D control net from two 2D curves using separable coordinates.

    Row index ``i`` follows the u-curve; column index ``j`` follows the v-curve.
    """
    u_cp = np.asarray(u_curve.control_points, dtype=float)
    v_cp = np.asarray(v_curve.control_points, dtype=float)
    if u_cp.ndim != 2 or u_cp.shape[1] != 2:
        raise ValueError("tensor_product_control_net expects 2D u-curve controls")
    if v_cp.ndim != 2 or v_cp.shape[1] != 2:
        raise ValueError("tensor_product_control_net expects 2D v-curve controls")

    n_u, n_v = u_cp.shape[0], v_cp.shape[0]
    net = np.zeros((n_u, n_v, 3), dtype=float)
    for i in range(n_u):
        for j in range(n_v):
            net[i, j, 0] = u_cp[i, 0]
            net[i, j, 1] = y_scale * v_cp[j, 0]
            net[i, j, 2] = z_scale * (u_cp[i, 1] + v_cp[j, 1])
    return net


def tensor_product_weights(
    u_curve: SimpleCurveExample,
    v_curve: SimpleCurveExample,
) -> np.ndarray | None:
    """Return an (n_u, n_v) weight net, or ``None`` for a polynomial surface."""
    w_u = np.ones(u_curve.control_points.shape[0], dtype=float)
    w_v = np.ones(v_curve.control_points.shape[0], dtype=float)
    if u_curve.weights is not None:
        w_u = np.asarray(u_curve.weights, dtype=float)
    if v_curve.weights is not None:
        w_v = np.asarray(v_curve.weights, dtype=float)
    if np.allclose(w_u, 1.0) and np.allclose(w_v, 1.0):
        return None
    return np.outer(w_u, w_v)


def infer_surface_type(
    u_curve: SimpleCurveExample,
    v_curve: SimpleCurveExample,
) -> SurfaceType:
    if u_curve.type == "nurbs" or v_curve.type == "nurbs":
        return "nurbs"
    return "bspline"


@dataclass(frozen=True)
class SimpleSurfaceExample:
    """Tensor-product surface built from two :class:`SimpleCurveExample` curves."""

    name: str
    type: SurfaceType
    u_curve_name: str
    v_curve_name: str
    control_points: np.ndarray
    knotvector_u: np.ndarray
    knotvector_v: np.ndarray
    degree_u: int
    degree_v: int
    weights: np.ndarray | None = None

    @property
    def n_u(self) -> int:
        return int(self.control_points.shape[0])

    @property
    def n_v(self) -> int:
        return int(self.control_points.shape[1])

    @property
    def is_rational(self) -> bool:
        return self.type == "nurbs" and self.weights is not None

    def default_title(self) -> str:
        kind = "NURBS" if self.type == "nurbs" else "B-spline"
        rational = " rational" if self.is_rational else ""
        return (
            f"{self.name} — {kind}{rational} surface "
            f"({self.u_curve_name} × {self.v_curve_name}, "
            f"{self.n_u}×{self.n_v} CPs, p_u={self.degree_u}, p_v={self.degree_v})"
        )

    @classmethod
    def from_curves(
        cls,
        name: str,
        u_curve: SimpleCurveExample,
        v_curve: SimpleCurveExample,
        *,
        y_scale: float = 1.0,
        z_scale: float = 1.0,
        surface_type: SurfaceType | None = None,
    ) -> SimpleSurfaceExample:
        stype = surface_type or infer_surface_type(u_curve, v_curve)
        return cls(
            name=name,
            type=stype,
            u_curve_name=u_curve.name,
            v_curve_name=v_curve.name,
            control_points=tensor_product_control_net(
                u_curve,
                v_curve,
                y_scale=y_scale,
                z_scale=z_scale,
            ),
            knotvector_u=np.asarray(u_curve.knots, dtype=float),
            knotvector_v=np.asarray(v_curve.knots, dtype=float),
            degree_u=int(u_curve.degree),
            degree_v=int(v_curve.degree),
            weights=tensor_product_weights(u_curve, v_curve),
        )

    def to_geomdl_surface(self, *, delta: float = 0.04) -> BSpline.Surface | NURBS.Surface:
        if self.type == "bspline":
            if self.weights is not None:
                raise ValueError("B-spline surface cannot carry a non-trivial weight net")
            return build_bspline_surface(
                self.control_points,
                self.knotvector_u,
                self.knotvector_v,
                self.degree_u,
                self.degree_v,
                delta=delta,
            )

        if self.weights is None:
            raise ValueError("NURBS surface requires a weight net")

        net = np.asarray(self.control_points, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        n_u, n_v = net.shape[0], net.shape[1]
        if weights.shape != (n_u, n_v):
            raise ValueError(f"weights must have shape ({n_u}, {n_v}), got {weights.shape}")

        ctrlptsw: list[list[float]] = []
        for i in range(n_u):
            for j in range(n_v):
                w = float(weights[i, j])
                x, y, z = net[i, j]
                ctrlptsw.append([x * w, y * w, z * w, w])

        surf = NURBS.Surface()
        surf.degree_u = self.degree_u
        surf.degree_v = self.degree_v
        surf.set_ctrlpts(ctrlptsw, n_u, n_v)
        surf.knotvector_u = self.knotvector_u.tolist()
        surf.knotvector_v = self.knotvector_v.tolist()
        surf.delta = delta
        return surf

    def plot(
        self,
        *,
        backend: PlotBackend = "vismpl",
        save_path: Path | str | None = None,
        show: bool = True,
        title: str | None = None,
        delta: float = 0.04,
        vismpl_style: Literal["surface", "wireframe"] = "surface",
    ) -> None:
        surface = self.to_geomdl_surface(delta=delta)
        label = title or self.default_title()
        out = Path(save_path) if save_path is not None else None

        if backend == "vismpl":
            write_file = None
            if out is not None:
                write_file = out if out.suffix else out.with_suffix(".png")
                write_file.parent.mkdir(parents=True, exist_ok=True)
            render_surface_vismpl(
                surface,
                style=vismpl_style,
                show=show,
                save_path=write_file,
            )
            if write_file is not None:
                print(f"Saved {write_file}")
            return

        if backend == "matplotlib":
            visualize_bspline_surface(
                surface,
                title=label,
                save_stem=out,
                show=show,
                use_vismpl=False,
                vismpl_style=vismpl_style,
            )
            return

        if backend == "breakdown":
            visualize_bspline_surface(
                surface,
                title=label,
                save_stem=out,
                show=show,
                use_vismpl=True,
                vismpl_style=vismpl_style,
            )
            return

        raise ValueError(f"unknown plot backend: {backend}")


# ------------------------------ Registered tensor-product surfaces ------------------------------
#
# Each entry: (name, u_curve, v_curve, y_scale, z_scale)

_SURFACE_SPECS: list[tuple[str, str, str, float, float]] = [
    # --- B-spline × B-spline (10) ---
    ("s_shaped_peak_saddle", "s_shaped", "single_peak_uniform", 0.8, 1.0),
    ("asymmetric_lopsided_ribbon", "asymmetric_s_shaped", "single_peak_lopsided", 0.75, 1.0),
    ("s_shaped_trailing_wave", "s_shaped", "single_peak_trailing", 1.0, 1.0),
    ("right_angled_uniform_ramp", "right_angled_curve", "single_peak_uniform", 0.9, 1.0),
    ("plateau_s_shaped_screen", "slow_ascent_plateau", "s_shaped", 1.0, 0.85),
    ("lopsided_asymmetric_valley", "single_peak_lopsided", "asymmetric_s_shaped", 0.8, 1.0),
    ("right_angled_plateau_shelf", "right_angled_curve", "slow_ascent_plateau", 1.0, 1.0),
    ("right_angled_s_shaped_ramp", "right_angled_curve", "s_shaped", 0.85, 1.0),
    ("right_angled_squared", "right_angled_curve", "right_angled_curve", 1.0, 1.0),
    ("trailing_lopsided_saddle", "single_peak_trailing", "single_peak_lopsided", 1.0, 1.0),
    # --- NURBS / rational (5) ---
    ("semicircle_plateau_shell", "semicircle", "slow_ascent_plateau", 1.0, 1.0),
    ("semicircle_lopsided_dome", "semicircle", "single_peak_lopsided", 0.85, 1.0),
    ("right_angled_semicircle_shell", "right_angled_curve", "semicircle", 1.0, 1.0),
    ("circular_ascent_peak_shell", "circular_ascent_plateau", "single_peak_uniform", 0.8, 1.0),
    ("circular_s_shaped_crown", "circular_ascent_plateau", "s_shaped", 0.7, 0.9),
]

SURFACE_EXAMPLES: dict[str, SimpleSurfaceExample] = {
    name: SimpleSurfaceExample.from_curves(
        name,
        load_simple_example(u_name),
        load_simple_example(v_name),
        y_scale=y_scale,
        z_scale=z_scale,
    )
    for name, u_name, v_name, y_scale, z_scale in _SURFACE_SPECS
}

surface_names = list(SURFACE_EXAMPLES)


def load_surface_example(name: str) -> SimpleSurfaceExample:
    if name not in SURFACE_EXAMPLES:
        raise KeyError(
            f"unknown surface {name!r}; choose from {list(SURFACE_EXAMPLES)}"
        )
    return SURFACE_EXAMPLES[name]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot a tensor-product surface built from simple curve examples."
    )
    parser.add_argument(
        "-s",
        "--surface",
        choices=surface_names,
        default="s_shaped_peak_saddle",
        help="registered surface example (default: s_shaped_peak_saddle)",
    )
    parser.add_argument(
        "--backend",
        choices=("vismpl", "matplotlib", "breakdown"),
        default="vismpl",
        help="vismpl: interactive 3D; matplotlib: static breakdown; breakdown: both",
    )
    parser.add_argument(
        "--vismpl-style",
        choices=("surface", "wireframe"),
        default="surface",
    )
    parser.add_argument("-o", "--save", type=Path, default=None)
    parser.add_argument("-n", "--no-show", action="store_true")
    parser.add_argument("--delta", type=float, default=0.04, help="geomdl sampling step")
    args = parser.parse_args()

    example = load_surface_example(args.surface)
    example.plot(
        backend=args.backend,
        save_path=args.save,
        show=not args.no_show,
        delta=args.delta,
        vismpl_style=args.vismpl_style,
    )
