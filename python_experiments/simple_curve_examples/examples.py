"""Simple B-spline / NURBS curve examples with plotting helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
from geomdl import BSpline, NURBS

_PYTHON_EXPERIMENTS = Path(__file__).resolve().parent.parent
if str(_PYTHON_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_PYTHON_EXPERIMENTS))

from plot_bspline_curve import plot_bspline_curve

CurveType = Literal["bspline", "nurbs"]
PlotBackend = Literal["matplotlib", "vismpl"]

# Populated from EXAMPLES after all examples are defined (single source of truth).
curve_names: list[str] = []

class SimpleCurveExample:
    """Lightweight curve holder with matplotlib / VisMPL plotting."""

    def __init__(
        self,
        name: str,
        type: CurveType,
        control_points: np.ndarray,
        knots: np.ndarray,
        degree: int,
        weights: np.ndarray | None = None,
    ) -> None:
        self.name = name
        self.type = type
        self.control_points = np.asarray(control_points, dtype=float)
        self.knots = np.asarray(knots, dtype=float)
        self.degree = int(degree)

        n = self.control_points.shape[0]
        expected_len = n + self.degree + 1
        if self.knots.size != expected_len:
            raise ValueError(
                f"{name}: len(knots)={self.knots.size} != n+p+1={expected_len}"
            )

        if type == "nurbs":
            if weights is None:
                self.weights = np.ones(n, dtype=float)
            else:
                w = np.asarray(weights, dtype=float)
                if w.shape != (n,):
                    raise ValueError(f"{name}: weights must have shape ({n},)")
                self.weights = w
        elif type == "bspline":
            self.weights = None
        else:
            raise ValueError(f"Invalid type: {type}")

    @property
    def spatial_dim(self) -> int:
        return int(self.control_points.shape[1])

    @property
    def is_rational(self) -> bool:
        return self.type == "nurbs" and not np.allclose(self.weights, 1.0)

    def default_title(self) -> str:
        kind = "NURBS" if self.type == "nurbs" else "B-spline"
        rational = " rational" if self.is_rational else ""
        return (
            f"{self.name} — {kind}{rational} "
            f"(p={self.degree}, {self.control_points.shape[0]} CPs)"
        )

    def to_geomdl_curve(self) -> BSpline.Curve | NURBS.Curve:
        """Build a geomdl curve for evaluation / VisMPL rendering."""
        cp = self.control_points
        kv = self.knots.tolist()

        if cp.shape[1] == 2:
            ctrlpts = [[float(x), float(y), 0.0] for x, y in cp]
        elif cp.shape[1] == 3:
            ctrlpts = [[float(x), float(y), float(z)] for x, y, z in cp]
        else:
            raise ValueError("control_points must be (n, 2) or (n, 3)")

        if self.type == "bspline":
            curve = BSpline.Curve()
            curve.degree = self.degree
            curve.ctrlpts = ctrlpts
            curve.knotvector = kv
            curve.name = self.name
            return curve

        curve = NURBS.Curve()
        curve.degree = self.degree
        curve.ctrlpts = ctrlpts
        curve.weights = [float(w) for w in self.weights]
        curve.knotvector = kv
        curve.name = self.name
        return curve

    def plot(
        self,
        *,
        backend: PlotBackend = "matplotlib",
        save_path: Path | str | None = None,
        show: bool = True,
        title: str | None = None,
    ) -> None:
        """
        Plot this example using existing project visualisation helpers.

        Parameters
        ----------
        backend:
            ``matplotlib`` — 2-panel figure via :func:`plot_bspline_curve`
            (evaluated curve + knot index plot).
            ``vismpl`` — interactive geomdl VisMPL window (2D or 3D).
        save_path:
            Optional PNG path (saved when provided, or when ``show=False``).
        show:
            Display an interactive window when supported.
        title:
            Figure title override.
        """
        out = Path(save_path) if save_path is not None else None
        label = title or self.default_title()

        if backend == "matplotlib":
            if self.spatial_dim > 2:
                raise ValueError(
                    "matplotlib backend supports 2D control points only; use backend='vismpl'"
                )
            plot_bspline_curve(
                self.control_points,
                self.knots,
                self.degree,
                title=label,
                save_path=out,
                show=show,
                weights=self.weights,
            )
            return

        if backend == "vismpl":
            self._plot_vismpl(label=label, save_path=out, show=show)
            return

        raise ValueError(f"unknown plot backend: {backend}")

    def _plot_vismpl(
        self,
        *,
        label: str,
        save_path: Path | None,
        show: bool,
    ) -> None:
        from geomdl.visualization import VisMPL

        curve = self.to_geomdl_curve()
        curve.name = label
        dim: Literal["2d", "3d"] = "3d" if self.spatial_dim >= 3 else "2d"

        config = VisMPL.VisConfig(
            figure_size=[12.0, 9.0],
            figure_dpi=120,
            axes_equal=True,
            display_axes=True,
            display_labels=True,
            display_legend=True,
            display_ctrlpts=True,
            display_evalpts=True,
        )
        vis_cls = VisMPL.VisCurve3D if dim == "3d" else VisMPL.VisCurve2D
        curve.vis = vis_cls(config)
        curve.evaluate()

        kwargs: dict = {
            "plot": show,
            "evalcolor": "#2a5a9e",
            "cpcolor": "#555555",
        }
        write_file = save_path if (save_path is not None and not show) else None
        if write_file is not None:
            write_file.parent.mkdir(parents=True, exist_ok=True)
            kwargs["filename"] = str(write_file)

        curve.render(**kwargs)
        if write_file is not None:
            print(f"Saved {write_file}")


# ------------------------------ Example 1: S-shaped curve ------------------------------
def s_shaped_control_points() -> np.ndarray:
    """
    8 CPs, smooth cubic S-shaped curve.
    
    Loops up through a crest over the first domain section, 
    crosses an inflection point, and loops down through a symmetric valley.
    """
    return np.array(
        [
            [0.0,  0.0],   # Start point (exactly interpolated)
            [1.0,  2.0],   # Left crest incoming tangent
            [3.0,  3.0],   # Top of the upper loop crest
            [4.5,  1.0],   # Approaching inflection slope
            [5.5, -1.0],   # Symmetrical inflection handoff
            [7.0, -3.0],   # Bottom of the lower loop trough
            [9.0, -2.0],   # Right valley exit tangent
            [10.0, 0.0]    # End point (exactly interpolated)
        ],
        dtype=float,
    )


def s_shaped_knots() -> np.ndarray:
    """
    Open-clamped knot vector for n=8, p=3 (Length 12).
    Every internal knot has a multiplicity of 2 (C1 joint boundaries).
    """
    return np.array(
        [
            0.0, 0.0, 0.0, 0.0,  # Clamped Left Boundary (p + 1 = 4 entries)
            0.35, 0.35,          # First internal non-uniform seam (Multiplicity 2)
            0.65, 0.65,          # Second internal non-uniform seam (Multiplicity 2)
            1.0, 1.0, 1.0, 1.0   # Clamped Right Boundary (p + 1 = 4 entries)
        ],
        dtype=float,
    )

s_shaped_example = SimpleCurveExample(
    name="s_shaped",
    type="bspline",
    control_points=s_shaped_control_points(),
    knots=s_shaped_knots(),
    degree=3,
)

# ------------------------------ Example 2: Asymmetric S-shaped curve ------------------------------
def asymmetric_s_shaped_control_points() -> np.ndarray:
    """
    8 CPs, asymmetric S-shaped curve.
    
    Loops up through a crest over the first domain section, 
    crosses an inflection point, and loops down through a symmetric valley.
    """
    return np.array(
        [
            [0.0,  0.0],   # Start point (exactly interpolated)
            [1.0,  3.5],   # Left crest incoming tangent
            [3.0,  4.0],   # Top of the upper loop crest
            [5.0,  3.0],   # Approaching inflection slope
            [6.0,  0.0],   # Symmetrical inflection handoff
            [8.0, -3.0],   # Bottom of the lower loop trough
            [9.0, -2.0],   # Right valley exit tangent
            [10.0, 0.0]    # End point (exactly interpolated)
        ],
        dtype=float,
    )

asymmetric_s_shaped_example = SimpleCurveExample(
    name="asymmetric_s_shaped",
    type="bspline",
    control_points=asymmetric_s_shaped_control_points(),
    knots=s_shaped_knots(),
    degree=3,
)


# ------------------------------ Example 3: Single peak uniform curve ------------------------------

def single_peak_uniform_control_points() -> np.ndarray:
    """
    8 CPs, single peak uniform curve.
    """
    return np.array(
        [
            [0.0,  0.0],
            [1.0, 1.0],
            [2.0, 1.5],
            [3.0, 1.65],
            [4.0, 1.65],
            [5.0, 1.5],
            [6.0, 1.0],
            [7.0, 0.0]
        ],
        dtype=float,
    )

def single_peak_uniform_knots() -> np.ndarray:
    """
    Open-clamped knot vector for n=8, p=3 (Length 12).
    Every internal knot has a multiplicity of 2 (C1 joint boundaries).
    """
    return np.array(
        [0.0, 0.0, 0.0, 0.0, 0.33, 0.33, 0.66, 0.66,1.0, 1.0, 1.0, 1.0],
        dtype=float,
    )
    
single_peak_uniform_example = SimpleCurveExample(
    name="single_peak_uniform",
    type="bspline",
    control_points=single_peak_uniform_control_points(),
    knots=single_peak_uniform_knots(),
    degree=3,
)



# ------------------------------ Example 4: Single peak lopsided curve ------------------------------
def single_peak_lopsided_control_points() -> np.ndarray:
    """
    8 CPs, single peak lopsided curve.
    """
    return np.array(
        [
            [0.0,  0.0],
            [1.0, 1.8],
            [2.0, 1.70],
            [3.0, 1.65],
            [4.0, 1.5],
            [5.0, 1.22],
            [6.0, 0.8],
            [7.0, 0.0]
        ],
        dtype=float,
    )

def single_peak_lopsided_knots() -> np.ndarray:
    """
    Open-clamped knot vector for n=8, p=3 (Length 12).
    Every internal knot has a multiplicity of 2 (C1 joint boundaries).
    """
    return np.array(
        [0.0, 0.0, 0.0, 0.0, 0.33, 0.33, 0.66, 0.66, 1.0, 1.0, 1.0, 1.0],
        dtype=float,
    )

single_peak_lopsided_example = SimpleCurveExample(
    name="single_peak_lopsided",
    type="bspline",
    control_points=single_peak_lopsided_control_points(),
    knots=single_peak_lopsided_knots(),
    degree=3,
)

# ------------------------------ Example 5: Semicircle  curve ------------------------------
def semicircle_control_points() -> np.ndarray:
    return np.array([
        [-4.0, 0.0],
        [-4.0, 2.0],
        [-2.6667, 4.0],
        [0.0, 4.0],
        [2.6667, 4.0],
        [4.0, 2.0],
        [4.0, 0.0],
    ], dtype=float)

def semicircle_knots() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=float)

def semicircle_weights() -> np.ndarray:
    return np.array([1.0, 2/3, 0.5, 0.5, 0.5, 2/3, 1.0], dtype=float)

semicircle_example = SimpleCurveExample(
    name="semicircle",
    type="nurbs",
    control_points=semicircle_control_points(),
    knots=semicircle_knots(),
    weights=semicircle_weights(),
    degree=3,
)

# ------------------------------ Example 6: Single peak trailing curve ------------------------------
def single_peak_trailing_control_points() -> np.ndarray:
    """6 CPs, asymmetric S-curve in the xy plane, degree p=3."""
    return np.array(
        [
            [0.0, 0.0],
            [0.5, 3.0],
            [2.0, 3.0],
            [2.5, 0.0],
            [6.0, -1.0],
            [9.0, -1.0],
        ],
        dtype=float,
    )


def single_peak_trailing_knots() -> np.ndarray:
    """Open knot vector on [0, 1]; internal seam at u=0.5 (multiplicity 2)."""
    return np.array(
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
        dtype=float,
    )

single_peak_trailing_example = SimpleCurveExample(
    name="single_peak_trailing",
    type="bspline",
    control_points=single_peak_trailing_control_points(),
    knots=single_peak_trailing_knots(),
    degree=3,
)


#  ------------------------------ Example 7: Slow ascent plateau curve ------------------------------
def slow_ascent_plateau_control_points() -> np.ndarray:
    """6 CPs, slow ascent plateau curve."""
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 3.0],
            [3.0, 4.0],
            [5.0, 4.0],
            [6.0, 4.0],
            [8.0, 4.0],
        ],
        dtype=float,
    )

def slow_ascent_plateau_knots() -> np.ndarray:
    """Open knot vector on [0, 1]; internal seam at u=0.5 (multiplicity 2)."""
    return np.array(
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
        dtype=float,
    )

slow_ascent_plateau_example = SimpleCurveExample(
    name="slow_ascent_plateau",
    type="bspline",
    control_points=slow_ascent_plateau_control_points(),
    knots=slow_ascent_plateau_knots(),
    degree=3,
)


# ------------------------------ Example 8: Circular ascent plateau curve ------------------------------

def circular_ascent_plateau_control_points() -> np.ndarray:
    return np.array([
        [-4.0, 0.0],
        [-4.0, 2.0],
        [-2.6667, 4.0],
        [0.0, 4.0],
        [2.0, 4.0],
        [4.0, 4.0],
        [6.0, 4.0],
    ], dtype=float)

def circular_ascent_plateau_knots() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=float)


def circular_ascent_plateau_weights() -> np.ndarray:
    return np.array([1.0, 2/3, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)

circular_ascent_plateau_example = SimpleCurveExample(
    name="circular_ascent_plateau",
    type="nurbs",
    control_points=circular_ascent_plateau_control_points(),
    knots=circular_ascent_plateau_knots(),
    weights=circular_ascent_plateau_weights(),
    degree=3,
)

# ------------------------------ Example 9: Right angled curve ------------------------------

def right_angled_curve_control_points() -> np.ndarray:
    """
    8 CPs, single peak uniform curve.
    """
    return np.array(
        [
            [0.0,  0.0],
            [0.5,  0.5],
            [1.0, 1.0],
            [2.0,  2.0],
            [3.0,  1.0],
            [5.0, -1.0],
            [6.0, -2.0],
            [7.0, -1.0],
            [7.5, -0.5],
            [8.0,  0.0],
        ],
        dtype=float,
    )

def right_angled_curve_knots() -> np.ndarray:
    """
    Open-clamped knot vector for n=10, p=3 (Length 14).
    Every internal knot has a multiplicity of 3.
    """
    return np.array(
        [0.0, 0.0, 0.0, 0.0, 0.33 , 0.33 , 0.33, 0.66, 0.66, 0.66 ,1.0, 1.0, 1.0, 1.0],
        dtype=float,
    )
    
right_angled_curve_example = SimpleCurveExample(
    name="right_angled_curve",
    type="bspline",
    control_points=right_angled_curve_control_points(),
    knots=right_angled_curve_knots(),
    degree=3,
)



EXAMPLES: dict[str, SimpleCurveExample] = {
    "s_shaped": s_shaped_example,
    "asymmetric_s_shaped": asymmetric_s_shaped_example,
    "single_peak_uniform": single_peak_uniform_example,
    "single_peak_lopsided": single_peak_lopsided_example,
    "semicircle": semicircle_example,
    "single_peak_trailing": single_peak_trailing_example,
    "slow_ascent_plateau": slow_ascent_plateau_example,
    "circular_ascent_plateau": circular_ascent_plateau_example,
    "right_angled_curve": right_angled_curve_example,
}

curve_names = list(EXAMPLES)

def load_simple_example(name: str) -> SimpleCurveExample:
    """Return a registered simple curve example by short name."""
    if name not in EXAMPLES:
        raise KeyError(f"unknown simple curve {name!r}; choose from {list(EXAMPLES)}")
    return EXAMPLES[name]


def simple_example_cli_name(name: str) -> str:
    """CLI name with ``simple_`` prefix (avoids nurbs_curve_examples name clashes)."""
    return f"simple_{name}"


def simple_example_from_cli(curve: str) -> SimpleCurveExample:
    """Resolve ``simple_<name>`` CLI flag to a :class:`SimpleCurveExample`."""
    prefix = "simple_"
    if not curve.startswith(prefix):
        raise ValueError(f"not a simple-curve CLI name: {curve!r}")
    return load_simple_example(curve[len(prefix) :])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot a simple curve example.")
    parser.add_argument(
        "--backend",
        choices=("matplotlib", "vismpl"),
        default="matplotlib",
    )
    parser.add_argument("-o", "--save", type=Path, default=None)
    parser.add_argument("-n", "--no-show", action="store_true")
    parser.add_argument("-c", choices=curve_names, default="single_peak_lopsided")
    args = parser.parse_args()

    example = load_simple_example(args.c)

    example.plot(
        backend=args.backend,
        save_path=args.save,
        show=not args.no_show,
    )
