"""
Dummy Algorithm 2: split at ``u = 0.5``, then one A5.11 pass per half.

Pipeline:
  1. Read *curve properties* (control points, knot vector, degree, weights) from input.
  2. Split the curve into two parameter segments at ``u = 0.5``.
  3. Read curve properties of each split piece.
  4. Run single-step degree reduction (A5.11) on each piece independently.

Intended for exercising multi-segment orchestration before wiring real splitting
strategies.

Run::

    cd python_experiments
    .venv/bin/python -m single_step_degree_reduction.dummy_approach
    .venv/bin/python -m single_step_degree_reduction.dummy_approach -c semicircle -n
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_PYTHON_EXPERIMENTS = Path(__file__).resolve().parent.parent
if str(_PYTHON_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_PYTHON_EXPERIMENTS))

import numpy as np
from geomdl import operations

from b_spline_curve_reduction import DegreeReduceCurve
from knot_refinement_experiments.common import (
    build_geomdl_from_geometry,
    geometry_from_geomdl,
)
from nurbs_degree_reduction import degree_reduce_nurbs

try:
    from .protocol import (
        Curve,
        SingleStepReductionResult,
        single_step_result_from_lists,
    )
except ImportError:
    from single_step_degree_reduction.protocol import (
        Curve,
        SingleStepReductionResult,
        single_step_result_from_lists,
    )

DEFAULT_SPLIT_U = 0.5


@dataclass(frozen=True)
class CurveProperties:
    """B-spline / NURBS geometry extracted from a geomdl curve."""

    control_points: np.ndarray
    knotvector: np.ndarray
    degree: int
    weights: np.ndarray | None = None
    domain: tuple[float, float] = (0.0, 1.0)

    @property
    def n_control_points(self) -> int:
        return int(self.control_points.shape[0])


def curve_properties_from_curve(curve: Curve) -> CurveProperties:
    """Extract control points, knot vector, degree, and optional weights."""
    control_points, knotvector, degree, weights = geometry_from_geomdl(curve)
    return CurveProperties(
        control_points=np.asarray(control_points, dtype=float),
        knotvector=np.asarray(knotvector, dtype=float),
        degree=int(degree),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        domain=(float(curve.domain[0]), float(curve.domain[1])),
    )


def split_curve_into_two(
    curve: Curve,
    u: float = DEFAULT_SPLIT_U,
) -> tuple[Curve, Curve]:
    """Split ``curve`` at parameter ``u``; return left and right sub-curves."""
    u0, u1 = float(curve.domain[0]), float(curve.domain[1])
    u_split = float(u)
    if u_split < u0 - 1e-9 or u_split > u1 + 1e-9:
        raise ValueError(f"split parameter u={u_split} outside domain [{u0}, {u1}]")
    if abs(u_split - u0) < 1e-9 or abs(u_split - u1) < 1e-9:
        raise ValueError(
            f"split parameter u={u_split} must lie strictly inside domain [{u0}, {u1}]"
        )
    left, right = operations.split_curve(curve, u_split)
    return left, right


def segment_error_from_reduction(error_array: np.ndarray) -> float:
    """Scalar segment error attributed to one A5.11 pass (max knot-removal error)."""
    err = np.asarray(error_array, dtype=float)
    if err.size == 0:
        return 0.0
    return float(np.max(err))


def reduce_curve_properties_once(
    props: CurveProperties,
    *,
    tol: float = float("inf"),
    name: str = "",
) -> tuple[Curve, float]:
    """
    One A5.11 degree drop on numpy geometry.

    Returns the reduced geomdl curve (degree ``p - 1``) and ``max(error_array)``.
    """
    n = props.n_control_points
    p = props.degree
    if p < 2:
        raise ValueError(f"degree reduction requires degree >= 2, got {p}")

    if props.weights is None:
        out = DegreeReduceCurve(n, p, props.knotvector, props.control_points, tol=tol)
        if out == 1:
            raise RuntimeError("A5.11 tolerance exceeded on polynomial segment")
        pw, uh, err = out
        weights_out = None
    else:
        out = degree_reduce_nurbs(
            n,
            p,
            props.knotvector,
            props.control_points,
            props.weights,
            tol=tol,
        )
        if out == 1:
            raise RuntimeError("A5.11 tolerance exceeded on NURBS segment")
        pw, weights_out, uh, err = out

    reduced = build_geomdl_from_geometry(
        pw,
        uh,
        p - 1,
        weights=weights_out,
        name=name,
    )
    return reduced, segment_error_from_reduction(err)


def dummy_split_reduce_single_step(
    initial_curve: Curve,
    max_error: float,
    *,
    split_u: float = DEFAULT_SPLIT_U,
) -> SingleStepReductionResult:
    """
    Dummy Algorithm 2 backend.

    Splits at ``split_u`` (default ``0.5``), reduces each half once, and returns
    two segments with A5.11 max errors.
    """
    original_props = curve_properties_from_curve(initial_curve)
    if original_props.degree < 2:
        raise ValueError(
            f"dummy split-reduce requires input degree >= 2, got {original_props.degree}"
        )

    left_curve, right_curve = split_curve_into_two(initial_curve, split_u)
    left_props = curve_properties_from_curve(left_curve)
    right_props = curve_properties_from_curve(right_curve)

    reduced_left, err_left = reduce_curve_properties_once(
        left_props,
        tol=max_error,
        name="dummy-left",
    )
    reduced_right, err_right = reduce_curve_properties_once(
        right_props,
        tol=max_error,
        name="dummy-right",
    )

    return single_step_result_from_lists(
        [reduced_left, reduced_right],
        [err_left, err_right],
    )


@dataclass
class DummySplitReduceStep:
    """Callable backend matching :class:`DegreeReductionSingleStep`."""

    split_u: float = DEFAULT_SPLIT_U

    def __call__(
        self,
        initial_curve: Curve,
        max_error: float,
    ) -> SingleStepReductionResult:
        return dummy_split_reduce_single_step(
            initial_curve,
            max_error,
            split_u=self.split_u,
        )


def curve_xy(curve: Curve) -> np.ndarray:
    """Sample the curve for plotting (xy only)."""
    _ = curve.evalpts
    return np.asarray(curve.evalpts, dtype=float)[:, :2]


def point_at_u(curve: Curve, u: float) -> np.ndarray:
    """Evaluate ``curve`` at parameter ``u`` (xy)."""
    pt = curve.evaluate_single(float(u))
    return np.asarray(pt, dtype=float)[:2]


def plot_curve_overlay(
    ax,
    curve: Curve,
    *,
    color: str,
    cp_color: str,
    label: str,
    lw: float = 2.0,
    ls: str = "-",
    cp_marker: str = "o",
) -> int:
    """Plot evaluated curve and control polygon; return control-point count."""
    xy = curve_xy(curve)
    n_cp = len(curve.ctrlpts)
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, ls=ls, label=label)
    cp = np.asarray(curve.ctrlpts, dtype=float)[:, :2]
    ax.plot(
        cp[:, 0],
        cp[:, 1],
        f"{cp_marker}--",
        color=cp_color,
        ms=4,
        lw=0.8,
        alpha=0.55,
    )
    return n_cp


def plot_dummy_split_reduce(
    curve: Curve,
    result: SingleStepReductionResult,
    *,
    split_u: float = DEFAULT_SPLIT_U,
    curve_name: str = "curve",
    max_error: float = float("inf"),
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Two-panel figure: full input, then each segment overlaid with its reduction."""
    import matplotlib.pyplot as plt

    from knot_refinement_experiments.common import (
        COLOR_INPUT,
        COLOR_INPUT_CP,
    )

    split_pieces = split_curve_into_two(curve, split_u)
    p_in = int(curve.degree)
    p_out = p_in - 1
    split_pt = point_at_u(curve, split_u)
    cmap = plt.get_cmap("tab10")

    fig, (ax_input, ax_overlay) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    plot_curve_overlay(
        ax_input,
        curve,
        color=COLOR_INPUT,
        cp_color=COLOR_INPUT_CP,
        label=f"input p={p_in} ({len(curve.ctrlpts)} CPs)",
        lw=2.4,
    )
    ax_input.plot(
        split_pt[0],
        split_pt[1],
        "x",
        color="crimson",
        ms=10,
        mew=2.5,
        label=f"split u={split_u:g}",
    )
    ax_input.set_title(f"{curve_name}: input p={p_in}")
    ax_input.set_aspect("equal", adjustable="datalim")
    ax_input.grid(True, alpha=0.3)
    ax_input.legend(loc="best", fontsize=8)

    segment_labels = ("left", "right")
    for index, segment in enumerate(result.segments):
        label = segment_labels[index] if index < len(segment_labels) else f"seg {index + 1}"
        color = cmap(index)
        input_piece = split_pieces[index] if index < len(split_pieces) else None

        if input_piece is not None:
            plot_curve_overlay(
                ax_overlay,
                input_piece,
                color=color,
                cp_color=color,
                label=f"{label} input p={p_in} ({len(input_piece.ctrlpts)} CPs)",
                lw=2.2,
                ls="-",
                cp_marker="o",
            )

        plot_curve_overlay(
            ax_overlay,
            segment.curve,
            color=color,
            cp_color=color,
            label=(
                f"{label} reduced p={p_out} ({len(segment.curve.ctrlpts)} CPs, "
                f"err={segment.segment_error:.4g})"
            ),
            lw=2.0,
            ls="--",
            cp_marker="s",
        )

    tol_label = "inf" if not np.isfinite(max_error) else f"{max_error:g}"
    ax_overlay.set_title(
        f"segments overlaid with reductions (p={p_in} solid, p={p_out} dashed)"
    )
    ax_overlay.set_aspect("equal", adjustable="datalim")
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"dummy split-reduce: {curve_name}  |  split u={split_u:g}  |  "
        f"max_error={tol_label}",
        fontsize=11,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"saved {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _main() -> None:
    import argparse
    from pathlib import Path

    from simple_curve_examples.examples import EXAMPLES
    try:
        from .paths import resolve_curve_save_path
    except ImportError:
        from single_step_degree_reduction.paths import resolve_curve_save_path

    try:
        from .protocol import degree_reduction_single_step
    except ImportError:
        from single_step_degree_reduction.protocol import degree_reduction_single_step

    parser = argparse.ArgumentParser(
        description="Dummy split-at-u degree reduction demo with plots.",
    )
    parser.add_argument(
        "-c",
        "--curve",
        choices=sorted(EXAMPLES),
        default="s_shaped",
        help="simple curve example (default: s_shaped)",
    )
    parser.add_argument(
        "--split-u",
        type=float,
        default=DEFAULT_SPLIT_U,
        help=f"parameter split site (default: {DEFAULT_SPLIT_U})",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=float("inf"),
        help="A5.11 tolerance passed to each half (default: inf)",
    )
    parser.add_argument("-o", "--save", type=Path, default=None)
    parser.add_argument("-n", "--no-show", action="store_true")
    args = parser.parse_args()

    example = EXAMPLES[args.curve]
    curve = example.to_geomdl_curve()

    props = curve_properties_from_curve(curve)
    print(
        f"input: degree={props.degree} n_cp={props.n_control_points} "
        f"domain={props.domain}"
    )

    left, right = split_curve_into_two(curve, args.split_u)
    for label, piece in ("left", left), ("right", right):
        p = curve_properties_from_curve(piece)
        print(
            f"  {label}: degree={p.degree} n_cp={p.n_control_points} "
            f"domain={p.domain}"
        )

    backend = DummySplitReduceStep(split_u=float(args.split_u))
    result = degree_reduction_single_step(curve, float(args.max_error), backend=backend)
    print(f"reduced segments: {len(result.segments)}")
    for index, segment in enumerate(result.segments):
        p = curve_properties_from_curve(segment.curve)
        print(
            f"  seg{index}: degree={p.degree} n_cp={p.n_control_points} "
            f"error={segment.segment_error:.6g}"
        )

    save_path = resolve_curve_save_path(
        args.save,
        name=f"dummy_split_reduce_{args.curve}",
        show=not args.no_show,
    )
    plot_dummy_split_reduce(
        curve,
        result,
        split_u=float(args.split_u),
        curve_name=args.curve,
        max_error=float(args.max_error),
        save_path=save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    _main()
