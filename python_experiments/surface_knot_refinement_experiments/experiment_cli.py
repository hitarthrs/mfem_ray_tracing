"""Shared CLI and geometry loading for surface knot-refinement experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from simple_curve_examples.surface_examples import (
    SURFACE_EXAMPLES,
    load_surface_example,
    surface_names,
)

DEFAULT_SURFACE = "s_shaped_peak_saddle"


@dataclass(frozen=True)
class ExperimentSurfaceGeometry:
    """Input surface geometry for knot-refinement experiments."""

    qw: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p_u: int
    p_v: int
    weights: np.ndarray | None = None
    name: str = ""

    @property
    def is_nurbs(self) -> bool:
        return self.weights is not None

    @property
    def is_rational(self) -> bool:
        return self.is_nurbs and not np.allclose(self.weights, 1.0)


def add_surface_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = DEFAULT_SURFACE,
) -> None:
    parser.add_argument(
        "-s",
        "--surface",
        choices=surface_names,
        default=default,
        help=f"registered surface example (default: {default})",
    )


def load_input_geometry(surface: str) -> ExperimentSurfaceGeometry:
    """Load a registered simple surface example."""
    if surface not in SURFACE_EXAMPLES:
        raise KeyError(f"unknown surface {surface!r}; choose from {list(SURFACE_EXAMPLES)}")
    ex = load_surface_example(surface)
    return ExperimentSurfaceGeometry(
        qw=np.asarray(ex.control_points, dtype=float),
        u=np.asarray(ex.knotvector_u, dtype=float),
        v=np.asarray(ex.knotvector_v, dtype=float),
        p_u=int(ex.degree_u),
        p_v=int(ex.degree_v),
        weights=None if ex.weights is None else np.asarray(ex.weights, dtype=float),
        name=ex.name,
    )


def degree_reduce_prefix(p_u: int, p_v: int, surface: str) -> str:
    """Filename stem prefix, e.g. ``p3p3_to_p2p2_s_shaped_peak_saddle``."""
    return f"p{p_u}p{p_v}_to_p{p_u - 1}p{p_v - 1}_{surface}"


def example_label(surface: str, p_u: int, p_v: int, *, geom: ExperimentSurfaceGeometry | None = None) -> str:
    kind = "NURBS" if geom is not None and geom.is_rational else "B-spline"
    return f"{surface} ({kind}, p_u={p_u}, p_v={p_v})"
