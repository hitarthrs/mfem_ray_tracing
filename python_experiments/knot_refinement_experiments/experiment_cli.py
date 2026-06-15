"""Shared CLI and geometry loading for knot-refinement experiment runners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from nurbs_curve_examples import available_degrees, load_example, load_example_weights
from simple_curve_examples.examples import (
    curve_names as simple_curve_names,
    simple_example_cli_name,
    simple_example_from_cli,
)

from knot_refinement_experiments.common import P_IN

LIBRARY_CURVE_CHOICES = ("s_shaped", "s_shaped_asymmetric", "multiple_peak")
SIMPLE_CURVE_CHOICES = tuple(simple_example_cli_name(name) for name in simple_curve_names)
CURVE_CHOICES = LIBRARY_CURVE_CHOICES + SIMPLE_CURVE_CHOICES
DEFAULT_CURVE = "multiple_peak"
DEFAULT_INPUT_DEGREE = P_IN
SIMPLE_CURVE_PREFIX = "simple_"


@dataclass(frozen=True)
class ExperimentGeometry:
    """Input curve geometry for knot-refinement experiments."""

    qw: np.ndarray
    u: np.ndarray
    p: int
    weights: np.ndarray | None = None

    @property
    def is_nurbs(self) -> bool:
        return self.weights is not None

    @property
    def is_rational(self) -> bool:
        return self.is_nurbs and not np.allclose(self.weights, 1.0)


def is_simple_curve(curve: str) -> bool:
    return curve.startswith(SIMPLE_CURVE_PREFIX)


def add_curve_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = DEFAULT_CURVE,
) -> None:
    parser.add_argument(
        "--curve",
        choices=CURVE_CHOICES,
        default=default,
        help=(
            "example curve from nurbs_curve_examples or simple_curve_examples "
            f"(simple curves use the {SIMPLE_CURVE_PREFIX}<name> prefix; default: {default})"
        ),
    )


def add_degree_argument(
    parser: argparse.ArgumentParser,
    *,
    default: int = DEFAULT_INPUT_DEGREE,
) -> None:
    degrees = available_degrees()
    parser.add_argument(
        "--degree",
        type=int,
        choices=degrees,
        default=default,
        help=f"input B-spline degree (default: {default}; simple curves require p=3)",
    )


def add_example_arguments(
    parser: argparse.ArgumentParser,
    *,
    curve_default: str = DEFAULT_CURVE,
    degree_default: int = DEFAULT_INPUT_DEGREE,
) -> None:
    """Add ``--curve`` and ``--degree``."""
    add_curve_argument(parser, default=curve_default)
    add_degree_argument(parser, default=degree_default)


def load_input_geometry(
    curve: str,
    *,
    degree: int = DEFAULT_INPUT_DEGREE,
) -> ExperimentGeometry:
    """Return geometry for a library or simple-curve example."""
    if is_simple_curve(curve):
        example = simple_example_from_cli(curve)
        if degree != example.degree:
            raise ValueError(
                f"{curve} is defined at degree {example.degree}; got --degree {degree}"
            )
        weights = example.weights if example.type == "nurbs" else None
        return ExperimentGeometry(
            qw=example.control_points,
            u=example.knots,
            p=example.degree,
            weights=weights,
        )

    qw, u, p = load_example(curve, degree=degree)
    weights = load_example_weights(curve, degree=degree)
    return ExperimentGeometry(qw=qw, u=u, p=p, weights=weights)


def degree_reduce_prefix(p_in: int, curve: str) -> str:
    """Filename stem prefix, e.g. ``p6_to_p5_multiple_peak``."""
    return f"p{p_in}_to_p{p_in - 1}_{curve}"


def example_label(curve: str, p_in: int, *, geom: ExperimentGeometry | None = None) -> str:
    if geom is not None and geom.is_rational:
        kind = "rational NURBS"
    elif geom is not None and geom.is_nurbs:
        kind = "NURBS"
    elif is_simple_curve(curve):
        kind = "simple B-spline"
    else:
        kind = "B-spline"
    return f"{curve} {kind} (p={p_in})"
