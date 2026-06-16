"""Simple B-spline / NURBS curve and surface examples."""

from .examples import (
    EXAMPLES,
    SimpleCurveExample,
    curve_names,
    load_simple_example,
    simple_example_cli_name,
    simple_example_from_cli,
    s_shaped_example,
)
from .surface_examples import (
    SURFACE_EXAMPLES,
    SimpleSurfaceExample,
    load_surface_example,
    surface_names,
)

__all__ = [
    "EXAMPLES",
    "SURFACE_EXAMPLES",
    "SimpleCurveExample",
    "SimpleSurfaceExample",
    "curve_names",
    "load_simple_example",
    "load_surface_example",
    "simple_example_cli_name",
    "simple_example_from_cli",
    "s_shaped_example",
    "surface_names",
]
