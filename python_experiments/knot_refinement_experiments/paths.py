"""Output directories for knot-refinement experiment runners."""

from __future__ import annotations

from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parent


def output_dir_for(experiment: str) -> Path:
    """``knot_refinement_experiments/<experiment>/outputs/``"""
    return EXPERIMENTS_ROOT / experiment / "outputs"


def ensure_output_dir(experiment: str) -> Path:
    out = output_dir_for(experiment)
    out.mkdir(parents=True, exist_ok=True)
    return out
