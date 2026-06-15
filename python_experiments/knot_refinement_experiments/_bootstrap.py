"""Ensure ``python_experiments`` is on ``sys.path`` when running scripts directly."""

from __future__ import annotations

import sys
from pathlib import Path

PYTHON_EXPERIMENTS = Path(__file__).resolve().parent.parent


def install() -> None:
    if str(PYTHON_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(PYTHON_EXPERIMENTS))


install()
