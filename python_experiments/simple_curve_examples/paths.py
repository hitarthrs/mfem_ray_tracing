"""Output directories for simple curve / surface example plots."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PACKAGE_DIR / "outputs"
CURVE_OUTPUTS_DIR = OUTPUTS_DIR / "curves"
SURFACE_OUTPUTS_DIR = OUTPUTS_DIR / "surfaces"


def ensure_curve_outputs_dir() -> Path:
    CURVE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return CURVE_OUTPUTS_DIR


def ensure_surface_outputs_dir() -> Path:
    SURFACE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return SURFACE_OUTPUTS_DIR


def _resolve_in_subdir(
    user_path: Path | str | None,
    *,
    name: str,
    subdir: Path,
    suffix: str,
    show: bool,
) -> Path | None:
    """
    Resolve a plot output path under ``subdir``.

    - ``user_path`` set, no parent → ``subdir / filename``
    - ``user_path`` set, with parent → use as given
    - ``user_path`` unset and ``show=False`` → ``subdir / {name}{suffix}``
    - otherwise → ``None`` (display only)
    """
    if user_path is not None:
        path = Path(user_path)
        if path.suffix == "":
            path = path.with_suffix(suffix)
        if path.parent == Path("."):
            path = subdir / path.name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if not show:
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{name}{suffix}"

    return None


def resolve_curve_save_path(
    user_path: Path | str | None,
    *,
    name: str,
    show: bool = True,
    suffix: str = ".png",
) -> Path | None:
    """Default curve plots → ``outputs/curves/``."""
    return _resolve_in_subdir(
        user_path,
        name=name,
        subdir=ensure_curve_outputs_dir(),
        suffix=suffix,
        show=show,
    )


def resolve_surface_save_path(
    user_path: Path | str | None,
    *,
    name: str,
    show: bool = True,
    suffix: str = ".png",
) -> Path | None:
    """
    Default surface plots → ``outputs/surfaces/``.

    For matplotlib / breakdown backends pass ``suffix=""`` to get a stem path.
    """
    return _resolve_in_subdir(
        user_path,
        name=name,
        subdir=ensure_surface_outputs_dir(),
        suffix=suffix,
        show=show,
    )
