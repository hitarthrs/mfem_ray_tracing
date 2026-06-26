"""Output directories for multi-step degree reduction plots."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PACKAGE_DIR / "outputs"
CURVE_OUTPUTS_DIR = OUTPUTS_DIR / "curves"


def ensure_curve_outputs_dir() -> Path:
    CURVE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return CURVE_OUTPUTS_DIR


def resolve_curve_save_path(
    user_path: Path | str | None,
    *,
    name: str,
    show: bool = True,
    suffix: str = ".png",
) -> Path | None:
    """
    Resolve a plot output path under ``outputs/curves/``.

    - ``user_path`` set, no parent → ``outputs/curves / filename``
    - ``user_path`` set, with parent → use as given
    - ``user_path`` unset and ``show=False`` → ``outputs/curves / {name}{suffix}``
    - otherwise → ``None`` (display only)
    """
    subdir = ensure_curve_outputs_dir()

    if user_path is not None:
        path = Path(user_path)
        if path.suffix == "":
            path = path.with_suffix(suffix)
        if path.parent == Path("."):
            path = subdir / path.name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if not show:
        return subdir / f"{name}{suffix}"

    return None
