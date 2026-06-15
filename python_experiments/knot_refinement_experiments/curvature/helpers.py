"""Curvature refinement helpers shared by curvature and hybrid runners."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from knot_refinement_algorithms.curvature_knot_refinement import (
    MarchMode,
    curvature_new_knots,
)
from knot_refinement_algorithms.knot_refinement_common import apply_new_knots


def profile_save_path(comparison_path: Path) -> Path:
    return comparison_path.with_name(comparison_path.stem + "_profile.png")


def save_curvature_profile(
    u_samples: np.ndarray,
    s_samples: np.ndarray,
    kappa: np.ndarray,
    spacing_samples: np.ndarray,
    grid: list[float],
    *,
    march_mode: MarchMode,
    save_path: Path,
) -> None:
    fig, (ax_k, ax_s, ax_ds) = plt.subplots(3, 1, figsize=(10, 8), dpi=150, sharex=True)

    ax_k.plot(u_samples, kappa, color="#5c4a9e", lw=1.5)
    ax_k.set_ylabel("κ(u)")
    ax_k.set_title("curvature profile (geomdl derivatives)")
    ax_k.grid(True, alpha=0.3)

    ax_s.plot(u_samples, s_samples, color="#4a7ab8", lw=1.5)
    ax_s.set_ylabel("s(u)")
    ax_s.set_title("cumulative arc length")
    ax_s.grid(True, alpha=0.3)

    spacing_label = "ds(u)" if march_mode == "arc" else "Δu(u)"
    ax_ds.plot(u_samples, spacing_samples, color="#2a9d6a", lw=1.5, label=spacing_label)
    for u in grid:
        ax_ds.axvline(u, color="#e07a3a", alpha=0.25, lw=0.8)
    ax_ds.set_xlabel("u")
    ax_ds.set_ylabel("ds" if march_mode == "arc" else "Δu")
    march_label = "arc-length" if march_mode == "arc" else "parameter u"
    ax_ds.set_title(f"{march_label} spacing + candidate grid ({len(grid)} points)")
    ax_ds.legend(loc="best")
    ax_ds.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def apply_curvature_refinement(
    curve,
    knotvector: np.ndarray,
    degree: int,
    *,
    h_min: float,
    h_max: float,
    alpha: float,
    beta: float,
    n_samples: int,
    insert_multiplicity: int,
    march_mode: MarchMode,
) -> tuple[list[tuple[float, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    new_knots, u_samples, s_samples, kappa, spacing_samples, grid = curvature_new_knots(
        curve,
        knotvector,
        degree,
        h_min=h_min,
        h_max=h_max,
        alpha=alpha,
        beta=beta,
        n_samples=n_samples,
        march_mode=march_mode,
    )

    u_span = float(u_samples[-1] - u_samples[0])
    s_total = float(s_samples[-1])
    scale = s_total / u_span if u_span > 1e-12 else 1.0
    if march_mode == "arc":
        print(
            f"march mode: arc-length (Δu_min={h_min:g}, Δu_max={h_max:g} "
            f"→ ds_min={h_min * scale:g}, ds_max={h_max * scale:g}), "
            f"alpha={alpha:g}, beta={beta:g}"
        )
    else:
        print(
            f"march mode: parameter u (Δu_min={h_min:g}, Δu_max={h_max:g}), "
            f"alpha={alpha:g}, beta={beta:g}"
        )
    print(f"κ max = {float(np.max(kappa)):.6g}, total arc length = {s_total:.6g}")
    print(f"candidate grid ({len(grid)} points): {grid}")

    log = apply_new_knots(curve, new_knots, insert_multiplicity, algorithm="curvature")
    return log, u_samples, s_samples, kappa, spacing_samples, grid
