"""Three-column comparison figure for hybrid curvature + adaptive refinement."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from knot_refinement_experiments.common import (
    P_IN,
    P_OUT,
    curve_pair_eval_points,
    plot_curve_pair_on_ax,
    plot_error_array_on_ax,
    square_axis_limits,
)


def render_hybrid_comparison_figure(
    *,
    qw_orig: np.ndarray,
    u_orig: np.ndarray,
    pw_orig: np.ndarray,
    uh_orig: np.ndarray,
    err_orig: np.ndarray,
    qw_curv: np.ndarray,
    u_curv: np.ndarray,
    pw_curv: np.ndarray,
    uh_curv: np.ndarray,
    err_curv: np.ndarray,
    qw_final: np.ndarray,
    u_final: np.ndarray,
    pw_final: np.ndarray,
    uh_final: np.ndarray,
    err_final: np.ndarray,
    save_path: Path,
    show: bool,
    p_in: int = P_IN,
) -> None:
    fig = plt.figure(figsize=(18.0, 10.0), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0], hspace=0.45, wspace=0.12)

    ax_curves_orig = fig.add_subplot(gs[0, 0])
    ax_curves_curv = fig.add_subplot(gs[0, 1], sharex=ax_curves_orig, sharey=ax_curves_orig)
    ax_curves_final = fig.add_subplot(gs[0, 2], sharex=ax_curves_orig, sharey=ax_curves_orig)
    ax_err_orig = fig.add_subplot(gs[1, 0])
    ax_err_curv = fig.add_subplot(gs[1, 1])
    ax_err_final = fig.add_subplot(gs[1, 2])

    p_curv = int(u_curv.size - qw_curv.shape[0] - 1)
    p_final = int(u_final.size - qw_final.shape[0] - 1)
    all_curve_pts = np.vstack(
        [
            curve_pair_eval_points(qw_orig, u_orig, p_in, pw_orig, uh_orig),
            curve_pair_eval_points(qw_curv, u_curv, p_curv, pw_curv, uh_curv),
            curve_pair_eval_points(qw_final, u_final, p_final, pw_final, uh_final),
        ]
    )
    xlim, ylim = square_axis_limits(all_curve_pts)

    plot_curve_pair_on_ax(
        ax_curves_orig, qw_orig, u_orig, p_in, pw_orig, uh_orig,
        title=f"original → p={p_in - 1}", input_label=f"original p={p_in}", xlim=xlim, ylim=ylim,
    )
    plot_curve_pair_on_ax(
        ax_curves_curv, qw_curv, u_curv, p_curv, pw_curv, uh_curv,
        title=f"curvature → p={p_curv - 1}",
        input_label=f"curvature p={p_curv} ({qw_curv.shape[0]} CPs)", xlim=xlim, ylim=ylim,
    )
    plot_curve_pair_on_ax(
        ax_curves_final, qw_final, u_final, p_final, pw_final, uh_final,
        title=f"hybrid final → p={p_final - 1}",
        input_label=f"hybrid p={p_final} ({qw_final.shape[0]} CPs)", xlim=xlim, ylim=ylim,
    )
    for ax in (ax_curves_curv, ax_curves_final):
        ax.tick_params(labelleft=False, labelbottom=False)

    plot_error_array_on_ax(ax_err_orig, err_orig, u_orig, title="error (original reduction)")
    plot_error_array_on_ax(ax_err_curv, err_curv, u_curv, title="error (curvature reduction)")
    plot_error_array_on_ax(ax_err_final, err_final, u_final, title="error (hybrid final reduction)")

    fig.suptitle(
        f"p{p_in} → p{p_in - 1} hybrid: original | curvature warm-start | adaptive polish",
        fontsize=13,
        y=0.98,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)
