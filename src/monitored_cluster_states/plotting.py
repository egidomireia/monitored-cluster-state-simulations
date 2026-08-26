"""Plotting functions for the thesis results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


# =============================================================================
# ENTROPY VS SYSTEM SIZE
# =============================================================================

def plot_entropy_vs_L(p_values, dimensions, S_matrix, S_matrix_std, p_c, vmin=0.0, vmax=1.0):
    """Plot the mean entropy against the system size for different probabilities."""

    colors_lower = plt.cm.Blues_r(np.linspace(0.0, 0.65, 128))
    colors_upper = plt.cm.Reds(np.linspace(0.35, 1.0, 128))
    colors_combined = np.vstack((colors_lower, colors_upper))
    custom_cmap = mcolors.ListedColormap(colors_combined, name="BlueRed_Visible")

    effective_vmax = vmax
    if p_c >= vmax:
        effective_vmax = p_c + 1e-5

    norm = mcolors.TwoSlopeNorm(vcenter=p_c, vmin=vmin, vmax=effective_vmax)
    fig, ax = plt.subplots(figsize=(11, 7))

    num_lines_to_plot = 12
    indices_to_plot = np.linspace(0, len(p_values) - 1, num_lines_to_plot, dtype=int)

    for idx in indices_to_plot:
        p = p_values[idx]
        means = S_matrix[:, idx]
        stds = S_matrix_std[:, idx]
        color = custom_cmap(norm(p))

        ax.plot(dimensions, means, "-o", color=color, linewidth=3.5, markersize=9, alpha=0.9)
        ax.fill_between(dimensions, means - stds, means + stds, color=color, alpha=0.2, edgecolor="none")

    sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.05, aspect=20)
    cbar.set_label(r"$p$", fontsize=28, labelpad=15)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.invert_yaxis()
    cbar.ax.axhline(p_c, color="black", linewidth=3, linestyle="-")
    cbar.ax.text(-0.17, p_c, r"$p_c$", color="black", va="center", ha="right", fontsize=24)

    ax.set_xlabel(r"$L$", fontsize=28)
    ax.set_ylabel(r"$\langle S \rangle$", fontsize=28)
    ax.tick_params(axis="both", which="major", labelsize=24, width=2, length=6)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(3, 3), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(22)
    ax.set_xlim(dimensions[0] - 10, dimensions[-1] + 10)
    ax.grid(alpha=0.4, linestyle="--")

    plt.tight_layout()
    plt.show()


# =============================================================================
# CV AND DV ENTROPY VS PROBABILITY AT FIXED L
# =============================================================================

def plot_CV_vs_DV_fixed_L(p_values, dimensions, S_matrix, S_matrix_std, S_DV_matrix, S_DV_matrix_std,
                          target_L, p_c=0.4070, p_err_minus=0.0003, p_err_plus=0.0010,
                          r_fixed=0.0, w_fixed=1.0):
    """
    Compare CV and DV entanglement entropy against measurement probability
    for a fixed system size L.
    """

    if target_L not in dimensions:
        raise ValueError(f"Target L={target_L} not found in the dimensions array {dimensions}")

    l_idx = np.where(dimensions == target_L)[0][0]

    cv_means = S_matrix[l_idx, :]
    cv_stds = S_matrix_std[l_idx, :]
    dv_means = S_DV_matrix[l_idx, :]
    dv_stds = S_DV_matrix_std[l_idx, :]

    styles = {
        "CV": {"color": "tab:purple", "ls": "-", "marker": "o", "lw": 4, "ms": 9},
        "DV": {"color": "tab:orange", "ls": "--", "marker": "s", "lw": 4, "ms": 9},
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(p_values, cv_means, color=styles["CV"]["color"], linestyle=styles["CV"]["ls"],
            marker=styles["CV"]["marker"], linewidth=styles["CV"]["lw"],
            markersize=styles["CV"]["ms"], zorder=5)

    ax.fill_between(p_values, cv_means - cv_stds, cv_means + cv_stds,
                    color=styles["CV"]["color"], alpha=0.2, edgecolor="none", zorder=4)

    ax.plot(p_values, dv_means, color=styles["DV"]["color"], linestyle=styles["DV"]["ls"],
            marker=styles["DV"]["marker"], linewidth=styles["DV"]["lw"],
            markersize=styles["DV"]["ms"], zorder=5)

    ax.fill_between(p_values, dv_means - dv_stds, dv_means + dv_stds,
                    color=styles["DV"]["color"], alpha=0.2, edgecolor="none", zorder=4)

    ax.axvline(x=p_c, color="black", linestyle="--", linewidth=4, zorder=10)
    ax.axvspan(p_c - p_err_minus, p_c + p_err_plus, color="black", alpha=0.15, zorder=9)

    error_text = rf"$p_c = {p_c:.4f}_{{-{p_err_minus:.4f}}}^{{+{p_err_plus:.4f}}}$"
    y_text_pos = np.nanmax(cv_means) * 0.5
    ax.text(p_c + 0.02, y_text_pos, error_text, color="black", fontsize=26, va="center")

    ax.set_xlabel(r"$p$", fontsize=28)
    ax.set_ylabel(r"$\langle S \rangle/L$", fontsize=28)
    ax.set_xlim(p_values[0] - 0.02, p_values[-1] + 0.02)
    ax.tick_params(axis="both", which="major", labelsize=24, width=2, length=6)
    ax.grid(alpha=0.4, linestyle="--")

    legend_elements = [
        Line2D([0], [0], color=styles["CV"]["color"], linestyle=styles["CV"]["ls"],
               marker=styles["CV"]["marker"], lw=styles["CV"]["lw"], ms=styles["CV"]["ms"],
               label="Continuous Variable (CV)"),
        Line2D([0], [0], color=styles["DV"]["color"], linestyle=styles["DV"]["ls"],
               marker=styles["DV"]["marker"], lw=styles["DV"]["lw"], ms=styles["DV"]["ms"],
               label="Discrete Variable (DV)"),
    ]

    ax.legend(handles=legend_elements, loc="upper right", fontsize=22, frameon=True, facecolor="white")

    plt.tight_layout()
    plt.show()


# =============================================================================
# CV AND DV ENTROPY VS SYSTEM SIZE WITH INSET
# =============================================================================

def plot_CV_vs_DV_with_inset(p_values_all, dimensions, S_matrix, S_matrix_std,
                             S_DV_matrix, S_DV_matrix_std, p_ordered=0.30, p_disordered=0.55):
    """
    Plot the ordered phase on the main axis and the disordered phase
    in an inset.
    """

    fig, ax = plt.subplots(figsize=(11, 7))

    color_CV = "tab:purple"
    color_DV = "tab:orange"

    idx_ord = np.argmin(np.abs(p_values_all - p_ordered))
    idx_dis = np.argmin(np.abs(p_values_all - p_disordered))

    actual_p_ord = p_values_all[idx_ord]
    actual_p_dis = p_values_all[idx_dis]

    # Main plot: ordered phase
    ax.plot(dimensions, S_matrix[:, idx_ord], color=color_CV, ls="-", marker="o", lw=3.5, ms=8, zorder=5)
    ax.fill_between(dimensions, S_matrix[:, idx_ord] - S_matrix_std[:, idx_ord],
                    S_matrix[:, idx_ord] + S_matrix_std[:, idx_ord],
                    color=color_CV, alpha=0.2, edgecolor="none", zorder=4)

    ax.plot(dimensions, S_DV_matrix[:, idx_ord], color=color_DV, ls="--", marker="s", lw=3.5, ms=8, zorder=5)
    ax.fill_between(dimensions, S_DV_matrix[:, idx_ord] - S_DV_matrix_std[:, idx_ord],
                    S_DV_matrix[:, idx_ord] + S_DV_matrix_std[:, idx_ord],
                    color=color_DV, alpha=0.2, edgecolor="none", zorder=4)

    ax.set_xlabel(r"$L$", fontsize=26)
    ax.set_ylabel(r"$\langle S \rangle$", fontsize=26)
    ax.tick_params(axis="both", which="major", labelsize=22, width=2, length=6)
    ax.grid(alpha=0.3, linestyle="--")

    ax.text(0.52, 0.12, rf"Ordered Phase ($p = {actual_p_ord:.2f}$)", transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round,pad=0.4"))

    legend_elements = [
        Line2D([0], [0], color=color_CV, marker="o", ls="-", lw=3.5, ms=8, label="Continuous Variable (CV)"),
        Line2D([0], [0], color=color_DV, marker="s", ls="--", lw=3.5, ms=8, label="Discrete Variable (DV)"),
    ]

    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.52, 0.52),
              fontsize=18, frameon=True, facecolor="white", edgecolor="gray")

    # Inset: disordered phase
    axins = ax.inset_axes([0.05, 0.45, 0.45, 0.45])

    axins.plot(dimensions, S_matrix[:, idx_dis], color=color_CV, ls="-", marker="o", lw=2.5, ms=6, zorder=5)
    axins.fill_between(dimensions, S_matrix[:, idx_dis] - S_matrix_std[:, idx_dis],
                       S_matrix[:, idx_dis] + S_matrix_std[:, idx_dis],
                       color=color_CV, alpha=0.2, edgecolor="none", zorder=4)

    axins.plot(dimensions, S_DV_matrix[:, idx_dis], color=color_DV, ls="--", marker="s", lw=2.5, ms=6, zorder=5)
    axins.fill_between(dimensions, S_DV_matrix[:, idx_dis] - S_DV_matrix_std[:, idx_dis],
                       S_DV_matrix[:, idx_dis] + S_DV_matrix_std[:, idx_dis],
                       color=color_DV, alpha=0.2, edgecolor="none", zorder=4)

    axins.set_title(rf"Disordered Phase ($p = {actual_p_dis:.2f}$)", fontsize=18, fontweight="bold")
    axins.tick_params(axis="both", which="major", labelsize=16)
    axins.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.show()


# =============================================================================
# ENTROPY VS SYSTEM SIZE FOR DIFFERENT MEASUREMENT ANGLES
# =============================================================================

def plot_entropy_subplots(datasets, p_c, phi_labels, vmin=0.0, vmax=1.0):
    """
    Plot entropy against system size for three measurement angles.

    Each dataset must contain:
    (p_values, dimensions, S_matrix, S_matrix_std)
    """

    colors_lower = plt.cm.Blues_r(np.linspace(0.0, 0.65, 128))
    colors_upper = plt.cm.Reds(np.linspace(0.35, 1.0, 128))
    colors_combined = np.vstack((colors_lower, colors_upper))

    cmap_1 = mcolors.ListedColormap(colors_combined, name="BlueRed_Visible")
    norm_1 = mcolors.TwoSlopeNorm(vcenter=p_c, vmin=vmin, vmax=vmax)

    colors_blue = plt.cm.Blues_r(np.linspace(0.0, 0.55, 256))
    cmap_2 = mcolors.ListedColormap(colors_blue, name="Blues_Visible")
    norm_2 = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    num_lines_to_plot = 12

    for i, ax in enumerate(axes):
        p_values, dimensions, S_matrix, S_matrix_std = datasets[i]

        if i == 0:
            current_cmap, current_norm = cmap_1, norm_1
        else:
            current_cmap, current_norm = cmap_2, norm_2

        indices_to_plot = np.linspace(0, len(p_values) - 1, num_lines_to_plot, dtype=int)

        for idx in indices_to_plot:
            p = p_values[idx]
            means = S_matrix[:, idx]
            stds = S_matrix_std[:, idx]
            color = current_cmap(current_norm(p))

            ax.plot(dimensions, means, "-o", color=color, linewidth=2.5, markersize=6, alpha=0.9)
            ax.fill_between(dimensions, means - stds, means + stds,
                            color=color, alpha=0.2, edgecolor="none")

        ax.set_xlabel(r"$L$", fontsize=22)
        ax.set_title(phi_labels[i], fontsize=24, pad=15)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax.set_xlim(dimensions[0] - 2, dimensions[-1] + 2)
        ax.grid(alpha=0.4, linestyle="--")

        if i == 0:
            ax.set_ylabel(r"$\langle S \rangle$", fontsize=22)

    sm1 = cm.ScalarMappable(cmap=cmap_1, norm=norm_1)
    sm1.set_array([])

    cbar1 = fig.colorbar(sm1, ax=axes[0], pad=0.03, aspect=20)
    cbar1.set_label(r"$p$", fontsize=20, labelpad=10)
    cbar1.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar1.ax.tick_params(labelsize=16)
    cbar1.ax.invert_yaxis()
    cbar1.ax.axhline(p_c, color="black", linewidth=3, linestyle="-")
    cbar1.ax.text(-0.17, p_c, r"$p_c$", color="black", va="center", ha="right", fontsize=18)

    sm2 = cm.ScalarMappable(cmap=cmap_2, norm=norm_2)
    sm2.set_array([])

    cbar2 = fig.colorbar(sm2, ax=axes[2], pad=0.03, aspect=20)
    cbar2.set_label(r"$p$", fontsize=20, labelpad=10)
    cbar2.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar2.ax.tick_params(labelsize=16)
    cbar2.ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


# =============================================================================
# ENTROPY VS PROBABILITY FOR DIFFERENT MEASUREMENT ANGLES
# =============================================================================

def plot_entropy_vs_p_fixed_L(datasets, target_L, phi_labels):
    """
    Plot entropy against measurement probability for a fixed system size.

    Each dataset must contain:
    (p_values, dimensions, S_matrix, S_matrix_std)
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#08519c", "#cb181d", "#238b45"]
    markers = ["o", "s", "^"]

    for i, data in enumerate(datasets):
        p_values, dimensions, S_matrix, S_matrix_std = data

        if target_L not in dimensions:
            raise ValueError(f"Target L={target_L} not found in dimensions array {dimensions}")

        l_idx = np.where(dimensions == target_L)[0][0]
        means = S_matrix[l_idx, :]
        stds = S_matrix_std[l_idx, :]

        ax.plot(p_values, means, marker=markers[i], color=colors[i], linewidth=2.5,
                markersize=7, label=phi_labels[i], alpha=0.9, markevery=5)

        ax.fill_between(p_values, means - stds, means + stds,
                        color=colors[i], alpha=0.15, edgecolor="none")

    ax.set_xlabel(r"$p$", fontsize=22)
    ax.set_ylabel(r"$\langle S \rangle/L$", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.yaxis.get_offset_text().set_fontsize(16)
    ax.set_xlim(p_values[0], p_values[-1])
    ax.legend(fontsize=18, loc="best", frameon=True, edgecolor="black", shadow=True)
    ax.grid(alpha=0.4, linestyle="--")

    plt.tight_layout()
    plt.show()