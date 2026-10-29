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

def plot_vertex_deletion_panels(
    p_values, dimensions, S_matrix, S_matrix_std,
    S_DV_matrix, S_DV_matrix_std, *, n_samples,
    p_ordered=0.35, p_disordered=0.55, p_c=0.4070,
    selected_sizes=None, fit_min_L=None,
):
    """Compare raw entropy scaling and entropy per system size with SEM bands.

    Inputs have shape (number of sizes, number of probabilities), as returned
    by initialize_data_q. n_samples is the number of independent realisations,
    either a scalar or an array of the same shape as the entropy matrices.
    Supplied sample standard deviations are converted to SEM = SD / sqrt(N).
    Shaded bands are mean +/- one SEM, not 95% confidence intervals.
    Panel (c) divides both means and SEMs by L, without amplitude rescaling.
    Panel (a) uses unweighted least squares S = slope * L + intercept;
    fit_min_L optionally restricts the fitted sizes. No fit is imposed in (b).
    Requested probabilities must lie within the data range; nearest stored
    values are used and returned. Sizes must match exactly; by default only the largest size is shown in (c).

    Returns (fig, axes, details), without calling show or saving files.
    """
    import warnings

    p = np.asarray(p_values, dtype=float)
    sizes = np.asarray(dimensions, dtype=float)
    if (p.ndim != 1 or sizes.ndim != 1 or len(p) < 2 or len(sizes) < 2
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(sizes))
            or np.any((p < 0) | (p > 1)) or np.any(sizes <= 0)
            or len(np.unique(p)) != len(p) or len(np.unique(sizes)) != len(sizes)):
        raise ValueError('Use unique finite probabilities in [0, 1] and positive sizes (at least two each).')
    arrays = [np.asarray(a, dtype=float) for a in
              (S_matrix, S_matrix_std, S_DV_matrix, S_DV_matrix_std)]
    if any(a.shape != (len(sizes), len(p)) or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('Entropy and standard-deviation arrays must be finite with shape (len(dimensions), len(p_values)).')
    if any(np.any(a < 0) for a in arrays[1::2]):
        raise ValueError('Standard deviations cannot be negative.')
    counts = np.asarray(n_samples, dtype=float)
    if (counts.shape not in ((), arrays[0].shape)
            or not np.all(np.isfinite(counts)) or np.any(counts < 2)
            or np.any(counts != np.floor(counts))):
        raise ValueError('n_samples must contain integer counts >= 2, as a scalar or entropy-shaped array.')
    arrays[1] = arrays[1] / np.sqrt(counts)
    arrays[3] = arrays[3] / np.sqrt(counts)
    pi, li = np.argsort(p), np.argsort(sizes)
    p, sizes = p[pi], sizes[li]
    cv, cv_sem, dv, dv_sem = [a[np.ix_(li, pi)] for a in arrays]
    indices = []
    for target in (p_ordered, p_disordered):
        if not np.isfinite(target) or not p[0] <= target <= p[-1]:
            raise ValueError(f'Requested p={target} lies outside the available range [{p[0]}, {p[-1]}].')
        idx = int(np.argmin(abs(p - target)))
        if not np.isclose(p[idx], target, rtol=0, atol=1e-10):
            warnings.warn(f'Requested p={target:g}; using nearest stored p={p[idx]:.8g}.', stacklevel=2)
        indices.append(idx)
    if not p[indices[0]] < p_c < p[indices[1]]:
        raise ValueError('Selected probabilities must bracket p_c in the specified regimes.')
    if selected_sizes is None:
        selected = np.array([len(sizes) - 1])
    else:
        requested = np.asarray(selected_sizes, dtype=float)
        if requested.ndim != 1 or len(requested) < 1 or len(np.unique(requested)) != len(requested):
            raise ValueError('Select at least one system size, without duplicates.')
        if not np.all(np.isin(requested, sizes)):
            raise ValueError(f'Selected sizes must be present in {sizes.tolist()}.')
        selected = np.array([np.flatnonzero(sizes == L)[0] for L in sorted(requested)])
    fit_mask = np.ones(len(sizes), dtype=bool) if fit_min_L is None else sizes >= fit_min_L
    if fit_mask.sum() < 2:
        raise ValueError('At least two sizes are needed for a linear fit.')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout='constrained', sharey=False)
    details = {'uncertainty': 'SEM', 'p_ordered': float(p[indices[0]]),
               'p_disordered': float(p[indices[1]]),
               'selected_sizes': sizes[selected].tolist(),
               'fit_sizes': sizes[fit_mask].tolist(), 'fits': {}}
    series = [('DV', dv, dv_sem, 'tab:orange', 's'),
              ('CV', cv, cv_sem, 'tab:purple', 'o')]
    for panel, (ax, idx) in enumerate(zip(axes[:2], indices)):
        for name, means, sems, color, marker in series:
            ax.fill_between(sizes, means[:, idx] - sems[:, idx],
                            means[:, idx] + sems[:, idx], color=color,
                            alpha=0.18, linewidth=0)
            ax.plot(sizes, means[:, idx], color=color, marker=marker, ms=4,
                    lw=1.2, ls='none' if panel == 0 else '-', label=name)
            if panel == 0:
                x = sizes[fit_mask]
                slope, intercept = np.polyfit(x, means[fit_mask, idx], 1)
                ax.plot(x, slope * x + intercept, '--', color=color, lw=1.5)
                details['fits'][name] = {'slope': float(slope), 'intercept': float(intercept)}
        regime = 'Percolating regime' if panel == 0 else 'Non-percolating regime'
        # ax.set_title(f'({chr(97 + panel)}) {regime}\n$p = {p[idx]:.5g}$', fontsize=12)
        ax.set_title(f'{regime} \n$p = {p[idx]:.2g}$', fontsize=12)
        ax.set_xlabel(r'$L$')
        ax.set_ylabel(r'$\langle S\rangle$')
        ax.legend(frameon=False)
    # axes[0].text(0.04, 0.96, r'Dashed: fits $aL+b$', transform=axes[0].transAxes,
    #              va='top', fontsize=9)
    # Draw dashed CV over solid DV; offset markers so coincident curves remain visible.
    step = max(2, len(p) // 12)
    for i in selected:
        L = sizes[i]
        for name, means, sems, scale, color, ls, marker, offset in (
            ('DV', dv, dv_sem, L, 'tab:orange', '-', 's', 0),
            ('CV', cv, cv_sem, L, 'tab:purple', '--', 'o', step // 2),
        ):
            y, err = means[i] / scale, sems[i] / scale
            label = name if len(selected) == 1 else rf'{name}, $L={L:g}$'
            axes[2].plot(p, y, ls=ls, color=color, marker=marker, ms=4,
                         markerfacecolor='white', lw=1.7,
                         markevery=(offset, step), label=label)
            axes[2].fill_between(p, y - err, y + err, color=color, alpha=0.08)
    axes[2].axvline(p_c, color='0.4', ls=':', lw=1.2, label=rf'$p_c={p_c:g}$')
    axes[2].set_title('Across the transition' +
                      (f'\n$L={sizes[selected[0]]:g}$' if len(selected) == 1 else ''), fontsize=12)
    axes[2].set_xlabel(r'$p$')
    axes[2].set_ylabel(r'$\langle S\rangle/L$')
    axes[2].legend(loc='upper right', frameon=False, fontsize=9)
    for ax in axes:
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
        ax.title.set_fontsize(20)
        ax.tick_params(axis="both", labelsize=18)

        # Scientific-notation labels, if present
        ax.xaxis.get_offset_text().set_fontsize(18)
        ax.yaxis.get_offset_text().set_fontsize(18)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(17)

        # Includes the “Dashed: fits aL+b” annotation
        for text in ax.texts:
            text.set_fontsize(16)
    return fig, axes, details


def export_vertex_deletion_panels(data_file, output_dir, *, r=1.0,
                                  p_ordered=0.35, p_disordered=0.55,
                                  p_c=0.4070, figsize=(6.0, 4.4)):
    """Export three separate PDF panels, with mean +/- SEM bands.

    Counts are taken from actual samples, assuming independent realisations.
    The existing loader supplies sample SD (ddof=1), converted here to SEM.
    Uses w=1 data, where the loader's spectral convention is unambiguous.
    Panel (a) fits aL+b by unweighted least squares over all sizes.
    Panel (c) uses the largest size and S/L for BOTH encodings, with no alpha
    rescaling. Uses spacious landscape panels with automatic layout and
    tightly cropped PDF output. No fixed margins or displaced axis labels.
    Returns a dictionary of figure objects, fit details and sample counts.
    """
    from pathlib import Path
    import pickle
    import warnings
    from .observables import initialize_data_q

    data_file, output_dir = Path(data_file), Path(output_dir)
    with data_file.open('rb') as handle:
        raw = pickle.load(handle)
    if raw.get('weight') != 1.0:
        raise ValueError('Use w=1 data; the loader spectral convention needs review for other weights.')
    p, sizes = np.asarray(raw['p_values']), np.asarray(raw['L_values'])
    counts = np.array([[len(raw['results'][prob][L]['dv_entropies'])
                        for prob in p] for L in sizes])
    cv_counts = np.array([[len(raw['results'][prob][L]['eigenvalues'])
                           for prob in p] for L in sizes])
    if not np.array_equal(counts, cv_counts) or np.any(counts < 2):
        raise ValueError('CV and DV need matching sample counts >= 2.')
    del raw
    _, _, _, _, cv, cv_sd, dv, dv_sd = initialize_data_q(data_file, r=r)
    cv_sem, dv_sem = cv_sd / np.sqrt(counts), dv_sd / np.sqrt(counts)
    pi, li = np.argsort(p), np.argsort(sizes)
    p, sizes = p[pi], sizes[li]
    cv, cv_sem, dv, dv_sem, counts = [a[np.ix_(li, pi)]
                                     for a in (cv, cv_sem, dv, dv_sem, counts)]
    if len(sizes) < 2 or not all(np.all(np.isfinite(a)) for a in (cv, cv_sem, dv, dv_sem)):
        raise ValueError('Need finite entropy statistics and at least two sizes.')
    idxs = []
    for target in (p_ordered, p_disordered):
        if not p[0] <= target <= p[-1]:
            raise ValueError(f'p={target} is outside [{p[0]}, {p[-1]}].')
        idx = int(np.argmin(abs(p-target)))
        if not np.isclose(p[idx], target, atol=1e-10, rtol=0):
            warnings.warn(f'Requested p={target}; using stored p={p[idx]:.8g}.', stacklevel=2)
        idxs.append(idx)
    if not p[idxs[0]] < p_c < p[idxs[1]]:
        raise ValueError('The selected probabilities must bracket p_c.')
    output_dir.mkdir(parents=True, exist_ok=True)
    figures, fits = {}, {}
    styles = [('DV', dv, dv_sem, 'tab:orange', 's'),
              ('CV', cv, cv_sem, 'tab:purple', 'o')]
    settings = {'font.size': 16, 'axes.labelsize': 22, 'axes.titlesize': 20,
                'xtick.labelsize': 18, 'ytick.labelsize': 18,
                'legend.fontsize': 16, 'pdf.fonttype': 42}
    with plt.rc_context(settings):
        for panel, (name, idx) in enumerate(zip(
                ('vertex_deletion_a', 'vertex_deletion_b'), idxs)):
            fig, ax = plt.subplots(figsize=figsize, layout='constrained')
            for model, means, errors, color, marker in styles:
                ax.fill_between(sizes, means[:,idx]-errors[:,idx],
                                means[:,idx]+errors[:,idx], color=color, alpha=.20, lw=0)
                ax.plot(sizes, means[:,idx], color=color, marker=marker,
                        ms=4.5, lw=1.3, ls='none' if panel == 0 else '-', label=model)
                if panel == 0:
                    a, b = np.polyfit(sizes, means[:,idx], 1)
                    ax.plot(sizes, a*sizes+b, color=color, ls='--', lw=1.4)
                    fits[model] = {'slope': float(a), 'intercept': float(b)}
            ax.set(xlabel=r'$L$', ylabel=r'$\langle S\rangle$ (bits)',
                   title=rf'$p={p[idx]:.5f}$')
            ax.legend(loc='upper left' if panel == 0 else 'upper right', frameon=False)
            figures[name] = fig
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        L = sizes[-1]
        for model, means, errors, color, marker in styles:
            y, e = means[-1]/L, errors[-1]/L
            ax.fill_between(p, y-e, y+e, color=color, alpha=.20, lw=0)
            ax.plot(p, y, color=color, marker=marker, ms=4.5,
                    markevery=max(1, len(p)//12), lw=1.5,
                    ls='-' if model == 'DV' else '--', label=model)
        ax.axvline(p_c, color='0.4', ls=':', lw=1.1, label=rf'$p_c={p_c:g}$')
        ax.set(xlabel=r'$p$', ylabel=r'$\langle S\rangle/L$ (bits)', title=rf'$L={L:g}$')
        ax.legend(loc='upper right', frameon=False)
        figures['vertex_deletion_c'] = fig
        for name, fig in figures.items():
            ax = fig.axes[0]
            ax.grid(alpha=.16, lw=.5)
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(direction='out', width=.8, length=3)
            fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight', pad_inches=0.04)
    # Return metadata after all three figures have been saved.
    # Keep this inside the function, outside the plotting loops/context.
    return {
        'figures': figures,
        'fits': fits,
        'counts': counts,
        'p_ordered': float(p[idxs[0]]),
        'p_disordered': float(p[idxs[1]]),
        'L': float(sizes[-1]),
        'cv_sem': cv_sem,
        'dv_sem': dv_sem,
    }
