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



# =============================================================================
# ENTROPY VS SYSTEM SIZE FOR CRITICAL PROBABILITY
# =============================================================================

def export_vertex_deletion_critical(data_file, output_dir, *, r=1.0,
                                     p_c=0.4070, model='CV', fit_L_min=None,
                                     allow_nearest=False, figsize=(6.0, 4.4)):
    """Export log-log entropy and compensated entropy PDFs.

    Fit log(mean S) = log(A) + gamma*log(L) by unweighted least
    squares. Independently fit log(A_ref) with gamma fixed at 43/48,
    using the same sizes and log-space objective. Bands show sample
    SEM, assuming independent realisations. A plateau is a diagnostic,
    not proof of asymptotic scaling. Exact p is required unless explicitly
    overridden; the actual stored p is always shown.
    """
    from pathlib import Path
    import pickle
    import warnings
    from .observables import initialize_data_q

    model = model.upper()
    if model not in ('CV', 'DV'):
        raise ValueError("model must be 'CV' or 'DV'.")
    with open(data_file, 'rb') as handle:
        raw = pickle.load(handle)
    if raw.get('weight') != 1.0:
        raise ValueError('This exporter currently requires w=1 data.')
    probabilities = np.asarray(raw['p_values'])
    idx = int(np.argmin(abs(probabilities-p_c)))
    p_actual = float(probabilities[idx])
    exact = bool(np.isclose(p_actual, p_c, atol=1e-10, rtol=0))
    if not exact:
        message = (f'No samples at p_c={p_c:g}; nearest stored p={p_actual:.8f}. '
                   'Off-critical data cannot establish critical asymptotic scaling.')
        if not allow_nearest:
            raise ValueError(message + ' Set allow_nearest=True for a labelled preview.')
        warnings.warn(message, stacklevel=2)
    sizes = np.asarray(raw['L_values'], dtype=float)
    key = 'eigenvalues' if model == 'CV' else 'dv_entropies'
    counts = np.array([len(raw['results'][probabilities[idx]][L][key]) for L in sizes])
    del raw
    if np.any(counts < 2):
        raise ValueError('At least two independent samples per point are required.')
    _, _, _, _, cv, cv_sd, dv, dv_sd = initialize_data_q(data_file, r=r)
    mean = (cv if model == 'CV' else dv)[:, idx]
    sem = (cv_sd if model == 'CV' else dv_sd)[:, idx] / np.sqrt(counts)
    order = np.argsort(sizes)
    sizes, mean, sem, counts = [a[order] for a in (sizes, mean, sem, counts)]
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sem)) or np.any(mean <= 0) or np.any(sizes <= 0):
        raise ValueError('Logarithmic plots require finite positive sizes and mean entropies.')
    mask = np.ones(len(sizes), dtype=bool) if fit_L_min is None else sizes >= fit_L_min
    if mask.sum() < 3:
        raise ValueError('Select at least three sizes for the fit.')
    exponent = 43/48
    x, y = np.log(sizes[mask]), np.log(mean[mask])
    gamma, log_amplitude = np.polyfit(x, y, 1)
    amplitude = np.exp(log_amplitude)
    reference_amplitude = np.exp(np.mean(y - exponent*x))
    line_L = np.geomspace(sizes[mask].min(), sizes[mask].max(), 300)
    title = rf'$p={p_actual:.5f}$'
    if not exact:
        title += rf' (target $p_c={p_c:.4f}$)'
    color = 'tab:purple' if model == 'CV' else 'tab:orange'
    settings = {'font.size': 16, 'axes.labelsize': 22, 'axes.titlesize': 18,
                'xtick.labelsize': 18, 'ytick.labelsize': 18,
                'legend.fontsize': 13, 'pdf.fonttype': 42}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(settings):
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Omit nonpositive lower bounds rather than inventing a log-space floor.
        lower = np.where(mean-sem > 0, mean-sem, np.nan)
        ax.fill_between(sizes, lower, mean+sem, color=color, alpha=.20, lw=0)
        ax.plot(sizes, mean, 'o-', color=color, ms=5, lw=1.4, label=model)
        ax.plot(line_L, amplitude*line_L**gamma, '--', color='tab:blue',
                label=rf'Fit: $\gamma={gamma:.4f}$', lw=1.8)
        ax.plot(line_L, reference_amplitude*line_L**exponent, ':', color='black',
                label=r'Reference: $\gamma=43/48$', lw=2)
        ax.set(xlabel=r'$L$', ylabel=r'$\langle S\rangle$ (bits)', title=title)
        ax.legend(frameon=False)
        fig_ratio, ar = plt.subplots(figsize=figsize, layout='constrained')
        ratio, error = mean/sizes**exponent, sem/sizes**exponent
        ar.fill_between(sizes, ratio-error, ratio+error, color=color, alpha=.20, lw=0)
        ar.plot(sizes, ratio, 'o-', color=color, ms=5, lw=1.4, label=model)
        ar.plot(line_L, np.full_like(line_L, reference_amplitude), ':', color='black',
                lw=2, label=r'Fitted $A_{43/48}$')
        ar.set(xlabel=r'$L$', ylabel=r'$\langle S\rangle/L^{43/48}$', title=title)
        ar.legend(frameon=False)
        figures = {'critical_loglog': fig, 'critical_compensated': fig_ratio}
        paths = {}
        for name, figure in figures.items():
            axis = figure.axes[0]
            axis.grid(alpha=.16, lw=.5, which='both')
            axis.spines[['top', 'right']].set_visible(False)
            path = output_dir / f'vertex_deletion_{model.lower()}_{name}.pdf'
            figure.savefig(path, bbox_inches='tight', pad_inches=.04)
            paths[name] = path
    return {'figures': figures, 'paths': paths, 'p_actual': p_actual,
            'exact_p': exact, 'gamma': float(gamma), 'amplitude': float(amplitude),
            'reference_amplitude': float(reference_amplitude), 'counts': counts,
            'sizes': sizes, 'mean': mean, 'sem': sem, 'fit_mask': mask}


def plot_vertex_deletion_critical(
    data_file,
    output_dir,
    *,
    r=1.0,
    p_c=0.4070,
    model="CV",
    fit_L_min=None,
    allow_nearest=False,
    figsize=(6.0, 4.4),
):
    """Plot critical entropy scaling and compensated entropy.

    Fits:
        S = A * L**gamma       (free exponent)
        S = A_ref * L**(43/48) (fixed exponent)

    Both fits minimise unweighted squared residuals in log space,
    over the same selected sizes. Bands show mean ± one SEM.
    """
    import pickle
    import warnings
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    from .observables import initialize_data_q

    model = model.upper()
    if model not in ("CV", "DV"):
        raise ValueError("model must be 'CV' or 'DV'.")

    # Read only the sizes actually stored at the selected probability.
    with Path(data_file).open("rb") as handle:
        raw = pickle.load(handle)

    if raw.get("weight") != 1.0:
        raise ValueError("This function currently requires w=1 data.")

    p_key = min(raw["results"], key=lambda p: abs(float(p) - p_c))
    p_actual = float(p_key)
    exact_p = bool(np.isclose(p_actual, p_c, atol=1e-10, rtol=0))

    if not exact_p:
        message = (
            f"No data at p={p_c:.8f}; nearest stored p={p_actual:.8f}."
        )
        if not allow_nearest:
            raise ValueError(message)
        warnings.warn(
            message + " This is an off-critical preview.",
            stacklevel=2,
        )

    results_at_p = raw["results"][p_key]
    size_keys = sorted(results_at_p, key=float)
    sizes = np.asarray(size_keys, dtype=float)

    missing = sorted(
        set(map(float, raw["L_values"])) - set(sizes)
    )
    if missing:
        warnings.warn(
            f"Skipping sizes without stored results: {missing}",
            stacklevel=2,
        )

    means, sems, sample_counts = [], [], []

    for L in size_keys:
        result = results_at_p[L]

        if model == "DV":
            samples = np.asarray(result["dv_entropies"], dtype=float)

        else:
            # Vertex-deletion spectra use the same convention as
            # initialize_data_q: nu = sqrt(1 + exp(4r) * eigenvalue**2).
            samples = []

            for spectrum in result["eigenvalues"]:
                spectrum = np.asarray(spectrum, dtype=float)
                nu = np.sqrt(1.0 + np.exp(4.0 * r) * spectrum**2)
                occupation = (nu - 1.0) / 2.0

                # Stable evaluation of:
                # (n+1) log2(n+1) - n log2(n), including n=0.
                entropy = np.log1p(occupation)
                positive = occupation > 0
                entropy[positive] += (
                    occupation[positive]
                    * np.log1p(1.0 / occupation[positive])
                )

                # An empty spectrum contributes zero entropy.
                samples.append(float(entropy.sum() / np.log(2.0)))

            samples = np.asarray(samples, dtype=float)

        if len(samples) < 2 or not np.all(np.isfinite(samples)):
            raise ValueError(
                f"L={L}: need at least two finite entropy samples."
            )

        sample_counts.append(len(samples))
        means.append(samples.mean())
        sems.append(samples.std(ddof=1) / np.sqrt(len(samples)))

    mean = np.asarray(means)
    sem = np.asarray(sems)
    counts = np.asarray(sample_counts)

    del raw

    order = np.argsort(sizes)
    sizes, mean, sem, counts = [
        values[order] for values in (sizes, mean, sem, counts)
    ]

    if (
        not all(np.all(np.isfinite(v)) for v in (sizes, mean, sem))
        or np.any(sizes <= 0)
        or np.any(mean <= 0)
        or np.any(sem < 0)
    ):
        raise ValueError(
            "Logarithmic plots require positive sizes and mean entropies, "
            "with finite, nonnegative SEM."
        )

    fit_mask = (
        np.ones(len(sizes), dtype=bool)
        if fit_L_min is None
        else sizes >= fit_L_min
    )

    if fit_mask.sum() < 3:
        raise ValueError("Select at least three sizes for the fits.")

    # Independent amplitude fits using the same log-space objective.
    exponent = 43 / 48
    log_L = np.log(sizes[fit_mask])
    log_S = np.log(mean[fit_mask])

    gamma, log_amplitude = np.polyfit(log_L, log_S, 1)
    amplitude = float(np.exp(log_amplitude))
    reference_amplitude = float(
        np.exp(np.mean(log_S - exponent * log_L))
    )

    fit_sizes = np.geomspace(
        sizes[fit_mask].min(), sizes[fit_mask].max(), 300
    )

    color = "tab:purple" if model == "CV" else "tab:orange"
    title = rf"$p={p_actual:.4f}$"
    if not exact_p:
        title = rf"$p={p_actual:.6f}$ (target $p_c={p_c:.4f}$)"

    settings = {
        "font.size": 16,
        "axes.labelsize": 22,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "pdf.fonttype": 42,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(settings):
        # Figure 1: log–log entropy.
        fig_log, ax = plt.subplots(
            figsize=figsize, layout="constrained"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        # A nonpositive lower bound cannot be drawn on a log axis.
        lower = np.where(mean - sem > 0, mean - sem, np.nan)
        if np.any(mean - sem <= 0):
            warnings.warn(
                "Some lower SEM bounds are nonpositive and are omitted "
                "from the logarithmic shading.",
                stacklevel=2,
            )

        ax.fill_between(
            sizes, lower, mean + sem,
            color=color, alpha=0.30, linewidth=0,
        )
        ax.plot(
            sizes, mean, "o", color=color,
            markersize=5, label=model,
        )
        ax.plot(
            fit_sizes, amplitude * fit_sizes**gamma,
            "--", color="tab:blue", linewidth=1.8,
            label=rf"Fit: $\gamma={gamma:.4f}$",
        )
        ax.plot(
            fit_sizes, reference_amplitude * fit_sizes**exponent,
            ":", color="black", linewidth=2,
            label=r"Reference: $\gamma=43/48$",
        )
        ax.set(
            xlabel=r"$L$",
            ylabel=r"$\langle S\rangle$",
            title=title,
        )

        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

        # Keep the logarithmic axis, but show fewer labels.
        ax.xaxis.set_major_locator(FixedLocator([100, 200, 400, 750]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())

        # Keep the logarithmic y-axis, displaying 100, 200, ... as 1, 2, ...
        ax.yaxis.set_major_locator(FixedLocator([100, 200, 300, 400, 500]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 100:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())

        # Common factor above the y-axis.
        ax.text(
            0, 1.02, r"$\times 10^{2}$",
            transform=ax.transAxes,
            ha="left", va="bottom", fontsize=18,
        )

        # Figure 2: compensated entropy.
        fig_comp, ax_comp = plt.subplots(
            figsize=figsize, layout="constrained"
        )
        compensated = mean / sizes**exponent
        compensated_sem = sem / sizes**exponent

        ax_comp.fill_between(
            sizes,
            compensated - compensated_sem,
            compensated + compensated_sem,
            color=color, alpha=0.30, linewidth=0,
        )
        ax_comp.plot(
            sizes, compensated, "o-", color=color,
            markersize=5, label=model,
        )
        ax_comp.plot(
            fit_sizes,
            np.full_like(fit_sizes, reference_amplitude),
            ":", color="black", linewidth=2,
            label=r"Fit: $\langle S\rangle=A L^{43/48}$",
        )
        ax_comp.set(
            xlabel=r"$L$",
            ylabel=r"$\langle S\rangle/L^{43/48}$",
            title=title,
        )

        ax_comp.set_ylim(0.9, 1.3)
        ax_comp.set_yticks([0.9, 1.0, 1.1, 1.2, 1.3])

        figures = {
            "critical_loglog": fig_log,
            "critical_compensated": fig_comp,
        }
        paths = {}

        for name, figure in figures.items():
            axis = figure.axes[0]
            axis.legend(frameon=False)
            axis.grid(alpha=0.16, linewidth=0.5, which="both")
            axis.spines[["top", "right"]].set_visible(False)

            paths[name] = (
                output_dir / f"vertex_deletion_{model.lower()}_{name}.pdf"
            )
            figure.savefig(
                paths[name], bbox_inches="tight", pad_inches=0.04
            )

    return {
        "figures": figures,
        "paths": paths,
        "p_actual": p_actual,
        "exact_p": exact_p,
        "gamma": float(gamma),
        "amplitude": amplitude,
        "reference_amplitude": reference_amplitude,
        "sizes": sizes,
        "mean": mean,
        "sem": sem,
        "counts": counts,
        "fit_mask": fit_mask,
    }



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




def plot_homodyne__panels(data_files, output_dir, *, target_L=70,
                                  target_p=.6, p_c_del=.4070,
                                  figsize=(6.0, 4.6), export_appendix=True,
                                  angle_datasets=None, sample_counts=None):
    """Export two separate panel PDFs and optionally a probability-scan appendix.

    Supply three files in angle order 0, pi/4, pi/2. All statistics are
    computed over raw realisations. Panel (a) divides both mean and SEM by L;
    panel (b) uses raw S at the nearest common stored probability above p_c_del.
    The nonzero-angle fits are unweighted least squares S=aL+b over all sizes.
    No interpolation or smoothing is applied to measured curves.
    Alternatively pass angle_datasets from initialize_data(meas="general") and
    sample_counts (a shared scalar or three arrays shaped like the entropies).
    In that mode data_files provide names only; no files are reloaded.
    """
    from pathlib import Path
    import warnings
    if len(angle_datasets) != 3 or len(data_files) != 3:
        raise ValueError('Provide three datasets in order 0, pi/4, pi/2.')
    counts_in = [sample_counts]*3 if np.isscalar(sample_counts) else sample_counts
    datasets = []
    for source, values, count in zip(data_files, angle_datasets, counts_in):
        p, L, mean, sd = [np.asarray(a) for a in values]
        shape = (len(L), len(p))
        counts = np.broadcast_to(count, shape)
        if mean.shape != shape or sd.shape != shape or np.any(counts < 2):
            raise ValueError('Invalid array shapes or sample counts.')
        if not all(np.all(np.isfinite(a)) for a in (mean, sd, counts)) or np.any(sd < 0):
            raise ValueError('Invalid entropy statistics.')
        order = np.ix_(np.argsort(L), np.argsort(p))
        datasets.append(dict(p=np.sort(p), L=np.sort(L), mean=mean[order],
                             sem=(sd/np.sqrt(counts))[order], counts=counts[order], source=str(source)))
    common = datasets[0]['p']
    for d in datasets[1:]: common = np.intersect1d(common, d['p'])
    if not len(common): raise ValueError('No common stored probability.')
    actual = float(common[np.argmin(abs(common-target_p))])
    if actual <= p_c_del: raise ValueError('Choose a probability above p_c.')
    if not np.isclose(actual, target_p):
        warnings.warn(f'Using stored p={actual:.8f} for requested p={target_p}.', stacklevel=2)
    colors = ['#0072B2', '#D55E00', '#009E73']
    markers = ['o', 's', '^']
    labels = [r'$\varphi=0$', r'$\varphi=\pi/4$', r'$\varphi=\pi/2$']
    angles = [0., np.pi/4, np.pi/2]
    settings = {'font.size':16, 'axes.labelsize':22, 'axes.titlesize':20,
                'xtick.labelsize':18, 'ytick.labelsize':18, 'legend.fontsize':16,
                'pdf.fonttype':42}
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fits, figures, paths = {}, {}, {}
    with plt.rc_context(settings):
        fa, aa = plt.subplots(figsize=figsize, layout='constrained')
        fb, ab = plt.subplots(figsize=figsize, layout='constrained')
        for k, (d, color, marker, label) in enumerate(zip(datasets, colors, markers, labels)):
            p,L,mean,sem = [d[key] for key in ('p','L','mean','sem')]
            if target_L not in L: raise ValueError(f'L={target_L} missing.')
            i = np.where(L==target_L)[0][0]; j=np.where(p==actual)[0][0]
            aa.fill_between(p,(mean[i]-sem[i])/target_L,(mean[i]+sem[i])/target_L,
                            color=color,alpha=.30,lw=0)
            aa.plot(p,mean[i]/target_L,color=color,marker=marker,
                    markevery=max(1,int(np.ceil(len(p)/12))),ms=5,lw=1.4,label=label)
            ab.fill_between(L,mean[:,j]-sem[:,j],mean[:,j]+sem[:,j],color=color,alpha=.30,lw=0)
            ab.plot(L,mean[:,j],color=color,marker=marker,ms=5,lw=1.1,label=label)
            if k:
                a,b=np.polyfit(L,mean[:,j],1)
                ab.plot(L,a*L+b,color=color,ls='--',lw=1.7,label='_nolegend_')
                fits[angles[k]]=dict(slope=float(a),intercept=float(b))
        aa.axvline(p_c_del,color='0.4',ls=':',lw=.9)
        aa.text(p_c_del+.012,.96,r'$p_c$',transform=aa.get_xaxis_transform(),va='top',fontsize=18)
        # aa.set(xlabel=r'$p$',ylabel=r'$\langle S\rangle/L$',title=rf'$L={target_L:g}$')
        # ab.set(xlabel=r'$L$',ylabel=r'$\langle S\rangle$',title=rf'$p={actual:.5f}$')
        aa.set(xlabel=r'$p$',ylabel=r'$\langle S\rangle/L$')
        ab.set(xlabel=r'$L$',ylabel=r'$\langle S\rangle$')  
        for key,fig,ax in [('a',fa,aa),('b',fb,ab)]:
            ax.legend(frameon=False);ax.grid(alpha=.16,lw=.5)
            ax.spines[['top','right']].set_visible(False)
            figures[key]=fig;paths[key]=output_dir/f'homodyne_panel_{key}.pdf'
            fig.savefig(paths[key],bbox_inches='tight',pad_inches=.04)
        if export_appendix:
            fig,axs=plt.subplots(1,3,figsize=(17,4.8),layout='constrained')
            norm=mcolors.Normalize(min(d['p'].min() for d in datasets),max(d['p'].max() for d in datasets))
            cmap=plt.get_cmap('viridis')
            for ax,d,label in zip(axs,datasets,labels):
                for j,p in enumerate(d['p']):
                    y,e=d['mean'][:,j],d['sem'][:,j]
                    ax.plot(d['L'],y,color=cmap(norm(p)),lw=1)
                    ax.fill_between(d['L'],y-e,y+e,color=cmap(norm(p)),alpha=.15,lw=0)
                ax.set(xlabel=r'$L$',title=label)
            axs[0].set_ylabel(r'$\langle S\rangle$ (bits)')
            fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=list(axs),label=r'$p$')
            figures['appendix']=fig;paths['appendix']=output_dir/'homodyne_appendix_probability_scan.pdf'
            fig.savefig(paths['appendix'],bbox_inches='tight',pad_inches=.04)
    return dict(figures=figures,paths=paths,datasets=datasets,fits=fits,p_actual=actual,L=target_L)


