"""Pauli-X/Y result loading and paper plots. Arrays have shape (p, L).

Place in src/monitored_cluster_states/pauli_plotting.py.
Loads trusted pickle files only. SEM uses actual stored sample counts.
"""
from pathlib import Path
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

STYLE = {
    "font.size": 16, "axes.labelsize": 22, "axes.titlesize": 18,
    "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 13,
    "lines.linewidth": 1.8, "lines.markersize": 5, "pdf.fonttype": 42,
}

def load_pauli_results(data_dir, file_name=None, *, basis="y"):
    """Load raw realisations or saved mean/variance/SEM; missing points stay NaN."""
    data_dir = Path(data_dir)
    if file_name is None:
        paths = sorted(data_dir.glob(f"pauli_{basis.lower()}_results*.pkl"))
        if len(paths) != 1:
            available = "\n".join(p.name for p in paths) or "(none found)"
            raise FileNotFoundError(f"Set FILE_NAME in {data_dir}. Available files:\n{available}")
        data_file = paths[0]
    else:
        data_file = data_dir / file_name
    with data_file.open("rb") as stream:
        data = pickle.load(stream)
    results = data["results"]
    if not results:
        raise ValueError("The results dictionary is empty.")
    probabilities = np.array(sorted(results), dtype=float)
    sizes = np.array(sorted({int(L) for entries in results.values() for L in entries}), dtype=int)
    S = np.full((len(probabilities), len(sizes)), np.nan)
    SEM = np.full_like(S, np.nan)
    counts = np.zeros_like(S, dtype=int)

    for i, p in enumerate(probabilities):
        for j, L in enumerate(sizes):
            entry = results[p].get(L)
            if entry is None:
                continue  # Missing sizes are gaps, never artificial zero entropies.
            if "dv_entropies" in entry:
                samples = np.asarray(entry["dv_entropies"], dtype=float)
                n = len(samples)
                if samples.ndim != 1 or n == 0 or not np.all(np.isfinite(samples)):
                    raise ValueError(f"Invalid entropy samples at p={p}, L={L}.")
                if entry.get("number_realizations", n) != n:
                    raise ValueError(f"Sample count mismatch at p={p}, L={L}.")
                mean = float(samples.mean())
                sem = float(samples.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            else:
                n = int(entry["number_realizations"])
                mean = float(entry["mean_dv_entropy"])
                saved_sem = entry.get("sem_dv_entropy")
                variance = entry.get("variance_dv_entropy")
                sem = (float(saved_sem) if saved_sem is not None else
                       np.sqrt(float(variance) / n) if variance is not None and n > 1 else np.nan)
                if n < 2:
                    sem = np.nan
            if n < 1 or not np.isfinite(mean) or mean < 0 or (np.isfinite(sem) and sem < 0):
                raise ValueError(f"Invalid statistics at p={p}, L={L}.")
            S[i, j], SEM[i, j], counts[i, j] = mean, sem, n

    valid_counts = counts[counts > 0]
    if not valid_counts.size:
        raise ValueError("No populated results found.")
    if np.any((counts > 0) & ~np.isfinite(SEM)):
        warnings.warn("SEM is unavailable for some points; their bands are omitted.")
    return dict(probabilities=probabilities, sizes=sizes, S=S, SEM=SEM,
                counts=counts, metadata={k:v for k,v in data.items() if k != "results"},
                data_file=Path(data_file))


def plot_pauli_entropy_vs_L(dataset, output_dir=None, *, p_targets=None, zoom=(0.9, 1.0), show_sem=True, figsize=(12.5, 5.8)):
    """Return figure, axes and selected values; optionally save a PDF."""
    probabilities, sizes, S, SEM = (dataset[k] for k in ("probabilities", "sizes", "S", "SEM"))
    if not (0 <= zoom[0] < zoom[1] <= 1):
        raise ValueError("zoom must satisfy 0 <= lower < upper <= 1.")
    data_file = dataset["data_file"]
    SHOW_SEM = show_sem
    SAVE_PDF = output_dir is not None
    if SAVE_PDF:
        FIGURES_DIR = Path(output_dir)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        P_TARGETS = np.asarray(p_targets if p_targets is not None else np.arange(1, 11) / 10, dtype=float)
        if P_TARGETS.ndim != 1 or not P_TARGETS.size or not np.all(np.isfinite(P_TARGETS)):
            raise ValueError("Choose at least one finite probability in p_targets.")

        left_indices = list(dict.fromkeys(
            int(np.argmin(abs(probabilities - target)))
            for target in P_TARGETS
        ))

        right_indices = np.flatnonzero(
            (probabilities >= zoom[0]) & (probabilities <= zoom[1])
        )

        fig_sl, axes = plt.subplots(
            1, 2, figsize=figsize, layout="constrained"
        )

        for ax, indices, title in zip(
            axes,
            [left_indices, right_indices],
            [r"(a) Selected probabilities", fr"(b) ${zoom[0]:g}\leq p\leq{zoom[1]:g}$"],
        ):
            colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, len(indices)))

            for color, i in zip(colors, indices):
                p = probabilities[i]

                if ax is axes[0]:
                    label = fr"$p={p:.5g}$"
                else:
                    label = None  # Use a colour bar for the dense probability scan.

                ax.plot(sizes, S[i], "o-", color=color, label=label)

                if SHOW_SEM:
                    ax.fill_between(
                        sizes, S[i] - SEM[i], S[i] + SEM[i],
                        color=color, alpha=0.24, linewidth=0,
                    )

            ax.set(
                xlabel=r"$L$",
                ylabel=r"$\langle S\rangle$",
                title=title,
            )
            ax.set_ylim(bottom=0)
            ax.margins(x=0.04, y=0.08)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(alpha=0.22, linestyle="--")

        axes[0].legend(
            frameon=False, ncol=2, fontsize=12, loc="best"
        )

        # Match the right panel's colours to the actual probabilities.
        if len(right_indices):
            from matplotlib.colors import Normalize, LinearSegmentedColormap
            from matplotlib.cm import ScalarMappable

            cmap = LinearSegmentedColormap.from_list(
                "viridis_trimmed", plt.get_cmap("viridis")(np.linspace(0.08, 0.88, 256))
            )
            norm = Normalize(vmin=zoom[0], vmax=zoom[1])

            # Recolour curves and shaded bands using their actual p values.
            for k, i in enumerate(right_indices):
                color = cmap(norm(probabilities[i]))
                axes[1].lines[k].set_color(color)
                if SHOW_SEM:
                    axes[1].collections[k].set_facecolor(color)

            cbar = fig_sl.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                ax=axes[1], pad=0.02,
            )
            cbar.set_label(r"$p$")
        else:
            axes[1].text(
                0.5, 0.5, "No stored probabilities in this interval",
                ha="center", va="center", transform=axes[1].transAxes,
            )

        if SAVE_PDF:
            path_sl = FIGURES_DIR / f"{data_file.stem}_S_vs_L_two_panels.pdf"
            fig_sl.savefig(path_sl, bbox_inches="tight")
            print("Saved:", path_sl)

        return {"figure": fig_sl, "axes": axes, "selected_probabilities": probabilities[left_indices], "zoom_probabilities": probabilities[right_indices]}


def plot_pauli_density_vs_p(dataset, output_dir=None, *, L_targets=None, zoom=(0.85, 1.0), show_sem=True, figsize=(12.5, 5.8)):
    """Return figure, axes and selected values; optionally save a PDF."""
    probabilities, sizes, S, SEM = (dataset[k] for k in ("probabilities", "sizes", "S", "SEM"))
    if not (0 <= zoom[0] < zoom[1] <= 1):
        raise ValueError("zoom must satisfy 0 <= lower < upper <= 1.")
    data_file = dataset["data_file"]
    SHOW_SEM = show_sem
    SAVE_PDF = output_dir is not None
    if SAVE_PDF:
        FIGURES_DIR = Path(output_dir)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    L_TARGETS = L_targets
    with plt.rc_context(STYLE):
        if L_TARGETS is None:
            L_indices = list(range(len(sizes)))
        else:
            if not len(L_TARGETS):
                raise ValueError("Choose at least one size in L_TARGETS.")
            L_indices = list(dict.fromkeys(
                int(np.argmin(abs(sizes - target)))
                for target in L_TARGETS
            ))

        print("Selected sizes:", sizes[L_indices])

        fig_density, axes = plt.subplots(
            1, 2, figsize=figsize, layout="constrained"
        )

        colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, len(L_indices)))
        markers = ["o", "s", "^", "D", "v", "P", "X"]

        masks = [
            (probabilities >= 0) & (probabilities <= 1),
            (probabilities >= zoom[0]) & (probabilities <= zoom[1]),
        ]

        for ax, mask, limits, title in zip(
            axes,
            masks,
            [(0, 1), zoom],
            [r"(a) Full range", fr"(b) ${zoom[0]:g}\leq p\leq{zoom[1]:g}$"],
        ):
            p_plot = probabilities[mask]

            for k, (color, j) in enumerate(zip(colors, L_indices)):
                L = sizes[j]
                mean = S[mask, j] / L
                error = SEM[mask, j] / L

                ax.plot(
                    p_plot, mean,
                    color=color,
                    linestyle="-",
                    marker="",
                    linewidth=1.8,
                    label=fr"$L={L:g}$",
                )

                if SHOW_SEM:
                    ax.fill_between(
                        p_plot, mean - error, mean + error,
                        color=color, alpha=0.24, linewidth=0,
                    )

            ax.set(
                xlabel=r"$p$",
                ylabel=r"$\langle S\rangle/L$",
                xlim=limits,
                title=title,
            )
            ax.margins(y=0.08)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(alpha=0.22, linestyle="--")
            ax.legend(frameon=False, ncol=2, fontsize=12, loc="best")

            if not len(p_plot):
                ax.text(
                    0.5, 0.5, "No data in this interval",
                    ha="center", va="center", transform=ax.transAxes,
                )

        if SAVE_PDF:
            path_density = (
                FIGURES_DIR / f"{data_file.stem}_S_over_L_vs_p_two_panels.pdf"
            )
            fig_density.savefig(path_density, bbox_inches="tight")
            print("Saved:", path_density)

        return {"figure": fig_density, "axes": axes, "selected_sizes": sizes[L_indices]}


