"""Homodyne scan loading and figures. Load only trusted pickle files.

Install in src/monitored_cluster_states/homodyne_scan_plotting.py.
This module is independent of plotting.py; existing CV figure APIs are unchanged.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

def load_homodyne_scan(data_dir, *, r=5.0, weight=1.0, file_settings=None):
    """Read trusted .pkl files recursively, without assuming equal p/L grids.

    file_settings optionally maps a relative filename to {'phi': ..., 'tolerance': ...}.
    Metadata is preferred; filename phi/r/w and folder tolerance tags are fallbacks.
    Stored cv_entropies are total entropy; eigenvalues are symplectic eigenvalues.
    """
    import pickle
    import re
    import warnings
    from pathlib import Path

    root = Path(data_dir)
    overrides = file_settings or {}
    number = r'([-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?)'
    records = []
    for path in sorted(root.rglob('*.pkl')):
        with path.open('rb') as stream:
            raw = pickle.load(stream)
        meta = {**raw, **raw.get('metadata', {})}
        explicit = overrides.get(path.relative_to(root).as_posix(), {})
        def parameter(keys, tag, folder=False):
            for key in keys:
                if key in explicit:
                    return float(explicit[key])
                if key in meta and meta[key] is not None:
                    return float(meta[key])
            candidates = [path.stem] + ([p.name for p in path.parents if p != root.parent] if folder else [])
            for candidate in candidates:
                match = re.search(r'(?:^|[_\-])' + tag + r'[_=\-]?' + number, candidate, re.I)
                if match:
                    return float(match.group(1))
            return None
        phi = parameter(('phi', 'angle'), 'phi')
        tol = parameter(('connectivity_tolerance', 'connect_tol', 'connect_tolerance', 'connect','tolerance', 'tol', 'edge_tolerance', 'graph_tolerance'),
                        '(?:connect|tolerance|tol)', True,)
        squeezing = parameter(('r_squeezing', 'r', 'squeezing'), 'r')
        coupling = parameter(('weight', 'w'), 'w')
        if squeezing is not None and r is not None and not np.isclose(squeezing, r):
            continue
        if coupling is not None and weight is not None and not np.isclose(coupling, weight):
            continue
        if phi is None or tol is None:
            warnings.warn(f'Skipping {path}: phi/tolerance not identified. Add file_settings to include it.', stacklevel=2)
            continue
        if (r is not None and squeezing is None) or (weight is not None and coupling is None):
            raise ValueError(f'Cannot verify r/weight for {path}. Add r/weight to file_settings, or use r=None/weight=None knowingly.')
        if 'results' not in raw or not raw['results']:
            raise ValueError(f'No results in {path}.')
        points = {}
        for p, sizes in raw['results'].items():
            for L, entry in sizes.items():
                if 'cv_entropies' in entry:
                    samples = np.asarray(entry['cv_entropies'], dtype=float)
                elif 'eigenvalues' in entry:
                    samples = []
                    for spectrum in entry['eigenvalues']:
                        sigma = np.asarray(spectrum, dtype=float)
                        if not np.all(np.isfinite(sigma)) or np.any(sigma < .5 - 1e-8):
                            raise ValueError(f'Invalid symplectic spectrum in {path}, p={p}, L={L}.')
                        occupation = np.maximum(sigma - .5, 0.)
                        entropy = np.log1p(occupation)
                        mask = occupation > 0
                        entropy[mask] += occupation[mask] * np.log1p(1. / occupation[mask])
                        samples.append(float(entropy.sum() / np.log(2.)))
                    samples = np.asarray(samples)
                else:
                    samples = None
                if samples is not None:
                    if samples.ndim != 1 or not samples.size or not np.all(np.isfinite(samples)) or np.any(samples < 0):
                        raise ValueError(f'Invalid entropy samples in {path}, p={p}, L={L}.')
                    n = len(samples)
                    if entry.get('number_realizations', n) != n:
                        raise ValueError(f'Count mismatch in {path}, p={p}, L={L}.')
                    mean = float(samples.mean())
                    sem = float(samples.std(ddof=1)/np.sqrt(n)) if n > 1 else np.nan
                else:
                    n_value = entry['number_realizations']
                    n = int(n_value)
                    if n < 1 or n != n_value:
                        raise ValueError(f'Invalid sample count in {path}.')
                    mean = float(entry['mean_cv_entropy'])
                    saved_sem = entry.get('sem_cv_entropy')
                    variance = entry.get('variance_cv_entropy')
                    if variance is not None and (not np.isfinite(variance) or variance < 0):
                        raise ValueError(f'Invalid variance in {path}.')
                    sem = (float(saved_sem) if saved_sem is not None else
                           float(np.sqrt(variance/n)) if variance is not None else np.nan)
                    if n < 2:
                        sem = np.nan
                if not np.isfinite(mean) or mean < 0 or (not np.isnan(sem) and (not np.isfinite(sem) or sem < 0)):
                    raise ValueError(f'Invalid entropy statistics in {path}.')
                if not (0 <= float(p) <= 1) or float(L) <= 0:
                    raise ValueError(f'Invalid p/L in {path}.')
                points[(float(p), float(L))] = (mean, sem, n)
        records.append(dict(phi=phi, tolerance=tol, r=squeezing, weight=coupling,
                            periodic=meta.get('periodic'), source=path, points=points))
    if not records:
        raise FileNotFoundError(f'No matching homodyne results under {root}.')
    print('Available angles (radians):', sorted({d['phi'] for d in records}))
    print('Available tolerances:', sorted({d['tolerance'] for d in records}))
    return records

def plot_homodyne_phi_tolerance(records, output_dir=None, *, target_L=70,
                                fixed_phi=None, fixed_tolerance=None,
                                show_sem=True, figsize=(13, 5.5)):
    """Left: all phi at fixed tolerance. Right: all tolerances at fixed phi.

    Both panels use exactly target_L. None selects an available setting and reports
    it: smallest positive tolerance and angle (zero if no positive value exists). Numeric settings must exist.
    No interpolation, fitting, or assumed transition threshold is used.
    """
    from pathlib import Path
    import warnings
    from matplotlib.ticker import MaxNLocator

    if not records:
        raise ValueError('No datasets supplied.')
    tolerances = sorted({d['tolerance'] for d in records})
    angles = sorted({d['phi'] for d in records})
    if fixed_tolerance is None:
        # fixed_tolerance = tolerances[-1]
        fixed_tolerance = min((tol for tol in tolerances if tol > 0), default=0.0)
    if fixed_phi is None:
        fixed_phi = min((a for a in angles if a > 0), default=0.0)
    def same(a, b):
        return np.isclose(a, b, rtol=1e-9, atol=0.)
    left = [d for d in records if same(d['tolerance'], fixed_tolerance)]
    right = [d for d in records if same(d['phi'], fixed_phi)]
    if not left or not right:
        raise ValueError(f'Requested settings unavailable. phi={angles}; tolerances={tolerances}. Use stored numeric values.')
    selected = left + right
    for key in ('r', 'weight', 'periodic'):
        values = {d.get(key) for d in selected}
        if len(values) > 1:
            raise ValueError(f'Mixed {key} settings in selected files: {values}. Select comparable files.')
    curves = []
    for group, variable in ((left, 'phi'), (right, 'tolerance')):
        merged = {}
        for d in group:
            points = merged.setdefault(d[variable], {})
            for (p, L), value in d['points'].items():
                if L == target_L:
                    if p in points:
                        raise ValueError(f'Duplicate p={p}, L={L} for {variable}={d[variable]}. Select one file per point.')
                    points[p] = value
        panel = []
        for setting, points in sorted(merged.items()):
            if not points:
                raise ValueError(f'L={target_L} missing for {variable}={setting}; choose a common stored size.')
            p = np.array(sorted(points))
            values = np.asarray([points[q] for q in p])
            panel.append(dict(setting=setting, p=p, mean=values[:,0]/target_L,
                              sem=values[:,1]/target_L, counts=values[:,2].astype(int)))
        curves.append(panel)
    if show_sem and any(np.any(~np.isfinite(c['sem'])) for panel in curves for c in panel):
        warnings.warn('SEM is unavailable for some points; those bands are omitted.', stacklevel=2)
    style = {'font.size':16, 'axes.labelsize':22, 'axes.titlesize':17,
             'xtick.labelsize':17, 'ytick.labelsize':17, 'legend.fontsize':12, 'pdf.fonttype':42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, layout='constrained')
        titles = [fr'(a) $L={target_L:g}$, tolerance $={fixed_tolerance:g}$',
                  fr'(b) $L={target_L:g}$, $\varphi={fixed_phi:.5g}$']
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
        for k, (ax, panel) in enumerate(zip(axes, curves)):
            colors = plt.get_cmap('viridis')(np.linspace(.08, .88, len(panel)))
            for j, (c, color) in enumerate(zip(panel, colors)):
                label = (fr'$\varphi={c["setting"]:.5g}$' if k == 0 else
                         fr'$\epsilon={c["setting"]:.5g}$')
                ax.plot(c['p'], c['mean'], color=color, marker=markers[j%len(markers)],
                        markevery=max(1, len(c['p'])//12), ms=4, lw=1.6, label=label)
                if show_sem:
                    ax.fill_between(c['p'], c['mean']-c['sem'], c['mean']+c['sem'],
                                    color=color, alpha=.24, linewidth=0)
            ax.set(xlabel=r'$p$', title=titles[k], xlim=(0,1), ylim=(0,None))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(alpha=.22, linestyle='--')
            ax.legend(frameon=False, ncol=2 if len(panel)>5 else 1,
                      title=None if k == 0 else r'$\epsilon$: tolerance')
        axes[0].set_ylabel(r'$\langle S\rangle/L$')
        path = None
        if output_dir is not None:
            folder = Path(output_dir)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'homodyne_phi_tolerance_L_{target_L:g}_phi_{fixed_phi:.6g}_tol_{fixed_tolerance:.6g}.pdf'
            fig.savefig(path, bbox_inches='tight')
    print(f'Both panels: L={target_L:g}; left tolerance={fixed_tolerance:g}; right phi={fixed_phi:.8g} rad')
    return dict(figure=fig, axes=axes, path=path, curves=curves,
                L=target_L, fixed_phi=fixed_phi, fixed_tolerance=fixed_tolerance)

def plot_homodyne_scan_heatmaps(base_dir, output_dir=None, *,
                                weights=(0.5, 1.0, 2.0), squeezings=(1.0,),
                                p_target=0.6, L_small=30, L_large=60,
                                records_override=None):
    """Plot ratio maps: weight columns, squeezing rows, shared colour scale.

    Missing folders/cells stay grey. Uses only p_target (absolute tolerance 1e-10).
    Returns figure, axes, path, heatmaps and sampled axes. Cells have equal
    display width; positive log-spaced scans therefore appear logarithmic.
    No interpolation, phase classification or cell annotations are applied.
    records_override is used internally by plot_homodyne_scaling_ratio.
    """
    BASE_DIR = Path(base_dir)
    W_VALUES, R_VALUES = list(weights), list(squeezings)
    if not W_VALUES or not R_VALUES:
        raise ValueError("Choose at least one weight and squeezing.")
    if len(set(W_VALUES)) != len(W_VALUES) or len(set(R_VALUES)) != len(R_VALUES):
        raise ValueError("Weight/squeezing selections must not contain duplicates.")
    if not 0 <= p_target <= 1 or not 0 < L_small < L_large:
        raise ValueError("Require 0 <= p_target <= 1 and 0 < L_small < L_large.")
    P_TARGET, L_SMALL, L_LARGE = p_target, L_small, L_large
    SAVE_PDF = output_dir is not None
    output_path = None
    datasets = {}
    all_phis = set()
    all_tolerances = set()

    for r in R_VALUES:
        for w in W_VALUES:
            folder = BASE_DIR / f"w_{float(w)}_r_{float(r)}"

            if records_override is not None:
                datasets[(w, r)] = list(records_override)
                all_phis.update(float(d["phi"]) for d in records_override)
                all_tolerances.update(float(d["tolerance"]) for d in records_override)
                continue

            if not folder.exists() or not any(folder.rglob("*.pkl")):
                print(f"Not available yet: w={w:g}, r={r:g}")
                datasets[(w, r)] = []
                continue

            records = load_homodyne_scan(
                folder,
                r=r,
                weight=w,
                file_settings={},
            )
            datasets[(w, r)] = records

            for record in records:
                all_phis.add(float(record["phi"]))
                all_tolerances.add(float(record["tolerance"]))

    if not all_phis or not all_tolerances:
        raise ValueError("No scan data found. Check BASE_DIR.")

    # Same grid in every panel, including partially completed scans.
    phis = sorted(all_phis)
    tolerances = sorted(all_tolerances)

    phi_index = {value: j for j, value in enumerate(phis)}
    tol_index = {value: i for i, value in enumerate(tolerances)}

    # ------------------------------------------------------------
    # Calculate each heatmap
    # ------------------------------------------------------------
    heatmaps = {}

    for (w, r), records in datasets.items():
        matrix = np.full((len(tolerances), len(phis)), np.nan)
        grouped = {}

        for record in records:
            key = (float(record["phi"]), float(record["tolerance"]))
            points = grouped.setdefault(key, {})

            for (p, L), values in record["points"].items():
                point = (float(p), float(L))
                if point in points:
                    raise ValueError(
                        f"Duplicate data: w={w}, r={r}, "
                        f"phi={key[0]}, tolerance={key[1]}, p={p}, L={L}"
                    )
                points[point] = values

        for (phi, tolerance), points in grouped.items():
            # Use the requested p in every panel, not different nearest values.
            matching_p = sorted({
                p for p, L in points
                if np.isclose(p, P_TARGET, rtol=0, atol=1e-10)
            })

            if not matching_p:
                continue
            if len(matching_p) > 1:
                raise ValueError(f"Multiple probabilities match p={P_TARGET}.")

            p = matching_p[0]
            small_entry = points.get((p, float(L_SMALL)))
            large_entry = points.get((p, float(L_LARGE)))

            # A simulation may not have reached both sizes yet.
            if small_entry is None or large_entry is None:
                continue

            small = small_entry[0]
            large = large_entry[0]

            if (
                np.isfinite(small)
                and np.isfinite(large)
                and small > 0
                and large >= 0
            ):
                matrix[tol_index[tolerance], phi_index[phi]] = (
                    (large / L_LARGE) / (small / L_SMALL)
                )

        heatmaps[(w, r)] = matrix
        print(
            f"w={w:g}, r={r:g}: "
            f"{np.isfinite(matrix).sum()}/{matrix.size} cells available"
        )

    # ------------------------------------------------------------
    # Common colour scale
    # ------------------------------------------------------------
    finite = np.concatenate([
        matrix[np.isfinite(matrix)]
        for matrix in heatmaps.values()
    ])

    if finite.size == 0:
        raise ValueError(
            f"No valid ratios at p={P_TARGET}, "
            f"L={L_SMALL} and L={L_LARGE}."
        )

    norm = Normalize(
        vmin=min(L_SMALL / L_LARGE, float(finite.min())),
        vmax=max(1.0, float(finite.max())),
    )

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("0.85")

    def sparse_log_ticks(values):
        positions, labels = [], []
        for i, value in enumerate(values):
            if value == 0:
                positions.append(i)
                labels.append(r"$0$")
            elif value > 0:
                exponent = int(np.rint(np.log10(value)))
                if np.isclose(value, 10.0**exponent, rtol=1e-6, atol=0):
                    positions.append(i)
                    labels.append(fr"$10^{{{exponent}}}$")
        return positions, labels

    xticks, xlabels = sparse_log_ticks(phis)
    yticks, ylabels = sparse_log_ticks(tolerances)

    # ------------------------------------------------------------
    # Plot: columns = w, rows = r
    # ------------------------------------------------------------
    nrows, ncols = len(R_VALUES), len(W_VALUES)

    with plt.rc_context({
        "font.size": 13,
        "axes.labelsize": 19,
        "axes.titlesize": 17,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "pdf.fonttype": 42,
    }):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.5 * ncols + 0.8, 4.4 * nrows + 0.8),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )

        for row, r in enumerate(R_VALUES):
            for col, w in enumerate(W_VALUES):
                ax = axes[row, col]
                matrix = heatmaps[(w, r)]

                im = ax.imshow(
                    np.ma.masked_invalid(matrix),
                    origin="lower",
                    interpolation="nearest",
                    aspect="auto",
                    cmap=cmap,
                    norm=norm,
                )

                ax.set_title(fr"$w={w:g},\quad r={r:g}$")
                ax.set_xticks(xticks, labels=xlabels)
                ax.set_yticks(yticks, labels=ylabels)

                if row == nrows - 1:
                    ax.set_xlabel(r"$\varphi\;(\mathrm{rad})$")
                    ax.tick_params(axis="x", labelrotation=45)

                if col == 0:
                    ax.set_ylabel(r"Tolerance $\varepsilon$")

                if 0.0 in phis:
                    ax.axvline(
                        phis.index(0.0) + 0.5,
                        color="white", linewidth=1,
                    )
                if 0.0 in tolerances:
                    ax.axhline(
                        tolerances.index(0.0) + 0.5,
                        color="white", linewidth=1,
                    )

        fig.suptitle(
            fr"$p={P_TARGET:g},\quad "
            fr"L_{{\mathrm{{small}}}}={L_SMALL},\quad "
            fr"L_{{\mathrm{{large}}}}={L_LARGE}$",
            fontsize=20,
        )

        cbar = fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.025,
            aspect=30,
        )
        cbar.set_label(r"$R$", rotation=0, labelpad=14)

        if any(np.any(~np.isfinite(m)) for m in heatmaps.values()):
            fig.legend(
                handles=[
                    Patch(facecolor="0.85", label="Missing or undefined ratio")
                ],
                loc="outside lower center",
                frameon=False,
            )

        if SAVE_PDF:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            w_tag = "_".join(f"{w:g}" for w in W_VALUES)
            r_tag = "_".join(f"{r:g}" for r in R_VALUES)

            output_path = output_dir / (
                f"heatmaps_p_{P_TARGET:g}_w_{w_tag}_r_{r_tag}"
                f"_L_{L_SMALL}_{L_LARGE}.pdf"
            )
            fig.savefig(output_path, bbox_inches="tight")
            print("Saved:", output_path)

    return dict(figure=fig, axes=axes, path=output_path, heatmaps=heatmaps,
                phis=phis, tolerances=tolerances, p_actual=P_TARGET, norm=norm)

def plot_homodyne_scaling_ratio(records, output_dir=None, *, p_target=0.6,
                                 L_small=30, L_large=60):
    """Plot one ratio map from loaded records without notebook globals."""
    records = list(records)
    if not records:
        raise ValueError("No records supplied.")
    for key in ("weight", "r", "periodic"):
        if len({d.get(key) for d in records}) > 1:
            raise ValueError(f"Mixed {key} settings; select one dataset.")
    w, r = records[0]["weight"], records[0]["r"]
    if w is None or r is None:
        raise ValueError("Weight and squeezing metadata are required.")
    result = plot_homodyne_scan_heatmaps(
        ".", output_dir, weights=[w], squeezings=[r], p_target=p_target,
        L_small=L_small, L_large=L_large, records_override=records,
    )
    result["ratio"] = result["heatmaps"][(w, r)]
    return result
