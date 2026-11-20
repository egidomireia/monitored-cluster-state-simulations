# Monitored cluster-state simulations

Simulations and plots for vertex deletion, general homodyne measurements, and Pauli-X/Y measurements on square lattices. The scripts generate the data; the notebooks load saved results and produce the figures.

## Files and folders

Use the filenames below in the repository, without the copy suffixes such as `(1)` or `(3)` added when downloading files.

| File | Purpose |
| --- | --- |
| `scripts/simulate_vertex_deletion.py` | Vertex deletion: LCC size, DV entropy and CV eigenvalue data |
| `scripts/simulate_general_homodyne.py` | CV homodyne scans over probability, angle, connectivity tolerance, weight and squeezing |
| `scripts/simulate_pauli_y.py` | Pauli-Y measurements and DV entropy |
| `scripts/simulate_pauli_x.py` | Pauli-X measurements and DV entropy |
| `notebooks/01_vertex_deletion_results.ipynb` | Vertex-deletion results |
| `notebooks/02_homodyne_general_results.ipynb` | General homodyne results |
| `notebooks/03_homodyne_general_scan.ipynb` | Angle/tolerance comparisons and scaling-ratio heatmaps |
| `notebooks/04_pauli_y_results.ipynb` | Pauli-Y entropy and entropy-density plots |
| `notebooks/05_pauli_x_results.ipynb` | Pauli-X entropy and entropy-density plots |
| `src/monitored_cluster_states/plotting.py` | Existing vertex-deletion and homodyne plotting functions |
| `src/monitored_cluster_states/homodyne_scan_plotting.py` | Loading and plotting the homodyne scans |
| `src/monitored_cluster_states/pauli_plotting.py` | Shared loading and plotting functions for Pauli-X/Y |
| `src/monitored_cluster_states/paths.py` | Shared `DATA_DIR` and `FIGURES_DIR` paths |

Simulation output goes into `data_files/vertex_deletion/`, `data_files/homodyne_general/`, `data_files/pauli_y/` or `data_files/pauli_x/`.

## Running the code

From the repository root, install the dependencies in the Python environment you will use:

```bash
python -m pip install -r requirements.txt
```

Edit the parameters near the top of the script, then run the required simulation:

```bash
python scripts/simulate_vertex_deletion.py
python scripts/simulate_general_homodyne.py
python scripts/simulate_pauli_y.py
python scripts/simulate_pauli_x.py
```

There are no checkpoints or automatic resume: Pauli scripts save after the complete run, vertex deletion saves after each weight, and homodyne saves after each angle's probability/size scan, with a separate file for each tolerance. Existing files with the same name are overwritten.


## Storage and error bands

`STORE_ALL_REALIZATIONS` controls what is retained, not how many realisations are simulated:

- `True`: save individual results for later analysis.
- `False`: save summary statistics, including the sample count, mean and SEM. Individual samples cannot be recovered from these summaries.

The exact saved keys differ between protocols; see below. `pauli_plotting.py` and `homodyne_scan_plotting.py` support both storage options. Their shaded bands show one standard error of the mean, using the actual number of realisations at each point:

$$\mathrm{SEM}=s/\sqrt{N},$$

where `s` is the sample standard deviation with `ddof=1`. For entropy-density plots, both the mean and SEM are divided by `L`. SEM is unavailable for a single realisation. Using the same samples, both storage modes give the same means and SEM up to numerical rounding.

## Vertex deletion

### Parameters

Edit the settings at the top of the script. For example:

```python
L_MIN = 100
L_MAX = 900
L_STEP = 50
dimensions = np.arange(L_MIN, L_MAX + 1, L_STEP, dtype=int)

P_INTERVALS = [(0.00, 0.35, 15), (0.35, 0.46, 111), (0.46, 1.00, 28)]
p_data = np.unique(np.round(np.concatenate([np.linspace(start, stop, number) for start, stop, number in P_INTERVALS]), decimals=12))

REALIZATIONS = 500
weight_values = [1.0]
lattice_type = "square"
SEED = 12345
STORE_ALL_REALIZATIONS = True
```

`L` is the side length, so each initial lattice has `L**2` vertices. Each tuple in `P_INTERVALS` gives the start, end, and number of probabilities, including both endpoints. This lets you sample more densely near the transition. `REALIZATIONS` sets the number of independent samples for each size, probability, and weight.

For a quick test, use:

```python
dimensions = np.array([10, 20, 40])
p_data = np.array([0.0, 0.2, 0.35, 0.407, 0.45, 0.6, 1.0])
REALIZATIONS = 10
```

### What the code calculates

`measurement_q` creates a square lattice with periodic boundaries and removes each vertex independently with probability `p`. `extract_graph_data_all` then selects the largest connected component (LCC) and divides it into two parts using the first lattice coordinate, with `cut_x = L // 2 - 0.5`.

Alongside the CV eigenvalues and DV entropy, `extract_graph_data_all` returns the classical LCC size, meaning the number of vertices in that component. We kept this so we could also plot the classical percolation results, using the mean LCC size divided by `L**2`, and compare them with the entanglement results as a check on the simulations.

For this geometry and `L >= 4`, the crossing edges have distinct endpoints. The DV entropy is their number, and the CV eigenvalues are their squared weights. Both periodic interfaces are already included. CV entropy is calculated afterwards from these eigenvalues, so the same simulation can be used for different squeezing values.

The general matrix-rank branch in the original function is not needed for this square-lattice cut. It would need a binary-field rank calculation before using it for DV entropy on other geometries.

`main` repeats the calculation over the requested parameters and saves the results. There are no checkpoints: one file is written per weight once all its calculations are finished. Running again with the same filename overwrites it.

### Saved results

The output folder is `data_files/vertex_deletion/`. The filename is set in the script; for example, `vertex_deletion_results_PBC_w_1.0.pkl`.

To open it from the repository root:

```python
import pickle

with open("data_files/vertex_deletion/vertex_deletion_results_PBC_w_1.0.pkl", "rb") as f:
    data = pickle.load(f)

p = sorted(data["results"])[0]
L = sorted(data["results"][p])[0]
entry = data["results"][p][L]
```

With `STORE_ALL_REALIZATIONS = True`, `entry` contains `LCC_sizes`, `eigenvalues`, and `dv_entropies`, with matching entries for each realization.

With `False`, it contains `number_realizations`, `mean_LCC_size`, `sem_LCC_size`, `mean_dv_entropy`, `sem_dv_entropy`, `cv_eigenvalue`, and `mean_cv_eigenvalue_count`. SEM means standard error of the mean. For this cut and uniform weight, the CV eigenvalues all equal `weight**2`, so their mean count is enough to recover the mean CV entropy at any chosen squeezing. Its SEM is the entropy contribution of one eigenvalue times `sem_dv_entropy`. The plotting code needs to use the keys corresponding to the storage option selected.

As a basic check, at `p = 0` the LCC should have `L**2` vertices, DV entropy should be `2*L`, and there should be `2*L` eigenvalues equal to `weight**2`. At `p = 1`, the LCC size and entropy should be zero.



## General homodyne scans

Edit `dimensions`, `p_data`, `phi_values`, `tolerance_values`, `weight_values`, `r_values`, `REALIZATIONS` and `SEED` in `scripts/simulate_general_homodyne.py`. The current scan uses periodic boundaries.

The active loop uses `tolerance_values`; changing only the standalone `CONNECTIVITY_TOL` setting will not change this scan. For each realisation, the measured Gaussian state is calculated once and analysed at every requested tolerance. Tolerance sets which couplings count for identifying the LCC; it does not zero those couplings in the Gaussian state used for the entropy calculation.

Each weight/squeezing combination gets its own `w_<weight>_r_<r>` folder. Filenames include the angle, tolerance, storage mode, weight and squeezing, for example:

```text
homodyne_results_PBC_phi_1e-12_connect_1e-12_mean_w_0.5_r_0.5.pkl
```

Each file stores parameter metadata and `results[p][L]`. With `True`, entries contain `LCC_sizes`, `eigenvalues` (symplectic eigenvalues) and `cv_entropies`. With `False`, they contain the count and the means and SEM of LCC size and CV entropy. Unlike vertex deletion, these homodyne results are calculated for the specified squeezing.

### Plotting the scans

Open `03_homodyne_general_scan.ipynb` and run the imports first. Its three sections can then run independently:

1. **Entropy density versus probability:** compare angles at fixed tolerance, and tolerances at fixed angle. Set `SCAN_DIR`, `TARGET_L`, `FIXED_PHI`, `FIXED_TOLERANCE`, `R_FIXED` and `W_FIXED`.
2. **Compare heatmaps:** set `BASE_DIR`, `W_VALUES` and `R_VALUES`. Columns show weights and rows show squeezing values, using a shared colour scale. Missing or undefined cells, including unfinished scans, appear grey.
3. **Single heatmap:** set `SINGLE_DIR`, the weight/squeezing pair, probability and two sizes.

The heatmap colour is the finite-size ratio

$$R=\frac{\langle S(L_{\mathrm{large}})\rangle/L_{\mathrm{large}}}{\langle S(L_{\mathrm{small}})\rangle/L_{\mathrm{small}}}.$$

For the default sizes 30 and 60, a strictly linear `S` through the origin gives `R=1`, while constant nonzero `S` gives `R=0.5`.

The current simulations at `p=0.6` support these heatmaps. To plot `S/L` versus `p`, choose a dataset with a probability scan or simulate additional probabilities. Set `SAVE_PDF = True` to save figures under `FIGURES_DIR / "homodyne_scan"`.

## Pauli-X and Pauli-Y

The Pauli scripts use independent site selection with probability `p`, process measured sites in the original node order, and update the current graph after each measurement. Pauli-Y uses local complementation followed by deletion; Pauli-X uses the implemented neighbour-pivot sequence followed by deletion of the measured site. Both calculate the LCC entropy across the fixed spatial cut using binary rank over GF(2), not `np.linalg.matrix_rank` over the real numbers.

Edit the settings at the top of the relevant script:

```python
PERIODIC = True
L_MIN, L_MAX, L_STEP = 40, 80, 10
dimensions = np.arange(L_MIN, L_MAX + 1, L_STEP, dtype=int)
REALIZATIONS = 50
SEED = 12345
STORE_ALL_REALIZATIONS = False
```

Also set `p_data` (or the `P_INTERVALS` used to build it) to the desired probabilities. `PERIODIC = True` enables periodic boundaries; `False` uses open boundaries.

Output filenames identify the basis, boundary conditions, binary-rank calculation, storage mode and maximum-size setting. Examples:

```text
data_files/pauli_y/pauli_y_results_PBC_binary_rank_mean_Lmax_80.pkl
data_files/pauli_x/pauli_x_results_PBC_binary_rank_all_Lmax_80.pkl
```

Files store metadata and `results[p][L]`. Both storage modes retain counts, means, sample variances and SEM for LCC size and DV entropy. `True` additionally retains `LCC_sizes` and `dv_entropies`; it does not retain the graphs themselves.

### Plotting Pauli results

Open `04_pauli_y_results.ipynb` or `05_pauli_x_results.ipynb`. Both import `pauli_plotting.py` and produce:

- `S/L` versus `p`, with a full-range panel and a high-probability zoom.
- `S` versus `L`, with selected probabilities on the left and all stored probabilities within the zoom interval on the right.

`BASIS` selects `"y"` or `"x"`. Keep `FILE_NAME = None` if there is exactly one matching dataset in the basis folder. Otherwise select its exact filename:

```python
FILE_NAME = "pauli_y_results_PBC_binary_rank_mean_Lmax_80.pkl"
```

`MAX_P_CURVES` limits the number of probability curves selected from the data; `L_TARGETS` initially includes all stored sizes. Set `zoom` separately in each plotting call. `SHOW_SEM` toggles the bands, and `SAVE_PDF` saves figures under `FIGURES_DIR / "pauli_y"` or `FIGURES_DIR / "pauli_x"`.

No plotting-code changes are needed when switching between `_all_` and `_mean_` files. Select the appropriate file and rerun the loading and plotting cells.

## Citation and license

See `LICENSE` for the repository's MIT license.
