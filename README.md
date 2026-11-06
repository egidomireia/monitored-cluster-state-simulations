# Vertex-deletion simulations

I've adapted `scripts/simulate_vertex_deletion.py` for the vertex-deletion calculations we discussed. The parameters are at the top of the file.

`STORE_ALL_REALIZATIONS` lets you choose how much data to save. Initially, we kept the results of every realization in case we needed them for further analysis without rerunning the simulations. Setting it to `True` keeps those individual results; setting it to `False` saves the means and standard errors, together with the information needed to reconstruct the mean CV entropy for this geometry and uniform weight. This makes the output files much smaller.

The main files and folders are:

| Location | Contents |
| --- | --- |
| `scripts/simulate_vertex_deletion.py` | Simulation code and parameters |
| `data_files/vertex_deletion/` | Simulation output |
| `notebooks/` | Existing analysis and plots |

The other scripts and notebooks haven't been cleaned up or adapted yet. I've left them in the repository for reference, but they may need changes before running. Everything needed for these simulations is in `simulate_vertex_deletion.py`.

## Running the code

From the repository root, run:

```bash
python -m pip install -r requirements.txt
python scripts/simulate_vertex_deletion.py
```


## Parameters

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

## What the code calculates

`measurement_q` creates a square lattice with periodic boundaries and removes each vertex independently with probability `p`. `extract_graph_data_all` then selects the largest connected component (LCC) and divides it into two parts using the first lattice coordinate, with `cut_x = L // 2 - 0.5`.

Alongside the CV eigenvalues and DV entropy, `extract_graph_data_all` returns the classical LCC size, meaning the number of vertices in that component. We kept this so we could also plot the classical percolation results, using the mean LCC size divided by `L**2`, and compare them with the entanglement results as a check on the simulations.

For this geometry and `L >= 4`, the crossing edges have distinct endpoints. The DV entropy is their number, and the CV eigenvalues are their squared weights. Both periodic interfaces are already included. CV entropy is calculated afterwards from these eigenvalues, so the same simulation can be used for different squeezing values.

The general matrix-rank branch in the original function is not needed for this square-lattice cut. It would need a binary-field rank calculation before using it for DV entropy on other geometries.

`main` repeats the calculation over the requested parameters and saves the results. There are no checkpoints: one file is written per weight once all its calculations are finished. Running again with the same filename overwrites it.

## Saved results

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



## Citation and license

See `LICENSE` for the repository's MIT license.
