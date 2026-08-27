# Entanglement Phase Transitions in Monitored Cluster States

This repository studies how local measurements modify graph connectivity and bipartite entanglement in discrete-variable (DV) and continuous-variable (CV) cluster states. The simulations compare vertex-deletion measurements with general homodyne measurements and analyse the resulting entanglement-scaling transitions.

## Main features

- Simulation of square-lattice cluster states under local measurements.
- Vertex-deletion measurements corresponding to graph-node removal.
- General homodyne measurements that can rewire the CV graph.
- DV and CV bipartite-entanglement observables across a fixed central cut.
- Finite-size-scaling and data-collapse analysis.
- Reusable plotting and data-processing functions.
- Notebooks for reproducing the main result figures.

## Installation

Python 3.10 or later is recommended. Run all commands from the repository root: the directory containing `pyproject.toml`.

### 1. Download the repository

Using Git:

```bash
git clone https://github.com/egidomireia/monitored-cluster-state-simulations.git
cd monitored-cluster-state-simulations
```

Alternatively, download the repository as a ZIP file from GitHub and extract it before continuing.

### 2. Create an isolated Python environment

#### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These commands use the environment directly, so Conda activation is not required.

#### macOS or Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The first line of `requirements.txt` is `-e .`, which installs the package from `src/` in editable mode together with its dependencies.

### 3. Verify the installation

On Windows:

```powershell
.\.venv\Scripts\python.exe -c "import monitored_cluster_states; print(monitored_cluster_states.__file__)"
```

On macOS or Linux:

```bash
.venv/bin/python -c "import monitored_cluster_states; print(monitored_cluster_states.__file__)"
```

The printed path should end with:

```text
src/monitored_cluster_states/__init__.py
```

## Running the notebooks

The analysis is divided into two notebooks:

- `notebooks/01_vertex_deletion_results.ipynb`: vertex-deletion results and comparison between DV and CV entanglement.
- `notebooks/02_general_homodyne_results.ipynb`: results for general homodyne measurements and different measurement angles.

To start JupyterLab on Windows:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

On macOS or Linux:

```bash
.venv/bin/python -m jupyter lab
```

When using VS Code, select the interpreter or notebook kernel located at:

```text
.venv/Scripts/python.exe    # Windows
.venv/bin/python            # macOS/Linux
```

The notebooks import the project package normally:

```python
from monitored_cluster_states.paths import RAW_DATA_DIR
from monitored_cluster_states.observables import initialize_data, initialize_data_q
from monitored_cluster_states.plotting import plot_entropy_vs_L
```

No absolute local paths or manual modifications to `sys.path` should be necessary after installation.

## Data

Simulation data are organised as follows:

- `data_file/vertex_deletion/`: raw simulation output for the vertex deletion case, stored as `.pkl` files.
- `data_file/vertex_deletion/`: raw simulation output for the general homodyne measurement case, stored as `.pkl` files.

If the full data are unavailable, they can be regenerated with the simulation scripts, although the largest system sizes can require substantial memory and computation time.

## Running simulations

Review the corresponding JSON file in `configs/` before starting a simulation. Parameters include the lattice dimensions, measurement probabilities, number of realisations, squeezing, edge weight, measurement angle and parallel-worker count.

From the repository root, the simulation entry points are:

```bash
python scripts/simulate_vertex_deletion.py --config configs/vertex_deletion.json
python scripts/simulate_general_homodyne.py --config configs/general_homodyne.json
```

Raw outputs should be written under `data/raw/`. The simulation code saves progress periodically so that an incomplete calculation can be resumed from the existing output file.

> **Computational note:** start with the smaller dimensions and realisation counts in the sample configuration. Large CV calculations can consume significant memory, particularly when several realisations are evaluated in parallel.

## Finite-size scaling

Finite-size-scaling analyses are run through:

```bash
python scripts/run_data_collapse.py
```

The project expects the external `autoScale.py` program at:

```text
external/autoscale/autoScale.py
```

This location is defined in `src/monitored_cluster_states/paths.py`. If `autoScale` is not already included, clone it from its original repository:

```bash
git clone https://github.com/omelchert/autoScale.git external/autoscale
```

The scaling outputs are stored in `results/finite_size_scaling/`.

For details of the method, see:

> O. Melchert, *autoScale.py - A program for automatic finite-size scaling analyses: A user's guide*, arXiv:0910.5403 (2009).  
> https://arxiv.org/abs/0910.5403


## Reproducibility notes

- Run commands from the repository root unless stated otherwise.
- Use the Python environment in which the repository was installed.
- Record the random seed, lattice dimensions, probability grid and number of realisations for new simulation runs.
- Avoid committing large generated `.pkl` files unless Git LFS or another data repository is used.
- Pickle files can execute arbitrary code when loaded; only open data files from trusted sources.

## Citation

If you use this repository, please cite the thesis and the code:

```bibtex
@software{Egido2026MonitoredClusterStates,
  author  = {Mireia Egido},
  title   = {Monitored Cluster State Simulations},
  year    = {2026},
  url     = {https://github.com/egidomireia/monitored-cluster-state-simulations}
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

The external `autoScale` code remains subject to its own license and attribution requirements.
