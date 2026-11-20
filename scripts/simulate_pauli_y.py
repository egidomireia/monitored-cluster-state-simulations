from pathlib import Path
import pickle
import numpy as np
import networkx as nx

# ==========================================
# 0. PARAMETERS -- edit these
# ==========================================
STORE_ALL_REALIZATIONS = False  # False retains counts, means, variances and SEM.
PERIODIC = True  # True uses PBC

L_MIN, L_MAX, L_STEP = 40, 80, 10

dimensions = np.arange(L_MIN, L_MAX + 1, L_STEP, dtype=int)

# defined as (start, stop, number of points).
P_INTERVALS = [
    (0.00, 0.7, 50),
    (0.7, 1.00, 50),
]

# Combine intervals, remove repeated endpoints, and sort.
p_data = np.unique(np.round( np.concatenate([np.linspace(start, stop, number) for start, stop, number in P_INTERVALS]), decimals=12))

REALIZATIONS = 50  
SEED = 12345

lattice_type = 'square'
# Resolve the output folder relative to this script, independently of the terminal working directory.
data_folder = Path(__file__).resolve().parents[1] / 'data_files' / 'pauli_y'

# ==========================================
# 1. GRAPH CASCADE
# ==========================================
# Apply the Y rule to the current graph representative at each selected site.
def measurement_pauli_y(p_measure, x, y, periodic=None):
    if periodic is None:
        periodic = PERIODIC
    G = nx.grid_2d_graph(x, y, periodic=periodic)
    Z = nx.to_numpy_array(G, dtype=np.int8)
    # Keep the original coordinate labels so matrix indices can later be converted back to lattice sites.
    initial_nodes = list(G.nodes())
    total_nodes = len(initial_nodes)
    # Select each site independently with probability p_measure
    measure_mask = np.random.random(total_nodes) < p_measure
    # Process selected sites in their original node order, using the updated graph at each step.
    measure_indices = np.where(measure_mask)[0]
    keep_indices = np.where(~measure_mask)[0]
    # If every qubit is measured, the surviving graph is empty and no rewiring needs to be computed.
    if len(keep_indices) == 0:
        return G, nx.Graph()
    for idx in measure_indices:
        # Read the current neighbours, including edges created or removed by earlier updates.
        n_idx = np.nonzero(Z[:, idx])[0]
        if len(n_idx) > 1:
            # Locally complement the neighbourhood: toggle connections between every pair of neighbours.
            Z[np.ix_(n_idx, n_idx)] ^= 1
            # Remove the diagonal entries introduced by the toggle; a simple graph has no self-loops.
            Z[n_idx, n_idx] = 0
        # Disconnect the measured site in both directions so it cannot affect subsequent neighbourhoods.
        Z[idx, :] = 0
        Z[:, idx] = 0
    # Select only surviving rows and columns; zeroed measured sites must not remain as isolated vertices.
    Z_final = Z[np.ix_(keep_indices, keep_indices)]

    # Restore physical coordinates rather than retaining the temporary integer labels from the matrix.
    surviving_nodes = [initial_nodes[i] for i in keep_indices]
    mapping = {i: surviving_nodes[i] for i in range(len(surviving_nodes))}
    G_meas_int = nx.from_numpy_array(Z_final)
    G_meas = nx.relabel_nodes(G_meas_int, mapping)
    return G, G_meas


def binary_rank(matrix):
    """Exact rank of a binary matrix over F_2 using XOR elimination."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError('Expected a two-dimensional binary matrix.')
    # Store one independent binary row per pivot column; the number of stored rows will be the rank.
    pivots = {}
    # Pack each binary row into a Python integer, with one bit per column.
    for values in matrix:
        row = 0
        for column in np.flatnonzero(values):
            # Set the bit corresponding to this nonzero column.
            row |= 1 << int(column)
        # Eliminate existing pivots using XOR, which is row addition modulo two.
        while row:
            # Choose the highest occupied bit as the pivot column.
            pivot = row.bit_length() - 1
            if pivot in pivots:
                row ^= pivots[pivot]
            else:
                # A previously unused pivot identifies another linearly independent row.
                pivots[pivot] = row
                break
    # Rows that reduce to zero are dependent and do not increase the rank.
    return len(pivots)


def entanglement_network_DV(G, G_meas, dim):

    G.add_edges_from(G.edges(), weight=1)
    G_meas.add_edges_from(G_meas.edges(), weight=1)
    total_nodes = dim*dim
    # Sort components by size and select the largest; ties retain the component iteration order.
    G_cc_components = sorted(nx.connected_components(G_meas), key=len, reverse=True)
    # An empty surviving graph has zero entanglement.
    if not G_cc_components:
        return 0.0
    G_cc = G_meas.subgraph(G_cc_components[0])
    # Partition the selected LCC by original horizontal position
    nodes_A = [n for n in G_cc.nodes() if n[0] < dim // 2]
    nodes_B = [n for n in G_cc.nodes() if n[0] >= dim // 2]
    # Map coordinate (i, j) to matrix index dim*i + j in the original square lattice.
    idx_A = [dim * n[0] + n[1] for n in nodes_A]
    idx_B = [dim * n[0] + n[1] for n in nodes_B]
    # Build the adjacency matrix in the original indexing; only the LCC cross-cut block is used below.
    nbrs = np.zeros((total_nodes, total_nodes))
    for x, y, data in G_meas.edges(data=True):
        idx_x = dim * x[0] + x[1]
        idx_y = dim * y[0] + y[1]
        nbrs[idx_x, idx_y] = G_meas[x][y]['weight']
        nbrs[idx_y, idx_x] = G_meas[y][x]['weight']
    # If the selected component lies entirely on one side, its entropy across this cut is zero.
    if idx_A == [] or idx_B == []:
        return 0.0
    else:
        # Keep edges crossing from the LCC's left half to its right half.
        Gamma_AB = nbrs[np.ix_(idx_A, idx_B)]
    # For a pure qubit graph state, the cut rank over F_2 gives the von Neumann entropy in bits.
    dv_entropy = binary_rank(Gamma_AB)
    return float(dv_entropy)

# ==========================================
# 2. STATISTICS
# ==========================================
def new_entry(store_all):
    # Maintain separate running statistics for the LCC size and its bipartite entropy.
    entry = {'number_realizations': 0}
    for name in ('LCC_size', 'dv_entropy'):
        entry.update({f'mean_{name}': 0.0, f'M2_{name}': 0.0, f'variance_{name}': None, f'sem_{name}': None})
    # Retain individual samples only when requested; summary statistics are available in either mode.
    if store_all:
        entry.update(LCC_sizes=[], dv_entropies=[])
    return entry


def add_realization(entry, size, entropy, store_all):
    # Update the statistics after adding one independent realization.
    n = entry['number_realizations'] + 1
    for name, value in (('LCC_size', size), ('dv_entropy', entropy)):
        # delta uses the previous mean.
        delta = value - entry[f'mean_{name}']
        entry[f'mean_{name}'] += delta / n
        # M2 accumulates squared deviations; the second factor uses the newly updated mean.
        entry[f'M2_{name}'] += delta * (value - entry[f'mean_{name}'])
        # Divide M2 by n-1 for the sample variance; clip tiny negative roundoff to zero.
        variance = max(entry[f'M2_{name}'], 0.0) / (n - 1) if n > 1 else None
        entry[f'variance_{name}'] = variance
        # SEM = sample standard deviation / sqrt(n); it is undefined here for a single realization.
        entry[f'sem_{name}'] = float(np.sqrt(variance / n)) if n > 1 else None
    # Retain individual samples only when requested; summary statistics are available in either mode.
    if store_all:
        entry['LCC_sizes'].append(int(size))
        entry['dv_entropies'].append(float(entropy))
    entry['number_realizations'] = n


def positive_integer(value, name, minimum=1):
    # Reject nonfinite, fractional or too-small values before converting to an integer.
    if not np.isfinite(value) or value != int(value) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)

# ==========================================
# 3. MAIN SIMULATION LOOP (SEQUENTIAL)
# ==========================================
def verify_boundary_conditions(sizes, periodic):
    """Check the actual initial lattice before spending time on simulations."""
    for size in sizes:
        L = int(size)
        G = nx.grid_2d_graph(L, L, periodic=periodic)
        # Check for edges joining opposite boundaries in both lattice directions.
        wrap_x = all(G.has_edge((0, j), (L - 1, j)) for j in range(L))
        wrap_y = all(G.has_edge((i, 0), (i, L - 1)) for i in range(L))
        # For the sizes allowed below, a periodic square lattice has 2L squared edges; an open one has 2L(L-1).
        expected_edges = 2 * L * L if periodic else 2 * L * (L - 1)
        if G.number_of_edges() != expected_edges:
            raise RuntimeError(f'Unexpected edge count for L={L}.')
        if periodic and not (wrap_x and wrap_y and all(d == 4 for _, d in G.degree())):
            raise RuntimeError(f'PBC validation failed for L={L}.')
        if not periodic and (wrap_x or wrap_y):
            raise RuntimeError(f'OBC validation failed for L={L}.')
    print('Verified PBC: wraparound edges in both directions; every initial node has degree 4.' if periodic else 'Verified OBC: no wraparound edges.', flush=True)


def main():
    # Validate the parameter arrays before generating any measurement realizations.
    probabilities = np.asarray(p_data, dtype=float)
    sizes_input = np.asarray(dimensions, dtype=float)
    if lattice_type != 'square':
        raise ValueError('Only square lattices are implemented.')
    if probabilities.ndim != 1 or probabilities.size == 0 or not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)) or np.unique(probabilities).size != probabilities.size:
        raise ValueError('p_data must be a nonempty sequence of distinct probabilities in [0, 1].')
    if sizes_input.ndim != 1 or sizes_input.size == 0:
        raise ValueError('dimensions must be a nonempty sequence.')
    # Require L >= 4 to avoid small periodic-grid degeneracies in the boundary checks.
    sizes = np.array([positive_integer(v, 'L', 4) for v in sizes_input])
    if np.unique(sizes).size != sizes.size:
        raise ValueError('dimensions must not contain duplicates.')
    target = positive_integer(REALIZATIONS, 'REALIZATIONS')
    seed = positive_integer(SEED, 'SEED', 0)
    if seed > 2**32 - 1:
        raise ValueError('SEED must be <= 2**32 - 1 for np.random.seed.')
    verify_boundary_conditions(sizes, PERIODIC)
    # Seed once for the entire run, not once per realization, so successive samples differ.
    np.random.seed(seed)
    folder = Path(data_folder)
    folder.mkdir(parents=True, exist_ok=True)
    boundary = 'PBC' if PERIODIC else 'OBC'
    storage = 'all' if STORE_ALL_REALIZATIONS else 'mean'
    path = folder / f'pauli_y_results_{boundary}_binary_rank_{storage}_Lmax_{L_MAX}.pkl'
    # Save the settings and conventions together with the data so the output remains interpretable.
    metadata = {
        'schema_version': 1, 'p_values': probabilities, 'L_values': sizes,
        'lattice_type': lattice_type, 'periodic': bool(PERIODIC), 'seed': seed,
        'store_all_realizations': bool(STORE_ALL_REALIZATIONS),
        'entropy_units': 'bits', 'entropy_method': 'binary_cut_rank_GF2',
        'measurement_protocol': 'Y_current_graph_LC_delete_with_local_corrections',
        'measurement_order': 'original_networkx_node_order',
        'cut': 'node[0] < L // 2', 'LCC_tie_break': 'first_in_node_order',
        'variance_ddof': 1,
        'rng_scheme': 'np.random.seed(seed); global RandomState MT19937; p,L,trial loop order',
    }
    # Store results as results[p][L], with one statistics entry for each parameter pair.
    data = dict(metadata, results={}, realizations=target)
    print(f'Protocol: {data["measurement_protocol"]}\nOutput: {path}')
    for p_index, p_value in enumerate(probabilities):
        p = float(p_value)
        per_p = data['results'].setdefault(p, {})
        for size in sizes:
            L = int(size)
            entry = per_p.setdefault(L, new_entry(STORE_ALL_REALIZATIONS))
            print(f'p={p:.6g}, L={L}: {target} realisations', flush=True)
            # Generate a fresh lattice and measurement pattern, then evaluate the surviving LCC.
            for trial in range(target):
                G, G_meas = measurement_pauli_y(p, L, L, periodic=PERIODIC)
                entropy = entanglement_network_DV(G, G_meas, L)
                lcc_size = max((len(c) for c in nx.connected_components(G_meas)), default=0)
                add_realization(entry, lcc_size, entropy, STORE_ALL_REALIZATIONS)
                # Release the graph objects before generating the next realization.
                del G, G_meas
            # M2 is only needed while accumulating statistics.
            for name in ('LCC_size', 'dv_entropy'):
                del entry[f'M2_{name}']
    # Write the output only after all parameter pairs finish; this is not an intermediate checkpoint.
    data['complete'] = True
    with path.open('wb') as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Complete: {path}')
    return path


# Run simulations when executed as a script, while allowing functions to be imported without running main().
if __name__ == '__main__':
    main()
