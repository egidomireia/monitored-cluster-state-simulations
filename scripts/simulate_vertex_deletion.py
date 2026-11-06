from pathlib import Path
import os
import pickle
import random

import networkx as nx
import numpy as np


STORE_ALL_REALIZATIONS = False  # True to store all realizations, False to store only averages and variances.

# ==========================================
# 0. PARAMETER SETUP 
# ==========================================

L_MIN = 100
L_MAX = 900
L_STEP = 50

dimensions = np.arange(L_MIN, L_MAX + 1, L_STEP, dtype=int)


# ==========================================
# VERTEX-DELETION PROBABILITIES
# ==========================================

# defined as (start, stop, number of points).
P_INTERVALS = [
    (0.00, 0.38, 15),    # p<p_c
    (0.38, 0.42, 100),   # p near p_c
    (0.42, 1.00, 30),    # p>p_c
]

# Combine intervals, remove repeated endpoints, and sort.
p_data = np.unique(np.round( np.concatenate([np.linspace(start, stop, number) for start, stop, number in P_INTERVALS]), decimals=12))


weight_values = [1.0] # this can be an array of weights, e.g., [1.0, 2.0, 3.0], for the CV case

REALIZATIONS = 1000

lattice_type = 'square'


data_folder = Path(__file__).resolve().parents[1] / "data_files" / "vertex_deletion"
data_folder.mkdir(parents=True, exist_ok=True)

# Reproducible initial random seed.
SEED = 12345



# ==========================================
# 1. GRAPH GENERATION & MEASUREMENT
# ==========================================
def measurement_q(lattice_type, p_remove, x, y, weight):
    """
    Generate a lattice and independently delete vertices with probability p_remove.

    Notes
    -----
    Triangular, hexagonal, and cubic lattices were also used in earlier
    simulations and are retained here. The current study focuses on the
    square lattice with periodic boundary conditions.

    Parameters
    ----------
    lattice_type : str
        Lattice geometry: 'square', 'triangular', 'hexagonal', or 'cubic'.
    p_remove : float
        Probability of deleting each vertex, between 0 and 1.
    x, y : int
        Size parameters passed to the lattice generator.
        The cubic lattice uses dimensions (x, y, y).
    weight : float or tuple of float
        Constant edge weight or bounds (a, b) for independent, uniformly
        distributed weights assigned to surviving edges.

    Returns
    -------
    G : networkx.Graph
        Original lattice before vertex deletion.
    G_meas : networkx.Graph
        Remaining graph with edge weights assigned.

    Raises
    ------
    ValueError
        If lattice_type is unsupported or weight has an invalid shape.
    """
    if lattice_type == 'square':
        G = nx.grid_2d_graph(x, y, periodic=True)
    elif lattice_type == 'triangular':
        G = nx.triangular_lattice_graph(x, y, periodic=False)
    elif lattice_type == 'hexagonal':
        G = nx.hexagonal_lattice_graph(x, y, periodic=False)
    elif lattice_type == 'cubic':
        G = nx.grid_graph(dim=(x, y, y), periodic=False)
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")
    
    G_meas = G.copy()
    nodes_removed = [node for node in G if random.random() < p_remove]
    G_meas.remove_nodes_from(nodes_removed)
    
    if np.shape(weight) == ():
        # Assign the same scalar weight to all edges instantly
        nx.set_edge_attributes(G_meas, float(weight), 'weight')
    elif np.shape(weight) == (2,):
        random_weights = {(u, v): {'weight': random.uniform(weight[0], weight[1])} 
                          for u, v in G_meas.edges()}
        nx.set_edge_attributes(G_meas, random_weights)
    else:
        raise ValueError('Select a uniform weight or random between (a,b)')
        
    return G, G_meas


# ==========================================
# 2. TOPOLOGICAL EXTRACTION FUNCTION
# ==========================================
def extract_graph_data_all(G, G_meas, cut):
    """
    Extracts all three key metrics simultaneously to maximize CPU efficiency:
    1. Classical LCC size
    2. Eigenvalues of the cut (for Continuous Variables / CV Entropy)
    3. Matrix Rank of the cut (for Discrete Variables / DV Entropy)
    """
    # 1. Isolate the Largest Connected Component (LCC)
    G_cc = sorted(nx.connected_components(G_meas), key=len, reverse=True)
    if not G_cc or not G_cc[0]:
        return 0, np.array([]), 0.0  # LCC = 0, No eigenvalues, DV Entropy = 0.0
        
    G_largest_cc = G_meas.subgraph(G_cc[0])
    classical_LCC_size = len(G_largest_cc.nodes())

    # 2. Extract physical positions to perform the cut
    pos = nx.get_node_attributes(G, 'pos')
    if not pos: 
        pos = {node: node for node in G.nodes()}
    
    left_nodes = [n for n in G_largest_cc.nodes() if pos[n][0] < cut]
    right_nodes = [n for n in G_largest_cc.nodes() if pos[n][0] >= cut]

    crossing_edges = list(nx.edge_boundary(G_largest_cc, left_nodes, right_nodes, data=True))
    number_links = len(crossing_edges)

    if number_links == 0:
        return classical_LCC_size, np.array([]), 0.0
    
    # 3. Check for multiple connections (branching/bifurcations) at the cut
    has_multiple_connections = False
    seen_u, seen_v = set(), set()
    for u, v, data in crossing_edges:
        if u in seen_u or v in seen_v:
            has_multiple_connections = True
            break
        seen_u.add(u)
        seen_v.add(v)

    # 4. Calculate Metrics (CV & DV)
    if not has_multiple_connections:
        # OPTIMIZATION: 1-to-1 topology
        # CV: Eigenvalues are just the squared weights
        eigenvalues = np.array([data.get('weight', 1.0)**2 for _, _, data in crossing_edges])
        
        # DV: Rank of a diagonal matrix is exactly the number of its elements
        dv_entropy = float(number_links)
        
    else:
        # GENERAL CASE: Branched topology. Build the submatrix ONCE.
        total_nodes = len(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        
        indx_A = [node_to_idx[n] for n in left_nodes]
        indx_B = [node_to_idx[n] for n in G.nodes() if n not in left_nodes]

        nbrs = np.zeros((total_nodes, total_nodes))
        for u, v, data in G_largest_cc.edges(data=True):
            idx_u, idx_v = node_to_idx[u], node_to_idx[v]
            w = data.get('weight', 1.0)
            nbrs[idx_u, idx_v] = w
            nbrs[idx_v, idx_u] = w
            
        # Extract the bipartite adjacency matrix 
        V_AB = nbrs[np.ix_(indx_A, indx_B)]
        
        # --- CV ENTROPY METRIC ---
        eigenvalues = np.linalg.eigvalsh(np.dot(V_AB, V_AB.T))
        eigenvalues = eigenvalues[eigenvalues > 1e-10] 
        
        # --- DV ENTROPY METRIC ---
        # The rank relies on linear independence. It evaluates correctly 
        # whether the matrix has weights=1 or continuous uniform weights.
        dv_entropy = float(np.linalg.matrix_rank(V_AB))
        
    return classical_LCC_size, eigenvalues, dv_entropy



def main():
    random.seed(SEED)
    data_folder.mkdir(parents=True, exist_ok=True)

    for weight in weight_values:
        print(f"\nVertex deletion: weight = {weight}")

        data_to_save = {
            "p_values": p_data.copy(),
            "L_values": dimensions.copy(),
            "realizations": REALIZATIONS,
            "weight": weight,
            "lattice_type": lattice_type,
            "periodic": True,
            "store_all_realizations": STORE_ALL_REALIZATIONS,
            "seed": SEED,
            "results": {},
        }

        for p in p_data:
            data_to_save["results"][p] = {}

            for L in dimensions:
                L = int(L)
                cut_x = L // 2 - 0.5
                entry = {"LCC_sizes": [], "eigenvalues": [], "dv_entropies": []}

                print(f"p = {p:.5f}, L = {L}")

                for realization in range(REALIZATIONS):
                    G, G_meas = measurement_q(lattice_type, p, L, L, weight)
                    lcc_size, eigenvalues, dv_entropy = extract_graph_data_all(G, G_meas, cut_x)

                    entry["LCC_sizes"].append(lcc_size)
                    entry["eigenvalues"].append(eigenvalues)
                    entry["dv_entropies"].append(dv_entropy)

                    del G, G_meas

                if STORE_ALL_REALIZATIONS:
                    data_to_save["results"][p][L] = entry
                else:
                    lcc_sizes = np.asarray(entry["LCC_sizes"], dtype=float)
                    dv_entropies = np.asarray(entry["dv_entropies"], dtype=float)
                    number_realizations = len(lcc_sizes)

                    data_to_save["results"][p][L] = {
                        "number_realizations": number_realizations,
                        "mean_LCC_size": float(lcc_sizes.mean()),
                        "sem_LCC_size": float(lcc_sizes.std(ddof=1) / np.sqrt(number_realizations)) if number_realizations > 1 else None,
                        "mean_dv_entropy": float(dv_entropies.mean()),
                        "sem_dv_entropy": float(dv_entropies.std(ddof=1) / np.sqrt(number_realizations)) if number_realizations > 1 else None,
                        "cv_eigenvalue": float(weight**2),
                        "mean_cv_eigenvalue_count": float(dv_entropies.mean()),
                    }

        file_name = data_folder / f"vertex_deletion_results_PBC_w_{weight}.pkl"
        with file_name.open("wb") as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved: {file_name}")


if __name__ == "__main__":
    main()