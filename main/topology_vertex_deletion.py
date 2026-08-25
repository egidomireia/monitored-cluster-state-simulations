
import numpy as np
import networkx as nx
import os
import pickle
from tqdm.auto import tqdm
import random
from joblib import Parallel, delayed 
import multiprocessing



# ==========================================
# 0. GRAPH GENERATION & MEASUREMENT
# ==========================================
def measurement_q(lattice_type, p_remove, x, y, weight):
    if lattice_type == 'square':
        G = nx.grid_2d_graph(x, y, periodic=False)
    elif lattice_type == 'triangular':
        G = nx.triangular_lattice_graph(x, y, periodic=False)
    elif lattice_type == 'hexagonal':
        G = nx.hexagonal_lattice_graph(x, y, periodic=False)
    elif lattice_type == 'cubic':
        G = nx.grid_graph(dim=(x, y, y), periodic=False)
    
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
# 1. TOPOLOGICAL EXTRACTION FUNCTION
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


# ==========================================
# NEW: THE WORKER FUNCTION (Updated for DV)
# ==========================================
def worker_realization(lattice_type, p, L, weight, cut_x):
    """
    This function does exactly ONE realization. 
    It will be executed simultaneously across dozens of different cores.
    Now uses extract_graph_data_all to get LCC, CV (eigenvalues), and DV (entropy).
    """
    G, G_meas = measurement_q(lattice_type, p, L, L, weight)
    # Ensure you are using the 'extract_graph_data_all' function defined previously
    lcc_size, eigenvalues, dv_entropy = extract_graph_data_all(G, G_meas, cut_x)
    return lcc_size, eigenvalues, dv_entropy

# ==========================================
# 2. PARAMETER SETUP 
# ==========================================
#dimensions = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450]) 
dimensions = np.array([500, 550, 600, 650, 700, 750]) 
#dimensions = np.array([650,700,750,800,850]) 
p_data = np.linspace(0.3, 0.5, 150)
weight_values = [1.0] 
REALIZATIONS = 1000
lattice_type = 'square' 

repo_path = r"data_files" 
str_folder = f"data_topology_{str(lattice_type)}_reps_2"
data_folder = os.path.join(repo_path, str_folder)
os.makedirs(data_folder, exist_ok=True)

total_cores = multiprocessing.cpu_count()
cores_to_use = max(1, 50) 

# ==========================================
# 3. MAIN SIMULATION LOOP (PARALLELIZED & BULLETPROOF)
# ==========================================
for weight in weight_values:
    print(f"\n--- Generating Topologies for weight={weight} ---")
    file_name = os.path.join(data_folder, f'Raw_Graph_Data_w_{weight}.pkl')
    
    # --- 1. LOAD BACKUP IF IT EXISTS ---
    if os.path.exists(file_name):
        print(f"[INFO] Backup found! Loading saved data from {file_name}...")
        with open(file_name, 'rb') as f:
            data_to_save = pickle.load(f)
    else:
        # Initialize fresh dictionary if no file exists
        data_to_save = {
            'p_values': p_data,
            'L_values': dimensions,
            'realizations': REALIZATIONS,
            'weight': weight,
            'results': {} 
        }
    
    for p in tqdm(p_data, desc="Calculating probabilities", position=0, leave=True):
        
        # --- 2. SAFE INITIALIZATION FOR p ---
        # Only create the dictionary for 'p' if it hasn't been created yet.
        # This prevents overwriting partial progress!
        if p not in data_to_save['results']:
            data_to_save['results'][p] = {}
        
        for L in dimensions:
            
            # --- 3. GRANULAR CHECKPOINTING FOR L ---
            # If this specific L has already been fully completed (200 realizations), skip it.
            if L in data_to_save['results'][p] and len(data_to_save['results'][p][L].get('LCC_sizes', [])) == REALIZATIONS:
                continue 
            
            # Prepare clean lists for this L (including the new dv_entropies)
            data_to_save['results'][p][L] = {
                'LCC_sizes': [], 
                'eigenvalues': [], 
                'dv_entropies': []
            }
            
            # --- CALCULATE CENTRAL CUT ---
            G_dummy, _ = measurement_q(lattice_type, 0.0, L, L, weight)
            pos_OG = nx.get_node_attributes(G_dummy, 'pos')
            if not pos_OG: pos_OG = {node: node for node in G_dummy.nodes()}
            all_x_coords = sorted(list(set([coords[0] for coords in pos_OG.values()])))
            mid_idx = len(all_x_coords) // 2
            cut_x = (all_x_coords[mid_idx] + all_x_coords[mid_idx - 1]) / 2.0
            
            # --- PARALLELIZATION ---
            parallel_results = Parallel(n_jobs=cores_to_use)(
                delayed(worker_realization)(lattice_type, p, L, weight, cut_x) 
                for _ in range(REALIZATIONS)
            )
            
            # Unpack the 3 variables and append them to the dictionaries
            for lcc_size, eigenvalues, dv_entropy in parallel_results:
                data_to_save['results'][p][L]['LCC_sizes'].append(lcc_size)
                data_to_save['results'][p][L]['eigenvalues'].append(eigenvalues)
                data_to_save['results'][p][L]['dv_entropies'].append(dv_entropy)

            # --- 4. ULTRA-FREQUENT SAVING ---
            # Saving inside the L loop guarantees that if the script crashes, 
            # you only lose the progress of the current L, not the whole p.
            with open(file_name, 'wb') as f:
                pickle.dump(data_to_save, f)
            
print("\nSimulation completed successfully! All topological data has been saved.")