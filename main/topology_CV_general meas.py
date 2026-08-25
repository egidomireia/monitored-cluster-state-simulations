import os

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
import scipy.linalg as la
import pickle
import random
from tqdm.auto import tqdm
from joblib import Parallel, delayed 
import multiprocessing
import gc # Added for garbage collection

# ============================================================
# 1. HOMODYNE MEASUREMENT
# ============================================================

def measurement_general_math_only(lattice_type, p_remove, phi, x, y, r, weight_range):
    """
    Creates the lattice, removes nodes, and applies homodyne
    measurements to the removed modes.

    Parameters
    ----------
    p_remove : float
        Probability of removing/measuring each node.

    phi : float
        Homodyne measurement angle in radians.

    r : float
        Squeezing parameter.

    Returns
    -------
    G : networkx.Graph
        Original lattice. Do not use this graph to find the LCC,
        because it still contains removed nodes.

    Z_final : ndarray, complex
        Complex Gaussian graph matrix of the surviving modes.

    surviving_nodes : list
        Coordinates of the modes retained in Z_final.
    """

    if not 0.0 <= p_remove <= 1.0:
        raise ValueError("p_remove must be between 0 and 1.")

    # --------------------------------------------------------
    # Create lattice
    # --------------------------------------------------------

    if lattice_type == "square":
        G = nx.grid_2d_graph(x, y, periodic=False)

    elif lattice_type == "triangular":
        G = nx.triangular_lattice_graph(x, y, periodic=False)

    elif lattice_type == "hexagonal":
        G = nx.hexagonal_lattice_graph(x, y, periodic=False)

    elif lattice_type == "cubic":
        G = nx.grid_graph(dim=(x, y, y), periodic=False)

    else:
        raise ValueError(
            f"Unknown lattice_type: {lattice_type}"
        )

    nodes_ini = list(G.nodes())
    total_nodes = len(nodes_ini)

    if total_nodes == 0:
        return G, np.empty((0, 0), dtype=complex), []

    node_to_idx = {
        node: index for index, node in enumerate(nodes_ini)
    }

    edges = list(G.edges())

    # --------------------------------------------------------
    # Sparse real adjacency matrix V
    # --------------------------------------------------------

    if edges:
        row = np.asarray(
            [node_to_idx[u] for u, _ in edges],
            dtype=int
        )

        col = np.asarray(
            [node_to_idx[v] for _, v in edges],
            dtype=int
        )

        if np.isscalar(weight_range):
            weights = np.full(
                len(edges),
                float(weight_range),
                dtype=float
            )
        else:
            weight_min, weight_max = map(float, weight_range)

            if weight_min > weight_max:
                raise ValueError(
                    "weight_range minimum is greater than maximum."
                )

            weights = np.random.uniform(
                weight_min,
                weight_max,
                size=len(edges)
            )

        row_full = np.concatenate((row, col))
        col_full = np.concatenate((col, row))
        weights_full = np.concatenate((weights, weights))

        V_sparse = sp.csr_matrix(
            (weights_full, (row_full, col_full)),
            shape=(total_nodes, total_nodes),
            dtype=float
        )

    else:
        V_sparse = sp.csr_matrix(
            (total_nodes, total_nodes),
            dtype=float
        )

    # --------------------------------------------------------
    # Select measured and surviving modes
    # --------------------------------------------------------

    remove_mask = (
        np.random.random(total_nodes) < p_remove
    )

    keep_indices = np.flatnonzero(~remove_mask)
    remove_indices = np.flatnonzero(remove_mask)

    if keep_indices.size == 0:
        return G, np.empty((0, 0), dtype=complex), []

    U_diag = np.full(
        total_nodes,
        np.exp(-2.0 * r),
        dtype=float
    )

    surviving_nodes = [
        nodes_ini[index] for index in keep_indices
    ]

    # --------------------------------------------------------
    # No nodes measured
    # --------------------------------------------------------

    if remove_indices.size == 0:
        Z_final = V_sparse.toarray().astype(
            np.complex128
        )

        diagonal = np.diag_indices_from(Z_final)
        Z_final[diagonal] += 1j * U_diag

    # --------------------------------------------------------
    # Some nodes measured
    # --------------------------------------------------------

    else:
        V_KK = V_sparse[keep_indices, :][:, keep_indices]
        V_KM = V_sparse[keep_indices, :][:, remove_indices]
        V_MK = V_sparse[remove_indices, :][:, keep_indices]
        V_MM = V_sparse[remove_indices, :][:, remove_indices]

        Z_KK_sparse = (
            V_KK.astype(np.complex128)
            + 1j * sp.diags(U_diag[keep_indices])
        )

        # q-homodyne measurement: simple node deletion
        if abs(np.sin(phi)) < 1e-12:
            Z_final = Z_KK_sparse.toarray()

        else:
            Z_MM_sparse = (
                V_MM.astype(np.complex128)
                + 1j * sp.diags(U_diag[remove_indices])
            )

            cot_phi = np.cos(phi) / np.sin(phi)

            Z_MM_shifted = Z_MM_sparse.toarray()

            diagonal = np.diag_indices_from(
                Z_MM_shifted
            )

            Z_MM_shifted[diagonal] += cot_phi

            Z_KM_dense = V_KM.toarray().astype(
                np.complex128
            )

            Z_MK_dense = V_MK.toarray().astype(
                np.complex128
            )

            # Solve instead of explicitly calculating an inverse
            X = la.solve(
                Z_MM_shifted,
                Z_MK_dense,
                assume_a="gen",
                check_finite=False
            )

            Z_final = (
                Z_KK_sparse.toarray()
                - Z_KM_dense @ X
            )

    # A Gaussian graph matrix must be complex symmetric,
    # not Hermitian. Therefore use .T, not .conj().T.
    Z_final = 0.5 * (Z_final + Z_final.T)

    if not np.all(np.isfinite(Z_final)):
        raise FloatingPointError(
            "Z_final contains NaN or infinite values."
        )

    # Do not threshold the complete Z_final matrix here.
    # Its small imaginary diagonal contains physical squeezing.

    return G, Z_final, surviving_nodes


# ============================================================
# 2. LCC AND SYMPLECTIC EIGENVALUES
# ============================================================

def extract_graph_data_math_only(
    Z_final,
    surviving_nodes,
    cut
):
    """
    Finds the largest connected component and calculates the
    symplectic eigenvalues of subsystem A.

    Subsystem A contains nodes satisfying:

        node_coordinate[0] < cut

    Returns
    -------
    N_cc : int
        Number of modes in the largest connected component.

    sigma : ndarray
        Symplectic eigenvalues of the reduced covariance matrix.
        The vacuum value is 0.5.
    """

    CONNECTIVITY_TOL = 1e-12
    PHYSICAL_TOL = 1e-7

    Z_final = np.asarray(
        Z_final,
        dtype=np.complex128
    )

    if Z_final.ndim != 2:
        raise ValueError("Z_final must be a matrix.")

    if Z_final.shape[0] != Z_final.shape[1]:
        raise ValueError("Z_final must be square.")

    N = Z_final.shape[0]

    if N == 0:
        return 0, np.array([], dtype=float)

    if len(surviving_nodes) != N:
        raise ValueError(
            "The number of surviving_nodes does not match "
            "the dimensions of Z_final."
        )

    if not np.all(np.isfinite(Z_final)):
        raise ValueError(
            "Z_final contains NaN or infinite values."
        )

    # --------------------------------------------------------
    # 1. Find the largest connected component
    # --------------------------------------------------------
    # Use a separate matrix for connectivity. Do not alter
    # Z_final, since its diagonal is needed for the covariance.

    Z_connectivity = np.abs(Z_final).astype(
        float,
        copy=True
    )

    np.fill_diagonal(Z_connectivity, 0.0)

    # Remove numerical Schur-complement noise
    Z_connectivity[
        Z_connectivity < CONNECTIVITY_TOL
    ] = 0.0

    connectivity_sparse = sp.csr_matrix(
        Z_connectivity
    )

    connectivity_sparse.eliminate_zeros()

    n_components, labels = csgraph.connected_components(
        connectivity_sparse,
        directed=False,
        return_labels=True
    )

    if n_components == 0:
        return 0, np.array([], dtype=float)

    counts = np.bincount(labels)
    largest_label = np.argmax(counts)

    lcc_indices = np.flatnonzero(
        labels == largest_label
    )

    N_cc = lcc_indices.size

    # --------------------------------------------------------
    # 2. Define subsystem A inside the LCC
    # --------------------------------------------------------

    idx_A_local = np.asarray(
        [
            local_index
            for local_index, global_index
            in enumerate(lcc_indices)
            if surviving_nodes[global_index][0] < cut
        ],
        dtype=int
    )

    n_A = idx_A_local.size

    if n_A == 0 or n_A == N_cc:
        return N_cc, np.array([], dtype=float)

    Z_lcc = Z_final[
        np.ix_(lcc_indices, lcc_indices)
    ]

    # Preserve the diagonal of V. Measurements at nonzero phi
    # can generate physically meaningful diagonal terms.
    V = np.real(Z_lcc)
    U = np.imag(Z_lcc)

    # Remove only numerical asymmetry
    V = 0.5 * (V + V.T)
    U = 0.5 * (U + U.T)

    # --------------------------------------------------------
    # 3. Cholesky factorization of U
    # --------------------------------------------------------
    # U must be positive definite for a valid pure Gaussian
    # graph state.

    try:
        U_factor = la.cho_factor(
            U,
            lower=True,
            check_finite=False
        )
    except la.LinAlgError as error:
        minimum_eigenvalue = np.min(
            la.eigvalsh(U, check_finite=False)
        )

        raise la.LinAlgError(
            "Im(Z_lcc) is not positive definite. "
            f"Smallest eigenvalue: {minimum_eigenvalue:.6e}"
        ) from error

    # E_A selects the columns corresponding to subsystem A
    E_A = np.zeros((N_cc, n_A), dtype=float)
    E_A[idx_A_local, np.arange(n_A)] = 1.0

    # Selected columns of U^{-1}
    U_inv_all_A = la.cho_solve(
        U_factor,
        E_A,
        check_finite=False
    )

    V_all_A = V[:, idx_A_local]
    V_A_all = V[idx_A_local, :]

    # U^{-1} V[:, A]
    U_inv_V_all_A = la.cho_solve(
        U_factor,
        V_all_A,
        check_finite=False
    )

    # --------------------------------------------------------
    # 4. Reduced covariance matrix
    # --------------------------------------------------------
    # Full covariance convention:
    #
    # covariance = 1/2 [
    #     U^-1,          U^-1 V
    #     V U^-1,        U + V U^-1 V
    # ]

    XX_A = U_inv_all_A[idx_A_local, :]

    XP_A = U_inv_all_A.T @ V_all_A

    PX_A = V_A_all @ U_inv_all_A

    PP_A = (
        U[np.ix_(idx_A_local, idx_A_local)]
        + V_A_all @ U_inv_V_all_A
    )

    cov_A = 0.5 * np.block([
        [XX_A, XP_A],
        [PX_A, PP_A]
    ])

    cov_A = 0.5 * (cov_A + cov_A.T)

    if not np.all(np.isfinite(cov_A)):
        raise FloatingPointError(
            "The reduced covariance matrix contains "
            "NaN or infinite values."
        )

    # --------------------------------------------------------
    # 5. Symplectic eigenvalues
    # --------------------------------------------------------

    identity_A = np.eye(n_A)
    zeros_A = np.zeros((n_A, n_A))

    Omega_A = np.block([
        [zeros_A, identity_A],
        [-identity_A, zeros_A]
    ])

    try:
        L_chol = la.cholesky(
            cov_A,
            lower=True,
            check_finite=False
        )
    except la.LinAlgError as error:
        minimum_eigenvalue = np.min(
            la.eigvalsh(cov_A, check_finite=False)
        )

        raise la.LinAlgError(
            "The reduced covariance matrix is not positive "
            "definite. "
            f"Smallest eigenvalue: {minimum_eigenvalue:.6e}"
        ) from error

    # Hermitian matrix whose eigenvalues occur in ±sigma pairs
    M = 1j * (L_chol.T @ Omega_A @ L_chol)

    # Remove numerical non-Hermiticity
    M = 0.5 * (M + M.conj().T)

    mu = la.eigvalsh(
        M,
        check_finite=False
    )

    # eigvalsh sorts the eigenvalues:
    # [-sigma_n, ..., -sigma_1, sigma_1, ..., sigma_n]
    sigma = np.real(mu[n_A:])

    if sigma.size != n_A or np.any(sigma <= 0.0):
        raise ValueError(
            "Invalid symplectic spectrum obtained."
        )

    # Physical covariance matrices require sigma >= 1/2
    minimum_sigma = np.min(sigma)

    if minimum_sigma < 0.5 - PHYSICAL_TOL:
        raise ValueError(
            "Unphysical symplectic eigenvalue detected: "
            f"{minimum_sigma:.10f}. "
            "Check the covariance matrix or numerical precision."
        )

    # Correct only tiny floating-point deviations below 0.5
    sigma = np.where(
        sigma < 0.5,
        0.5,
        sigma
    )

    return N_cc, sigma

# ==========================================
# 3. THE WORKER
# ==========================================
def worker_realization(lattice_type, p, phi, L, r, weight, cut_x):
    G_dummy, Z_final, surviving_nodes = measurement_general_math_only(lattice_type, p, phi, L, L, r, weight)
    if len(surviving_nodes) == 0:
        return 0, np.array([])
    lcc_size, eigenvalues = extract_graph_data_math_only(Z_final, surviving_nodes, cut_x)
    return lcc_size, eigenvalues

# ==========================================
# 2. PARAMETER SETUP
# ==========================================
#dimensions = np.array([50, 100, 150, 200, 250, 300, 350, 400]) 
dimensions = np.array([30,40,50,60,70])
p_data = np.linspace(0.0, 1.0, 200)
weight_values = [1.0] 
r_values = [1.0]
#phi_values = [0.0, np.pi/2, np.pi/4]
phi_values = [np.pi/2, np.pi/4]

REALIZATIONS = 500
lattice_type = 'square' 

repo_path = r"data_files" 
str_folder = f"data_topology_{str(lattice_type)}_general_repeat_1"
data_folder = os.path.join(repo_path, str_folder)
os.makedirs(data_folder, exist_ok=True)

total_cores = multiprocessing.cpu_count()
cores_to_use = max(1, 50) # Leave 2 cores free for the OS

# ==========================================
# 3. MAIN SIMULATION LOOP (PARALLELIZED)
# ==========================================
for r in r_values:
    for weight in weight_values:
        for phi in phi_values:
            
            print(f"\n--- Generating Topologies (w={weight}, r={r}, phi={phi:.3f}) ---")
            
            file_suffix = f"w_{float(weight)}_r_{r}_phi_{phi:.4f}"
            file_name = os.path.join(data_folder, f'Raw_Graph_Data_{file_suffix}.pkl')
            
            if os.path.exists(file_name):
                print(f"[INFO] Backup found! Loading saved data from {file_name}...")
                with open(file_name, 'rb') as f:
                    data_to_save = pickle.load(f)
            else:
                data_to_save = {
                    'p_values': p_data,
                    'L_values': dimensions,
                    'realizations': REALIZATIONS,
                    'weight': weight,
                    'r_squeezing': r,
                    'phi': phi,
                    'results': {} 
                }
            
            for p in tqdm(p_data, desc=f"Calculating p (phi={phi:.3f})"):
                if p in data_to_save['results'] and len(data_to_save['results'][p]) == len(dimensions):
                    continue
                    
                data_to_save['results'][p] = {}
                
                for L in dimensions:
                    data_to_save['results'][p][L] = {'LCC_sizes': [], 'eigenvalues': []}
                    
                    # --- CALCULATE CENTRAL CUT ---
                    G_dummy, _, _ = measurement_general_math_only(lattice_type, 0.0, phi, L, L, r, weight)
                    pos_OG = nx.get_node_attributes(G_dummy, 'pos')
                    if not pos_OG: pos_OG = {node: node for node in G_dummy.nodes()}
                    all_x_coords = sorted(list(set([coords[0] for coords in pos_OG.values()])))
                    mid_idx = len(all_x_coords) // 2
                    cut_x = (all_x_coords[mid_idx] + all_x_coords[mid_idx - 1]) / 2.0
                    
                    # ==================================================
                    # THE MEMORY SAVIOR: DYNAMIC CORE THROTTLING
                    # ==================================================
                    # Calculate effective surviving nodes: N_eff = L^2 * (1 - p)
                    # ==================================================
                    # THE MEMORY SAVIOR: STRICT DYNAMIC THROTTLING
                    # ==================================================
                    # Calculate effective surviving nodes: N_eff = L^2 * (1 - p)
                    N_eff = (L**2) * (1.0 - p)
                    
                    # LAPACK eigvals requires massive workspace memory. 
                    # A 10k node graph needs ~14 GB per worker at peak.
                    est_gb_per_worker = 14.0 * (N_eff / 10000.0)**2 
                    
                    # Target max RAM usage: ~140 GB (Leaves ~40 GB free for OS safety)
                    safe_cores = max(1, int(140.0 / max(est_gb_per_worker, 0.1)))
                    cores_to_use_dynamic = min(cores_to_use, safe_cores)
                    
                    actual_realizations = REALIZATIONS
                    if p == 0.0 and np.isscalar(weight):
                        actual_realizations = 1
                        cores_to_use_dynamic = 1

                    # --- BULLETPROOF PARALLEL EXECUTION ---
                    # max_tasks_per_child=1 forces the OS to kill the process and reclaim 
                    # ALL leaked memory after a single run.
                    # pre_dispatch='n_jobs' stops joblib from queuing arrays into RAM.
                    parallel_results = Parallel(
                        n_jobs=cores_to_use_dynamic, 
                        batch_size=1, 
                        pre_dispatch='n_jobs', 
                        max_tasks_per_child=1
                    )(
                        delayed(worker_realization)(lattice_type, p, phi, L, r, weight, cut_x) 
                        for _ in range(actual_realizations)
                    )
                    
                    if p == 0.0 and np.isscalar(weight):
                        parallel_results = [parallel_results[0]] * REALIZATIONS
                    
                    for lcc_size, eigenvalues in parallel_results:
                        data_to_save['results'][p][L]['LCC_sizes'].append(lcc_size)
                        data_to_save['results'][p][L]['eigenvalues'].append(eigenvalues)
                        
                    # Force Python to clean up the main loop memory
                    gc.collect()
                    
                # --- FIXED: INDENTED PROGRESSIVE SAVE ---
                # This must be INSIDE the p loop, but OUTSIDE the L loop.
                # Now it saves the backup instantly after every p step!
                with open(file_name, 'wb') as f:
                    pickle.dump(data_to_save, f)

print("\nSimulation completed successfully! All topological data has been saved.")