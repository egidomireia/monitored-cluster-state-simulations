from pathlib import Path
import pickle

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
import scipy.linalg as la


STORE_ALL_REALIZATIONS = False  # True to store all realizations, False to store only averages and variances.

CONNECTIVITY_TOL = 1e-12  # Keep every nonzero link.  


# ==========================================
# 0. PARAMETER SETUP
# ==========================================
L_MIN = 20
L_MAX = 60
L_STEP = 10

dimensions = np.arange(L_MIN, L_MAX + 1, L_STEP, dtype=int)

# defined as (start, stop, number of points).
P_INTERVALS = [
    (0.00, 0.38, 6),    # p<p_c
    (0.38, 0.42, 4),   # p near p_c
    (0.42, 1.00, 8),    # p>p_c
]

# Combine intervals, remove repeated endpoints, and sort.
p_data = np.unique(np.round( np.concatenate([np.linspace(start, stop, number) for start, stop, number in P_INTERVALS]), decimals=12))

# p_data = np.array([0.6])    
# phi_values = np.r_[0.0, np.logspace(-12, -7, 26)].tolist()
# tolerance_values = np.r_[0.0, np.logspace(-12, -7, 26)].tolist()


weight_values = [1.0]
r_values = [1.0]
# phi_values = [1e-6]
REALIZATIONS = 20
lattice_type = "square"
SEED = 12345


# Place this script in the repository's scripts folder.
data_folder = Path(__file__).resolve().parents[1] / "data_files" / "homodyne_general" / "Scanning_phi_tolerance_SCAN_w_r"

# ============================================================
# 1. HOMODYNE MEASUREMENT
# ============================================================

def measurement_general_math_only(lattice_type, p_remove, phi, x, y, r, weight_range):
    """
    Creates the lattice, removes nodes, and applies homodyne measurements to the removed modes.

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
        Original lattice. Do not use this graph to find the LCC, because it still contains removed nodes.

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
        G = nx.grid_2d_graph(x, y, periodic=True)

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
        return (G, np.empty((0, 0), dtype=complex), [])
    
    node_to_idx = {node: index for index, node in enumerate(nodes_ini)}

    edges = list(G.edges())
    if edges:
        row = np.asarray([node_to_idx[u] for u, _ in edges], dtype=int)
        col = np.asarray([node_to_idx[v] for _, v in edges], dtype=int)

        if np.isscalar(weight_range):
            weights = np.full(len(edges), float(weight_range), dtype=float)
        else:
            weight_min, weight_max = map(float, weight_range)

            if weight_min > weight_max:
                raise ValueError('weight_range minimum is greater than maximum.')
            
            weights = np.random.uniform(weight_min, weight_max, size=len(edges))

        row_full = np.concatenate((row, col))
        col_full = np.concatenate((col, row))
        weights_full = np.concatenate((weights, weights))
        V_sparse = sp.csr_matrix((weights_full, (row_full, col_full)), shape=(total_nodes, total_nodes), dtype=float)

    else:
        V_sparse = sp.csr_matrix((total_nodes, total_nodes), dtype=float)

    remove_mask = np.random.random(total_nodes) < p_remove
    keep_indices = np.flatnonzero(~remove_mask)
    remove_indices = np.flatnonzero(remove_mask)

    if keep_indices.size == 0:
        return (G, np.empty((0, 0), dtype=complex), [])
    
    U_diag = np.full(total_nodes, np.exp(-2.0 * r), dtype=float)
    surviving_nodes = [nodes_ini[index] for index in keep_indices]

    if remove_indices.size == 0:
        Z_final = V_sparse.toarray().astype(np.complex128)
        diagonal = np.diag_indices_from(Z_final)
        Z_final[diagonal] += 1j * U_diag
    else:
        V_KK = V_sparse[keep_indices, :][:, keep_indices]
        V_KM = V_sparse[keep_indices, :][:, remove_indices]
        V_MK = V_sparse[remove_indices, :][:, keep_indices]
        V_MM = V_sparse[remove_indices, :][:, remove_indices]
        Z_KK_sparse = V_KK.astype(np.complex128) + 1j * sp.diags(U_diag[keep_indices])

        if phi == 0.0:
            Z_final = Z_KK_sparse.toarray()

        else:
            Z_MM_sparse = V_MM.astype(np.complex128) + 1j * sp.diags(U_diag[remove_indices])
            cot_phi = np.cos(phi) / np.sin(phi)
            Z_MM_shifted = Z_MM_sparse.toarray()
            diagonal = np.diag_indices_from(Z_MM_shifted)
            Z_MM_shifted[diagonal] += cot_phi
            Z_KM_dense = V_KM.toarray().astype(np.complex128)
            Z_MK_dense = V_MK.toarray().astype(np.complex128)
            X = la.solve(Z_MM_shifted, Z_MK_dense, assume_a='gen', check_finite=False)
            Z_final = Z_KK_sparse.toarray() - Z_KM_dense @ X

    Z_final = 0.5 * (Z_final + Z_final.T)

    if not np.all(np.isfinite(Z_final)):
        raise FloatingPointError("Z_final contains NaN or infinite values.")

    # Do not threshold the complete Z_final matrix here.
    # Its small imaginary diagonal contains physical squeezing.

    return G, Z_final, surviving_nodes


# ============================================================
# 2. LCC AND SYMPLECTIC EIGENVALUES
# ============================================================

def extract_graph_data_math_only(Z_final, surviving_nodes, cut, connectivity_tol=0.0):
    """
    Finds the largest connected component and calculates the
    symplectic eigenvalues of subsystem A.

    connectivity_tol is a nonnegative threshold on off-diagonal abs(Z).
    Zero keeps every nonzero link. Positive values exclude links strictly
    below the threshold from the LCC search, without altering Z_final.

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
    PHYSICAL_TOL = 1e-07

    Z_final = np.asarray(Z_final, dtype=np.complex128)
    if Z_final.ndim != 2:
        raise ValueError('Z_final must be a matrix.')
    if Z_final.shape[0] != Z_final.shape[1]:
        raise ValueError('Z_final must be square.')
    
    N = Z_final.shape[0]
    if N == 0:
        return (0, np.array([], dtype=float))
    if len(surviving_nodes) != N:
        raise ValueError('The number of surviving_nodes does not match the dimensions of Z_final.')
    if not np.all(np.isfinite(Z_final)):
        raise ValueError('Z_final contains NaN or infinite values.')
    
    Z_connectivity = np.abs(Z_final).astype(float, copy=True)
    np.fill_diagonal(Z_connectivity, 0.0)
    Z_connectivity[Z_connectivity < connectivity_tol] = 0.0
    connectivity_sparse = sp.csr_matrix(Z_connectivity)
    connectivity_sparse.eliminate_zeros()

    n_components, labels = csgraph.connected_components(connectivity_sparse, directed=False, return_labels=True)
    if n_components == 0:
        return (0, np.array([], dtype=float))
    
    counts = np.bincount(labels)
    largest_label = np.argmax(counts)
    lcc_indices = np.flatnonzero(labels == largest_label)
    N_cc = lcc_indices.size

    idx_A_local = np.asarray([local_index for local_index, global_index in enumerate(lcc_indices) if surviving_nodes[global_index][0] < cut], dtype=int)
    n_A = idx_A_local.size
    if n_A == 0 or n_A == N_cc:
        return (N_cc, np.array([], dtype=float))
    
    Z_lcc = Z_final[np.ix_(lcc_indices, lcc_indices)]
    V = np.real(Z_lcc)
    U = np.imag(Z_lcc)
    V = 0.5 * (V + V.T)
    U = 0.5 * (U + U.T)
    try:
        U_factor = la.cho_factor(U, lower=True, check_finite=False)
    except la.LinAlgError as error:
        minimum_eigenvalue = np.min(la.eigvalsh(U, check_finite=False))
        raise la.LinAlgError(f'Im(Z_lcc) is not positive definite. Smallest eigenvalue: {minimum_eigenvalue:.6e}') from error
    
    E_A = np.zeros((N_cc, n_A), dtype=float)
    E_A[idx_A_local, np.arange(n_A)] = 1.0

    U_inv_all_A = la.cho_solve(U_factor, E_A, check_finite=False)
    V_all_A = V[:, idx_A_local]
    V_A_all = V[idx_A_local, :]
    U_inv_V_all_A = la.cho_solve(U_factor, V_all_A, check_finite=False)

    XX_A = U_inv_all_A[idx_A_local, :]
    XP_A = U_inv_all_A.T @ V_all_A
    PX_A = V_A_all @ U_inv_all_A
    PP_A = U[np.ix_(idx_A_local, idx_A_local)] + V_A_all @ U_inv_V_all_A

    cov_A = 0.5 * np.block([[XX_A, XP_A], [PX_A, PP_A]])
    cov_A = 0.5 * (cov_A + cov_A.T)

    if not np.all(np.isfinite(cov_A)):
        raise FloatingPointError('The reduced covariance matrix contains NaN or infinite values.')
    
    identity_A = np.eye(n_A)
    zeros_A = np.zeros((n_A, n_A))
    Omega_A = np.block([[zeros_A, identity_A], [-identity_A, zeros_A]])
    try:
        L_chol = la.cholesky(cov_A, lower=True, check_finite=False)
    except la.LinAlgError as error:
        minimum_eigenvalue = np.min(la.eigvalsh(cov_A, check_finite=False))
        raise la.LinAlgError(f'The reduced covariance matrix is not positive definite. Smallest eigenvalue: {minimum_eigenvalue:.6e}') from error
    
    M = 1j * (L_chol.T @ Omega_A @ L_chol)
    M = 0.5 * (M + M.conj().T)

    mu = la.eigvalsh(M, check_finite=False)
    sigma = np.real(mu[n_A:])
    if sigma.size != n_A or np.any(sigma <= 0.0):
        raise ValueError('Invalid symplectic spectrum obtained.')
    
    minimum_sigma = np.min(sigma)
    if minimum_sigma < 0.5 - PHYSICAL_TOL:
        raise ValueError(f'Unphysical symplectic eigenvalue detected: {minimum_sigma:.10f}. Check the covariance matrix or numerical precision.')
    
    sigma = np.where(sigma < 0.5, 0.5, sigma)
    return (N_cc, sigma)


def cv_entropy(sigma):
    """Return the von Neumann entropy in bits from symplectic eigenvalues."""
    n = np.maximum(np.asarray(sigma, dtype=float) - 0.5, 0.0)
    n = n[n > 0]
    return float(np.sum((np.log1p(n) + n * np.log1p(1.0 / n)) / np.log(2.0)))



# ==========================================
# 3. MAIN SIMULATION LOOP (PARALLELIZED)
# ==========================================
def main():
    probabilities = np.asarray(p_data, dtype=float)
    sizes = np.asarray(dimensions)
    tolerances = np.asarray(tolerance_values, dtype=float)

    if lattice_type != "square":
        raise ValueError("This simulation loop uses the central cut of a square lattice.")

    if not isinstance(REALIZATIONS, (int, np.integer)) or REALIZATIONS < 1:
        raise ValueError("REALIZATIONS must be a positive integer.")

    if sizes.ndim != 1 or sizes.size == 0 or not np.all(np.isfinite(sizes) & (sizes >= 4) & (sizes == np.floor(sizes))):
        raise ValueError("System sizes must be integers L >= 4.")

    if probabilities.ndim != 1 or probabilities.size == 0 or not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    if tolerances.ndim != 1 or tolerances.size == 0 or not np.all(np.isfinite(tolerances) & (tolerances >= 0)):
        raise ValueError("Tolerances must be finite and nonnegative.")

    sizes = np.unique(sizes.astype(int))
    probabilities = np.unique(probabilities)
    tolerances = np.unique(tolerances)

    np.random.seed(SEED)
    data_folder.mkdir(parents=True, exist_ok=True)

    for r in r_values:
        for weight in weight_values:
            run_folder = data_folder / f"w_{float(weight)}_r_{float(r)}"
            run_folder.mkdir(parents=True, exist_ok=True)

            for phi in phi_values:
                print(f"\nHomodyne measurements: w={weight}, r={r}, phi={phi}")

                data_by_tolerance = {}

                for tolerance in tolerances:
                    data_by_tolerance[tolerance] = {
                        "p_values": probabilities.copy(),
                        "L_values": sizes.copy(),
                        "realizations": REALIZATIONS,
                        "weight": weight,
                        "r_squeezing": r,
                        "phi": phi,
                        "lattice_type": lattice_type,
                        "periodic": True,
                        "seed": SEED,
                        "store_all_realizations": STORE_ALL_REALIZATIONS,
                        "entropy_units": "bits",
                        "eigenvalues_type": "symplectic, vacuum = 0.5",
                        "connectivity_tolerance": float(tolerance),
                        "tolerance_mode": "connectivity_only",
                        "results": {},
                    }

                for p in probabilities:
                    for tolerance in tolerances:
                        data_by_tolerance[tolerance]["results"][p] = {}

                    for L in sizes:
                        L = int(L)
                        cut_x = L // 2 - 0.5
                        print(f"p={p:.6g}, L={L}")

                        entries = {}
                        means = {}
                        squared_deviations = {}

                        for tolerance in tolerances:
                            if STORE_ALL_REALIZATIONS:
                                entries[tolerance] = {
                                    "number_realizations": REALIZATIONS,
                                    "LCC_sizes": [],
                                    "eigenvalues": [],
                                    "cv_entropies": [],
                                }
                            else:
                                means[tolerance] = np.zeros(2, dtype=float)
                                squared_deviations[tolerance] = np.zeros(2, dtype=float)

                        for realization in range(REALIZATIONS):
                            # Use the same measured state for every tolerance.
                            G, Z_final, surviving_nodes = measurement_general_math_only(lattice_type, p, phi, L, L, r, weight)

                            for tolerance in tolerances:
                                lcc_size, sigma = extract_graph_data_math_only(Z_final, surviving_nodes, cut_x, connectivity_tol=tolerance)
                                entropy = cv_entropy(sigma)

                                if STORE_ALL_REALIZATIONS:
                                    entries[tolerance]["LCC_sizes"].append(int(lcc_size))
                                    entries[tolerance]["eigenvalues"].append(sigma)
                                    entries[tolerance]["cv_entropies"].append(entropy)
                                else:
                                    # Update the means and squared deviations without retaining samples.
                                    values = np.array([lcc_size, entropy], dtype=float)
                                    delta = values - means[tolerance]
                                    means[tolerance] += delta / (realization + 1)
                                    squared_deviations[tolerance] += delta * (values - means[tolerance])

                                del sigma

                            del G, Z_final, surviving_nodes

                        for tolerance in tolerances:
                            if STORE_ALL_REALIZATIONS:
                                entry = entries[tolerance]
                            else:
                                sem = np.sqrt(np.maximum(squared_deviations[tolerance], 0) / ((REALIZATIONS - 1) * REALIZATIONS)) if REALIZATIONS > 1 else [None, None]

                                entry = {
                                    "number_realizations": REALIZATIONS,
                                    "mean_LCC_size": float(means[tolerance][0]),
                                    "sem_LCC_size": float(sem[0]) if REALIZATIONS > 1 else None,
                                    "mean_cv_entropy": float(means[tolerance][1]),
                                    "sem_cv_entropy": float(sem[1]) if REALIZATIONS > 1 else None,
                                }

                            data_by_tolerance[tolerance]["results"][p][L] = entry

                storage = "all" if STORE_ALL_REALIZATIONS else "mean"
                if np.isscalar(weight):
                    weight_label = str(float(weight))
                else:
                    weight_array = np.atleast_1d(np.asarray(weight, dtype=float))
                    weight_label = "_".join(str(float(value)) for value in weight_array)

                for tolerance in tolerances:
                    file_name = run_folder / f"homodyne_results_PBC_phi_{float(phi)}_connect_{float(tolerance)}_{storage}_w_{weight_label}_r_{float(r)}.pkl"

                    with file_name.open("wb") as f:
                        pickle.dump(data_by_tolerance[tolerance], f, protocol=pickle.HIGHEST_PROTOCOL)

                    print(f"Saved: {file_name}")

    print("\nSimulation completed.")


if __name__ == "__main__":
    main()
