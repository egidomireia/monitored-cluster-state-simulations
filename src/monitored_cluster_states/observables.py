"""Functions for extracting observables from the simulation data."""

import pickle
import numpy as np


# =============================================================================
# VERTEX-DELETION DATA: q-HOMODYNE AND PAULI-Z
# =============================================================================

def initialize_data_q(file_name, r=1.0):
    """Load vertex-deletion data and calculate the LCC, CV entropy and DV entropy."""

    with open(file_name, "rb") as f:
        data = pickle.load(f)

    def percolation(p, L, data):
        """Calculate the mean LCC fraction for a fixed p and L."""

        total_nodes_original = L * L
        lcc_sizes = data["results"][p][L]["LCC_sizes"]

        p_cc_realizations = []
        for size in lcc_sizes:
            surviving_nodes = total_nodes_original * (1 - p)

            if surviving_nodes > 0:
                p_cc_realizations.append(size / total_nodes_original)
            else:
                p_cc_realizations.append(0.0)

        avg_p_cc = np.mean(p_cc_realizations)
        std_p_cc = np.std(p_cc_realizations)

        return avg_p_cc, std_p_cc

    def P_vs_L(data, p_target, L):
        """Calculate the mean LCC fraction using the stored p closest to p_target."""

        p_values = data["p_values"]
        idx = np.argmin(np.abs(p_values - p_target))
        p_closest = p_values[idx]

        lcc_sizes = data["results"][p_closest][L]["LCC_sizes"]
        total_nodes_original = L * L
        surviving_nodes = total_nodes_original * (1 - p_closest)

        p_cc_realizations = []
        for size in lcc_sizes:
            if surviving_nodes > 0:
                p_cc_realizations.append(size / total_nodes_original)
            else:
                p_cc_realizations.append(0.0)

        return np.mean(p_cc_realizations)

    def entropy_from_topology(p, L, data, r, meas):
        """Calculate the mean CV entropy and its standard deviation."""

        def exact_f_r(A_eff, r):
            nu = np.sqrt(1 + (A_eff ** 2) * np.exp(4 * r))
            eps = 1e-12
            term1 = ((nu + 1) / 2) * np.log2(((nu + 1) / 2) + eps)
            term2 = ((nu - 1) / 2) * np.log2(((nu - 1) / 2) + eps)
            return term1 - term2

        def all_equal(iterable):
            it = iter(iterable)
            return all(a == b for a, b in zip(it, it))

        spectra = data["results"][p][L]["eigenvalues"]
        entropies = []

        for eigenvalues in spectra:
            if meas in ["q", "z", "Z"]:
                if len(eigenvalues) == 0:
                    entropies.append(0.0)
                elif all_equal(eigenvalues):
                    weight = eigenvalues[0]
                    S = exact_f_r(weight, r) * len(eigenvalues)
                    entropies.append(S)
                else:
                    S = sum(exact_f_r(w, r) for w in eigenvalues)
                    entropies.append(S)

            elif meas in ["CV general", "CV", "general"]:
                if len(eigenvalues) == 0:
                    entropies.append(0.0)
                else:
                    n = eigenvalues - 0.5
                    h2 = (n + 1) * np.log2(n + 1) - n * np.log2(np.clip(n, 1e-15, None))
                    entropies.append(np.sum(h2) / L)

        S_matrix = np.mean(entropies)
        S_matrix_std = np.std(entropies, ddof=1)

        return S_matrix, S_matrix_std

    def entropy_DV(p, L, data):
        """Calculate the mean DV entropy and its standard deviation."""

        stored_entropies = data["results"][p][L]["dv_entropies"]
        entropies = []

        for entropy in stored_entropies:
            entropies.append(entropy)

        S_matrix = np.mean(entropies)
        S_matrix_std = np.std(entropies, ddof=1)

        return S_matrix, S_matrix_std

    p_values = data["p_values"]
    dimensions = data["L_values"]

    C_matrix = np.zeros((len(dimensions), len(p_values)))
    C_matrix_std = np.zeros((len(dimensions), len(p_values)))
    S_matrix = np.zeros((len(dimensions), len(p_values)))
    S_matrix_std = np.zeros((len(dimensions), len(p_values)))
    S_DV_matrix = np.zeros((len(dimensions), len(p_values)))
    S_DV_matrix_std = np.zeros((len(dimensions), len(p_values)))

    for i, L in enumerate(dimensions):
        for j, p in enumerate(p_values):
            C_matrix[i, j], C_matrix_std[i, j] = percolation(p, L, data)
            S_matrix[i, j], S_matrix_std[i, j] = entropy_from_topology(p, L, data, r, meas="q")
            S_DV_matrix[i, j], S_DV_matrix_std[i, j] = entropy_DV(p, L, data)

    return (
        p_values, dimensions, C_matrix, C_matrix_std, S_matrix, S_matrix_std,
        S_DV_matrix, S_DV_matrix_std
    )


# =============================================================================
# GENERAL HOMODYNE-MEASUREMENT DATA
# =============================================================================

def initialize_data(file_name, meas, r=1.0):
    """Load general homodyne data and calculate the LCC and CV entropy."""

    with open(file_name, "rb") as f:
        data = pickle.load(f)

    def percolation(p, L, data):
        """Calculate the mean LCC fraction for a fixed p and L."""

        total_nodes_original = L * L
        lcc_sizes = data["results"][p][L]["LCC_sizes"]

        p_cc_realizations = []
        for size in lcc_sizes:
            surviving_nodes = total_nodes_original * (1 - p)

            if surviving_nodes > 0:
                p_cc_realizations.append(size / total_nodes_original)
            else:
                p_cc_realizations.append(0.0)

        avg_p_cc = np.mean(p_cc_realizations)
        std_p_cc = np.std(p_cc_realizations)

        return avg_p_cc, std_p_cc

    def P_vs_L(data, p_target, L):
        """Calculate the mean LCC fraction using the stored p closest to p_target."""

        p_values = data["p_values"]
        idx = np.argmin(np.abs(p_values - p_target))
        p_closest = p_values[idx]

        lcc_sizes = data["results"][p_closest][L]["LCC_sizes"]
        total_nodes_original = L * L
        surviving_nodes = total_nodes_original * (1 - p_closest)

        p_cc_realizations = []
        for size in lcc_sizes:
            if surviving_nodes > 0:
                p_cc_realizations.append(size / total_nodes_original)
            else:
                p_cc_realizations.append(0.0)

        return np.mean(p_cc_realizations)

    def entropy_from_topology(p, L, data, r, meas="q"):
        """Calculate the mean CV entropy and its standard deviation."""

        def exact_f_r(A_eff, r):
            nu = np.sqrt(1 + (A_eff ** 2) * np.exp(4 * r))
            eps = 1e-12
            term1 = ((nu + 1) / 2) * np.log2(((nu + 1) / 2) + eps)
            term2 = ((nu - 1) / 2) * np.log2(((nu - 1) / 2) + eps)
            return term1 - term2

        def all_equal(iterable):
            it = iter(iterable)
            return all(a == b for a, b in zip(it, it))

        spectra = data["results"][p][L]["eigenvalues"]
        entropies = []

        for eigenvalues in spectra:
            if meas in ["q", "z", "Z"]:
                if len(eigenvalues) == 0:
                    entropies.append(0.0)
                elif all_equal(eigenvalues):
                    weight = eigenvalues[0]
                    S = exact_f_r(weight, r) * len(eigenvalues)
                    entropies.append(S / L)
                else:
                    S = sum(exact_f_r(w, r) for w in eigenvalues)
                    entropies.append(S / L)

            elif meas in ["CV general", "CV", "general"]:
                if len(eigenvalues) == 0:
                    entropies.append(0.0)
                else:
                    n = eigenvalues - 0.5
                    h2 = (n + 1) * np.log2(n + 1) - n * np.log2(np.clip(n, 1e-15, None))
                    entropies.append(np.sum(h2))

        S_matrix = np.mean(entropies)
        S_matrix_std = np.std(entropies, ddof=1)

        return S_matrix, S_matrix_std

    def entropy_DV(p, L, data):
        """Calculate the mean DV entropy density and its standard deviation."""

        stored_entropies = data["results"][p][L]["dv_entropies"]
        entropies = []

        for entropy in stored_entropies:
            entropies.append(entropy / L)

        S_matrix = np.mean(entropies)
        S_matrix_std = np.std(entropies, ddof=1)

        return S_matrix, S_matrix_std

    p_values_full = data["p_values"]
    dimensions = data["L_values"]
    valid_p_values = []

    for p in p_values_full:
        if p in data["results"] and len(data["results"][p]) == len(dimensions):
            valid_p_values.append(p)

    p_values = np.array(valid_p_values)

    if len(p_values) == 0:
        print("[WARNING] No complete data found in this file!")
        return [], [], [], [], [], []

    print(
        f"[INFO] Successfully loaded {len(p_values)} / "
        f"{len(p_values_full)} completed p-values."
    )

    C_matrix = np.zeros((len(dimensions), len(p_values)))
    C_matrix_std = np.zeros((len(dimensions), len(p_values)))
    S_matrix = np.zeros((len(dimensions), len(p_values)))
    S_matrix_std = np.zeros((len(dimensions), len(p_values)))

    for i, L in enumerate(dimensions):
        for j, p in enumerate(p_values):
            C_matrix[i, j], C_matrix_std[i, j] = percolation(p, L, data)
            S_matrix[i, j], S_matrix_std[i, j] = entropy_from_topology(p, L, data, r, meas)

    return p_values, dimensions, C_matrix, C_matrix_std, S_matrix, S_matrix_std