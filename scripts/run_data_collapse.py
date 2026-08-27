"""Prepare entropy data, run autoScale, estimate S+1 errors, and plot collapse.

Important statistical convention
--------------------------------
The third column in every autoScale input file is the standard error of the
mean (SEM), not the standard deviation. Points with an undefined, non-finite,
or exactly zero SEM are omitted because autoScale uses this value as a weight
and therefore requires dy > 0.
"""

from __future__ import annotations

import csv
import itertools
import math
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from monitored_cluster_states.paths import (
    DATA_DIR,
    SCALING_RESULTS_DIR,
    AUTOSCALE_SCRIPT,
)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# File path to the topological data (pickle file).
DATA_FILE = (
    DATA_DIR
    / "vertex_deletion"
    / "Raw_Graph_Data_w_1.0.pkl"
)

# Directory where the autoScale .dat files and results will be saved.
OUTPUT_DIR = SCALING_RESULTS_DIR / "vertex_deletion"

# Use the corrected Python 3 autoScale file supplied previously.
# AUTOSCALE_SCRIPT = AUTOSCALE_SCRIPT

R_VALUE = 5
MIN_L_THRESHOLD = 300
MIN_VALID_POINTS_PER_SIZE = 3
MIN_INCLUDED_SYSTEM_SIZES = 3

# Initial guesses for the multi-start optimization.
# This grid performs 2 * 20 * 5 = 200 independent autoScale fits.
# XC_GUESSES = np.linspace(0.406, 0.407, 2, dtype=float)
A_GUESSES = np.linspace(0.70, 0.80, 25, dtype=float)
B_GUESSES = np.linspace(0.09, 0.13, 25, dtype=float)
XC_GUESSES = np.array([0.407], dtype=float)
# A_GUESSES = np.array([0.75], dtype=float)
# B_GUESSES = np.array([0.105], dtype=float)

# Range of the SCALED coordinate x = (p - pc) L^a.
MIN_SCALED_X = -0.15
MAX_SCALED_X = 0.15


# =============================================================================
# 2. STATISTICS AND OBSERVABLES
# =============================================================================

def mean_and_sem(values: Iterable[float], context: str = "values") -> Tuple[float, float]:
    """Return the arithmetic mean and sample SEM.

    A zero SEM is returned as zero here and is handled during export by
    omitting the affected point. This is legitimate when every realization is
    identical, for example far from the critical region.
    """

    array = np.asarray(list(values), dtype=float).reshape(-1)

    if array.size < 2:
        raise ValueError(
            f"{context}: at least two realizations are required to estimate a SEM"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context}: realizations contain NaN or infinity")

    mean = float(np.mean(array))
    sem = float(np.std(array, ddof=1) / math.sqrt(array.size))
    return mean, sem


def percolation(p: float, L: int, data: Mapping) -> Tuple[float, float]:
    """Return mean LCC fraction and its SEM for one (p, L)."""

    total_nodes = L * L
    lcc_sizes = data["results"][p][L]["LCC_sizes"]

    fractions = []
    for size in lcc_sizes:
        if total_nodes * (1.0 - p) > 0.0:
            fractions.append(float(size) / total_nodes)
        else:
            fractions.append(0.0)

    return mean_and_sem(fractions, context=f"percolation p={p}, L={L}")


def P_vs_L(data: Mapping, p_target: float, L: int) -> float:
    """Return the mean LCC fraction at the stored p closest to p_target."""

    p_values = np.asarray(data["p_values"], dtype=float)
    p_closest = data["p_values"][int(np.argmin(np.abs(p_values - p_target)))]
    mean, _ = percolation(p_closest, L, data)
    return mean


def entropy_from_topology(
    p: float, L: int, data: Mapping, r: float
) -> Tuple[float, float]:
    """Return the mean topology entropy per L and its SEM."""

    def exact_f_r(A_eff: float, squeezing: float) -> float:
        nu = math.sqrt(1.0 + (float(A_eff) ** 2) * math.exp(4.0 * squeezing))
        plus = (nu + 1.0) / 2.0
        minus = (nu - 1.0) / 2.0

        # lim(x log x) = 0 as x -> 0, so handle minus == 0 explicitly.
        term_plus = plus * math.log2(plus)
        term_minus = 0.0 if minus == 0.0 else minus * math.log2(minus)
        return term_plus - term_minus

    spectra = data["results"][p][L]["eigenvalues"]
    entropies: List[float] = []

    for eigenvalues in spectra:
        eigenvalues = np.asarray(eigenvalues, dtype=float).reshape(-1)
        if eigenvalues.size == 0:
            entropies.append(0.0)
        else:
            entropy = sum(exact_f_r(weight, r) for weight in eigenvalues) / L
            entropies.append(float(entropy))

    return mean_and_sem(entropies, context=f"topology entropy p={p}, L={L}")


def entropy_DV(p: float, L: int, data: Mapping) -> Tuple[float, float]:
    """Return the mean DV entropy per L and its SEM."""

    entropies = [float(entropy) / L for entropy in data["results"][p][L]["dv_entropies"]]
    return mean_and_sem(entropies, context=f"DV entropy p={p}, L={L}")


# =============================================================================
# 3. AUTOSCALE INPUT/OUTPUT HELPERS
# =============================================================================

FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

SCORE_PATTERN = re.compile(
    rf"xc\s*=\s*({FLOAT_PATTERN}).*?"
    rf"\ba\s*=\s*({FLOAT_PATTERN}).*?"
    rf"\bb\s*=\s*({FLOAT_PATTERN}).*?"
    rf"\bS\s*=\s*({FLOAT_PATTERN})"
)

ERROR_PATTERN = re.compile(
    rf"^\s*(xc|a|b)\s*=\s*({FLOAT_PATTERN})\s+"
    rf"({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s*$"
)


@dataclass(frozen=True)
class OptimizationRun:
    """One successful multi-start fit."""

    run_number: int
    initial_xc: float
    initial_a: float
    initial_b: float
    optimized_xc: float
    optimized_a: float
    optimized_b: float
    score: float
    result_line: str


def export_entropy_files(
    p_values: Sequence[float],
    dimensions: Sequence[int],
    entropy_means: np.ndarray,
    entropy_sems: np.ndarray,
    output_dir: Path,
) -> Tuple[Dict[int, Path], Dict[int, int]]:
    """Write valid autoScale rows and return paths and row counts by L."""

    generated_files: Dict[int, Path] = {}
    valid_counts: Dict[int, int] = {}
    skipped_total = 0

    for i, raw_L in enumerate(dimensions):
        L = int(raw_L)
        path = output_dir / f"entropy_L{L}.dat"
        rows: List[str] = []

        for j, raw_p in enumerate(p_values):
            p = float(raw_p)
            value = float(entropy_means[i, j])
            sem = float(entropy_sems[i, j])

            if not math.isfinite(value) or not math.isfinite(sem) or sem <= 0.0:
                skipped_total += 1
                print(
                    f"  Skipping p={p:.12g}, L={L}: invalid or zero SEM "
                    f"(mean={value:.12g}, SEM={sem:.12g})"
                )
                continue

            # Scientific notation prevents small positive SEMs from being
            # rounded to 0.00000000 in the output file.
            rows.append(f"{p:.16e} {value:.16e} {sem:.16e}\n")

        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("# <x-Values> <y-Values> <dy-Values (SEM)>\n")
            stream.writelines(rows)

        generated_files[L] = path
        valid_counts[L] = len(rows)
        print(f"  L={L}: wrote {len(rows)} valid points to {path.name}")

    print(f"Omitted {skipped_total} point(s) with unusable SEM values.")
    return generated_files, valid_counts


def create_master_file(
    generated_files: Mapping[int, Path],
    valid_counts: Mapping[int, int],
    output_dir: Path,
) -> Tuple[Path, List[int]]:
    """Create inputFiles.dat using only sufficiently large, usable datasets."""

    master_file = output_dir / "inputFiles.dat"
    included_sizes: List[int] = []

    with master_file.open("w", encoding="utf-8", newline="\n") as stream:
        for L in sorted(generated_files):
            if L < MIN_L_THRESHOLD:
                print(f"Excluding L={L}: below MIN_L_THRESHOLD={MIN_L_THRESHOLD}")
                continue
            if valid_counts[L] < MIN_VALID_POINTS_PER_SIZE:
                print(
                    f"Excluding L={L}: only {valid_counts[L]} valid points "
                    f"(minimum {MIN_VALID_POINTS_PER_SIZE})"
                )
                continue

            stream.write(f"{generated_files[L].name}\t{L}\n")
            included_sizes.append(L)

    if len(included_sizes) < MIN_INCLUDED_SYSTEM_SIZES:
        raise RuntimeError(
            f"Only {len(included_sizes)} usable system sizes remain: {included_sizes}. "
            f"autoScale should be run with at least {MIN_INCLUDED_SYSTEM_SIZES}."
        )

    print(f"Included system sizes: {included_sizes}")
    return master_file, included_sizes


def run_command(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a subprocess and retain both streams for useful diagnostics."""

    return subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def autoscale_command(
    output_name: str,
    xc: float,
    a: float,
    b: float,
    *extra_options: str,
) -> List[str]:
    """Build an autoScale command with the current Python interpreter."""

    return [
        sys.executable,
        str(AUTOSCALE_SCRIPT),
        "-f",
        "inputFiles.dat",
        "-o",
        output_name,
        "-xc",
        str(float(xc)),
        "-a",
        str(float(a)),
        "-b",
        str(float(b)),
        "-xr",
        str(MIN_SCALED_X),
        str(MAX_SCALED_X),
        *extra_options,
    ]


def parse_best_result(result_file: Path) -> Tuple[float, float, float, float, str]:
    """Return xc, a, b, S, and the full line for the lowest finite S."""

    best: Tuple[float, float, float, float, str] | None = None

    with result_file.open("r", encoding="utf-8") as stream:
        for line in stream:
            match = SCORE_PATTERN.search(line)
            if not match:
                continue

            xc, a, b, score = (float(value) for value in match.groups())
            if not all(math.isfinite(value) for value in (xc, a, b, score)):
                continue
            if best is None or score < best[3]:
                best = (xc, a, b, score, line.strip())

    if best is None:
        raise RuntimeError(
            f"Optimization completed, but no valid 'xc=... a=... b=... S=...' "
            f"line was found in {result_file}."
        )
    return best


def write_multistart_summary(
    output_dir: Path,
    successful_runs: Sequence[OptimizationRun],
    failed_runs: Sequence[Tuple[int, float, float, float, str]],
) -> None:
    """Save all successful fits, ordered from lowest to highest S."""

    ordered_runs = sorted(successful_runs, key=lambda run: run.score)

    csv_path = output_dir / "multistart_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "rank",
                "run_number",
                "initial_xc",
                "initial_a",
                "initial_b",
                "optimized_xc",
                "optimized_a",
                "optimized_b",
                "S",
            ]
        )
        for rank, run in enumerate(ordered_runs, start=1):
            writer.writerow(
                [
                    rank,
                    run.run_number,
                    f"{run.initial_xc:.16g}",
                    f"{run.initial_a:.16g}",
                    f"{run.initial_b:.16g}",
                    f"{run.optimized_xc:.16g}",
                    f"{run.optimized_a:.16g}",
                    f"{run.optimized_b:.16g}",
                    f"{run.score:.16g}",
                ]
            )

    # Keep the traditional output filename as a readable list of all final
    # autoScale states, ordered by S. It is regenerated on every execution.
    output_file = output_dir / "scaled_output.out"
    with output_file.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Successful multi-start fits, ordered by increasing S\n")
        for run in ordered_runs:
            stream.write(run.result_line + "\n")

    failure_path = output_dir / "multistart_failures.txt"
    if failed_runs:
        with failure_path.open("w", encoding="utf-8", newline="\n") as stream:
            for run_number, xc, a, b, diagnostic in failed_runs:
                stream.write(
                    f"run={run_number} initial_xc={xc:.16g} "
                    f"initial_a={a:.16g} initial_b={b:.16g}\n"
                )
                stream.write(diagnostic.rstrip() + "\n\n")
    else:
        failure_path.unlink(missing_ok=True)

    print(f"Saved ranked fits to {csv_path.name}")
    print(f"Saved autoScale result lines to {output_file.name}")
    if failed_runs:
        print(f"Saved failed-run diagnostics to {failure_path.name}")


def optimize_with_autoscale(output_dir: Path) -> Tuple[float, float, float, float]:
    """Run every initial guess and return the converged fit with lowest S.

    Each fit uses a separate output file. This avoids accidentally accepting a
    result left over by a previous run when one subprocess fails.
    """

    run_dir = output_dir / "multistart_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    for old_run_file in run_dir.glob("run_*.out"):
        old_run_file.unlink()

    initial_guesses = list(
        itertools.product(XC_GUESSES, A_GUESSES, B_GUESSES)
    )
    total = len(initial_guesses)
    successful_runs: List[OptimizationRun] = []
    failed_runs: List[Tuple[int, float, float, float, str]] = []

    print(
        f"Testing {total} initial guesses over scaled x in "
        f"[{MIN_SCALED_X}, {MAX_SCALED_X}]..."
    )

    for run_number, (raw_xc, raw_a, raw_b) in enumerate(
        initial_guesses, start=1
    ):
        xc = float(raw_xc)
        a = float(raw_a)
        b = float(raw_b)
        run_file = run_dir / f"run_{run_number:04d}.out"
        relative_output = Path(run_dir.name) / run_file.name

        # The final output line always contains S. Omitting -showS prevents
        # hundreds of intermediate simplex iterations from filling stdout.
        command = autoscale_command(str(relative_output), xc, a, b)
        result = run_command(command, output_dir)

        if result.returncode != 0:
            diagnostic = (
                result.stderr or result.stdout or "No diagnostic output"
            ).strip()
            failed_runs.append((run_number, xc, a, b, diagnostic))
        else:
            try:
                opt_xc, opt_a, opt_b, score, line = parse_best_result(run_file)
                successful_runs.append(
                    OptimizationRun(
                        run_number=run_number,
                        initial_xc=xc,
                        initial_a=a,
                        initial_b=b,
                        optimized_xc=opt_xc,
                        optimized_a=opt_a,
                        optimized_b=opt_b,
                        score=score,
                        result_line=line,
                    )
                )
            except (OSError, RuntimeError, ValueError) as error:
                failed_runs.append((run_number, xc, a, b, str(error)))

        if run_number == 1 or run_number % 10 == 0 or run_number == total:
            print(
                f"  Completed {run_number}/{total}: "
                f"{len(successful_runs)} successful, "
                f"{len(failed_runs)} failed"
            )

    if not successful_runs:
        first_failure = failed_runs[0][4] if failed_runs else "No diagnostic output"
        raise RuntimeError(
            "Every autoScale optimization failed. The first failure was:\n"
            + first_failure
        )

    write_multistart_summary(output_dir, successful_runs, failed_runs)

    best = min(successful_runs, key=lambda run: run.score)
    print("Lowest quality score found:")
    print(f"  Initial guess: xc={best.initial_xc:.10g}, "
          f"a={best.initial_a:.10g}, b={best.initial_b:.10g}")
    print(f"  Optimized fit: {best.result_line}")
    print(f"  S = {best.score:.12g}")
    return (
        best.optimized_xc,
        best.optimized_a,
        best.optimized_b,
        best.score,
    )


def calculate_parameter_errors(
    output_dir: Path, xc: float, a: float, b: float
) -> Dict[str, Tuple[float, float]]:
    """Run the S+1 analysis and return (minus_error, plus_error) by parameter."""

    error_file = output_dir / "error.out"
    error_file.unlink(missing_ok=True)

    command = autoscale_command(error_file.name, xc, a, b, "-getError")
    result = run_command(command, output_dir)

    if result.returncode != 0:
        raise RuntimeError(
            "autoScale -getError failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    if not error_file.exists():
        raise RuntimeError(f"autoScale did not create {error_file}")

    errors: Dict[str, Tuple[float, float]] = {}
    fitted_values: Dict[str, float] = {}

    with error_file.open("r", encoding="utf-8") as stream:
        for line in stream:
            match = ERROR_PATTERN.match(line)
            if not match:
                continue

            name = match.group(1)
            fitted_values[name] = float(match.group(2))
            errors[name] = (abs(float(match.group(3))), abs(float(match.group(4))))

    missing = {"xc", "a", "b"} - errors.keys()
    if missing:
        raise RuntimeError(
            f"Could not parse S+1 errors for {sorted(missing)} from {error_file}."
        )

    print("S+1 parameter intervals:")
    for name in ("xc", "a", "b"):
        minus, plus = errors[name]
        print(f"  {name} = {fitted_values[name]:.8g} -{minus:.6g} +{plus:.6g}")

    return errors


# =============================================================================
# 4. PLOTTING
# =============================================================================

def scaling_plot(
    p_values: np.ndarray,
    dimensions: np.ndarray,
    entropy_means: np.ndarray,
    entropy_sems: np.ndarray,
    included_sizes: Sequence[int],
    xc: float,
    a: float,
    b: float,
    errors: Mapping[str, Tuple[float, float]],
) -> None:
    """Plot the same entropy observable that was fitted by autoScale."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )

    figure, axis = plt.subplots(figsize=(7.0, 5.5))
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=min(included_sizes), vmax=max(included_sizes))

    size_to_index = {int(L): i for i, L in enumerate(dimensions)}
    for L in included_sizes:
        i = size_to_index[int(L)]
        scaled_x = (p_values - xc) * (L ** a)
        scaled_y = entropy_means[i, :] * (L ** b)
        in_fit_window = (
            np.isfinite(scaled_x)
            & np.isfinite(scaled_y)
            & np.isfinite(entropy_sems[i, :])
            & (entropy_sems[i, :] > 0.0)
            # & (scaled_x >= MIN_SCALED_X)
            # & (scaled_x <= MAX_SCALED_X)
        )
        axis.plot(
            scaled_x[in_fit_window],
            scaled_y[in_fit_window],
            marker=".",
            color=cmap(norm(L)),
            label=f"L={L}",
        )

    axis.set_xlabel(rf"$(p - {xc:.6g})L^{{{a:.4f}}}$")
    axis.set_ylabel(rf"$\mathcal{{S}}L^{{{b:.4f}}}$")

    xc_minus, xc_plus = errors["xc"]
    a_minus, a_plus = errors["a"]
    b_minus, b_plus = errors["b"]
    text = (
        rf"$x_c={xc:.5f}_{{-{xc_minus:.5f}}}^{{+{xc_plus:.5f}}}$" + "\n"
        rf"$a={a:.5f}_{{-{a_minus:.5f}}}^{{+{a_plus:.5f}}}$" + "\n"
        rf"$b={b:.5f}_{{-{b_minus:.5f}}}^{{+{b_plus:.5f}}}$"
    )
    axis.text(
        0.03,
        0.05,
        text,
        transform=axis.transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.9,
        },
        fontsize=12,
    )

    scalar_map = cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array([])
    colorbar = figure.colorbar(scalar_map, ax=axis)
    colorbar.set_label("System size $L$", fontsize=14)
    colorbar.ax.tick_params(labelsize=12)

    # axis.set_xlim(MIN_SCALED_X, MAX_SCALED_X)
    axis.grid(True, alpha=0.3, linestyle="--")
    figure.tight_layout()
    plt.show()


# =============================================================================
# 5. MAIN PROGRAM
# =============================================================================

def main() -> None:
    print(f"Python interpreter: {sys.executable}")
    print(f"Loading data from: {DATA_FILE}")

    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Data file does not exist: {DATA_FILE}")
    if not AUTOSCALE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"autoScale script does not exist: {AUTOSCALE_SCRIPT}\n"
            "Update AUTOSCALE_SCRIPT so it points to autoScale_corrected.py."
        )
    if MIN_SCALED_X >= MAX_SCALED_X:
        raise ValueError("MIN_SCALED_X must be smaller than MAX_SCALED_X")
    for name, guesses in (
        ("XC_GUESSES", XC_GUESSES),
        ("A_GUESSES", A_GUESSES),
        ("B_GUESSES", B_GUESSES),
    ):
        guesses = np.asarray(guesses, dtype=float)
        if guesses.size == 0 or not np.all(np.isfinite(guesses)):
            raise ValueError(f"{name} must contain at least one finite value")

    with DATA_FILE.open("rb") as stream:
        data = pickle.load(stream)

    p_values = np.asarray(data["p_values"], dtype=float)
    dimensions = np.asarray(data["L_values"], dtype=int)

    if p_values.ndim != 1 or dimensions.ndim != 1:
        raise ValueError("p_values and L_values must be one-dimensional")
    if len(np.unique(p_values)) != len(p_values):
        raise ValueError("p_values contains duplicates")
    if len(np.unique(dimensions)) != len(dimensions):
        raise ValueError("L_values contains duplicates")

    shape = (len(dimensions), len(p_values))
    percolation_means = np.empty(shape, dtype=float)
    percolation_sems = np.empty(shape, dtype=float)
    entropy_means = np.empty(shape, dtype=float)
    entropy_sems = np.empty(shape, dtype=float)
    dv_entropy_means = np.empty(shape, dtype=float)
    dv_entropy_sems = np.empty(shape, dtype=float)

    print("Calculating means and standard errors...")
    for i, L in enumerate(dimensions):
        for j, p in enumerate(p_values):
            percolation_means[i, j], percolation_sems[i, j] = percolation(
                p, int(L), data
            )
            entropy_means[i, j], entropy_sems[i, j] = entropy_from_topology(
                p, int(L), data, R_VALUE
            )
            dv_entropy_means[i, j], dv_entropy_sems[i, j] = entropy_DV(
                p, int(L), data
            )

    # These arrays are intentionally calculated for possible later analysis.
    # The current autoScale fit and plot both use entropy_means/entropy_sems.
    _ = (percolation_means, percolation_sems, dv_entropy_means, dv_entropy_sems)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Exporting entropy data for autoScale...")
    generated_files, valid_counts = export_entropy_files(
        p_values,
        dimensions,
        entropy_means,
        entropy_sems,
        OUTPUT_DIR,
    )
    _, included_sizes = create_master_file(
        generated_files, valid_counts, OUTPUT_DIR
    )

    print("Running multi-start autoScale optimization...")
    best_xc, best_a, best_b, best_S = optimize_with_autoscale(OUTPUT_DIR)
    print(
        f"Best parameters: xc={best_xc:.10g}, a={best_a:.10g}, "
        f"b={best_b:.10g}, S={best_S:.10g}"
    )

    print("Calculating S+1 parameter errors...")
    errors = calculate_parameter_errors(
        OUTPUT_DIR, best_xc, best_a, best_b
    )

    print("Generating entropy-collapse plot...")
    scaling_plot(
        p_values,
        dimensions,
        entropy_means,
        entropy_sems,
        included_sizes,
        best_xc,
        best_a,
        best_b,
        errors,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise