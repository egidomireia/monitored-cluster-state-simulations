# autoScale.py -- a program for automatic finite-size scaling analyses
# Copyright (C) 2007-2009 Oliver Melchert
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
"""Automatic finite-size scaling analysis.

Python 3 adaptation of Oliver Melchert's autoScale.py (2007-2009).
The program measures the quality of a data collapse using the method of
Houdayer and Hartmann, Phys. Rev. B 70, 014418 (2004), and minimizes that
quality with a Nelder-Mead simplex.

This corrected version keeps the original command-line interface while fixing
the Python 3 class initialization, root bracketing, state restoration, input
validation, and several non-terminating-loop edge cases.  Its ``-getError``
option performs a profile-S analysis: when one scaling parameter is fixed at a
trial value, all other fitted parameters are re-optimized before S is tested
against S_min + 1.

Original program copyright (C) 2007-2009 Oliver Melchert.
Licensed under the GNU General Public License, version 2 or later.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, TextIO, Tuple


VERSION = "1.2-python3-profile"


@dataclass
class myValue:
    """One unscaled or scaled data point."""

    L: float
    x: float
    y: float
    dy: float

    def __repr__(self) -> str:
        return "%lf %lf %lf %lf" % (self.L, self.x, self.y, self.dy)


class myRawData:
    """Storage and input handling for all finite-size data sets."""

    def __init__(self) -> None:
        self.dataSet: Dict[float, List[myValue]] = {}
        self.nSets = 0

    def fetchData(self, file_list_name: str) -> None:
        """Read the master file and all data files referenced by it.

        Each non-comment line in the master file must contain

            data_file_path  L

        Each non-comment line in a data file must contain

            x  y  dy

        Relative data-file paths are resolved relative to the master file.
        Uncertainties must be finite and strictly positive.
        """

        master_path = os.path.abspath(file_list_name)
        master_dir = os.path.dirname(master_path)

        with open(master_path, "r", encoding="utf-8") as file_list:
            for line_number, line in enumerate(file_list, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                fields = stripped.split()
                if len(fields) < 2:
                    raise ValueError(
                        f"{master_path}:{line_number}: expected '<data file> <L>'"
                    )

                data_path = fields[0]
                if not os.path.isabs(data_path):
                    data_path = os.path.join(master_dir, data_path)
                data_path = os.path.abspath(data_path)

                try:
                    L = float(fields[1])
                except ValueError as exc:
                    raise ValueError(
                        f"{master_path}:{line_number}: invalid system size "
                        f"{fields[1]!r}"
                    ) from exc

                if not math.isfinite(L) or L <= 0.0:
                    raise ValueError(
                        f"{master_path}:{line_number}: L must be finite and positive"
                    )
                if L in self.dataSet:
                    raise ValueError(
                        f"{master_path}:{line_number}: duplicate data set for L={L}"
                    )
                if not os.path.exists(data_path):
                    raise FileNotFoundError(
                        f"Data file listed at {master_path}:{line_number} "
                        f"does not exist: {data_path}"
                    )

                values: List[myValue] = []
                with open(data_path, "r", encoding="utf-8") as data_file:
                    for data_line_number, data_line in enumerate(data_file, start=1):
                        data_stripped = data_line.strip()
                        if not data_stripped or data_stripped.startswith("#"):
                            continue

                        data_fields = data_stripped.split()
                        if len(data_fields) < 3:
                            raise ValueError(
                                f"{data_path}:{data_line_number}: expected '<x> <y> <dy>'"
                            )

                        try:
                            x, y, dy = map(float, data_fields[:3])
                        except ValueError as exc:
                            raise ValueError(
                                f"{data_path}:{data_line_number}: x, y and dy must be "
                                "numbers"
                            ) from exc

                        if not all(math.isfinite(v) for v in (x, y, dy)):
                            raise ValueError(
                                f"{data_path}:{data_line_number}: x, y and dy must be "
                                "finite"
                            )
                        if dy <= 0.0:
                            raise ValueError(
                                f"{data_path}:{data_line_number}: dy must be strictly "
                                "positive. Supply the standard error of the mean, not "
                                "a zero uncertainty."
                            )

                        values.append(myValue(L, x, y, dy))

                if len(values) < 2:
                    raise ValueError(
                        f"{data_path}: each data set needs at least two data points"
                    )

                self.dataSet[L] = values

        self.nSets = len(self.dataSet)
        if self.nSets < 2:
            raise ValueError(
                "At least two system sizes are required for interpolation; "
                "three or more are strongly recommended."
            )

    def listDataSets(self) -> None:
        """Print all raw data sets."""

        for L, data in self.dataSet.items():
            for value in data:
                print(L, value.x, value.y, value.dy)

    def listDataSetsScaled(self, scale_assumption: "myScaleAssumption") -> None:
        """Print all data sets after applying a scaling assumption."""

        for data in self.dataSet.values():
            for value in data:
                print(scale_assumption.scale(value))


class myScaleAssumption:
    """The finite-size scaling parameters and transformation."""

    def __init__(self) -> None:
        self.xc = 0.0
        self.xco = 1
        self.a = 0.0
        self.ao = 1
        self.b = 0.0
        self.bo = 1
        self.scaledXMin = -99999.0
        self.scaledXMax = 99999.0

    def setScalePar(self, name: str, value: float, opt_flag: int) -> None:
        """Set a scaling parameter and whether it is optimized."""

        if name == "xc":
            self.xc, self.xco = value, opt_flag
        elif name == "a":
            self.a, self.ao = value, opt_flag
        elif name == "b":
            self.b, self.bo = value, opt_flag
        else:
            raise ValueError(f"Scaling parameter {name!r} does not exist")

    def updateByHand(self, xc: float, a: float, b: float) -> None:
        self.xc = float(xc)
        self.a = float(a)
        self.b = float(b)

    def updateFromList(self, parameters: Sequence[float]) -> None:
        if len(parameters) != 3:
            raise ValueError("Scaling parameter list must be [xc, a, b]")
        self.updateByHand(parameters[0], parameters[1], parameters[2])

    def scaleParNames(self) -> List[str]:
        return ["xc", "a", "b"]

    def scaleParList(self) -> Tuple[List[float], List[int]]:
        return [self.xc, self.a, self.b], [self.xco, self.ao, self.bo]

    def scale(self, value: myValue) -> myValue:
        """Apply x -> (x-xc)L^a and y,dy -> y,dy times L^b."""

        scale_x = math.pow(value.L, self.a)
        scale_y = math.pow(value.L, self.b)
        return myValue(
            value.L,
            (value.x - self.xc) * scale_x,
            value.y * scale_y,
            value.dy * scale_y,
        )

    def listScalePar(self, fileStream: TextIO = sys.stdout) -> None:
        fileStream.write("xc=%f a=%f b=%f\n" % (self.xc, self.a, self.b))


class myFunc(myRawData, myScaleAssumption):
    """Raw data plus the scaling model and quality function."""

    def __init__(self) -> None:
        myRawData.__init__(self)
        myScaleAssumption.__init__(self)

    def scaleData(self, scale_parameters: Sequence[float]) -> float:
        """Return the mean normalized squared residual S."""

        self.updateFromList(scale_parameters)
        quality_terms: List[float] = []

        for data in self.dataSet.values():
            for value in data:
                scaled_value = self.scale(value)
                x, y, dy = scaled_value.x, scaled_value.y, scaled_value.dy

                if not all(math.isfinite(v) for v in (x, y, dy)):
                    return math.inf
                if dy <= 0.0:
                    raise ValueError("Scaled uncertainty dy must be strictly positive")

                if self.scaledXMin <= x <= self.scaledXMax:
                    subset = self.selectSubset(value)
                    if subset:
                        Y, dY2 = self.llsFit(x, subset)
                        denominator = dy * dy + dY2
                        if denominator <= 0.0 or not math.isfinite(denominator):
                            raise ValueError(
                                "Invalid uncertainty in quality calculation; check dy"
                            )
                        chi2 = (y - Y) ** 2 / denominator
                        if math.isfinite(chi2):
                            quality_terms.append(float(chi2))

        if not quality_terms:
            return 99999.99
        return sum(quality_terms) / float(len(quality_terms))

    def selectSubset(self, pivot_value: myValue) -> List[myValue]:
        """Select bracketing points from every non-pivot system size."""

        scaled_pivot = self.scale(pivot_value)
        subset: List[myValue] = []

        for L, values in self.dataSet.items():
            if L == pivot_value.L:
                continue

            lower: Optional[myValue] = None
            upper: Optional[myValue] = None

            for value in values:
                scaled_value = self.scale(value)
                if scaled_value.x <= scaled_pivot.x:
                    if lower is None or scaled_value.x >= lower.x:
                        lower = scaled_value
                else:
                    if upper is None or scaled_value.x <= upper.x:
                        upper = scaled_value

            if lower is not None and upper is not None:
                subset.extend((lower, upper))

        return subset

    def llsFit(self, pivot_x: float, subset: Sequence[myValue]) -> Tuple[float, float]:
        """Weighted straight-line fit, evaluated at pivot_x."""

        K = Kx = Ky = Kxx = Kxy = 0.0
        for value in subset:
            if value.dy <= 0.0:
                raise ValueError("All uncertainties must be strictly positive")
            weight = 1.0 / (value.dy * value.dy)
            K += weight
            Kx += value.x * weight
            Ky += value.y * weight
            Kxx += value.x * value.x * weight
            Kxy += value.x * value.y * weight

        determinant = K * Kxx - Kx * Kx
        tolerance = 1e-14 * max(1.0, abs(K * Kxx), abs(Kx * Kx))
        if abs(determinant) <= tolerance:
            raise ValueError(
                "Singular interpolation fit. Check for repeated x values and ensure "
                "that several system sizes overlap in the selected scaled x range."
            )

        A = (Ky * Kxx - Kx * Kxy) / determinant
        B = (K * Kxy - Kx * Ky) / determinant
        Y = A + B * pivot_x
        dY2 = (
            Kxx - 2.0 * pivot_x * Kx + pivot_x * pivot_x * K
        ) / determinant

        if dY2 < 0.0 and abs(dY2) <= 1e-12:
            dY2 = 0.0
        if not math.isfinite(Y) or not math.isfinite(dY2) or dY2 < 0.0:
            raise ValueError("Invalid result from weighted linear interpolation")
        return Y, dY2

    def listState(
        self, scale_parameters: Sequence[float], fileStream: TextIO = sys.stdout
    ) -> None:
        quality = self.scaleData(scale_parameters)
        fileStream.write(
            "dx = [%lf:%lf]  xc = %f  a = %f  b = %f  S = %f\n"
            % (
                self.scaledXMin,
                self.scaledXMax,
                self.xc,
                self.a,
                self.b,
                quality,
            )
        )


def amotry(
    function: Callable[[Sequence[float]], float],
    simplex: List[List[float]],
    values: List[float],
    coordinate_sums: List[float],
    dimensions: int,
    highest: int,
    factor: float,
    opt_flags: Sequence[int],
) -> float:
    """Try a reflected, expanded, or contracted simplex point."""

    factor1 = (1.0 - factor) / float(dimensions)
    factor2 = factor1 - factor
    trial = [0.0] * dimensions

    for j in range(dimensions):
        if opt_flags[j] != 0:
            trial[j] = coordinate_sums[j] * factor1 - simplex[highest][j] * factor2
        else:
            trial[j] = simplex[highest][j]

    trial_value = function(trial)
    if trial_value < values[highest]:
        values[highest] = trial_value
        for j in range(dimensions):
            coordinate_sums[j] += trial[j] - simplex[highest][j]
            simplex[highest][j] = trial[j]

    return trial_value


def amoeba(
    function: Callable[[Sequence[float]], float],
    simplex: List[List[float]],
    values: List[float],
    opt_flags: Sequence[int],
    tolerance: float,
    report_fitness: int,
    max_iterations: int = 500,
) -> Tuple[List[float], float, int]:
    """Minimize function with the original Nelder-Mead simplex scheme."""

    tiny = 1.0e-10
    dimensions = len(values) - 1
    iterations = 0
    coordinate_sums = [
        sum(simplex[row][column] for row in range(dimensions + 1))
        for column in range(dimensions)
    ]

    while True:
        lowest = min(range(dimensions + 1), key=lambda i: values[i])
        ordered_worst = sorted(
            range(dimensions + 1), key=lambda i: values[i], reverse=True
        )
        highest, next_highest = ordered_worst[:2]

        relative_tolerance = (
            2.0
            * abs(values[highest] - values[lowest])
            / (abs(values[highest]) + abs(values[lowest]) + tiny)
        )

        if report_fitness:
            print(iterations, values[lowest], relative_tolerance)

        if relative_tolerance < tolerance:
            values[0], values[lowest] = values[lowest], values[0]
            simplex[0], simplex[lowest] = simplex[lowest], simplex[0]
            return list(simplex[0]), values[0], iterations

        if iterations >= max_iterations:
            raise RuntimeError(
                f"Maximum of {max_iterations} simplex iterations exceeded"
            )

        iterations += 2
        trial_value = amotry(
            function,
            simplex,
            values,
            coordinate_sums,
            dimensions,
            highest,
            -1.0,
            opt_flags,
        )

        if trial_value <= values[lowest]:
            amotry(
                function,
                simplex,
                values,
                coordinate_sums,
                dimensions,
                highest,
                2.0,
                opt_flags,
            )
        elif trial_value >= values[next_highest]:
            saved_highest = values[highest]
            trial_value = amotry(
                function,
                simplex,
                values,
                coordinate_sums,
                dimensions,
                highest,
                0.5,
                opt_flags,
            )

            if trial_value >= saved_highest:
                for i in range(dimensions + 1):
                    if i == lowest:
                        continue
                    point = [0.0] * dimensions
                    for j in range(dimensions):
                        if opt_flags[j] != 0:
                            simplex[i][j] = 0.5 * (
                                simplex[i][j] + simplex[lowest][j]
                            )
                        point[j] = simplex[i][j]
                    values[i] = function(point)

                iterations += 1
                coordinate_sums = [
                    sum(simplex[row][column] for row in range(dimensions + 1))
                    for column in range(dimensions)
                ]
        else:
            iterations -= 1


def iniSimplex(
    function: Callable[[Sequence[float]], float],
    trial_point: Sequence[float],
    delta: float,
    opt_flags: Sequence[int],
) -> Tuple[List[List[float]], List[float]]:
    """Construct the initial simplex.

    A zero initial parameter receives a small additive displacement, because a
    purely multiplicative displacement would otherwise leave it unchanged.
    """

    dimensions = len(trial_point)
    simplex = [list(map(float, trial_point)) for _ in range(dimensions + 1)]

    for i in range(1, dimensions + 1):
        j = i - 1
        if opt_flags[j] != 0:
            base = float(trial_point[j])
            step = abs(base) * delta if base != 0.0 else delta
            simplex[i][j] = base + step

    values = [function(point) for point in simplex]
    return simplex, values


def rootBisection(
    function: Callable[[float], float],
    x_min: float,
    x_max: float,
    epsilon: float = 1e-5,
    max_iterations: int = 200,
) -> float:
    """Find a root inside an interval whose endpoints bracket it."""

    if epsilon <= 0.0:
        raise ValueError("Bisection epsilon must be positive")
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    f_min = function(x_min)
    f_max = function(x_max)
    if not math.isfinite(f_min) or not math.isfinite(f_max):
        raise ValueError("Non-finite function value at a bisection boundary")
    if f_min == 0.0:
        return x_min
    if f_max == 0.0:
        return x_max
    if f_min * f_max > 0.0:
        raise ValueError(
            "Root is not bracketed: "
            f"f({x_min})={f_min}, f({x_max})={f_max}"
        )

    for _ in range(max_iterations):
        midpoint = 0.5 * (x_min + x_max)
        f_midpoint = function(midpoint)
        if not math.isfinite(f_midpoint):
            raise ValueError("Non-finite function value during bisection")

        if f_midpoint == 0.0 or abs(x_max - x_min) <= 2.0 * epsilon:
            return midpoint

        if f_min * f_midpoint <= 0.0:
            x_max = midpoint
            f_max = f_midpoint
        else:
            x_min = midpoint
            f_min = f_midpoint

    raise RuntimeError(
        f"Bisection did not converge after {max_iterations} iterations"
    )


def getBrackets(
    function: Callable[[float], float],
    mid_value: float,
    relative_step: float = 0.01,
    growth: float = 1.6,
    max_iterations: int = 100,
) -> Tuple[List[float], List[float]]:
    """Find intervals bracketing the two S+1 roots around a minimum."""

    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    if growth <= 1.0:
        raise ValueError("growth must be greater than 1")

    f_mid = function(mid_value)
    if not math.isfinite(f_mid):
        raise ValueError(f"Non-finite S+1 function at optimum: {f_mid}")
    if f_mid >= 0.0:
        raise ValueError(
            "The supplied parameter is not inside the S+1 interval: "
            f"objective({mid_value})={f_mid}"
        )

    initial_step = max(abs(mid_value) * relative_step, 1e-6)

    def search(direction: float) -> List[float]:
        inner = mid_value
        f_inner = f_mid
        step = initial_step

        for _ in range(max_iterations):
            outer = mid_value + direction * step
            f_outer = function(outer)
            if not math.isfinite(f_outer):
                side = "left" if direction < 0.0 else "right"
                raise ValueError(
                    f"Non-finite S+1 function while bracketing the {side} "
                    f"root at {outer}"
                )

            if f_inner * f_outer <= 0.0:
                return [outer, inner] if direction < 0.0 else [inner, outer]

            inner = outer
            f_inner = f_outer
            step *= growth

        side = "left" if direction < 0.0 else "right"
        raise RuntimeError(
            f"Could not bracket the {side} S+1 root after "
            f"{max_iterations} attempts. Widen -xr, check dy, and verify that "
            "the optimum is well constrained by overlapping data sets."
        )

    return search(-1.0), search(1.0)


def minimizeNuisanceParameters(
    function_object: myFunc,
    fixed_parameter_id: int,
    fixed_value: float,
    starting_parameters: Sequence[float],
    opt_flags: Sequence[int],
    simplex_delta: float = 0.001,
    simplex_tolerance: float = 1.0e-6,
    max_iterations: int = 1000,
) -> Tuple[List[float], float]:
    """Minimize S over every fitted parameter except one fixed parameter.

    The existing Nelder-Mead implementation is applied only to the active
    nuisance coordinates.  This is important: passing a three-coordinate
    simplex with one zero optimization flag would leave a degenerate simplex
    and would not be a genuine two-dimensional minimization.
    """

    full_start = list(map(float, starting_parameters))
    if len(full_start) != len(opt_flags):
        raise ValueError("Parameter values and optimization flags do not match")
    if not 0 <= fixed_parameter_id < len(full_start):
        raise IndexError("Fixed parameter index is outside the parameter vector")

    full_start[fixed_parameter_id] = float(fixed_value)
    nuisance_ids = [
        parameter_id
        for parameter_id, optimize in enumerate(opt_flags)
        if optimize != 0 and parameter_id != fixed_parameter_id
    ]

    if not nuisance_ids:
        quality = function_object.scaleData(full_start)
        return full_start, quality

    nuisance_start = [full_start[parameter_id] for parameter_id in nuisance_ids]

    def nuisance_objective(nuisance_values: Sequence[float]) -> float:
        trial = full_start.copy()
        for parameter_id, value in zip(nuisance_ids, nuisance_values):
            trial[parameter_id] = float(value)
        return function_object.scaleData(trial)

    # Retain the starting point as a safe candidate.  Numerical convergence of
    # the nested simplex must never make the reported profile worse than the
    # valid starting nuisance values.
    starting_quality = nuisance_objective(nuisance_start)
    simplex, simplex_values = iniSimplex(
        nuisance_objective,
        nuisance_start,
        simplex_delta,
        [1] * len(nuisance_ids),
    )
    nuisance_best, optimized_quality, _ = amoeba(
        nuisance_objective,
        simplex,
        simplex_values,
        [1] * len(nuisance_ids),
        simplex_tolerance,
        0,
        max_iterations=max_iterations,
    )

    if starting_quality <= optimized_quality:
        nuisance_best = nuisance_start
        optimized_quality = starting_quality

    full_best = full_start.copy()
    for parameter_id, value in zip(nuisance_ids, nuisance_best):
        full_best[parameter_id] = float(value)

    # Re-evaluate the returned full vector so state and reported S agree.
    optimized_quality = function_object.scaleData(full_best)
    return full_best, optimized_quality


def errorAnalysis(
    function_object: myFunc, best_parameters: Sequence[float]
) -> Dict[int, List[float]]:
    """Perform profile-S+1 error analysis.

    For a trial value of one parameter, all remaining fitted parameters are
    re-optimized.  The returned entries are
    ``[best value, lower error, upper error]``.
    """

    best = list(map(float, best_parameters))
    _, opt_flags = function_object.scaleParList()
    function_object.updateFromList(best)
    best_quality = function_object.scaleData(best)

    if not math.isfinite(best_quality):
        raise ValueError(f"Best quality S is not finite: {best_quality}")

    print(f"# PERFORM PROFILE-S+1 ERROR ANALYSIS; S_min = {best_quality:.12g}")
    parameter_errors: Dict[int, List[float]] = {}
    parameter_names = function_object.scaleParNames()

    try:
        for parameter_id, optimize in enumerate(opt_flags):
            pivot = best[parameter_id]

            if optimize == 0:
                parameter_errors[parameter_id] = [pivot, 0.0, 0.0]
                continue

            # Cache exact trial values because root bracketing and bisection
            # repeatedly request the same endpoints.  Each cached entry also
            # stores its optimized nuisance parameters for warm starts.
            profile_cache: Dict[float, Tuple[List[float], float]] = {
                pivot: (best.copy(), best_quality)
            }

            def profile_quality(value: float) -> float:
                value = float(value)
                if value in profile_cache:
                    return profile_cache[value][1]

                nearest_value = min(
                    profile_cache,
                    key=lambda cached_value: abs(cached_value - value),
                )
                warm_start = profile_cache[nearest_value][0]
                profiled_parameters, quality = minimizeNuisanceParameters(
                    function_object,
                    fixed_parameter_id=parameter_id,
                    fixed_value=value,
                    starting_parameters=warm_start,
                    opt_flags=opt_flags,
                )
                if not math.isfinite(quality):
                    raise ValueError(
                        f"Non-finite profile S for "
                        f"{parameter_names[parameter_id]}={value}"
                    )
                profile_cache[value] = (profiled_parameters, quality)
                return quality

            def objective(value: float) -> float:
                return profile_quality(value) - (best_quality + 1.0)

            left_bracket, right_bracket = getBrackets(objective, pivot)
            left_root = rootBisection(
                objective, left_bracket[0], left_bracket[1], epsilon=1e-5
            )
            right_root = rootBisection(
                objective, right_bracket[0], right_bracket[1], epsilon=1e-5
            )

            parameter_errors[parameter_id] = [
                pivot,
                abs(pivot - left_root),
                abs(right_root - pivot),
            ]

            print(
                f"# {parameter_names[parameter_id]} profile used "
                f"{len(profile_cache)} nested optimizations"
            )
    finally:
        function_object.updateFromList(best)

    return parameter_errors


def usage(program_name: str) -> None:
    print(
        f"""
NAME
    {program_name} -- automated finite-size scaling analysis

SYNTAX
    python {program_name} -f inFile [-o outFile]
        [-xc value | -xc! value] [-a value | -a! value]
        [-b value | -b! value] [-xr min max] [-showS] [-getError]

OPTIONS
    -help              Show this help and exit.
    -version           Show the program version and exit.
    -f inFile          Master file containing '<data-file> <L>' lines.
    -o outFile         Append results to this file; default is stdout.
    -xc value          Initial critical point; optimize it.
    -xc! value         Critical point fixed during optimization.
    -a value           Initial exponent a; optimize it.
    -a! value          Exponent a fixed during optimization.
    -b value           Initial exponent b; optimize it.
    -b! value          Exponent b fixed during optimization.
    -xr min max        Interval on the scaled x axis used in the analysis.
    -showS             Print S during simplex minimization.
    -getError          Calculate profile-S+1 intervals.  At each trial value
                       of one parameter, re-optimize all other fitted
                       parameters.

SCALING ASSUMPTION
    x -> (x-xc)L^a
    y -> yL^b

EXAMPLE
    python {program_name} -f inputFiles.dat -xc 0.592541 \\
        -a 0.754524 -b 0.107421 -xr -0.15 0.15 -getError -o error.out
""".strip()
    )


def _require_values(
    arguments: Sequence[str], index: int, option: str, count: int
) -> None:
    if index + count >= len(arguments):
        plural = "s" if count != 1 else ""
        raise ValueError(f"Option {option} requires {count} value{plural}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv if argv is None else argv)
    program_name = os.path.basename(arguments[0]) if arguments else "autoScale.py"

    if len(arguments) == 1:
        usage(program_name)
        return 1
    if "-help" in arguments[1:]:
        usage(program_name)
        return 0
    if "-version" in arguments[1:]:
        print(f"{program_name}, version: {VERSION}")
        return 0

    output: TextIO = sys.stdout
    output_to_close: Optional[TextIO] = None
    file_list_name: Optional[str] = None
    delta = 0.001
    report_fitness = 0
    simplex_tolerance = 1e-6
    get_error = False
    function_object = myFunc()

    index = 1
    while index < len(arguments):
        option = arguments[index]

        if option == "-f":
            _require_values(arguments, index, option, 1)
            file_list_name = arguments[index + 1]
            index += 2
        elif option == "-o":
            _require_values(arguments, index, option, 1)
            output_name = arguments[index + 1]
            output_to_close = open(output_name, "a", encoding="utf-8")
            output = output_to_close
            index += 2
        elif option in ("-xc", "-a", "-b", "-xc!", "-a!", "-b!"):
            _require_values(arguments, index, option, 1)
            name = option[1:].replace("!", "")
            try:
                value = float(arguments[index + 1])
            except ValueError as exc:
                raise ValueError(
                    f"Option {option} requires a floating-point value"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"Option {option} requires a finite value")
            function_object.setScalePar(name, value, 0 if option.endswith("!") else 1)
            index += 2
        elif option == "-showS":
            report_fitness = 1
            index += 1
        elif option == "-xr":
            _require_values(arguments, index, option, 2)
            try:
                x_min = float(arguments[index + 1])
                x_max = float(arguments[index + 2])
            except ValueError as exc:
                raise ValueError("Option -xr requires two floating-point values") from exc
            if not math.isfinite(x_min) or not math.isfinite(x_max):
                raise ValueError("Option -xr boundaries must be finite")
            if x_min >= x_max:
                raise ValueError("Option -xr requires min < max")
            function_object.scaledXMin = x_min
            function_object.scaledXMax = x_max
            index += 3
        elif option == "-getError":
            get_error = True
            index += 1
        else:
            raise ValueError(f"Unknown option: {option}")

    if file_list_name is None:
        raise ValueError("No input file specified; use -f inputFiles.dat")
    if not os.path.exists(file_list_name):
        raise FileNotFoundError(f"Input file does not exist: {file_list_name}")

    try:
        starting_parameters, opt_flags = function_object.scaleParList()
        function_object.fetchData(file_list_name)

        simplex, simplex_values = iniSimplex(
            function_object.scaleData,
            starting_parameters,
            delta,
            opt_flags,
        )
        best_parameters, best_quality, iterations = amoeba(
            function_object.scaleData,
            simplex,
            simplex_values,
            opt_flags,
            simplex_tolerance,
            report_fitness,
        )
        del best_quality, iterations
        function_object.updateFromList(best_parameters)

        if get_error:
            parameter_errors = errorAnalysis(function_object, best_parameters)
            parameter_names = function_object.scaleParNames()
            output.write("# Profile-S+1 error analysis yields:\n")
            output.write("# Scaling analysis restricted to\n")
            output.write(
                "%4s = [%lf : %lf]\n"
                % (
                    "xr",
                    function_object.scaledXMin,
                    function_object.scaledXMax,
                )
            )
            output.write("# <scalePar>  <best>  <-Err>  <+Err>\n")
            for parameter_id in range(len(parameter_names)):
                values = parameter_errors[parameter_id]
                output.write(
                    "%4s = %lf %lf %lf\n"
                    % (
                        parameter_names[parameter_id],
                        values[0],
                        values[1],
                        values[2],
                    )
                )
        else:
            function_object.listState(best_parameters, output)
    finally:
        if output_to_close is not None:
            output_to_close.close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
