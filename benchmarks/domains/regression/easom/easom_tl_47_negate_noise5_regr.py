"""TL regression benchmark for Easom function with negated output and noise.

Target task: Standard negated Easom function  
Source task: Negated Easom function with noise (std=0.05)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.parameters.base import DiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.random import temporary_seed
from benchmarks.definition import (
    TransferLearningRegressionBenchmark,
    TransferLearningRegressionBenchmarkSettings,
)
from benchmarks.domains.regression.base import run_tl_regression_benchmark
from benchmarks.domains.transfer_learning.easom.easom_tl_47_negate_noise5 import easom


def load_data() -> pd.DataFrame:
    """Load data for Easom benchmark - extracted from original convergence benchmark.
    
    Creates synthetic data using the same parameters as the original:
    • 47 points per dimension
    • Bounds: [-10, 10] for both dimensions
    • Target function: negated Easom
    • Source function: negated Easom with noise (std=0.05)
    
    Returns:
        DataFrame containing the Easom benchmark data
    """
    points_per_dim = 47
    negate = True
    bounds: np.ndarray = np.array([[-10] * 2, [10] * 2])
    grid_locations = {
        f"x{d}": np.linspace(lower, upper, points_per_dim)
        for d, (lower, upper) in enumerate(bounds.T)
    }

    functions = {
        "Target_Function": lambda x: easom(x, negate=negate),
        "Source_Function": lambda x: easom(x, noise_std=0.05, negate=negate),
    }

    # Create meshgrid for all combinations
    meshgrid = np.meshgrid(*[points for points in grid_locations.values()])
    
    # Use same random seed as original benchmark for reproducibility
    with temporary_seed(1337):
        lookups = []
        for function_name, function in functions.items():
            lookup = pd.DataFrame(
                {f"x{d}": grid_d.ravel() for d, grid_d in enumerate(meshgrid)}
            )
            lookup["Target"] = lookup.apply(
                function, axis=1
            )  # Apply function to get Target values
            lookup["Function"] = function_name
            lookups.append(lookup)
        
    return pd.concat(lookups, ignore_index=True)


def make_searchspace(
    data: pd.DataFrame,
    use_task_parameter: bool,
) -> SearchSpace:
    """Create the search space for Easom benchmark - extracted from original.
    
    Args:
        data: The Easom dataset
        use_task_parameter: Whether to include task parameter (True for TL, False for vanilla)
        
    Returns:
        SearchSpace for the benchmark
    """
    # Extract unique values for each dimension from the data
    grid_locations = {
        "x0": sorted(data["x0"].unique()),
        "x1": sorted(data["x1"].unique()),
    }
    
    params: list[DiscreteParameter] = [
        NumericalDiscreteParameter(name=name, values=values)
        for name, values in grid_locations.items()
    ]
    
    if use_task_parameter:
        task_param = TaskParameter(
            name="Function",
            values=["Target_Function", "Source_Function"],
            active_values=["Target_Function"],
        )
        params.append(task_param)
    
    return SearchSpace.from_product(parameters=params)


def make_objective() -> SingleTargetObjective:
    """Create the objective for Easom benchmark - extracted from original.
    
    Returns:
        SingleTargetObjective for Target maximization (negated Easom)
    """
    # Original uses negated Easom, so mode="MAX" (maximizing the negated function)
    return SingleTargetObjective(
        target=NumericalTarget(name="Target", mode="MAX")
    )


def easom_tl_47_negate_noise5_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses different versions of the Easom function as tasks:
    • Target task: Standard negated Easom function
    • Source task: Negated Easom function with added noise (std=0.05)

    Uses the same configuration as the original convergence benchmark:
    • 47 points per dimension 
    • Bounds: [-10, 10] for both x0 and x1
    • Discrete grid search space
    • Negated Easom function (maximization objective)

    Key characteristics:
    • Parameters:
      - x0: Numerical discrete (47 points from -10 to 10)
      - x1: Numerical discrete (47 points from -10 to 10)
      - Function: Task parameter (Target_Function, Source_Function)
    • Target: Easom function output (continuous)
    • Objective: Maximization (negated Easom)
    • Compares TL models vs vanilla GP on predictive performance

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame with benchmark results
    """
    return run_tl_regression_benchmark(
        settings=settings,
        data_loader=load_data,
        searchspace_factory=make_searchspace,
        objective_factory=make_objective,
    )


# Define the benchmark settings
benchmark_config = TransferLearningRegressionBenchmarkSettings(
    n_mc_iterations=30,
    max_n_train_points=10,
    source_fractions=(0.01, 0.05, 0.1),
    noise_std=0.0,  # Noise is already built into source function
)

# Create the benchmark
easom_tl_47_negate_noise5_regr_benchmark = TransferLearningRegressionBenchmark(
    function=easom_tl_47_negate_noise5_regr, 
    settings=benchmark_config
)