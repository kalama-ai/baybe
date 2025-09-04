"""TL regression benchmark for Hartmann function in 3D.

Target task: Standard Hartmann function (3D)
Source task: Hartmann function with noise (std=0.15)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import Hartmann

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


def load_data() -> pd.DataFrame:
    """Load data for Hartmann benchmark - extracted from original convergence benchmark.
    
    Creates synthetic data using the same parameters as the original:
    • 3D Hartmann function
    • 20 points per dimension
    • Bounds: [0, 1] for all dimensions
    • Target function: standard Hartmann
    • Source function: Hartmann with noise (std=0.15)
    
    Returns:
        DataFrame containing the Hartmann benchmark data
    """
    target_function = Hartmann(dim=3)
    source_function = Hartmann(dim=3, noise_std=0.15)
    
    points_per_dim = 20
    
    # Create grid locations for the parameters
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    grid_locations = {
        f"x{d}": np.linspace(lower, upper, points_per_dim)
        for d, (lower, upper) in enumerate(bounds.T)
    }
    
    # Create meshgrid for all combinations
    meshgrid = np.meshgrid(*[points for points in grid_locations.values()])
    
    # Create coordinate columns
    coord_columns = [f"x{d}" for d in range(3)]
    
    # Generate source data (with noise)
    source_data = pd.DataFrame(
        {f"x{d}": grid_d.ravel() for d, grid_d in enumerate(meshgrid)},
        columns=coord_columns,
    )
    
    # Convert coordinates to PyTorch tensor for function evaluation
    source_data_tensor = torch.tensor(source_data[coord_columns].values)
    
    # Use same random seed as original benchmark for reproducibility
    with temporary_seed(1337):
        source_target_values = source_function(source_data_tensor)
    
    source_data["Target"] = source_target_values.detach().numpy()
    source_data["Function"] = "Source_Function"
    
    # Generate target data (clean) - create all combinations and evaluate
    target_data = pd.DataFrame(
        {f"x{d}": grid_d.ravel() for d, grid_d in enumerate(meshgrid)},
        columns=coord_columns,
    )
    
    # Convert coordinates to PyTorch tensor for function evaluation
    target_data_tensor = torch.tensor(target_data[coord_columns].values)
    target_target_values = target_function(target_data_tensor)
    
    target_data["Target"] = target_target_values.detach().numpy()
    target_data["Function"] = "Target_Function"
    
    # Combine source and target data
    return pd.concat([source_data, target_data], ignore_index=True)


def make_searchspace(
    data: pd.DataFrame,
    use_task_parameter: bool,
) -> SearchSpace:
    """Create the search space for Hartmann benchmark - extracted from original.
    
    Args:
        data: The Hartmann dataset
        use_task_parameter: Whether to include task parameter (True for TL, False for vanilla)
        
    Returns:
        SearchSpace for the benchmark
    """
    # Extract unique values for each dimension from the data
    grid_locations = {
        f"x{d}": sorted(data[f"x{d}"].unique()) for d in range(3)
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
    """Create the objective for Hartmann benchmark - extracted from original.
    
    Returns:
        SingleTargetObjective for Target minimization (Hartmann function)
    """
    # Original uses Hartmann function minimization
    return SingleTargetObjective(
        target=NumericalTarget(name="Target", mode="MIN")
    )


def hartmann_tl_3_20_15_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses different versions of the 3D Hartmann function as tasks:
    • Target task: Standard 3D Hartmann function
    • Source task: 3D Hartmann function with added noise (std=0.15)

    Uses the same configuration as the original convergence benchmark:
    • 20 points per dimension 
    • Bounds: [0, 1] for all three dimensions (x0, x1, x2)
    • Discrete grid search space
    • Hartmann function (minimization objective)

    Key characteristics:
    • Parameters:
      - x0, x1, x2: Numerical discrete (20 points each from 0 to 1)
      - Function: Task parameter (Target_Function, Source_Function)
    • Target: Hartmann function output (continuous)
    • Objective: Minimization (standard Hartmann)
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
    max_n_train_points=15,
    source_fractions=(0.01, 0.05, 0.1),
    noise_std=0.0,  # Noise is already built into source function
)

# Create the benchmark
hartmann_tl_3_20_15_regr_benchmark = TransferLearningRegressionBenchmark(
    function=hartmann_tl_3_20_15_regr, 
    settings=benchmark_config
)