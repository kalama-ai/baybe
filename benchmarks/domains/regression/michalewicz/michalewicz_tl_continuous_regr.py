"""TL regression benchmark for Michalewicz function in 5D continuous space.

Target task: Standard negated Michalewicz function (5D)
Source task: Negated Michalewicz function with noise (std=0.15)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import Michalewicz

from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.parameters.base import Parameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.random import temporary_seed
from benchmarks.definition import (
    TransferLearningRegressionBenchmark,
    TransferLearningRegressionBenchmarkSettings,
)
from benchmarks.domains.regression.base import run_tl_regression_benchmark


def wrap_function(
    function, function_name: str, df: pd.DataFrame
) -> pd.DataFrame:
    """Wrap the given function such that it operates on DataFrames.
    
    Imported from original convergence benchmark - handles function evaluation
    and proper column management for Task parameters.
    """
    input_columns = df.columns.tolist()

    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    # Apply Michalewicz function to each row
    def apply_function(row):
        # Remove the "Function" column but record its value such that it can be added
        # back to the row after the function evaluation
        if "Function" in row.index:
            function_value = row["Function"]
            row = row.drop("Function")
        else:
            function_value = None  # Default if column doesn't exist

        x = torch.tensor(row.values.astype(float), dtype=torch.float64).unsqueeze(0)
        # Re-add the "Function" column to the row
        if function_value is not None:
            row["Function"] = function_value
        return function(x).item()

    result_df["Target"] = (
        result_df[input_columns].apply(apply_function, axis=1).astype(float)
    )
    # Add a column "Function" with the function name
    result_df["Function"] = function_name

    return result_df


def load_data() -> pd.DataFrame:
    """Load data for Michalewicz benchmark - extracted from original convergence benchmark.
    
    Creates synthetic data using the same parameters as the original:
    • 5D Michalewicz function 
    • Bounds: [0, π] for all dimensions
    • Target function: standard negated Michalewicz
    • Source function: negated Michalewicz with noise (std=0.15)
    • Uses random sampling (not grid-based like other benchmarks)
    
    Returns:
        DataFrame containing the Michalewicz benchmark data
    """
    # Define functions same as original
    target_function = Michalewicz(dim=5, negate=True)
    source_function = Michalewicz(dim=5, negate=True, noise_std=0.15)
    
    # Generate more comprehensive data for regression evaluation
    # Use larger sample size than original to ensure sufficient data
    n_samples_per_task = 1000  # Sufficient for train/test splits
    
    # Use same random seed as original benchmark for reproducibility
    with temporary_seed(1337):
        # Generate source data
        source_samples = np.random.uniform(
            low=0, high=math.pi, size=(n_samples_per_task, 5)
        )
        source_df = pd.DataFrame(
            source_samples, columns=[f"x{i}" for i in range(5)]
        )
        source_df = wrap_function(source_function, "Source_Function", source_df)
        
        # Generate target data  
        target_samples = np.random.uniform(
            low=0, high=math.pi, size=(n_samples_per_task, 5)
        )
        target_df = pd.DataFrame(
            target_samples, columns=[f"x{i}" for i in range(5)]
        )
        target_df = wrap_function(target_function, "Target_Function", target_df)
    
    # Combine source and target data
    return pd.concat([source_df, target_df], ignore_index=True)


def make_searchspace(
    data: pd.DataFrame,
    use_task_parameter: bool,
) -> SearchSpace:
    """Create the search space for Michalewicz benchmark - extracted from original.
    
    Args:
        data: The Michalewicz dataset (not used, continuous bounds are fixed)
        use_task_parameter: Whether to include task parameter (True for TL, False for vanilla)
        
    Returns:
        SearchSpace for the benchmark
    """
    # 5D continuous parameters with bounds [0, π] as in original
    params: list[Parameter] = [
        NumericalContinuousParameter(
            name=f"x{k}",
            bounds=(0, math.pi),
        )
        for k in range(5)
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
    """Create the objective for Michalewicz benchmark - extracted from original.
    
    Returns:
        SingleTargetObjective for Target maximization (negated Michalewicz)
    """
    # Original uses negated Michalewicz, so mode="MAX"
    return SingleTargetObjective(
        target=NumericalTarget(name="Target", mode="MAX")
    )


def michalewicz_tl_continuous_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses different versions of the 5D Michalewicz function as tasks:
    • Target task: Standard negated 5D Michalewicz function
    • Source task: Negated 5D Michalewicz function with added noise (std=0.15)

    Uses the same configuration as the original convergence benchmark:
    • 5D continuous space
    • Bounds: [0, π] for all dimensions (x0, x1, x2, x3, x4)
    • Continuous search space (not discrete grid)
    • Negated Michalewicz function (maximization objective)

    Key characteristics:
    • Parameters:
      - x0, x1, x2, x3, x4: Numerical continuous (bounds [0, π])
      - Function: Task parameter (Target_Function, Source_Function)
    • Target: Michalewicz function output (continuous)
    • Objective: Maximization (negated Michalewicz)
    • Compares TL models vs vanilla GP on predictive performance
    • Uses random sampling (not grid-based like other benchmarks)

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
    n_mc_iterations=20,
    max_n_train_points=20,  # Reasonable for 5D continuous space
    source_fractions=(0.01, 0.05, 0.1),
    noise_std=0.0,  # Noise is already built into source function
)

# Create the benchmark
michalewicz_tl_continuous_regr_benchmark = TransferLearningRegressionBenchmark(
    function=michalewicz_tl_continuous_regr, 
    settings=benchmark_config
)