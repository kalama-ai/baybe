"""TL regression benchmark for partially inverted Hartmann function.

Target task: Hartmann(x) for x≤1, -Hartmann(x-1) for x>1 (partially inverted)
Source task: -Hartmann(x) for x≤1, -Hartmann(x-1) for x>1 (fully inverted)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.definition import (
    TransferLearningRegressionBenchmark,
    TransferLearningRegressionBenchmarkSettings,
)
from benchmarks.domains.regression.base import run_tl_regression_benchmark
from benchmarks.domains.transfer_learning.hartmann.base import (
    hartmann_mod,
    load_data,
    make_objective,
    make_searchspace,
)


def hartmann_partially_inverted_tl_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses partially inverted Hartmann functions:
    • Target task: Hartmann(x) for x≤1, -Hartmann(x-1) for x>1 (partially inverted)
    • Source task: -Hartmann(x) for x≤1, -Hartmann(x-1) for x>1 (fully inverted)

    Uses the same configuration as the original convergence benchmark:
    • 9 points per dimension 
    • Bounds: [0, 2] for all three dimensions (x0, x1, x2)
    • Discrete grid search space
    • Partially inverted Hartmann function (minimization objective)

    Key characteristics:
    • Parameters:
      - x0, x1, x2: Numerical discrete (9 points each from 0 to 2)
      - Function: Task parameter (Target_Function, Source_Function)
    • Target: Partially inverted Hartmann function output (continuous)
    • Objective: Minimization 
    • Compares TL models vs vanilla GP on predictive performance

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame with benchmark results
    """
    return run_tl_regression_benchmark(
        settings=settings,
        data_loader=lambda: load_data(
            target_function_factory=lambda: hartmann_mod(invert_first=False, invert_second=True),
            source_function_factory=lambda: hartmann_mod(invert_first=True, invert_second=True),
            bounds=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
            points_per_dim=9,
            random_seed=settings.random_seed,
        ),
        searchspace_factory=make_searchspace,
        objective_factory=make_objective,
    )


# Define the benchmark settings  
benchmark_config = TransferLearningRegressionBenchmarkSettings(
    n_mc_iterations=50,
    max_n_train_points=10,
    source_fractions=(0.1,),
    noise_std=0.0,  # No additional noise
)

# Create the benchmark
hartmann_partially_inverted_tl_regr_benchmark = TransferLearningRegressionBenchmark(
    function=hartmann_partially_inverted_tl_regr, 
    settings=benchmark_config
)