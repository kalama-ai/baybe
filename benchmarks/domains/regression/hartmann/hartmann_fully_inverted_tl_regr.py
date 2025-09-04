"""TL regression benchmark for fully inverted Hartmann function.

Target task: Standard 3D Hartmann function
Source task: Negated 3D Hartmann function with noise (std=0.15)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from botorch.test_functions.synthetic import Hartmann

from benchmarks.definition import (
    TransferLearningRegressionBenchmark,
    TransferLearningRegressionBenchmarkSettings,
)
from benchmarks.domains.regression.base import run_tl_regression_benchmark
from benchmarks.domains.transfer_learning.hartmann.base import (
    load_data,
    make_objective,
    make_searchspace,
)


def hartmann_fully_inverted_tl_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses standard vs fully inverted Hartmann functions:
    • Target task: Standard 3D Hartmann function
    • Source task: Negated 3D Hartmann function with noise (std=0.15)

    Uses the same configuration as the original convergence benchmark:
    • 5 points per dimension 
    • Bounds: [0, 1] for all three dimensions (x0, x1, x2)
    • Discrete grid search space
    • Hartmann function (minimization objective)

    Key characteristics:
    • Parameters:
      - x0, x1, x2: Numerical discrete (5 points each from 0 to 1)
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
        data_loader=lambda: load_data(
            target_function_factory=lambda: Hartmann(dim=3),
            source_function_factory=lambda: Hartmann(dim=3, negate=True, noise_std=0.15),
            bounds=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            points_per_dim=5,
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
    noise_std=0.0,  # Noise is already built into source function
)

# Create the benchmark
hartmann_fully_inverted_tl_regr_benchmark = TransferLearningRegressionBenchmark(
    function=hartmann_fully_inverted_tl_regr, 
    settings=benchmark_config
)