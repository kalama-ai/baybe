"""TL regression benchmark for sigmoid function partially inverted.

Target task: sin(x) for x ∈ [0,π] ∪ [π,2π]  
Source task: sin(x) for x ∈ [0,π], -sin(x) for x ∈ [π,2π] (partially inverted)
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    TransferLearningRegressionBenchmark,
    TransferLearningRegressionBenchmarkSettings,
)
from benchmarks.domains.regression.base import run_tl_regression_benchmark
from benchmarks.domains.transfer_learning.sigmoid.base import (
    load_data,
    make_objective,
    make_searchspace,
)


def sigmoid_partially_inverted_tl_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses partially inverted sine functions as tasks:
    • Target task: sin(x) across [0,2π]
    • Source task: sin(x) for x ∈ [0,π], -sin(x) for x ∈ [π,2π] (partially inverted)

    Uses the same configuration as the original convergence benchmark:
    • 17 discrete points across [0,2π] domain
    • No noise added to functions
    • Sine function (minimization objective)

    Key characteristics:
    • Parameters:
      - x: Numerical discrete (17 points from 0 to 2π)
      - Function: Task parameter (Test, Training)
    • Target: Sine function output (continuous)
    • Objective: Minimization (minimum of sin(x) is -1)
    • Compares TL models vs vanilla GP on predictive performance

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame with benchmark results
    """
    return run_tl_regression_benchmark(
        settings=settings,
        data_loader=lambda: load_data(
            target_noise_std=0.0,
            source_noise_std=0.0,
            random_seed=settings.random_seed,
        ),
        searchspace_factory=make_searchspace,
        objective_factory=make_objective,
    )


# Define the benchmark settings
benchmark_config = TransferLearningRegressionBenchmarkSettings(
    n_mc_iterations=50,
    max_n_train_points=10,
    source_fractions=(0.01, 0.05, 0.1),
    noise_std=0.0,  # No additional noise
)

# Create the benchmark
sigmoid_partially_inverted_tl_regr_benchmark = TransferLearningRegressionBenchmark(
    function=sigmoid_partially_inverted_tl_regr, 
    settings=benchmark_config
)