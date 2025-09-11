"""Forrester TL regression benchmark with negative shift -0.15 and noise 0.5.

This regression benchmark corresponds to the transfer learning convergence
benchmark and evaluates predictive accuracy of TL vs non-TL models:
• Target: Clean original Forrester function (a=1, b=0, c=0, noise=0)
• Source: Shifted Forrester (x_shift=-0.15) with noise_std=0.5
• Evaluates regression metrics on held-out target data
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    TransferLearningRegressionBenchmark,
    TransferLearningRegressionBenchmarkSettings,
)
from benchmarks.domains.regression.base import run_tl_regression_benchmark
from benchmarks.domains.transfer_learning.forrester.base import (
    TARGET_CONFIG,
    load_data,
    make_objective,
    make_searchspace,
)


def forrester_neg_shift_015_noise_05_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Forrester TL regression benchmark with negative shift -0.15 and noise 0.5.

    This benchmark evaluates regression performance when transferring from
    left-shifted source to clean target. Tests TL model ability to handle
    spatial domain shifts under moderate noise conditions.

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame with regression benchmark results.
    """
    source_config = {**TARGET_CONFIG, "x_shift": -0.15, "output_noise": 0.5}
    
    return run_tl_regression_benchmark(
        settings=settings,
        data_loader=lambda: load_data(
            target_config=TARGET_CONFIG,
            source_config=source_config,
        ),
        searchspace_factory=make_searchspace,
        objective_factory=make_objective,
    )


# Benchmark configuration
benchmark_config = TransferLearningRegressionBenchmarkSettings(
    n_mc_iterations=30,
    max_n_train_points=10,
    source_fractions=(0.01, 0.02, 0.04, 0.08, 0.1, 0.16, 0.2, 0.5),
    noise_std=0.0,
)

# Benchmark instance
forrester_neg_shift_015_noise_05_regr_benchmark = TransferLearningRegressionBenchmark(
    function=forrester_neg_shift_015_noise_05_regr,
    settings=benchmark_config,
)