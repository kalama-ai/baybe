"""Forrester TL regression benchmark with low-fidelity source and noise 2.0.

This regression benchmark corresponds to the transfer learning convergence
benchmark and evaluates predictive accuracy of TL vs non-TL models:
• Target: Clean original Forrester function (a=1, b=0, c=0, noise=0)
• Source: Low-fidelity Forrester (a=0.5, b=10, c=-5) with noise_std=2.0
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
    LOW_FIDELITY_CONFIG,
    TARGET_CONFIG,
    load_data,
    make_objective,
    make_searchspace,
)


def forrester_low_fid_noise_2_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Forrester TL regression benchmark with low-fidelity function and noise 2.0.

    This benchmark evaluates regression performance when transferring from
    low-fidelity to high-fidelity functions under high noise conditions.
    Tests TL model ability to handle both parameter differences and noise.

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame with regression benchmark results.
    """
    source_config = {**LOW_FIDELITY_CONFIG, "output_noise": 2.0}
    
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
forrester_low_fid_noise_2_regr_benchmark = TransferLearningRegressionBenchmark(
    function=forrester_low_fid_noise_2_regr,
    settings=benchmark_config,
)