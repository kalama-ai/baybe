"""Transfer learning benchmark with inverted Hartmann functions as tasks.

This benchmark uses standard vs noisy Hartmann functions:
• Target: standard 3D Hartmann
• Source: 3D Hartmann with added noise (noise_std=0.15)
• 20 points per dimension, bounds [0,1]³
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from botorch.test_functions.synthetic import Hartmann

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.hartmann.base import hartmann_tl_benchmark


def hartmann_tl_3_20_15(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Benchmark function for transfer learning with the Hartmann function in 3D.

    Key characteristics:
    • Compares two versions of Hartmann function:
      - Target: standard Hartmann
      - Source: Hartmann with added noise (noise_std=0.15)
    • Uses 20 points per dimension
    • Tests transfer learning with different source data percentages:
      - 1% of source data
      - 5% of source data
      - 10% of source data

    Args:
        settings: Configuration settings for the convergence benchmark.

    Returns:
        DataFrame containing benchmark results.
    """
    return hartmann_tl_benchmark(
        settings=settings,
        target_function_factory=lambda: Hartmann(dim=3),
        source_function_factory=lambda: Hartmann(dim=3, noise_std=0.15),
        bounds=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        points_per_dim=20,
        percentages=[0.01, 0.05, 0.1],
    )


# Benchmark configurations
benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=25,
    n_mc_iterations=75,
)

# Benchmark instance
hartmann_tl_3_20_15_benchmark = ConvergenceBenchmark(
    function=hartmann_tl_3_20_15,
    optimal_target_values={"Target": -3.8324342573},
    settings=benchmark_config,
)