"""Hartmann function with increased noise transfer learning benchmark.

This benchmark uses standard vs very high noise Hartmann functions:
• Target: standard 3D Hartmann
• Source: 3D Hartmann with very high noise (noise_std=5)  
• 5 points per dimension, bounds [0,1]³

Extracted from Karin's notebook research on TL convergence benchmarking.
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


def hartmann_increased_noise_tl(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Hartmann with increased noise transfer learning benchmark.

    Key characteristics:
    • Target task: standard 3D Hartmann function
    • Source task: 3D Hartmann with very high noise (noise_std=5)
    • 5 points per dimension, bounds [0,1]³  
    • Tests transfer learning vs. non-transfer learning approaches
    • Source data percentage: 10%

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results
    """
    return hartmann_tl_benchmark(
        settings=settings,
        target_function_factory=lambda: Hartmann(dim=3),
        source_function_factory=lambda: Hartmann(dim=3, noise_std=5),
        bounds=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        points_per_dim=5,
        percentages=[0.1],
    )


# Benchmark configurations
benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=1,
    n_doe_iterations=10,
    n_mc_iterations=50,
)

# Benchmark instance
hartmann_increased_noise_tl_benchmark = ConvergenceBenchmark(
    function=hartmann_increased_noise_tl,
    optimal_target_values={"Target": -2.9997168385},
    settings=benchmark_config,
)