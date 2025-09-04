"""Hartmann function fully inverted transfer learning benchmark.

This benchmark uses standard vs fully inverted Hartmann functions:
• Target: standard 3D Hartmann
• Source: negated 3D Hartmann with noise (noise_std=0.15)
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


def hartmann_fully_inverted_tl(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Hartmann fully inverted transfer learning benchmark.

    Key characteristics:
    • Target task: standard 3D Hartmann function
    • Source task: negated 3D Hartmann with noise (noise_std=0.15)
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
        source_function_factory=lambda: Hartmann(dim=3, negate=True, noise_std=0.15),
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
hartmann_fully_inverted_tl_benchmark = ConvergenceBenchmark(
    function=hartmann_fully_inverted_tl,
    optimal_target_values={"Target": -2.9997168385},
    settings=benchmark_config,
)