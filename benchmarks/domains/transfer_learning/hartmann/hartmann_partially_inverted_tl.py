"""Hartmann function partially inverted transfer learning benchmark.

This benchmark uses partially inverted Hartmann functions:
• Target: Hartmann(x) for x≤1, -Hartmann(x-1) for x>1 (partially inverted)
• Source: -Hartmann(x) for x≤1, -Hartmann(x-1) for x>1 (fully inverted)
• 9 points per dimension, bounds [0,2]³

Extracted from Karin's notebook research on TL convergence benchmarking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.hartmann.base import (
    hartmann_mod,
    hartmann_tl_benchmark,
)


def hartmann_partially_inverted_tl(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Hartmann partially inverted transfer learning benchmark.

    Key characteristics:
    • Target task: Hartmann(x) for x≤1, -Hartmann(x-1) for x>1 (partially inverted)
    • Source task: -Hartmann(x) for x≤1, -Hartmann(x-1) for x>1 (fully inverted)
    • 9 points per dimension, bounds [0,2]³
    • Tests transfer learning vs. non-transfer learning approaches  
    • Source data percentage: 10%

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results
    """
    return hartmann_tl_benchmark(
        settings=settings,
        target_function_factory=lambda: hartmann_mod(invert_first=False, invert_second=True),
        source_function_factory=lambda: hartmann_mod(invert_first=True, invert_second=True),
        bounds=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        points_per_dim=9,
        percentages=[0.1],
    )


# Benchmark configurations  
benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=1,
    n_doe_iterations=10,
    n_mc_iterations=50,
)

# Benchmark instance
hartmann_partially_inverted_tl_benchmark = ConvergenceBenchmark(
    function=hartmann_partially_inverted_tl,
    optimal_target_values={"Target": -2.9997168385},
    settings=benchmark_config,
)