"""Forrester transfer learning benchmark with inverted source and noise 0.5.

This benchmark tests transfer learning from an inverted, noisy Forrester function
to the clean original target function:
• Target: Clean original Forrester function (a=1, b=0, c=0, noise=0)
• Source: Inverted Forrester (a=-1, b=0, c=0) with noise_std=0.5
• 100 source points in [0,1] domain
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.forrester.base import (
    INVERTED_CONFIG,
    TARGET_CONFIG,
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_DOE_ITERATIONS,
    DEFAULT_N_MC_ITERATIONS,
    forrester_tl_benchmark,
)


def forrester_inverted_noise_05(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Forrester TL benchmark with inverted function and noise level 0.5.

    This benchmark evaluates transfer learning from negatively correlated source
    to positive target under noise conditions. Tests TL model ability to handle
    sign inversion and extract useful structural information despite correlation
    flip.

    Args:
        settings: Configuration settings for the convergence benchmark.

    Returns:
        DataFrame containing benchmark results across source data fractions.
    """
    source_config = {**INVERTED_CONFIG, "output_noise": 0.5}
    
    return forrester_tl_benchmark(
        settings=settings,
        target_config=TARGET_CONFIG,
        source_config=source_config,
    )


# Benchmark configuration
benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=DEFAULT_BATCH_SIZE,
    n_doe_iterations=DEFAULT_N_DOE_ITERATIONS,
    n_mc_iterations=DEFAULT_N_MC_ITERATIONS,
)

# Benchmark instance
forrester_inverted_noise_05_benchmark = ConvergenceBenchmark(
    function=forrester_inverted_noise_05,
    optimal_target_values={"Target": 6.02074},  # Approximate Forrester maximum
    settings=benchmark_config,
)