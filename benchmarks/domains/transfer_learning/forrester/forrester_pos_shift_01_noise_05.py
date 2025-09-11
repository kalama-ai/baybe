"""Forrester transfer learning benchmark with positive shift 0.1 and noise 0.5.

This benchmark tests transfer learning from a spatially shifted, noisy Forrester
function to the clean original target function:
• Target: Clean original Forrester function (a=1, b=0, c=0, shift=0, noise=0)
• Source: Right-shifted Forrester (shift=0.1) with noise_std=0.5
• 100 source points in [0,1] domain
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.forrester.base import (
    TARGET_CONFIG,
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_DOE_ITERATIONS,
    DEFAULT_N_MC_ITERATIONS,
    forrester_tl_benchmark,
)


def forrester_pos_shift_01_noise_05(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Forrester TL benchmark with positive shift 0.1 and noise level 0.5.

    This benchmark evaluates transfer learning from moderately shifted source
    domains in positive direction. Tests TL model ability to handle smaller
    positive domain shifts where g(x) = f(x - 0.1).

    Args:
        settings: Configuration settings for the convergence benchmark.

    Returns:
        DataFrame containing benchmark results across source data fractions.
    """
    source_config = {**TARGET_CONFIG, "x_shift": 0.1, "output_noise": 0.5}
    
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
forrester_pos_shift_01_noise_05_benchmark = ConvergenceBenchmark(
    function=forrester_pos_shift_01_noise_05,
    optimal_target_values={"Target": 6.02074},  # Approximate Forrester maximum
    settings=benchmark_config,
)