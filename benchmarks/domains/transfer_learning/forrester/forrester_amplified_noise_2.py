"""Forrester transfer learning benchmark with amplified high-noise source.

This benchmark tests transfer learning from an amplified, highly noisy Forrester
function to the clean original target function:
• Target: Clean original Forrester function (a=1, b=0, c=0, noise=0)
• Source: Amplified Forrester (a=1.5, b=-2, c=1) with noise_std=2.0
• 100 source points in [0,1] domain
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.forrester.base import (
    AMPLIFIED_CONFIG,
    TARGET_CONFIG,
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_DOE_ITERATIONS,
    DEFAULT_N_MC_ITERATIONS,
    forrester_tl_benchmark,
)


def forrester_amplified_noise_2(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Forrester TL benchmark with amplified function and noise level 2.0.

    This benchmark evaluates transfer learning under challenging conditions:
    both significant parameter differences and high noise levels. Tests the
    robustness limits of TL models for cross-parameter transfer.

    Args:
        settings: Configuration settings for the convergence benchmark.

    Returns:
        DataFrame containing benchmark results across source data fractions.
    """
    source_config = {**AMPLIFIED_CONFIG, "output_noise": 2.0}
    
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
forrester_amplified_noise_2_benchmark = ConvergenceBenchmark(
    function=forrester_amplified_noise_2,
    optimal_target_values={"Target": 6.02074},  # Approximate Forrester maximum
    settings=benchmark_config,
)