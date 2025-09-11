"""Forrester transfer learning benchmark with low-fidelity source and noise 2.0.

This benchmark tests transfer learning from a low-fidelity, highly noisy Forrester
function to the clean original target function:
• Target: Clean original Forrester function (a=1, b=0, c=0, noise=0)
• Source: Low-fidelity Forrester (a=0.5, b=10, c=-5) with noise_std=2.0
• 100 source points in [0,1] domain
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.forrester.base import (
    LOW_FIDELITY_CONFIG,
    TARGET_CONFIG,
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_DOE_ITERATIONS,
    DEFAULT_N_MC_ITERATIONS,
    forrester_tl_benchmark,
)


def forrester_low_fid_noise_2(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Forrester TL benchmark with low-fidelity function and noise level 2.0.

    This benchmark evaluates transfer learning under challenging conditions:
    low-fidelity source with high noise levels. Tests the limits of TL model
    performance when source quality is severely degraded.

    Args:
        settings: Configuration settings for the convergence benchmark.

    Returns:
        DataFrame containing benchmark results across source data fractions.
    """
    source_config = {**LOW_FIDELITY_CONFIG, "output_noise": 2.0}
    
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
forrester_low_fid_noise_2_benchmark = ConvergenceBenchmark(
    function=forrester_low_fid_noise_2,
    optimal_target_values={"Target": 6.02074},  # Approximate Forrester maximum
    settings=benchmark_config,
)