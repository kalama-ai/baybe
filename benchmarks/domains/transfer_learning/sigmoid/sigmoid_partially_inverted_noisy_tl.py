"""Sigmoid function partially inverted with noise transfer learning benchmark.

This benchmark uses a partially inverted sine function with noise as source and target tasks:
• Target: sin(x) + noise(0, 0.5) for x ∈ [0,π] ∪ [π,2π]  
• Source: sin(x) + noise(0, 0.1) for x ∈ [0,π], -sin(x) + noise(0, 0.1) for x ∈ [π,2π]

Extracted from Karin's notebook research on TL convergence benchmarking.
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.sigmoid.base import sigmoid_tl_benchmark


def sigmoid_partially_inverted_noisy_tl(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Sigmoid partially inverted transfer learning benchmark with noise.

    Key characteristics:
    • Target task: sin(x) + noise(0, 0.5) across [0,2π]
    • Source task: sin(x) + noise(0, 0.1) for x ∈ [0,π], -sin(x) + noise(0, 0.1) for x ∈ [π,2π]
    • 17 discrete points across domain
    • Tests transfer learning vs. non-transfer learning approaches
    • Source data percentages: [1%, 5%, 10%]

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results
    """
    return sigmoid_tl_benchmark(
        settings=settings,
        target_noise_std=0.5,
        source_noise_std=0.1,
        optimal_target_value=-1.5,  # Approximate considering noise
    )


# Benchmark configurations
benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=1,
    n_doe_iterations=10,
    n_mc_iterations=50,
)

# Benchmark instance
sigmoid_partially_inverted_noisy_tl_benchmark = ConvergenceBenchmark(
    function=sigmoid_partially_inverted_noisy_tl,
    optimal_target_values={"Target": -1.5},
    settings=benchmark_config,
)