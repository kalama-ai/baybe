"""Sigmoid function partially inverted transfer learning benchmark.

This benchmark uses a partially inverted sine function as source and target tasks:
• Target: sin(x) for x ∈ [0,π] ∪ [π,2π]  
• Source: sin(x) for x ∈ [0,π], -sin(x) for x ∈ [π,2π] (partially inverted)

Extracted from Karin's notebook research on TL convergence benchmarking.
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    ConvergenceBenchmark,
    ConvergenceBenchmarkSettings,
)
from benchmarks.domains.transfer_learning.sigmoid.base import sigmoid_tl_benchmark


def sigmoid_partially_inverted_tl(settings: ConvergenceBenchmarkSettings) -> pd.DataFrame:
    """Sigmoid partially inverted transfer learning benchmark without noise.

    Key characteristics:
    • Target task: sin(x) across [0,2π]
    • Source task: sin(x) for x ∈ [0,π], -sin(x) for x ∈ [π,2π] (partially inverted)
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
        target_noise_std=0.0,
        source_noise_std=0.0,
        optimal_target_value=-1.0,
    )


# Benchmark configurations
benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=1,
    n_doe_iterations=10,
    n_mc_iterations=50,
)

# Benchmark instance
sigmoid_partially_inverted_tl_benchmark = ConvergenceBenchmark(
    function=sigmoid_partially_inverted_tl,
    optimal_target_values={"Target": -1.0},
    settings=benchmark_config,
)