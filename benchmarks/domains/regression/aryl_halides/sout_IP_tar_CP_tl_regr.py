"""TL regression benchmark for aryl halide reactions.

Source task: 2-iodopyridine
Target task: 3-chloropyridine
"""

from __future__ import annotations

import pandas as pd

from benchmarks.definition import (
    TransferLearningRegressionBenchmark,
    TransferLearningRegressionBenchmarkSettings,
)
from benchmarks.domains.regression.aryl_halides.base import (
    run_aryl_halide_tl_regression_benchmark,
)


def aryl_halide_IP_CP_tl_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses different aryl halide substrates as tasks:
    - Source task: 2-iodopyridine
    - Target task: 3-chloropyridine

    It trains models with varying amounts of source and target data, and evaluates
    their predictive performance on held-out target data.

    Key characteristics:
    • Parameters:
      - Base: Substance with MORDRED encoding
      - Ligand: Substance with MORDRED encoding
      - Additive: Substance with MORDRED encoding
      - aryl_halide: Task parameter
    • Target: Reaction yield (continuous)
    • Objective: Maximization
    • Compares TL models (SourcePrior, MHGP, SHGP, Index Kernel) vs vanilla GP

    Args:
        settings: The benchmark settings.

    Returns:
        DataFrame with benchmark results
    """
    return run_aryl_halide_tl_regression_benchmark(
        settings=settings,
        source_tasks=["2-iodopyridine"],
        target_tasks=["3-chloropyridine"],
    )


# Define the benchmark settings
benchmark_config = TransferLearningRegressionBenchmarkSettings(
    n_mc_iterations=30,
    max_n_train_points=10,
    source_fractions=(0.01, 0.05, 0.1, 0.2),
    noise_std=0.0,
)

# Create the benchmark
aryl_halide_IP_CP_tl_regr_benchmark = TransferLearningRegressionBenchmark(
    function=aryl_halide_IP_CP_tl_regr, settings=benchmark_config
)
