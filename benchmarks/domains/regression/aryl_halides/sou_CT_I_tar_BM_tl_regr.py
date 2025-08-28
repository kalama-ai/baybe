"""TL regression benchmark for aryl halide reactions.

Source task:
    - 1-chloro-4-(trifluoromethyl)benzene
    # - 2-iodopyridine  # Commented out for SourcePrior GP compatibility (single source only)
Target task: 1-iodo-4-methoxybenzene
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


def aryl_halide_CT_I_BM_tl_regr(
    settings: TransferLearningRegressionBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark function for comparing regression performance of GP vs TL models.

    This benchmark uses different aryl halide substrates as tasks:
    - Source task: 1-chloro-4-(trifluoromethyl)benzene
    # - Source task: 2-iodopyridine (commented out for SourcePrior GP compatibility)
    - Target task: 1-iodo-4-methoxybenzene

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
        source_tasks=[
            "1-chloro-4-(trifluoromethyl)benzene"
        ],  # Single source for SourcePrior GP compatibility
        target_tasks=["1-iodo-4-methoxybenzene"],
    )


# Define the benchmark settings
benchmark_config = TransferLearningRegressionBenchmarkSettings(
    n_mc_iterations=30,
    max_n_train_points=10,
    source_fractions=(0.01, 0.05, 0.1, 0.2),
    noise_std=0.0,
)

# Create the benchmark
aryl_halide_CT_I_BM_tl_regr_benchmark = TransferLearningRegressionBenchmark(
    function=aryl_halide_CT_I_BM_tl_regr, settings=benchmark_config
)
