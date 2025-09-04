"""Base functionality for aryl halides transfer learning regression benchmarks."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from baybe.objectives import SingleTargetObjective
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from benchmarks.definition import TransferLearningRegressionBenchmarkSettings
from benchmarks.domains.regression.base import run_tl_regression_benchmark
from benchmarks.domains.transfer_learning.aryl_halides.base import (
    load_data,
)
from benchmarks.domains.transfer_learning.aryl_halides.base import (
    make_searchspace as _make_searchspace,
)


def make_searchspace(
    data: pd.DataFrame,
    use_task_parameter: bool,
    source_tasks: Sequence[str] | None = None,
    target_tasks: Sequence[str] | None = None,
) -> SearchSpace:
    """Create the search space for aryl halides regression benchmarks.

    Args:
        data: The aryl halides dataset
        use_task_parameter: Whether to include task parameter (True for TL, False for vanilla)
        source_tasks: List of source task identifiers (for TL benchmarks)
        target_tasks: List of target task identifiers (for TL benchmarks)

    Returns:
        SearchSpace for the benchmark
    """
    if use_task_parameter and source_tasks is not None and target_tasks is not None:
        return _make_searchspace(
            data=data,
            source_tasks=source_tasks,
            target_tasks=target_tasks,
        )
    else:
        return _make_searchspace(data=data)


def make_objective() -> SingleTargetObjective:
    """Create the objective for aryl halides benchmarks.

    Returns:
        SingleTargetObjective for yield maximization
    """
    return SingleTargetObjective(NumericalTarget(name="yield", mode="MAX"))


def run_aryl_halide_tl_regression_benchmark(
    settings: TransferLearningRegressionBenchmarkSettings,
    source_tasks: Sequence[str],
    target_tasks: Sequence[str],
) -> pd.DataFrame:
    """Run aryl halides transfer learning regression benchmark.

    Args:
        settings: The benchmark settings
        source_tasks: List of source task identifiers
        target_tasks: List of target task identifiers

    Returns:
        DataFrame with benchmark results
    """

    # Create wrapper functions that match the expected signature for run_tl_regression_benchmark
    def make_searchspace_wrapper(
        data: pd.DataFrame, use_task_parameter: bool
    ) -> SearchSpace:
        return make_searchspace(
            data=data,
            use_task_parameter=use_task_parameter,
            source_tasks=source_tasks,
            target_tasks=target_tasks,
        )

    return run_tl_regression_benchmark(
        settings=settings,
        data_loader=load_data,
        searchspace_factory=make_searchspace_wrapper,
        objective_factory=make_objective,
    )
