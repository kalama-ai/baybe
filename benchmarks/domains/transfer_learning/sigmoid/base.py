"""Base functionality for sigmoid transfer learning benchmarks.

This module provides shared functionality for benchmarks using partially inverted sine functions:
• Target: sin(x) + noise(0, target_noise_std) for x ∈ [0,π] ∪ [π,2π]  
• Source: sin(x) + noise(0, source_noise_std) for x ∈ [0,π], -sin(x) + noise(0, source_noise_std) for x ∈ [π,2π]

Extracted from Karin's notebook research on TL convergence benchmarking.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.parameters.base import DiscreteParameter
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.source_prior import SourcePriorGaussianProcessSurrogate
from baybe.surrogates.transfergpbo import (
    MHGPGaussianProcessSurrogate,
    SHGPGaussianProcessSurrogate,
)
from baybe.targets import NumericalTarget
from baybe.utils.random import temporary_seed
from benchmarks.definition import ConvergenceBenchmarkSettings


def load_data(
    target_noise_std: float = 0.0,
    source_noise_std: float = 0.0,
    random_seed: int = 1337,
) -> pd.DataFrame:
    """Load data for sigmoid partially inverted benchmark with configurable noise.
    
    Args:
        target_noise_std: Standard deviation of noise added to target function
        source_noise_std: Standard deviation of noise added to source function  
        random_seed: Random seed for reproducible noise generation
    
    Returns:
        Combined DataFrame with both source and target data
    """
    # Generate x values exactly as in notebook
    x1 = [(math.pi/8*i) for i in range(9)]  # [0, π/8, ..., π]
    x2 = [(math.pi/8*i)+math.pi for i in range(1, 9)]  # [π + π/8, ..., 2π]
    x = x1 + x2
    
    # Use fixed random seed for reproducible results
    np.random.seed(random_seed)
    
    # Target: sin(x) + noise for all x
    y_target_clean = [math.sin(i) for i in x1+x2]
    if target_noise_std > 0:
        y_target = np.array(y_target_clean) + np.random.normal(0, target_noise_std, size=len(x))
    else:
        y_target = y_target_clean
    
    # Source: sin(x) + noise for x ∈ [0,π], -sin(x) + noise for x ∈ [π,2π]  
    y_source_clean = [math.sin(i) for i in x1] + [-math.sin(i) for i in x2]
    if source_noise_std > 0:
        y_source = np.array(y_source_clean) + np.random.normal(0, source_noise_std, size=len(x))
    else:
        y_source = y_source_clean
    
    # Create DataFrames
    target_df = pd.DataFrame({
        'x': x,
        'Target': y_target,
        'Function': 'Test'
    })
    
    source_df = pd.DataFrame({
        'x': x,
        'Target': y_source,
        'Function': 'Training'
    })
    
    return pd.concat([source_df, target_df], ignore_index=True)


def make_searchspace(
    data: pd.DataFrame,
    use_task_parameter: bool,
) -> SearchSpace:
    """Create searchspace for sigmoid benchmark.
    
    Args:
        data: The sigmoid dataset  
        use_task_parameter: Whether to include task parameter
        
    Returns:
        SearchSpace for the benchmark
    """
    # Extract x values from data
    x_values = sorted(data['x'].unique())
    
    params: list[DiscreteParameter] = [
        NumericalDiscreteParameter(
            name="x",
            values=x_values,
        )
    ]
    
    if use_task_parameter:
        task_param = TaskParameter(
            name="Function",
            values=["Test", "Training"],
            active_values=["Test"],
        )
        params.append(task_param)
    
    return SearchSpace.from_product(parameters=params)


def make_objective() -> SingleTargetObjective:
    """Create objective for sigmoid benchmark.
    
    Returns:
        SingleTargetObjective for Target minimization
    """
    return SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))


def make_lookup(data: pd.DataFrame) -> pd.DataFrame:
    """Create lookup table for target task."""
    return data[data["Function"] == "Test"]


def make_initial_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create initial data for source task."""
    return data[data["Function"] == "Training"]


def sigmoid_tl_benchmark(
    settings: ConvergenceBenchmarkSettings,
    target_noise_std: float = 0.0,
    source_noise_std: float = 0.0,
    optimal_target_value: float = -1.0,
) -> pd.DataFrame:
    """Configurable sigmoid partially inverted transfer learning benchmark.

    Key characteristics:
    • Target task: sin(x) + noise(0, target_noise_std) across [0,2π]
    • Source task: sin(x) + noise(0, source_noise_std) for x ∈ [0,π], 
                  -sin(x) + noise(0, source_noise_std) for x ∈ [π,2π]
    • 17 discrete points across domain
    • Tests transfer learning vs. non-transfer learning approaches
    • Source data percentages: [1%, 5%, 10%]

    Args:
        settings: Configuration settings for the convergence benchmark
        target_noise_std: Standard deviation of noise added to target function
        source_noise_std: Standard deviation of noise added to source function
        optimal_target_value: Expected optimal target value (for tracking)

    Returns:
        DataFrame containing benchmark results
    """
    data = load_data(
        target_noise_std=target_noise_std,
        source_noise_std=source_noise_std,
        random_seed=settings.random_seed,
    )
    
    searchspace = make_searchspace(
        data=data,
        use_task_parameter=True,
    )
    searchspace_nontl = make_searchspace(
        data=data,
        use_task_parameter=False,
    )
    
    lookup = make_lookup(data)
    initial_data = make_initial_data(data)
    objective = make_objective()
    
    # Create campaigns
    index_kernel_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(surrogate_model=GaussianProcessSurrogate()),
        ),
    )
    source_prior_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(
                surrogate_model=SourcePriorGaussianProcessSurrogate()
            ),
        ),
    )
    mhgp_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(
                surrogate_model=MHGPGaussianProcessSurrogate()
            ),
        ),
    )
    shgp_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(
                surrogate_model=SHGPGaussianProcessSurrogate()
            ),
        ),
    )
    nontl_campaign = Campaign(searchspace=searchspace_nontl, objective=objective)
    
    percentages = [0.01, 0.05, 0.1]
    
    initial_data_samples = {}
    with temporary_seed(settings.random_seed):
        for p in percentages:
            initial_data_samples[p] = [
                initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
            ]
    
    results = []
    for p in percentages:
        results.append(
            simulate_scenarios(
                {
                    f"{int(100 * p)}_index_kernel": index_kernel_campaign,
                    f"{int(100 * p)}_source_prior": source_prior_campaign,
                    f"{int(100 * p)}_mhgp": mhgp_campaign,
                    f"{int(100 * p)}_shgp": shgp_campaign,
                    f"{int(100 * p)}_naive": nontl_campaign,
                },
                lookup,
                initial_data=initial_data_samples[p],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
                random_seed=settings.random_seed,
            )
        )
    results.append(
        simulate_scenarios(
            {"0": index_kernel_campaign, "0_naive": nontl_campaign},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
            random_seed=settings.random_seed,
        )
    )
    return pd.concat(results)