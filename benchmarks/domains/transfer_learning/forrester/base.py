"""Base functionality for Forrester transfer learning benchmarks.

This module provides shared functionality for benchmarks using various Forrester
function configurations:
• Original Forrester function with different noise levels
• Parameter variants (amplified, low-fidelity, inverted)
• Shifted versions with spatial domain transformations
• Various noise levels for robustness testing

The target function is always the clean original Forrester function.
Source functions use different configurations to test transfer learning scenarios.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch

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


def forrester_function(
    x: torch.Tensor,
    a: float = 1.0,
    b: float = 0.0,
    c: float = 0.0,
    x_shift: float = 0.0,
    output_noise: float = 0.0,
) -> torch.Tensor:
    """Parameterized Forrester function with shift and noise support.
    
    Implements: f(x) = a * (6(x-x_shift) - 2)² * sin(12(x-x_shift) - 4) + 
                       b * (x - 0.5) - c + noise
    
    Args:
        x: Input values, tensor of shape (n_points,) or (n_points, 1)
        a: Scaling factor for the high-fidelity oscillatory term
        b: Linear term coefficient 
        c: Constant offset
        x_shift: Horizontal shift applied to input (positive = right shift)
        output_noise: Standard deviation of Gaussian noise to add
        
    Returns:
        Function values of shape (n_points, 1)
    """
    x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1)
    x_shifted = x.flatten() - x_shift
    
    # Core Forrester function on shifted input
    y_high = torch.pow(6 * x_shifted - 2, 2) * torch.sin(12 * x_shifted - 4)
    y = (a * y_high + b * (x.flatten() - 0.5) - c).reshape(-1, 1)
    
    # Add noise if specified
    if output_noise > 0:
        noise = torch.normal(0.0, output_noise, size=y.shape)
        y += noise
        
    return y


def load_data(
    target_config: dict[str, float],
    source_config: dict[str, float],
    n_points: int = 100,
    bounds: tuple[float, float] = (0.0, 1.0),
    random_seed: int = 1337,
) -> pd.DataFrame:
    """Load data for Forrester benchmarks with configurable source and target.
    
    Args:
        target_config: Parameters for target Forrester function (a, b, c, etc.)
        source_config: Parameters for source Forrester function (a, b, c, etc.)
        n_points: Number of grid points to generate
        bounds: Input domain bounds as (lower, upper)
        random_seed: Random seed for reproducible noise generation
        
    Returns:
        Combined DataFrame with both source and target data
    """
    # Create input grid
    x_values = torch.linspace(bounds[0], bounds[1], n_points)
    
    # Generate target data (always clean original function)
    target_data = pd.DataFrame({"x": x_values.numpy()})
    target_values = forrester_function(x_values, **target_config)
    target_data["Target"] = target_values.numpy().flatten()
    target_data["Function"] = "Target_Function"
    
    # Generate source data with specified configuration
    source_data = pd.DataFrame({"x": x_values.numpy()})
    
    # Set random seed for reproducible source noise
    with temporary_seed(random_seed):
        source_values = forrester_function(x_values, **source_config)
    
    source_data["Target"] = source_values.numpy().flatten()
    source_data["Function"] = "Source_Function"
    
    return pd.concat([source_data, target_data], ignore_index=True)


def make_searchspace(
    data: pd.DataFrame,
    use_task_parameter: bool,
) -> SearchSpace:
    """Create searchspace for Forrester benchmark.
    
    Args:
        data: The Forrester dataset with 'x' coordinate column
        use_task_parameter: Whether to include TaskParameter for TL scenarios
        
    Returns:
        SearchSpace for the benchmark
    """
    params: list[DiscreteParameter] = [
        NumericalDiscreteParameter(
            name="x",
            values=sorted(data["x"].unique()),
        )
    ]
    
    if use_task_parameter:
        task_param = TaskParameter(
            name="Function",
            values=["Target_Function", "Source_Function"],
            active_values=["Target_Function"],
        )
        params.append(task_param)
    
    return SearchSpace.from_product(parameters=params)


def make_objective() -> SingleTargetObjective:
    """Create objective for Forrester benchmark.
    
    Returns:
        SingleTargetObjective for Target maximization
    """
    return SingleTargetObjective(target=NumericalTarget(name="Target", mode="MAX"))


def make_lookup(data: pd.DataFrame) -> pd.DataFrame:
    """Create lookup table for target task.
    
    Args:
        data: Combined source and target data
        
    Returns:
        DataFrame containing only target task data
    """
    return data[data["Function"] == "Target_Function"]


def make_initial_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create initial data for source task.
    
    Args:
        data: Combined source and target data
        
    Returns:
        DataFrame containing only source task data
    """
    return data[data["Function"] == "Source_Function"]


def forrester_tl_benchmark(
    settings: ConvergenceBenchmarkSettings,
    target_config: dict[str, float],
    source_config: dict[str, float],
    percentages: list[float] | None = None,
    n_points: int = 100,
) -> pd.DataFrame:
    """Configurable Forrester transfer learning convergence benchmark.

    Args:
        settings: Configuration settings for the convergence benchmark
        target_config: Parameters for target Forrester function
        source_config: Parameters for source Forrester function
        percentages: Source data percentages to test
        n_points: Number of points in source domain

    Returns:
        DataFrame containing benchmark results
    """
    if percentages is None:
        percentages = [0.01, 0.02, 0.04, 0.08, 0.1, 0.16, 0.2, 0.5]
        
    data = load_data(
        target_config=target_config,
        source_config=source_config,
        n_points=n_points,
        random_seed=settings.random_seed,
    )
    
    searchspace = make_searchspace(data=data, use_task_parameter=True)
    searchspace_nontl = make_searchspace(data=data, use_task_parameter=False)
    
    lookup = make_lookup(data)
    initial_data = make_initial_data(data)
    objective = make_objective()
    
    # Create transfer learning campaigns
    index_kernel_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(
                surrogate_model=GaussianProcessSurrogate()
            ),
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
    
    # Generate initial data samples for each percentage
    initial_data_samples = {}
    with temporary_seed(settings.random_seed):
        for p in percentages:
            initial_data_samples[p] = [
                initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
            ]
    
    # Run benchmark scenarios
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
    
    # Add baseline scenario without source data
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


# Standard configurations for different Forrester variants
TARGET_CONFIG = {"a": 1.0, "b": 0.0, "c": 0.0, "x_shift": 0.0, "output_noise": 0.0}

AMPLIFIED_CONFIG = {"a": 1.5, "b": -2.0, "c": 1.0, "x_shift": 0.0}
LOW_FIDELITY_CONFIG = {"a": 0.5, "b": 10.0, "c": -5.0, "x_shift": 0.0}
INVERTED_CONFIG = {"a": -1.0, "b": 0.0, "c": 0.0, "x_shift": 0.0}

# Global benchmark settings
DEFAULT_N_DOE_ITERATIONS = 10
DEFAULT_BATCH_SIZE = 1
DEFAULT_N_MC_ITERATIONS = 30