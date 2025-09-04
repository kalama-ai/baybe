"""Base functionality for Hartmann transfer learning benchmarks.

This module provides shared functionality for benchmarks using various Hartmann function configurations:
• Standard Hartmann vs noisy Hartmann
• Standard vs negated Hartmann  
• Partially inverted Hartmann functions
• Various grid sizes and bounds

Supports both existing and new benchmarks from Karin's notebook research.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
from botorch.test_functions.synthetic import Hartmann

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


def botorch_function_wrapper(test_function):
    """Turn a BoTorch test function into a format accepted by lookup in simulations."""
    def wrapper(*x: float) -> float:
        x_tensor = torch.tensor(x, dtype=torch.float64)
        result = test_function.forward(x_tensor)
        return float(result)
    return wrapper


def hartmann_mod(invert_first: bool, invert_second: bool):
    """Create partially inverted Hartmann function.
    
    Expects inputs in range [0,2]³. When inputs are all ≤1 use first function,
    when they are all >1 use second function on (x-1) transformed space.
    
    Args:
        invert_first: Whether to negate Hartmann for x ≤ 1 region
        invert_second: Whether to negate Hartmann for x > 1 region
        
    Returns:
        Partially inverted Hartmann function
    """
    def wrapper(x):
        x = np.array(x)
        if (x <= 1).all():
            y = Hartmann(dim=3, negate=invert_first).forward(torch.tensor(x))
        elif (x > 1).all():
            y = Hartmann(dim=3, negate=invert_second).forward(torch.tensor(x) - 1)
        else:
            y = 0  # Boundary region
        return float(y)
    return wrapper


def load_data(
    target_function_factory: Callable[[], Callable],
    source_function_factory: Callable[[], Callable],
    bounds: np.ndarray,
    points_per_dim: int,
    random_seed: int = 1337,
) -> pd.DataFrame:
    """Load data for Hartmann benchmarks with configurable functions and grid.
    
    Args:
        target_function_factory: Factory function that creates target function
        source_function_factory: Factory function that creates source function  
        bounds: Array of shape (2, dim) specifying lower and upper bounds
        points_per_dim: Number of grid points per dimension
        random_seed: Random seed for reproducible results
        
    Returns:
        Combined DataFrame with both source and target data
    """
    # Create grid locations
    grid_locations = {
        f"x{d}": np.linspace(lower, upper, points_per_dim)
        for d, (lower, upper) in enumerate(bounds.T)
    }
    
    # Create functions
    target_function = target_function_factory()
    source_function = source_function_factory()
    
    # Create meshgrid for all combinations
    meshgrid = np.meshgrid(*[points for points in grid_locations.values()])
    coord_columns = [f"x{d}" for d in range(len(bounds.T))]
    
    # Generate source data
    source_data = pd.DataFrame(
        {f"x{d}": grid_d.ravel() for d, grid_d in enumerate(meshgrid)},
        columns=coord_columns,
    )
    
    # Use random seed for reproducible source function evaluation (if noisy)
    with temporary_seed(random_seed):
        if hasattr(source_function, 'forward'):
            # BoTorch function - use tensor input
            source_tensor = torch.tensor(source_data[coord_columns].values)
            source_targets = source_function(source_tensor).detach().numpy()
        else:
            # Custom function - apply row-wise
            source_targets = source_data[coord_columns].apply(source_function, axis=1).values
    
    source_data["Target"] = source_targets
    source_data["Function"] = "Source_Function"
    
    # Generate target data
    target_data = pd.DataFrame(
        {f"x{d}": grid_d.ravel() for d, grid_d in enumerate(meshgrid)},
        columns=coord_columns,
    )
    
    if hasattr(target_function, 'forward'):
        # BoTorch function - use tensor input
        target_tensor = torch.tensor(target_data[coord_columns].values)
        target_targets = target_function(target_tensor).detach().numpy()
    else:
        # Custom function - apply row-wise  
        target_targets = target_data[coord_columns].apply(target_function, axis=1).values
        
    target_data["Target"] = target_targets
    target_data["Function"] = "Target_Function"
    
    return pd.concat([source_data, target_data], ignore_index=True)


def make_searchspace(
    data: pd.DataFrame,
    use_task_parameter: bool,
) -> SearchSpace:
    """Create searchspace for Hartmann benchmark.
    
    Args:
        data: The Hartmann dataset  
        use_task_parameter: Whether to include task parameter
        
    Returns:
        SearchSpace for the benchmark
    """
    # Extract coordinate columns (exclude Target and Function)
    coord_columns = [col for col in data.columns if col.startswith('x')]
    
    params: list[DiscreteParameter] = [
        NumericalDiscreteParameter(
            name=col,
            values=sorted(data[col].unique()),
        )
        for col in coord_columns
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
    """Create objective for Hartmann benchmark.
    
    Returns:
        SingleTargetObjective for Target minimization
    """
    return SingleTargetObjective(target=NumericalTarget(name="Target", mode="MIN"))


def make_lookup(data: pd.DataFrame) -> pd.DataFrame:
    """Create lookup table for target task."""
    return data[data["Function"] == "Target_Function"]


def make_initial_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create initial data for source task."""
    return data[data["Function"] == "Source_Function"]


def hartmann_tl_benchmark(
    settings: ConvergenceBenchmarkSettings,
    target_function_factory: Callable[[], Callable],
    source_function_factory: Callable[[], Callable], 
    bounds: np.ndarray,
    points_per_dim: int,
    percentages: list[float] = None,
) -> pd.DataFrame:
    """Configurable Hartmann transfer learning benchmark.

    Args:
        settings: Configuration settings for the convergence benchmark
        target_function_factory: Factory function that creates target function
        source_function_factory: Factory function that creates source function
        bounds: Array of shape (2, dim) specifying lower and upper bounds
        points_per_dim: Number of grid points per dimension
        percentages: Source data percentages to test (default: [0.01, 0.05, 0.1])

    Returns:
        DataFrame containing benchmark results
    """
    if percentages is None:
        percentages = [0.01, 0.05, 0.1]
        
    data = load_data(
        target_function_factory=target_function_factory,
        source_function_factory=source_function_factory,
        bounds=bounds,
        points_per_dim=points_per_dim,
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