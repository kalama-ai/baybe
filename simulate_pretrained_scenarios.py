"""Specialized simulate_scenarios for pre-trained models.

This module provides Option 2: a specialized simulate_scenarios variant that handles 
pre-trained models. It maintains the exact same interface as simulate_scenarios but
pre-trains models using initial_data before running optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal
from copy import deepcopy

import pandas as pd

from baybe.campaign import Campaign
from baybe.simulation import simulate_scenarios
from baybe.surrogates.source_prior import SourcePriorGaussianProcessSurrogate
from baybe.surrogates.botorchwrapper import PretrainedSingleTaskGPSurrogate
from baybe.parameters import TaskParameter
from baybe.recommenders import BotorchRecommender, TwoPhaseMetaRecommender, RandomRecommender
from baybe.searchspace import SearchSpace

if TYPE_CHECKING:
    from baybe.objectives import Objective
    from baybe.searchspace import SearchSpace


def simulate_pretrained_scenarios(
    scenarios: dict[Any, Campaign],
    lookup: pd.DataFrame | Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    /,
    *,
    # Pre-training configuration
    pretrain_model_factory: Callable[[], Any],
    pretrain_searchspace: SearchSpace,
    pretrain_objective: Objective,
    # Standard simulate_scenarios parameters
    batch_size: int = 1,
    n_doe_iterations: int | None = None,
    initial_data: list[pd.DataFrame] | None = None,
    groupby: list[str] | None = None,
    n_mc_iterations: int = 1,
    random_seed: int | None = None,
    impute_mode: Literal["error", "worst", "best", "mean", "random", "ignore"] = "error",
    noise_percent: float | None = None,
    parallel_runs: bool | None = None,
) -> pd.DataFrame:
    """Simulate multiple scenarios with pre-trained models.
    
    This function provides the same interface as simulate_scenarios but pre-trains
    models using initial_data before running optimization. Each MC iteration gets
    its own pre-trained model.
    
    Workflow per scenario and MC iteration:
    1. Create fresh model from pretrain_model_factory
    2. Pre-train model using initial_data[mc_iteration] 
    3. Create new campaign with pre-trained model
    4. Run single MC iteration with simulate_scenarios
    5. Combine all MC results into unified DataFrame
    
    Args:
        scenarios: Dictionary mapping scenario names to campaign templates.
                  The surrogate models in these campaigns will be replaced with
                  pre-trained models.
        lookup: Same as simulate_scenarios.
        pretrain_model_factory: Function that creates fresh model instances for pre-training.
                               E.g., lambda: SourcePriorGaussianProcessSurrogate()
        pretrain_searchspace: SearchSpace for pre-training (typically includes task parameter).
        pretrain_objective: Objective for pre-training.
        batch_size: Same as simulate_scenarios.
        n_doe_iterations: Same as simulate_scenarios.
        initial_data: List of DataFrames for pre-training, one per MC iteration.
        groupby: Same as simulate_scenarios.
        n_mc_iterations: Same as simulate_scenarios.
        random_seed: Same as simulate_scenarios.
        impute_mode: Same as simulate_scenarios.
        noise_percent: Same as simulate_scenarios.
        parallel_runs: Same as simulate_scenarios.
        
    Returns:
        DataFrame with same structure as simulate_scenarios, with proper MC iteration indexing.
        
    Example:
        >>> # Pre-train SourcePrior, then optimize without task parameter
        >>> pretrain_factory = lambda: SourcePriorGaussianProcessSurrogate()
        >>> 
        >>> scenarios = {
        >>>     "10_source_prior_wrapped": Campaign(
        >>>         searchspace=searchspace_no_task,  # No task parameter
        >>>         objective=objective,
        >>>         recommender=BotorchRecommender(surrogate_model=None)  # Will be replaced
        >>>     )
        >>> }
        >>> 
        >>> results = simulate_pretrained_scenarios(
        >>>     scenarios=scenarios,
        >>>     lookup=lookup,
        >>>     pretrain_model_factory=pretrain_factory,
        >>>     pretrain_searchspace=searchspace_with_task,  # With task parameter
        >>>     pretrain_objective=objective,
        >>>     initial_data=initial_data_samples[0.1],
        >>>     n_mc_iterations=55
        >>> )
    """
    if initial_data is None:
        raise ValueError("initial_data is required for pre-trained scenarios")
    
    # Collect results from all scenarios and MC iterations
    all_results = []
    
    for scenario_name, template_campaign in scenarios.items():
        print(f"Processing pre-trained scenario: {scenario_name}")
        
        # Collect MC results for this scenario
        scenario_results = []
        
        for mc_iteration in range(n_mc_iterations):
            # Get training data for this MC iteration
            if mc_iteration >= len(initial_data):
                raise ValueError(f"MC iteration {mc_iteration} exceeds initial_data length {len(initial_data)}")
            
            training_data = initial_data[mc_iteration]
            
            # Create and pre-train model (like temperature_tl_old.py)
            source_prior_surrogate = pretrain_model_factory()
            source_prior_surrogate.fit(
                searchspace=pretrain_searchspace,
                objective=pretrain_objective,
                measurements=training_data
            )
            
            # Extract the target GP and wrap it (like temperature_tl_old.py)
            target_gp = source_prior_surrogate._target_gp
            wrapped_surrogate = PretrainedSingleTaskGPSurrogate.from_botorch_model(target_gp)
            
            # Create reduced searchspace without task parameter
            reduced_searchspace = _create_reduced_searchspace(pretrain_searchspace)
            
            # Create new campaign with wrapped model (like temperature_tl_old.py)
            wrapped_campaign = Campaign(
                searchspace=reduced_searchspace,
                objective=pretrain_objective,
                recommender=_create_wrapped_recommender(template_campaign.recommender, wrapped_surrogate)
            )
            
            # Run single MC iteration 
            mc_seed = None if random_seed is None else random_seed + mc_iteration
            
            single_result = simulate_scenarios(
                {scenario_name: wrapped_campaign},
                lookup,
                batch_size=batch_size,
                n_doe_iterations=n_doe_iterations,
                # No initial_data - model is pre-trained
                groupby=groupby,
                n_mc_iterations=1,  # Single iteration
                random_seed=mc_seed,
                impute_mode=impute_mode,
                noise_percent=noise_percent,
                parallel_runs=parallel_runs,
            )
            
            # Fix Monte_Carlo_Run indexing 
            single_result['Monte_Carlo_Run'] = mc_iteration
            scenario_results.append(single_result)
        
        # Combine all MC iterations for this scenario
        if scenario_results:
            combined_scenario_result = pd.concat(scenario_results, ignore_index=True)
            all_results.append(combined_scenario_result)
    
    # Combine all scenarios
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        # Return empty DataFrame with expected structure
        return pd.DataFrame()


def _create_reduced_searchspace(full_searchspace: SearchSpace) -> SearchSpace:
    """Create reduced searchspace by removing TaskParameter instances."""
    # Get all parameters from the search space
    parameters = list(full_searchspace.parameters)
    
    # Filter out TaskParameter instances
    filtered_parameters = [
        param for param in parameters if not isinstance(param, TaskParameter)
    ]
    
    # If no TaskParameter was found, return the original searchspace
    if len(filtered_parameters) == len(parameters):
        return full_searchspace
    
    # Create new SearchSpace with filtered parameters and constraints
    return SearchSpace.from_product(
        parameters=filtered_parameters,
        constraints=list(full_searchspace.constraints),
    )


def _create_wrapped_recommender(template_recommender, wrapped_surrogate):
    """Create new recommender with wrapped surrogate, preserving structure."""
    # Handle TwoPhaseMetaRecommender (like in temperature_tl_old.py)
    if hasattr(template_recommender, 'initial_recommender'):
        return TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(
                surrogate_model=wrapped_surrogate
            ),
        )
    # Handle direct BotorchRecommender
    elif hasattr(template_recommender, '_surrogate_model'):
        return BotorchRecommender(surrogate_model=wrapped_surrogate)
    else:
        # Fallback: create TwoPhase structure
        return TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(
                surrogate_model=wrapped_surrogate
            ),
        )
