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
            
            # Create and pre-train model
            pretrained_model = pretrain_model_factory()
            pretrained_model.fit(
                searchspace=pretrain_searchspace,
                objective=pretrain_objective,
                measurements=training_data
            )
            
            # Create campaign copy with pre-trained model
            campaign_copy = deepcopy(template_campaign)
            
            # Replace the surrogate model in the recommender
            # Navigate through the recommender structure to find and replace the surrogate
            _replace_surrogate_in_recommender(campaign_copy.recommender, pretrained_model)
            
            # Run single MC iteration 
            mc_seed = None if random_seed is None else random_seed + mc_iteration
            
            single_result = simulate_scenarios(
                scenarios={scenario_name: campaign_copy},
                lookup=lookup,
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


def _replace_surrogate_in_recommender(recommender, new_surrogate):
    """Helper function to replace surrogate model in recommender hierarchy.
    
    This navigates through BotorchRecommender, TwoPhaseMetaRecommender, etc.
    to find and replace the surrogate_model attribute.
    """
    # Handle TwoPhaseMetaRecommender
    if hasattr(recommender, 'recommender'):
        _replace_surrogate_in_recommender(recommender.recommender, new_surrogate)
    
    # Handle BotorchRecommender  
    if hasattr(recommender, 'surrogate_model'):
        recommender.surrogate_model = new_surrogate
        
    # Handle other recommender types that might have surrogate models
    if hasattr(recommender, 'initial_recommender'):
        _replace_surrogate_in_recommender(recommender.initial_recommender, new_surrogate)