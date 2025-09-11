#!/usr/bin/env python3
"""Store transfer learning model predictions for Forrester benchmarks.

This script runs campaigns for all Forrester convergence benchmarks and stores
the evolution of model predictions throughout DOE iterations. The predictions
are saved to JSON files for subsequent visualization.

Usage:
    python store_tl_predictions.py --n_mc_iterations 5 --n_doe_iterations 10
    python store_tl_predictions.py --test  # Quick test mode: 2 MC, 2 DOE
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add BayBE root directory to Python path
script_dir = Path(__file__).resolve().parent
baybe_root = script_dir.parent.parent.parent.parent
sys.path.insert(0, str(baybe_root))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.simulation.lookup import look_up_targets
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.source_prior import SourcePriorGaussianProcessSurrogate
from baybe.surrogates.transfergpbo import (
    MHGPGaussianProcessSurrogate,
    SHGPGaussianProcessSurrogate,
)
from baybe.utils.random import temporary_seed

from benchmarks.domains.transfer_learning.forrester.base import (
    AMPLIFIED_CONFIG,
    INVERTED_CONFIG,
    LOW_FIDELITY_CONFIG,
    TARGET_CONFIG,
    load_data,
    make_objective,
    make_searchspace,
)


def get_forrester_benchmark_configs() -> List[Tuple[str, Dict[str, float], Dict[str, float]]]:
    """Get all Forrester benchmark configurations.
    
    Returns:
        List of tuples: (benchmark_name, target_config, source_config)
    """
    configs = [
        # Noise-based benchmarks
        ("forrester_noise_05", TARGET_CONFIG, {**TARGET_CONFIG, "output_noise": 0.5}),
        ("forrester_noise_1", TARGET_CONFIG, {**TARGET_CONFIG, "output_noise": 1.0}),
        ("forrester_noise_2", TARGET_CONFIG, {**TARGET_CONFIG, "output_noise": 2.0}),
        
        # Amplified configuration benchmarks
        ("forrester_amplified_noise_05", TARGET_CONFIG, {**AMPLIFIED_CONFIG, "output_noise": 0.5}),
        ("forrester_amplified_noise_2", TARGET_CONFIG, {**AMPLIFIED_CONFIG, "output_noise": 2.0}),
        
        # Low-fidelity configuration benchmarks
        ("forrester_low_fid_noise_02", TARGET_CONFIG, {**LOW_FIDELITY_CONFIG, "output_noise": 0.2}),
        ("forrester_low_fid_noise_05", TARGET_CONFIG, {**LOW_FIDELITY_CONFIG, "output_noise": 0.5}),
        ("forrester_low_fid_noise_2", TARGET_CONFIG, {**LOW_FIDELITY_CONFIG, "output_noise": 2.0}),
        
        # Inverted configuration benchmark
        ("forrester_inverted_noise_05", TARGET_CONFIG, {**INVERTED_CONFIG, "output_noise": 0.5}),
        
        # Shifted configuration benchmarks
        ("forrester_neg_shift_015_noise_05", TARGET_CONFIG, {**TARGET_CONFIG, "x_shift": -0.15, "output_noise": 0.5}),
        ("forrester_neg_shift_01_noise_05", TARGET_CONFIG, {**TARGET_CONFIG, "x_shift": -0.1, "output_noise": 0.5}),
        ("forrester_pos_shift_015_noise_05", TARGET_CONFIG, {**TARGET_CONFIG, "x_shift": 0.15, "output_noise": 0.5}),
        ("forrester_pos_shift_01_noise_05", TARGET_CONFIG, {**TARGET_CONFIG, "x_shift": 0.1, "output_noise": 0.5}),
    ]
    return configs


def create_campaigns(searchspace, searchspace_nontl, objective) -> Dict[str, Campaign]:
    """Create all campaign types for prediction storage.
    
    Args:
        searchspace: Search space with task parameter for TL models
        searchspace_nontl: Search space without task parameter for baselines
        objective: Optimization objective
        
    Returns:
        Dictionary mapping model names to campaign instances
    """
    campaigns = {
        "index_kernel": Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=TwoPhaseMetaRecommender(
                initial_recommender=RandomRecommender(),
                recommender=BotorchRecommender(
                    surrogate_model=GaussianProcessSurrogate()
                ),
            ),
        ),
        "source_prior": Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=TwoPhaseMetaRecommender(
                initial_recommender=RandomRecommender(),
                recommender=BotorchRecommender(
                    surrogate_model=SourcePriorGaussianProcessSurrogate()
                ),
            ),
        ),
        "mhgp": Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=TwoPhaseMetaRecommender(
                initial_recommender=RandomRecommender(),
                recommender=BotorchRecommender(
                    surrogate_model=MHGPGaussianProcessSurrogate()
                ),
            ),
        ),
        "shgp": Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=TwoPhaseMetaRecommender(
                initial_recommender=RandomRecommender(),
                recommender=BotorchRecommender(
                    surrogate_model=SHGPGaussianProcessSurrogate()
                ),
            ),
        ),
        "0_reduced": Campaign(
            searchspace=searchspace_nontl,
            objective=objective,
            recommender=TwoPhaseMetaRecommender(
                initial_recommender=RandomRecommender(),
                recommender=BotorchRecommender(
                    surrogate_model=GaussianProcessSurrogate()
                ),
            ),
        ),
        "0_full": Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=TwoPhaseMetaRecommender(
                initial_recommender=RandomRecommender(),
                recommender=BotorchRecommender(
                    surrogate_model=GaussianProcessSurrogate()
                ),
            ),
        ),
    }
    return campaigns


def extract_predictions(surrogate, test_data: pd.DataFrame) -> Dict[str, List[float]]:
    """Extract model predictions on test grid.
    
    Args:
        surrogate: Fitted surrogate model
        test_data: Test data with x values and task information
        
    Returns:
        Dictionary with mean and std predictions
    """
    try:
        posterior = surrogate.posterior(test_data)
        mean_vals = posterior.mean.detach().numpy().flatten().astype(float)
        var_vals = posterior.variance.detach().numpy().flatten().astype(float)
        std_vals = np.sqrt(var_vals).astype(float)
        
        return {
            "mean": [float(x) for x in mean_vals],
            "std": [float(x) for x in std_vals]
        }
    except Exception as e:
        print(f"Warning: Failed to extract predictions: {e}")
        return {"mean": [0.0] * len(test_data), "std": [1.0] * len(test_data)}


def get_source_predictions(surrogate, test_data: pd.DataFrame, model_name: str) -> Dict[str, List[float]] | None:
    """Extract source model predictions for TL models.
    
    Args:
        surrogate: Fitted surrogate model
        test_data: Test data with x values
        model_name: Name of the model type
        
    Returns:
        Dictionary with source predictions or None if not applicable
    """
    try:
        if model_name in ["mhgp", "shgp"]:
            if hasattr(surrogate, '_model') and hasattr(surrogate._model, 'source_gps'):
                source_gps = surrogate._model.source_gps
                if source_gps and len(source_gps) > 0:
                    source_gp = source_gps[0]
                    x_tensor = torch.tensor(test_data["x"].values.reshape(-1, 1), dtype=torch.float32)
                    with torch.no_grad():
                        posterior = source_gp.posterior(x_tensor)
                        mean = posterior.mean.detach().numpy().flatten().astype(float)
                        std = posterior.variance.sqrt().detach().numpy().flatten().astype(float)
                    return {"mean": [float(x) for x in mean], "std": [float(x) for x in std]}
        
        elif model_name == "source_prior":
            if hasattr(surrogate, '_source_gp') and surrogate._source_gp is not None:
                source_gp = surrogate._source_gp
                x_tensor = torch.tensor(test_data["x"].values.reshape(-1, 1), dtype=torch.float32)
                with torch.no_grad():
                    posterior = source_gp.posterior(x_tensor)
                    mean = posterior.mean.detach().numpy().flatten().astype(float)
                    std = posterior.variance.sqrt().detach().numpy().flatten().astype(float)
                return {"mean": [float(x) for x in mean], "std": [float(x) for x in std]}
        
        elif model_name == "index_kernel":
            # For index kernel, we can extract source predictions by setting task to source
            source_test_data = test_data.copy()
            source_test_data["Function"] = "Source_Function"
            posterior = surrogate.posterior(source_test_data)
            mean = posterior.mean.detach().numpy().flatten().astype(float)
            std = posterior.variance.sqrt().detach().numpy().flatten().astype(float)
            return {"mean": [float(x) for x in mean], "std": [float(x) for x in std]}
        
    except Exception as e:
        print(f"Warning: Could not extract source predictions for {model_name}: {e}")
    
    return None


def run_benchmark_predictions(
    benchmark_name: str,
    target_config: Dict[str, float],
    source_config: Dict[str, float],
    n_mc_iterations: int,
    n_doe_iterations: int,
) -> Dict[str, Any]:
    """Run prediction storage for a single benchmark.
    
    Args:
        benchmark_name: Name of the benchmark
        target_config: Target function configuration
        source_config: Source function configuration  
        n_mc_iterations: Number of Monte Carlo iterations
        n_doe_iterations: Number of DOE iterations
        
    Returns:
        Complete predictions data structure
    """
    print(f"Running benchmark: {benchmark_name}")
    
    # Load data and create search spaces
    data = load_data(target_config=target_config, source_config=source_config)
    searchspace = make_searchspace(data=data, use_task_parameter=True)
    searchspace_nontl = make_searchspace(data=data, use_task_parameter=False)
    objective = make_objective()
    
    # Create lookup data for target task
    lookup_data = data[data["Function"] == "Target_Function"].copy()
    source_data_full = data[data["Function"] == "Source_Function"].copy()
    
    # Prediction grid on [0,1]
    x_grid = np.linspace(0, 1, 100)
    
    predictions_data = {
        "benchmark_name": benchmark_name,
        "source_config": source_config,
        "target_config": target_config,
        "n_mc_iterations": n_mc_iterations,
        "n_doe_iterations": n_doe_iterations,
        "prediction_grid": [float(x) for x in x_grid],
        "mc_iterations": []
    }
    
    # Run MC iterations
    for mc_iter in tqdm(range(n_mc_iterations), desc=f"{benchmark_name} MC iterations"):
        print(f"  MC iteration {mc_iter + 1}/{n_mc_iterations}")
        
        # Sample source data (0.08 fraction, consistent across models within MC iteration)
        with temporary_seed(1337 + mc_iter):
            source_data = source_data_full.sample(frac=0.08)
        
        mc_data = {
            "mc_iter": mc_iter,
            "source_data": {
                "x": [float(x) for x in source_data["x"].values],
                "Target": [float(x) for x in source_data["Target"].values]
            },
            "models": {}
        }
        
        # Create campaigns
        campaigns = create_campaigns(searchspace, searchspace_nontl, objective)
        
        # Run each model
        for model_name, campaign in campaigns.items():
            print(f"    Running model: {model_name}")
            
            # Reset campaign for this MC iteration
            campaign_copy = deepcopy(campaign)
            
            # Add source data (except for baseline models)
            if model_name not in ["0_reduced", "0_full"]:
                campaign_copy.add_measurements(source_data)
            
            # Prepare test data for predictions
            if model_name == "0_reduced":
                # No task parameter
                test_data = pd.DataFrame({"x": x_grid})
            else:
                # With task parameter set to target
                test_data = pd.DataFrame({
                    "x": x_grid,
                    "Function": ["Target_Function"] * len(x_grid)
                })
            
            model_data = {"doe_iterations": []}
            
            # Get initial predictions (source-only for TL models, empty for baselines)
            try:
                if model_name not in ["0_reduced", "0_full"] and len(campaign_copy.measurements) > 0:
                    # TL models: get source predictions on target domain
                    surrogate = campaign_copy.get_surrogate()
                    source_preds = get_source_predictions(surrogate, test_data, model_name)
                    model_data["source_predictions"] = source_preds
                else:
                    # Baseline models: no source predictions
                    model_data["source_predictions"] = None
            except Exception as e:
                print(f"      Warning: Could not get initial source predictions: {e}")
                model_data["source_predictions"] = None
            
            # Run DOE iterations
            observed_data = []
            for doe_iter in range(n_doe_iterations):
                print(f"      DOE iteration {doe_iter + 1}/{n_doe_iterations}")
                
                try:
                    # Recommend next point
                    recommended = campaign_copy.recommend(batch_size=1)
                    
                    # Look up target values
                    look_up_targets(recommended, campaign_copy.targets, lookup_data, "error")
                    
                    # Store observed data
                    obs_point = {
                        "x": float(recommended["x"].iloc[0]),
                        "Target": float(recommended["Target"].iloc[0])
                    }
                    observed_data.append(obs_point)
                    
                    # Add measurements to campaign
                    campaign_copy.add_measurements(recommended)
                    
                    # Extract predictions
                    surrogate = campaign_copy.get_surrogate()
                    predictions = extract_predictions(surrogate, test_data)
                    
                    iter_data = {
                        "iteration": doe_iter,
                        "predictions": predictions,
                        "observed_point": obs_point,
                        "observed_data_so_far": [dict(pt) for pt in observed_data]  # Deep copy to avoid reference issues
                    }
                    
                    model_data["doe_iterations"].append(iter_data)
                    
                except Exception as e:
                    print(f"      Error in DOE iteration {doe_iter}: {e}")
                    # Add empty iteration data
                    iter_data = {
                        "iteration": doe_iter,
                        "predictions": {"mean": [0.0] * len(x_grid), "std": [1.0] * len(x_grid)},
                        "observed_point": {"x": 0.5, "Target": 0.0},
                        "observed_data_so_far": [dict(pt) for pt in observed_data]  # Deep copy
                    }
                    model_data["doe_iterations"].append(iter_data)
            
            mc_data["models"][model_name] = model_data
        
        predictions_data["mc_iterations"].append(mc_data)
    
    return predictions_data


def main():
    """Main execution function."""
    # Get script directory for default output
    script_dir = Path(__file__).resolve().parent
    
    parser = argparse.ArgumentParser(description="Store TL model predictions for Forrester benchmarks")
    parser.add_argument("--n_mc_iterations", type=int, default=5, help="Number of MC iterations")
    parser.add_argument("--n_doe_iterations", type=int, default=10, help="Number of DOE iterations") 
    parser.add_argument("--test", action="store_true", help="Run in test mode (2 MC, 2 DOE)")
    parser.add_argument("--benchmark", type=str, help="Run only specific benchmark (by name)")
    parser.add_argument("--output_dir", type=str, default=str(script_dir), help="Output directory for JSON files")
    
    args = parser.parse_args()
    
    # Set parameters based on mode
    if args.test:
        n_mc_iterations = 2
        n_doe_iterations = 2
        print("🧪 Running in TEST mode: 2 MC iterations, 2 DOE iterations")
    else:
        n_mc_iterations = args.n_mc_iterations
        n_doe_iterations = args.n_doe_iterations
        print(f"🚀 Running in FULL mode: {n_mc_iterations} MC iterations, {n_doe_iterations} DOE iterations")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    # Get benchmark configurations
    all_configs = get_forrester_benchmark_configs()
    
    # Filter benchmarks if specified
    if args.benchmark:
        all_configs = [(name, tc, sc) for name, tc, sc in all_configs if name == args.benchmark]
        if not all_configs:
            print(f"❌ Benchmark '{args.benchmark}' not found!")
            return
    
    print(f"📊 Will run {len(all_configs)} benchmarks")
    
    # Run benchmarks
    for benchmark_name, target_config, source_config in all_configs:
        try:
            predictions_data = run_benchmark_predictions(
                benchmark_name=benchmark_name,
                target_config=target_config,
                source_config=source_config,
                n_mc_iterations=n_mc_iterations,
                n_doe_iterations=n_doe_iterations,
            )
            
            # Save to JSON file
            output_file = os.path.join(args.output_dir, f"{benchmark_name}_predictions.json")
            with open(output_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            
            print(f"✅ Saved predictions for {benchmark_name} to {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to run benchmark {benchmark_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("🎉 All benchmarks completed!")


if __name__ == "__main__":
    main()