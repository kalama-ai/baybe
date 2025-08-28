"""Script to visualize TL regression benchmark results organized by source fraction."""

import base64
import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_benchmark_data(json_file_path):
    """Load and decode the data field from a benchmark result JSON file."""
    with open(json_file_path) as f:
        result = json.load(f)

    # Decode the base64 encoded pickled DataFrame
    data_str = result["data"]
    pickled_df = base64.b64decode(data_str.encode("utf-8"))
    df = pickle.loads(pickled_df)

    return df, result


def extract_scenarios_and_metrics(df):
    """Extract scenario names and metrics from DataFrame columns."""
    # Define metadata columns
    metadata_cols = ['scenario', 'mc_iter', 'n_train_pts', 'fraction_source', 'n_source_pts', 'n_test_pts', 'source_data_seed']
    metric_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Map new metric column names to old display names
    metric_mapping = {
        'root_mean_squared_error': 'RMSE',
        'mean_squared_error': 'MSE', 
        'r2_score': 'R2',
        'mean_absolute_error': 'MAE',
        'max_error': 'MAX_ERROR',
        'explained_variance_score': 'EXPLAINED_VARIANCE',
        'kendall_tau_score': 'KENDALL_TAU',
        'spearman_rho_score': 'SPEARMAN_RHO'
    }
    
    # Get unique scenarios and available metrics
    scenarios = sorted(df['scenario'].unique()) if 'scenario' in df.columns else []
    metrics = [metric_mapping[col] for col in metric_cols if col in metric_mapping]
    
    return scenarios, sorted(metrics)


def clean_scenario_name(scenario_name):
    """Convert internal scenario names to clean display names."""
    if scenario_name == "0_reduced_searchspace":
        return "GP 0% (reduced searchspace)"
    elif scenario_name == "0_full_searchspace":
        return "GP 0% (full searchspace)"  
    elif scenario_name == "1_index_kernel":
        return "GPIndex 1%"
    elif scenario_name == "5_index_kernel":
        return "GPIndex 5%"
    else:
        return scenario_name


def visualize_tl_regression_per_source(json_file_path):
    """Create visualization organized by source fraction."""
    # Load data
    df, metadata = load_benchmark_data(json_file_path)
    
    print(f"Loaded regression data with {len(df)} rows and {len(df.columns)} columns")
    
    # Extract scenarios and metrics
    scenarios, metrics = extract_scenarios_and_metrics(df)
    print(f"Found scenarios: {scenarios}")
    print(f"Found metrics: {metrics}")
    
    # Separate baseline scenarios from TL scenarios
    baseline_scenarios = [s for s in scenarios if s.startswith("0_")]
    tl_scenarios = [s for s in scenarios if not s.startswith("0_")]
    
    # Get source fractions
    source_fractions = sorted(df["fraction_source"].unique())
    print(f"Source fractions: {source_fractions}")
    print(f"Baseline scenarios: {baseline_scenarios}")
    print(f"TL scenarios: {tl_scenarios}")
    
    # Define metric direction (higher is better or lower is better)
    metrics_higher_is_better = {
        "R2": True,
        "EXPLAINED_VARIANCE": True,
        "KENDALL_TAU": True,
        "SPEARMAN_RHO": True,
        "RMSE": False,
        "MSE": False,
        "MAE": False,
        "MAX_ERROR": False,
    }
    
    # Set up the plot style
    plt.style.use("default")
    sns.set_palette("tab10")
    
    # Create figure: n_metrics rows × n_source_fractions columns
    n_rows = len(metrics)
    n_cols = len(source_fractions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey='row')
    
    # Handle single row/column cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(
        f"TL Regression by Source Fraction: {metadata['name']}", 
        fontsize=14, fontweight="bold"
    )
    
    # Define colors for scenarios
    scenario_colors = {
        "0_reduced_searchspace": "#8c564b",
        "0_full_searchspace": "#9467bd",
        "1_index_kernel": "#1f77b4",
        "5_index_kernel": "#ff7f0e"
    }
    
    # Map new column names to old column names for metrics
    metric_column_mapping = {
        'RMSE': 'root_mean_squared_error',
        'MSE': 'mean_squared_error',
        'R2': 'r2_score', 
        'MAE': 'mean_absolute_error',
        'MAX_ERROR': 'max_error',
        'EXPLAINED_VARIANCE': 'explained_variance_score',
        'KENDALL_TAU': 'kendall_tau_score',
        'SPEARMAN_RHO': 'spearman_rho_score'
    }
    
    # Plot each metric × source combination
    for row_idx, metric in enumerate(metrics):
        for col_idx, source_fraction in enumerate(source_fractions):
            ax = axes[row_idx, col_idx]
            
            # Get metric column name for new format
            metric_col = metric_column_mapping.get(metric)
            if not metric_col or metric_col not in df.columns:
                continue
                
            # Filter data for this source fraction
            source_data = df[df["fraction_source"] == source_fraction]
            
            # Plot baseline scenarios
            for baseline_scenario in baseline_scenarios:
                baseline_data = source_data[source_data["scenario"] == baseline_scenario]
                if len(baseline_data) > 0:
                    baseline_stats = (
                        baseline_data.groupby("n_train_pts")[metric_col]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    baseline_stats["std"] = baseline_stats["std"].fillna(0)
                    
                    color = scenario_colors.get(baseline_scenario, "#cccccc")
                    linestyle = ":" if "reduced" in baseline_scenario else "--"
                    
                    ax.plot(
                        baseline_stats["n_train_pts"],
                        baseline_stats["mean"],
                        color=color,
                        linestyle=linestyle,
                        linewidth=2,
                        alpha=0.8,
                        label=clean_scenario_name(baseline_scenario)
                    )
                    ax.fill_between(
                        baseline_stats["n_train_pts"],
                        baseline_stats["mean"] - baseline_stats["std"],
                        baseline_stats["mean"] + baseline_stats["std"],
                        color=color,
                        alpha=0.15,
                    )
            
            # Plot TL scenarios for this source fraction
            for tl_scenario in tl_scenarios:
                tl_data = source_data[source_data["scenario"] == tl_scenario]
                if len(tl_data) > 0:
                    tl_stats = (
                        tl_data.groupby("n_train_pts")[metric_col]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    tl_stats["std"] = tl_stats["std"].fillna(0)
                    
                    color = scenario_colors.get(tl_scenario, "#cccccc")
                    
                    ax.plot(
                        tl_stats["n_train_pts"],
                        tl_stats["mean"],
                        color=color,
                        linestyle="-",
                        linewidth=2,
                        alpha=1.0,
                        label=clean_scenario_name(tl_scenario)
                    )
                    ax.fill_between(
                        tl_stats["n_train_pts"],
                        tl_stats["mean"] - tl_stats["std"],
                        tl_stats["mean"] + tl_stats["std"],
                        color=color,
                        alpha=0.15,
                    )
            
            # Customize subplot
            if row_idx == 0:  # Column title only on top row
                ax.set_title(f"{int(source_fraction * 100)}% Source Data", fontsize=11)
            
            if row_idx == n_rows - 1:  # X-label only on bottom row
                ax.set_xlabel("Number of Target Training Points", fontsize=10)
            
            if col_idx == 0:  # Y-label only on leftmost column
                higher_better = metrics_higher_is_better.get(metric, True)
                direction = "higher better" if higher_better else "lower better"
                ax.set_ylabel(f"{metric} ({direction})", fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            
            # Add legend only to first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best", fontsize=8)
            
            # Set y-axis limits based on mean values only (not std bands)
            all_means = []
            
            if metric_col and metric_col in df.columns:
                # Collect means from all scenarios for this source fraction
                for scenario in scenarios:
                    scenario_data = source_data[source_data["scenario"] == scenario]
                    if len(scenario_data) > 0:
                        scenario_means = scenario_data.groupby("n_train_pts")[metric_col].mean()
                        # Filter out NaN and infinite values
                        valid_means = [val for val in scenario_means.values if np.isfinite(val)]
                        all_means.extend(valid_means)
                
                # Set y-axis range based on means with some padding
                if all_means:
                    mean_min, mean_max = min(all_means), max(all_means)
                    mean_range = mean_max - mean_min
                    padding = mean_range * 0.1 if mean_range > 0 else abs(mean_max) * 0.1
                    
                    # Only set limits if they are finite
                    if np.isfinite(mean_min - padding) and np.isfinite(mean_max + padding):
                        ax.set_ylim(mean_min - padding, mean_max + padding)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.2)
    
    # Save the plot
    input_path = Path(json_file_path)
    output_path = input_path.parent / (
        input_path.stem.replace("_result", "_tl_regression_per_source") + ".png"
    )
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"TL regression per source visualization saved to: {output_path}")
    
    # Show basic statistics
    print("\nDataset statistics:")
    print(f"- Total rows: {len(df)}")
    print(f"- Monte Carlo runs: {df['mc_iter'].nunique()}")
    print(f"- Training points range: {df['n_train_pts'].min()} to {df['n_train_pts'].max()}")
    print(f"- Source fractions: {source_fractions}")
    
    plt.close()
    return output_path