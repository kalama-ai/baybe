"""Script to visualize TL regression benchmark results organized by model type."""

import base64
import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from benchmarks.visualization.scenario_parsing import parse_scenario


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
    source_pct, model_name, is_baseline = parse_scenario(scenario_name)
    if is_baseline:
        return model_name  # Use original baseline names like "0", "0_naive"
    else:
        return f"{model_name} {source_pct}%"  # e.g., "source_prior 10%"


def get_model_type_from_scenario(scenario_name):
    """Extract model type from scenario name for grouping."""
    source_pct, model_name, is_baseline = parse_scenario(scenario_name)
    return model_name  # Use simplified model names directly


def visualize_tl_regression_per_model(json_file_path):
    """Create visualization organized by model type."""
    # Load data
    df, metadata = load_benchmark_data(json_file_path)
    
    print(f"Loaded regression data with {len(df)} rows and {len(df.columns)} columns")
    
    # Extract scenarios and metrics
    scenarios, metrics = extract_scenarios_and_metrics(df)
    print(f"Found scenarios: {scenarios}")
    print(f"Found metrics: {metrics}")
    
    # Group scenarios by model type
    model_groups = {}
    for scenario in scenarios:
        model_type = get_model_type_from_scenario(scenario)
        if model_type not in model_groups:
            model_groups[model_type] = []
        model_groups[model_type].append(scenario)
    
    # Order model types for consistent display
    model_type_order = ["0", "0_naive", "index_kernel", "source_prior", "mhgp", "shgp", "naive"]
    ordered_model_types = [mt for mt in model_type_order if mt in model_groups]
    
    # Add any unknown models not in the preferred order
    remaining_models = set(model_groups.keys()) - set(ordered_model_types)
    ordered_model_types.extend(sorted(remaining_models))
    
    # Get source fractions
    source_fractions = sorted(df["fraction_source"].unique())
    print(f"Source fractions: {source_fractions}")
    print(f"Model groups: {model_groups}")
    
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
    
    # Create figure: n_model_types rows × n_metrics columns (switched from original)
    n_rows = len(ordered_model_types)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    
    # Handle single row/column cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(
        f"TL Regression by Model Type: {metadata['name']}", 
        fontsize=14, fontweight="bold"
    )
    
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
    
    # Define extensible color palettes
    # Bright, distinct colors for source fractions (easily extensible)
    source_fraction_palette = [
        '#3498db',  # Bright blue
        '#e74c3c',  # Bright red
        '#2ecc71',  # Bright green
        '#f39c12',  # Bright orange
        '#9b59b6',  # Bright purple
        '#34495e',  # Dark blue-gray
        '#e67e22',  # Darker orange
        '#1abc9c',  # Teal
        '#8e44ad',  # Dark purple
        '#2c3e50',  # Very dark blue
        '#16a085',  # Dark teal
        '#c0392b',  # Dark red
        '#27ae60',  # Dark green
        '#d35400',  # Dark orange
        '#7f8c8d',  # Medium gray
    ]
    
    # Baseline colors (fixed, muted grays)
    baseline_color = '#95a5a6'        # Light gray for reduced searchspace
    baseline_full_color = '#7f8c8d'   # Darker gray for full searchspace
    
    # Dynamically assign colors to detected source fractions
    detected_fractions = sorted(source_fractions)
    source_fraction_colors = {}
    for i, fraction in enumerate(detected_fractions):
        color_idx = i % len(source_fraction_palette)
        source_fraction_colors[fraction] = source_fraction_palette[color_idx]
    
    def get_source_fraction_from_scenario(scenario_name):
        """Extract source fraction from scenario name."""
        try:
            if 'searchspace' in scenario_name:
                return None  # Baseline scenarios
            # Extract percentage and convert to fraction
            percentage = int(scenario_name.split('_')[0])
            return percentage / 100.0
        except (ValueError, IndexError):
            return None
    
    def get_color_and_style_for_scenario(scenario_name):
        """Get color and line style for a scenario."""
        if 'reduced_searchspace' in scenario_name:
            return baseline_color, ':'
        elif 'full_searchspace' in scenario_name:
            return baseline_full_color, '--'
        else:
            fraction = get_source_fraction_from_scenario(scenario_name)
            color = source_fraction_colors.get(fraction, '#999999')
            return color, '-'
    
    # Plot each model_type × metric combination (switched from original)
    for row_idx, model_type in enumerate(ordered_model_types):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            # Get metric column name for new format
            metric_col = metric_column_mapping.get(metric)
            if not metric_col or metric_col not in df.columns:
                continue
                
            # Get scenarios for this model type
            model_scenarios = model_groups[model_type]
            
            # Plot all scenarios for this model type
            for scenario in model_scenarios:
                scenario_data = df[df["scenario"] == scenario]
                if len(scenario_data) > 0:
                    scenario_stats = (
                        scenario_data.groupby("n_train_pts")[metric_col]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    scenario_stats["std"] = scenario_stats["std"].fillna(0)
                    
                    color, linestyle = get_color_and_style_for_scenario(scenario)
                    
                    ax.plot(
                        scenario_stats["n_train_pts"],
                        scenario_stats["mean"],
                        color=color,
                        linestyle=linestyle,
                        linewidth=2,
                        alpha=0.8,
                        label=clean_scenario_name(scenario)
                    )
                    ax.fill_between(
                        scenario_stats["n_train_pts"],
                        scenario_stats["mean"] - scenario_stats["std"],
                        scenario_stats["mean"] + scenario_stats["std"],
                        color=color,
                        alpha=0.15,
                    )
            
            # Customize subplot
            if row_idx == 0:  # Column title only on top row
                higher_better = metrics_higher_is_better.get(metric, True)
                direction = "higher better" if higher_better else "lower better"
                ax.set_title(f"{metric} ({direction})", fontsize=11)
            
            if row_idx == n_rows - 1:  # X-label only on bottom row
                ax.set_xlabel("Number of Target Training Points", fontsize=10)
            
            if col_idx == 0:  # Y-label only on leftmost column
                ax.set_ylabel(model_type.replace("_", " "), fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            
            # Add baseline models to all plots
            # Reduced searchspace baseline
            baseline_reduced = "0_reduced_searchspace"
            if baseline_reduced in scenarios:
                baseline_data = df[df["scenario"] == baseline_reduced]
                if len(baseline_data) > 0:
                    baseline_stats = (
                        baseline_data.groupby("n_train_pts")[metric_col]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    baseline_stats["std"] = baseline_stats["std"].fillna(0)
                    
                    ax.plot(
                        baseline_stats["n_train_pts"],
                        baseline_stats["mean"],
                        color=baseline_color,
                        linestyle=":",
                        linewidth=2,
                        alpha=0.8,
                        label="0% (reduced)"
                    )
                    ax.fill_between(
                        baseline_stats["n_train_pts"],
                        baseline_stats["mean"] - baseline_stats["std"],
                        baseline_stats["mean"] + baseline_stats["std"],
                        color=baseline_color,
                        alpha=0.15,
                    )
            
            # Full searchspace baseline
            baseline_full = "0_full_searchspace"
            if baseline_full in scenarios:
                baseline_full_data = df[df["scenario"] == baseline_full]
                if len(baseline_full_data) > 0:
                    baseline_full_stats = (
                        baseline_full_data.groupby("n_train_pts")[metric_col]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    baseline_full_stats["std"] = baseline_full_stats["std"].fillna(0)
                    
                    ax.plot(
                        baseline_full_stats["n_train_pts"],
                        baseline_full_stats["mean"],
                        color=baseline_full_color,
                        linestyle="--",
                        linewidth=2,
                        alpha=0.8,
                        label="0% (full)"
                    )
                    ax.fill_between(
                        baseline_full_stats["n_train_pts"],
                        baseline_full_stats["mean"] - baseline_full_stats["std"],
                        baseline_full_stats["mean"] + baseline_full_stats["std"],
                        color=baseline_full_color,
                        alpha=0.15,
                    )
            
            # Set y-axis limits based on mean values only (not std bands)
            all_means = []
            
            # Collect means from all scenarios for this model type
            for scenario in model_scenarios:
                scenario_data = df[df["scenario"] == scenario]
                if len(scenario_data) > 0:
                    scenario_means = scenario_data.groupby("n_train_pts")[metric_col].mean()
                    # Filter out NaN and infinite values
                    valid_means = [val for val in scenario_means.values if np.isfinite(val)]
                    all_means.extend(valid_means)
            
            # Also include both baseline models in y-limits calculation
            for baseline_scenario in ["0_reduced_searchspace", "0_full_searchspace"]:
                if baseline_scenario in scenarios:
                    baseline_data = df[df["scenario"] == baseline_scenario]
                    if len(baseline_data) > 0 and metric_col in baseline_data.columns:
                        baseline_means = baseline_data.groupby("n_train_pts")[metric_col].mean()
                        valid_baseline_means = [val for val in baseline_means.values if np.isfinite(val)]
                        all_means.extend(valid_baseline_means)
            
            # Set y-axis range based on means with some padding
            if all_means:
                mean_min, mean_max = min(all_means), max(all_means)
                mean_range = mean_max - mean_min
                padding = mean_range * 0.1 if mean_range > 0 else abs(mean_max) * 0.1
                
                # Only set limits if they are finite
                if np.isfinite(mean_min - padding) and np.isfinite(mean_max + padding):
                    ax.set_ylim(mean_min - padding, mean_max + padding)
    
    # Create legend mapping colors to source fractions and baseline
    legend_elements = []
    
    # Add source fractions (solid lines)
    for fraction in sorted(source_fractions):
        if fraction in source_fraction_colors:
            color = source_fraction_colors[fraction]
            percentage = int(fraction * 100)
            legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='-', 
                                            linewidth=2, label=f'{percentage}%'))
    
    # Add baseline models
    if "0_reduced_searchspace" in scenarios:
        legend_elements.append(plt.Line2D([0], [0], color=baseline_color, linestyle=':', 
                                        linewidth=2, label='0% (reduced)'))
    
    if "0_full_searchspace" in scenarios:
        legend_elements.append(plt.Line2D([0], [0], color=baseline_full_color, linestyle='--', 
                                        linewidth=2, label='0% (full)'))
    
    # Add the legend to the first subplot (top-left) in lower right corner
    first_ax = axes[0, 0]
    first_ax.legend(handles=legend_elements, loc='lower right', fontsize=8, 
                   frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout and fix title positioning
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.2)
    
    # Save the plot
    input_path = Path(json_file_path)
    output_path = input_path.parent / (
        input_path.stem.replace("_result", "_tl_regression_per_model") + ".png"
    )
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"TL regression per model visualization saved to: {output_path}")
    
    # Show basic statistics
    print("\nDataset statistics:")
    print(f"- Total rows: {len(df)}")
    print(f"- Monte Carlo runs: {df['mc_iter'].nunique()}")
    print(f"- Training points range: {df['n_train_pts'].min()} to {df['n_train_pts'].max()}")
    print(f"- Source fractions: {source_fractions}")
    print(f"- Scenarios: {scenarios}")
    
    plt.close()
    return output_path