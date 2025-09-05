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
    if "_reduced_searchspace" in scenario_name:
        fraction = scenario_name.split("_")[0]
        return f"GP {fraction}% (reduced searchspace)"
    elif "_full_searchspace" in scenario_name:
        fraction = scenario_name.split("_")[0]
        return f"GP {fraction}% (full searchspace)"
    elif "_index_kernel" in scenario_name:
        fraction = scenario_name.split("_")[0]
        return f"GPIndex {fraction}%"
    elif "_source_prior" in scenario_name:
        fraction = scenario_name.split("_")[0]
        return f"SourcePrior {fraction}%"
    elif "_naive" in scenario_name:
        fraction = scenario_name.split("_")[0]
        return f"NaiveGP {fraction}%"
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
    baseline_scenarios = [s for s in scenarios if "_searchspace" in s]
    tl_scenarios = [s for s in scenarios if "_searchspace" not in s]
    
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
    
    # Create figure: n_source_fractions rows × n_metrics columns (switched from original)
    n_rows = len(source_fractions)
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
        f"TL Regression by Source Fraction: {metadata['name']}", 
        fontsize=14, fontweight="bold"
    )
    
    # Define extensible color palettes (matches per-model script)
    # Bright, distinct colors for TL models (easily extensible)
    tl_model_palette = [
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
    baseline_reduced_color = '#95a5a6'   # Light gray for reduced searchspace
    baseline_full_color = '#7f8c8d'      # Darker gray for full searchspace
    
    # Dynamically detect TL model types and assign colors
    tl_model_types = set()
    for scenario in tl_scenarios:
        # Extract model type from scenario name
        if '_index_kernel' in scenario:
            tl_model_types.add('index_kernel')
        elif '_mhgp' in scenario:
            tl_model_types.add('mhgp')
        elif '_shgp' in scenario:
            tl_model_types.add('shgp')
        elif '_source_prior' in scenario:
            tl_model_types.add('source_prior')
        elif '_naive' in scenario:
            tl_model_types.add('naive')
        # Add more patterns as new models are introduced
    
    # Assign colors to detected TL model types
    model_colors = {
        'reduced_searchspace': baseline_reduced_color,
        'full_searchspace': baseline_full_color,
    }
    
    for i, model_type in enumerate(sorted(tl_model_types)):
        color_idx = i % len(tl_model_palette)
        model_colors[model_type] = tl_model_palette[color_idx]
    
    def get_model_color_and_style(scenario_name):
        """Get consistent color and line style for a scenario."""
        if 'reduced_searchspace' in scenario_name:
            return model_colors['reduced_searchspace'], ':'
        elif 'full_searchspace' in scenario_name:
            return model_colors['full_searchspace'], '--'
        else:
            # Dynamically detect TL model type from scenario name
            for model_type in tl_model_types:
                if f'_{model_type}' in scenario_name:
                    return model_colors[model_type], '-'
            # Fallback for unknown model types
            return '#cccccc', '-'
    
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
    
    # Plot each source × metric combination (switched from original)
    for row_idx, source_fraction in enumerate(source_fractions):
        for col_idx, metric in enumerate(metrics):
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
                    
                    color, linestyle = get_model_color_and_style(baseline_scenario)
                    
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
                    
                    color, linestyle = get_model_color_and_style(tl_scenario)
                    
                    ax.plot(
                        tl_stats["n_train_pts"],
                        tl_stats["mean"],
                        color=color,
                        linestyle=linestyle,
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
                higher_better = metrics_higher_is_better.get(metric, True)
                direction = "higher better" if higher_better else "lower better"
                ax.set_title(f"{metric} ({direction})", fontsize=11)
            
            if row_idx == n_rows - 1:  # X-label only on bottom row
                ax.set_xlabel("Number of Target Training Points", fontsize=10)
            
            if col_idx == 0:  # Y-label only on leftmost column
                ax.set_ylabel(f"{int(source_fraction * 100)}% Source Data", fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            
            # No individual legends - we'll create a single figure legend later
            
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
    
    # Create single legend with all model types
    legend_elements = []
    
    # Add transfer learning models (solid lines) - dynamically detected
    model_name_mapping = {
        'index_kernel': 'GPIndex',
        'mhgp': 'MHGP', 
        'shgp': 'SHGP',
        'source_prior': 'SourcePrior',
        'naive': 'NaiveGP'
        # Add more mappings as new models are introduced
    }
    
    for model_type in sorted(tl_model_types):
        if model_type in model_colors:
            color = model_colors[model_type]
            clean_name = model_name_mapping.get(model_type, model_type.replace('_', ' ').title())
            legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='-', 
                                            linewidth=2, label=clean_name))
    
    # Add baseline models
    if any('reduced_searchspace' in scenario for scenario in baseline_scenarios):
        legend_elements.append(plt.Line2D([0], [0], color=model_colors['reduced_searchspace'], 
                                        linestyle=':', linewidth=2, 
                                        label='GP 0% (reduced)'))
    
    if any('full_searchspace' in scenario for scenario in baseline_scenarios):
        legend_elements.append(plt.Line2D([0], [0], color=model_colors['full_searchspace'], 
                                        linestyle='--', linewidth=2, 
                                        label='GP 0% (full)'))
    
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