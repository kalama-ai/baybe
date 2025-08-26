"""Script to visualize TL regression benchmark results organized by model type."""

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


def extract_models_and_metrics(df):
    """Extract model names and metrics from DataFrame columns."""
    # Define metadata columns
    metadata_cols = ['mc_iter', 'n_train_pts', 'fraction_source', 'n_source_pts', 'n_test_pts']
    metric_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Define metric suffixes
    metric_suffixes = ['_rmse', '_mse', '_r2', '_mae', '_max_error', 
                      '_explained_variance', '_kendall_tau', '_spearman_rho']
    
    # Extract models and metrics
    models = set()
    metrics = set()
    
    for col in metric_cols:
        for suffix in metric_suffixes:
            if col.endswith(suffix):
                model_name = col[:-len(suffix)]
                metric_name = suffix[1:].upper()  # Remove underscore and uppercase
                models.add(model_name)
                metrics.add(metric_name)
                break
    
    return sorted(models), sorted(metrics)


def clean_model_name(model_name, source_fraction=None):
    """Convert internal model names to clean display names."""
    if model_name == "vanilla":
        return "GP 0% (reduced searchspace)"
    elif model_name == "GP_Index_Kernel":
        if source_fraction is not None:
            return f"GPIndex {int(source_fraction * 100)}%"
        return "GPIndex"
    elif model_name == "MHGP":
        if source_fraction is not None:
            return f"MHGP {int(source_fraction * 100)}%"
        return "MHGP"  
    elif model_name == "SHGP":
        if source_fraction is not None:
            return f"SHGP {int(source_fraction * 100)}%"
        return "SHGP"
    elif model_name == "Karins_Source_Prior":
        if source_fraction is not None:
            return f"SourceGP {int(source_fraction * 100)}%"
        return "SourceGP"
    else:
        return model_name


def visualize_tl_regression_per_model(json_file_path):
    """Create visualization organized by model type."""
    # Load data
    df, metadata = load_benchmark_data(json_file_path)
    
    print(f"Loaded regression data with {len(df)} rows and {len(df.columns)} columns")
    
    # Extract models and metrics
    models, metrics = extract_models_and_metrics(df)
    print(f"Found models: {models}")
    print(f"Found metrics: {metrics}")
    
    # Separate baseline from TL models
    baseline_model = "vanilla"
    tl_models = [m for m in models if m != baseline_model]
    
    # Order TL models for consistent display
    model_order = ["GP_Index_Kernel", "MHGP", "SHGP", "Karins_Source_Prior"]
    tl_models = [m for m in model_order if m in tl_models]
    
    # Get source fractions
    source_fractions = sorted(df["fraction_source"].unique())
    print(f"Source fractions: {source_fractions}")
    print(f"TL models: {tl_models}")
    
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
    
    # Create figure: n_metrics rows × n_tl_models columns
    n_rows = len(metrics)
    n_cols = len(tl_models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey='row')
    
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
    
    # Define colors for different source fractions
    colors = plt.cm.viridis(np.linspace(0, 1, len(source_fractions)))
    source_colors = dict(zip(source_fractions, colors))
    
    # Define baseline style (unique color, dotted line)
    baseline_color = "#8c564b"
    
    # Plot each metric × model combination
    for row_idx, metric in enumerate(metrics):
        for col_idx, model in enumerate(tl_models):
            ax = axes[row_idx, col_idx]
            
            # Plot baseline (vanilla) - same for all models
            baseline_col = f"{baseline_model}_{metric.lower()}"
            if baseline_col in df.columns:
                # Use data from first source fraction (baseline doesn't depend on source)
                baseline_data = df[df["fraction_source"] == source_fractions[0]]
                baseline_stats = (
                    baseline_data.groupby("n_train_pts")[baseline_col]
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
                    label=clean_model_name(baseline_model)
                )
                ax.fill_between(
                    baseline_stats["n_train_pts"],
                    baseline_stats["mean"] - baseline_stats["std"],
                    baseline_stats["mean"] + baseline_stats["std"],
                    color=baseline_color,
                    alpha=0.15,
                )
            
            # Plot this model at different source fractions
            model_col = f"{model}_{metric.lower()}"
            if model_col in df.columns:
                for source_fraction in source_fractions:
                    source_data = df[df["fraction_source"] == source_fraction]
                    
                    model_stats = (
                        source_data.groupby("n_train_pts")[model_col]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    model_stats["std"] = model_stats["std"].fillna(0)
                    
                    color = source_colors[source_fraction]
                    
                    ax.plot(
                        model_stats["n_train_pts"],
                        model_stats["mean"],
                        color=color,
                        linestyle="-",
                        linewidth=2,
                        alpha=1.0,
                        label=f"{int(source_fraction * 100)}% source"
                    )
                    ax.fill_between(
                        model_stats["n_train_pts"],
                        model_stats["mean"] - model_stats["std"],
                        model_stats["mean"] + model_stats["std"],
                        color=color,
                        alpha=0.15,
                    )
            
            # Customize subplot
            if row_idx == 0:  # Column title only on top row
                ax.set_title(clean_model_name(model), fontsize=11)
            
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
            
            # Collect baseline means
            if baseline_col in df.columns:
                baseline_data = df[df["fraction_source"] == source_fractions[0]]
                baseline_means = baseline_data.groupby("n_train_pts")[baseline_col].mean()
                # Filter out NaN and infinite values
                valid_means = [val for val in baseline_means.values if np.isfinite(val)]
                all_means.extend(valid_means)
            
            # Collect this model's means across all source fractions
            model_col = f"{model}_{metric.lower()}"
            if model_col in df.columns:
                for source_fraction in source_fractions:
                    source_data = df[df["fraction_source"] == source_fraction]
                    model_means = source_data.groupby("n_train_pts")[model_col].mean()
                    # Filter out NaN and infinite values
                    valid_means = [val for val in model_means.values if np.isfinite(val)]
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
    
    plt.close()
    return output_path