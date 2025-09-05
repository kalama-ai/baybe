"""Script to visualize TL convergence benchmark results organized by model type."""

import base64
import json
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


def parse_scenario(scenario):
    """Parse scenario string to extract source percentage and model type.
    
    Args:
        scenario: Scenario string like '10_mhgp', '0_naive', '0', etc.
        
    Returns:
        Tuple of (source_percent, model_name, is_baseline)
    """
    if scenario == "0":
        return 0, "GPIndex_0pct_full", True
    elif scenario == "0_naive":
        return 0, "GP_0pct_reduced", True
    elif "_" in scenario:
        parts = scenario.split("_", 1)
        source_percent = int(parts[0])
        model_name = parts[1]
        if model_name == "naive":
            return source_percent, "GP_reduced", False
        elif model_name == "index_kernel":
            return source_percent, "GPIndex", False
        elif model_name == "mhgp":
            return source_percent, "MHGP", False
        elif model_name == "shgp":
            return source_percent, "SHGP", False
        elif model_name == "source_prior":
            return source_percent, "SourceGP", False
        elif model_name == "source_prior_wrapped":
            return source_percent, "SourceGP_Wrapped", False
        else:
            return source_percent, f"TL_{model_name}", False
    else:
        return int(scenario), "TL_baseline", False


def visualize_tl_convergence_per_model(json_file_path):
    """Create visualization organized by model type."""
    # Load data
    df, metadata = load_benchmark_data(json_file_path)

    # Automatically detect target column names
    iter_best_col = None
    cum_best_col = None
    target_name = None

    for col in df.columns:
        if col.endswith("_IterBest"):
            iter_best_col = col
            target_name = col.replace("_IterBest", "")
        elif col.endswith("_CumBest"):
            cum_best_col = col

    if iter_best_col is None or cum_best_col is None:
        raise ValueError(
            f"Could not find IterBest and CumBest columns in {json_file_path}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"Detected target columns: {iter_best_col}, {cum_best_col}")

    # Parse scenarios and organize data
    baselines = {}
    model_data = {}
    
    for scenario in df["Scenario"].unique():
        source_pct, model_name, is_baseline = parse_scenario(scenario)
        
        if is_baseline:
            baselines[model_name] = scenario
        else:
            if model_name not in model_data:
                model_data[model_name] = {}
            model_data[model_name][source_pct] = scenario

    # Define the desired order of models for visualization
    model_order = ["GPIndex", "MHGP", "SHGP", "SourceGP", "SourceGP_Wrapped", "GP_reduced"]
    available_models = [model for model in model_order if model in model_data]
    
    print(f"Available models: {available_models}")
    print(f"Baselines: {list(baselines.keys())}")

    # Get all source percentages (excluding 0 since that's handled as baselines)
    all_source_pcts = set()
    for source_dict in model_data.values():
        all_source_pcts.update(source_dict.keys())
    all_source_pcts = sorted([pct for pct in all_source_pcts if pct > 0])
    print(f"Source percentages: {all_source_pcts}")

    # Set up the plot style
    plt.style.use("default")
    sns.set_palette("tab10")
    
    # Create figure with subplots: 2 rows (CumBest, IterBest) x (1 baseline + n_models) columns
    n_cols = 1 + len(available_models)  # 1 for baselines + one per model
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), sharey='row')
    
    # Handle single column case
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(
        f"TL Convergence by Model Type: {metadata['name']}", 
        fontsize=14, fontweight="bold"
    )

    # Define unified, extensible color scheme (matches regression scripts)
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
    
    # Dynamically assign colors to detected source percentages
    source_colors = {}
    for i, source_pct in enumerate(all_source_pcts):
        color_idx = i % len(source_fraction_palette)
        source_colors[source_pct] = source_fraction_palette[color_idx]
    
    # Define baseline colors (matches regression scripts)
    baseline_styles = {
        "GPIndex_0pct_full": {"color": "#7f8c8d", "linestyle": "--", "alpha": 0.8, "linewidth": 2},
        "GP_0pct_reduced": {"color": "#95a5a6", "linestyle": ":", "alpha": 0.8, "linewidth": 2}
    }

    def plot_model_scenarios(ax_cum, ax_iter, scenarios_to_plot, title_text, add_legend=False):
        """Helper function to plot model scenarios on given axes."""
        for source_pct, scenario in scenarios_to_plot.items():
            scenario_df = df[df["Scenario"] == scenario]
            
            # Group by iteration and calculate mean and std
            grouped = (
                scenario_df.groupby("Iteration")
                .agg({iter_best_col: ["mean", "std"], cum_best_col: ["mean", "std"]})
                .reset_index()
            )
            
            # Flatten column names
            grouped.columns = [
                "Iteration",
                "IterBest_mean",
                "IterBest_std", 
                "CumBest_mean",
                "CumBest_std",
            ]
            
            # Fill NaN std values with 0
            grouped = grouped.fillna(0)
            
            iterations = grouped["Iteration"]
            
            # Use source percentage color
            color = source_colors[source_pct]
            label = f"{source_pct}% source"
            
            # Plot CumBest
            ax_cum.plot(
                iterations,
                grouped["CumBest_mean"],
                label=label,
                color=color,
                linestyle="-",
                alpha=1.0,
                linewidth=2,
            )
            ax_cum.fill_between(
                iterations,
                grouped["CumBest_mean"] - grouped["CumBest_std"],
                grouped["CumBest_mean"] + grouped["CumBest_std"],
                color=color,
                alpha=0.15,
            )
            
            # Plot IterBest
            ax_iter.plot(
                iterations,
                grouped["IterBest_mean"],
                label=label,
                color=color,
                linestyle="-",
                alpha=1.0,
                linewidth=2,
            )
            ax_iter.fill_between(
                iterations,
                grouped["IterBest_mean"] - grouped["IterBest_std"],
                grouped["IterBest_mean"] + grouped["IterBest_std"],
                color=color,
                alpha=0.15,
            )

        # Add optimal target line if available
        if (
            "optimal_target_values" in metadata
            and target_name in metadata["optimal_target_values"]
        ):
            optimal_value = metadata["optimal_target_values"][target_name]
            ax_cum.axhline(
                y=optimal_value,
                color="red",
                linestyle=":",
                alpha=0.7,
                linewidth=1,
            )
            ax_iter.axhline(
                y=optimal_value,
                color="red",
                linestyle=":",
                alpha=0.7,
                linewidth=1,
            )

        # Customize subplots
        ax_cum.set_title(title_text, fontsize=11)
        ax_cum.grid(True, alpha=0.3)
        ax_cum.tick_params(labelsize=9)
        
        ax_iter.set_xlabel("Iteration", fontsize=10)
        ax_iter.grid(True, alpha=0.3)
        ax_iter.tick_params(labelsize=9)
        
        # No individual legends - unified legend will be created later

    def plot_baseline_scenarios(ax_cum, ax_iter, baseline_scenarios, title_text, add_legend=False):
        """Helper function to plot baseline scenarios on given axes."""
        for model_name, scenario in baseline_scenarios.items():
            scenario_df = df[df["Scenario"] == scenario]
            
            # Group by iteration and calculate mean and std
            grouped = (
                scenario_df.groupby("Iteration")
                .agg({iter_best_col: ["mean", "std"], cum_best_col: ["mean", "std"]})
                .reset_index()
            )
            
            # Flatten column names
            grouped.columns = [
                "Iteration",
                "IterBest_mean",
                "IterBest_std", 
                "CumBest_mean",
                "CumBest_std",
            ]
            
            # Fill NaN std values with 0
            grouped = grouped.fillna(0)
            
            iterations = grouped["Iteration"]
            
            # Use baseline styling
            style = baseline_styles[model_name]
            if model_name == "GPIndex_0pct_full":
                label = "GPIndex 0% (full searchspace)"
            else:
                label = "GP 0% (reduced searchspace)"
            
            # Plot CumBest
            ax_cum.plot(
                iterations,
                grouped["CumBest_mean"],
                label=label,
                **style
            )
            ax_cum.fill_between(
                iterations,
                grouped["CumBest_mean"] - grouped["CumBest_std"],
                grouped["CumBest_mean"] + grouped["CumBest_std"],
                color=style["color"],
                alpha=0.15,
            )
            
            # Plot IterBest
            ax_iter.plot(
                iterations,
                grouped["IterBest_mean"],
                label=label,
                **style
            )
            ax_iter.fill_between(
                iterations,
                grouped["IterBest_mean"] - grouped["IterBest_std"],
                grouped["IterBest_mean"] + grouped["IterBest_std"],
                color=style["color"],
                alpha=0.15,
            )

        # Add optimal target line if available
        if (
            "optimal_target_values" in metadata
            and target_name in metadata["optimal_target_values"]
        ):
            optimal_value = metadata["optimal_target_values"][target_name]
            ax_cum.axhline(
                y=optimal_value,
                color="red",
                linestyle=":",
                alpha=0.7,
                linewidth=1,
            )
            ax_iter.axhline(
                y=optimal_value,
                color="red",
                linestyle=":",
                alpha=0.7,
                linewidth=1,
            )

        # Customize subplots
        ax_cum.set_title(title_text, fontsize=11)
        ax_cum.grid(True, alpha=0.3)
        ax_cum.tick_params(labelsize=9)
        
        ax_iter.set_xlabel("Iteration", fontsize=10)
        ax_iter.grid(True, alpha=0.3)
        ax_iter.tick_params(labelsize=9)
        
        # No individual legends - unified legend will be created later

    # First column: Baselines only
    ax_cum_baseline = axes[0, 0]
    ax_iter_baseline = axes[1, 0]
    plot_baseline_scenarios(
        ax_cum_baseline, ax_iter_baseline, 
        baselines, "Baselines", add_legend=False
    )

    # Remaining columns: Each model with its source percentages + baselines
    for col_idx, model_name in enumerate(available_models, 1):
        source_scenarios = model_data[model_name]
        
        ax_cum = axes[0, col_idx]
        ax_iter = axes[1, col_idx]
        
        # First plot baselines on this column
        plot_baseline_scenarios(
            ax_cum, ax_iter, 
            baselines, "", add_legend=False
        )
        
        # Then plot the model scenarios
        plot_model_scenarios(
            ax_cum, ax_iter, 
            source_scenarios, "", add_legend=False
        )
        
        # Set title and customize after plotting
        if model_name == "GP_reduced":
            model_title = "GP (reduced searchspace)"
        elif model_name == "SourceGP_Wrapped":
            model_title = "SourcePrior (Wrapped)"
        else:
            model_title = model_name
            
        ax_cum.set_title(model_title, fontsize=11)
        
    # Create unified legend in the first subplot (top-left) in lower right corner
    legend_elements = []
    
    # Add source fractions (solid lines)
    for source_pct in sorted(all_source_pcts):
        if source_pct in source_colors:
            color = source_colors[source_pct]
            legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='-', 
                                            linewidth=2, label=f'{source_pct}%'))
    
    # Add baseline models
    legend_elements.append(plt.Line2D([0], [0], color="#95a5a6", linestyle=':', 
                                    linewidth=2, label='0% (reduced)'))
    legend_elements.append(plt.Line2D([0], [0], color="#7f8c8d", linestyle='--', 
                                    linewidth=2, label='0% (full)'))
    
    # Add the legend to the first subplot (top-left) in lower right corner
    first_ax = axes[0, 0]
    first_ax.legend(handles=legend_elements, loc='lower right', fontsize=8, 
                   frameon=True, fancybox=True, shadow=True)

    # Set row labels
    axes[0, 0].set_ylabel(f"{cum_best_col}", fontsize=10)
    axes[1, 0].set_ylabel(f"{iter_best_col}", fontsize=10)

    # Adjust layout and fix title positioning
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.25, wspace=0.15)

    # Save the plot
    input_path = Path(json_file_path)
    output_path = input_path.parent / (
        input_path.stem.replace("_result", "_tl_convergence_per_model") + ".png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"TL convergence per model visualization saved to: {output_path}")

    # Show basic statistics
    print("\nDataset statistics:")
    print(f"- Total rows: {len(df)}")
    print(f"- Scenarios: {len(df['Scenario'].unique())}")
    print(f"- Monte Carlo runs: {df['Monte_Carlo_Run'].nunique()}")
    print(f"- Iterations per run: {df['Iteration'].max() + 1}")

    plt.close()
    return output_path