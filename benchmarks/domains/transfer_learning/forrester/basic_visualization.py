"""Basic visualization of Forrester function variants for transfer learning benchmarks.

This script visualizes different parameter configurations and noise levels of the Forrester
function to help understand the transfer learning scenarios we can create.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")


def forrester_torch(x, a=1.0, b=0.0, c=0.0, output_noise=0.0):
    """Forrester function implemented in PyTorch.
    
    Args:
        x: Input values, tensor of shape (n_points,)
        a: Scaling factor for the high-fidelity term
        b: Linear term coefficient 
        c: Constant offset
        output_noise: Standard deviation of Gaussian noise to add
        
    Returns:
        Function values of shape (n_points, 1)
    """
    x = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
    y_high = torch.pow(6 * x - 2, 2) * torch.sin(12 * x - 4)
    y = (a * y_high + b * (x - 0.5) - c).reshape(-1, 1)
    
    if output_noise > 0:
        noise = torch.normal(0.0, output_noise, size=y.shape)
        y += noise
        
    return y


def create_forrester_visualization():
    """Create comprehensive visualization of Forrester function variants."""
    # Global shift parameters - easily adjustable (
    # negative = left shift, positive = right shift)
    x_shifts = [-0.15, -0.12, -0.1, 0.1, 0.12, 0.15]
    max_shift = max(abs(s) for s in x_shifts)  # Use absolute value for domain calculation
    
    # Create input grids
    x = torch.linspace(0, 1, 200)
    # Extended range to accommodate both negative and positive shifts
    min_shift = min(x_shifts)
    max_shift_abs = max(abs(s) for s in x_shifts)
    x_shifted_plot = torch.linspace(min_shift, 1 + max_shift_abs, 200)
    
    # Define parameter configurations
    configs = [
        {"a": 1.0, "b": 0.0, "c": 0.0, "name": "Original\n(a=1, b=0, c=0)"},
        {"a": 0.5, "b": 10.0, "c": -5.0, "name": "Low-Fidelity\n(a=0.5, b=10, c=-5)"},
        {"a": 1.5, "b": -2.0, "c": 1.0, "name": "Amplified\n(a=1.5, b=-2, c=1)"},
        # {"a": 0.8, "b": 3.0, "c": -2.0, "name": "Moderate\n(a=0.8, b=3, c=-2)"},
        {"a": 1.2, "b": -5.0, "c": 3.0, "name": "Steep Linear\n(a=1.2, b=-5, c=3)"},
        {"a": -1.0, "b": 0.0, "c": 0.0, "name": "Inverted\n(a=-1, b=0, c=0)"},
    ]
    
    # Increased noise levels as requested
    noise_levels = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0]
    
    # Create figure with denser layout (6 configs + len(x_shifts) shifted columns)
    total_cols = len(configs) + len(x_shifts)
    fig, axes = plt.subplots(len(noise_levels), total_cols, 
                           figsize=(3.5 * total_cols, 2.8 * len(noise_levels)), 
                           sharex=False, sharey='row')
    fig.suptitle("Forrester Function Variants for Transfer Learning Benchmarks\n" +
                r"$f(x) = a \cdot (6x - 2)^2 \cdot \sin(12x - 4) + b(x - 0.5) - c$", 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Get original function for overlay (always show as dashed red reference)
    x_orig = x
    y_orig = forrester_torch(x_orig, a=1.0, b=0.0, c=0.0, output_noise=0.0)
    
    # Set random seed for reproducible noise
    torch.manual_seed(42)
    
    # Calculate global y-limits for each row to share y-axis
    row_ylims = []
    for row, noise in enumerate(noise_levels):
        all_y_values = []
        
        # Collect y values for all configurations in this row
        for config in configs:
            y_temp = forrester_torch(x, **{k: v for k, v in config.items() if k != "name"}, 
                                   output_noise=noise)
            all_y_values.extend(y_temp.numpy().flatten())
            # Also include original function for overlay
            all_y_values.extend(y_orig.numpy().flatten())
        
        # Add shifted function values - both functions evaluated over (0, 1) domain
        # Use restricted domain for y-limits calculation
        x_plot_range_temp = torch.linspace(0, 1, 200)
        
        # Original function f(x) evaluated over (0, 1)
        y_orig_temp = forrester_torch(x_plot_range_temp, a=1.0, b=0.0, c=0.0, 
                                    output_noise=noise)
        all_y_values.extend(y_orig_temp.numpy().flatten())
        
        # All shifted functions g(x) = f(x-x_shift) for each shift parameter over (0, 1)
        for x_shift in x_shifts:
            x_shifted_input_temp = x_plot_range_temp - x_shift
            y_shifted_temp = forrester_torch(x_shifted_input_temp, a=1.0, b=0.0, c=0.0, 
                                           output_noise=noise)
            all_y_values.extend(y_shifted_temp.numpy().flatten())
        
        # Calculate row limits with some padding
        y_min, y_max = np.min(all_y_values), np.max(all_y_values)
        y_range = y_max - y_min
        padding = 0.05 * y_range if y_range > 0 else 1.0
        row_ylims.append((y_min - padding, y_max + padding))
    
    for row, noise in enumerate(noise_levels):
        # Set the same seed for each row to ensure consistent noise patterns
        torch.manual_seed(42 + row)
        
        for col, config in enumerate(configs):
            ax = axes[row, col]
            
            # Generate function values
            y = forrester_torch(x, **{k: v for k, v in config.items() if k != "name"}, 
                              output_noise=noise)
            
            # Always plot original function as dashed red reference
            ax.plot(x.numpy(), y_orig.numpy().flatten(), 'r--', 
                   linewidth=1.5, alpha=0.7)
            
            # Plot current configuration
            if noise == 0.0:
                if config["name"].startswith("Original"):
                    line_color = 'red'
                elif config["name"].startswith("Inverted"):
                    line_color = 'darkblue'
                else:
                    line_color = 'darkgreen'
                ax.plot(x.numpy(), y.numpy().flatten(), color=line_color, 
                       linewidth=2.5, alpha=0.9)
            else:
                # Plot noisy version with scatter
                if config["name"].startswith("Original"):
                    scatter_color = 'red'
                elif config["name"].startswith("Inverted"):
                    scatter_color = 'blue'
                else:
                    scatter_color = 'green'
                ax.scatter(x[::8].numpy(), y[::8].numpy().flatten(), 
                          s=25, alpha=0.7, c=scatter_color, edgecolors='white', linewidth=0.5)
                
                # Add noise-free reference line for non-Original configurations
                if not config["name"].startswith("Original"):
                    y_clean = forrester_torch(x, **{k: v for k, v in config.items() if k != "name"}, 
                                            output_noise=0.0)
                    if config["name"].startswith("Inverted"):
                        dashed_color = 'b--'  # Blue dashed for inverted
                    else:
                        dashed_color = 'g--'  # Green dashed for others
                    ax.plot(x.numpy(), y_clean.numpy().flatten(), dashed_color, 
                           linewidth=1.5, alpha=0.7)
            
            # Formatting
            ax.set_xlim(0, 1)
            ax.set_ylim(row_ylims[row])
            ax.set_xticks([])  # Remove all x-ticks
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#fafafa')
            
            # Enhanced styling
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('#333333')
            
            # Titles and labels - only on edges
            if row == 0:
                ax.set_title(config["name"], fontsize=10, fontweight='bold', 
                           pad=8, color='#2c3e50')
            if col == 0:
                ax.set_ylabel(f'σ={noise}', fontsize=10, fontweight='bold',
                            color='#2c3e50')
            else:
                ax.set_yticklabels([])  # Remove y-tick labels for non-leftmost columns
            # Remove x-axis labels and tick labels completely
            ax.set_xticklabels([])
        
        # Add shifted function columns (one for each shift parameter)
        for shift_idx, x_shift in enumerate(x_shifts):
            ax = axes[row, len(configs) + shift_idx]
            
            # Restrict plotting to (0, 1) domain for consistency
            x_plot_range = torch.linspace(0, 1, 200)
            
            # Plot original function f(x) evaluated over (0, 1)
            y_orig_restricted = forrester_torch(x_plot_range, a=1.0, b=0.0, c=0.0, output_noise=noise)
            
            # Plot shifted function g(x) = f(x-x_shift) evaluated over (0, 1)
            x_shifted_input = x_plot_range - x_shift
            y_shifted_restricted = forrester_torch(x_shifted_input, a=1.0, b=0.0, c=0.0, output_noise=noise)
            
            if noise == 0.0:
                # Always plot original function f(x) as red dashed line over (0, 1)
                ax.plot(x_plot_range.numpy(), y_orig_restricted.numpy().flatten(), 'r--', 
                       linewidth=1.5, alpha=0.7)
                # Plot shifted function g(x) = f(x-x_shift) over (0, 1)
                ax.plot(x_plot_range.numpy(), y_shifted_restricted.numpy().flatten(), 'purple', 
                       linewidth=2.0, alpha=0.8)
            else:
                # Always plot original function f(x) as red dashed line over (0, 1)
                y_orig_clean = forrester_torch(x_plot_range, a=1.0, b=0.0, c=0.0, output_noise=0.0)
                ax.plot(x_plot_range.numpy(), y_orig_clean.numpy().flatten(), 'r--', 
                       linewidth=1.5, alpha=0.7)
                
                # Scatter plot for noisy shifted function over (0, 1)
                ax.scatter(x_plot_range[::8].numpy(), y_shifted_restricted[::8].numpy().flatten(), 
                          s=20, alpha=0.7, c='purple', edgecolors='white', linewidth=0.5,
                          marker='o')
                
                # Also plot clean shifted version as reference over (0, 1)
                y_shifted_clean = forrester_torch(x_shifted_input, a=1.0, b=0.0, c=0.0, output_noise=0.0)
                ax.plot(x_plot_range.numpy(), y_shifted_clean.numpy().flatten(), 'purple', 
                       linewidth=1.0, alpha=0.4, linestyle='--')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(row_ylims[row])
            ax.set_xticks([])  # Remove all x-ticks
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#fafafa')
            
            # Enhanced styling
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color('#333333')
            
            # Titles and labels - only on edges
            if row == 0:
                # Format the shift properly (avoid double negative)
                if x_shift >= 0:
                    shift_str = f"g(x)=f(x-{x_shift})"
                else:
                    shift_str = f"g(x)=f(x+{abs(x_shift)})"
                ax.set_title(f"Shift={x_shift}\n{shift_str}", 
                            fontsize=10, fontweight='bold', pad=8, color='#2c3e50')
            # Remove y-tick labels for all shifted columns (they're never leftmost)
            ax.set_yticklabels([])
            # Remove x-axis labels and tick labels completely
            ax.set_xticklabels([])
    
    # No legends needed
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.04, right=0.98, bottom=0.02, wspace=0.02, hspace=0.05)
    
    # Save the plot in the same directory as this script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "forrester_function_variants.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved as: {output_path}")
    
    plt.show()
    
    return fig


def print_function_characteristics():
    """Print key characteristics of each Forrester variant."""
    # Use the same shift parameters as in visualization
    x_shifts = [-0.12, -0.1, -0.1]
    max_shift_abs = max(abs(s) for s in x_shifts)
    min_shift = min(x_shifts)
    
    print("\n" + "="*80)
    print("FORRESTER FUNCTION VARIANTS - KEY CHARACTERISTICS")
    print("="*80)
    
    x = torch.linspace(0, 1, 1000)
    
    configs = [
        {"a": 1.0, "b": 0.0, "c": 0.0, "name": "Original"},
        {"a": 0.5, "b": 10.0, "c": -5.0, "name": "Low-Fidelity"},
        {"a": 1.5, "b": -2.0, "c": 1.0, "name": "Amplified"},
        #{"a": 0.8, "b": 3.0, "c": -2.0, "name": "Moderate"},
        {"a": 1.2, "b": -5.0, "c": 3.0, "name": "Steep Linear"},
        {"a": -1.0, "b": 0.0, "c": 0.0, "name": "Inverted"},
    ]
    
    for config in configs:
        y = forrester_torch(x, **{k: v for k, v in config.items() if k != "name"})
        y_flat = y.numpy().flatten()
        
        min_val = np.min(y_flat)
        max_val = np.max(y_flat)
        min_idx = np.argmin(y_flat)
        max_idx = np.argmax(y_flat)
        
        print(f"\n{config['name']} (a={config['a']}, b={config['b']}, c={config['c']}):")
        print(f"  Range: [{min_val:.3f}, {max_val:.3f}]")
        print(f"  Global minimum: {min_val:.3f} at x={x[min_idx]:.3f}")
        print(f"  Global maximum: {max_val:.3f} at x={x[max_idx]:.3f}")
    
    # Shifted functions
    x_extended = torch.linspace(min_shift, 1 + max_shift_abs, 1000)
    for x_shift in x_shifts:
        x_shifted_input = x_extended - x_shift
        y_shifted = forrester_torch(x_shifted_input, a=1.0, b=0.0, c=0.0)
        y_shifted_flat = y_shifted.numpy().flatten()
        
        min_val = np.min(y_shifted_flat)
        max_val = np.max(y_shifted_flat)
        min_idx = np.argmin(y_shifted_flat)
        max_idx = np.argmax(y_shifted_flat)
        
        # Format shift description properly
        if x_shift >= 0:
            shift_desc = f"f(x-{x_shift})"
        else:
            shift_desc = f"f(x+{abs(x_shift)})"
        
        print(f"\nShifted g(x) = {shift_desc} on [{min_shift:.1f}, {1+max_shift_abs:.1f}]:")
        print(f"  Range: [{min_val:.3f}, {max_val:.3f}]")
        print(f"  Global minimum: {min_val:.3f} at x={x_extended[min_idx]:.3f}")
        print(f"  Global maximum: {max_val:.3f} at x={x_extended[max_idx]:.3f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Define shifts for the output
    x_shifts = [-0.12, -0.1, -0.1]
    
    # Create visualization
    fig = create_forrester_visualization()
    
    # Print function characteristics
    print_function_characteristics()
    
    print("\nTransfer Learning Scenarios:")
    print("1. Original → Low-Fidelity: High-fidelity to low-fidelity transfer")
    print("2. Low-Fidelity → Original: Low-fidelity to high-fidelity transfer")
    print("3. Original → Inverted: Positive to negative correlation transfer")
    print(f"4. Original → Shifted: Spatial domain shift transfer (shifts={x_shifts})")
    print("5. Any variant → Any other: Cross-parameter transfer learning")
    print("6. Multiple sources: Combine multiple variants as source tasks")

