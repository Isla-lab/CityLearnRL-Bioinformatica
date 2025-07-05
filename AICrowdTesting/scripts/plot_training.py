#!/usr/bin/env python3
"""
Training Rewards Visualization Tool

This script helps visualize the training progress by plotting the episode rewards
from TensorBoard log files for both 2-KPI and multi-KPI training.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Default directories
DEFAULT_2KPI_LOG_DIR = os.path.join("results", "ppo_tensorboard_logs_2kpi")
DEFAULT_MULTI_KPI_LOG_DIR = os.path.join("results", "ppo_tensorboard_logs_multi_kpi")

def find_latest_event_file(log_dir):
    """Find the most recent event file in the given directory."""
    event_files = list(Path(log_dir).rglob('events.out.tfevents.*'))
    if not event_files:
        return None
    return max(event_files, key=lambda x: x.stat().st_mtime)

def load_training_data(log_dir, metric_tags=None):
    """
    Load training data from TensorBoard event files.
    
    Args:
        log_dir: Directory containing the TensorBoard logs
        metric_tags: List of metric tags to extract (None for all)
    """
    event_file = find_latest_event_file(log_dir)
    if not event_file:
        print(f"No event files found in {log_dir}")
        return None, {}
    
    print(f"\nLoading training data from: {event_file}")
    
    try:
        # Load the event file
        acc = EventAccumulator(str(event_file.parent))
        acc.Reload()
        
        # Get available tags
        available_tags = acc.Tags()['scalars']
        print(f"Available metrics: {', '.join(available_tags)}")
        
        # If specific tags are requested, filter them
        tags_to_load = metric_tags if metric_tags else available_tags
        
        data = {}
        for tag in tags_to_load:
            if tag in available_tags:
                events = acc.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                data[tag] = pd.DataFrame({'step': steps, tag: values})
                print(f"Loaded {len(values)} data points for {tag}")
            else:
                print(f"Warning: Metric '{tag}' not found in the event file")
        
        return event_file.parent, data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, {}

def plot_training_comparison(log_dirs, labels, metric='rollout/ep_rew_mean'):
    """
    Plot comparison of training metrics across different runs.
    
    Args:
        log_dirs: List of log directories to compare
        labels: List of labels for each log directory
        metric: Metric to plot (default: 'rollout/ep_rew_mean')
    """
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.get_cmap('tab10', len(log_dirs)) # Get distinct colors

    for i, (log_dir, label) in enumerate(zip(log_dirs, labels)):
        _, data = load_training_data(log_dir, [metric])
        if not data or metric not in data:
            print(f"Skipping {log_dir} - no data for {metric}")
            continue
            
        df = data[metric]
        # Calculate moving average and standard deviation
        window = max(1, len(df) // 20) # Ensure window is at least 1
        if window == 0 and len(df) > 0: # Handle case where df is very short
             window = len(df)
        elif len(df) == 0:
            print(f"Skipping {label} for metric {metric} due to empty dataframe.")
            continue

        df_ma = df[metric].rolling(window=window, min_periods=1).mean()
        df_std = df[metric].rolling(window=window, min_periods=1).std()
        
        # Plot moving average with standard deviation
        plt.plot(df['step'], df_ma, linewidth=2, color=colors(i),
                label=f'{label}')
        plt.fill_between(df['step'], df_ma - df_std, df_ma + df_std, alpha=0.2, color=colors(i))
    
    # Formatting
    plt.title(f'Training Comparison: {metric} with ±1 Std Dev', fontsize=16, pad=20)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Improve legend to avoid duplicate labels from fill_between
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if "(Std Dev)" in label: # Keep raw and MA, but only one entry for Std Dev conceptually
            base_label = label.replace(" (Std Dev)", "").replace(" (Raw)","").replace(f" (MA {window} steps)","").strip()
            if base_label not in unique_labels or "(MA" in label : # Prioritize MA label or first entry
                 unique_labels[base_label] = handle # This isn't perfect, but helps declutter
        elif label not in unique_labels:
            unique_labels[label] = handle

    # Create a more organized legend for MA and Std Dev bands
    # We want to show: Run1 (MA), Run1 (Std Dev), Run2 (MA), Run2 (Std Dev)
    # The raw plot can be less prominent or optional if too cluttered.
    # For simplicity, the current auto-generated legend might be acceptable,
    # or a more complex legend handling is needed.
    # The code above with unique_labels tries to simplify it a bit.

    # For now, let's try a simpler approach to legend:
    # Only show MA and one label for the shaded region.
    # This requires careful label management in plot and fill_between.
    # Re-creating handles/labels for clarity:

    new_handles = []
    new_labels = []
    # Group labels by run name to combine MA and StdDev
    # This is a bit tricky with how fill_between creates separate legend entries.
    # A simpler way is to ensure fill_between does not add a new label if MA is already there for that color.

    # Reset legend and add manually for better control if needed, or accept matplotlib's default.
    # Given the current code, matplotlib will create three entries per run: Raw, MA, Std Dev.
    # This is acceptable.
    plt.legend()

    # Improve x-axis formatting
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else int(x))
    )
    
    plt.tight_layout()
    return plt.gcf()

def plot_kpi_metrics(log_dir, title_suffix=""):
    """
    Plot all available KPI metrics from the training run.
    """
    _, data = load_training_data(log_dir)
    if not data:
        return None
    
    # Filter for KPI-related metrics
    kpi_metrics = [k for k in data.keys() if any(x in k.lower() for x in ['kpi', 'reward', 'loss'])]
    if not kpi_metrics:
        print("No KPI metrics found")
        return None
    
    # Create subplots
    n_metrics = len(kpi_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(kpi_metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        df = data[metric]
        
        # Calculate moving average
        window = max(1, len(df) // 20)
        df[f'{metric}_ma'] = df[metric].rolling(window=window, min_periods=1).mean()
        
        # Plot moving average
        ax.plot(df['step'], df[f'{metric}_ma'], linewidth=2, label=f'Moving Average ({window} steps)')
        
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else int(x))
        )
    
    # Remove any extra subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f'Training Metrics {title_suffix}'.strip(), fontsize=16, y=1.02)
    plt.tight_layout()
    return fig

def save_plot(fig, filename, output_dir=os.path.join('results', 'plots')):
    """Save the plot to a file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"\nPlot saved to: {os.path.abspath(filepath)}")
    return filepath
    
def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from TensorBoard logs')
    parser.add_argument('--log-dir-2kpi', type=str, default=DEFAULT_2KPI_LOG_DIR,
                       help=f'Directory containing 2-KPI TensorBoard logs (default: {DEFAULT_2KPI_LOG_DIR})')
    parser.add_argument('--log-dir-multi', type=str, default=DEFAULT_MULTI_KPI_LOG_DIR,
                       help=f'Directory containing multi-KPI TensorBoard logs (default: {DEFAULT_MULTI_KPI_LOG_DIR})')
    parser.add_argument('--output-dir', type=str, default=os.path.join('results', 'training_plots'),
                       help='Directory to save output plots (default: results/training_plots)')
    parser.add_argument('--all-plots', action='store_true',
                       help='Generate all individual KPI plots in addition to the main comparison plot.')
    # The --compare flag is effectively removed as comparison is default.
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    # Using a fixed name for the primary comparison plot for easier access
    # timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    log_dir_2kpi_exists = os.path.exists(args.log_dir_2kpi)
    log_dir_multi_exists = os.path.exists(args.log_dir_multi)

    if log_dir_2kpi_exists and log_dir_multi_exists:
        print("\n" + "="*50)
        print("GENERATING PRIMARY TRAINING COMPARISON PLOT")
        print("="*50)
        fig_compare_rew = plot_training_comparison(
            [args.log_dir_2kpi, args.log_dir_multi],
            ['2-KPI', 'Multi-KPI'],
            'rollout/ep_rew_mean'
        )
        if fig_compare_rew:
            save_plot(fig_compare_rew, "training_comparison_with_std.png", args.output_dir)
        else:
            print("Could not generate primary comparison plot for rollout/ep_rew_mean.")

        # Optionally, compare episode lengths if all_plots is not the main focus
        # For now, focusing on the primary reward plot as per instructions.
        # If you want to add ep_len_mean comparison by default:
        # print("\n" + "="*50)
        # print("GENERATING EPISODE LENGTH COMPARISON PLOT")
        # print("="*50)
        # fig_compare_len = plot_training_comparison(
        #     [args.log_dir_2kpi, args.log_dir_multi],
        #     ['2-KPI', 'Multi-KPI'],
        #     'rollout/ep_len_mean'
        # )
        # if fig_compare_len:
        #     save_plot(fig_compare_len, f"ep_len_comparison_with_std_{timestamp}.png", args.output_dir)
        # else:
        #     print("Could not generate episode length comparison plot.")

    else:
        print("\n" + "="*50)
        print("SKIPPING COMPARISON PLOT: One or both log directories not found.")
        print(f"2-KPI directory ({args.log_dir_2kpi}) found: {log_dir_2kpi_exists}")
        print(f"Multi-KPI directory ({args.log_dir_multi}) found: {log_dir_multi_exists}")
        print("="*50)

    # Plot individual training runs if --all-plots is specified
    if args.all_plots:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S') # Use timestamp for these optional plots
        if log_dir_2kpi_exists:
            print("\n" + "="*50)
            print("PLOTTING INDIVIDUAL 2-KPI TRAINING METRICS")
            print("="*50)
            fig_2kpi = plot_kpi_metrics(args.log_dir_2kpi, "(2-KPI Training)")
            if fig_2kpi:
                save_plot(fig_2kpi, f"2kpi_metrics_{timestamp}.png", args.output_dir)
        
        if log_dir_multi_exists:
            print("\n" + "="*50)
            print("PLOTTING INDIVIDUAL MULTI-KPI TRAINING METRICS")
            print("="*50)
            fig_multi = plot_kpi_metrics(args.log_dir_multi, "(Multi-KPI Training)")
            if fig_multi:
                save_plot(fig_multi, f"multi_kpi_metrics_{timestamp}.png", args.output_dir)
    
    print("\n" + "="*50)
    print("PLOTTING SCRIPT EXECUTION COMPLETE")
    print("="*50)
    print(f"\nTo view the TensorBoard logs, run:")
    if os.path.exists(args.log_dir_2kpi):
        print(f"  2-KPI:    tensorboard --logdir={os.path.abspath(args.log_dir_2kpi)}")
    if os.path.exists(args.log_dir_multi):
        print(f"  Multi-KPI: tensorboard --logdir={os.path.abspath(args.log_dir_multi)}")
    print(f"\nPlots saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
