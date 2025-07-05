import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def find_latest_event_file(log_dir):
    """Finds the latest event file in a directory, searching recursively."""
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        return None
    return max(event_files, key=os.path.getctime)

def extract_scalar_data(event_file, scalar_keys):
    """Extracts scalar data from an event file."""
    ea = event_accumulator.EventAccumulator(event_file,
                                            size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    data = {}
    for key in scalar_keys:
        if key in ea.Tags()['scalars']:
            events = ea.Scalars(key)
            data[key] = pd.DataFrame([(s.step, s.value) for s in events], columns=['step', 'value'])
        else:
            print(f"Warning: Scalar key '{key}' not found in {event_file}")
            data[key] = pd.DataFrame(columns=['step', 'value']) # Empty dataframe if key not found
    return data

def main():
    log_dirs = {
        "2KPI": os.path.join("results", "ppo_tensorboard_logs_2kpi"),
        "MultiKPI": os.path.join("results", "ppo_tensorboard_logs_multi_kpi")
    }

    scalar_keys_to_extract = ["rollout/ep_rew_mean", "train/loss", "eval/mean_reward"]

    all_data = {}

    for run_name, log_dir_path in log_dirs.items():
        latest_event_file = find_latest_event_file(log_dir_path)
        if latest_event_file:
            print(f"Processing {run_name}: {latest_event_file}")
            all_data[run_name] = extract_scalar_data(latest_event_file, scalar_keys_to_extract)
        else:
            print(f"Warning: No event file found for {run_name} in {log_dir_path}")
            all_data[run_name] = {key: pd.DataFrame(columns=['step', 'value']) for key in scalar_keys_to_extract}

    # Plotting
    num_plots = 0
    if any(not all_data[run_name]["rollout/ep_rew_mean"].empty for run_name in log_dirs):
        num_plots += 1
    if any(not all_data[run_name]["train/loss"].empty for run_name in log_dirs if "train/loss" in all_data[run_name]):
        num_plots +=1
    if any(not all_data[run_name]["eval/mean_reward"].empty for run_name in log_dirs if "eval/mean_reward" in all_data[run_name]):
        num_plots +=1

    if num_plots == 0:
        print("No data available to plot.")
        return

    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), squeeze=False)
    plot_idx = 0

    # Plot rollout/ep_rew_mean
    if any(not all_data[run_name]["rollout/ep_rew_mean"].empty for run_name in log_dirs):
        ax = axs[plot_idx, 0]
        for run_name, data in all_data.items():
            df = data.get("rollout/ep_rew_mean")
            if not df.empty:
                ax.plot(df['step'], df['value'], label=f"{run_name} - Episode Reward Mean")
        ax.set_xlabel("Step")
        ax.set_ylabel("Episode Reward Mean")
        ax.set_title("rollout/ep_rew_mean")
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    # Plot train/loss
    if any(not all_data[run_name]["train/loss"].empty for run_name in log_dirs if "train/loss" in all_data[run_name]):
        ax = axs[plot_idx, 0]
        for run_name, data in all_data.items():
            df = data.get("train/loss")
            if df is not None and not df.empty:
                ax.plot(df['step'], df['value'], label=f"{run_name} - Training Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("train/loss")
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    # Plot eval/mean_reward
    if any(not all_data[run_name]["eval/mean_reward"].empty for run_name in log_dirs if "eval/mean_reward" in all_data[run_name]):
        ax = axs[plot_idx, 0]
        for run_name, data in all_data.items():
            df = data.get("eval/mean_reward")
            if df is not None and not df.empty:
                ax.plot(df['step'], df['value'], label=f"{run_name} - Evaluation Mean Reward")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Reward")
        ax.set_title("eval/mean_reward")
        ax.legend()
        ax.grid(True)
        plot_idx +=1

    plt.tight_layout()
    plt.savefig("analysis_plots.png")
    print("Plots saved to analysis_plots.png")

if __name__ == "__main__":
    main()
