import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_single(path_csv, algo_name, output_dir, smoothing_window=25):
    df = pd.read_csv(path_csv)
    rewards = df['reward']
    episodes = df['episode']
    
    # Applies mean / smoothing to the rewards
    smoothed = rewards.rolling(window=smoothing_window).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="Raw Reward")
    plt.plot(episodes, smoothed, color='blue', linewidth=2, label=f"Smoothed Mean ({smoothing_window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{algo_name} – Training Reward Over Episodes")
    plt.grid(True)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{algo_name}_reward_plot.png"))
    plt.close()

def plot_combined(paths_csv, labels, output_dir, smoothing_window=50):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.figure(figsize=(12, 6))
    colors = ['royalblue', 'darkorange', 'seagreen']
    
    for path, label, color in zip(paths_csv, labels, colors):
        df = pd.read_csv(path)
        rewards = df['reward']
        episodes = df['episode']
        smoothed = rewards.rolling(window=smoothing_window).mean()

        # Raw Reward
        plt.plot(episodes, rewards, alpha=0.2, color=color, linestyle='--', linewidth=1)

        # Smoothed Reward
        plt.plot(episodes, smoothed, label=f"{label} (Smoothed Mean)", color=color, linewidth=2.5)

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.title("Reward Overview: PPO vs TD3", fontsize=14, weight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "combined_plot.png"))
    plt.close()

