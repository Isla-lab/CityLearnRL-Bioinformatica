"""
Training script for an RL agent on the CustomCityEnv.

This script demonstrates a two-phase training process:
1. Initial training focused on two Key Performance Indicators (KPIs): comfort and emissions.
2. Continued training incorporating additional KPIs (e.g., grid impact),
   starting from the policy learned in the first phase.

It uses stable-baselines3 for the PPO agent and CustomCityEnv for the environment.
Evaluation results and trained models are saved for each phase.
TensorBoard logging is enabled for monitoring training progress.
"""
import os
import sys
from pathlib import Path

# Ensure the src package is discoverable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import time
import torch
import random
import numpy as np
import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import matplotlib.pyplot as plt
import json
import shutil
import glob
from pathlib import Path

# Import stable_baselines3 directly to ensure all components are available
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
# BaseCallback is now imported with other callbacks above
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor

# Import callbacks from stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import get_schedule_fn

# ===== Utility Functions =====

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize_reward(reward: float, min_reward: float, max_reward: float) -> float:
    """Normalize reward to a 0-10 scale for comparison."""
    return ((reward - min_reward) / (max_reward - min_reward)) * 10 if max_reward > min_reward else reward

class TrainingLogger:
    """Helper class to log training metrics for comparison."""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {
            'episode': [],
            'reward': [],
            'normalized_reward': [],
            'episode_length': [],
            'phase': []
        }
    
    def log_episode(self, episode: int, reward: float, episode_length: int, phase: str, 
                   min_reward: float = 0, max_reward: float = 10) -> None:
        """Log metrics for a single episode."""
        norm_reward = normalize_reward(reward, min_reward, max_reward)
        self.metrics['episode'].append(episode)
        self.metrics['reward'].append(reward)
        self.metrics['normalized_reward'].append(norm_reward)
        self.metrics['episode_length'].append(episode_length)
        self.metrics['phase'].append(phase)
        
        # Print episode summary
        print(f"\nEpisode {episode + 1} ({phase}):")
        print(f"  Raw reward: {reward:.4f}")
        print(f"  Normalized reward: {norm_reward:.4f}")
        print(f"  Episode length: {episode_length}")
    
    def save_to_csv(self, filename: str) -> None:
        """Save logged metrics to a CSV file."""
        df = pd.DataFrame(self.metrics)
        filepath = os.path.join(self.log_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"\nMetrics saved to {filepath}")

# Clean up previous training artifacts
def cleanup_previous_runs():
    """
    Remove all previous training artifacts including models, logs, evaluations, and plots.
    Cleans both /results and /scripts/results directories.
    This ensures a clean start for each training run.
    """
    # Define base directories to clean
    base_dirs = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results')),  # /results
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))  # /scripts/results
    ]
    
    # Define subdirectories to clean within each base directory
    sub_dirs = [
        '',  # The base directory itself
        'logs',
        'models',
        'evaluation',
        'tensorboard_logs',
        'plots',
        'checkpoints',
        'ppo_tensorboard_logs_2kpi',
        'ppo_tensorboard_logs_multi_kpi'
    ]
    
    # Build full paths to clean
    dirs_to_clean = []
    for base in base_dirs:
        for sub in sub_dirs:
            dirs_to_clean.append(os.path.join(base, sub) if sub else base)
    
    # Add specific patterns to clean
    patterns_to_clean = [
        'ppo_*_model*',  # Model files
        '*.log',        # Log files
        '*.csv',        # CSV result files
        '*.png',        # Plot images
        '*.jpg',
        '*.jpeg',
        '*.gif',
        '*.pdf',
        'events.out.tfevents.*',  # TensorBoard event files
        '*.pt',         # PyTorch files
        '*.pth',
        '*.bin',
        '*.h5',
        '*.zip',
        '*.pkl',
        '*.json',
        '*.yaml',
        '*.yml'
    ]
    
    print("\n" + "="*50)
    print("CLEANING UP PREVIOUS RUNS")
    print("="*50)
    
    # Remove matching files in the root directory
    for pattern in patterns_to_clean:
        for file_path in glob.glob(pattern):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Removed directory: {file_path}")
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")
    
    # Clean up directories
    for dir_path in dirs_to_clean:
        try:
            if os.path.exists(dir_path):
                if os.path.isfile(dir_path):
                    os.remove(dir_path)
                    print(f"Removed file: {dir_path}")
                else:
                    shutil.rmtree(dir_path)
                    print(f"Removed directory: {dir_path}")
        except Exception as e:
            print(f"Warning: Could not remove {dir_path}: {e}")
    
    # Recreate necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs(os.path.join('results', 'logs'), exist_ok=True)
    os.makedirs(os.path.join('results', 'models'), exist_ok=True)
    os.makedirs(os.path.join('results', 'evaluation'), exist_ok=True)
    os.makedirs(os.path.join('results', 'plots'), exist_ok=True)
    
    print("\nCleanup complete. Ready to start a fresh training run.")
    print("="*50 + "\n")

# Run cleanup at the start
cleanup_previous_runs()

# Import environment and reward calculation from the src package
from citylearn_rl.reward import PhaseWeights, BASELINE_KPIS  # BASELINE_KPIS is used by CustomCityEnv
from citylearn_rl.env import CustomCityEnv

# --- Constants for training phases ---
SEED = 42
set_seed(SEED)

TOTAL_TIMESTEPS_2KPI = 200000
TOTAL_TIMESTEPS_MULTI = 500000
EVAL_FREQ = 5000  # Evaluate every 5k steps
N_EVAL_EPISODES = 5  # Number of episodes for evaluation

MODEL_SAVE_PATH_2KPI_BASE = os.path.join("results", "models", "ppo_2kpi_model")
MODEL_SAVE_PATH_MULTI_BASE = os.path.join("results", "models", "ppo_multi_kpi_model")
TENSORBOARD_LOG_PATH_2KPI_BASE = os.path.join("results", "ppo_tensorboard_logs_2kpi")
TENSORBOARD_LOG_PATH_MULTI_BASE = os.path.join("results", "ppo_tensorboard_logs_multi_kpi")

REWARD_BOUNDS = {
    '2kpi': {'min': 0, 'max': 10},    # Adjust these values based on your environment
    'multi': {'min': 0, 'max': 10}    # Adjust these values based on your environment
}

training_logger = TrainingLogger(log_dir=os.path.join("results", "training_logs"))
EVAL_RESULTS_CSV_2KPI = os.path.join("results", "evaluation_results_2kpi.csv")
EVAL_RESULTS_CSV_MULTI = os.path.join("results", "evaluation_results_multi_kpi.csv")

def smooth_data(y, window_size=5):
    """Smooth 1D array using a moving average."""
    if len(y) < window_size:
        return y, np.zeros_like(y)
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(y, window, mode='same')
    
    # Calculate rolling standard deviation
    rolling_std = np.zeros_like(y)
    for i in range(len(y) - window_size + 1):
        rolling_std[i + window_size//2] = np.std(y[i:i+window_size])
    
    # Handle edges
    rolling_std[:window_size//2] = rolling_std[window_size//2]
    rolling_std[-(window_size//2):] = rolling_std[-(window_size//2)-1]
    
    return smoothed, rolling_std

def plot_training_results(window_size=25):
    """
    Generate training and KPI visualisations using a unified style.
    
    Args:
        window_size (int): Size of the moving average window.
    
    Returns:
        None
    """
    try:
        import glob
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from pathlib import Path

        def load_rewards(log_base_path: str) -> Optional[pd.DataFrame]:
            """
            Load reward history from TensorBoard event files.
            Handles SB3's typical subdirectory structure for runs.
            
            Args:
                log_base_path (str): Base path for the log directory (e.g., "ppo_tensorboard_logs_2kpi").
            
            Returns:
                Optional[pd.DataFrame]: Reward history with timesteps as index, or None if loading fails.
            """
            try:
                base_path = Path(log_base_path)
                if not base_path.exists():
                    print(f"Log base path {log_base_path} does not exist.")
                    return None

                run_dirs = [d for d in base_path.iterdir() if d.is_dir()]
                if not run_dirs:
                    # Check if event files are directly in log_base_path
                    event_files_in_base = list(base_path.glob('events.out.tfevents.*'))
                    if event_files_in_base:
                        run_dirs = [base_path] # Treat base itself as the run_dir
                    else:
                        print(f"No run directories or event files found in {log_base_path}.")
                        return None

                latest_run_dir = None
                latest_event_time = -1

                for rundir_candidate in run_dirs:
                    event_files = list(rundir_candidate.glob('events.out.tfevents.*'))
                    if not event_files:
                        continue
                    
                    current_latest_event_file = max(event_files, key=lambda x: x.stat().st_mtime)
                    current_event_time = current_latest_event_file.stat().st_mtime

                    if current_event_time > latest_event_time:
                        latest_event_time = current_event_time
                        latest_run_dir = rundir_candidate
                
                if not latest_run_dir:
                    print(f"No event files found in any subdirectories of {log_base_path}.")
                    return None

                print(f"Loading events from: {latest_run_dir}")
                acc = EventAccumulator(str(latest_run_dir))
                acc.Reload()

                required_tag = 'rollout/ep_rew_mean'
                if required_tag not in acc.Tags()['scalars']:
                    print(f"'{required_tag}' not found in {latest_run_dir}.")
                    print(f"Available scalar tags: {list(acc.Tags()['scalars'])}")
                    alternative_tags = ['eval/mean_reward', 'ep_reward_mean'] # Common alternatives
                    for alt_tag in alternative_tags:
                        if alt_tag in acc.Tags()['scalars']:
                            print(f"Found alternative tag: {alt_tag}")
                            required_tag = alt_tag
                            break
                    else: # If no alternative found
                        return None
                
                events = acc.Scalars(required_tag)
                if not events:
                    print(f"No data found for tag '{required_tag}' in {latest_run_dir}.")
                    return None

                steps = [e.step for e in events]
                rewards = [e.value for e in events]
                df = pd.DataFrame({'step': steps, 'reward': rewards}).set_index('step')
                print(f"Loaded {len(df)} reward entries from {latest_run_dir} for tag '{required_tag}'.")
                return df

            except Exception as e:
                print(f"Error loading rewards from {log_base_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                return None

        # Load reward histories
        print("Loading training histories...")
        history_2kpi = load_rewards(TENSORBOARD_LOG_PATH_2KPI_BASE)
        history_multi = load_rewards(TENSORBOARD_LOG_PATH_MULTI_BASE)
        
        # Create figure for training curves
        print("Generating training curves...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each training phase
        for df, label, color in [
            (history_2kpi, '2-KPI Training', 'green'),
            (history_multi, 'Multi-KPI Training', 'orange')
        ]:
            if df is None or df.empty:
                print(f"No data available for {label}")
                continue
                
            try:
                # Ensure we have a DataFrame with numeric data
                df = df.apply(pd.to_numeric, errors='coerce')
                if 'reward' not in df.columns:
                    print(f"'reward' column not found in {label} data")
                    continue
                
                # Calculate rolling mean and std
                window = min(window_size, len(df) // 4)  # Ensure window is not too large
                if window < 1:
                    window = 1
                
                df['smoothed'] = df['reward'].rolling(window=window, min_periods=1, center=True).mean()
                df['std'] = df['reward'].rolling(window=window, min_periods=1, center=True).std().fillna(0)
                
                # Plot the smoothed line
                ax.plot(df.index, df['smoothed'], 
                       label=f'{label} (smoothed)', 
                       color=color,
                       linewidth=2)
                
                # Plot the standard deviation
                ax.fill_between(df.index, 
                              df['smoothed'] - df['std'], 
                              df['smoothed'] + df['std'],
                              color=color, 
                              alpha=0.2,
                              label=f'{label} ±1 std' if label == '2-KPI Training' else "_")
                
                print(f"Plotted {label} with {len(df)} data points")
                
            except Exception as e:
                print(f"Error plotting {label}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Customize the plot
        ax.set_title('Training Progress', fontsize=14, pad=20)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Average Episode Reward', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Ensure plots directory exists
        os.makedirs(os.path.join('results', 'plots'), exist_ok=True)
        
        # Save the figure
        plot_path = os.path.join('results', 'plots', 'training_curves.png')
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        print(f"\nTraining curves saved to: {os.path.abspath(plot_path)}")
        
        # Show the plot if in interactive mode
        if 'IPython' in sys.modules:
            plt.show()
        else:
            plt.close(fig)

        # --- KPI Metrics ---
        print("\nProcessing KPI metrics...")
        
        # Check if we have any data to plot
        if history_2kpi is None and history_multi is None:
            print("No training data available for KPI metrics.")
            return
            
        # Define KPIs to plot with their display properties
        kpi_configs = [
            {'id': 'comfort', 'title': 'Comfort Score', 'color': '#1f77b4'},
            {'id': 'emissions', 'title': 'Emissions Score', 'color': '#ff7f0e'},
            {'id': 'grid_impact', 'title': 'Grid Impact Score', 'color': '#2ca02c'},
            {'id': 'resilience', 'title': 'Resilience Score', 'color': '#d62728'}
        ]
        
        # Create figure for KPI comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Track if we actually have any data to plot
        has_data = False
        
        # Plot each KPI
        for idx, kpi_cfg in enumerate(kpi_configs):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            kpi = kpi_cfg['id']
            color = kpi_cfg['color']
            
            # Track if we have data for this specific KPI
            kpi_has_data = False
            
            # Plot 2-KPI training data if available
            if history_2kpi is not None and f'kpi_{kpi}' in history_2kpi.columns:
                try:
                    df = history_2kpi[[f'kpi_{kpi}']].copy()
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()
                    if not df.empty:
                        # Smooth the KPI data
                        window = min(10, len(df) // 10) or 1
                        df['smoothed'] = df[f'kpi_{kpi}'].rolling(window=window, min_periods=1, center=True).mean()
                        
                        # Plot the smoothed line
                        ax.plot(df.index, df['smoothed'], 
                               label='2-KPI Training', 
                               color=color,
                               linewidth=2)
                        
                        kpi_has_data = True
                        has_data = True
                        print(f"Plotted 2-KPI {kpi} with {len(df)} data points")
                except Exception as e:
                    print(f"Error plotting 2-KPI {kpi}: {str(e)}")
            
            # Plot multi-KPI training data if available
            if history_multi is not None and f'kpi_{kpi}' in history_multi.columns:
                try:
                    df = history_multi[[f'kpi_{kpi}']].copy()
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()
                    if not df.empty:
                        # Smooth the KPI data
                        window = min(10, len(df) // 10) or 1
                        df['smoothed'] = df[f'kpi_{kpi}'].rolling(window=window, min_periods=1, center=True).mean()
                        
                        # Plot the smoothed line with a different style
                        ax.plot(df.index, df['smoothed'], 
                               label='Multi-KPI Training', 
                               color=color,
                               linestyle='--',
                               linewidth=2,
                               alpha=0.8)
                        
                        kpi_has_data = True
                        has_data = True
                        print(f"Plotted Multi-KPI {kpi} with {len(df)} data points")
                except Exception as e:
                    print(f"Error plotting Multi-KPI {kpi}: {str(e)}")
            
            # Only customize the subplot if we have data
            if kpi_has_data:
                ax.set_title(kpi_cfg['title'], fontsize=12, pad=10)
                ax.set_xlabel('Training Steps', fontsize=10)
                ax.set_ylabel('Score', fontsize=10)
                ax.legend(loc='lower right' if kpi in ['comfort', 'resilience'] else 'upper right')
                ax.grid(True, alpha=0.2)
                
                # Improve tick label formatting
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else int(x)))
                
                # Add a subtle border
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#dddddd')
            else:
                # Hide empty subplots
                ax.axis('off')
        
        # Add a main title
        plt.suptitle('KPI Metrics Comparison', fontsize=16, y=1.02)
        
        # Adjust layout with more padding
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save the KPI comparison plot
        kpi_plot_path = os.path.join('results', 'plots', 'kpi_comparison.png')
        plt.savefig(kpi_plot_path, dpi=120, bbox_inches='tight')
        print(f"\nKPI comparison plot saved to: {os.path.abspath(kpi_plot_path)}")
        
        # Show the plot if in interactive mode
        if 'IPython' in sys.modules:
            plt.show()
        else:
            plt.close(fig)
        
        print("\nAll plots generated successfully!")
        
    except Exception as e:
        print(f"\nError generating plots: {str(e)}")
        import traceback
        traceback.print_exc()
        if not eval_dfs:
            print("No evaluation data available for KPI comparison")
        raise
            
    except ImportError as e:
        print(f"Required libraries not found: {e}")
    except Exception as e:
        print(f"Error in plot_training_results: {str(e)}")
    finally:
        # Ensure all figures are closed
        plt.close('all')

def main_multi_kpi_training(num_buildings: int, timesteps_per_episode: int) -> PPO:
    """
    Trains a PPO agent focusing on multiple KPIs (comfort, emissions, grid impact, and resilience).
    
    Args:
        num_buildings: Number of buildings in the environment
        timesteps_per_episode: Number of timesteps per episode
        
    Returns:
        PPO: Trained PPO agent
    """
    # Import EvalCallback here to ensure it's in scope
    from stable_baselines3.common.callbacks import EvalCallback
    
    print("\n=== Starting Multi-KPI Training Phase ===")
    
    # Set fixed random seed for reproducibility
    set_seed(SEED)
    
    # Define weights for multi-KPI training (comfort, emissions, grid impact, resilience)
    multi_kpi_weights = PhaseWeights(
        w1=0.3,  # Comfort
        w2=0.1,  # Emissions
        w3=0.3,  # Grid impact
        w4=0.3   # Resilience
    )
    
    # Ensure directories exist
    os.makedirs(TENSORBOARD_LOG_PATH_MULTI_BASE, exist_ok=True)
    os.makedirs(os.path.join('results', 'models'), exist_ok=True)
    os.makedirs(os.path.join('results', 'logs'), exist_ok=True)

    # --- Environment Setup ---
    print(f"Setting up CustomCityEnv for Multi-KPI training (Buildings: {num_buildings}, Timesteps/Episode: {timesteps_per_episode})...")
    env_multi = CustomCityEnv(
        phase_weights=multi_kpi_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    # Set environment seed for reproducibility
    env_multi.seed(SEED)
    env_multi.reset()

    # --- Agent Training (Multi-KPIs) ---
    print(f"Instantiating PPO agent for Multi-KPI training (TensorBoard logs: {TENSORBOARD_LOG_PATH_MULTI_BASE})...")
    
    # Enhanced policy network configuration for multi-KPI training
    policy_kwargs = {
        'net_arch': {
            'pi': [512, 512, 256],  # Deeper policy network
            'vf': [512, 512, 256]   # Deeper value network
        },
        'activation_fn': torch.nn.ReLU,
        'ortho_init': True,
        'share_features_extractor': False,  # Separate feature extractors for policy and value
        'log_std_init': 0.0,  # Initial log standard deviation for action distribution
        'full_std': True,
        'squash_output': False
    }
    
    # Initialize PPO model with enhanced hyperparameters for better learning
    model = PPO(
        "MlpPolicy",
        env_multi,
        n_steps=2048,         # Reduced n_steps
        batch_size=512,       # Kept batch_size
        n_epochs=10,          # Reduced n_epochs
        gamma=0.99,           # Standardized gamma
        gae_lambda=0.95,      # Standardized gae_lambda
        clip_range_vf=0.2,    # Matched to clip_range
        ent_coef=0.01,        # Reduced ent_coef
        vf_coef=0.5,          # Standardized vf_coef
        max_grad_norm=0.5,    # Kept max_grad_norm
        use_sde=False,
        target_kl=0.01,       # Kept target_kl
        tensorboard_log=TENSORBOARD_LOG_PATH_MULTI_BASE,
        policy_kwargs=policy_kwargs,
        verbose=2,
        device='auto',
        normalize_advantage=True,
        seed=SEED,
        learning_rate=2.5e-4, # Constant learning rate
        clip_range=0.2        # Constant clip range
    )
    
    # --- Callbacks ---
    # Create evaluation environment
    eval_env = CustomCityEnv(
        phase_weights=multi_kpi_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    eval_env.seed(SEED)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join('results', 'models'),
        log_path=os.path.join('results', 'evaluation'),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Custom callback for additional logging
    class CustomCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.log_freq = 1000  # Log every 1000 steps
            
        def _on_step(self) -> bool:
            # Log additional metrics at specified frequency
            if self.n_calls % self.log_freq == 0:
                # Track episode rewards and lengths
                if len(self.model.ep_info_buffer) > 0:
                    for info in self.model.ep_info_buffer:
                        if 'r' in info:
                            self.episode_rewards.append(info['r'])
                        if 'l' in info:
                            self.episode_lengths.append(info['l'])
                
                # Log mean reward and length
                if len(self.episode_rewards) > 0:
                    self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
                if len(self.episode_lengths) > 0:
                    self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))
                
                # Log training metrics
                if hasattr(self.model, 'log_std'):
                    self.logger.record('train/entropy', self.model.log_std.exp().mean().item())
                
                # Log learning rate and clip range
                if hasattr(self.model, 'lr_schedule'):
                    self.logger.record('train/learning_rate', self.model.lr_schedule(self.model._current_progress_remaining))
                if hasattr(self.model, 'clip_range'):
                    clip_range = self.model.clip_range(self.model._current_progress_remaining)
                    if isinstance(clip_range, (list, tuple)):
                        clip_range = clip_range[0]
                    self.logger.record('train/clip_range', clip_range)
                
                # Log explained variance if available
                if hasattr(self.model, 'explained_variance'):
                    self.logger.record('train/explained_variance', self.model.explained_variance)
                
                # Log to console
                print(f"Step: {self.num_timesteps}")
                if len(self.episode_rewards) > 0:
                    print(f"Last 100 episodes mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")
                
                # Force log to TensorBoard
                self.logger.dump(self.num_timesteps)
                
            return True
    
    # Progress bar callback
    class ProgressBarCallback(BaseCallback):
        def __init__(self, total_timesteps):
            super(ProgressBarCallback, self).__init__()
            self.progress_bar = None
            self.total_timesteps = total_timesteps
            
        def _on_training_start(self):
            self.progress_bar = tqdm.tqdm(total=self.total_timesteps, desc="Training progress")
            
        def _on_step(self) -> bool:
            if self.progress_bar is not None:
                self.progress_bar.update(1)  # Update progress by 1 step
            return True
            
        def _on_training_end(self):
            if self.progress_bar is not None:
                self.progress_bar.close()
                self.progress_bar = None
    
    # Combine all callbacks
    callbacks = [
        eval_callback,
        CustomCallback(),
        ProgressBarCallback(TOTAL_TIMESTEPS_MULTI)
    ]
    
    # --- Training ---
    print(f"Starting Multi-KPI training for {TOTAL_TIMESTEPS_MULTI} timesteps...")
    
    try:
        # Evaluate initial random policy
        print("\n=== Evaluating Initial Random Policy ===")
        initial_rewards = []
        for _ in range(5):  # Run 5 episodes to get a good estimate
            obs = env_multi.reset()
            done = False
            total_reward = 0
            while not done:
                action = env_multi.action_space.sample()  # Random action
                obs, reward, done, _ = env_multi.step(action)
                total_reward += reward
            initial_rewards.append(total_reward)
        initial_avg_reward = np.mean(initial_rewards)
        print(f"Initial average reward (random policy): {initial_avg_reward:.2f}")

        # Train the model
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS_MULTI,
            callback=callbacks,
            progress_bar=True,
            tb_log_name="PPO_MultiKPI"
        )
            
        # Save the final model
        model_path = os.path.join('results', 'models', 'ppo_multi_kpi_model')
        print(f"Multi-KPI Training finished. Saving model to {model_path} ...")
        model.save(model_path)
            
        # Evaluate final policy
        print("\n=== Evaluating Trained Policy ===")
        final_rewards = []
        for _ in range(5):  # Run 5 episodes to get a good estimate
            obs = eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                total_reward += reward
            final_rewards.append(total_reward)
        final_avg_reward = np.mean(final_rewards)
        
        # Calculate improvement
        improvement = ((final_avg_reward - initial_avg_reward) / (abs(initial_avg_reward) + 1e-10)) * 100
        
        # Print training summary
        print("\n=== Training Summary ===")
        print(f"Initial Average Reward (Random Policy): {initial_avg_reward:.2f}")
        print(f"Final Average Reward (Trained Policy): {final_avg_reward:.2f}")
        print(f"Improvement: {improvement:+.2f}%")
        print("=====================")
        print(f"\nMulti-KPI Training complete! Model saved to {model_path}")
        
        # Save the final evaluation results
        eval_results = []
        obs = eval_env.reset()
        for _ in range(5):  # Run 5 evaluation episodes
            done = False
            episode_reward = 0
            obs = eval_env.reset()
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
            eval_results.append(episode_reward)
        
        print(f"\nEvaluation Results (5 episodes):")
        print(f"Mean Reward: {np.mean(eval_results):.2f} ± {np.std(eval_results):.2f}")
        
        return model
        
    except Exception as e:
        print(f"\nError during Multi-KPI training: {str(e)}")
        raise
    finally:
        # Clean up
        env_multi.close()
        eval_env.close()

def main_2kpi_training(num_buildings: int, timesteps_per_episode: int) -> PPO:
    """
    Trains a PPO agent focusing on 2 KPIs (comfort and emissions).
    
    Args:
        num_buildings: Number of buildings in the environment
        timesteps_per_episode: Number of timesteps per episode
        
    Returns:
        PPO: Trained PPO agent
    """
    import numpy as np  # Import numpy for numerical operations
    import torch as th  # Import PyTorch with th alias for activation functions
    # Import EvalCallback here to ensure it's in scope
    from stable_baselines3.common.callbacks import EvalCallback
    
    print("\n=== Starting 2-KPI Training Phase ===")
    
    # Set fixed random seed for reproducibility
    set_seed(SEED)
    
    # Set up weights for comfort and emissions only, with higher weight on comfort
    comfort_emissions_weights = PhaseWeights(w1=0.8, w2=0.2, w3=0.0, w4=0.0)  # Focus more on comfort (w1)
    
    # Ensure directories exist
    os.makedirs(TENSORBOARD_LOG_PATH_2KPI_BASE, exist_ok=True)
    os.makedirs(os.path.join('results', 'models'), exist_ok=True)
    os.makedirs(os.path.join('results', 'logs'), exist_ok=True)

    # --- Environment Setup ---
    print(f"Setting up CustomCityEnv for 2-KPI training (Buildings: {num_buildings}, Timesteps/Episode: {timesteps_per_episode})...")
    env_2kpi = CustomCityEnv(
        phase_weights=comfort_emissions_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    # Set environment seed for reproducibility
    env_2kpi.seed(SEED)
    env_2kpi.reset()

    # --- Agent Training (2-KPIs) ---
    print(f"\nInstantiating PPO agent for 2-KPI training (TensorBoard logs: {TENSORBOARD_LOG_PATH_2KPI_BASE})...")
    
    # Enhanced network architecture with better exploration
    policy_kwargs = {
        'net_arch': {
            'pi': [256, 256],  # Deeper policy network
            'vf': [256, 256]   # Deeper value network
        },
        'activation_fn': th.nn.ReLU,
        'ortho_init': True,
        'log_std_init': 0.0,  # Changed log_std_init
        'share_features_extractor': False,
        'optimizer_kwargs': {
            'weight_decay': 1e-5,
            'eps': 1e-5
        }
    }
    
    # Enhanced PPO agent with optimized hyperparameters for 2-KPI training
    model_2kpi = PPO(
        'MlpPolicy',
        env_2kpi,
        n_steps=2048,             # Reduced n_steps
        batch_size=256,           # Kept batch_size
        n_epochs=10,              # Kept n_epochs
        gamma=0.99,               # Standardized gamma
        gae_lambda=0.95,          # Kept gae_lambda
        clip_range=0.2,           # Kept clip_range
        clip_range_vf=0.2,        # Kept clip_range_vf
        ent_coef=0.01,            # Reduced ent_coef
        vf_coef=0.5,              # Standardized vf_coef
        max_grad_norm=0.5,        # Standardized max_grad_norm
        policy_kwargs=policy_kwargs,
        tensorboard_log=TENSORBOARD_LOG_PATH_2KPI_BASE,
        verbose=2,
        normalize_advantage=True,
        target_kl=0.02,           # Kept target_kl
        seed=SEED,
        stats_window_size=20,
        use_sde=False,
        learning_rate=2.5e-4      # Constant learning rate
    )

    print(f"Starting 2-KPI training for {TOTAL_TIMESTEPS_2KPI} timesteps...")
    
    # Setup evaluation environment
    eval_env = CustomCityEnv(
        phase_weights=comfort_emissions_weights,
        num_buildings=num_buildings,
        timesteps_per_episode=timesteps_per_episode
    )
    
    # Create necessary directories
    os.makedirs(os.path.join('results', 'models'), exist_ok=True)
    os.makedirs(os.path.join('results', 'logs'), exist_ok=True)
    
    # Enhanced evaluation callback with more frequent evaluation and better logging
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join('results', 'models', 'best_2kpi'),
        log_path=os.path.join('results', 'logs', '2kpi_eval'),
        eval_freq=max(5000 // num_buildings, 1),  # More frequent evaluation
        deterministic=False,  # Use stochastic actions for evaluation for better exploration
        render=False,
        n_eval_episodes=10,  # More episodes for more stable metrics
        warn=False,
        # Add custom eval callback for additional metrics
        callback_on_new_best=None,
        # Log additional info
        verbose=1
    )
    
    # Add a simple custom callback for logging
    from stable_baselines3.common.callbacks import BaseCallback
    import numpy as np
    
    class CustomCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.log_freq = 1000  # Log every 1000 steps
            
        def _on_step(self) -> bool:
            # Log additional metrics at specified frequency
            if self.n_calls % self.log_freq == 0:
                # Track episode rewards and lengths
                if len(self.model.ep_info_buffer) > 0:
                    for info in self.model.ep_info_buffer:
                        if 'r' in info:
                            self.episode_rewards.append(info['r'])
                        if 'l' in info:
                            self.episode_lengths.append(info['l'])
                
                # Log mean reward and length
                if len(self.episode_rewards) > 0:
                    self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
                    self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))
                
                # Log learning rate
                if hasattr(self.model, 'lr_schedule'):
                    if callable(self.model.lr_schedule):
                        lr = self.model.lr_schedule(1.0 - (self.num_timesteps / self.model._total_timesteps))
                    else:
                        lr = self.model.lr_schedule
                    self.logger.record('train/learning_rate', lr)
                
                # Log clip range
                if hasattr(self.model, 'clip_range'):
                    if callable(self.model.clip_range):
                        clip_range = self.model.clip_range(1.0 - (self.num_timesteps / self.model._total_timesteps))
                    else:
                        clip_range = self.model.clip_range
                    self.logger.record('train/clip_range', clip_range)
                
                # Log explained variance if available
                if hasattr(self.model, 'explained_variance'):
                    self.logger.record('train/explained_variance', self.model.explained_variance)
                
                # Log to console
                print(f"Step: {self.num_timesteps}")
                if len(self.episode_rewards) > 0:
                    print(f"Last 100 episodes mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")
                
                # Force log to TensorBoard
                self.logger.dump(self.num_timesteps)
                
            return True
    
    # Combine callbacks
    callbacks = [eval_callback, CustomCallback()]
    
    # Start the training with progress bar
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50 + "\n")
    
    try:
        # Train the model
        model_2kpi.learn(
            total_timesteps=TOTAL_TIMESTEPS_2KPI,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=True,
            tb_log_name="PPO_2KPI"
        )
        
        # Save the final model
        model_path = os.path.join('results', 'models', 'ppo_2kpi_final')
        model_2kpi.save(model_path)
        print(f"\nTraining completed. Model saved to {model_path}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Plot training and evaluation results
    plot_training_results()
    
    return model_2kpi  # Return the trained model

def compare_training_results():
    """Compare and visualize results from 2-KPI and Multi-KPI training."""
    try:
        # Define file paths
        eval_results_2kpi = EVAL_RESULTS_CSV_2KPI
        
        # Check if 2-KPI results exist
        if not os.path.exists(eval_results_2kpi):
            print(f"2-KPI evaluation results not found at {eval_results_2kpi}")
            return
            
        # Load and print 2-KPI results
        results_2kpi = pd.read_csv(eval_results_2kpi)
        print("\n=== 2-KPI Training Results ===")
        print(f"Mean Reward: {results_2kpi['reward'].mean():.4f} ± {results_2kpi['reward'].std():.4f}")
        print(f"Mean Normalized Reward: {results_2kpi['normalized_reward'].mean():.4f} ± {results_2kpi['normalized_reward'].std():.4f}")
        
        # Check for Multi-KPI results
        eval_results_multi = EVAL_RESULTS_CSV_MULTI
        if os.path.exists(eval_results_multi):
            results_multi = pd.read_csv(eval_results_multi)
            print("\n=== Multi-KPI Training Results ===")
            print(f"Mean Reward: {results_multi['reward'].mean():.4f} ± {results_multi['reward'].std():.4f}")
            print(f"Mean Normalized Reward: {results_multi['normalized_reward'].mean():.4f} ± {results_multi['normalized_reward'].std():.4f}")
        
    except Exception as e:
        print(f"Error comparing training results: {str(e)}")

if __name__ == "__main__":
    # Set up logging
    os.makedirs(os.path.join('results', 'logs'), exist_ok=True)
    os.makedirs(os.path.join('results', 'models'), exist_ok=True)
    os.makedirs(os.path.join('results', 'evaluation'), exist_ok=True)
    
    # Training parameters
    NUM_BUILDINGS_MAIN = 1  # Number of buildings to train on
    TIMESTEPS_PER_EPISODE_MAIN = 24 * 7  # One week of hourly timesteps
    TOTAL_TIMESTEPS_2KPI = 400000  # Total number of timesteps for 2-KPI training
    TOTAL_TIMESTEPS_MULTI = 400000  # Total number of timesteps for Multi-KPI training
    ENABLE_MULTI_KPI_TRAINING = True  # Enable multi-KPI training
    
    # Generate a random seed for reproducibility
    SEED = random.randint(0, 10000)  # Random seed between 0 and 9999
    set_seed(SEED)
    
    print("\n" + "="*50)
    print("Starting Training with Fixed Random Seeds")
    print(f"Using random seed: {SEED}")
    print("="*50 + "\n")
    
    # Clean up previous runs
    cleanup_previous_runs()
    
    # ... (rest of the code remains the same)
    print("\n" + "="*50)
    print("STARTING 2-KPI TRAINING PHASE")
    print("="*50)
    
    model_2kpi = main_2kpi_training(
        num_buildings=NUM_BUILDINGS_MAIN,
        timesteps_per_episode=TIMESTEPS_PER_EPISODE_MAIN
    )
    
    # After 2-KPI training completes, check if multi-KPI training is enabled
    if ENABLE_MULTI_KPI_TRAINING:
        print("\n" + "=" * 50)
        print("MULTI-KPI TRAINING")
        print("=" * 50 + "\n")

        try:
            model_multi_kpi = main_multi_kpi_training(
                num_buildings=NUM_BUILDINGS_MAIN,
                timesteps_per_episode=TIMESTEPS_PER_EPISODE_MAIN
            )

            # Save the final multi-KPI model if training was successful
            model_multi_kpi.save(os.path.join('results', 'models', 'ppo_multi_kpi_model'))
            print("Multi-KPI Training finished. Model saved to results/models/ppo_multi_kpi_model.zip")
        except NameError:
            print("Skipping Multi-KPI training: main_multi_kpi_training function not implemented.")
        except Exception as e:
            print(f"Error during Multi-KPI training: {str(e)}")
    else:
        print("\nMulti-KPI training is disabled. Set ENABLE_MULTI_KPI_TRAINING = True to enable it.")

    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print("\nAll training phases complete.")
