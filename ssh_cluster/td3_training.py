#!/usr/bin/env python3
"""
TD3 Training Script for CityLearn Environment

This script trains a TD3 agent on the CityLearn environment.

To set up the environment, run: bash setup.sh
"""

import os
import sys
import subprocess
import platform
import gym
import numpy as np
import torch
import json
import time
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gym')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set matplotlib to use 'Agg' to avoid display issues in headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Check for required packages
try:
    import tensorboard
except ImportError:
    print("Warning: TensorBoard is not installed. Logging will be limited.")
    TENSORBOARD_AVAILABLE = False
else:
    TENSORBOARD_AVAILABLE = True
    from torch.utils.tensorboard import SummaryWriter

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'training.log'))
    ]
)
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for optimal performance

import matplotlib
# Use Agg backend for non-interactive plotting (for server/SSH)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import StableBaselines3Wrapper

try:
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
    from citylearn.citylearn import CityLearnEnv
    from citylearn.wrappers import StableBaselines3Wrapper
except ImportError as e:
    print("Error: Required packages not found. Please run 'bash setup.sh' first.")
    print(f"Missing package: {e}")
    sys.exit(1)

class RewardScaler(gym.Wrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important for good performance.
    """
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward * 0.01, done, info

    def reset(self):
        return self.env.reset()

# Custom callback for logging and saving models
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                # Try to load results
                monitor_data = load_results(self.log_dir)
                x, y = ts2xy(monitor_data, 'timesteps')
                if len(x) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    # Save the best model
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.model.save(os.path.join(self.log_dir, 'best_model'))
                    # Save the current model
                    self.model.save(os.path.join(self.log_dir, 'latest_model'))
            except Exception as e:
                # If monitor files aren't available yet, just continue
                if "No monitor files" in str(e):
                    pass
                else:
                    print(f"Error in callback: {e}")
                    
                    # Plot results
                    plot_results([self.log_dir], 1000, results_plotter.X_TIMESTEPS, "TD3 CityLearn")
                    plt.savefig(os.path.join(self.log_dir, 'training_results.png'))
                    plt.close()
        return True

def plot_results(log_folders, num_timesteps, x_axis, title):
    """
    Plot the results
    """
    plt.figure(figsize=(12, 6))
    
    for folder in log_folders:
        x, y = ts2xy(load_results(folder), x_axis)
        # Limit to num_timesteps
        if len(x) > 0:
            plt.plot(x, y, label=folder.split('/')[-1])
    
    plt.title(title)
    plt.legend()
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.grid(True)
    plt.tight_layout()

# Set up logging
log_dir = "logs/td3_citylearn_" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(log_dir, exist_ok=True)

# Set random seeds for reproducibility
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Save hyperparameters
hyperparams = {
    'learning_rate': 0.0003,
    'buffer_size': 1000000,
    'batch_size': 100,
    'gamma': 0.99,
    'tau': 0.005,
    'policy_kwargs': dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
}

with open(os.path.join(log_dir, 'hyperparams.json'), 'w') as f:
    json.dump(hyperparams, f, indent=4)

# Load environment
dataset_name = 'citylearn_challenge_2022_phase_all'

# Initialize the CityLearn environment
env = CityLearnEnv(dataset_name, central_agent=True)

# The noise objects for TD3
n_actions = env.action_space[0].shape[0]  # Get number of actions from the first building
print(f"Number of actions: {n_actions}")

action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), 
    sigma=0.1 * np.ones(n_actions)
)

# Wrap environment for Stable Baselines3
env = StableBaselines3Wrapper(env)

# Verify the environment
print("Environment wrapped successfully")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Create a simple evaluation environment that doesn't use Monitor
eval_env = StableBaselines3Wrapper(CityLearnEnv(dataset_name, central_agent=True))

# Optimized hyperparameters for better learning
hyperparams.update({
    'learning_rate': 3e-4,  # Slightly higher learning rate for faster learning
    'buffer_size': 200000,  # Larger replay buffer for more stable training
    'learning_starts': 5000,  # Collect more samples before starting to learn
    'batch_size': 512,  # Larger batch size for more stable updates
    'tau': 0.005,  # Target network update rate (standard for TD3)
    'gamma': 0.99,  # Discount factor (standard for most RL tasks)
    'train_freq': (1, "step"),
    'gradient_steps': 1,  # Update every step
    'policy_kwargs': dict(
        net_arch=dict(pi=[400, 300], qf=[400, 300]),  # Larger networks (standard for TD3)
        activation_fn=torch.nn.ReLU,
    ),
    'policy_delay': 2,  # TD3 specific: Policy update frequency
    'target_policy_noise': 0.2,  # Noise added to target policy
    'target_noise_clip': 0.5,  # Range to clip target policy noise
})

# Initialize the TD3 agent
model = TD3(
    "MlpPolicy",
    env,
    **hyperparams,
    action_noise=action_noise,
    verbose=1,
    seed=RANDOM_SEED,
    tensorboard_log=log_dir
)

# Create the callback
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# Enhanced exploration with adaptive noise
n_actions = env.action_space.shape[-1]

# Start with higher noise and decay it over time
initial_noise = 0.5
final_noise = 0.1
noise_decay = 0.9999

def get_action_noise(step, total_steps):
    # Linear decay from initial_noise to final_noise
    progress = min(step / total_steps, 1.0)
    current_noise = initial_noise * (1 - progress) + final_noise * progress
    
    # Add some temporal correlation for smoother exploration
    return OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=current_noise * np.ones(n_actions),
        theta=0.15,
        dt=1e-2,
    )

# Initialize with starting noise
hyperparams['action_noise'] = get_action_noise(0, 10000)

# Enhanced reward scaling with clipping to prevent exploding gradients
class RewardScaler(gym.Wrapper):
    """Scale and shape rewards for better learning"""
    def __init__(self, env, scale=0.001, clip=10.0, shift=0.0):
        super().__init__(env)
        self.scale = scale
        self.clip = clip
        self.shift = shift
        self.episode_reward = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        
        # Scale and shift the reward
        scaled_reward = (reward + self.shift) * self.scale
        
        # Clip to avoid extreme values
        clipped_reward = np.clip(scaled_reward, -self.clip, self.clip)
        
        # Add small penalty for large actions to encourage energy efficiency
        action_penalty = 0.01 * np.mean(np.square(action))
        final_reward = clipped_reward - action_penalty
        
        if done:
            # Optional: Add terminal reward based on episode performance
            terminal_reward = -0.1 * np.abs(self.episode_reward)
            final_reward += terminal_reward
            self.episode_reward = 0
            
        return obs, final_reward, done, info

    def reset(self):
        self.episode_reward = 0
        return self.env.reset()

# Apply enhanced reward scaling
env = RewardScaler(env, scale=0.001, clip=1.0)

# For CityLearn, we'll use a custom evaluation function instead of EvalCallback
# since the environment doesn't work well with the default Monitor wrapper
class CityLearnEvalCallback(BaseCallback):
    """
    A custom callback that evaluates the model periodically.
    """
    def __init__(self, eval_env, eval_freq=1000, deterministic=True, verbose=0):
        super(CityLearnEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.best_model_path = os.path.join(log_dir, 'best_model')
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Run evaluation
            mean_reward, _, _ = evaluate_model(
                self.model, 
                self.eval_env,
                num_episodes=1,
                deterministic=self.deterministic
            )
            
            # Save the best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.best_model_path)
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f} - Saving best model to {self.best_model_path}")
            
            # Log the evaluation results
            if self.logger is not None:
                self.logger.record('eval/mean_reward', float(mean_reward))
                self.logger.dump(self.num_timesteps)
                
        return True

# Create the evaluation callback
eval_callback = CityLearnEvalCallback(
    eval_env=eval_env,
    eval_freq=1000,  # Evaluate every 1000 steps
    deterministic=True,
    verbose=1
)

# Train the model
print("Starting training...")
start_time = time.time()

class ProgressCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    """
    def __init__(self, check_freq: int, verbose: int = 1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.last_time = time.time()
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        if self.n_calls % self.check_freq == 0:
            current_time = time.time()
            time_elapsed = current_time - self.last_time
            steps_per_sec = self.check_freq / time_elapsed if time_elapsed > 0 else float('inf')
            self.last_time = current_time
            
            # Get the last 100 rewards from the buffer
            if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                # Extract rewards from the buffer
                rewards = []
                for info in self.model.ep_info_buffer:
                    if isinstance(info, dict) and 'r' in info:
                        rewards.append(info['r'])
                    elif isinstance(info, (int, float)):
                        rewards.append(info)
                
                if rewards:  # Only calculate mean if we have valid rewards
                    mean_reward = np.mean(rewards[-100:])  # Last 100 rewards
                    print(f"Step: {self.num_timesteps}, "
                          f"Mean Reward (last 100): {mean_reward:.2f}, "
                          f"Steps/sec: {steps_per_sec:.2f}")
                else:
                    print(f"Step: {self.num_timesteps}, No valid rewards in buffer yet")
        return True

try:
        # Optimized training parameters for CityLearn
    total_training_steps = 500000  # More training steps for better convergence
    warmup_steps = 10000  # Longer warmup for stability
    save_freq = 10000  # Save model every 10k steps
    
    # Parameters for better learning
    batch_size = 1024  # Reduced batch size for better generalization
    buffer_size = 500000  # Smaller buffer for more recent experiences
    
    # Update model hyperparameters for better learning
    hyperparams.update({
        'device': device,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'optimize_memory_usage': True,
        'learning_starts': 10000,  # Collect more samples before learning
        'train_freq': (1, 'step'),  # Update every step
        'gradient_steps': 1,  # Single gradient step per update
        'learning_rate': 1e-4,  # Lower learning rate for stability
        'tau': 0.005,  # Slower target network updates
        'gamma': 0.99,  # Standard discount factor
        'policy_delay': 2,  # Standard TD3 delay
        'target_policy_noise': 0.2,  # More exploration
        'target_noise_clip': 0.5,  # Standard noise clipping
        'replay_buffer_kwargs': {
            'handle_timeout_termination': False,
        },
        'policy_kwargs': {
            'net_arch': dict(pi=[256, 256], qf=[256, 256])  # Larger networks
        }
    })
    
    # Reinitialize model with updated hyperparameters
    model = TD3(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        **{k: v for k, v in hyperparams.items() if k in [

# Save the final model
model.save(os.path.join(log_dir, 'final_model'))

# Save training metrics with serializable data only
with open(os.path.join(log_dir, 'training_metrics.json'), 'w') as f:
    # Create a serializable version of hyperparameters
    serializable_hyperparams = {}
    for key, value in hyperparams.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            serializable_hyperparams[key] = value
        elif isinstance(value, dict):
            # Handle nested dictionaries
            serializable_hyperparams[key] = {
                k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                for k, v in value.items()
            }
        else:
            # Convert other types to string
            serializable_hyperparams[key] = str(value)
    
    # Save the serializable data
    json.dump({
        'training_time_seconds': training_time,
        'total_timesteps': model.num_timesteps,
        'training_completed': True,
        'hyperparameters': serializable_hyperparams
    }, f, indent=4)

# Save the final model
model.save(os.path.join(log_dir, 'final_model'))

# Save training time
with open(os.path.join(log_dir, 'training_metrics.json'), 'w') as f:
    json.dump({
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'training_time_hours': training_time / 3600
    }, f, indent=4)

# Create evaluation directory
eval_dir = os.path.join(log_dir, 'evaluation')
os.makedirs(eval_dir, exist_ok=True)

# Evaluate the model
def evaluate_model(model, env, num_episodes=2, max_steps=1000):
    """
    Evaluate a RL agent with safety measures against infinite loops
    :param model: (BaseRLModel) the RL Agent
    :param env: (gym.Env) the environment
    :param num_episodes: (int) number of episodes to evaluate
    :param max_steps: (int) maximum steps per episode
    :return: (tuple) mean_reward, std_reward, all_rewards
    """
    episode_rewards = []
    
    for i in range(num_episodes):
        print(f"\nStarting evaluation episode {i+1}/{num_episodes}")
        reset_return = env.reset()
        
        # Handle different return types from reset()
        if isinstance(reset_return, tuple) and len(reset_return) == 2:
            obs, _ = reset_return  # Newer Gym API
        else:
            obs = reset_return  # Older Gym API
            
        episode_reward = 0.0
        step_count = 0
        done = False
        
        while not done and step_count < max_steps:
            try:
                # Get action from the model
                action, _ = model.predict(obs, deterministic=True)
                
                # Step the environment
                step_return = env.step(action)
                
                # Handle different return types from step()
                if len(step_return) == 4:  # Older Gym API
                    obs, reward, done, _ = step_return
                else:  # Newer Gym API
                    obs, reward, terminated, truncated, _ = step_return
                    done = terminated or truncated
                
                # Update episode reward and step count
                episode_reward += float(reward)
                step_count += 1
                
                # Print progress
                if step_count % 100 == 0:
                    print(f"Step {step_count}: Current episode reward = {episode_reward:.2f}")
                
            except Exception as e:
                print(f"Error during evaluation: {e}")
                done = True
        
        # Handle episode end
        if step_count >= max_steps:
            print(f"Episode {i+1} reached max steps ({max_steps})")
        else:
            print(f"Episode {i+1} completed in {step_count} steps")
            
        print(f"Episode {i+1} total reward: {episode_reward:.2f}")
        episode_rewards.append(episode_reward)
    
    # Calculate statistics
    if episode_rewards:
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        print(f"\nEvaluation completed over {len(episode_rewards)} episodes")
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward, episode_rewards
    else:
        print("No episodes completed successfully")
        return 0.0, 0.0, []

# Run evaluation
print("Starting evaluation...")
mean_reward, std_reward, all_rewards = evaluate_model(model, env)

# Save evaluation results
eval_results = {
    'mean_reward': float(mean_reward),
    'std_reward': float(std_reward),
    'all_rewards': [float(r) for r in all_rewards]
}

with open(os.path.join(eval_dir, 'evaluation_results.json'), 'w') as f:
    json.dump(eval_results, f, indent=4)

# Plot training and evaluation results
try:
    # Plot training rewards
    results_df = load_results(log_dir)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(results_df['time'].values, results_df['r'].rolling(window=100).mean())
    plt.title('Training Rewards (Smoothed)')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    
    # Plot evaluation rewards
    plt.subplot(1, 2, 2)
    plt.plot(all_rewards, 'b-')
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(eval_dir, 'training_evaluation_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training and evaluation plot saved to {plot_path}")
    
except Exception as e:
    print(f"Error generating plots: {e}")
plt.ylabel('Reward')
plt.title('Evaluation Results')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, 'evaluation_results.png'))
plt.close()

print(f"Evaluation completed. Results saved to {eval_dir}")

# Close the environment
env.close()

# Save the final plot
plt.figure(figsize=(10, 6))
plt.plot(all_rewards, 'b-', label='Evaluation Reward')
plt.title('Evaluation Rewards Over Time')
plt.xlabel('Evaluation Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_filename = f'evaluation_results_{timestamp}.png'
plt.savefig(plot_filename)
plt.close()

print(f"\nTraining completed! Evaluation results saved to: {os.path.abspath(plot_filename)}")
print("To view the plot, you can download it from the server using scp or view it in a Jupyter notebook.")

# Print instructions for remote visualization
if 'SSH_CONNECTION' in os.environ:
    print("\nSince you're connected via SSH, you can download the plot using:")
    print(f"scp {os.getlogin()}@{platform.node()}:{os.path.abspath(plot_filename)} .")
