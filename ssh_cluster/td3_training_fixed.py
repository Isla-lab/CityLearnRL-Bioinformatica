#!/usr/bin/env python3
"""
TD3 Training Script for CityLearn Environment

This script trains a TD3 agent on the CityLearn environment with proper error handling and logging.
"""

import os
import sys
import time
import json
import logging
import warnings
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np
import torch
import gym
from gym import spaces

# Configure device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from typing import Dict, List, Tuple, Union, Optional
from collections import OrderedDict

# Dummy SummaryWriter implementation
class DummySummaryWriter:
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, *args, **kwargs):
        return lambda *args, **kwargs: None
    def add_scalar(self, *args, **kwargs):
        pass
    def close(self):
        pass

# Global flag for tensorboard availability
TENSORBOARD_AVAILABLE = False
SummaryWriter = DummySummaryWriter

class CityLearnSB3Wrapper(gym.Env):
    """
    Wrapper for CityLearn environment to make it compatible with Stable Baselines3.
    This wrapper converts the multi-agent environment into a single-agent environment
    by concatenating observations and actions.
    """
    def __init__(self, env):
        self.env = env
        self.buildings = env.buildings
        
        # Calculate combined observation and action space sizes
        obs_shapes = [space.shape[0] for space in env.observation_space]
        action_shapes = [space.shape[0] for space in env.action_space]
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(sum(obs_shapes),), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0, high=1,
            shape=(sum(action_shapes),),
            dtype=np.float32
        )
        
        self.observation_shapes = obs_shapes
        self.action_shapes = action_shapes
        
    def reset(self):
        obs_list = self.env.reset()
        return self._flatten_obs(obs_list)
    
    def step(self, action):
        # Split the action into individual building actions
        actions = []
        start_idx = 0
        for shape in self.action_shapes:
            actions.append(action[start_idx:start_idx + shape])
            start_idx += shape
        
        # Step the environment
        obs_list, reward, done, info = self.env.step(actions)
        
        # Flatten observations and sum rewards
        flat_obs = self._flatten_obs(obs_list)
        total_reward = sum(reward) if isinstance(reward, (list, tuple)) else reward
        
        return flat_obs, total_reward, done, info
    
    def _flatten_obs(self, obs_list):
        return np.concatenate([np.asarray(obs).flatten() for obs in obs_list])
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def close(self):
        return self.env.close()

def wrap_citylearn(env):
    """Wrap the CityLearn environment for SB3 compatibility"""
    return CityLearnSB3Wrapper(env)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gym')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# TensorBoard logging is disabled - using dummy implementation

class ProgressCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    
    :param check_freq: (int) Number of steps between logging FPS
    :param verbose: (int) Verbosity level: 0 for no output, 1 for info messages
    """
    def __init__(self, check_freq=1000, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.last_time = time.time()
        self.last_step = 0
        
    def _on_training_start(self) -> None:
        """This method is called before the first training step."""
        self.last_time = time.time()
        self.last_step = self.model.num_timesteps
        if self.verbose > 0:
            logger.info("Starting training progress tracking...")
    
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        
        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.n_calls % self.check_freq == 0:
            current_time = time.time()
            steps = self.model.num_timesteps - self.last_step
            if steps > 0 and current_time > self.last_time:
                fps = steps / (current_time - self.last_time)
                if self.verbose > 0:
                    logger.info(f"Step: {self.model.num_timesteps}, FPS: {fps:.1f}")
            self.last_time = current_time
            self.last_step = self.model.num_timesteps
        return True

def train_td3(env, eval_env, hyperparams, log_dir, total_steps=100000, warmup_steps=10000):
    """
    Train a TD3 model on the CityLearn environment
    
    Args:
        env: Training environment
        eval_env: Evaluation environment
        hyperparams: Dictionary of hyperparameters
        log_dir: Directory to save logs and models
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps for learning rate
        
    Returns:
        tuple: (success, training_time_seconds, model_path)
    """
    start_time = time.time()
    writer = None
    model = None
    
    try:
        # Set up TensorBoard writer (will be a dummy if not available)
        tb_log_dir = os.path.join(log_dir, 'tensorboard')
        os.makedirs(tb_log_dir, exist_ok=True)
        
        # Initialize the dummy writer - tensorboard is disabled
        writer = DummySummaryWriter()
            
        # Set random seeds for reproducibility
        seed = 42
        if hasattr(env, 'seed'):
            env.seed(seed)
        if hasattr(eval_env, 'seed'):
            eval_env.seed(seed)
        set_random_seed(seed, using_cuda=False)
            
        # Handle multi-agent action spaces
        action_space = env.action_space
        if isinstance(action_space, list):
            logger.info(f"Training with {len(action_space)} agents")
            logger.info(f"Action spaces: {[space.shape for space in action_space]}")
        
        # Log device information
        if torch.cuda.is_available():
            logger.info("CUDA is available. PyTorch will automatically use GPU if configured.")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        else:
            logger.warning("CUDA is not available. Training will use CPU.")
        logger.info(f"PyTorch Version: {torch.__version__}")
        
        # Configure PyTorch for better performance if GPU is available
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Get policy kwargs without device specification
        policy_kwargs = hyperparams.get('policy_kwargs', {})
        hyperparams['policy_kwargs'] = policy_kwargs
        
        logger.info("Initializing TD3 model...")
        
        # Create the model - SB3 will automatically use GPU if available
        model = TD3(
            "MultiInputPolicy" if isinstance(env.observation_space, list) else "MlpPolicy",
            env, 
            verbose=1,
            tensorboard_log=os.path.join(log_dir, 'tb_logs') if TENSORBOARD_AVAILABLE else None,
            **hyperparams
        )
        
        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, 'best_model'),
            log_path=log_dir,
            eval_freq=5000,
            deterministic=True,
            render=False,
            n_eval_episodes=1,
            warn=False,  # Disable warnings about training/eval mode
            verbose=1
        )
        
        # Fix for total_timesteps in eval callback
        eval_callback._init_callback = lambda: None  # Override the init to prevent resetting timesteps
        
        progress_callback = ProgressCallback(check_freq=500, verbose=1)
        
        # Initialize callbacks
        callback_list = [eval_callback, progress_callback]
        for callback in callback_list:
            callback.init_callback(model)
        
        # Training loop
        logger.info(f"Starting training for {total_steps} steps...")
        
        # Create a combined callback that handles both evaluation and progress tracking
        class CombinedCallback(BaseCallback):
            def __init__(self, eval_callback, progress_callback):
                super(CombinedCallback, self).__init__()
                self.eval_callback = eval_callback
                self.progress_callback = progress_callback
                
            def _on_training_start(self) -> None:
                self.eval_callback.init_callback(self.model)
                self.progress_callback.init_callback(self.model)
                
            def _on_step(self) -> bool:
                # Apply learning rate warmup
                if self.model.num_timesteps < warmup_steps:
                    lr_scale = self.model.num_timesteps / warmup_steps
                    current_lr = hyperparams['learning_rate'] * lr_scale
                    for param_group in self.model.actor.optimizer.param_groups + self.model.critic.optimizer.param_groups:
                        param_group['lr'] = current_lr
                
                # Call both callbacks
                self.eval_callback._on_step()
                self.progress_callback._on_step()
                
                # Save model periodically
                if self.model.num_timesteps % 10000 == 0:
                    model_path = os.path.join(log_dir, f'model_step_{self.model.num_timesteps}')
                    self.model.save(model_path)
                    logger.info(f"Model saved to {model_path}")
                    
                    # Log to TensorBoard if available
                    if writer is not None and hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                        for info in self.model.ep_info_buffer:
                            if 'r' in info:
                                writer.add_scalar('train/episode_reward', info['r'], self.model.num_timesteps)
                
                return True
        
        # First, test the environment to make sure it steps properly
        logger.info("Testing environment step...")
        obs = env.reset()
        logger.info(f"Environment reset successful. Observation shape: {np.array(obs).shape}")
        
        # Initialize reward tracking
        initial_reward = None
        episode_rewards = []
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            # Track initial reward if not set
            if initial_reward is None:
                initial_reward = reward
            
            # Calculate reward increment
            reward_increment = reward - initial_reward if initial_reward is not None else 0
            
            logger.info(f"Step {i+1}: "
                       f"Reward: {reward:.2f}, "
                       f"Increment: {reward_increment:+.2f}, "
                       f"Done: {done}, "
                       f"Obs shape: {np.array(obs).shape}")
            
            if done:
                obs = env.reset()
                logger.info("Environment reset after done")
        
        logger.info("Environment test completed. Starting training...")
        
        # Initialize the combined callback
        combined_callback = CombinedCallback(eval_callback, progress_callback)
        combined_callback.init_callback(model)
        
        # Train the model with the combined callback
        with tqdm(total=total_steps, desc="Training Progress", unit="step") as pbar:
            # Custom callback for tracking rewards and progress
            class RewardTrackingCallback(BaseCallback):
                def __init__(self, pbar, initial_reward=None):
                    super(RewardTrackingCallback, self).__init__()
                    self.pbar = pbar
                    self.last_step = 0
                    self.episode_rewards = []
                    self.episode_steps = []
                    self.start_time = time.time()
                    self.initial_reward = initial_reward
                    self.current_episode_reward = 0
                    self.current_episode_steps = 0
                    self.best_reward = -np.inf
                    logger.info("RewardTrackingCallback initialized")
                
                def _on_step(self) -> bool:
                    # Get current reward from the environment
                    if len(self.model.ep_info_buffer) > 0 and 'r' in self.model.ep_info_buffer[0]:
                        reward = self.model.ep_info_buffer[0]['r']
                        self.current_episode_reward += reward
                        self.current_episode_steps += 1
                        
                        # Track best reward
                        if reward > self.best_reward:
                            self.best_reward = reward
                        
                        # Calculate reward increment from initial
                        reward_increment = reward - self.initial_reward if self.initial_reward is not None else 0
                        
                        # Log the reward information
                        print(f"\nStep {self.num_timesteps}: "
                              f"Reward: {reward:10.2f} | "
                              f"Increment: {reward_increment:10.2f} | "
                              f"Best: {self.best_reward:10.2f} | "
                              f"Steps: {self.current_episode_steps}")
                    
                    # Update progress bar
                    current_step = self.model.num_timesteps
                    steps_done = current_step - self.last_step
                    if steps_done > 0:
                        self.pbar.update(steps_done)
                        self.last_step = current_step
                    
                    return True
                
                def _on_training_start(self) -> None:
                    logger.info("Training started")
                    self.start_time = time.time()
                
                def _on_rollout_end(self) -> None:
                    # Log at the end of each rollout
                    if self.current_episode_steps > 0:
                        avg_reward = self.current_episode_reward / max(1, self.current_episode_steps)
                        logger.info(f"Rollout completed - "
                                  f"Avg Reward: {avg_reward:.2f} | "
                                  f"Total Steps: {self.current_episode_steps} | "
                                  f"Total Reward: {self.current_episode_reward:.2f}")
                        
                        # Reset for next episode
                        self.episode_rewards.append(self.current_episode_reward)
                        self.episode_steps.append(self.current_episode_steps)
                        self.current_episode_reward = 0
                        self.current_episode_steps = 0
            
            # Create the final callback list with reward tracking
            reward_callback = RewardTrackingCallback(pbar, initial_reward=initial_reward)
            final_callback = [combined_callback, reward_callback]
            
            logger.info("Starting model.learn()...")
            logger.info(f"Initial reward: {initial_reward:.2f}")
            logger.info("Monitoring rewards and increments...")
            try:
                # Train the model with a smaller number of steps first
                model.learn(
                    total_timesteps=total_steps,
                    callback=final_callback,
                    progress_bar=False,
                    log_interval=1,
                    tb_log_name="td3_citylearn"
                )
                logger.info("Model training completed successfully")
                pbar.update(total_steps - pbar.n)
            except Exception as e:
                logger.error(f"Error during model.learn(): {str(e)}", exc_info=True)
                raise
        
        # Save final model
        final_model_path = os.path.join(log_dir, 'final_model')
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        return True, time.time() - start_time, final_model_path
        
    except ImportError as e:
        logger.error(f"Missing required package: {str(e)}")
        logger.info("Please install the required packages: pip install tensorboard")
        return False, time.time() - start_time, None
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        
        # Save the model at the point of failure with a timestamp
        if model is not None:
            timestamp = int(time.time())
            model_path = os.path.join(log_dir, f'model_error_{timestamp}')
            try:
                model.save(model_path)
                logger.info(f"Model saved to {model_path} after error")
                return False, time.time() - start_time, model_path
            except Exception as save_error:
                logger.error(f"Failed to save model after error: {str(save_error)}")
        
        return False, time.time() - start_time, None
        
    finally:
        if writer is not None:
            writer.close()

def main():
    # Create log directory
    log_dir = f"logs/td3_citylearn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Environment setup for CityLearn
    try:
        # Import CityLearn environment
        from citylearn.citylearn import CityLearnEnv
        from citylearn.data import DataSet
        from citylearn.wrappers import NormalizedObservationWrapper, NormalizedActionWrapper
        
        # List available datasets
        available_datasets = DataSet.get_names()
        logger.info(f"Available CityLearn datasets: {available_datasets}")
        
        if not available_datasets:
            logger.error("No built-in CityLearn datasets found. Please check your installation.")
            return
            
        # Use the first available dataset (you can modify this to choose a specific one)
        dataset_name = available_datasets[0]
        logger.info(f"Using CityLearn dataset: {dataset_name}")
        
        # Create environment with the selected dataset
        env = CityLearnEnv(dataset_name, simulation_days=365)  # Using 1 year of simulation
        eval_env = CityLearnEnv(dataset_name, simulation_days=30)  # Shorter evaluation period
        
        # Apply wrappers for normalization
        env = NormalizedObservationWrapper(env)
        env = NormalizedActionWrapper(env)
        
        eval_env = NormalizedObservationWrapper(eval_env)
        eval_env = NormalizedActionWrapper(eval_env)
        
        # Wrap for SB3 compatibility
        env = wrap_citylearn(env)
        eval_env = wrap_citylearn(eval_env)
        
        logger.info(f"CityLearn environment created with {len(env.buildings)} buildings")
        logger.info(f"Action space: {env.action_space}")
        logger.info(f"Observation space: {env.observation_space}")
        
    except ImportError as e:
        logger.error(f"Failed to import CityLearn: {str(e)}")
        logger.error("Please install the CityLearn package: pip install citylearn")
        return
    except Exception as e:
        logger.error(f"Failed to create CityLearn environment: {str(e)}", exc_info=True)
        return
    
    # Hyperparameters optimized for CityLearn
    hyperparams = {
        'learning_rate': 3e-4,  # Standard learning rate
        'buffer_size': 1000000,  # Large buffer for better exploration
        'learning_starts': 10000,  # Initial exploration steps
        'batch_size': 256,  # Batch size for training
        'tau': 0.005,  # Target network update rate
        'gamma': 0.99,  # Discount factor
        'train_freq': (1, 'step'),  # Update after each step
        'gradient_steps': 1,  # Number of gradient steps per update
        'policy_delay': 2,  # Policy update delay (TD3 specific)
        'target_policy_noise': 0.2,  # Noise added to target policy
        'target_noise_clip': 0.5,  # Clip target policy noise
        'policy_kwargs': {
            'net_arch': dict(pi=[400, 300], qf=[400, 300]),  # Network architecture
            'activation_fn': torch.nn.ReLU,  # Activation function
        },
        'replay_buffer_kwargs': {
            'handle_timeout_termination': True  # Handle timeouts properly
        }
    }
    
    # Initialize action noise for each building in the environment
    try:
        if isinstance(env.action_space, list):
            # Multi-agent setting - create a list of action noises
            action_noises = []
            for i, space in enumerate(env.action_space):
                n_actions = space.shape[0]
                noise = NormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=0.1 * np.ones(n_actions)
                )
                action_noises.append(noise)
                logger.info(f"Action noise for building {i} initialized with shape: {n_actions}")
            hyperparams['action_noise'] = action_noises
        else:
            # Single agent setting
            n_actions = env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.1 * np.ones(n_actions)
            )
            hyperparams['action_noise'] = action_noise
            logger.info(f"Action noise initialized with shape: {n_actions}")
    except Exception as e:
        logger.warning(f"Could not initialize action noise: {str(e)}")
        logger.warning("Continuing without action noise")
    
    # Start training
    try:
        success, training_time, model_path = train_td3(
            env=env,
            eval_env=eval_env,
            hyperparams=hyperparams,
            log_dir=log_dir,
            total_steps=50000,
            warmup_steps=10000
        )
        
        # Save training metrics
        metrics = {
            'success': success,
            'training_time_seconds': training_time,
            'model_path': model_path,
            'final_learning_rate': hyperparams['learning_rate']
        }
        
        metrics_path = os.path.join(log_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        if success:
            print(f"\n✅ Training completed successfully in {training_time:.2f} seconds")
            print(f"Model saved to: {model_path}")
            print(f"Training metrics saved to: {metrics_path}")
        else:
            print(f"\n❌ Training failed after {training_time:.2f} seconds")
            if model_path:
                print(f"Partial model saved to: {model_path}")
            print("Check the logs for more information")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"\n❌ An unexpected error occurred: {str(e)}")
    finally:
        env.close()
        eval_env.close()
        print(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    start_time = time.time()
    main()
