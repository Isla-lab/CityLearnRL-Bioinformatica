import numpy as np
from stable_baselines3 import PPO
from utils import make_env, save_rewards
from wrappers import wrap_env

def train_and_save(seed=0, timesteps=100_000):
    env = make_env()
    env = wrap_env(env)
    model = PPO("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=timesteps)
    model.save(f"results/ppo/ppo_seed_{seed}")
    save_rewards(env.envs[0].episode_returns, f"results/ppo/rewards_seed_{seed}.npy")
    env.close()

# Se vuoi, puoi lasciare questo per eseguire singolarmente
if __name__ == "__main__":
    train_and_save(seed=0, timesteps=100_000)

    