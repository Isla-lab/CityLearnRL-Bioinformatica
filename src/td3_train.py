import numpy as np
from stable_baselines3 import TD3
from utils import make_env, save_rewards
from wrappers import wrap_env

TIMESTEPS = 100_000
SEEDS = [0, 42, 123, 999, 2024]

for seed in SEEDS:
    env = make_env()
    env = wrap_env(env)
    model = TD3("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(f"results/td3/td3_seed_{seed}")
    save_rewards(env.envs[0].episode_rewards, f"results/td3/rewards_seed_{seed}.npy")
    env.close()
