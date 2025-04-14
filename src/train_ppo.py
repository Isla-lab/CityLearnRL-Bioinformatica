from stable_baselines3 import PPO
from citylearn.citylearn import CityLearnEnv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import make_env, evaluate_model

def main(seed=0):
  env = make_env(seed)
  model = PPO("MlpPolicy", env, verbose=1, seed=seed)
  model.learn(total_timesteps=100_000)
  model.save(f"results/ppo/ppo_seed{seed}")
  evaluate_model(mode, env, "ppo", seed)


if __name__ == "__main__":
  main()
