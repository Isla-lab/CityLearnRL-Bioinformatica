importa numpy as np
import matplotlib.pyplot as plt
from citylearn.citylearn import CityLearnEnv

def make_env(seed):
  env = CityLearnEnv(schema="citylearn_challenge_2022_phase_1")
  env.seed(seed)
  return env

def evaluate_model(model, env, algo_name, seed):
  obs = env.reset()
  rewards = []
  for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    rewards.append(reward)
  np.save(f"results/{algo_name}/rewards_seed{seed}.npy", rewards)
