import gym
from gym.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from citylearn.citylearn import CityLearnEnv
import numpy as np

from src.wrappers import SingleBuildingWrapper

def make_env(seed=0):
    env = CityLearnEnv(
        schema='citylearn_challenge_2022_phase_1',
        building_ids=['Building_1']
    )
    env = SingleBuildingWrapper(env)
    env.reset(seed=seed)
    return env




def evaluate_model(model, env, algo_name, seed):
  obs = env.reset()
  rewards = []
  for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    rewards.append(reward)
  np.save(f"results/{algo_name}/rewards_seed{seed}.npy", rewards)
