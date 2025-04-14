import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

class FlattenObservationWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env

        self.action_space = env.action_space[0]

        # Calcolo la nuova observation_space (flattened)
        sample_obs = self._flatten_obs(self.env.reset())
        obs_shape = sample_obs.shape
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._flatten_obs(obs), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step([action])
        reward = reward[0] if isinstance(reward, list) else reward

        done = terminated or truncated
        if done:
            print("👉 Episodio terminato, reward:", reward)
    
        return self._flatten_obs(obs), reward, terminated, truncated, info




   
    def _flatten_obs(self, obs):
        # Filtra solo gli array validi, scarta None e float
        obs = [np.ravel(o) for o in obs if isinstance(o, (np.ndarray, list))]
        return np.concatenate(obs, axis=0)



    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

def wrap_env(env):
    env = FlattenObservationWrapper(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    return env
