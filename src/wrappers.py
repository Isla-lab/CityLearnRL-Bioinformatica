import gym
import numpy as np

class SingleBuildingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Assumiamo che ogni osservazione sia un singolo elemento di una lista
        assert isinstance(env.observation_space, list), "Expected list of observation spaces"
        self.observation_space = env.observation_space[0]

    def observation(self, observation):
        return observation[0]  # Restituisci solo l’osservazione del primo building