import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import StableBaselines3Wrapper
from gym import Wrapper

from AICrowd.AICrowdControl import ControlTrackReward, PHASE_I

BASELINE_KPIS = {
    "carbon_emissions": 1.0,
    "ramping": 1.0,
    "1-load_factor": 1.0,
    "daily_peak": 1.0,
    "all_time_peak": 1.0,
    "unmet_hours": 1.0,
    "1-thermal_resilience": 1.0,
    "normalized_unserved_energy": 1.0,
}

class ChallengeRewardWrapper(Wrapper):
    """Wrapper that replaces environment reward with control track score."""

    def __init__(self, env, reward_calc=None):
        super().__init__(env)
        self.reward_calc = reward_calc or ControlTrackReward(BASELINE_KPIS, PHASE_I)

    def step(self, action):
        obs, _reward, done, info = self.env.step(action)
        evaluation = self.env.evaluate_citylearn_challenge()
        kpis = {
            "carbon_emissions": evaluation["carbon_emissions_total"]["value"],
            "unmet_hours": evaluation["discomfort_proportion"]["value"],
            "ramping": evaluation["ramping_average"]["value"],
            "1-load_factor": evaluation["daily_one_minus_load_factor_average"]["value"],
            "daily_peak": evaluation["daily_peak_average"]["value"],
            "all_time_peak": evaluation["annual_peak_average"]["value"],
            "1-thermal_resilience": evaluation["one_minus_thermal_resilience_proportion"]["value"],
            "normalized_unserved_energy": evaluation["power_outage_normalized_unserved_energy_total"]["value"],
        }
        reward = self.reward_calc.score(kpis)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_env(dataset_name="citylearn_challenge_2022_phase_all"):
    env = CityLearnEnv(dataset_name, central_agent=True)
    env = StableBaselines3Wrapper(env)
    env = ChallengeRewardWrapper(env)
    return env


def main(total_timesteps=10000, model_path="sac_aicrowd.zip"):
    env = DummyVecEnv([make_env])
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    env.close()


if __name__ == "__main__":
    total_timesteps = int(os.environ.get("TIMESTEPS", 10000))
    main(total_timesteps)
