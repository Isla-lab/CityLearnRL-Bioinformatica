# Repository Overview

This repository provides a minimal reinforcement learning (RL) framework to experiment with the **CityLearn** challenge KPIs. It includes a custom Gym environment (`CustomCityEnv`), reward calculation utilities, and training scripts based on Stable Baselines3.

## Environment Setup

The environment is defined in `src/citylearn_rl/env.py`. Initial episode data comes from `generate_dummy_environment_data`, which builds synthetic profiles for temperature, electricity demand, setpoints, occupancy, emission rates, and outage timesteps.

`CustomCityEnv` supports five discrete actions representing different electricity consumption levels. Observations combine building metrics (consumption and temperature with trends), district metrics (aggregate consumption, emission rate, outage flag), time features, and recent action history. The observation space size is computed as `4 * num_buildings + 22`.

Reference lines from `env.py`:
```python
class CustomCityEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, phase_weights: PhaseWeights, num_buildings=2, timesteps_per_episode=24*7):
        ...
        self.action_space = spaces.Discrete(5)
        self.expected_obs_size = 4 * num_buildings + 22
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.expected_obs_size,),
            dtype=np.float32
        )
```

## Reward Formulas

Reward computation is handled by `ControlTrackReward` in `src/citylearn_rl/reward.py`. Raw KPIs are calculated and then combined into a weighted score.

### KPI Definitions
- **Carbon Emissions:**
  $\( G = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} e_{i,t} \cdot c_t \)$
- **Thermal Comfort:**
  $\( U = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \mathbb{1}(|T_{i,t}-S_{i,t}|>B) \cdot \mathbb{1}(O_{i,t}>0) \)$
- **Grid Metrics:** ramping, load factor (1-load_factor), daily peak and all-time peak.
- **Resilience Metrics:** thermal resilience fraction and unserved energy fraction.

### Final Score
The overall score is
$\[ score = w_1 S_{\text{comfort}} + w_2 S_{\text{emissions}} + w_3 S_{\text{grid}} + w_4 S_{\text{resilience}} \]$
where each component score is normalized against `BASELINE_KPIS`.

`PhaseWeights` provide two standard configurations:
- **Phase I:** w1=0.3, w2=0.1, w3=0.6, w4=0.0
- **Phase II:** w1=0.3, w2=0.1, w3=0.3, w4=0.3

## Training Functions

Training is performed via `scripts/train.py` which defines two entry points:
- `main_2kpi_training` – focuses on comfort and emissions using `PhaseWeights(w1=0.8, w2=0.2, w3=0.0, w4=0.0)`.
- `main_multi_kpi_training` – trains on all four KPIs using weights `w1=0.3, w2=0.1, w3=0.3, w4=0.3`.

Both functions instantiate `CustomCityEnv`, configure PPO hyperparameters, attach evaluation callbacks and run `.learn()` for a configured number of timesteps. Models and logs are stored in the `results/` directory.
