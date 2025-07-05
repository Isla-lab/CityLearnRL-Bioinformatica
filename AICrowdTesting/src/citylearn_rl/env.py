"""
Custom Environment for CityLearn-like KPI testing with stable-baselines3.

This module defines `CustomCityEnv`, a Gymnasium-compatible environment that
simulates building energy control scenarios. It uses a dummy data generator
and the AICrowdControl module for KPI calculation and reward scoring.
The environment is designed for episodic rewards, where KPIs and scores are
calculated at the end of each episode.
"""
import time
import gym
from gym import spaces
import numpy as np
import pandas as pd # Required for generate_dummy_environment_data
from .reward import ControlTrackReward, PhaseWeights, BASELINE_KPIS

# Copied from train_agent.py - consider refactoring this into a common utility module later
def generate_dummy_environment_data(num_timesteps, num_buildings, action=None):
    """
    Generates a dictionary of dummy data for all KPIs for one episode.
    'action': A conceptual parameter. In this version, if provided, it applies a simple
    modification to the baseline electricity consumption array ('e_episode_array') for the episode.
    """
    # --- Temperature Dynamics ---
    # Sinusoidal daily outdoor temperature pattern
    outdoor_temp_amplitude = 5  # Variation around the mean
    outdoor_temp_mean = 10  # Mean outdoor temperature
    outdoor_temp_episode_array = outdoor_temp_mean + outdoor_temp_amplitude * \
                                 np.sin(2 * np.pi * np.arange(num_timesteps) / 24)

    # Initialize temp_episode_array with a starting temperature for each building
    # The rest will be calculated in the `step` method.
    start_temp_per_building = np.random.uniform(20, 24, size=num_buildings) # e.g., random start between 20-24C
    temp_episode_array = np.zeros((num_timesteps, num_buildings))
    temp_episode_array[0, :] = start_temp_per_building

    # --- Electricity Consumption (e_episode_array) ---
    # Baseline demand profile (e.g., related to occupancy or a typical load shape)
    # For simplicity, let's make a typical office-like pattern for all buildings
    # Higher during the day (8 AM - 6 PM), lower otherwise.
    baseline_e_per_building = np.zeros((num_timesteps, num_buildings))
    day_hours = (np.arange(num_timesteps) % 24 >= 8) & (np.arange(num_timesteps) % 24 < 18)
    # Base load for each building (can be varied per building)
    base_load_factor = np.random.uniform(20, 40, size=num_buildings) # Random base load between 20-40 units
    peak_load_factor = base_load_factor * np.random.uniform(1.5, 2.5, size=num_buildings) # Peak is 1.5x to 2.5x base

    for t in range(num_timesteps):
        if 8 <= (t % 24) < 18: # Daytime: 8 AM to 6 PM
            baseline_e_per_building[t, :] = peak_load_factor
        else: # Nighttime or early morning
            baseline_e_per_building[t, :] = base_load_factor
    # Add some noise
    baseline_e_per_building += np.random.normal(0, 5, size=baseline_e_per_building.shape)
    baseline_e_per_building = np.clip(baseline_e_per_building, 5, 100) # Clip to reasonable values

    e_episode_array = baseline_e_per_building.copy() # This will be modified by agent action in `step`

    # --- Other Profiles ---
    # Setpoint: Different for occupied/unoccupied times or day/night
    setpoint_episode_array = np.full_like(temp_episode_array, 20.0) # Default night/unoccupied setpoint
    day_setpoint = 23.0 # Target during the day
    for t in range(num_timesteps):
        if 8 <= (t % 24) < 18: # Daytime: 8 AM to 6 PM
            setpoint_episode_array[t, :] = day_setpoint

    # Occupancy: Typical office hours
    occupancy_episode_array = np.zeros_like(temp_episode_array)
    for t in range(num_timesteps):
        if 8 <= (t % 24) < 18: # Occupied from 8 AM to 6 PM
            occupancy_episode_array[t, :] = 1.0 # Assume 1 means occupied

    # Emission Rate: Simple daily pattern (e.g., higher during peak grid load)
    emission_rate_episode = np.zeros(num_timesteps)
    base_emission = 0.2
    peak_increase = 0.3 # Max increase
    for t in range(num_timesteps):
        # Higher emissions during typical peak load hours (e.g., afternoon)
        if 12 <= (t % 24) < 20:
            emission_rate_episode[t] = base_emission + peak_increase * (( (t % 24) - 12) / 8.0) # Ramp up
        elif (t % 24) >= 20 or (t % 24) < 6: # Nighttime lower
            emission_rate_episode[t] = base_emission * 0.8
        else: # Morning ramp up
            emission_rate_episode[t] = base_emission + peak_increase * 0.3 * (( (t % 24) - 6) / 6.0)
    emission_rate_episode = np.clip(emission_rate_episode, 0.1, 0.6) # Ensure bounds

    # Demand and Served: Related to e_episode_array, served affected by outages
    # For now, demand is what would be consumed if no issues. Agent modifies 'e'.
    # Let's assume 'e' is the actual consumption the agent tries to achieve.
    # 'demand' could be a theoretical value if HVAC was running ideally for comfort,
    # but for now, let's tie it more directly to 'e' as the baseline.
    demand_episode_array = e_episode_array.copy() # Agent's target consumption before outages

    # Outage Timesteps: Can remain random
    outage_timesteps_episode = np.random.choice([True, False], size=num_timesteps, p=[0.05, 0.95]) # Less frequent outages

    served_episode_array = demand_episode_array.copy()
    for t in range(num_timesteps):
        if outage_timesteps_episode[t]:
            served_episode_array[t, :] = 0 # No electricity served during outage

    # District consumption will be calculated based on the agent-modified 'e' in `step` or `reset`
    # Initialize it here based on the baseline 'e'
    district_consumption_episode = np.sum(e_episode_array, axis=1)


    # This function in the plan was intended to generate data for the whole episode.
    # For a gym step, we need data for ONE timestep.
    # So, this function will be used by reset() to generate data for the whole episode,
    # and step() will iterate through it.

    return {
        'e': e_episode_array, # This is the baseline, to be copied to self.original_e_episode_array
        'emission_rate': emission_rate_episode,
        'temp': temp_episode_array, # Initial temp at t=0, rest to be filled
        'setpoint': setpoint_episode_array,
        'band': 1.0, # Comfort band (e.g., +/- 1 degree)
        'occupancy': occupancy_episode_array,
        'district_consumption': district_consumption_episode, # Based on baseline 'e'
        'hours_per_day': 24,
        'outage_timesteps': outage_timesteps_episode,
        'demand': demand_episode_array, # Based on baseline 'e'
        'served': served_episode_array, # Based on baseline 'e' and outages
        'outdoor_temp': outdoor_temp_episode_array # New outdoor temperature profile
    }

class CustomCityEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, phase_weights: PhaseWeights, num_buildings=2, timesteps_per_episode=24*7):
        """
        Initializes the CustomCityEnv.

        Args:
            phase_weights (PhaseWeights): Weights for different KPI components in reward calculation.
            num_buildings (int): Number of buildings to simulate.
            timesteps_per_episode (int): Number of timesteps that constitute one episode.
        """
        super(CustomCityEnv, self).__init__()

        self.num_buildings = num_buildings
        self.timesteps_per_episode = timesteps_per_episode
        self.current_timestep = 0
        self.current_step = 0  # Initialize current_step
        self.episode_length = timesteps_per_episode  # Initialize episode_length
        
        # Initialize random number generator
        self.np_random = np.random.default_rng()
        self._seed = None

        self.reward_calculator = ControlTrackReward(baseline=BASELINE_KPIS, phase=phase_weights)
        self.phase_weights = phase_weights # Store for direct access if needed

        # Temperature model parameters
        self.k_loss = 0.1  # Heat loss coefficient
        self.k_action = 0.5  # Coefficient for how much energy change affects temperature
        self.baseline_energy_for_comfort = 20.0  # Assumed energy needed to maintain comfort (e.g., in kWh)
                                                # This could be building-specific or dynamic later

        # Enhanced action space with 5 discrete actions for more granular control
        # 0: Very low consumption (aggressive energy saving)
        # 1: Low consumption (energy saving)
        # 2: Medium consumption (balanced)
        # 3: High consumption (comfort focus)
        # 4: Very high consumption (maximum comfort)
        self.action_space = spaces.Discrete(5)

        # Calculate the expected observation size
        # For N buildings, the observation size is:
        # - N buildings * (2 metrics + 2 trends) = 4N
        # - 3 district metrics = 3
        # - 4 time features = 4
        # - 15 action history (3 actions * 5 one-hot) = 15
        # Total = 4N + 3 + 4 + 15 = 4N + 22
        self.expected_obs_size = 4 * num_buildings + 22
        
        # Define the observation space with the correct size
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.expected_obs_size,),
            dtype=np.float32
        )

        self.episode_data = None # To store data for the current episode
        self.original_e_episode_array = None # To store original e data for consistent action effects

    def _get_observation(self):
        """Construct observation vector with enhanced state information.
        
        The observation includes:
        - Building-level metrics (consumption, temperature, setpoints, occupancy)
        - District-level metrics (aggregate consumption, emission rates)
        - Time features (hour of day, day of week)
        - Recent history statistics (moving averages, trends)
        - Action history (last few actions)
        """
        # Initialize observation vector with zeros
        obs = []
        
        # Ensure data is available and we're not at the end of the episode
        if self.episode_data is None:
            # If we're here during normal operation, it's a bug - log it but don't spam
            if hasattr(self, '_last_warning') and (time.time() - self._last_warning) < 10:
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            self._last_warning = time.time()
            print(f"Warning: _get_observation called with no episode data at timestep {self.current_timestep}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        if self.current_timestep >= self.timesteps_per_episode:
            # This is normal at the end of an episode, no need to warn
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Track building-specific features
        for b in range(self.num_buildings):
            # Current timestep values
            current_e = self.episode_data['e'][self.current_timestep, b]
            current_temp = self.episode_data['temp'][self.current_timestep, b]
            
            # Add normalized features
            # Max 'e' is ABSOLUTE_MAX_E_PER_BUILDING (e.g. 100.0) due to clipping in step().
            # Scaling by ABSOLUTE_MAX_E_PER_BUILDING.
            # Define ABSOLUTE_MAX_E_PER_BUILDING, e.g. 100.0, consistent with step() method's clipping
            ABSOLUTE_MAX_E_PER_BUILDING = 100.0
            norm_current_e = current_e / ABSOLUTE_MAX_E_PER_BUILDING
            obs.append(norm_current_e)
            
            # Temp is clipped to [10, 35] in step(). Normalizing to [0, 1].
            # (temp - min_temp) / (max_temp - min_temp)
            norm_current_temp = (current_temp - 10.0) / (35.0 - 10.0)
            norm_current_temp = np.clip(norm_current_temp, 0.0, 1.0) # Ensure it stays in [0,1] due to potential float issues
            obs.append(norm_current_temp)

            # Add short-term trends (difference from previous normalized timestep)
            if self.current_timestep > 0:
                prev_e = self.episode_data['e'][self.current_timestep-1, b]
                prev_temp = self.episode_data['temp'][self.current_timestep-1, b]

                norm_prev_e = prev_e / ABSOLUTE_MAX_E_PER_BUILDING
                obs.append(norm_current_e - norm_prev_e)  # Consumption trend will be roughly [-1, 1]

                norm_prev_temp = (prev_temp - 10.0) / (35.0 - 10.0)
                norm_prev_temp = np.clip(norm_prev_temp, 0.0, 1.0)
                obs.append(norm_current_temp - norm_prev_temp)  # Temperature trend will be roughly [-1, 1]
            else:
                obs.extend([0.0, 0.0])  # No trend data for first step
        
        # District-level metrics
        district_consumption = self.episode_data['district_consumption'][self.current_timestep]
        emission_rate = self.episode_data['emission_rate'][self.current_timestep]
        outage = self.episode_data['outage_timesteps'][self.current_timestep]
        
        # District consumption: sum of individual 'e' values. Max num_buildings * ABSOLUTE_MAX_E_PER_BUILDING
        # Define ABSOLUTE_MAX_E_PER_BUILDING, e.g. 100.0, consistent with step() method's clipping
        ABSOLUTE_MAX_E_PER_BUILDING = 100.0
        max_district_consumption = self.num_buildings * ABSOLUTE_MAX_E_PER_BUILDING
        obs.append(district_consumption / max_district_consumption if max_district_consumption > 0 else 0.0)

        # Emission rate is [0.1, 0.6]. Dividing by 0.6 gives [0.166, 1].
        obs.append(emission_rate / 0.6)
        obs.append(1.0 if outage else 0.0)  # Binary outage flag
        
        # Time features
        hour_of_day = self.current_timestep % 24
        day_of_week = (self.current_timestep // 24) % 7
        
        # Circular encoding for time features (sine/cosine)
        obs.append(np.sin(2 * np.pi * hour_of_day / 24.0))  # Hour of day (sine)
        obs.append(np.cos(2 * np.pi * hour_of_day / 24.0))  # Hour of day (cosine)
        obs.append(np.sin(2 * np.pi * day_of_week / 7.0))   # Day of week (sine)
        obs.append(np.cos(2 * np.pi * day_of_week / 7.0))   # Day of week (cosine)
        
        # Action history (last 3 actions, one-hot encoded for 5 possible actions)
        action_history = np.zeros(3 * 5)  # 3 actions, each one-hot encoded for 5 possible actions
        for i, action in enumerate(self.action_history[-3:]):  # Last 3 actions
            if i < 3:  # Ensure we don't go out of bounds
                action_history[i*5 + min(action, 4)] = 1.0  # Cap action at 4 (0-4 range)
        obs.extend(action_history)
        
        return np.array(obs, dtype=np.float32)

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s)."""
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        return [seed]
        
    def reset(self, **kwargs):
        self.current_timestep = 0
        self.current_step = 0  # Reset current_step
        # Initialize or clear action history
        self.action_history = []
        
        # Handle seed if provided in kwargs (for Gym compatibility)
        if 'seed' in kwargs:
            self.seed(kwargs['seed'])
        
        # Generate new episode data
        self.episode_data = generate_dummy_environment_data(
            num_timesteps=self.timesteps_per_episode,
            num_buildings=self.num_buildings,
            action=None
        )
        # Store a copy of the original (baseline) electricity consumption data
        # The 'e' from generate_dummy_environment_data is now the baseline.
        self.original_e_episode_array = self.episode_data['e'].copy()

        # Ensure 'temp' for the first timestep is correctly set from generate_dummy_environment_data
        # self.episode_data['temp'][0, :] is already initialized by generate_dummy_environment_data.
        # No further action needed here for temp initialization if generate_dummy_environment_data handles t=0.

        # Initialize the actual 'e' for t=0 to be the baseline 'e' for t=0.
        # Actions will modify 'e' from t=0 onwards in the first call to step().
        # So, self.episode_data['e'] will store the *actual, agent-modified* consumption.
        # The self.original_e_episode_array stores the baseline.
        # For t=0, before any action is taken, actual 'e' is the baseline 'e'.
        # This is already handled as episode_data['e'] is initialized from baseline.

        # Get initial observation (based on t=0 data)
        obs = self._get_observation()
        
        # Verify observation shape matches the space
        if obs.shape[0] != self.expected_obs_size:
            raise ValueError(
                f"Observation shape {obs.shape} does not match expected shape "
                f"({self.expected_obs_size},). Please check _get_observation() implementation."
            )
        
        return obs

    def step(self, action):
        if self.episode_data is None:
            raise Exception("Must call reset() before step()")

        # Convert action to integer if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
            
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {self.action_space}")
            
        # Store the action in history before processing
        self.action_history.append(action)

        # --- 1. Electricity Consumption Update (based on action) ---
        action_effect_multiplier = {
            0: 0.4,   # Very low (60% reduction from baseline)
            1: 0.7,   # Low (30% reduction)
            2: 1.0,   # Medium (baseline consumption)
            3: 1.4,   # High (40% increase)
            4: 1.8    # Very high (80% increase)
        }[action]
        
        ABSOLUTE_MIN_E_PER_BUILDING = 1.0
        ABSOLUTE_MAX_E_PER_BUILDING = 100.0 # Arbitrary cap, can be profile-dependent

        # Get baseline consumption for the current timestep
        baseline_e_t = self.original_e_episode_array[self.current_timestep, :].copy()

        # Apply action effect to get target consumption
        target_e_t = baseline_e_t * action_effect_multiplier

        # Clip to absolute limits
        actual_e_t = np.clip(target_e_t, ABSOLUTE_MIN_E_PER_BUILDING, ABSOLUTE_MAX_E_PER_BUILDING)

        # Store the agent-modified electricity consumption for the current timestep
        self.episode_data['e'][self.current_timestep, :] = actual_e_t

        # Update district consumption based on the agent-modified 'e' values
        self.episode_data['district_consumption'][self.current_timestep] = np.sum(actual_e_t)

        # If there's an outage, served electricity is zero for affected buildings
        if self.episode_data['outage_timesteps'][self.current_timestep]:
            # Assuming outage affects all buildings for simplicity, can be building-specific
            self.episode_data['served'][self.current_timestep, :] = 0.0
            # Also, actual electricity consumed ('e') should be 0 if served is 0.
            self.episode_data['e'][self.current_timestep, :] = 0.0
            # Re-update district consumption if 'e' changed due to outage
            self.episode_data['district_consumption'][self.current_timestep] = np.sum(
                self.episode_data['e'][self.current_timestep, :]
            )
        else:
            # If no outage, served is equal to the (modified) demand 'e'
             self.episode_data['served'][self.current_timestep, :] = self.episode_data['e'][self.current_timestep, :].copy()


        # --- 2. Temperature Update (dynamic model) ---
        # This must happen *after* 'e' for the current timestep is determined.
        if self.current_timestep < self.timesteps_per_episode:
            T_outdoor_current = self.episode_data['outdoor_temp'][self.current_timestep]
            
            for b in range(self.num_buildings):
                T_old = self.episode_data['temp'][self.current_timestep -1, b] if self.current_timestep > 0 else self.episode_data['temp'][0,b]

                # Action_effect_on_Energy is the actual electricity consumed by HVAC for building b
                # We'll use self.episode_data['e'][self.current_timestep, b] as a proxy for this.
                # This implies 'e' is primarily HVAC, or HVAC is a dominant part of it.
                Action_effect_on_Energy = self.episode_data['e'][self.current_timestep, b]

                # Baseline_Energy_for_comfort could be related to maintaining setpoint,
                # or a fixed value representing average energy for comfort.
                # For now, using the fixed value from __init__.

                delta_T_loss = self.k_loss * (T_old - T_outdoor_current)
                delta_T_action = self.k_action * (Action_effect_on_Energy - self.baseline_energy_for_comfort)

                T_new = T_old - delta_T_loss + delta_T_action

                # Clip temperature to realistic bounds (e.g., 10C to 35C)
                T_new = np.clip(T_new, 10.0, 35.0)

                self.episode_data['temp'][self.current_timestep, b] = T_new

        # --- 3. Intermediate Reward (Discomfort Penalty) ---
        discomfort_penalty_current_step = 0.0
        # Use data from the current timestep for penalty calculation
        # Temp for current_timestep was just updated. Setpoint and Occupancy are from profile.
        for b in range(self.num_buildings):
            current_temp_b = self.episode_data['temp'][self.current_timestep, b]
            current_setpoint_b = self.episode_data['setpoint'][self.current_timestep, b]
            current_occupancy_b = self.episode_data['occupancy'][self.current_timestep, b]
            comfort_band = self.episode_data['band'] # Assuming 'band' is scalar or correctly shaped

            if current_occupancy_b > 0 and abs(current_temp_b - current_setpoint_b) > comfort_band:
                discomfort_penalty_current_step += \
                    (abs(current_temp_b - current_setpoint_b) - comfort_band) * 0.1 # Small penalty factor

        intermediate_reward = -discomfort_penalty_current_step

        # --- Increment timestep and check for done ---
        self.current_timestep += 1
        self.current_step += 1
        done = self.current_timestep >= self.timesteps_per_episode
        info = {}

        # --- 4. Episodic Reward (if done) ---
        if done:
            try:
                # Ensure all necessary data in self.episode_data is up-to-date before KPI calculation.
                # 'e' (agent-modified), 'temp' (dynamically calculated), 'setpoint', 'occupancy',
                # 'emission_rate', 'district_consumption', 'served', 'demand' etc. should be final.
                # 'demand' in this context is the baseline demand before agent action,
                # which is self.original_e_episode_array.
                # However, the reward calculator might expect 'demand' to be something else.
                # For CityLearn, 'demand' is often the energy consumed by the building.
                # Let's ensure our `self.episode_data` matches expectations or adjust.
                # The AICrowdControl kpi functions expect `self.episode_data['e']` as consumption.
                # Let's make sure `self.episode_data['demand']` is consistent with `['e']` if it's used by KPIs,
                # or pass `self.original_e_episode_array` if that's the intended 'demand' for some KPI.
                # For now, let's assume KPIs use `self.episode_data['e']` as the primary consumption metric.
                # The `generate_dummy_environment_data` sets `demand` based on baseline `e`.
                # If KPIs need demand that *would have been* without agent, then `original_e_episode_array` is better.
                # Let's assume for now `self.episode_data['demand']` is the relevant one.
                # The current setup:
                # self.episode_data['e'] is agent-modified consumption.
                # self.episode_data['demand'] was initialized from baseline_e_per_building.
                # self.episode_data['served'] is self.episode_data['e'] unless outage.

                raw_kpis = self.reward_calculator.get_all_kpi_values(self.episode_data)
                
                # Calculate episodic reward using the reward calculator's score method
                episodic_reward_raw = self.reward_calculator.score(raw_kpis)
                
                # Scale the reward
                scaled_episodic_reward = episodic_reward_raw * 100.0 # Scaling factor (e.g., 100 or 200)
                
                # The final reward for this step (when done) is the scaled episodic reward
                # plus the (negative) discomfort penalty for this very last step.
                reward = scaled_episodic_reward + intermediate_reward # intermediate_reward is already negative
                
                info = {
                    'episode_kpis': raw_kpis,
                    'raw_episodic_reward': episodic_reward_raw,
                    'scaled_episodic_reward': scaled_episodic_reward,
                    'final_discomfort_penalty_contrib': intermediate_reward, # For this last step
                    'final_reward': reward,
                    # Include individual KPIs from raw_kpis for detailed logging if needed
                }
                # Add all raw_kpis to info for easier access
                if isinstance(raw_kpis, dict):
                    for k,v in raw_kpis.items():
                        info[f"kpi_{k}"] = v

            except Exception as e:
                print(f"Error calculating KPIs/reward at end of episode: {e}")
                reward = -100.0  # Significant penalty for failure
                info = {'error': str(e)}
                
            # For the terminal state, the observation is typically not used by SB3,
            # but providing a zero vector or the last valid observation are common.
            # Let's use the last valid observation.
            # _get_observation uses self.current_timestep, which is now self.timesteps_per_episode.
            # We need obs for t = self.timesteps_per_episode - 1.
            # However, _get_observation might try to access self.current_timestep,
            # so we might need to temporarily decrement it or pass the step index.
            # For now, a zero vector is safer if _get_observation isn't robust to this.
            # Let's ensure _get_observation can handle current_timestep == timesteps_per_episode for terminal state.
            # The existing _get_observation has a check:
            # if self.episode_data is None or self.current_timestep >= self.timesteps_per_episode:
            # This means it would return zeros, which is fine.
            observation = self._get_observation()

        else: # Not done
            reward = intermediate_reward # Per-step reward is just the discomfort penalty
            observation = self._get_observation()

        # Update last_action after processing the current action
        self.last_action = action

        return observation, reward, done, info

    def render(self, mode='human'):
        # For now, no complex rendering
        pass

    def close(self):
        pass

if __name__ == '__main__':
    # Example Usage and Test
    print("Testing CustomCityEnv...")
    phase_weights = PhaseWeights(w1=0.5, w2=0.5, w3=0.0, w4=0.0) # Comfort and Emissions
    env = CustomCityEnv(phase_weights=phase_weights, num_buildings=2, timesteps_per_episode=24*1) # 1 day episode
    env.seed(42) # Set seed for reproducibility
    obs = env.reset()
    print("Initial Observation Shape:", obs.shape)
    print("Initial Observation:", obs)
    # print("Initial Info:", info) # Info is not returned by reset in gym 0.21.0

    done = False
    total_reward = 0
    step_count = 0
    while not done:
        action = env.action_space.sample() # Random action
        obs, reward, done, info = env.step(action) # Use 'done'
        total_reward += reward # Accumulate reward (though it's only non-zero at the end)
        step_count += 1
        if done: # Check 'done'
            print(f"Episode finished after {step_count} steps. Done: {done}")
            print(f"Final Reward: {reward}") # This is the episode reward
            if 'episode_kpis' in info:
                print("Episode KPIs:")
                for k, v in info['episode_kpis'].items():
                    print(f"  {k}: {v:.4f}")
        # else:
            # print(f"Step {step_count}, Action: {action}, Obs: {obs[:3]}..., Reward: {reward}, Done: {done}")


    print("Environment test complete.")

    # Test with stable-baselines3 checker
    from stable_baselines3.common.env_checker import check_env
    try:
        check_env(env, warn=True)
        print("stable-baselines3 check_env passed!")
    except Exception as e:
        print(f"stable-baselines3 check_env failed: {e}")
