"""
AICrowdControl - Reward Calculation Module for CityLearn Challenge Control Track

This module implements the reward calculation logic for the CityLearn Challenge Control Track.
It provides a flexible framework for evaluating building control strategies based on multiple
Key Performance Indicators (KPIs) including energy efficiency, grid impact, and occupant comfort.

Key Components:
- PhaseWeights: Dataclass for defining weight distribution across different reward components
- ControlTrackReward: Main class for calculating and aggregating various KPIs into a final score

"""

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Mapping, Union, Dict, Any

@dataclass
class PhaseWeights:
    """
    Dataclass defining the weight distribution for control track score calculation.
    
    The weights determine the relative importance of different components in the final score.
    All weights should sum to 1.0 for proper normalization.
    
    Attributes:
        w1 (float): Weight for thermal comfort component (0.0 to 1.0)
        w2 (float): Weight for carbon emissions component (0.0 to 1.0)
        w3 (float): Weight for grid impact component (0.0 to 1.0)
        w4 (float): Weight for resilience component (0.0 to 1.0)
    """
    w1: float  # Comfort (thermal comfort of occupants)
    w2: float  # Emissions (carbon footprint)
    w3: float  # Grid (load shifting and peak shaving)
    w4: float  # Resilience (performance during grid outages)

# Predefined phase weight configurations
PHASE_I = PhaseWeights(w1=0.3, w2=0.1, w3=0.6, w4=0.0)  # Phase I focuses more on grid impact
PHASE_II = PhaseWeights(w1=0.3, w2=0.1, w3=0.3, w4=0.3)  # Phase II adds resilience component

# Baseline KPIs for reward calculation normalization
# These values are used to normalize the KPIs to a common scale (0-1)
# The actual baseline values should be set based on your specific scenario
BASELINE_KPIS = {
    "carbon_emissions": 1.0,           # Baseline for carbon emissions (kgCO2)
    "ramping": 1.0,                    # Baseline for load ramping (kW)
    "1-load_factor": 1.0,              # Baseline for load factor (1 - actual_load_factor)
    "daily_peak": 1.0,                 # Baseline for daily peak demand (kW)
    "all_time_peak": 1.0,              # Baseline for all-time peak demand (kW)
    "unmet_hours": 1.0,                # Baseline for thermal comfort violations (hours)
    "1-thermal_resilience": 1.0,        # Baseline for thermal resilience during outages
    "normalized_unserved_energy": 1.0,   # Baseline for unserved energy during outages (kWh)
}

class ControlTrackReward:
    """
    Main class for calculating the Control Track reward based on various KPIs.
    
    This class implements the reward calculation logic as specified in the CityLearn Challenge.
    It computes multiple KPIs and combines them into a single score using configurable weights.
    
    Args:
        baseline (Mapping[str, float]): Dictionary containing baseline KPI values for normalization
        phase (Union[dict, PhaseWeights], optional): Phase configuration or custom weights. 
            If dict, should contain keys w1-w4. Defaults to PHASE_I.
    """

    def __init__(self, baseline: Mapping[str, float], phase: Union[dict, PhaseWeights] = None):
        """
        Initialize the reward calculator with baseline values and phase configuration.
        
        Args:
            baseline: Dictionary mapping KPI names to their baseline values
            phase: Either a PhaseWeights instance or a dictionary with weights (w1-w4)
        """
        self.baseline = baseline
        
        # Convert phase dict to PhaseWeights if needed
        if isinstance(phase, dict):
            self.phase = PhaseWeights(
                w1=phase.get('w1', 0.3),  # Default weights if not specified
                w2=phase.get('w2', 0.1),
                w3=phase.get('w3', 0.6),
                w4=phase.get('w4', 0.0)
            )
        else:
            # Use provided PhaseWeights or default to PHASE_I
            self.phase = phase if phase is not None else PHASE_I

    def carbon_emissions(self, e: np.ndarray, emission_rate: Sequence[float]) -> float:
        """
        Calculate the total carbon emissions based on electricity consumption.
        
        This method computes the carbon emissions by multiplying the electricity
        consumption by the emission rate at each timestep, then averaging across all buildings.
        
        Args:
            e (np.ndarray): Net electricity consumption with shape (time_steps, num_buildings).
                          Negative values are treated as zero (no carbon credits for export).
            emission_rate (Sequence[float]): Carbon intensity of the grid (kgCO2/kWh) for each timestep.
                                          Length should match the number of timesteps in 'e'.
                                          
        Returns:
            float: Mean carbon emissions across all buildings (kgCO2).
            
        Note:
            Only positive electricity consumption is considered (net consumption).
        """
        # Ensure we only consider positive electricity consumption (no carbon credits for export)
        e = np.maximum(e, 0.0)
        # Calculate emissions: sum over time (emission_rate * consumption) for each building
        g_i = np.sum(e * np.asarray(emission_rate)[:, None], axis=0)
        # Return the mean across all buildings
        return float(np.mean(g_i))

    def unmet_hours(self, temp: np.ndarray, setpoint: np.ndarray, band: float, occupancy: np.ndarray) -> float:
        """
        Calculate the fraction of occupied hours with temperature outside the comfort band.
        
        This metric quantifies thermal discomfort by measuring how often the indoor
        temperature deviates from the setpoint beyond the specified comfort band
        during occupied hours.
        
        Args:
            temp (np.ndarray): Actual temperature measurements (time_steps, num_zones).
            setpoint (np.ndarray): Target temperature setpoints (time_steps, num_zones).
            band (float): Allowed temperature deviation from setpoint (±°C).
            occupancy (np.ndarray): Binary array indicating occupancy (1=occupied, 0=unoccupied).
            
        Returns:
            float: Fraction of occupied hours with temperature outside comfort band,
                  averaged across all zones (0.0 to 1.0).
                  
        Note:
            Only counts violations during occupied hours (occupancy > 0).
        """
        # Identify timesteps where temperature is outside comfort band
        diff = np.abs(temp - setpoint) > band
        # Only count violations during occupied hours
        unmet = diff & (occupancy > 0)
        # Return the mean violation rate across all zones and timesteps
        return float(np.mean(unmet))

    def ramping(self, district_consumption: Sequence[float]) -> float:
        """
        Calculate the total ramping of district electricity consumption.
        
        This metric quantifies the total variation in electricity consumption between
        consecutive timesteps, which is important for grid stability.
        
        Args:
            district_consumption (Sequence[float]): Total electricity consumption (kW)
                                                 for the district at each timestep.
                                                 
        Returns:
            float: Sum of absolute differences between consecutive consumption values (kW).
            
        Note:
            Higher values indicate more volatile load profiles.
            This metric is sensitive to the sampling interval of the data.
        """
        e = np.asarray(district_consumption)
        # Calculate absolute differences between consecutive timesteps and sum them
        return float(np.sum(np.abs(np.diff(e))))

    def load_factor(self, district_consumption: Sequence[float], hours_per_day: int = 24) -> float:
        """
        Calculate the complement of the average daily load factor.
        
        The load factor is the ratio of average load to peak load. This function returns
        1 - load_factor, where a higher value indicates more peaky (less efficient) load profiles.
        
        Args:
            district_consumption (Sequence[float]): Total electricity consumption (kW)
                                                 for the district at each timestep.
            hours_per_day (int, optional): Number of timesteps in a day. Defaults to 24.
            
        Returns:
            float: 1.0 - average_daily_load_factor (0.0 to 1.0).
            
        Note:
            - A value of 0.0 indicates perfectly flat consumption (ideal).
            - A value approaching 1.0 indicates very peaky consumption.
        """
        e = np.asarray(district_consumption)
        days = len(e) // hours_per_day
        ratios = []
        
        # Calculate daily load factors
        for d in range(days):
            # Get one day of data
            daily = e[d*hours_per_day:(d+1)*hours_per_day]
            if len(daily) == 0:
                continue  # Skip empty days
                
            # Calculate load factor for this day (mean/peak)
            daily_max = np.max(daily)
            if daily_max > 0:  # Avoid division by zero
                ratios.append(np.mean(daily) / daily_max)
            else:
                ratios.append(0.0)  # If max is zero, load factor is zero
                
        # Return complement of average load factor
        lf = np.mean(ratios) if ratios else 0.0
        return 1.0 - lf

    def daily_peak(self, district_consumption: Sequence[float], hours_per_day: int = 24) -> float:
        """
        Calculate the average daily peak electricity consumption.
        
        This metric helps understand the peak demand that the grid must be able to serve
        on a typical day.
        
        Args:
            district_consumption (Sequence[float]): Total electricity consumption (kW)
                                                 for the district at each timestep.
            hours_per_day (int, optional): Number of timesteps in a day. Defaults to 24.
            
        Returns:
            float: Average of daily peak consumption values (kW).
            
        Note:
            This is different from all_time_peak() which finds the single highest peak
            across the entire simulation period.
        """
        e = np.asarray(district_consumption)
        days = len(e) // hours_per_day
        if days == 0:
            return 0.0
            
        # Calculate peak for each day and take the average
        peaks = [np.max(e[d*hours_per_day:(d+1)*hours_per_day]) for d in range(days)]
        return float(np.mean(peaks))

    def all_time_peak(self, district_consumption: Sequence[float]) -> float:
        """
        Calculate the maximum electricity consumption across the entire simulation period.
        
        This represents the absolute peak demand that the grid must be able to serve.
        
        Args:
            district_consumption (Sequence[float]): Total electricity consumption (kW)
                                                 for the district at each timestep.
                                                 
        Returns:
            float: Maximum consumption value observed (kW).
            
        Note:
            This is different from daily_peak() which calculates the average of daily peaks.
        """
        if len(district_consumption) == 0:
            return 0.0
        return float(np.max(district_consumption))

    def thermal_resilience(self, temp: np.ndarray, setpoint: np.ndarray, band: float,
                          occupancy: np.ndarray, outage: np.ndarray) -> float:
        """
        Calculate the fraction of time thermal comfort is violated during grid outages.
        
        This metric evaluates how well the building maintains comfort during power outages,
        which is critical for resilience assessment.
        
        Args:
            temp (np.ndarray): Actual temperature measurements (time_steps, num_zones).
            setpoint (np.ndarray): Target temperature setpoints (time_steps, num_zones).
            band (float): Allowed temperature deviation from setpoint (±°C).
            occupancy (np.ndarray): Binary array indicating occupancy (1=occupied, 0=unoccupied).
            outage (np.ndarray): Boolean array indicating grid outage conditions (time_steps,).
            
        Returns:
            float: Fraction of time with comfort violations during outages (0.0 to 1.0).
            
        Note:
            Only counts violations that occur during both occupied hours and grid outages.
        """
        # Identify comfort violations
        diff = np.abs(temp - setpoint) > band
        # Only count violations during occupied hours AND during outages
        unmet = diff & (occupancy > 0) & np.asarray(outage, dtype=bool)[:, None]
        return float(np.mean(unmet))

    def unserved_energy(self, demand: np.ndarray, served: np.ndarray, outage: np.ndarray) -> float:
        """
        Calculate the fraction of energy demand not met during grid outages.
        
        This metric measures the system's ability to meet energy demand when the grid is down,
        which is a key aspect of resilience.
        
        Args:
            demand (np.ndarray): Total energy demand (kWh) with shape (time_steps, num_buildings).
            served (np.ndarray): Energy actually served (kWh) with same shape as demand.
            outage (np.ndarray): Boolean array indicating grid outage conditions (time_steps,).
            
        Returns:
            float: Ratio of unserved energy to total demand during outages (0.0 to 1.0).
            
        Note:
            - Only considers periods with grid outages.
            - A small epsilon (1e-9) is added to the denominator to avoid division by zero.
            - Values are clipped to ensure the result is between 0.0 and 1.0.
        """
        # Calculate unserved energy (only during outages, only positive values)
        unmet = np.where(outage[:, None], np.maximum(demand - served, 0.0), 0.0)
        # Total expected demand during outages
        expected = np.where(outage[:, None], demand, 0.0)
        total_expected = expected.sum()
        # Return ratio of unserved to expected energy (avoid division by zero)
        if total_expected <= 0:
            return 0.0
        return float(np.clip(unmet.sum() / total_expected, 0.0, 1.0))

    def score(self, kpis: Mapping[str, float]) -> float:
        """
        Calculate the overall control track score based on the provided KPIs.
        
        The final score is a weighted sum of four main components:
        1. Comfort: Based on thermal comfort violations
        2. Emissions: Based on carbon emissions
        3. Grid: Average of ramping, load factor, and peak metrics
        4. Resilience: Average of thermal resilience and unserved energy metrics
        
        Args:
            kpis (Mapping[str, float]): Dictionary containing calculated KPI values.
                                      Should include all keys present in BASELINE_KPIS.
                                      
        Returns:
            float: Overall score between 0.0 and 1.0, where higher is better.
            
        Note:
            - All KPIs are normalized by their baseline values.
            - The weights for each component are determined by the phase configuration.
            - The grid score is the average of ramping, load factor, and peak metrics.
            - The resilience score is the average of thermal resilience and unserved energy metrics.
        """
        # Calculate normalized scores for each KPI
        # Emissions score (lower emissions is better)
        score_emissions = 1.0 - kpis['carbon_emissions'] / self.baseline['carbon_emissions']
        
        # Grid-related scores (lower is better for all)
        score_ramping = 1.0 - kpis['ramping'] / self.baseline['ramping']
        score_load_factor = 1.0 - kpis['1-load_factor'] / self.baseline['1-load_factor']
        score_daily_peak = 1.0 - kpis['daily_peak'] / self.baseline['daily_peak']
        score_all_time_peak = 1.0 - kpis['all_time_peak'] / self.baseline['all_time_peak']
        
        # Calculate composite grid score (average of grid-related metrics)
        score_grid = (score_ramping + score_load_factor + score_daily_peak + score_all_time_peak) / 4.0
        
        # Resilience-related scores (lower is better in kpis, so we invert them)
        score_thermal_resilience = 1.0 - kpis['1-thermal_resilience']
        score_unserved_energy = 1.0 - kpis['normalized_unserved_energy']
        
        # Calculate composite resilience score (average of resilience metrics)
        score_resilience = (score_thermal_resilience + score_unserved_energy) / 2.0
        
        # Comfort score (lower unmet_hours is better)
        score_comfort = 1.0 - kpis['unmet_hours']
        
        # Get weights from phase configuration
        weights = self.phase
        
        # Calculate final weighted score
        score_control = (
            weights.w1 * score_comfort +         # Comfort component
            weights.w2 * score_emissions +       # Emissions component
            weights.w3 * score_grid +            # Grid component
            weights.w4 * score_resilience        # Resilience component
        )
        
        # Ensure the score is within valid range
        return float(np.clip(score_control, 0.0, 1.0))
