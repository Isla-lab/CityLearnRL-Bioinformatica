import numpy as np
from dataclasses import dataclass
from typing import Sequence, Mapping

@dataclass
class PhaseWeights:
    """Weights for control track score calculation."""
    w1: float  # Comfort
    w2: float  # Emissions
    w3: float  # Grid
    w4: float  # Resilience

PHASE_I = PhaseWeights(w1=0.3, w2=0.1, w3=0.6, w4=0.0)
PHASE_II = PhaseWeights(w1=0.3, w2=0.1, w3=0.3, w4=0.3)

class ControlTrackReward:
    """Compute Control Track reward following CityLearn Challenge formulas."""

    def __init__(self, baseline: Mapping[str, float], phase: PhaseWeights = PHASE_I):
        self.baseline = baseline
        self.phase = phase

    def carbon_emissions(self, e: np.ndarray, emission_rate: Sequence[float]) -> float:
        """Carbon emissions G.

        Parameters
        ----------
        e: np.ndarray
            Net electricity consumption of shape (time, buildings).
        emission_rate: Sequence[float]
            Emission rate per timestep.
        """
        e = np.maximum(e, 0.0)
        g_i = np.sum(e * np.asarray(emission_rate)[:, None], axis=0)
        return float(np.mean(g_i))

    def unmet_hours(self, temp: np.ndarray, setpoint: np.ndarray, band: float, occupancy: np.ndarray) -> float:
        """Thermal discomfort U."""
        diff = np.abs(temp - setpoint) > band
        unmet = diff & (occupancy > 0)
        return float(np.mean(unmet))

    def ramping(self, district_consumption: Sequence[float]) -> float:
        e = np.asarray(district_consumption)
        return float(np.sum(np.abs(np.diff(e))))

    def load_factor(self, district_consumption: Sequence[float], hours_per_day: int = 24) -> float:
        e = np.asarray(district_consumption)
        days = len(e) // hours_per_day
        ratios = []
        for d in range(days):
            daily = e[d*hours_per_day:(d+1)*hours_per_day]
            ratios.append(np.mean(daily) / np.max(daily))
        lf = np.mean(ratios)
        return 1.0 - lf

    def daily_peak(self, district_consumption: Sequence[float], hours_per_day: int = 24) -> float:
        e = np.asarray(district_consumption)
        days = len(e) // hours_per_day
        peaks = [np.max(e[d*hours_per_day:(d+1)*hours_per_day]) for d in range(days)]
        return float(np.mean(peaks))

    def all_time_peak(self, district_consumption: Sequence[float]) -> float:
        return float(np.max(district_consumption))

    def thermal_resilience(self, temp: np.ndarray, setpoint: np.ndarray, band: float,
                            occupancy: np.ndarray, outage: np.ndarray) -> float:
        diff = np.abs(temp - setpoint) > band
        unmet = diff & (occupancy > 0) & outage.astype(bool)
        return float(np.mean(unmet))

    def unserved_energy(self, demand: np.ndarray, served: np.ndarray, outage: np.ndarray) -> float:
        unmet = np.where(outage, np.maximum(demand - served, 0.0), 0.0)
        expected = np.where(outage, demand, 0.0)
        return float(unmet.sum() / (expected.sum() + 1e-9))

    def score(self, kpis: Mapping[str, float]) -> float:
        score_emissions = 1.0 - kpis['carbon_emissions'] / self.baseline['carbon_emissions']
        score_ramping = 1.0 - kpis['ramping'] / self.baseline['ramping']
        score_load_factor = 1.0 - kpis['1-load_factor'] / self.baseline['1-load_factor']
        score_daily_peak = 1.0 - kpis['daily_peak'] / self.baseline['daily_peak']
        score_all_time_peak = 1.0 - kpis['all_time_peak'] / self.baseline['all_time_peak']
        score_grid = (score_ramping + score_load_factor + score_daily_peak + score_all_time_peak) / 4.0
        score_thermal_resilience = 1.0 - kpis['1-thermal_resilience']
        score_unserved_energy = 1.0 - kpis['normalized_unserved_energy']
        score_resilience = (score_thermal_resilience + score_unserved_energy) / 2.0
        score_comfort = 1.0 - kpis['unmet_hours']
        weights = self.phase
        score_control = (
            weights.w1 * score_comfort +
            weights.w2 * score_emissions +
            weights.w3 * score_grid +
            weights.w4 * score_resilience
        )
        return score_control
