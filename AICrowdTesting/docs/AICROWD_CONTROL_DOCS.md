# AICrowdControl Module Documentation

## Overview
The `AICrowdControl` module implements the reward calculation logic for the CityLearn Challenge Control Track. It provides a flexible framework for evaluating building control strategies based on multiple Key Performance Indicators (KPIs) including energy efficiency, grid impact, and occupant comfort.

## Table of Contents
1. [Core Classes](#core-classes)
2. [Key Performance Indicators (KPIs)](#key-performance-indicators)
3. [Reward Calculation](#reward-calculation)
4. [Usage Examples](#usage-examples)
5. [Implementation Details](#implementation-details)
6. [Phase Weights](#phase-weights)

## Core Classes

### `PhaseWeights`
A dataclass that defines the weight distribution for different components of the reward calculation.

**Attributes:**
- `w1` (float): Weight for comfort component (default: 0.3)
- `w2` (float): Weight for emissions component (default: 0.1)
- `w3` (float): Weight for grid component (default: 0.6)
- `w4` (float): Weight for resilience component (default: 0.0)

### `ControlTrackReward`
Main class responsible for calculating the control track reward based on various KPIs.

**Initialization:**
```python
def __init__(self, baseline: Mapping[str, float], phase: Union[dict, PhaseWeights] = None)
```
- `baseline`: Dictionary containing baseline KPI values for normalization
- `phase`: Either a `PhaseWeights` instance or dictionary with weights (w1-w4)

## Key Performance Indicators

The module calculates several KPIs that contribute to the final reward:

### 1. Carbon Emissions (`carbon_emissions`)
- **Purpose**: Measures the total carbon footprint of the building operations
- **Formula**: \( G = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} e_{i,t} \cdot c_t \)
  where \( e_{i,t} \) is the net electricity consumption and \( c_t \) is the emission rate

### 2. Thermal Comfort (`unmet_hours`)
- **Purpose**: Measures occupant discomfort due to temperature deviations
- **Formula**: \( U = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \mathbb{1}(|T_{i,t} - S_{i,t}| > B) \cdot \mathbb{1}(O_{i,t} > 0) \)
  where \( T \) is temperature, \( S \) is setpoint, \( B \) is comfort band, and \( O \) is occupancy

### 3. Grid Impact Metrics
- **Ramping**: Measures load variability between consecutive timesteps
- **Load Factor**: Measures the ratio of average to peak load
- **Daily Peak**: Tracks the maximum load during each day
- **All-time Peak**: Tracks the overall maximum load

### 4. Resilience Metrics
- **Thermal Resilience**: Measures comfort violations during grid outages
- **Unserved Energy**: Quantifies energy demand not met during outages

## Reward Calculation

The final reward is a weighted sum of four main components:

\[ \text{score} = w_1 \cdot S_{\text{comfort}} + w_2 \cdot S_{\text{emissions}} + w_3 \cdot S_{\text{grid}} + w_4 \cdot S_{\text{resilience}} \]

Where:
- \( S_{\text{comfort}} = 1 - U \) (normalized comfort score)
- \( S_{\text{emissions}} = 1 - G/G_{\text{baseline}} \) (normalized emissions score)
- \( S_{\text{grid}} \): Average of normalized grid metrics
- \( S_{\text{resilience}} \): Average of resilience metrics

## Phase Weights

Two predefined phases are available:

1. **Phase I** (Default):
   - Comfort (w1): 0.3
   - Emissions (w2): 0.1
   - Grid (w3): 0.6
   - Resilience (w4): 0.0

2. **Phase II**:
   - Comfort (w1): 0.3
   - Emissions (w2): 0.1
   - Grid (w3): 0.3
   - Resilience (w4): 0.3

## Usage Examples

```python
# Initialize with default Phase I weights
reward_calculator = ControlTrackReward(baseline=BASELINE_KPIS)

# Or specify custom weights
custom_weights = PhaseWeights(w1=0.4, w2=0.1, w3=0.4, w4=0.1)
reward_calculator = ControlTrackReward(baseline=BASELINE_KPIS, phase=custom_weights)

# Calculate KPIs
kpis = {
    'carbon_emissions': 0.8,  # Example values
    'ramping': 0.9,
    '1-load_factor': 0.7,
    'daily_peak': 120.5,
    'all_time_peak': 150.0,
    'unmet_hours': 0.05,
    '1-thermal_resilience': 0.9,
    'normalized_unserved_energy': 0.1
}

# Get final score
score = reward_calculator.score(kpis)
```

## Implementation Notes

1. **Normalization**: All KPIs are normalized against baseline values to ensure fair comparison
2. **Handling Zeros**: Division by zero is prevented using small epsilon values
3. **Vectorization**: Uses NumPy for efficient array operations
4. **Flexibility**: Supports both Phase I and Phase II of the challenge through configurable weights

## References
- CityLearn Challenge Documentation
- Related academic papers on building control and reinforcement learning
