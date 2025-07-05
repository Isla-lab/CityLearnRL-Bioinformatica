# CityLearn RL Training

This repository contains a minimal reinforcement learning setup for experimenting with the CityLearn challenge KPIs. It includes a custom Gym environment, reward calculation utilities and training scripts based on Stable Baselines3.

## Directory Structure

- `src/citylearn_rl/` – Python package with the environment and reward modules.
- `scripts/` – Command line utilities for training and plotting results.
- `results/` – Location for models, logs, tensorboard files and evaluation results.
- `docs/` – Additional documentation and analysis.

## Installation

1. Create a virtual environment (optional but recommended).
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the agent with default settings run:
```bash
python scripts/train.py
```
Models and logs will be saved in `results/`.

## Visualizing Progress

TensorBoard logs are saved in `results/ppo_tensorboard_logs_*/`. Launch TensorBoard with:
```bash
tensorboard --logdir results/ppo_tensorboard_logs_2kpi
```
Replace the path with the multi KPI log directory to visualize the second training phase.

You can also generate summary plots using:
```bash
python scripts/plot_training.py
```

## Analyzing TensorBoard Logs

To extract scalar values from TensorBoard event files and save them to CSV:
```bash
python scripts/analyze_tb.py
```

---

See the documents in `docs/` for more details on KPI calculations and analysis results.
