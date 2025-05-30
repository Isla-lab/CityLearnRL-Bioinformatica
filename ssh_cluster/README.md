# CityLearn TD3 Training

This project implements a TD3 (Twin Delayed DDPG) agent for the CityLearn environment. The code is designed to be portable across different machines and can be run on both local and remote systems.

## Setup

1. **Prerequisites**:
   - Miniconda or Anaconda installed
   - Git (for cloning the repository)

2. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

3. **Setup the environment**:
   ```bash
   # Make the setup script executable
   chmod +x setup.sh
   
   # Run the setup script
   ./setup.sh
   ```
   This will:
   - Create a new conda environment called `citylearn_td3`
   - Install all required Python packages

## Running the Training

1. **Activate the environment**:
   ```bash
   conda activate citylearn_td3
   ```

2. **Run the training script**:
   ```bash
   python td3_training.py
   ```

## Monitoring and Results

- **Logs**: Training logs are saved in the `logs` directory
- **Models**: Trained models are saved in the `models` directory
- **Plots**: Evaluation plots are saved in the current directory with timestamps

# CityLearn TD3 Training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable_Baselines3-1.8.0-9cf.svg)](https://stable-baselines3.readthedocs.io/)

A robust implementation of Twin Delayed DDPG (TD3) for the CityLearn environment, featuring automated setup, comprehensive logging, and support for both local and remote execution.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Monitoring & Evaluation](#monitoring--evaluation)
- [Remote Execution](#remote-execution)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- 🚀 **TD3 Implementation**: State-of-the-art reinforcement learning algorithm
- 🏙️ **CityLearn Integration**: Optimized for building energy management tasks
- 📊 **Comprehensive Logging**: TensorBoard integration and model checkpointing
- ☁️ **Cloud Ready**: Easy setup for remote execution on GPU instances
- 🔄 **Reproducible**: Environment pinning and deterministic training options
- 📈 **Performance Optimized**: Includes reward scaling and environment wrappers

## Prerequisites

- Linux/macOS (Windows not officially supported)
- Miniconda/Anaconda
- Git
- NVIDIA GPU (optional but recommended for faster training)

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/CityLearnRL-Bioinformatica.git
   cd CityLearnRL-Bioinformatica/ssh_cluster
   ```

2. **Setup the environment**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Start training**:
   ```bash
   conda activate citylearn_td3
   python td3_training.py
   ```

## Project Structure

```
ssh_cluster/
├── README.md               # This documentation
├── create_env.sh           # Conda environment creation
├── install_packages.sh     # Package installation
├── requirements.txt        # Python dependencies
├── setup.sh               # Main setup script
├── sync.sh                # Remote sync utility
├── td3_training.py        # Main training script
└── logs/                  # Training outputs
    └── td3_citylearn_*/   # Timestamped runs
        ├── best_model/    # Best model checkpoints
        ├── final_model.zip # Final trained model
        ├── hyperparams.json # Training configuration
        └── training_metrics.json # Performance metrics
```

## Configuration

### Hyperparameters
Key training parameters in `td3_training.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0003 | Learning rate for both actor and critic |
| `batch_size` | 256 | Minibatch size for training |
| `buffer_size` | 1000000 | Replay buffer size |
| `learning_starts` | 10000 | Steps before learning starts |
| `train_freq` | 1 | Update the model every `train_freq` steps |
| `gradient_steps` | 1 | How many gradient steps after each rollout |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Target network update rate |
| `policy_delay` | 2 | Policy update delay for TD3 |

### Environment Configuration
Modify the CityLearn environment setup in `td3_training.py` to customize:
- Building configurations
- Weather data
- Reward function
- Simulation parameters

## Training

### Starting Training
```bash
# Basic training
python td3_training.py

# With custom log directory
python td3_training.py --log-dir custom_logs
```

### Training Parameters
```bash
python td3_training.py \
    --total-timesteps 1000000 \
    --learning-rate 0.0003 \
    --batch-size 256 \
    --gamma 0.99
```

## Monitoring & Evaluation

### TensorBoard
Monitor training progress with TensorBoard:
```bash
tensorboard --logdir=logs/
```

### Model Evaluation
Evaluate a trained model:
```python
from stable_baselines3 import TD3

model = TD3.load("logs/td3_citylearn_<timestamp>/best_model")
mean_reward, std_reward = evaluate_model(model, env, num_episodes=10)
print(f"Mean reward: {mean_reward} ± {std_reward}")
```

## Remote Execution

### 1. Initial Setup on Remote Server
```bash
# On your local machine
scp -r ssh_cluster/ user@remote:~/citylearn/

# SSH into remote server
ssh user@remote
cd ~/citylearn/ssh_cluster

# Setup environment
./setup.sh
```

### 2. Start Training in Background
```bash
# Using nohup to keep process running after logout
nohup python td3_training.py > training.log 2>&1 &

# Check training progress
tail -f training.log
```

### 3. Sync Results Locally
```bash
# From your local machine
rsync -avz user@remote:~/citylearn/ssh_cluster/logs/ ./logs/
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size: `--batch-size 128`
- Use gradient accumulation

**Training Instability**
- Try different random seeds
- Adjust learning rate
- Modify reward scaling

**Package Conflicts**
```bash
conda create -n fresh_env python=3.8
conda activate fresh_env
pip install -r requirements.txt --no-deps
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this code in your research, please cite:
```
@misc{citylearn2023,
  author = {Your Name},
  title = {CityLearn TD3 Implementation},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/CityLearnRL-Bioinformatica}}
}
```
