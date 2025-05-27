#!/bin/bash
#SBATCH --job-name=td3_training
#SBATCH --output=logs/td3_%j.out
#SBATCH --error=logs/td3_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1  # If you have GPU available

# Load required modules
module load python/3.8
module load cuda/11.1  # If using GPU

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run training for different seeds
for seed in {0..4}
do
    echo "Running TD3 with seed $seed"
    python train_td3.py --seed $seed --episodes 1000 --save-dir results
done