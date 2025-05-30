#!/bin/bash

# Exit on any error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Environment name
ENV_NAME="citylearn_td3"

# Function to print status messages
status() {
    echo -e "${GREEN}[STATUS]${NC} $1"
}

# Function to print warnings
warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to print errors and exit
error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    error "Conda is not installed. Please install Miniconda or Anaconda first."
fi

# Clean up existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    status "Removing existing environment: $ENV_NAME"
    conda deactivate 2>/dev/null || true
    conda remove -n $ENV_NAME --all -y || warning "Failed to remove existing environment"
fi

# Create a new conda environment with Python 3.8.20
status "Creating conda environment: $ENV_NAME with Python 3.8.20"
conda create -n $ENV_NAME python=3.8.20 -y || error "Failed to create conda environment"

# Initialize conda for shell interaction
status "Initializing conda..."
eval "$(conda shell.bash hook)" || error "Failed to initialize conda"

# Activate the environment
status "Activating environment: $ENV_NAME"
conda activate $ENV_NAME || error "Failed to activate environment"

# Verify Python version
CURRENT_PYTHON=$(python --version 2>&1 | cut -d ' ' -f2)
if [[ "$CURRENT_PYTHON" != "3.8.20" ]]; then
    error "Python version mismatch. Expected 3.8.20, found $CURRENT_PYTHON"
else
    status "Python version confirmed: $CURRENT_PYTHON"
fi

# Install specific versions of packages from PPO_TD3_tutorial.ipynb
status "Installing Python packages with specific versions from PPO_TD3_tutorial..."

# First install setuptools and wheel with specific versions
pip install --no-cache-dir setuptools==65.5.0 "wheel<0.40.0" || error "Failed to install setuptools and wheel"

# Install main packages
pip install --no-cache-dir \
    gym==0.21.0 \
    stable-baselines3==1.8.0 \
    CityLearn==2.1.2 \
    ipywidgets \
    matplotlib \
    seaborn \
    shimmy \
    requests \
    beautifulsoup4 \
    || error "Failed to install main packages"

# Install PyTorch with CPU support (version not specified in the tutorial, using a compatible one)
pip install --no-cache-dir torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html \
    || warning "Failed to install PyTorch with specific version, trying without version constraint"

# If PyTorch installation with specific version failed, try without version constraint
if ! python -c "import torch" &> /dev/null; then
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
        || error "Failed to install PyTorch"
fi

# Install system dependencies if available (won't fail if not available)
if command -v apt-get &> /dev/null; then
    status "Installing system dependencies..."
    sudo apt-get update && sudo apt-get install -y python3-tk python3-dev graphviz || warning "Failed to install system dependencies"
fi

# Verify installations
status "Verifying installations..."
python -c "
import sys
import pkg_resources

required = {
    'numpy': '1.24.3',
    'torch': '2.0.1',
    'gym': '0.21.0',
    'stable_baselines3': '1.8.0',
    'citylearn': '2.0.0',
    'matplotlib': '3.7.1',
    'Pillow': '10.0.0',
    'pyparsing': '3.0.9'
}

success = True
for pkg, version in required.items():
    try:
        installed = pkg_resources.get_distribution(pkg).version
        if installed != version:
            print(f'Warning: {pkg} version mismatch. Expected {version}, got {installed}')
            success = False
    except pkg_resources.DistributionNotFound:
        print(f'Error: {pkg} is not installed')
        success = False

if success:
    print('All required packages are installed and versions match!')
    # Test imports
    try:
        import torch
        import gym
        import stable_baselines3
        import citylearn
        import matplotlib
        print('All packages imported successfully!')
    except ImportError as e:
        print(f'Error during import test: {e}')
        success = False

if not success:
    sys.exit(1)
" || error "Verification failed"

# Create necessary directories
mkdir -p logs models

# Make the script executable
chmod +x td3_training.py

# Print success message
echo -e "\n${GREEN}Setup completed successfully!${NC}"
echo -e "To activate the environment, run: ${YELLOW}conda activate $ENV_NAME${NC}"
echo -e "To run the training script: ${YELLOW}python td3_training.py${NC}"
echo -e "\nNote: If you're using this on a cluster, you might need to load CUDA modules first."
echo -e "Check with your cluster's documentation for the appropriate module commands."
