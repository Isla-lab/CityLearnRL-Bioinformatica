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

# Ensure pip is installed and at the right version
python -m ensurepip --upgrade
python -m pip install --upgrade pip==23.3.2

# Install setuptools and wheel with specific versions
python -m pip install --no-cache-dir setuptools==59.5.0 "wheel<0.40.0"

# Install core dependencies first
python -m pip install --no-cache-dir "numpy<1.24.0" "pyyaml>=5.4.1" "pandas>=1.0.0" "tqdm>=4.50.0"

# Install gym and its dependencies
python -m pip install --no-cache-dir "gym==0.21.0" "shimmy>=0.1.0"

# Install stable-baselines3 and CityLearn
python -m pip install --no-cache-dir "stable-baselines3==1.8.0" "CityLearn==2.1.2"

# Install visualization packages
python -m pip install --no-cache-dir "matplotlib>=3.3.0" "seaborn>=0.11.0" "ipywidgets>=7.6.0" "Pillow>=9.0.0"

# Install PyTorch with CPU support
python -m pip install --no-cache-dir torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install system dependencies if available (skip if no sudo access)
if command -v apt-get &> /dev/null; then
    status "Installing system dependencies..."
    sudo -k apt-get update && sudo apt-get install -y python3-tk python3-dev graphviz 2>/dev/null || warning "Failed to install system dependencies (sudo access required)"
fi

# Verify installations
status "Verifying installations..."
python -c "
import sys
import importlib.metadata as metadata
from importlib import import_module

# Required packages with minimum versions
required = {
    'numpy': '1.20.0',
    'torch': '1.10.0',
    'gym': '0.21.0',
    'stable_baselines3': '1.8.0',
    'citylearn': '2.1.2',
    'matplotlib': '3.3.0',
    'Pillow': '9.0.0',
    'pandas': '1.0.0',
    'pyyaml': '5.4.1',
    'tqdm': '4.50.0',
    'requests': '2.25.0',
    'beautifulsoup4': '4.9.0',
    'shimmy': '0.1.0'
}

success = True
for pkg, min_version in required.items():
    try:
        # Try to get the installed version
        installed_version = metadata.version(pkg)
        
        # Compare versions
        from packaging import version
        if version.parse(installed_version) < version.parse(min_version):
            print(f'Warning: {pkg} version {installed_version} is below minimum required {min_version}')
            success = False
        else:
            print(f'✓ {pkg} {installed_version} >= {min_version}')
            
        # Try to import the package
        import_module(pkg.replace('-', '_'))
        
    except metadata.PackageNotFoundError:
        print(f'Error: {pkg} is not installed')
        success = False
    except ImportError as e:
        print(f'Error importing {pkg}: {str(e)}')
        success = False

if success:
    print('\n✅ All required packages are installed and meet version requirements!')
    print('✅ All packages imported successfully!')
else:
    print('\n❌ Some packages failed verification')
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
