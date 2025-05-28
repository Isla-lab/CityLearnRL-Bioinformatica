#!/bin/bash

# Exit on any error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Check if we're in the correct environment
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "citylearn_td3" ]]; then
    error "Please activate the citylearn_td3 environment first. Run:\n    source create_env.sh"
fi

# Verify Python version
CURRENT_PYTHON=$(python --version 2>&1 | cut -d ' ' -f2)
if [[ "$CURRENT_PYTHON" != "3.8.20" ]]; then
    error "Python version mismatch. Expected 3.8.20, found $CURRENT_PYTHON"
else
    status "Python version confirmed: $CURRENT_PYTHON"
fi

# Install specific versions of packages from PPO_TD3_tutorial.ipynb
status "Installing Python packages with specific versions..."

# First downgrade pip to a version that works with gym 0.21.0
status "Ensuring compatible pip version..."
python -m pip install --upgrade pip==23.3.2 || error "Failed to set pip version"

# Install setuptools and wheel with specific versions
python -m pip install --no-cache-dir setuptools==65.5.0 "wheel<0.40.0" || error "Failed to install setuptools and wheel"

# Install main packages
status "Installing main packages..."

# First install gym with --no-deps to avoid dependency conflicts
python -m pip install --no-cache-dir --no-deps gym==0.21.0 || error "Failed to install gym"

# Then install other packages
python -m pip install --no-cache-dir \
    stable-baselines3==1.8.0 \
    CityLearn==2.1.2 \
    ipywidgets \
    matplotlib \
    seaborn \
    shimmy \
    requests \
    beautifulsoup4 \
    tensorboard \
    pandas \
    scikit-learn \
    tqdm \
    pyyaml \
    rich \
    || error "Failed to install main packages"

# Install PyTorch with CUDA support if available
status "Checking for CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    # Install PyTorch with CUDA 11.3 (compatible with most modern GPUs)
    status "CUDA detected! Installing PyTorch with CUDA support..."
    python -m pip install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    
    # Verify CUDA is available
    if python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}');" | grep -q "True"; then
        status "PyTorch with CUDA installed successfully!"
    else
        warning "CUDA installation might have failed. Falling back to CPU version."
        python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
else
    status "No CUDA detected. Installing CPU-only PyTorch..."
    python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch installation
status "Verifying PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA version: N/A')
    print('Current device: CPU')
    print('Device name: CPU')
" || error "Failed to verify PyTorch installation"

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
    'gym': '0.21.0',
    'stable_baselines3': '1.8.0',
    'citylearn': '2.1.2',
    'torch': '1.10.0'
}

success = True
for pkg, version in required.items():
    try:
        installed = pkg_resources.get_distribution(pkg).version
        print(f'{pkg}: {installed}')
        if installed != version:
            print(f'Warning: {pkg} version mismatch. Expected {version}, got {installed}')
            success = False
    except pkg_resources.DistributionNotFound:
        print(f'Error: {pkg} is not installed')
        success = False

if success:
    print('\nAll required packages are installed with correct versions!')
    # Test imports
    try:
        import torch
        import gym
        import stable_baselines3
        import citylearn
        print('\nAll packages imported successfully!')
    except ImportError as e:
        print(f'\nError during import test: {e}')
        success = False

if not success:
    sys.exit(1)
"

status "Installation complete!"
echo -e "To activate this environment in the future, run:\n    ${YELLOW}conda activate citylearn_td3${NC}"
