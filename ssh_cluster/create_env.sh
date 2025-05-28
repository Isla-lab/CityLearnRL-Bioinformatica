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
    echo -e "\n${GREEN}Environment setup complete!${NC}"
    echo -e "To activate this environment in the future, run:\n    ${YELLOW}conda activate $ENV_NAME${NC}"
    echo -e "\nTo install the required packages, run:\n    ${YELLOW}./install_packages.sh${NC}"
fi
