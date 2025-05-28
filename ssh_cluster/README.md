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

## Remote Execution (SSH)

If running on a remote server:

1. Transfer the files to the server:
   ```bash
   scp -r /local/path/to/project username@server:~/remote/path/
   ```

2. SSH into the server:
   ```bash
   ssh username@server
   ```

3. Follow the setup and running instructions above.

4. To download the results:
   ```bash
   # From your local machine
   scp username@server:/remote/path/to/plot.png .
   ```

## Customization

- **Hyperparameters**: Modify the hyperparameters in `td3_training.py`
- **Training duration**: Adjust `total_training_steps` in the script
- **Environment**: Change the CityLearn environment configuration as needed

## Troubleshooting

- If you get package errors, try:
  ```bash
  conda activate citylearn_td3
  pip install -r requirements.txt --upgrade
  ```

- For visualization issues, ensure you're using the Agg backend (already set in the script) for non-interactive environments.
