#!/bin/bash

echo "📈 Starting plot generation..."

# Single plots
python3 -c "from utils.plot_utils import plot_single; \
plot_single('notebooks/results/log_PPO.csv', 'PPO', 'notebooks/results/plot_PPO')"

python3 -c "from utils.plot_utils import plot_single; \
plot_single('notebooks/results/log_TD3.csv', 'TD3', 'notebooks/results/plot_TD3')"

# Combined plot
python3 -c "from utils.plot_utils import plot_combined; \
plot_combined(['notebooks/results/log_PPO.csv', 'notebooks/results/log_TD3.csv'], \
['PPO', 'TD3'], 'notebooks/combined_plot')"

echo "✅ Plot generation successful and located in -dir 'results/'"
