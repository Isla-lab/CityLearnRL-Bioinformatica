# 🏙️ CityLearnRL-Bioinformatics: Advanced RL Algorithms for Urban Energy Optimization

This project explores the application of advanced Reinforcement Learning (RL) algorithms—specifically Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO), and Twin Delayed DDPG (TD3)—for optimizing urban energy consumption using the CityLearn simulation environment. Building upon the foundational SAC implementation from the official CityLearn tutorial, this work extends the analysis to include PPO and TD3, providing a comparative study of their performance in terms of reward, stability, and learning speed. The goal is to identify effective RL strategies for smart grid management and energy efficiency in urban settings.

[![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of Contents

*   [About The Project](#about-the-project)
*   [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
*   [Usage](#usage)
*   [Results](#results)
*   [Project Status](#project-status)
*   [Documentation](#documentation)
    *   [Contributing Guidelines](#contributing-guidelines)
    *   [Code of Conduct](#code-of-conduct)
    *   [Setup Instructions](#setup-instructions)
    *   [Technical Compatibility](#technical-compatibility)
*   [License](#license)
*   [Author](#author)
*   [Acknowledgements](#acknowledgements)
*   [Citation](#citation)
*   [Glossary](#glossary)

---

## About The Project

CityLearn is an open-source OpenAI Gym environment for research in demand response and smart grid energy management. It simulates building energy dynamics, enabling the development and testing of control strategies. This project uses CityLearn to assess modern RL algorithms (SAC, PPO, TD3) for complex energy optimization. By comparing these algorithms, we aim to highlight their effectiveness for efficiently managing building energy resources—cutting costs, reducing emissions, and enhancing grid stability—especially in dynamic environments where traditional methods may be insufficient, thereby contributing to more intelligent and adaptive urban energy systems.

The original objectives were:
- Build upon the **tutorial.ipynb** notebook from the official CityLearn repository, which implements the **SAC (Soft Actor-Critic)** algorithm.
- Extend it with additional advanced algorithms:
    - 🔁 **PPO** (Proximal Policy Optimization)  
    - 🎯 **TD3** (Twin Delayed DDPG)
- Analyze and **compare the performance** of these algorithms in terms of:
    - Reward
    - Stability
    - Learning Speed

---

## Getting Started

### Prerequisites

*   Python 3.8.x
*   pip
It is highly recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/djacoo/CityLearnRL-Bioinformatics.git
    cd CityLearnRL-Bioinformatics
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary dependencies, including CityLearn, Stable Baselines3 (v1.x), and Gym (v0.21.0). For detailed compatibility information, see the [Technical Compatibility](#technical-compatibility) section.

---

## Usage

This project primarily uses Jupyter Notebooks for RL algorithm implementation and analysis.

**Running the Notebooks:**
*   `notebooks/tutorial.ipynb`: This is the original CityLearn tutorial notebook focusing on the SAC algorithm.
*   `notebooks/PPO_TD3_tutorial.ipynb`: This notebook contains the implementations and comparative analysis of PPO and TD3 algorithms, alongside SAC.
To run these, start Jupyter Lab from the root directory:
```bash
jupyter lab
```
Then, navigate to the `notebooks/models/` directory in the Jupyter interface and open the desired notebook.
Then, open the desired notebook and execute the cells.

**Generating Plots:**
Visualizations from saved results (e.g., CSV files in `notebooks/results/`) can be generated using the analysis script:
```bash
bash scripts/generate_plots.sh
```
This script processes data from `notebooks/results/` and saves plots in `notebooks/plots/`.

**Models and Results:**
*   **Saved Models:** Trained RL models are stored in `notebooks/models/` (e.g., `ppo_model_seed0.zip`).
*   **Raw Results:** CSV files containing logs and rewards data are saved in `notebooks/results/` (e.g., `log_PPO.csv`, `ppo_rewards_seed0.csv`).
*   **Plots:** Visualizations are found in `notebooks/plots/` (e.g., `combined_plot.png`, `PPO_reward_plot.png`, `TD3_reward_plot.png`).

---

## Results

The project conducts a comparative analysis of SAC, PPO, and TD3 algorithms based on metrics such as cumulative reward, learning stability, and convergence speed. Detailed performance metrics and visualizations can be found within the `notebooks/PPO_TD3_tutorial.ipynb` notebook.
Key comparative plots include:
*   Combined performance plot: `notebooks/combined_plot/combined_plot.png`
*   PPO-specific plots: `notebooks/results/plot_PPO/PPO_reward_plot.png`
*   TD3-specific plots: `notebooks/results/plot_TD3/TD3_reward_plot.png`
The raw CSV data backing these results is available in the `notebooks/results/` directory.

---

## Project Status

Status: **Work in Progress**. This project was developed as part of a Bachelor's Thesis and is now work-in-progress. It may receive occasional maintenance for compatibility or bug fixes.

---

## Documentation

For more detailed information about the project, please check the documentation in the [docs](docs/) directory:

*   [Contributing Guidelines](docs/CONTRIBUTING.md)
*   [Code of Conduct](docs/CODE_OF_CONDUCT.md)
*   [Setup Instructions](docs/SETUP.md)
*   [Technical Compatibility](docs/COMPATIBILITY_GUIDE.md)

## Contributing

Contributions are welcome! Please follow our [Contributing Guidelines](docs/CONTRIBUTING.md) and review our [Code of Conduct](docs/CODE_OF_CONDUCT.md) before submitting any contributions.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Author

**Jacopo Parretti**  
Bachelor’s Thesis in **Bioinformatics** at the **Department of Computer Science**, University of Verona

📧 **Email**: [jacopo.parretti@gmail.com](mailto:jacopo.parretti@gmail.com)  
📄 **GitHub**: [github.com/djacoo](https://github.com/djacoo)

---

## Acknowledgements

This project builds upon the CityLearn environment and the original SAC tutorial provided by the CityLearn developers. I acknowledge their foundational work which made this research possible.

---


## Glossary

| Acronym | Description |
|--------|-------------|
| **AI**   | Artificial Intelligence |
| **API**  | Application Programming Interface |
| **DER**  | Distributed Energy Resource |
| **ESS**  | Energy Storage System |
| **EV**   | Electric Vehicle |
| **GEB**  | Grid-Interactive Efficient Building |
| **GHG**  | Greenhouse Gas |
| **HVAC** | Heating, Ventilation and Air Conditioning |
| **KPI**  | Key Performance Indicator |
| **MPC**  | Model Predictive Control |
| **PV**   | Photovoltaic |
| **RBC**  | Rule-Based Control |
| **RLC**  | Reinforcement Learning Control |
| **SoC**  | State of Charge |
| **TES**  | Thermal Energy Storage |
| **ToU**  | Time of Use |
| **ZNE**  | Zero Net Energy |

---

📌 The glossary is continuously being updated and will be expanded with new terms throughout the development of the project.

---

## Technical Compatibility

For detailed information on package versions (Python, Gym, Stable-Baselines3, CityLearn), compatibility challenges, and specific steps required to align with older Gym/SB3 releases for reproducing the original tutorial environment, please see the [Technical Compatibility Guide](docs/COMPATIBILITY_GUIDE.md) in the documentation.

This guide covers:
*   Original and target versions for Python, Gym, and Stable-Baselines3.
*   Required code changes for Gym 0.21 API.
*   SB3 API and wrapper adjustments.
*   Version alignment and dependency details.
*   Observed issues and their resolutions.

---

## Original Repo Links

The **CityLearn project** is an open-source simulator for urban energy optimization. You can find the original repository at the following [GitHub link](https://github.com/CityLearn/CityLearn).

For more information about the project and to run the base code, visit the official repository:

- [CityLearn GitHub Repository](https://github.com/CityLearn/CityLearn)

---
