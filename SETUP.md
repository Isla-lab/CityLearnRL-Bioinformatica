# Setup Instructions

## Prerequisites

- Python 3.8+
- Git
- Jupyter Notebook/Lab

## Installation

1. Clone the repository:
```bash
git clone https://github.com/djacoo/CityLearnRL-Bioinformatica.git
cd CityLearnRL-Bioinformatica
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
CityLearnRL-Bioinformatica/
├── notebooks/                # Jupyter notebooks
│   ├── data/                # Input data
│   ├── models/              # Trained models
│   ├── results/             # Experiment results
│   └── plots/               # Generated plots
├── src/                     # Source code
│   ├── __init__.py
│   ├── citylearn/           # CityLearn implementation
│   └── utils/              # Utility functions
├── tests/                   # Test files
├── config/                  # Configuration files
└── docs/                    # Documentation
```

## Running Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open the desired notebook from the `notebooks/` directory

## Development

1. Run tests:
```bash
pytest tests/
```

2. Format code:
```bash
black src/ notebooks/
```

3. Lint code:
```bash
flake8 src/ notebooks/
```
