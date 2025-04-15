
# 🏙️ CityLearnRL-Bioinformatics

Questo repository raccoglie il codice e i risultati sviluppati durante il lavoro di tesi sull'utilizzo di algoritmi di **Reinforcement Learning (RL)** per l'ottimizzazione energetica urbana nel simulatore **CityLearn**.

---

## 🎯 Obiettivo del Progetto

L'obiettivo principale della tesi è partire dal notebook _`tutorial.ipynb`_ del repository ufficiale CityLearn, che utilizza l'algoritmo **SAC (Soft Actor-Critic)**, ed **estenderlo con altri algoritmi avanzati**:

- 🔁 **PPO** (Proximal Policy Optimization)  
- 🎯 **TD3** (Twin Delayed DDPG)

L'analisi si concentra sul **confronto delle performance** tra gli algoritmi in termini di:

- Reward
- Stabilità
- Rapidità di apprendimento

---

## 👨‍💻 Autore

**Jacopo Parretti**  
Tesi di Laurea Triennale in **Bioinformatica** presso il **Dipartimento di Informatica**, Università degli Studi di Verona  


📧 **Email**: [jacopo.parretti@gmail.com](mailto:jacopo.parretti@gmail.com)  
📄 **GitHub**: [github.com/djacoo](https://github.com/djacoo)

---



## 📁 Struttura del Repository

La struttura del progetto è la seguente **(IN AGGIORNAMENTO)**

- `tools/`                   : Contiene il notebook originale dal repository CityLearn (SAC)
- `src/`                     : dir contenente il codice sorgente
  - `sac_train.py`           : IN AGGIORNAMENTO
  - `ppo_train.py`           : PPO
  - `td3_train.py`           : IN AGGIORNAMENTO
  - `utils.py`               : utils for python RL
  - `wrappers.py`            : wrappers for python RL
  - `plot_rewards.py`        : matplotlib plotting single-seed
  - `train_multiple_rewards.py`    : training for multiple rewards
  - `plot_multiple_rewards.py`     : plotting for multiple rewards
- `notebooks/`               : Implementazione dell'algoritmo TD3
- `results/`                 : Risultati sperimentali (grafici, metriche, log)
  - `ppo/`                   : Results for PPO
  - `td3/`                   : Results for TD3
  - `sac/`                   : Results for SAC
- `data/`                    : Data for citylearn_challenge_2021 (schema.json && BuildingX data)
- `README.md`                : Documentazione del progetto
- `LICENSE`                  : MIT License for project
- `requirements.txt`         : bash pip install -r requirements.txt for software requirements in conda



---

## ⚙️ Requisiti

Per eseguire i notebook è necessario installare i seguenti pacchetti:

- `citylearn`
- `stable-baselines3`
- `matplotlib`
- `doe_xstock>=1.1.0`
- `gymnasium`
- `nrel-pysam`
- `numpy<2.0.0`
- `pandas`
- `pyyaml`
- `scikit-learn<=1.2.2`
- `simplejson`
- `torch`
- `torchvision`
- `openstudio<=3.3.0`

Puoi installarli tramite pip dalla root della repo:

```bash
pip install -r requirements.txt
```



---


## 🌐 Link al Repository Originale

Il progetto **CityLearn** è un simulatore open-source per l'ottimizzazione energetica urbana. Puoi trovare il repository originale sul seguente [link GitHub](https://github.com/CityLearn/CityLearn).

Per maggiori informazioni sul progetto e per eseguire il codice di base, visita il repository ufficiale:

- [CityLearn GitHub Repository](https://github.com/CityLearn/CityLearn)

---

# 📘 Glossario

Si fornisce di seguito il glossario presente in **tutorial.ipynb** nella repo originale, al fine di comprendere appieno i termini abbreviati e gli acronimi.
Questo glossario é pensato per facilitare la comprensione dei concetti legati ai sistemi energetici, agli edifici intelligenti e alle tecnologie di controllo.  

---

### 🔤 Acronimi e Definizioni

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

📌 *Il glossario è in continuo aggiornamento e sarà ampliato con nuovi termini nel corso dello sviluppo del progetto.*

---

# 💻 Requisiti Software

Questa sezione installa e importa i pacchetti software che verranno utilizzati nel resto del tutorial. Iniziamo verificando la versione di Python dell’ambiente corrente.  
CityLearn e le sue dipendenze funzionano correttamente con **Python >= 3.7.x**.

### 🔍 Verifica della versione di Python

```bash
!python --version
```

### 📦 Pacchetti Python richiesti

L'installazione dei seguenti pacchetti richiede circa 3 minuti per essere completata:

```python
%%capture

# Ambiente di simulazione principale
!pip install CityLearn==2.1.2

# Per interazioni con i partecipanti (es. pulsanti)
!pip install ipywidgets

# Per generare grafici statici
!pip install matplotlib
!pip install seaborn

# Algoritmi standard di Reinforcement Learning
!pip install stable-baselines3

# Compatibilità con gym per versioni successive di stable-baselines3
!pip install shimmy

# Per l'invio dei risultati
!pip install requests
!pip install beautifulsoup4
```

📌 Assicurarsi di eseguire l’installazione in un ambiente virtuale o notebook compatibile per evitare conflitti di dipendenze.

---

## 🧩 Importazione dei Moduli

Di seguito vengono importati tutti i moduli, classi e funzioni necessari per eseguire il tutorial. Le importazioni sono organizzate per ambito di utilizzo:

```python
# 🔧 Operazioni di sistema
import inspect
import os
import uuid

# 🗓️ Data e ora
from datetime import datetime

# 🧠 Type hinting
from typing import Any, List, Mapping, Tuple, Union

# 📊 Visualizzazione dei dati
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# 🎛️ Interazione con l'utente
from IPython.display import clear_output
from ipywidgets import Button, FloatSlider, HBox, HTML
from ipywidgets import IntProgress, Text, VBox

# 🧹 Manipolazione dei dati
from bs4 import BeautifulSoup
import math
import numpy as np
import pandas as pd
import random
import re
import requests
import simplejson as json

# 🏙️ CityLearn
from citylearn.agents.rbc import HourRBC
from citylearn.agents.q_learning import TabularQLearning
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn.wrappers import StableBaselines3Wrapper
from citylearn.wrappers import TabularQLearningWrapper

# 🤖 Algoritmi RL di baseline
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
```


## ⚙️ Impostazioni Globali del Notebook

Per uniformare la visualizzazione dei grafici nel notebook, viene disattivato il margine automatico delle figure:

```python
# Impostare i grafici matplotlib senza margini
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
```

📌 Queste impostazioni garantiscono che i grafici siano visualizzati in modo più compatto, migliorando la leggibilità durante l'analisi dei risultati.
