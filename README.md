
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

# 💻 Tutorial sul training e sul plotting dei risultati

Assicurarsi di essere posizionati nella root e, possibilmente, con ambiente **Conda** attivo per facilitare l'installazione e l'uso dei pacchetti.

1. Installare i pacchetti necessari tramite pip:

```bash
pip install -r requirements.txt
```

2. Assicurarsi che in data/ sia presente una directory contenente lo **schema.json** e i vari .csv degli edifici da analizzare, oltre ai dataset **weather.csv** e **carbon_intensity.csv**

3. Spostarsi in src/

```bash
cd src/
```

4. Eseguire lo script **ppo_train.py** da terminale. Il completamento dello script puó essere lungo o breve a seconda dei total_episodes (puó essere sistemato in fase di tuning con la modifica degli stessi e di altri parametri)

```bash
python3 ppo_train.py
```

Sul terminale a questo punto, appaiono in sequenza box contenenti i risultati degli addestramenti per episodio, nonché una verifica che il singolo episodio sia completato correttamente e venga salvato in **results/**.

5. Terminato il training, si possono visualizzare a terminale le sequenze di dati e, tramite funzione **plot_rewards.py**, é possibile visualizzare un plot a schermo dell'andamento reward/episodio

```bash
python3 plot_rewards.py
```

6. Infine, nella dir presente in root **results/ppo/** sono presenti X file .zip e X file .npy contenenti i dati di reward per ogni training. In caso di multiple_seed_training && plotting, é presente anche **rewards_plot.png** (da implementare anche nel singolo seed) con smoothing e shadows.






