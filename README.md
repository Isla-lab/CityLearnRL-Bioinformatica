
# 🏙️ CityLearnRL-Bioinformatics

This repository contains the code and results developed as part of a thesis project focused on the use of **Reinforcement Learning (RL)** algorithms for urban energy optimization using the **CityLearn** simulator.

---

## 🎯 Objectives

The main objective of this thesis is to build upon the **tutorial.ipynb** notebook from the official CityLearn repository, which implements the **SAC (Soft Actor-Critic)** algorithm, and extend it with additional advanced algorithms:

- 🔁 **PPO** (Proximal Policy Optimization)  
- 🎯 **TD3** (Twin Delayed DDPG)

The analysis focuses on **comparing the performance** of the algorithms in terms of:

- Reward
- Stability
- Learning Speed

---

## 👨‍💻 Author

**Jacopo Parretti**  
Bachelor’s Thesis in **Bioinformatics** at the **Department of Computer Science**, University of Verona


📧 **Email**: [jacopo.parretti@gmail.com](mailto:jacopo.parretti@gmail.com)  
📄 **GitHub**: [github.com/djacoo](https://github.com/djacoo)

---


## 🌐 Original Repo Links

The **CityLearn project** is an open-source simulator for urban energy optimization. You can find the original repository at the following [GitHub link](https://github.com/CityLearn/CityLearn).

For more information about the project and to run the base code, visit the official repository:

- [CityLearn GitHub Repository](https://github.com/CityLearn/CityLearn)

---

# Installation

Install needed requirements by **requirements.txt** on root:

```python
pip install -r requirements.txt
```

---


# 📘 Glossary

Below is the glossary included in the original **tutorial.ipynb** from the official repository. It is provided to facilitate a full understanding of the abbreviated terms and acronyms.

This glossary is designed to support the comprehension of concepts related to energy systems, smart buildings, and control technologies. 

---

### 🔤 Acronyms & Descriptions

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

# 🛠️ Update Overview for CityLearn Tutorial on Python 3.8, Gym 0.21.0, and SB3

The CityLearn tutorial.ipynb and examples were originally written against older Gym and SB3 releases. In particular, CityLearn v2.0+ targeted Python 3.7+, OpenAI Gym 0.21.x, and Stable-Baselines3 1.x. (Later CityLearn versions switched to the Gymnasium API, but the tutorial code assumes the pre-Gymnasium Gym interface). Below I'll summarize the required changes and version alignments:

### 🧾 Original Versions (CityLearn examples)

- Python: CityLearn’s setup.py requires Python ≥3.7.7; the tutorial was typically run on Python 3.7–3.9. We target Python 3.8.20 to stay within this range.
- Gym: The tutorial was created for Gym 0.21.x (legacy OpenAI Gym) (CityLearn v2.0+ lists gym>=0.21.0 as a requirement.) Later CityLearn releases moved to Gymnasium, but for compatibility we use the older Gym API.
- Stable-Baselines3: The tutorial used SB3 in the 1.x series (e.g. 1.4–1.8). SB3 v1.5.0 (Mar 2022) “switched minimum Gym version to 0.21.0”, so any SB3 1.5+ will work with Gym 0.21. I recommend SB3 1.8.0 (the last 1.x release, Apr 2023) for full Gym support.

### 🔧 Code Changes for Gym 0.21 API

Gym 0.21 (pre-Gymnasium) differs from newer Gym/Gymnasium in the reset()/step() signatures. The tutorial code must be adjusted accordingly:

- **env.reset()** – In Gym 0.21 this returns only the initial observation (older Gym style). In contrast, Gymnasium’s env.reset() returns (obs, info). The original tutorial code expects a single return value, which matches Gym 0.21. If running against a Gymnasium-based CityLearn, change any unpacking of two values to use only one. For example:

```python
# Gymnasium style (newer): returns (obs, info)
observations, _ = env.reset()
# Gym 0.21 style: returns obs
observations = env.reset()
```

In practice, the tutorial already uses the latter form, so no code change is needed for Gym 0.21. (If porting from a newer CityLearn/Gymnasium, ensure you remove the extra info)

- **env.step()** – In Gym 0.21 this returns (obs, reward, done, info). The tutorial loops use exactly four outputs, e.g.:

```python
observations, _, _, _ = env.step(actions)
```

which works with Gym 0.21. (Gymnasium’s step returns five values (obs, reward, terminated, truncated, info), but those extra signals aren’t used in the original tutorial.) Thus the tutorial’s step calls match Gym 0.21; just ensure you do not unpack five values.

- **Episode termination**: The tutorial loops use while not env.done: or while not env.unwrapped.terminated:. In Gym 0.21, the done flag from step signals episode end. If using CityLearn’s new API, you may need to replace references to .done or .terminated accordingly. In CityLearn 2.x (Gym 0.21), env.done is set when the episode ends. No change is needed if using the old API, but if errors occur check that you use the correct termination flag (for Gym 0.21, use done; CityLearn’s Gymnasium version uses terminated/truncated).

---
# 🧩 SB3 API and Wrappers Adjustments

With Gym 0.21, we must use an SB3 release that supports the old Gym backend. Key points:

- **SB3 Version**: Use stable-baselines3 1.5.0 or later (but <2.0). SB3 1.5+ explicitly requires Gym≥0.21. SB3 2.x is Gymnasium-centric (and requires Python 3.8+), so for Gym 0.21 we stick to SB3 1.x. I recommend SB3 1.8.0.

- **Model instantiation**: The tutorial does e.g.:

```python
sac_model = SAC(policy='MlpPolicy', env=sac_env, seed=RANDOM_SEED, **sac_kwargs)
```

This syntax is valid in SB3 1.x. (In SB3 2.x the import path changed slightly, but we are not using 2.x.) No changes to SAC, PPO, etc. instantiation are needed for SB3 1.x.

- **Wrapper usage**: The tutorial wraps CityLearn with NormalizedObservationWrapper and StableBaselines3Wrapper for SB3, and with TabularQLearningWrapper for Q-learning. These wrappers are compatible with Gym 0.21, since they simply transform observations/actions into vectors. (Note: CityLearn’s current wrappers import from Gymnasium; if running under Gym 0.21 you may need to install the gymnasium compatibility shim or use an older CityLearn release. Alternatively, simply ensure your code uses the older wrappers or wraps via Gym spaces.)

- **Training loop**: The tutorial uses model.learn(total_timesteps=...) with callbacks. SB3 1.x’s callback API (BaseCallback) works the same. Just be sure to pass env=... to the SB3 model (as in the tutorial) and not rely on Gymnasium’s return values.

---

# 📦 Version Alignment and Dependencies

To ensure compatibility, align package versions as follows:

- **Python 3.8.20** – Install Python 3.8 (specifically 3.8.20 if possible) to meet CityLearn’s requirements (≥3.7) and SB3’s (1.x supports 3.7+).

- **Gym 0.21.0** – Pin Gym to exactly version 0.21.0. This restores the legacy Gym API.

- **CityLearn** – Use a CityLearn release that supports Gym 0.21. CityLearn 2.2.0 (Nov 2024) is appropriate (it requires gym>=0.21.0). Later versions (2.3.x) assume Gymnasium.

- **Stable-Baselines3** – Install SB3 1.8.0 (or any 1.x ≥1.5). This version supports Gym 0.21 as its minimum dependency.

- **Other Dependencies**: Ensure all CityLearn prerequisites match (e.g. doe-xstock>=1.1.0, nrel-pysam>=7.0.0, torch, etc.) consistent with CityLearn 2.2.0. The CityLearn docs or setup.py can be referenced for exact versions.

---

# 🛠️ Observed Issues and Resolutions

## 🔁 reset()

reset() return mismatch: If running the tutorial without changes, a ValueError (“not enough values to unpack”) may occur because CityLearn’s new env.reset() (Gymnasium style) returns two values. Pinning Gym 0.21 (or using CityLearn 2.2.0) fixes this by making env.reset() return only one value. Alternatively, edit the tutorial to drop the extra return (e.g. use obs = env.reset() instead of obs, _ = env.reset()).

## 🧩 Wrapper imports

CityLearn’s wrappers.py imports from gymnasium. If you truly run under Gym 0.21, either install the gymnasium package as well (so import gymnasium succeeds), or use CityLearn 2.2.0 which uses only gym. In practice, installing Gym 0.21 and letting CityLearn import Gymnasium (if needed) works if gymnasium is present.

## 🎓 Model training differences

With the version alignment above, SB3 training and callbacks should run as in the original tutorial. Just be sure to reset the environment with single-output env.reset() each episode.

---

By aligning to Python 3.8.20, Gym 0.21.0, and SB3 1.8.0 (with CityLearn 2.2.0), all tutorial cells run with minimal code changes. In summary: use Gym’s old API calls (env.reset() → one output; env.step() → 4 outputs) and install SB3 1.x. This restores the original tutorial behavior

---


