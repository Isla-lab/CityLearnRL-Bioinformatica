import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pathlib import Path

# Parametri
reward_file = Path("results/ppo/rewards_seed_0.npy")
save_path = Path("results/ppo/rewards_plot.png")
window_size = 10  # Puoi modificarlo per cambiare il grado di smoothing

# Caricamento dei reward
rewards = np.load(reward_file)

# Calcolo della media mobile
smoothed_rewards = uniform_filter1d(rewards, size=window_size)

# Creazione del grafico
plt.figure(figsize=(12, 6))
plt.plot(rewards, alpha=0.3, label="Reward grezzo", color="blue")
plt.plot(smoothed_rewards, label=f"Media mobile (finestra={window_size})", color="red")
plt.title("Andamento del Reward PPO per Episodio (con smoothing)")
plt.xlabel("Episodio")
plt.ylabel("Reward Totale")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Salvataggio e visualizzazione
plt.savefig(save_path)
plt.show()
