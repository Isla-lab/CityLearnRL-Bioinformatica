from ppo_train import train_and_save
import os

SEEDS = [0, 1, 2, 3, 4]

for seed in SEEDS:
    print(f"🔥 Avvio training PPO con seed={seed}")
    train_and_save(seed=seed, timesteps=100_000)
