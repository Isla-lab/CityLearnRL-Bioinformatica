import numpy as np
import matplotlib.pyplot as plt
import glob

files = glob.glob('results/ppo/rewards_seed_*.npy')
rewards_all = [np.load(f) for f in files]
min_len = min(len(r) for r in rewards_all)
rewards_all = [r[:min_len] for r in rewards_all]

rewards_array = np.vstack(rewards_all)
mean_rewards = np.mean(rewards_array, axis=0)
std_rewards = np.std(rewards_array, axis=0)

episodes = np.arange(min_len)

plt.plot(episodes, mean_rewards, label='Media')
plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, label='Deviazione standard')
plt.title('Reward medio e varianza su PPO con seed multipli')
plt.xlabel('Episodio')
plt.ylabel('Reward Totale')
plt.legend()
plt.grid()
plt.show()
