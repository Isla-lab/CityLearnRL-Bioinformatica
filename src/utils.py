import numpy as np
import matplotlib.pyplot as plt


from citylearn.citylearn import CityLearnEnv


def make_env():
    schema_path = "data/citylearn_challenge_2021/schema.json"
    env = CityLearnEnv(schema=schema_path, episode_time_steps=96)
    return env





def save_rewards(rewards, path):
    np.save(path, np.array(rewards))

def load_all_rewards(folder, num_seeds):
    rewards_list = []
    for i in range(num_seeds):
        rewards = np.load(f"{folder}/rewards_seed_{i}.npy")
        rewards_list.append(rewards)
    return rewards_list

def plot_reward_curves(data_list, label):
    data_array = np.array(data_list)
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    x = np.arange(len(mean))

    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward vs Episode")
    plt.legend()
    plt.grid()
    plt.show()
