import csv
import matplotlib.pyplot as plt
import numpy as np

def calculate_mean_rewards(csv_file):
    episode_rewards = {}
    domain_shifts = {}

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode = int(row['episode'])
            domain_shift = float(row['domain_shift'])
            cumulative_reward = float(row['cumulative_reward'])

            if episode not in episode_rewards:
                episode_rewards[episode] = []
            if episode not in domain_shifts:
                domain_shifts[episode] = []

            episode_rewards[episode].append(cumulative_reward)
            domain_shifts[episode].append(domain_shift)

    mean_rewards = {episode: np.mean(rewards) for episode, rewards in episode_rewards.items()}
    mean_domain_shifts = {episode: np.mean(shifts) for episode, shifts in domain_shifts.items()}

    return mean_rewards, mean_domain_shifts

def plot_mean_rewards(mean_rewards, mean_domain_shifts):
    episodes = list(mean_rewards.keys())
    mean_cumulative_rewards = list(mean_rewards.values())
    mean_domain_shift_sizes = list(mean_domain_shifts.values())

    plt.figure(figsize=(10, 5))
    plt.scatter(episodes, mean_cumulative_rewards, c=mean_domain_shift_sizes, cmap='viridis')
    plt.colorbar(label='Mean Domain Shift Size')
    plt.title('Mean Cumulative Reward per Episode (Colored by Domain Shift Size)')
    plt.xlabel('Episode')
    plt.ylabel('Mean Cumulative Reward')
    plt.grid(True)
    plt.show()

def plot_best_run(csv_file):
    best_cumulative_reward = -float('inf')
    best_run = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cumulative_reward = float(row['cumulative_reward'])
            if cumulative_reward > best_cumulative_reward:
                best_cumulative_reward = cumulative_reward
                best_run = [(int(row['episode']), cumulative_reward)]

    episodes, cumulative_rewards = zip(*best_run)

    # Plot the cumulative rewards for the best run
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, cumulative_rewards, label='Best Run', color='blue')
    plt.title('Best Run: Cumulative Reward over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_reward_distribution(csv_file):
    cumulative_rewards = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cumulative_rewards.append(float(row['cumulative_reward']))

    # Plot the reward distribution
    plt.figure(figsize=(10, 5))
    plt.hist(cumulative_rewards, bins=20, density=True, alpha=0.6, color='green')
    plt.title('Reward Distribution')
    plt.xlabel('Cumulative Reward')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

def plot_domain_shift_impact(csv_file):
    domain_shifts = []
    cumulative_rewards = []
    episodes = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain_shifts.append(float(row['domain_shift']))
            cumulative_rewards.append(float(row['cumulative_reward']))
            episodes.append(float(row['episode']))

    # Plot the domain shift impact on cumulative reward
    plt.figure(figsize=(10, 5))
    plt.scatter(domain_shifts, cumulative_rewards, alpha=0.6, s = 5)
    plt.title('Domain Shift Impact on Cumulative Reward')
    plt.xlabel('Domain Shift (Difference in Pole Length)')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.show()


csv_file = 'friction_and _mass_random_change_training_data_with_predictor.csv'
mean_rewards, mean_domain_shifts = calculate_mean_rewards(csv_file)
plot_mean_rewards(mean_rewards, mean_domain_shifts)

#plot_best_run(csv_file)
#plot_reward_distribution(csv_file)
#plot_domain_shift_impact(csv_file)
