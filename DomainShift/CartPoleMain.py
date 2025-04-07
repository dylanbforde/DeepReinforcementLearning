#!/usr/bin/env python3
# CartPole implementation with domain shift prediction

import numpy as np
import os
import time
from itertools import count
import torch
import torch.nn.functional as F
import optuna
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn
import argparse

# custom imports
from ReplayMemoryClass import ReplayMemory
from CartPoleDQN import CartPoleDQN
from PlotFunction import plot_function
from CartPoleInit import config, initialize_environment
from DataLoggerClass import DataLogger
from DomainShiftPredictor import DomainShiftPredictor

# Parse command line arguments
parser = argparse.ArgumentParser(description='Deep Reinforcement Learning with Domain Shift (CartPole)')
parser.add_argument('--cloud', action='store_true', help='Run in cloud mode without visualization')
parser.add_argument('--trials', type=int, default=10, help='Number of trials to run')
parser.add_argument('--episodes', type=int, default=500, help='Max number of episodes per trial')
args = parser.parse_args()

# Update config based on command line arguments
if args.cloud:
    os.environ['CLOUD_MODE'] = 'true'
    config['cloud_mode'] = True
config['n_trials'] = args.trials
config['num_episodes'] = args.episodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print('Using CUDA with deterministic mode')

set_seed(42)
# best model
global best_value
best_value = -float('inf')

# Create output directory for results
output_dir = os.path.join(os.getcwd(), 'results_cartpole')
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved in: {output_dir}")

# Initialize environment
env, policy_net, target_net, optimizer, action_selector, optimizer_instance = initialize_environment(config)

def objective(trial):
    global best_value

    # suggest values for tunable hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    eps_decay = trial.suggest_int('eps_decay', 100, 2000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    
    # Update the config with the suggested values
    config.update({
        "lr": lr,
        "eps_decay": eps_decay,
        "batch_size": batch_size,
        "gamma": gamma,
    })

    # reinitialize the environment with the updated values
    env, policy_net, target_net, optimizer, action_selector, optimizer_instance = initialize_environment(config)

    # Use the hyperparameters from the config dictionary
    PERFORMANCE_THRESHOLD = config['performance_threshold']

    # initialise environment and components
    memory = ReplayMemory(config['replay_memory_size'])
    optimizer_instance.memory = memory

    # domain shift predictor necessary values
    input_dim = 4 + 1  # CartPole state dim (4) + domain shift value (1)
    hidden_dim = 128
    output_dim = 1
    suitability_threshold = 0.4
    adjustment_factor = 0.9

    # Instantiate the domain shift class
    domain_shift_module = DomainShiftPredictor(input_dim, hidden_dim, output_dim, lr, suitability_threshold, adjustment_factor, device)

    # For plotting function
    fig, axs = plt.subplots(4, 1, figsize=(10, 7))
    episode_durations = []
    losses = optimizer_instance.losses
    eps_thresholds = []
    episode_rewards = []

    # Logging function
    log_filename = os.path.join(output_dir, f'cartpole_domain_shift_predictor_trial_{trial.number}.csv')
    logger = DataLogger(log_filename)
    env.set_logger(logger)

    num_episodes = config.get('num_episodes', 500)
    print(f"Starting trial {trial.number}, running for {num_episodes} episodes")
    start_time = time.time()
    
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        episode_total_reward = 0
        policy_net.train()

        for t in count():
            domain_shift_metric = env.quantify_domain_shift()
            domain_shift_tensor = torch.tensor([domain_shift_metric], dtype=torch.float32, device=device)
            
            # Get suitability prediction from domain shift predictor
            predicted_suitability = domain_shift_module.predict_suitability(state, domain_shift_tensor)
            
            # Use the action selector to get an action
            action = action_selector.select_action(state, domain_shift_tensor)

            # Take the action and observe the new state and reward
            observation, reward, terminated, truncated, info = env.step(action.item())[0]
            domain_shift = env.quantify_domain_shift()
            
            # Convert observation and reward to tensors
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            reward = torch.tensor([reward], device=device)
            
            # Determine true suitability based on the episode outcome
            true_suitability = torch.tensor([[1.0]], device=device) if not (terminated or truncated) else torch.tensor([[0.0]], device=device)

            # Update the domain shift predictor
            if i_episode >= 100:  # Start updating after collecting some experience
                loss, _ = domain_shift_module.update(state, domain_shift_tensor, true_suitability)
                
            done = terminated or truncated
            episode_total_reward += reward.item()

            if not done:
                next_state = state
            else:
                next_state = None
            
            # Push experience to replay memory
            memory.push(
                torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0),  # state
                action,  # action
                next_state,  # next_state
                reward,  # reward
                domain_shift_tensor  # domain_shift
            )
            
            # Perform one step of optimization
            loss = optimizer_instance.optimize()

            if loss is not None:
                # Log step data
                logger.log_step(
                    episode=i_episode,
                    step=t,
                    original_gravity=env.original_gravity,
                    current_gravity=env.unwrapped.gravity,
                    action=action.item(),
                    reward=reward.item(),
                    domain_shift=domain_shift,
                    cumulative_reward=episode_total_reward,
                    epsilon=action_selector.get_epsilon_thresholds()[-1] if action_selector.get_epsilon_thresholds() else 0,
                    loss=loss.item() if loss is not None else 0,
                    predicted_suitability=predicted_suitability.item() if predicted_suitability is not None else 0.0,
                )

            if done:
                episode_durations.append(t + 1)
                break
        
        # Reset exploration if environment suitability is low
        if predicted_suitability is not None and predicted_suitability.item() < suitability_threshold:
            action_selector.reset_epsilon()
        
        episode_rewards.append(episode_total_reward)

        # Update epsilon if performance threshold is reached
        if len(episode_rewards) >= 100:
            average_reward = np.mean(episode_rewards[-100:])
            if average_reward > PERFORMANCE_THRESHOLD:
                action_selector.update_epsilon()
        
        # Get current epsilon threshold
        if action_selector.get_epsilon_thresholds():
            current_eps_threshold = action_selector.get_epsilon_thresholds()[-1]
            eps_thresholds.append(current_eps_threshold)
        else:
            current_eps_threshold = None

        # Plot training metrics
        plot_function(fig, axs, episode_durations, losses, eps_thresholds, episode_rewards, optimization_mode=True)
        
        # Report to Optuna for pruning
        if i_episode % 10 == 0:
            mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            trial.report(mean_reward, i_episode)
            
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Trial {trial.number} completed in {execution_time:.2f} seconds")
    
    # Compute final reward
    mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    
    # Save best model
    if mean_reward > best_value:
        best_value = mean_reward
        model_path = os.path.join(output_dir, 'cartpole_domain_shift_predictor.pth')
        torch.save(policy_net.state_dict(), model_path)
        print(f"New best model saved with mean reward: {mean_reward:.2f}")

    # Save training plots
    if not config.get('cloud_mode', False):
        plot_path = os.path.join(output_dir, f'plot_cartpole_trial_{trial.number}.png')
        plt.savefig(plot_path)
        plt.close(fig)

    return mean_reward

# Optuna study configuration
db_path = os.path.join(output_dir, 'optuna_cartpole_study.db')
storage_url = f"sqlite:///{db_path}"
study_name = 'cartpole_domain_shift_predictor'

# Create or load study
pruner = optuna.pruners.PercentilePruner(98)
study = optuna.create_study(
    study_name=study_name, 
    storage=storage_url, 
    direction='maximize',
    load_if_exists=True, 
    pruner=pruner
)

print(f"Starting optimization with {config['n_trials']} trials")
total_start_time = time.time()

try:
    study.optimize(objective, n_trials=config['n_trials'])
except Exception as e:
    print(f"An error occurred during optimization: {e}")

total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f"Total optimization time: {total_time:.2f} seconds")

# Get best trial and model
best_trial = study.best_trial
best_model_path = os.path.join(output_dir, 'cartpole_domain_shift_predictor.pth')

print("\n--- Best Trial Results ---")
print(f"Value: {best_trial.value}")
print("Parameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Save a summary file
summary_path = os.path.join(output_dir, 'cartpole_summary.txt')
with open(summary_path, 'w') as f:
    f.write(f"Total trials: {len(study.trials)}\n")
    f.write(f"Best trial: {best_trial.number}\n")
    f.write(f"Best value: {best_trial.value}\n")
    f.write("Parameters:\n")
    for key, value in best_trial.params.items():
        f.write(f"    {key}: {value}\n")
    f.write(f"\nTotal optimization time: {total_time:.2f} seconds\n")

print(f"Summary saved to {summary_path}")
print(f"Best model saved to {best_model_path}")

if __name__ == "__main__":
    print("CartPole training with domain shift predictor completed.")