# module imports
import numpy as np
from itertools import count
import torch
import torch.nn.functional as F
import optuna
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn
import logging

# custom imports
from ReplayMemoryClass import ReplayMemory
from DQNClass import DQN
from PlotFunction import plot_function
from InitEnvironment import config, initialize_environment
from DataLoggerClass import DataLogger
from DomainShiftPredictor import DomainShiftPredictor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        logging.info('Using CUDA')

set_seed(1)
# best model
global best_value
best_value = -float('inf')

env, policy_net, target_net, optimizer, action_selector, optimizer_instance = initialize_environment(config)


def objective(trial):
    global best_value

    # suggest values for tunable hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    eps_decay = trial.suggest_int('eps_decay', 100, 2000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.8, 0.9999)
    
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
    input_dim = env.observation_space.shape[0] + 1 # size of the input (state + domain shift value)
    hidden_dim = 128 # size of the hidden layers
    output_dim = 1 # Size of the output (1 if suitable, 0 otherwise)
    suitability_threshold = 0.4
    adjustment_factor = 0.9 # factor to readjust hyperparams

    # Instantiate the domain shift class
    domain_shift_module = DomainShiftPredictor(input_dim, hidden_dim, output_dim, lr, suitability_threshold, adjustment_factor, device)            

    # For plotting function
    fig, axs = plt.subplots(4, 1, figsize=(10, 7))  # Create them once here
    episode_durations = []
    losses = optimizer_instance.losses
    eps_thresholds = []
    episode_rewards = []

    # Logging function
    logger = DataLogger('bipedal_walker_gravity_change_DSP.csv')
    env.set_logger(logger)

    num_episodes = 40000
    try:
        for i_episode in range(num_episodes):
            state, info = env.reset()
            state = state[0][0]  # Extract the array from the nested tuple
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            episode_total_reward = 0
            policy_net.train()
            predicted_suitability = None  # Initialize outside the loop

            for t in count():
                domain_shift_metric = env.quantify_domain_shift()
                domain_shift_tensor = torch.tensor([domain_shift_metric], dtype=torch.float32, device=device)

                predicted_suitability = domain_shift_module.predict_suitability(state, domain_shift_tensor)
                action = torch.tensor(np.random.uniform(low=-1, high=1, size=(env.action_space.shape[0],)), dtype=torch.float32, device=device).unsqueeze(0)

                # Take the action and observe the new state and reward
                (observation, reward, terminated, truncated, info), domain_shift = env.step(action.squeeze(0).detach().cpu().numpy())
                state = np.array(observation, dtype=np.float32)
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                reward = torch.tensor([reward], device=device)
                # Determine true suitability based on the episode outcome
                true_suitability = torch.tensor([[1.0]], device=device) if not (terminated or truncated) else torch.tensor([[0.0]], device=device)

                # Update the domain shift model
                if predicted_suitability is not None:
                    loss, _ = domain_shift_module.update(state, domain_shift_tensor, true_suitability)

                if i_episode >= 200:
                        # Update the DSP model after the first 200 episodes
                    loss, _ = domain_shift_module.update(state, domain_shift_tensor, true_suitability)

                done = terminated or truncated
                episode_total_reward += reward.item() # accumulate reward

                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                done_tensor = torch.tensor([done], device=device, dtype=torch.bool)

                memory.push(state, action, next_state, reward, domain_shift_tensor, done_tensor)

                if not done:
                    state = next_state
                else:
                    state = None
                loss = optimizer_instance.optimize()

                if loss is not None:
                    # Log step data
                    logger.log_step(
                    episode=i_episode,
                    step=t,
                    original_gravity=env.original_gravity[1],
                    current_gravity=env.world.gravity[1],
                    action=action.squeeze(0).detach().cpu().numpy(),
                    reward=reward.item(),
                    domain_shift=domain_shift,
                    cumulative_reward=episode_total_reward,
                    epsilon=action_selector.get_epsilon_thresholds()[-1] if action_selector.get_epsilon_thresholds() else 0,
                    loss=loss.item() if loss is not None else 0,
                    predicted_suitability=predicted_suitability.item() if predicted_suitability is not None else 0,
                )

                if done:
                    episode_durations.append(t + 1)
                    break

            if predicted_suitability.item() < suitability_threshold:
                action_selector.reset_epsilon()

            episode_rewards.append(episode_total_reward)

            if len(episode_rewards) >= 100:
                average_reward = np.mean(episode_rewards[-100:])
                if average_reward > PERFORMANCE_THRESHOLD:
                    action_selector.update_epsilon()

            # Get the current epsilon threshold after update
            if action_selector.get_epsilon_thresholds():
                current_eps_threshold = action_selector.get_epsilon_thresholds()[-1]
                eps_thresholds.append(current_eps_threshold)  # Append the latest epsilon value
            else:
                current_eps_threshold = None

            # Plot the graphs wanted
            plot_function(fig, axs, episode_durations, losses, eps_thresholds, episode_rewards, optimization_mode=True)

            trial.report(episode_durations[-1], i_episode)

            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        if mean_reward > best_value:
            best_value = mean_reward
            torch.save(policy_net.state_dict(), 'bipedal_walker_gravity_DSP.pth')

        return mean_reward
    finally:
        logger.close()

# study organisation
storage_url = "sqlite:///optuna_study.db"
study_name = 'bipedal_walker_gravity_DSP_final'

# Create a new study or load an existing study
pruner = optuna.pruners.PercentilePruner(99)
study = optuna.create_study(study_name=study_name, storage=storage_url, direction='maximize', load_if_exists=True, pruner=pruner)


try:
    study.optimize(objective, n_trials=80)
except Exception as e:
    print(f"An error occurred during optimization: {e}")


# After optimization, use the best trial to set the state of policy_net
best_trial = study.best_trial
best_model_path = 'bipedal_walker_gravity_DSP.pth'
best_model_state = torch.load(best_model_path)

# Reinitialize the environment with the best trial's hyperparameters
config.update(best_trial.params)
env, policy_net, target_net, optimizer, action_selector, optimizer_instance = initialize_environment(config)

policy_net.load_state_dict(best_model_state)
torch.save(policy_net.state_dict(), 'bipedal_walker_gravity_DSP.pth')

# Load the study
study = optuna.load_study(study_name=study_name, storage=storage_url)
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")