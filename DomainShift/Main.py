import argparse
import logging
import os
from itertools import count

import numpy as np
import optuna
import torch
from matplotlib import pyplot as plt

from ReplayMemoryClass import ReplayMemory
from PlotFunction import plot_function
from InitEnvironment import get_environment_config, initialize_environment
from DataLoggerClass import DataLogger
from DomainShiftPredictor import DomainShiftPredictor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
active_config = get_environment_config('bipedal', 'dsp')
best_value = -float('inf')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        logging.info('Using CUDA')


def _state_tensor(observation):
    if isinstance(observation, tuple):
        observation = observation[0]
    return torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


def _reset_state(env):
    state, _ = env.reset()
    return _state_tensor(state)


def _action_for_env(action, action_mode):
    if action_mode == 'discrete':
        return int(action.item())
    return action.squeeze(0).detach().cpu().numpy()


def _action_for_log(action, action_mode):
    if action_mode == 'discrete':
        return int(action.item())
    return action.squeeze(0).detach().cpu().numpy()


def _true_suitability(environment, done, reward, step):
    if environment == 'cartpole':
        episode_progress = min(step / 500.0, 1.0)
        survival_factor = 0.8 if not done else 0.0
        return 0.2 * episode_progress + survival_factor
    return 1.0 if not done else 0.0


def _gravity_values(env, environment):
    if environment == 'cartpole':
        return env.original_gravity, env.current_gravity
    return env.original_gravity[1], env.world.gravity[1]


def objective(trial):
    global best_value

    cfg = active_config
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    eps_decay = trial.suggest_int('eps_decay', 100, 2000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.8 if cfg['environment'] == 'bipedal' else 0.9, 0.9999)

    cfg.update({
        "lr": lr,
        "eps_decay": eps_decay,
        "batch_size": batch_size,
        "gamma": gamma,
    })

    env, policy_net, target_net, optimizer, action_selector, optimizer_instance = initialize_environment(cfg)
    memory = ReplayMemory(cfg['replay_memory_size'])
    optimizer_instance.memory = memory

    domain_shift_module = None
    suitability_threshold = cfg.get('suitability_threshold', 0.4)
    if cfg['mode'] == 'dsp':
        domain_shift_module = DomainShiftPredictor(
            env.observation_space.shape[0] + 1,
            cfg.get('hidden_dim', 128),
            1,
            lr,
            suitability_threshold,
            cfg.get('adjustment_factor', 0.9),
            device,
        )

    fig, axs = plt.subplots(4, 1, figsize=(10, 7))
    episode_durations = []
    losses = optimizer_instance.losses
    eps_thresholds = []
    episode_rewards = []

    log_file = os.path.join(cfg['output_dir'], f"{cfg['environment']}_{cfg['mode']}_trial_{trial.number}.csv")
    logger = DataLogger(log_file)
    env.set_logger(logger)

    try:
        for i_episode in range(cfg['num_episodes']):
            state = _reset_state(env)
            episode_total_reward = 0.0
            policy_net.train()
            predicted_suitability = None

            for t in count():
                domain_shift_metric = env.quantify_domain_shift()
                domain_shift_tensor = torch.tensor([domain_shift_metric], dtype=torch.float32, device=device)

                if domain_shift_module is not None:
                    predicted_suitability = domain_shift_module.predict_suitability(state, domain_shift_tensor)

                action = action_selector.select_action(state, domain_shift_tensor)
                (observation, reward, terminated, truncated, info), domain_shift = env.step(
                    _action_for_env(action, cfg['action_mode'])
                )
                done = terminated or truncated
                next_state = _state_tensor(observation)
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
                done_tensor = torch.tensor([done], dtype=torch.bool, device=device)

                if domain_shift_module is not None:
                    true_suitability = torch.tensor(
                        [[_true_suitability(cfg['environment'], done, reward, t)]],
                        dtype=torch.float32,
                        device=device,
                    )
                    domain_shift_module.update(
                        state,
                        domain_shift_tensor,
                        true_suitability,
                        predicted_suitability=predicted_suitability,
                    )

                episode_total_reward += float(reward)
                memory.push(state, action, next_state, reward_tensor, domain_shift_tensor, done_tensor)
                state = next_state if not done else None
                loss = optimizer_instance.optimize()

                if loss is not None:
                    original_gravity, current_gravity = _gravity_values(env, cfg['environment'])
                    logger.log_step(
                        episode=i_episode,
                        step=t,
                        original_gravity=original_gravity,
                        current_gravity=current_gravity,
                        action=_action_for_log(action, cfg['action_mode']),
                        reward=float(reward),
                        domain_shift=domain_shift,
                        cumulative_reward=episode_total_reward,
                        epsilon=action_selector.get_epsilon_thresholds()[-1] if action_selector.get_epsilon_thresholds() else 0,
                        loss=loss.item(),
                        predicted_suitability=predicted_suitability.item() if predicted_suitability is not None else 0.0,
                    )

                if done:
                    episode_durations.append(t + 1)
                    break

            if predicted_suitability is not None and predicted_suitability.item() < suitability_threshold:
                action_selector.reset_epsilon()

            episode_rewards.append(episode_total_reward)
            if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) > cfg['performance_threshold']:
                action_selector.update_epsilon()

            if action_selector.get_epsilon_thresholds():
                eps_thresholds.append(action_selector.get_epsilon_thresholds()[-1])

            plot_function(fig, axs, episode_durations, losses, eps_thresholds, episode_rewards, optimization_mode=True)

            if episode_durations:
                report_value = (
                    np.mean(episode_rewards[-100:])
                    if cfg['environment'] == 'cartpole'
                    else episode_durations[-1]
                )
                trial.report(report_value, i_episode)

            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        if mean_reward > best_value:
            best_value = mean_reward
            torch.save(policy_net.state_dict(), cfg['model_file'])

        if not cfg.get('cloud_mode', False):
            plt.savefig(os.path.join(cfg['output_dir'], f"{cfg['environment']}_{cfg['mode']}_trial_{trial.number}.png"))
        return mean_reward
    finally:
        logger.close()
        plt.close(fig)
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description='Train domain-shift RL experiments')
    parser.add_argument('--env', choices=['bipedal', 'cartpole'], default='bipedal')
    parser.add_argument('--mode', choices=['dsp', 'control'], default='dsp')
    parser.add_argument('--cloud', action='store_true', help='Run without interactive rendering')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--output-dir', default='results', help='Directory for CSV logs, models, studies, and plots')
    parser.add_argument('--num-episodes', '--episodes', dest='num_episodes', type=int)
    parser.add_argument('--n-trials', '--trials', dest='n_trials', type=int)
    return parser


def main(argv=None):
    global active_config, best_value

    args = build_parser().parse_args(argv)
    best_value = -float('inf')
    set_seed(args.seed)

    if args.cloud:
        os.environ['CLOUD_MODE'] = 'true'

    active_config = get_environment_config(args.env, args.mode)
    active_config['cloud_mode'] = args.cloud or active_config.get('cloud_mode', False)
    active_config['output_dir'] = args.output_dir
    if args.num_episodes is not None:
        active_config['num_episodes'] = args.num_episodes
    if args.n_trials is not None:
        active_config['n_trials'] = args.n_trials

    os.makedirs(active_config['output_dir'], exist_ok=True)
    active_config['model_file'] = os.path.join(active_config['output_dir'], f"{args.env}_{args.mode}.pth")
    active_config['study_db'] = os.path.join(active_config['output_dir'], f"{args.env}_{args.mode}.db")

    storage_url = f"sqlite:///{active_config['study_db']}"
    pruner = optuna.pruners.PercentilePruner(99)
    study = optuna.create_study(
        study_name=active_config['study_name'],
        storage=storage_url,
        direction='maximize',
        load_if_exists=True,
        pruner=pruner,
    )

    try:
        study.optimize(objective, n_trials=active_config['n_trials'])
    except Exception as e:
        logging.exception("An error occurred during optimization: %s", e)

    try:
        trial = study.best_trial
    except ValueError:
        logging.warning("No completed trials were available.")
        return None

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    return trial


if __name__ == "__main__":
    main()
