import torch
import torch.optim as optim
import os
from copy import deepcopy
from ReplayMemoryClass import ReplayMemory
from ActionSelection import ActionSelector
from OptimizeModel import Optimizer
from DQNClass import DQN

# Define the configuration dictionary with the necessary hyperparameters.
config = {
    "lr": 1e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_decay": 1000,
    "batch_size": 128,
    "replay_memory_size": 10000,
    "performance_threshold": 195,
    "clip_value": 100,
    "hidden_dim": 128,
    "environment": "bipedal",
    "mode": "dsp",
    "log_file": "bipedal_walker_gravity_change_DSP.csv",
    "model_file": "bipedal_walker_gravity_DSP.pth",
    "study_db": "optuna_study.db",
    "num_episodes": 40000,
    "n_trials": 80,
    "cloud_mode": os.environ.get('CLOUD_MODE', 'false').lower() == 'true',
    # ... other hyperparameters ...
}

ENVIRONMENT_DEFAULTS = {
    "bipedal": {
        "lr": 1e-4,
        "gamma": 0.99,
        "performance_threshold": 195,
        "num_episodes": 40000,
        "n_trials": 80,
        "action_mode": "continuous",
        "output_activation": "tanh",
        "study_name": "bipedal_walker_gravity",
    },
    "cartpole": {
        "lr": 1e-3,
        "gamma": 0.99,
        "performance_threshold": 195,
        "num_episodes": 500,
        "n_trials": 10,
        "action_mode": "discrete",
        "output_activation": "identity",
        "study_name": "cartpole_domain_shift",
    },
}


def get_environment_config(environment, mode):
    env_defaults = ENVIRONMENT_DEFAULTS[environment]
    cfg = deepcopy(config)
    cfg.update(env_defaults)
    cfg.update({
        "environment": environment,
        "mode": mode,
        "log_file": f"{environment}_{mode}.csv",
        "model_file": f"{environment}_{mode}.pth",
        "study_db": f"{environment}_{mode}.db",
        "study_name": f"{env_defaults['study_name']}_{mode}",
    })
    return cfg

# Set up the device for training (either CPU or CUDA if available).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a function to initialize the environment and all related components.
def initialize_environment(config):
    """
    Initializes the environment along with the policy and target networks, action selector, and optimizer instance.
    
    Args:
        config (dict): A dictionary containing hyperparameters and other configuration settings.
    
    Returns:
        tuple: A tuple containing the environment, policy network, target network, optimizer, action selector,
               and optimizer instance.
    """
    
    render_mode = None if config.get('cloud_mode', False) else 'human'
    environment = config.get('environment', 'bipedal')
    if environment == 'bipedal':
        from CustomBipedalWalkerEnv import CustomBipedalWalkerEnv
        env = CustomBipedalWalkerEnv(render_mode=render_mode)
        action_dim = env.action_space.shape[0]
    elif environment == 'cartpole':
        from CustomCartPoleEnv import CustomCartPoleEnv
        env = CustomCartPoleEnv(render_mode=render_mode)
        action_dim = env.action_space.n
    else:
        raise ValueError(f"Unsupported environment: {environment}")

    memory = ReplayMemory(config['replay_memory_size'])  # Access from config
    state_dim = env.observation_space.shape[0]

    # Initialize policy and target networks with the proper device
    domain_shift_input_dim = 1
    hidden_dim = config.get('hidden_dim', 128)
    output_activation = config.get('output_activation', 'tanh')
    policy_net = DQN(state_dim, action_dim, domain_shift_input_dim, hidden_dim, output_activation).to(device)
    target_net = DQN(state_dim, action_dim, domain_shift_input_dim, hidden_dim, output_activation).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Set up the optimizer using the learning rate from config
    optimizer = optim.AdamW(policy_net.parameters(), lr=config['lr'], amsgrad=True)

    # Initialize the action selector with parameters from config
    action_selector = ActionSelector(
        policy_net,
        action_dim,
        device,
        config['eps_start'],
        config['eps_end'],
        config['eps_decay'],
        config.get('action_mode', 'continuous')
    )

    # Initialize the optimizer instance with parameters from config
    optimizer_instance = Optimizer(
        policy_net,
        target_net,
        optimizer,
        memory,
        device,
        config['batch_size'],
        config['gamma'],
        config['tau'],
        config.get('clip_value', 100),
        config.get('action_mode', 'continuous')
    )

    # Return all the initialized components
    return env, policy_net, target_net, optimizer, action_selector, optimizer_instance
