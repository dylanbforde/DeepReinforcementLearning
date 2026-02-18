import torch
import torch.optim as optim
from CustomBipedalWalkerEnv import CustomBipedalWalkerEnv
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
    # ... other hyperparameters ...
}

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
    
    env = CustomBipedalWalkerEnv(render_mode='human')
    memory = ReplayMemory(config['replay_memory_size'])  # Access from config
    state_dim = env.observation_space.shape[0]
    action_dim = env.observation_space.shape[0]

    # Initialize policy and target networks with the proper device
    domain_shift_input_dim = 1
    policy_net = DQN(state_dim, action_dim, domain_shift_input_dim, config.get('hidden_dim', 128)).to(device)
    target_net = DQN(state_dim, action_dim, domain_shift_input_dim, config.get('hidden_dim', 128)).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Set up the optimizer using the learning rate from config
    optimizer = optim.AdamW(policy_net.parameters(), lr=config['lr'], amsgrad=True)

    # Initialize the action selector with parameters from config
    action_selector = ActionSelector(
        policy_net,
        env.action_space.shape[0],
        device,
        config['eps_start'],
        config['eps_end'],
        config['eps_decay']
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
        config.get('clip_value', 100)
    )

    # Return all the initialized components
    return env, policy_net, target_net, optimizer, action_selector, optimizer_instance
