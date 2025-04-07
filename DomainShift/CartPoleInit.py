import torch
import torch.optim as optim
import os
from CustomCartPoleEnv import CustomCartPoleEnv
from ReplayMemoryClass import ReplayMemory
from CartPoleActionSelector import CartPoleActionSelector
from CartPoleOptimizer import CartPoleOptimizer
from CartPoleDQN import CartPoleDQN

# CartPole-specific configuration
config = {
    "lr": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_decay": 1000,
    "batch_size": 128,
    "replay_memory_size": 10000,
    "performance_threshold": 195,  # CartPole is solved when average reward is ≥ 195 over 100 episodes
    "n_trials": 10,
    "cloud_mode": os.environ.get('CLOUD_MODE', 'false').lower() == 'true',
}

# Set up the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def initialize_environment(config):
    """
    Initializes the CartPole environment and all related components.
    
    Args:
        config (dict): Configuration parameters.
        
    Returns:
        tuple: Environment, policy network, target network, optimizer, action selector, and optimizer instance.
    """
    # Use 'rgb_array' render mode when in cloud mode, or 'human' otherwise
    render_mode = None if config.get('cloud_mode', False) else 'human'
    print(f"Initializing CartPole environment with render_mode: {render_mode}")
    
    env = CustomCartPoleEnv(render_mode=render_mode)
    memory = ReplayMemory(config['replay_memory_size'])
    
    # CartPole has 4 state dimensions and 2 possible actions (left/right)
    state_dim = 4  # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    action_dim = 2  # [left, right]
    
    # Initialize networks for CartPole
    domain_shift_input_dim = 1
    policy_net = CartPoleDQN(state_dim, action_dim, domain_shift_input_dim).to(device)
    target_net = CartPoleDQN(state_dim, action_dim, domain_shift_input_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # Set up optimizer
    optimizer = optim.AdamW(policy_net.parameters(), lr=config['lr'], amsgrad=True)
    
    # Initialize CartPole-specific action selector
    action_selector = CartPoleActionSelector(
        policy_net,
        action_dim,
        device,
        config['eps_start'],
        config['eps_end'],
        config['eps_decay']
    )
    
    # Initialize optimizer instance
    optimizer_instance = CartPoleOptimizer(
        policy_net,
        target_net,
        optimizer,
        memory,
        device,
        config['batch_size'],
        config['gamma'],
        config['tau']
    )
    
    return env, policy_net, target_net, optimizer, action_selector, optimizer_instance