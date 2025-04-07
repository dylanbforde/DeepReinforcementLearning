import random
import math
import torch
import numpy as np

class CartPoleActionSelector:
    """
    Action selector for CartPole environment with discrete actions.
    Uses epsilon-greedy policy with decaying exploration rate.
    
    Attributes:
        policy_net: The Q-network for selecting actions.
        num_actions: Number of discrete actions (2 for CartPole).
        device: Device to run computations on (CPU/CUDA).
        EPS_START: Initial exploration rate.
        EPS_END: Final exploration rate.
        EPS_DECAY: Rate of exploration decay.
    """
    
    def __init__(self, policy_net, num_actions, device, EPS_START, EPS_END, EPS_DECAY):
        self.policy_net = policy_net
        self.num_actions = num_actions
        self.device = device
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0
        self.eps_thresholds = []
    
    def select_action(self, state, domain_shift):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation.
            domain_shift: Domain shift metric.
            
        Returns:
            torch.Tensor: Selected action (0 or 1 for CartPole).
        """
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        self.eps_thresholds.append(eps_threshold)
        
        # Set the model to evaluation mode
        self.policy_net.eval()
        
        with torch.no_grad():
            if sample > eps_threshold:
                # Exploit: select best action from Q-values
                q_values = self.policy_net(state, domain_shift)
                return torch.argmax(q_values, dim=1).view(1, 1)
            else:
                # Explore: select random action
                return torch.tensor([[random.randrange(self.num_actions)]], 
                                    device=self.device, dtype=torch.long)

    def get_epsilon_thresholds(self):
        """Get the history of epsilon values."""
        return self.eps_thresholds
    
    def update_epsilon(self):
        """Gradually reduce epsilon when performance threshold is met."""
        self.EPS_START = max(self.EPS_START * 0.9, self.EPS_END)
    
    def reset_epsilon(self, factor=0.8):
        """
        Reset exploration rate (epsilon) to encourage exploration in new domains.
        
        Args:
            factor (float): Factor to determine how aggressive the reset should be.
                           Higher values (closer to 1) mean more exploration.
        """
        # Calculate new epsilon based on the factor and ensure it's not below minimum
        self.EPS_START = max(self.EPS_START, self.EPS_END, factor)
        # Reset step counter to restart decay
        self.steps_done = 0
        # Log the reset for tracking
        self.eps_thresholds.append(self.EPS_START)