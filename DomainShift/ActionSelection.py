import random
import math
import torch

class ActionSelector:
    """
    This class is responsible for selecting actions using an epsilon-greedy policy.
    It supports updating the exploration rate (epsilon) over time.
    
    Attributes:
        policy_net (torch.nn.Module): The neural network used to select actions.
        num_actions (int): The number of possible actions to choose from.
        device (torch.device): The device on which to perform tensor operations.
        EPS_START (float): The initial value of epsilon for the epsilon-greedy policy.
        EPS_END (float): The minimum value of epsilon after decay.
        EPS_DECAY (float): The rate at which epsilon decays.
        steps_done (int): The number of steps taken (used for epsilon decay).
        eps_thresholds (list): A list to store the value of epsilon after each step.
    """
    
    def __init__(self, policy_net, action_dim, device, EPS_START, EPS_END, EPS_DECAY, action_mode='continuous'):
        self.policy_net = policy_net
        self.action_dim = action_dim
        self.device = device
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.action_mode = action_mode
        self.steps_done = 0
        self.eps_thresholds = []
    
    def select_action(self, state, domain_shift):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        self.eps_thresholds.append(eps_threshold)
        
        # Set the model to evaluation mode
        self.policy_net.eval()
        
        with torch.no_grad():
            if sample > eps_threshold:
                q_values = self.policy_net(state, domain_shift)
                if self.action_mode == 'discrete':
                    return q_values.argmax(dim=1).view(1, 1)
                return q_values

            if self.action_mode == 'discrete':
                return torch.randint(self.action_dim, (1, 1), dtype=torch.long, device=self.device)
            return torch.empty((1, self.action_dim), dtype=torch.float32, device=self.device).uniform_(-1.0, 1.0)

    def get_epsilon_thresholds(self):
        return self.eps_thresholds
    
    def update_epsilon(self, factor=0.9):
        self.EPS_START = max(self.EPS_START * factor, self.EPS_END)
    
    def reset_epsilon(self, factor=0.8):
        self.EPS_START = max(self.EPS_START * factor, self.EPS_END)
        self.steps_done = 0
