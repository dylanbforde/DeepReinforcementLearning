import random
import math
import torch
import numpy as np

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
    
    def __init__(self, policy_net, action_dim, device, EPS_START, EPS_END, EPS_DECAY):
        self.policy_net = policy_net
        self.action_dim = action_dim
        self.device = device
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
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
                return self.policy_net(state, domain_shift)
            else:
                return torch.tensor(np.random.uniform(low=-1, high=1, size=(self.action_dim,)), dtype=torch.float32, device=self.device).unsqueeze(0)

    def get_epsilon_thresholds(self):
        return self.eps_thresholds
    
    def update_epsilon(self):
        self.EPS_START = max(self.EPS_START * (1 - self.EPS_DECAY), self.EPS_END)
    
    def reset_epsilon(self):
        # self.EPS_START = self.eps_thresholds[0] # original
        self.EPS_START = max(self.EPS_START * (1 - self.EPS_DECAY), self.EPS_END, 0.8)
        self.steps_done = 0