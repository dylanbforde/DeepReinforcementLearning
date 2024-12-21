import torch
import torch.nn.functional as F
from ReplayMemoryClass import Transition

class Optimizer:
    """
    Encapsulates the optimization process of the Q-network using experiences sampled from the replay memory.
    
    Attributes:
        policy_net (torch.nn.Module): The current policy network.
        target_net (torch.nn.Module): The target network, used for stable Q-value estimation.
        optimizer (torch.optim.Optimizer): The optimizer used for training the policy network.
        memory (ReplayMemory): The replay memory storing experience tuples.
        device (torch.device): The device on which to perform tensor operations.
        BATCH_SIZE (int): The size of the batch sampled from the replay memory.
        GAMMA (float): The discount factor for future rewards.
        TAU (float): The interpolation parameter for soft updating the target network.
        losses (list): A list to store the loss values after each optimization step.
    """
    
    def __init__(self, policy_net, target_net, optimizer, replay_memory, device, batch_size, gamma, tau):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.memory = replay_memory
        self.device = device
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau
        self.losses = []

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_domain_shifts = torch.cat([ds for s, ds in zip(batch.next_state, batch.domain_shift) if s is not None])  # Next state domain shifts

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).float()  # Cast reward_batch to float
        domain_shift_batch = torch.cat(batch.domain_shift)  # Current state domain shifts

        state_action_values = self.policy_net(state_batch, domain_shift_batch)
        state_action_values = state_action_values.gather(1, action_batch.argmax(dim=1).unsqueeze(1)).squeeze(1)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device, dtype=torch.float32)  # Specify dtype as float32
        with torch.no_grad():
            next_state_actions = self.policy_net(non_final_next_states, non_final_domain_shifts)
            next_state_values[non_final_mask] = next_state_actions.max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values)
        self.losses.append(loss.item())  # Store the loss value

        self.optimizer.zero_grad()  # Zero the gradients before the backward pass
        loss.backward()  # Compute the backward pass
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)  # Gradient clipping
        self.optimizer.step()  # Take a step with the optimizer

        # Soft update the target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.TAU * policy_param.data + (1.0 - self.TAU) * target_param.data)

        return loss