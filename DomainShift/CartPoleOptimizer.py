import torch
import torch.nn.functional as F
from ReplayMemoryClass import Transition

class CartPoleOptimizer:
    """
    Optimizer for CartPole DQN with discrete actions.
    Implements experience replay and target network for stable learning.
    
    Attributes:
        policy_net: The network being trained.
        target_net: Slowly-updated target network for stable Q-value estimates.
        optimizer: Optimizer for updating policy network.
        memory: Replay buffer for storing experience tuples.
        device: Device to run computations (CPU/CUDA).
        batch_size: Number of transitions to sample for each update.
        gamma: Discount factor for future rewards.
        tau: Rate for soft target network updates.
    """
    
    def __init__(self, policy_net, target_net, optimizer, replay_memory, device, batch_size, gamma, tau):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.memory = replay_memory
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.losses = []

    def optimize(self):
        """
        Perform one step of optimization using experience replay.
        
        Returns:
            torch.Tensor: The loss value for this optimization step, or None if the memory is insufficient.
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample a batch of transitions from memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Create mask for non-final states (where next_state is not None)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                     device=self.device, dtype=torch.bool)
        
        # Collect non-final next states and their domain shifts
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_domain_shifts = torch.cat([ds for s, ds in zip(batch.next_state, batch.domain_shift) 
                                          if s is not None])

        # Prepare batch data
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)  # This will be indices for CartPole (0 or 1)
        reward_batch = torch.cat(batch.reward)
        domain_shift_batch = torch.cat(batch.domain_shift)

        # Get current Q-values for the states and actions in the batch
        state_action_values = self.policy_net(state_batch, domain_shift_batch).gather(1, action_batch)

        # Compute next state values using the target network
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # Using double Q-learning approach: select actions using policy net, evaluate using target net
            next_action_indices = self.policy_net(non_final_next_states, non_final_domain_shifts).max(1)[1].detach()
            next_state_values[non_final_mask] = self.target_net(non_final_next_states, non_final_domain_shifts).gather(1, next_action_indices.unsqueeze(1)).squeeze(1)

        # Compute the expected Q values using the Bellman equation
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute the loss (Huber loss for stability)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.losses.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        return loss