
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import MagicMock

from OptimizeModel import Optimizer
from ReplayMemoryClass import Transition

class TestOptimizeModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.input_dim = 4
        self.output_dim = 2

        # Simple policy and target networks
        self.policy_net = nn.Linear(self.input_dim, self.output_dim)
        self.target_net = nn.Linear(self.input_dim, self.output_dim)

        # Initialize weights to known values
        nn.init.constant_(self.policy_net.weight, 1.0)
        nn.init.constant_(self.policy_net.bias, 1.0)
        nn.init.constant_(self.target_net.weight, 0.0)
        nn.init.constant_(self.target_net.bias, 0.0)

        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=0.1)

        # Mock replay memory
        self.memory = MagicMock()
        self.BATCH_SIZE = 2
        self.memory.__len__.return_value = self.BATCH_SIZE

        # Create dummy transitions
        state = torch.randn(1, self.input_dim)
        action = torch.zeros(1, self.output_dim)
        action[0, 0] = 1 # One-hot action
        next_state = torch.randn(1, self.input_dim)
        reward = torch.tensor([1.0])
        domain_shift = torch.randn(1, 1) # dummy domain shift
        done = torch.tensor([False])

        transition = Transition(state, action, next_state, reward, domain_shift, done)
        self.memory.sample.return_value = [transition] * self.BATCH_SIZE

        self.GAMMA = 0.99
        self.TAU = 0.5

        self.optimizer_instance = Optimizer(
            policy_net=self.policy_net,
            target_net=self.target_net,
            optimizer=self.optimizer,
            replay_memory=self.memory,
            device=self.device,
            batch_size=self.BATCH_SIZE,
            gamma=self.GAMMA,
            tau=self.TAU
        )

    def test_soft_update(self):
        # Initial check
        self.assertTrue(torch.allclose(self.target_net.weight, torch.zeros_like(self.target_net.weight)))

        # Run optimize step
        # We need to mock the forward pass slightly because the architecture in OptimizeModel expects specific inputs
        # But wait, OptimizeModel calls self.policy_net(state_batch, domain_shift_batch)
        # My simple linear model only takes one input.
        # So I need to mock the forward method of my policy_net or change the policy net to accept two inputs.

        class MockNet(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(MockNet, self).__init__()
                self.layer = nn.Linear(input_dim, output_dim)
            def forward(self, x, domain_shift):
                # Ignore domain shift for this test
                return self.layer(x)

        self.policy_net = MockNet(self.input_dim, self.output_dim)
        self.target_net = MockNet(self.input_dim, self.output_dim)

        nn.init.constant_(self.policy_net.layer.weight, 1.0)
        nn.init.constant_(self.policy_net.layer.bias, 1.0)
        nn.init.constant_(self.target_net.layer.weight, 0.0)
        nn.init.constant_(self.target_net.layer.bias, 0.0)

        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=0.1)

        self.optimizer_instance = Optimizer(
            policy_net=self.policy_net,
            target_net=self.target_net,
            optimizer=self.optimizer,
            replay_memory=self.memory,
            device=self.device,
            batch_size=self.BATCH_SIZE,
            gamma=self.GAMMA,
            tau=self.TAU
        )

        # Before optimization
        target_weight_before = self.target_net.layer.weight.clone()

        self.optimizer_instance.optimize()

        # Get policy weights AFTER optimization step
        policy_weight_after = self.policy_net.layer.weight

        # Check if soft update happened using the updated policy weights
        # target_new = tau * policy_new + (1-tau) * target_old

        expected_weight = self.TAU * policy_weight_after + (1.0 - self.TAU) * target_weight_before

        self.assertTrue(torch.allclose(self.target_net.layer.weight, expected_weight, atol=1e-6),
                        f"Expected {expected_weight}, got {self.target_net.layer.weight}")

if __name__ == "__main__":
    unittest.main()
