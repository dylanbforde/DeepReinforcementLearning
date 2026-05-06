import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.optim as optim

from OptimizeModel import Optimizer
from ReplayMemoryClass import Transition


class MockNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x, domain_shift):
        return self.layer(x)


class TestOptimizeModel(unittest.TestCase):
    def test_soft_update_uses_linear_interpolation(self):
        input_dim = 4
        output_dim = 2
        batch_size = 2
        tau = 0.5

        policy_net = MockNet(input_dim, output_dim)
        target_net = MockNet(input_dim, output_dim)
        nn.init.constant_(policy_net.layer.weight, 1.0)
        nn.init.constant_(policy_net.layer.bias, 1.0)
        nn.init.constant_(target_net.layer.weight, 0.0)
        nn.init.constant_(target_net.layer.bias, 0.0)

        optimizer = optim.SGD(policy_net.parameters(), lr=0.1)

        memory = MagicMock()
        memory.__len__.return_value = batch_size
        state = torch.randn(1, input_dim)
        action = torch.zeros(1, output_dim)
        action[0, 0] = 1
        next_state = torch.randn(1, input_dim)
        reward = torch.tensor([1.0])
        domain_shift = torch.randn(1, 1)
        done = torch.tensor([False])
        transition = Transition(state, action, next_state, reward, domain_shift, done)
        memory.sample.return_value = [transition] * batch_size

        optimizer_instance = Optimizer(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_memory=memory,
            device=torch.device('cpu'),
            batch_size=batch_size,
            gamma=0.99,
            tau=tau,
        )

        target_weight_before = target_net.layer.weight.clone()
        optimizer_instance.optimize()

        expected_weight = tau * policy_net.layer.weight + (1.0 - tau) * target_weight_before
        self.assertTrue(torch.allclose(target_net.layer.weight, expected_weight, atol=1e-6))

    def test_discrete_action_optimizer_accepts_index_actions(self):
        input_dim = 4
        output_dim = 2
        batch_size = 2

        policy_net = MockNet(input_dim, output_dim)
        target_net = MockNet(input_dim, output_dim)
        optimizer = optim.SGD(policy_net.parameters(), lr=0.1)

        memory = MagicMock()
        memory.__len__.return_value = batch_size
        state = torch.randn(1, input_dim)
        action = torch.tensor([[1]], dtype=torch.long)
        next_state = torch.randn(1, input_dim)
        reward = torch.tensor([1.0])
        domain_shift = torch.randn(1, 1)
        done = torch.tensor([False])
        transition = Transition(state, action, next_state, reward, domain_shift, done)
        memory.sample.return_value = [transition] * batch_size

        optimizer_instance = Optimizer(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_memory=memory,
            device=torch.device('cpu'),
            batch_size=batch_size,
            gamma=0.99,
            tau=0.5,
            action_mode='discrete',
        )

        loss = optimizer_instance.optimize()

        self.assertIsNotNone(loss)


if __name__ == '__main__':
    unittest.main()
