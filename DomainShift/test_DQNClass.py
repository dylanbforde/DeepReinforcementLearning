import unittest
import torch
import torch.nn as nn
import sys
import os

# Ensure we can import modules from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DQNClass import DQN

class TestDQN(unittest.TestCase):
    def test_default_initialization(self):
        state_dim = 10
        action_dim = 2
        domain_shift_input_dim = 1
        model = DQN(state_dim, action_dim, domain_shift_input_dim)

        # Check hidden layer sizes (assuming default is 128)
        self.assertEqual(model.layer1.out_features, 128)
        self.assertEqual(model.layer2.out_features, 128)
        self.assertEqual(model.layer3.in_features, 128)

    def test_custom_initialization(self):
        state_dim = 10
        action_dim = 2
        domain_shift_input_dim = 1
        hidden_dim = 64

        model = DQN(state_dim, action_dim, domain_shift_input_dim, hidden_dim=hidden_dim)
        # Check if sizes match hidden_dim
        self.assertEqual(model.layer1.out_features, hidden_dim)
        self.assertEqual(model.layer2.out_features, hidden_dim)
        self.assertEqual(model.layer3.in_features, hidden_dim)

    def test_forward_pass(self):
        state_dim = 10
        action_dim = 2
        domain_shift_input_dim = 1
        model = DQN(state_dim, action_dim, domain_shift_input_dim)

        batch_size = 5
        # Generate dummy input
        x = torch.randn(batch_size, state_dim)
        # Domain shift input
        # Based on DomainShift/DQNClass.py:
        # domain_shift = domain_shift.view(-1, 1)
        # x = torch.cat((x, domain_shift), dim=1)
        # So domain_shift should be compatible with view(-1, 1) -> likely [batch_size] or [batch_size, 1]

        domain_shift = torch.randn(batch_size)

        output = model(x, domain_shift)
        self.assertEqual(output.shape, (batch_size, action_dim))

if __name__ == '__main__':
    unittest.main()
