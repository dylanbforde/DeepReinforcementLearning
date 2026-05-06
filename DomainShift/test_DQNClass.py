import unittest

import torch

from DQNClass import DQN


class TestDQN(unittest.TestCase):
    def test_default_hidden_dim(self):
        model = DQN(state_dim=10, action_dim=2, domain_shift_input_dim=1)

        self.assertEqual(model.layer1.out_features, 128)
        self.assertEqual(model.layer2.out_features, 128)
        self.assertEqual(model.layer3.in_features, 128)

    def test_custom_hidden_dim(self):
        model = DQN(state_dim=10, action_dim=2, domain_shift_input_dim=1, hidden_dim=64)

        self.assertEqual(model.layer1.out_features, 64)
        self.assertEqual(model.layer2.out_features, 64)
        self.assertEqual(model.layer3.in_features, 64)

    def test_forward_shape(self):
        model = DQN(state_dim=10, action_dim=2, domain_shift_input_dim=1)

        x = torch.randn(5, 10)
        domain_shift = torch.randn(5)
        output = model(x, domain_shift)

        self.assertEqual(output.shape, (5, 2))

    def test_identity_output_activation(self):
        model = DQN(state_dim=10, action_dim=2, domain_shift_input_dim=1, output_activation='identity')

        x = torch.randn(5, 10)
        domain_shift = torch.randn(5)
        output = model(x, domain_shift)

        self.assertEqual(output.shape, (5, 2))


if __name__ == '__main__':
    unittest.main()
