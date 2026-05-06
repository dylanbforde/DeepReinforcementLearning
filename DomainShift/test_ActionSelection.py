import unittest

import torch
import torch.nn as nn

from ActionSelection import ActionSelector


class DummyPolicy(nn.Module):
    def forward(self, state, domain_shift):
        return torch.ones((state.shape[0], 3), dtype=torch.float32)


class TestActionSelector(unittest.TestCase):
    def test_random_action_shape_and_bounds(self):
        selector = ActionSelector(
            policy_net=DummyPolicy(),
            action_dim=3,
            device=torch.device('cpu'),
            EPS_START=1.0,
            EPS_END=1.0,
            EPS_DECAY=1000,
        )

        action = selector.select_action(torch.zeros((1, 4)), torch.zeros(1))

        self.assertEqual(action.shape, (1, 3))
        self.assertTrue(torch.all(action >= -1.0))
        self.assertTrue(torch.all(action <= 1.0))

    def test_discrete_random_action_shape_type_and_bounds(self):
        selector = ActionSelector(
            policy_net=DummyPolicy(),
            action_dim=3,
            device=torch.device('cpu'),
            EPS_START=1.0,
            EPS_END=1.0,
            EPS_DECAY=1000,
            action_mode='discrete',
        )

        action = selector.select_action(torch.zeros((1, 4)), torch.zeros(1))

        self.assertEqual(action.shape, (1, 1))
        self.assertEqual(action.dtype, torch.long)
        self.assertGreaterEqual(action.item(), 0)
        self.assertLess(action.item(), 3)

    def test_epsilon_helpers_apply_factors(self):
        selector = ActionSelector(
            policy_net=DummyPolicy(),
            action_dim=3,
            device=torch.device('cpu'),
            EPS_START=0.9,
            EPS_END=0.05,
            EPS_DECAY=1000,
        )

        selector.update_epsilon(factor=0.5)
        self.assertAlmostEqual(selector.EPS_START, 0.45)

        selector.steps_done = 10
        selector.reset_epsilon(factor=0.8)
        self.assertAlmostEqual(selector.EPS_START, 0.36)
        self.assertEqual(selector.steps_done, 0)


if __name__ == '__main__':
    unittest.main()
