import unittest

from InitEnvironment import get_environment_config, initialize_environment


class TestEnvironmentConfig(unittest.TestCase):
    def test_bipedal_dsp_defaults(self):
        config = get_environment_config('bipedal', 'dsp')

        self.assertEqual(config['environment'], 'bipedal')
        self.assertEqual(config['mode'], 'dsp')
        self.assertEqual(config['action_mode'], 'continuous')
        self.assertEqual(config['output_activation'], 'tanh')

    def test_cartpole_control_defaults(self):
        config = get_environment_config('cartpole', 'control')

        self.assertEqual(config['environment'], 'cartpole')
        self.assertEqual(config['mode'], 'control')
        self.assertEqual(config['action_mode'], 'discrete')
        self.assertEqual(config['output_activation'], 'identity')

    def test_bipedal_reset_returns_observation_info_pair(self):
        config = get_environment_config('bipedal', 'dsp')
        env, *_ = initialize_environment(config)

        try:
            observation, info = env.reset()
            self.assertEqual(len(observation), env.observation_space.shape[0])
            self.assertIsInstance(info, dict)
        finally:
            env.close()

    def test_cartpole_reset_returns_observation_info_pair(self):
        config = get_environment_config('cartpole', 'control')
        env, *_ = initialize_environment(config)

        try:
            observation, info = env.reset()
            self.assertEqual(len(observation), env.observation_space.shape[0])
            self.assertIsInstance(info, dict)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
