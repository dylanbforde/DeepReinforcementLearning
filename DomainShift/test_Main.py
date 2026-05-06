import unittest

from Main import build_parser


class TestMainParser(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.env, 'bipedal')
        self.assertEqual(args.mode, 'dsp')

    def test_cartpole_control_args(self):
        args = build_parser().parse_args(['--env', 'cartpole', '--mode', 'control', '--trials', '2', '--episodes', '3'])

        self.assertEqual(args.env, 'cartpole')
        self.assertEqual(args.mode, 'control')
        self.assertEqual(args.n_trials, 2)
        self.assertEqual(args.num_episodes, 3)


if __name__ == '__main__':
    unittest.main()
