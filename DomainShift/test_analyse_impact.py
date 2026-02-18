import sys
import unittest
from unittest.mock import MagicMock
import os
import tempfile

# Mock dependencies that might be missing in the environment
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["numpy"] = MagicMock()
import numpy as np
sys.modules["numpy"].mean.side_effect = lambda x: sum(x)/len(x) if x else 0

import Analyse_Domain_Shift_Impact as analyse

class TestAnalyseImpact(unittest.TestCase):
    def setUp(self):
        self.test_fd, self.test_csv = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(self.test_fd, "w") as f:
            f.write("episode,domain_shift,cumulative_reward\n")
            f.write("1,0.1,10.5\n")
            f.write("1,0.1,11.5\n")
            f.write("2,0.2,20.0\n")
            f.write("2,0.2,22.0\n")
            f.write("3,0.3,30.0\n")

    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

    def test_calculate_mean_rewards(self):
        mean_rewards, mean_domain_shifts = analyse.calculate_mean_rewards(self.test_csv)

        self.assertEqual(mean_rewards[1], 11.0)
        self.assertEqual(mean_rewards[2], 21.0)
        self.assertEqual(mean_rewards[3], 30.0)

        self.assertAlmostEqual(mean_domain_shifts[1], 0.1)
        self.assertAlmostEqual(mean_domain_shifts[2], 0.2)
        self.assertAlmostEqual(mean_domain_shifts[3], 0.3)

if __name__ == "__main__":
    unittest.main()
