import unittest
import os
import csv
from DomainShift.DataLoggerClass import DataLogger

class TestDataLogger(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_log.csv"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_buffering(self):
        logger = DataLogger(self.test_file, buffer_size=5)

        # Log 4 steps (should be buffered)
        for i in range(4):
            logger.log_step(episode=1, step=i, original_gravity=0, current_gravity=0, action=0, reward=0, domain_shift=0, cumulative_reward=0, epsilon=0, loss=0, predicted_suitability=0)

        # Check file content - should only have header
        with open(self.test_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1) # Header only

        # Log 1 more step (should trigger flush)
        logger.log_step(episode=1, step=4, original_gravity=0, current_gravity=0, action=0, reward=0, domain_shift=0, cumulative_reward=0, epsilon=0, loss=0, predicted_suitability=0)

        with open(self.test_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1 + 5) # Header + 5 rows

    def test_flush_explicit(self):
        logger = DataLogger(self.test_file, buffer_size=10)
        logger.log_step(episode=1, step=1, original_gravity=0, current_gravity=0, action=0, reward=0, domain_shift=0, cumulative_reward=0, epsilon=0, loss=0, predicted_suitability=0)

        with open(self.test_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)

        logger.flush()

        with open(self.test_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)

    def test_close_flushes(self):
        logger = DataLogger(self.test_file, buffer_size=10)
        logger.log_step(episode=1, step=1, original_gravity=0, current_gravity=0, action=0, reward=0, domain_shift=0, cumulative_reward=0, epsilon=0, loss=0, predicted_suitability=0)

        logger.close()

        with open(self.test_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)

if __name__ == '__main__':
    unittest.main()
