import unittest
import os
import csv
from DataLoggerClass import DataLogger

class TestDataLogger(unittest.TestCase):
    def setUp(self):
        self.filename = 'test_log.csv'
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_buffer_flush(self):
        # Buffer size 5
        logger = DataLogger(self.filename, buffer_size=5)

        # Log 4 steps (should not write to file yet)
        for i in range(4):
            logger.log_step(
                episode=1, step=i, original_gravity=-10, current_gravity=-9.8,
                action=[0.0, 0.0], reward=1.0, domain_shift=0.0,
                cumulative_reward=1.0, epsilon=0.1, loss=0.01, predicted_suitability=0.5
            )

        # Check file content (should only have header)
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1) # Only header

        # Log 1 more step (should trigger flush)
        logger.log_step(
            episode=1, step=4, original_gravity=-10, current_gravity=-9.8,
            action=[0.0, 0.0], reward=1.0, domain_shift=0.0,
            cumulative_reward=1.0, epsilon=0.1, loss=0.01, predicted_suitability=0.5
        )

        # Check file content (should have header + 5 lines)
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 6)

    def test_close_flushes(self):
        logger = DataLogger(self.filename, buffer_size=10)

        # Log 3 steps
        for i in range(3):
            logger.log_step(
                episode=1, step=i, original_gravity=-10, current_gravity=-9.8,
                action=[0.0, 0.0], reward=1.0, domain_shift=0.0,
                cumulative_reward=1.0, epsilon=0.1, loss=0.01, predicted_suitability=0.5
            )

        logger.close()

        # Check file content (should have header + 3 lines)
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 4)

if __name__ == '__main__':
    unittest.main()
