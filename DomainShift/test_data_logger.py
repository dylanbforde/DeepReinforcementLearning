import unittest
import os
import csv
from DataLoggerClass import DataLogger

class TestDataLogger(unittest.TestCase):
    def setUp(self):
        self.filename = 'test_logger.csv'
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_log_step_buffering(self):
        # Buffer size of 2
        logger = DataLogger(self.filename, buffer_size=2)

        # Log first step
        logger.log_step(
            episode=1, step=1, original_gravity=9.8, current_gravity=9.8,
            action=[0.0], reward=1.0, domain_shift=0.0, cumulative_reward=1.0,
            epsilon=0.1, loss=0.0, predicted_suitability=0.5
        )

        # Verify NOT in file yet
        with open(self.filename, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 0, "Data should be buffered, not written yet")

        # Log second step (should trigger flush)
        logger.log_step(
            episode=1, step=2, original_gravity=9.8, current_gravity=9.8,
            action=[0.0], reward=1.0, domain_shift=0.0, cumulative_reward=1.0,
            epsilon=0.1, loss=0.0, predicted_suitability=0.5
        )

        # Verify BOTH in file
        with open(self.filename, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2, "Data should be flushed after buffer limit reached")
            self.assertEqual(rows[0]['step'], '1')
            self.assertEqual(rows[1]['step'], '2')

    def test_close_flushes(self):
        logger = DataLogger(self.filename, buffer_size=10)
        logger.log_step(
            episode=1, step=1, original_gravity=9.8, current_gravity=9.8,
            action=[0.0], reward=1.0, domain_shift=0.0, cumulative_reward=1.0,
            epsilon=0.1, loss=0.0, predicted_suitability=0.5
        )

        # Verify NOT in file yet
        with open(self.filename, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 0)

        logger.close()

        # Verify in file
        with open(self.filename, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)

if __name__ == '__main__':
    unittest.main()
