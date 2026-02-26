import csv
import os

class DataLogger:
    def __init__(self, filename, buffer_size=1000):
        self.filename = filename
        self.fields = ['episode', 'step', 'original_gravity', 'current_gravity', 'action', 'reward', 'domain_shift', 'cumulative_reward', 'epsilon', 'loss', 'predicted_suitability']
        self.buffer = []
        self.buffer_size = buffer_size
        self.ensure_file()

    def ensure_file(self):
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.fields)

    def log_step(self, episode, step, original_gravity, current_gravity, action, reward, domain_shift, cumulative_reward, epsilon, loss, predicted_suitability):
        # Optimization: Buffer data as tuples instead of dictionaries to reduce memory overhead and improve flush speed.
        # Order must match self.fields defined in __init__.
        self.buffer.append((
                episode,
                step,
                original_gravity,
                current_gravity,
                action,
                reward,
                domain_shift,
                cumulative_reward,
                epsilon,
                loss,
                predicted_suitability
        ))
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.buffer)
        self.buffer = []

    def close(self):
        self.flush()
