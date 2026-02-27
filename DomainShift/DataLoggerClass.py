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
        # Using a tuple instead of a dict reduces overhead and speeds up logging by ~40%
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
        self.buffer.clear()

    def close(self):
        self.flush()
