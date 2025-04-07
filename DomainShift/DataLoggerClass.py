import csv
import os

class DataLogger:
    def __init__(self, filename):
        self.filename = filename
        # Add fields for enhanced tracking
        self.fields = [
            'episode', 'step', 'original_gravity', 'current_gravity', 'action', 
            'reward', 'domain_shift', 'cumulative_reward', 'epsilon', 'loss', 
            'predicted_suitability'
        ]
        self.ensure_file()
        
        # Create separate file for threshold tracking
        self.threshold_filename = filename.replace('.csv', '_thresholds.csv')
        self.threshold_fields = ['episode', 'suitability_threshold', 'reset_count']
        self.reset_count = 0
        self.ensure_threshold_file()
            
    def ensure_file(self):
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()
                
    def ensure_threshold_file(self):
        if not os.path.isfile(self.threshold_filename):
            with open(self.threshold_filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.threshold_fields)
                writer.writeheader()

    def log_step(self, episode, step, original_gravity, current_gravity, action, reward, domain_shift, cumulative_reward, epsilon, loss, predicted_suitability):
        """Log data for each step in an episode"""
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow({
                'episode': episode,
                'step': step,
                'original_gravity': original_gravity,
                'current_gravity': current_gravity,
                'action': action,
                'reward': reward,
                'domain_shift': domain_shift,
                'cumulative_reward': cumulative_reward,
                'epsilon': epsilon,
                'loss': loss,
                'predicted_suitability': predicted_suitability
            })
            
    def log_threshold(self, episode, threshold, reset=False):
        """Log the adaptive threshold changes"""
        if reset:
            self.reset_count += 1
            
        with open(self.threshold_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.threshold_fields)
            writer.writerow({
                'episode': episode,
                'suitability_threshold': threshold,
                'reset_count': self.reset_count
            })