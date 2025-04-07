import gymnasium as gym
import numpy as np
import random
import math

class CustomCartPoleEnv(gym.Wrapper):
    """
    Custom CartPole environment that implements domain shifts by changing gravity and cart mass.
    """
    def __init__(self, render_mode=None):
        # Create the base CartPole environment
        env = gym.make('CartPole-v1', render_mode=render_mode)
        super().__init__(env)
        
        # Store the original parameters
        self.original_gravity = 9.8
        self.original_masscart = 1.0
        self.original_masspole = 0.1
        self.original_length = 0.5
        
        # Domain shift parameters
        self.min_gravity_change = -2.0
        self.max_gravity_change = 2.0
        self.min_masscart_change = -0.3
        self.max_masscart_change = 0.3
        
        # Track episode and step count
        self.episode = 0
        self.current_step = 0
        
        # Current domain shift parameters
        self.current_gravity = self.original_gravity
        self.current_masscart = self.original_masscart
        
        self.logger = None

    def change_domain(self):
        """Apply a random domain shift by changing gravity and cart mass."""
        gravity_change = random.uniform(self.min_gravity_change, self.max_gravity_change)
        masscart_change = random.uniform(self.min_masscart_change, self.max_masscart_change)
        
        # Update the unwrapped environment parameters
        self.unwrapped.gravity = self.original_gravity + gravity_change
        self.unwrapped.masscart = self.original_masscart + masscart_change
        
        # Store current values for logging
        self.current_gravity = self.unwrapped.gravity
        self.current_masscart = self.unwrapped.masscart
        
        # Return to update the physics model
        return self.unwrapped.gravity, self.unwrapped.masscart

    def step(self, action):
        """Override step method to introduce domain shift at each step."""
        # Change domain parameters
        self.change_domain()
        domain_shift = self.quantify_domain_shift()
        
        # Execute the step in the environment
        observation, reward, terminated, truncated, info = super().step(action)
        self.current_step += 1
        
        return (observation, reward, terminated, truncated, info), domain_shift

    def reset(self, **kwargs):
        """Reset the environment and domain parameters."""
        # Reset to original parameters
        self.unwrapped.gravity = self.original_gravity
        self.unwrapped.masscart = self.original_masscart
        
        # Reset episode counter
        self.episode += 1
        self.current_step = 0
        
        # Call the parent reset method
        state = super().reset(**kwargs)
        return state, {}

    def quantify_domain_shift(self):
        """Calculate a metric that represents the magnitude of domain shift."""
        gravity_diff = abs(self.original_gravity - self.unwrapped.gravity) / self.original_gravity
        mass_diff = abs(self.original_masscart - self.unwrapped.masscart) / self.original_masscart
        
        # Combine the two shift metrics
        domain_shift = (gravity_diff + mass_diff) / 2.0
        return domain_shift

    def set_logger(self, logger):
        """Set a logger to record environment data."""
        self.logger = logger