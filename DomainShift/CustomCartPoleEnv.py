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
        """
        Calculate an enhanced metric that represents the magnitude of domain shift.
        Uses Euclidean distance in the parameter space for better representation.
        Returns a value between 0 and 1 representing the magnitude of the shift.
        """
        # Calculate the normalized gravity shift
        gravity_diff = abs(self.original_gravity - self.unwrapped.gravity) / self.original_gravity
        
        # Calculate the normalized mass shift
        mass_diff = abs(self.original_masscart - self.unwrapped.masscart) / self.original_masscart
        
        # Add pole angle component (if the pole is already significantly tilted, environment is more challenging)
        pole_angle_component = 0.0
        if hasattr(self, 'unwrapped') and hasattr(self.unwrapped, 'state'):
            # CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
            if len(self.unwrapped.state) >= 3:
                # Normalize pole angle to [0, 1] range where 1 is close to falling
                # The pole angle threshold at which the episode terminates is typically 12 degrees (0.2095 radians)
                pole_angle = abs(self.unwrapped.state[2])
                pole_angle_component = min(pole_angle / 0.209, 1.0) * 0.5  # Scale component to max 0.5
            
        # Use Euclidean distance for more accurate shift representation
        # Square root of weighted sum of squares: sqrt(w1*x1^2 + w2*x2^2 + w3*x3^2)
        domain_shift = math.sqrt(
            0.4 * (gravity_diff ** 2) +  # 40% weight to gravity
            0.4 * (mass_diff ** 2) +     # 40% weight to mass
            0.2 * (pole_angle_component ** 2)  # 20% weight to pole angle
        )
        
        # Ensure result is in [0, 1] range
        return min(domain_shift, 1.0)

    def set_logger(self, logger):
        """Set a logger to record environment data."""
        self.logger = logger