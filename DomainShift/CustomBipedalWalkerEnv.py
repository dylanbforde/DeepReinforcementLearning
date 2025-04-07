from gym.envs.box2d import bipedal_walker
import random
import numpy as np

class CustomBipedalWalkerEnv(bipedal_walker.BipedalWalker):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        # Domain shifts
        self.original_gravity = self.world.gravity
        self.min_gravity_change = -0.5
        self.max_gravity_change = 1.0
        self.min_velocity_threshold = 0.1
        self.max_stuck_steps = 300
        self.stuck_steps = 0
        self.episode = 0
        self.current_step = 0  # Renamed from 'step' to 'current_step'
    
    def change_gravity(self):
        gravity_change = random.uniform(self.min_gravity_change, self.max_gravity_change)
        self.world.gravity = (0.0, self.original_gravity[1] + gravity_change)

    def step(self, action):
        self.change_gravity()
        domain_shift = self.quantify_domain_shift()
        observation, reward, terminated, truncated, info = super().step(action)
        self.current_step += 1

        # Check if the robot is stuck
        velocity = np.abs(observation[2])  # Get the horizontal velocity from the observation
        if velocity < self.min_velocity_threshold:
            self.stuck_steps += 1
            if self.stuck_steps >= self.max_stuck_steps:
                self.stuck_steps = 0
                terminated = True
        else:
            self.stuck_steps = 0

        return (observation, reward, terminated, truncated, info), domain_shift

    def reset(self):
        self.world.gravity = self.original_gravity
        self.episode += 1
        self.current_step = 0
        state = super().reset()
        return state, {}
    
    def quantify_domain_shift(self):
        """
        Calculate a normalized domain shift metric.
        Returns a value between 0 and 1 representing the magnitude of the shift.
        """
        # Calculate normalized gravity shift
        gravity_diff = abs(self.original_gravity[1] - self.world.gravity[1])
        # Normalize by the maximum possible shift range
        max_shift_range = abs(self.max_gravity_change - self.min_gravity_change)
        normalized_gravity_shift = gravity_diff / max_shift_range if max_shift_range > 0 else 0
        
        # Add velocity-based component for more comprehensive shift detection
        velocity_component = 0
        if hasattr(self, 'stuck_steps') and self.max_stuck_steps > 0:
            velocity_component = self.stuck_steps / self.max_stuck_steps
        
        # Combine metrics (weighted toward gravity as primary shift factor)
        domain_shift = 0.7 * normalized_gravity_shift + 0.3 * velocity_component
        
        return domain_shift

    def set_logger(self, logger):
        self.logger = logger