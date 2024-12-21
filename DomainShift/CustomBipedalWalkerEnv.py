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
        gravity_shift = abs(self.original_gravity[1] - self.world.gravity[1])
        return gravity_shift

    def set_logger(self, logger):
        self.logger = logger