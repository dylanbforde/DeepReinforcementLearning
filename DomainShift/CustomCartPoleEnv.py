import math
import random

import gymnasium as gym


class CustomCartPoleEnv(gym.Wrapper):
    """
    CartPole environment with step-level gravity and cart-mass domain shifts.
    """

    def __init__(self, render_mode=None):
        super().__init__(gym.make('CartPole-v1', render_mode=render_mode))
        self.original_gravity = 9.8
        self.original_masscart = 1.0
        self.original_masspole = 0.1
        self.original_length = 0.5
        self.min_gravity_change = -2.0
        self.max_gravity_change = 2.0
        self.min_masscart_change = -0.3
        self.max_masscart_change = 0.3
        self.current_gravity = self.original_gravity
        self.current_masscart = self.original_masscart
        self.episode = 0
        self.current_step = 0
        self.logger = None

    def change_domain(self):
        gravity_change = random.uniform(self.min_gravity_change, self.max_gravity_change)
        masscart_change = random.uniform(self.min_masscart_change, self.max_masscart_change)
        self.unwrapped.gravity = self.original_gravity + gravity_change
        self.unwrapped.masscart = self.original_masscart + masscart_change
        self.current_gravity = self.unwrapped.gravity
        self.current_masscart = self.unwrapped.masscart

    def step(self, action):
        self.change_domain()
        domain_shift = self.quantify_domain_shift()
        observation, reward, terminated, truncated, info = super().step(action)
        self.current_step += 1
        return (observation, reward, terminated, truncated, info), domain_shift

    def reset(self, **kwargs):
        self.unwrapped.gravity = self.original_gravity
        self.unwrapped.masscart = self.original_masscart
        self.current_gravity = self.original_gravity
        self.current_masscart = self.original_masscart
        self.episode += 1
        self.current_step = 0
        return super().reset(**kwargs)

    def quantify_domain_shift(self):
        gravity_diff = abs(self.original_gravity - self.unwrapped.gravity) / self.original_gravity
        mass_diff = abs(self.original_masscart - self.unwrapped.masscart) / self.original_masscart
        pole_angle_component = 0.0
        if getattr(self.unwrapped, 'state', None) is not None and len(self.unwrapped.state) >= 3:
            pole_angle = abs(self.unwrapped.state[2])
            pole_angle_component = min(pole_angle / 0.209, 1.0) * 0.5
        domain_shift = math.sqrt(
            0.4 * (gravity_diff ** 2)
            + 0.4 * (mass_diff ** 2)
            + 0.2 * (pole_angle_component ** 2)
        )
        return min(domain_shift, 1.0)

    def set_logger(self, logger):
        self.logger = logger
