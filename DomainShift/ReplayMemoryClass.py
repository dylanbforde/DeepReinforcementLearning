import random
from collections import namedtuple


class ReplayMemory(object):
    """
    A simple implementation of replay memory.
    
    Attributes:
        memory (list): A list to store the transitions with a maximum length.
        capacity (int): The maximum number of transitions to store.
        position (int): The current position in the memory to store the next transition.
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'domain_shift', 'done'))
