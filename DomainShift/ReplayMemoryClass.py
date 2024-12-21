import random
from collections import deque, namedtuple


class ReplayMemory(object):
    """
    A simple implementation of replay memory.
    
    Attributes:
        memory (deque): A double-ended queue to store the transitions with a maximum length.
    """
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'domain_shift'))
