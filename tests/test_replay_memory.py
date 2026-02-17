import unittest
import random
from DomainShift.ReplayMemoryClass import ReplayMemory, Transition

class TestReplayMemory(unittest.TestCase):
    def setUp(self):
        self.capacity = 10
        self.memory = ReplayMemory(self.capacity)

    def test_initialization(self):
        """Test if the ReplayMemory is correctly initialized."""
        self.assertEqual(len(self.memory), 0)
        self.assertEqual(self.memory.memory.maxlen, self.capacity)

    def test_push(self):
        """Test if pushing transitions works correctly."""
        state = [0.1, 0.2]
        action = 1
        next_state = [0.2, 0.3]
        reward = 1.0
        domain_shift = 0.5

        self.memory.push(state, action, next_state, reward, domain_shift)
        self.assertEqual(len(self.memory), 1)

        transition = self.memory.memory[0]
        self.assertIsInstance(transition, Transition)
        self.assertEqual(transition.state, state)
        self.assertEqual(transition.action, action)
        self.assertEqual(transition.next_state, next_state)
        self.assertEqual(transition.reward, reward)
        self.assertEqual(transition.domain_shift, domain_shift)

    def test_capacity_limit(self):
        """Test if the memory correctly handles maximum capacity."""
        for i in range(self.capacity + 5):
            self.memory.push(i, i, i, i, i)

        self.assertEqual(len(self.memory), self.capacity)
        # Check if the oldest elements are removed (deque behavior)
        # Elements pushed were 0, 1, ..., 14.
        # Capacity is 10, so it should keep 5, 6, ..., 14.
        self.assertEqual(self.memory.memory[0].state, 5)
        self.assertEqual(self.memory.memory[-1].state, 14)

    def test_sample(self):
        """Test if sampling from memory returns the correct number of transitions."""
        for i in range(self.capacity):
            self.memory.push(i, i, i, i, i)

        batch_size = 5
        sample = self.memory.sample(batch_size)
        self.assertEqual(len(sample), batch_size)
        for s in sample:
            self.assertIsInstance(s, Transition)

    def test_sample_all(self):
        """Test sampling the entire memory."""
        for i in range(5):
            self.memory.push(i, i, i, i, i)

        sample = self.memory.sample(5)
        self.assertEqual(len(sample), 5)

    def test_sample_exceeds_memory(self):
        """Test that sampling more items than available raises a ValueError."""
        for i in range(5):
            self.memory.push(i, i, i, i, i)

        with self.assertRaises(ValueError):
            self.memory.sample(10)

    def test_len(self):
        """Test the __len__ method."""
        for i in range(3):
            self.memory.push(i, i, i, i, i)
        self.assertEqual(len(self.memory), 3)

if __name__ == '__main__':
    unittest.main()
