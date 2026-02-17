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
        # self.memory.memory is a list, so it has no maxlen attribute
        self.assertIsInstance(self.memory.memory, list)

    def test_push(self):
        """Test if pushing transitions works correctly."""
        state = [0.1, 0.2]
        action = 1
        next_state = [0.2, 0.3]
        reward = 1.0
        domain_shift = 0.5
        done = False

        self.memory.push(state, action, next_state, reward, domain_shift, done)
        self.assertEqual(len(self.memory), 1)

        transition = self.memory.memory[0]
        self.assertIsInstance(transition, Transition)
        self.assertEqual(transition.state, state)
        self.assertEqual(transition.action, action)
        self.assertEqual(transition.next_state, next_state)
        self.assertEqual(transition.reward, reward)
        self.assertEqual(transition.domain_shift, domain_shift)
        self.assertEqual(transition.done, done)

    def test_capacity_limit(self):
        """Test if the memory correctly handles maximum capacity."""
        for i in range(self.capacity + 5):
            # i serves as state, action, etc.
            self.memory.push(i, i, i, i, i, False)

        self.assertEqual(len(self.memory), self.capacity)

        # Circular buffer verification
        # Capacity is 10. We pushed 0 to 14.
        # 0-9 filled the list.
        # 10 overwrote index 0.
        # 11 overwrote index 1.
        # ...
        # 14 overwrote index 4.
        # So memory contents should be:
        # Index 0: 10
        # Index 4: 14
        # Index 5: 5 (from original fill)
        # Index 9: 9 (from original fill)

        self.assertEqual(self.memory.memory[0].state, 10)
        self.assertEqual(self.memory.memory[4].state, 14)
        self.assertEqual(self.memory.memory[5].state, 5)
        self.assertEqual(self.memory.memory[9].state, 9)

    def test_sample(self):
        """Test if sampling from memory returns the correct number of transitions."""
        for i in range(self.capacity):
            self.memory.push(i, i, i, i, i, False)

        batch_size = 5
        sample = self.memory.sample(batch_size)
        self.assertEqual(len(sample), batch_size)
        for s in sample:
            self.assertIsInstance(s, Transition)

    def test_sample_all(self):
        """Test sampling the entire memory."""
        for i in range(5):
            self.memory.push(i, i, i, i, i, False)

        sample = self.memory.sample(5)
        self.assertEqual(len(sample), 5)

    def test_sample_exceeds_memory(self):
        """Test that sampling more items than available raises a ValueError."""
        for i in range(5):
            self.memory.push(i, i, i, i, i, False)

        with self.assertRaises(ValueError):
            self.memory.sample(10)

    def test_len(self):
        """Test the __len__ method."""
        for i in range(3):
            self.memory.push(i, i, i, i, i, False)
        self.assertEqual(len(self.memory), 3)

if __name__ == '__main__':
    unittest.main()
