from collections import deque
from collections import namedtuple
import random
max_size = 20000
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'terminal'])
Batch = namedtuple('Batch', ['data', 'size'])
class replay_memory:
    def __init__(self):
        self.data = deque(maxlen=max_size)
        print("memory size ", max_size)
        pass

    def __len__(self):
        return len(self.data)

    def is_full(self):
        return self.__len__() == max_size

    def add(self, state, action, reward, next_state, done):
        terminal = 1
        if done:
            terminal = 0

        self.data.append(Transition(state.copy(), action, reward, next_state.copy(), terminal))

    def sample(self, batch_size):
        sample_size = min(self.__len__(), batch_size)
        return Batch(random.sample(self.data, sample_size), sample_size)


