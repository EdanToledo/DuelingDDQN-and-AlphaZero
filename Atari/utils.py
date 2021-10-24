from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done"))

class Replay_Memory:

    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            return None
        return random.sample(self.memory, batch_size)
    
