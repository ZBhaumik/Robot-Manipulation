import numpy as np
from collections import deque, namedtuple
import torch

Experience = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, maxsize):
        self.buffer = deque(maxlen=maxsize)
        pass

    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float()
        action = torch.tensor(action)
        reward = torch.tensor([reward])
        next_state = torch.from_numpy(next_state).float()
        t = Experience(state, action, reward, next_state, done)
        self.buffer.append(t)
        pass

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def sample(self, batch_size):
        rng = np.random.default_rng()
        indices = rng.choice(np.arange(0, len(self.buffer)), batch_size, replace=False)
        bufferList = list(self.buffer)
        samples =[bufferList[i] for i in indices]
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for sample in samples:
            states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            next_states.append(sample[3])
            dones.append(float(sample[4]))
        return (torch.stack(states), torch.tensor(actions).unsqueeze(-1), torch.tensor(rewards), torch.stack(next_states), torch.tensor(dones))