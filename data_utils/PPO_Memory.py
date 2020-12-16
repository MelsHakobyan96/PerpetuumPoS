import torch
from torch.utils.data import Dataset

class PPO_Memory(Dataset):
    def __init__(self):
        self.actions = []
        self.images = []
        self.meta_data = []
        self.logprobs = []
        self.state_values = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.images[:]
        del self.meta_data[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return torch.FloatTensor(self.actions[index]), torch.FloatTensor(self.images[index]), torch.FloatTensor(self.meta_data[index]), torch.FloatTensor(self.logprobs[index])