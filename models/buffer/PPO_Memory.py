class PPO_Memory():
    def __init__(self):
        self.actions = []
        self.images = []
        self.meta_data = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.images[:]
        del self.meta_data[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]