import torch
from torch.utils.data import Dataset


class RewardDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(self)

    def __getitem__(self, index):
        return


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return torch.tensor(data), torch.tensor(target)
