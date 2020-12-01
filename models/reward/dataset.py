import torch
from torch.utils.data import Dataset


class RewardDataset(Dataset):
    def __init__(self, s1, s2, data, label):
        self.s1 = s1
        self.s2 = s2
        self.data = self.preprocess(data)
        self.target = label
        self.sizes = (self.s1.shape[1], self.data.shape[1])

    @staticmethod
    def preprocess(data):
        def _flatten(array):
            return array.flatten()

        output = []
        for val_dict in data:
            arrays = [_flatten(torch.tensor(val)) for val in val_dict.values()]
            arrays = torch.cat(arrays)
            output.append(_flatten(arrays))
        return torch.stack(output, dim=0)

    def __len__(self):
        return self.s1.shape[0]

    def __getitem__(self, index):
        return self.s1[index], self.s2[index], self.data, torch.tensor(self.target)
