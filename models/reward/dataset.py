import json

import numpy as np
import torch
from torch.utils.data import Dataset
from train import train


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
        return (torch.tensor(self.s1[index]),
                torch.tensor(self.s2[index]),
                torch.tensor(self.data),
                torch.tensor(self.target))


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return torch.tensor(data), torch.tensor(target)


if __name__ == '__main__':
    json_file_path = './data/test.json'

    with open(json_file_path, 'r') as j:
        d = json.loads(j.read())

    arr = np.expand_dims(np.zeros((3, 85, 85)), axis=0)
    target = [0, 1]
    rd = RewardDataset(arr, arr, d, target)
    train(rd, 1, lr=3e-4)
