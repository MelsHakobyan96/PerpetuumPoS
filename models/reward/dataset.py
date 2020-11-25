from operator import itemgetter

import numpy as np
import torch
from torch.utils.data import Dataset


class RewardDataset(Dataset):
    def __init__(self, s1, s2, data, label):
        self.s1 = s1
        self.s2 = s2
        self.data = self.preprocess(data)
        self.target = label
        self.sizes = (self.s1.shape[1], len(self.data), self.s1.shape[0])

    @staticmethod
    def preprocess(data):
        # TODO
        # Add encodings for categorical variables
        def _check(x):
            return isinstance(x, bool)

        def _pad(x, max_len):
            return np.pad(x, (0, max_len), 'constant')

        # Flattening the matrix
        data['vehicle_positions'] = data['vehicle_positions'].flatten()

        # Getting the numpy arrays from the dictionary
        num_keys = [x for x, y in data.items() if _check(y) is False]
        num_arrays = itemgetter(*num_keys)(data)

        # Getting the maximum length of the numpy arrays
        max_length = max(list(map(lambda x: len(x), num_arrays)))

        # Padding with zeroes of match the max length
        padded = [_pad(x, max_length - len(x)) for x in num_arrays]

        return padded

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
