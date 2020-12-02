import torch
from torch.utils.data import Dataset


class RewardDataset(Dataset):
    def __init__(self, s1, s2, data, label):
        def _min_frame(s1, s2):
            return min(len(s1), len(s2))

        self.min_frame = _min_frame(s1, s2)

        self.s1 = self.image_preprocess(s1)
        self.s2 = self.image_preprocess(s2)
        self.data = self.parameter_preprocess(data)
        self.target = label
        self.mlp_inp_size = self.data.shape[1]

    def parameter_preprocess(self, data):
        def _flatten(array):
            return array.flatten()

        output = []
        for val_dict in data:
            val_dict = val_dict[:self.min_frame]
            for i in range(self.min_frame):
                arrays = [_flatten(torch.tensor(val)) for val in val_dict[i].values()]
                arrays = torch.cat(arrays)
                output.append(_flatten(arrays))

        output = torch.stack(output, dim=0)
        return output

    def image_preprocess(self, image):
        return torch.Tensor(image[:self.min_frame])

    def __len__(self):
        return self.s1.shape[0]

    def __getitem__(self, index):
        return self.s1[index], self.s2[index], self.data[index], torch.Tensor(self.target)
