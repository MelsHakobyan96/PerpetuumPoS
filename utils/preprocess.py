import json
import pickle

import torch


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return torch.tensor(data), torch.tensor(target)


def unpickle(path='./data/test.pickle'):
    with open(path, 'rb') as fin:
        data = pickle.load(fin)
    return data


def read_json(path='./data/test.json'):
    with open(path, 'r') as j:
        data = json.loads(j.read())

    return data


def read_txt(path='./data/keys.txt'):
    file = open(path, 'r')
    return file.read().split('\n')
