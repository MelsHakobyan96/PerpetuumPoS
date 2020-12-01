import collections
import json
import pickle

import numpy as np
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


def image_preprocess(image):
    image += np.mean(image)
    image = image.astype(np.uint8)
    return np.expand_dims(image, axis=0)


def preprocess(episode, keys=None):
    data = episode['data']
    length = len(data)
    values = []

    for i in range(length):
        d = flatten_nested_dicts(data[i])

        # Getting the image
        image = image_preprocess(d['image'])

        # Getting the required data
        img_data = {key: val for key, val in d.items() if key in keys}
        values.append((image, img_data))

    return values


def flatten_nested_dicts(d):
    """
        Flattening (collapsing the nested keys) of the given dict into one big one.
    """
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dicts(v).items())
        elif isinstance(v, list):
            if len(v) < 1:
                items.append((new_key, v))
            else:
                items.extend(v[0].items())
        else:
            items.append((new_key, v))

    return dict(items)
