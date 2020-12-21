import random
from os.path import isfile, join
from os import listdir
import numpy as np
import collections
import json
import pickle

import torch
from models.reward.dataset import RewardDataset


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
    return image


def preprocess(data, keys=None):
    # data = episode['data']
    length = len(data)
    images = []
    values = []

    for i in range(length):
        d = flatten_nested_dicts(data[i])

        # Getting the image
        image = image_preprocess(d['image'])

        # Getting the required data
        img_data = {key: val for key, val in d.items() if key in keys}
        images.append(image)
        values.append(img_data)

    return images, values


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


def save_json(data, path):
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=4)


def unpickle(path='./data/test.pickle'):
    with open(path, 'rb') as fin:
        data = pickle.load(fin)
    return data


def select_random_data(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    rand_file_names = random.sample(files, 2)
    values = list()

    for file_name in rand_file_names:
        values.extend(read_json(path + file_name))

    return values


def random_data(path='./data/reward/'):
    episodes_data = select_random_data(path)

    episode_1, episode_2 = episodes_data
    images_1, images_2 = episode_1['images'], episode_2['images']

    del episode_1['images']
    del episode_2['images']

    # TODO:
    target = None

    rd = RewardDataset(images_1, images_2, (episode_1, episode_2), target)
    return rd
