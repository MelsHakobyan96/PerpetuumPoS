import random
import pickle
import collections
import numpy as np

from os import listdir
from os.path import isfile, join


def unpickle(path='./data/test.pickle'):
    with open(path, 'rb') as fin:
        data = pickle.load(fin)
    return data


def image_preprocess(image):
    image += np.mean(image)
    image = image.astype(np.uint8)
    return image


def preprocess(episode, keys=None):
    data = episode['data']
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


def get_clean_data(path, keys):
    data = unpickle(path)
    values = preprocess(data, keys)

    return values

def random_data(path='./data/reward/', keys=None):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    rand_file_names = random.sample(files, 2)
    values = list()

    for file_name in rand_file_names:
        values.extend(get_clean_data(path + file_name, keys))


    return values
