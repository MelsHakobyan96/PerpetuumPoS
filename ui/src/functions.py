import random
import pickle
from os import listdir, walk, path
from os.path import isfile, join
from cv2 import VideoWriter, VideoWriter_fourcc
import torch
import numpy as np
import imageio
import csv
import os


def submit(form):
    if form.get("btn1"):
        return [1, 0]
    elif form.get("btn2"):
        return [0, 1]
    elif form.get("btn3"):
        return [0.5, 0.5]
    else:
        return [0, 0]


def unpickle(path):
    with open(path, 'rb') as fin:
        data = pickle.load(fin)
    return data


def select_random_data(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    rand_file_names = random.sample(files, 2)
    values = list()

    for file_name in rand_file_names:
        values.append(unpickle(path + file_name))
        values.append(file_name)

    return values


def random_data(path='./data/reward/'):
    episodes_data = select_random_data(path)
    episode_1, name_1, episode_2, name_2 = episodes_data
    images_1, images_2 = episode_1['images'], episode_2['images']

    return {
        'episode_1': np.array(torch.stack(images_1), dtype=np.uint8),
        'episode_1_name': name_1,
        'episode_2': np.array(torch.stack(images_2), dtype=np.uint8),
        'episode_2_name': name_2,
    }


def make_video(images, name):
    # have a folder of output where output files could be stored.
    write_to = '{}.mp4'.format(name)

    if os.path.exists(write_to):
        os.remove(write_to)

    writer = imageio.get_writer(write_to, format='mp4', mode='I', fps=30)

    for im in images:
        writer.append_data(im.T)
    writer.close()


def convert_episodes(data, path='./ui/src/static/videos/'):
    for i in range(1, 3):
        name = 'episode_' + str(i)
        make_video(data[name], path + name)


def save_csv(row, path='./data/target.csv'):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def extra_files(path='./ui/src/static/videos/'):
    extras = list()

    for dirname, dirs, files in walk(path):
        for filename in files:
            filename = join(dirname, filename)
            if isfile(filename):
                extras.append(filename)

    return extras
