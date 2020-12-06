from models.reward.dataset import RewardDataset
from models.reward.train import train
from utils.preprocess import *
from models.buffer.select import random_data

if __name__ == '__main__':
    keys = read_txt()
    images_1, data_1, images_2, data_2 = random_data(path='./data/reward/', keys=keys)

    target = [0, 1]
    rd = RewardDataset(images_1, images_2, (data_1, data_2), target)
    train(rd, 2, lr=3e-4)
