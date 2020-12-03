from models.reward.dataset import RewardDataset
from models.reward.train import train
from utils.preprocess import *

if __name__ == '__main__':
    keys = read_txt()
    images_1, data_1 = get_clean_data('./data/test1.pickle', keys)
    images_2, data_2 = get_clean_data('./data/test2.pickle', keys)

    target = [0, 1]
    rd = RewardDataset(images_1, images_2, (data_1, data_2), target)
    train(rd, 2, lr=3e-4)
