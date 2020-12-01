from models.reward.dataset import RewardDataset
from models.reward.train import train
from models.util.preprocess import *

if __name__ == '__main__':
    data = unpickle()
    keys = read_txt()
    values = preprocess(data, keys)

    arr_1 = values[0][0]
    arr_2 = values[1][0]
    d = [values[0][1], values[1][1]]
    target = [0, 1]
    rd = RewardDataset(arr_1, arr_2, d, target)
    train(rd, 1, lr=3e-4)
