from models.reward.dataset import RewardDataset
from models.reward.train import train
from utils.preprocess import *
from models.buffer.select import random_data
from models.agent.algorithm import PPO
from data_utils.PPO_Memory import PPO_Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # keys = read_txt()
    # images_1, data_1, images_2, data_2 = random_data(path='./data/reward/', keys=keys)

    # target = [0, 1]
    # rd = RewardDataset(images_1, images_2, (data_1, data_2), target)
    # print(rd.mlp_inp_size)
    # # train(rd, 2, lr=3e-4)

    mlp_input_size=14
    action_size=2
    agent_lr=3e-4
    betas=(0.9, 0.999)
    gamma=0.99
    K_epochs=10
    eps_clip=0.2
    reward_pred_lr=3e-4

    

