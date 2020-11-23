import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, Dropout2d, Linear, ReLU


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class RewardPredictor(nn.Module):
    def __init__(self):
        super(RewardPredictor, self).__init__()

    def forward(self):
        pass

    def reward(self):
        pass


class MLPBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=64):
        """
            The input in the paper for Simulated Robotics Tasks.
            Copied the architecture for now.
            Will change it according to our own task.
        """
        super(MLPBase, self).__init__()

        self.layers = nn.Sequential(
            Linear(num_inputs, hidden_size),
            LeakyReLU(0.01, inplace=True),

            Linear(hidden_size, hidden_size),
            LeakyReLU(0.01, inplace=True),
        )

        self.critic_linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = self.layers(inputs)
        return self.critic_linear(x)


class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=64):
        """
            The input in the paper for Atari are images of size 84x84.
            Copied the architecture for now.
            Will change it according to our own task.
        """
        super(CNNBase, self).__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs

        self.layers = nn.Sequential(
            Conv2d(self.num_inputs, 16, kernel_size=7, stride=3),
            BatchNorm2d(16),
            Dropout2d(p=0.5),
            LeakyReLU(0.01, inplace=True),

            Conv2d(16, 16, kernel_size=5, stride=2),
            BatchNorm2d(16),
            Dropout2d(p=0.5),
            LeakyReLU(0.01, inplace=True),

            Conv2d(16, 16, kernel_size=3, stride=1),
            BatchNorm2d(16),
            Dropout2d(p=0.5),
            LeakyReLU(0.01, inplace=True),

            Conv2d(16, 16, kernel_size=3, stride=1),
            BatchNorm2d(16),
            Dropout2d(p=0.5),
            LeakyReLU(0.01, inplace=True),

            Flatten(),
            Linear(16 * 6 * 6, self.hidden_size)
        )

        self.critic_linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = self.layers(inputs / 255.0)
        return self.critic_linear(x)
