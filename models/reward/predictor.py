import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, Dropout2d, Linear, ReLU


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class RewardPredictor(nn.Module):
    def __init__(self, cnn_input_size, nn_input_size, output_size, base, base_kwargs=None):
        """
            Uses the base function (CNN or MLP) to train the predictor.
        """
        super(RewardPredictor, self).__init__()
        self.cnn_layer = CNNBase(cnn_input_size)
        self.mlp_layer = MLPBase(nn_input_size)

        self.last_layer = LastNNLayer(output_size)

    def forward(self, s1, s2, data):
        """
            Given 2 segments we train the network based on the preference of the user.
        """
        # Raise error if the input shapes do not match
        assert s1.shape == s2.shape

        cnn_input = torch.cat((s1, s2), dim=0)
        cnn_output = self.cnn_layer.forward(cnn_input)

        mlp_output = self.mlp_layer.forward(data)

        #  Raise error if the input shapes do not match
        assert cnn_output.shape == mlp_output.shape

        last_layer_input = torch.cat((cnn_output, mlp_output), dim=0)
        last_layer_output = self.last_layer.forward(last_layer_input)

        return last_layer_output

    def reward(self):
        raise NotImplemented

    def __str__(self):
        return 'Reward Predictor'


class MLPBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=1024, output_size=576):
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

        self.critic_linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = self.layers(inputs)
        return self.critic_linear(x)

    def __str__(self):
        return 'MLP'


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

    def __str__(self):
        return 'CNN'


class LastNNLayer(nn.Module):
    def __init__(self, num_inputs, hidden_size=512, output_size=1):
        """
            A layer to take the concatenated output vectors and produce a prediction.
        """
        super(LastNNLayer, self).__init__()

        self.layers = nn.Sequential(
            Linear(num_inputs, hidden_size),
            ReLU(),
        )

        self.critic_linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = self.layers(inputs)
        return self.critic_linear(x)

    def __str__(self):
        return 'Prediction Layer'
