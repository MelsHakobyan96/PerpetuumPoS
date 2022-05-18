import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class CNN_GRU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, hidden_size, gru_num_layers):
        super(CNN_GRU, self).__init__()

        self.hidden_size = hidden_size
        self.rnn_input_size = 896
        self.gru_num_layers = gru_num_layers

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels * 4,
                               kernel_size=kernel_size)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels * 4)

        self.conv3 = nn.Conv2d(in_channels=out_channels * 4,
                               out_channels=out_channels * 4,
                               kernel_size=kernel_size)
        self.batchnorm3 = nn.BatchNorm2d(num_features=out_channels * 4)

        self.conv4 = nn.Conv2d(in_channels=out_channels * 4,
                               out_channels=out_channels * 4,
                               kernel_size=kernel_size)
        self.batchnorm4 = nn.BatchNorm2d(num_features=out_channels * 4)

        self.conv5 = nn.Conv2d(in_channels=out_channels * 4,
                               out_channels=out_channels * 4,
                               kernel_size=kernel_size)
        self.batchnorm5 = nn.BatchNorm2d(num_features=out_channels * 4)

        self.conv6 = nn.Conv2d(in_channels=out_channels * 4,
                               out_channels=out_channels * 4,
                               kernel_size=kernel_size)
        self.batchnorm6 = nn.BatchNorm2d(num_features=out_channels * 4)

        self.conv7 = nn.Conv2d(in_channels=out_channels * 4,
                               out_channels=out_channels,
                               kernel_size=kernel_size)
        self.batchnorm7 = nn.BatchNorm2d(num_features=out_channels)

        self.pool = nn.AvgPool2d(2)
        self.gru = nn.GRU(self.rnn_input_size, hidden_size,
                          num_layers=gru_num_layers)

    def forward(self, inputs, prev_hidden=None, is_new_video=False):
        """
        Inputs are in shape of (batch_size, agent_num, channel_num, map_width, map_height)
        """
        batch_size = inputs.size(0)

        # reduce the dimension of the input to fit the CNN
        inputs = inputs.view(-1, inputs.size(-3),
                             inputs.size(-2), inputs.size(-1))

        conved_1 = self.conv1(inputs)
        batched_1 = self.batchnorm1(conved_1)
        leaky_relued_1 = F.leaky_relu(batched_1)
        pooled_1 = self.pool(leaky_relued_1)

        conved_2 = self.conv2(pooled_1)
        batched_2 = self.batchnorm2(conved_2)
        leaky_relued_2 = F.leaky_relu(batched_2)
        pooled_2 = self.pool(leaky_relued_2)

        conved_3 = self.conv3(pooled_2)
        batched_3 = self.batchnorm3(conved_3)
        leaky_relued_3 = F.leaky_relu(batched_3)
        pooled_3 = self.pool(leaky_relued_3)

        conved_4 = self.conv4(pooled_3)
        batched_4 = self.batchnorm4(conved_4)
        leaky_relued_4 = F.leaky_relu(batched_4)
        pooled_4 = self.pool(leaky_relued_4)

        conved_5 = self.conv5(pooled_4)
        batched_5 = self.batchnorm5(conved_5)
        leaky_relued_5 = F.leaky_relu(batched_5)
        pooled_5 = self.pool(leaky_relued_5)

        conved_6 = self.conv5(pooled_5)
        batched_6 = self.batchnorm5(conved_6)
        leaky_relued_6 = F.leaky_relu(batched_6)
        pooled_6 = self.pool(leaky_relued_6)

        conved_7 = self.conv5(pooled_6)
        batched_7 = self.batchnorm5(conved_7)
        leaky_relued_7 = F.leaky_relu(batched_7)
        pooled_7 = self.pool(leaky_relued_7)

        flat = torch.flatten(pooled_7, 1)
        flat = flat.view(1, batch_size, -1)

        if is_new_video or prev_hidden is None:
            gru_hidden = self.init_hidden(batch_size)
        else:
            gru_hidden = prev_hidden

        output, gru_hidden = self.gru(flat, gru_hidden)
        return output, gru_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.gru_num_layers, batch_size, self.hidden_size).zero_()))
