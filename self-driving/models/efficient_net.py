import torch
import torch.nn as nn

from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet


class EfficientGRU(nn.Module):
    def __init__(self, hidden_size, gru_num_layers):
        super(EfficientGRU, self).__init__()

        self.hidden_size = hidden_size
        self.rnn_input_size = 1280
        self.gru_num_layers = gru_num_layers

        self.eff_net = EfficientNet.from_name(
            'efficientnet-b0', include_top=False)

        self.pool = nn.AvgPool2d(2)
        self.gru = nn.GRU(self.rnn_input_size, hidden_size,
                          num_layers=gru_num_layers)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs, prev_hidden=None, is_new_video=False):
        """
        inputs is in shape of (batch_size, agent_num, channel_num, map_width, map_height)
        """
        batch_size = inputs.size(0)

        eff_out = self.eff_net(inputs)

        flat = torch.flatten(eff_out, 1)
        flat = flat.view(1, batch_size, -1)

        if is_new_video or prev_hidden is None:
            gru_hidden = self.init_hidden(batch_size)
        else:
            gru_hidden = prev_hidden

        output, gru_hidden = self.gru(flat, gru_hidden)
        output = self.linear(output)
        return output, gru_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.gru_num_layers, batch_size, self.hidden_size).zero_())
