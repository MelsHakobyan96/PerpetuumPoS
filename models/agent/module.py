import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#from aim.pytorch_lightning import AimLogger

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

class CNN_base(nn.Module):

	def __init__(self, height, width, in_channels, out_channels, kernel_size):
		super(CNN_base, self).__init__()

		self.height = height
		self.width = width
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size

		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.flatten = Flatten()

		self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
		self.bn_1 = nn.BatchNorm2d(out_channels)
		self.pool_1 = nn.MaxPool2d(4, stride=2)
		self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*4, kernel_size=kernel_size)
		self.bn_2 = nn.BatchNorm2d(out_channels*4)
		self.pool_2 = nn.MaxPool2d(4, stride=2)
		self.conv_3 = nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*4, kernel_size=kernel_size)
		self.bn_3 = nn.BatchNorm2d(out_channels*4)
		self.pool_3 = nn.MaxPool2d(4, stride=2)
		self.conv_4 = nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*4, kernel_size=kernel_size)
		self.bn_4 = nn.BatchNorm2d(out_channels*4)
		self.conv_5 = nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*4, kernel_size=kernel_size)
		self.bn_5 = nn.BatchNorm2d(out_channels*4)
		self.fcn_1 = nn.Linear(1600, 512)
		self.fcn_2 = nn.Linear(512, 512)
		self.fcn_3 = nn.Linear(512, 256)
		self.fcn_4 = nn.Linear(256, 128)
		self.fcn_5 = nn.Linear(128, 128)
		self.fcn_6 = nn.Linear(128, 128)


	def forward(self, x):

		conv_1_out = self.bn_1(self.pool_1(self.relu(self.conv_1(x))))
		conv_2_out = self.bn_2(self.pool_2(self.relu(self.conv_2(conv_1_out))))
		conv_3_out = self.bn_3(self.pool_3(self.relu(self.conv_3(conv_2_out))))
		conv_4_out = self.bn_4(self.relu(self.conv_4(conv_3_out)))
		conv_5_out = self.bn_5(self.relu(self.conv_5(conv_4_out)))
		flatten = self.flatten(conv_5_out)
		fcn_1_out = self.tanh(self.fcn_1(flatten))
		fcn_2_out = self.tanh(self.fcn_2(fcn_1_out))
		fcn_3_out = self.tanh(self.fcn_3(fcn_2_out))
		fcn_4_out = self.tanh(self.fcn_4(fcn_3_out))
		fcn_5_out = self.tanh(self.fcn_5(fcn_4_out))
		fcn_6_out = self.tanh(self.fcn_6(fcn_5_out))

		return self.sigmoid(fcn_6_out)


class MLP_base(nn.Module):

	def __init__(self, n_inputs, n_outputs)