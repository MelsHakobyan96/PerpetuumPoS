import torch
import torch.nn as nn

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
		self.fcn_1 = nn.Linear(7776, 1024)
		self.fcn_2 = nn.Linear(1024, 512)
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

		return self.tanh(fcn_6_out)


class MLP_base(nn.Module):

	def __init__(self, n_inputs):
		super(MLP_base, self).__init__()

		self.tanh = nn.Tanh()

		self.layer1 = nn.Linear(n_inputs, 256)
		self.layer2 = nn.Linear(256, 256)
		self.layer3 = nn.Linear(256, 128)
		self.layer4 = nn.Linear(128, 128)
		self.layer5 = nn.Linear(128, 128)

	def forward(self, x):

		layer1_out = self.layer1(self.tanh(x))
		layer2_out = self.layer2(self.tanh(layer1_out))
		layer3_out = self.layer3(self.tanh(layer2_out))
		layer4_out = self.layer4(self.tanh(layer3_out))
		layer5_out = self.layer5(self.tanh(layer4_out))

		return layer5_out

class PPO_model(nn.Module):

	def __init__(self, height, width, in_channels, out_channels, kernel_size, mlp_input_size, action_size):
		super(PPO_model, self).__init__()

		self.CNN = CNN_base(height, width, in_channels, out_channels, kernel_size)
		self.MLP = MLP_base(mlp_input_size)

		self.tanh = nn.Tanh()
		self.softplus = nn.Softplus()

		self.layer1 = nn.Linear(256, 256)
		self.layer2 = nn.Linear(256, 256)
		self.layer3 = nn.Linear(256, 128)
		self.layer4 = nn.Linear(128, 128)
		self.layer5 = nn.Linear(128, 128)
		self.layer6 = nn.Linear(128, 64)
		self.layer7 = nn.Linear(64, 64)
		self.layer8 = nn.Linear(64, 64)
		self.layer9 = nn.Linear(64, 64)
		self.layer10 = nn.Linear(64, 64)

		self.policy_layer1 = nn.Linear(64, 64)
		self.policy_layer2 = nn.Linear(64, 64)
		self.policy_layer3 = nn.Linear(64, 64)
		self.policy_layer4 = nn.Linear(64, 32)
		self.policy_mean = nn.Linear(32, action_size)
		self.policy_var = nn.Linear(32, action_size)

		self.value_layer1 = nn.Linear(64, 64)
		self.value_layer2 = nn.Linear(64, 64)
		self.value_layer3 = nn.Linear(64, 32)
		self.value_layer4 = nn.Linear(32, 16)
		self.value_layer5 = nn.Linear(16, 1)


	def forward(self, image, meta_data):

		CNN_out = self.CNN(image)
		MLP_out = self.MLP(meta_data)

		PPO_input = torch.cat((CNN_out, MLP_out)).view(image.size()[0], 1, -1)

		layer1_out = self.layer1(self.tanh(PPO_input))
		layer2_out = self.layer2(self.tanh(layer1_out))
		layer3_out = self.layer3(self.tanh(layer2_out))
		layer4_out = self.layer4(self.tanh(layer3_out))
		layer5_out = self.layer5(self.tanh(layer4_out))
		layer6_out = self.layer6(self.tanh(layer5_out))
		layer7_out = self.layer7(self.tanh(layer6_out))
		layer8_out = self.layer8(self.tanh(layer7_out))
		layer9_out = self.layer9(self.tanh(layer8_out))
		layer10_out = self.layer10(self.tanh(layer9_out))

		policy_layer1_out = self.policy_layer1(self.tanh(layer10_out))
		policy_layer2_out = self.policy_layer2(self.tanh(policy_layer1_out))
		policy_layer3_out = self.policy_layer3(self.tanh(policy_layer2_out))
		policy_layer4_out = self.policy_layer4(self.tanh(policy_layer3_out))

		policy_mean_out = self.policy_mean(self.tanh(policy_layer4_out))
		policy_var_out = self.policy_var(self.softplus(policy_layer4_out))

		value_layer1_out = self.value_layer1(self.tanh(layer10_out))
		value_layer2_out = self.value_layer2(self.tanh(value_layer1_out))
		value_layer3_out = self.value_layer3(self.tanh(value_layer2_out))
		value_layer4_out = self.value_layer4(self.tanh(value_layer3_out))
		value_layer5_out = self.value_layer5(value_layer4_out)

		return (policy_mean_out, policy_var_out), value_layer5_out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PPO_model(height=227, width=227, in_channels=3, out_channels=6, kernel_size=4, mlp_input_size=8, action_size=2).to(device)

image = torch.zeros((8, 3, 227, 227)).to(device)
meta = torch.zeros((8, 8)).to(device)

print(model.forward(image, meta))