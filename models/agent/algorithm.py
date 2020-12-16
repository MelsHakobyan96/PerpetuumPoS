import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from module import PPO_model


class PPO:
	def __init__(self, height, width, in_channels, out_channels, kernel_size, mlp_input_size, action_size, lr, betas, gamma, K_epochs, eps_clip, pre_trained=False, pre_trained_weights_path=None):
		self.lr = lr
		self.betas = betas
		self.gamma = gamma
		self.eps_clip = eps_clip
		self.K_epochs = K_epochs

		self.policy = PPO_model(height, width, in_channels, out_channels, kernel_size, mlp_input_size, action_size).to(device)
		if pre_trained_weights_path:
			self.policy.load_state_dict(torch.load(pre_trained_weights_path))
		else:
			pass

		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
		
		self.policy_old = PPO_model(height, width, in_channels, out_channels, kernel_size, mlp_input_size, action_size).to(device)
		self.policy_old.load_state_dict(self.policy.state_dict())
		
		self.MseLoss = nn.MSELoss()

	def _MC_calculator(self, reward_list):
		state_values = []
		for i, _ in enumerate(reward_list):
			state_values.append(reward_list[i] + self.gamma*np.sum(reward_list[i+1:]))
		  
		return state_values

	def select_action(self, image, meta_data, memory):
		return self.policy_old.act(image, meta_data, memory)

	def update(self, memory, path_write='/home/msi/ML_projects/PerpetuumPoS/saved_agent_models'):
		data_loader = DataLoader(memory, batch_size=8, shuffle=True)

		state_values = torch.tensor(memory.state_values).to(device)
		state_values = (state_values - state_values.mean()) / (state_values.std() + 1e-5)

		for k in range(K_epochs):
			for i, batch in enumerate(data_loader)
				old_actions, old_images, old_metadata, old_logprobs = batch

				logprobs, dist_entropy, state_values = self.policy.evaluate(old_states, old_actions)

				ratios = torch.exp(logprobs - old_logprobs.detach())

				advantages = rewards - state_values.detach()   
				surr1 = ratios * advantages
				surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
				loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
				
				self.optimizer.zero_grad()
				loss.mean().backward()
				self.optimizer.step()

		self.policy_old.load_state_dict(self.policy.state_dict())
		torch.save(self.policy.state_dict(), path_write)



