from models.reward.dataset import RewardDataset
from models.reward.module import RewardPredictor
from models.reward.train import train
from models.reward.predict import predict
from utils.preprocess import *
from models.buffer.select import random_data
from models.agent.algorithm import PPO
from data_utils.PPO_Memory import PPO_Memory
import sim
import config
from sim import DrivingStyle

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
	random_seed=None
	pre_trained_agent_path=None
	reward_pred_batch_size=32
	reward_update_episode_count=10
	agent_update_episode_count=6
	total_episode_count=100

	reward_model_save_path="./logs/reward/experiment_1.pth"
	agent_model_save_path="./logs/agent/experiment_1.pth"

	env = sim.start(is_sync=True,
					render=False,
					enable_traffic=True,
					experiment='my_experiment',
					cameras=[config.DEFAULT_CAM],
					# fps=config.DEFAULT_FPS,  # Agent steps per second
					sim_step_time=config.DEFAULT_SIM_STEP_TIME,
					is_discrete=False,  # Discretizes the action space
					driving_style=DrivingStyle.NORMAL,
					is_remote_client=False,
					max_steps=500,
					max_episodes=100,
					should_record=False,  # HDF5/numpy recordings
					recording_dir=config.RECORDING_DIR,
					randomize_view_mode=False,
					view_mode_period=None,  # Domain randomization
					randomize_sun_speed=False,
					randomize_shadow_level=False,
					randomize_month=False,
					map='kevindale_bare',
					scenario_index=2)

	if random_seed:
		print("Random Seed: {}".format(random_seed))
		torch.manual_seed(random_seed)
		env.seed(random_seed)
		np.random.seed(random_seed)

	reward_predictor = RewardPredictor(cnn_input=example_cnn_inp, mlp_input=example_mlp_inp)
	memory = PPO_Memory()
	agent = PPO(action_size=action_size, lr=agent_lr, betas=betas, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip, pre_trained_weights_path=pre_trained_agent_path)

	episodic_rewards = []
	done = False
	episode_count=0

	# maybe run UI in async to constantly show videos of driving

	for i in range(total_episode_count):
		state = env.reset()
		
		###########################################
		# preprocess(state)      #
		# returns flattened dictionary #
		# takes the image and meta_data     #
		###########################################
		while not done:
			action = agent.select_action(image, meta_data, memory)
			state, env_reward, done, _ = env.step(action)
			#####################################################
			# state.preprocess()      #
			# returns flattened dictionary #
			# takes the image and meta_data     #
			# we should define where an episode starts and ends #
			#####################################################

			# reward_pred model predicts rewards in batches, so the full episode should be runned sequentially
			# and give the appropriate reward for every batch, append values in episodic_rewards list and given to MC_calculator and append to memory as state_values

			# add images and meta_data to buffer for later training of reward_predictor

			memory.is_terminal.append(done)

		episodic_rewards = []

		if episode_count%agent_update_episode_count == 0:
			agent.update(memory, path_write=agent_model_save_path)
			memory.clear_memory()

		if episode_count%reward_update_episode_count == 0:
			# train_db needs to be in the right format from buffer
			train(data=train_db, batch_size=reward_pred_batch_size, device=None, lr=3e-4, save=False, path=reward_model_save_path)

		episode_count+=1

			



