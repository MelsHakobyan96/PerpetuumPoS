from models.reward.module import RewardPredictor
from models import reward
from models.reward.predict import predict
from utils.functions import *
from models.agent.algorithm import PPO
from data_utils.PPO_Memory import PPO_Memory
import sim
import config
from sim import DrivingStyle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    keys = read_txt()
    mlp_input_size = 14
    action_size = 2
    agent_lr = 3e-4
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 10
    eps_clip = 0.2
    reward_pred_lr = 3e-4
    random_seed = None
    pre_trained_agent_path = None
    reward_pred_batch_size = 32
    reward_update_episode_count = 10
    agent_update_episode_count = 6
    give_rewards_episode_count = 2
    total_episode_count = 100

    reward_model_save_path = "./logs/reward/experiment_1.pth"
    agent_model_save_path = "./logs/agent/experiment_1.pth"

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

    reward_predictor = RewardPredictor(
        cnn_input=example_cnn_inp, mlp_input=example_mlp_inp)
    memory = PPO_Memory()
    agent = PPO(action_size=action_size, lr=agent_lr, betas=betas, gamma=gamma,
                K_epochs=K_epochs, eps_clip=eps_clip, pre_trained_weights_path=pre_trained_agent_path)

    episodic_rewards = []
    done = False
    episode_count = 0

    # maybe run UI in async to constantly show videos of driving

    for i in range(total_episode_count):
        episode_images = list()
        episode_meta_data = list()

        state = env.reset()

        image, meta_data = preprocess(state, keys)

        while not done:
            action = agent.select_action(image, meta_data, memory)
            state, env_reward, done, _ = env.step(action)

            image, meta_data = preprocess(state, keys)
            episode_images.append(image)
            episode_meta_data.append(meta_data)
            memory.is_terminal.append(done)

        if done:
            buffer_data = episode_meta_data.copy()
            buffer_data['images'] = episode_images

            file_name = 'episode_' + str(i)
            save_json(buffer_data, './data/reward/')

    memory.images.extend(episode_images)
    memory.meta_data.extend(episode_meta_data)

    episodic_rewards = []

    if episode_count % give_rewards_episode_count == 0:
        # get rewards for the episode_images
        rd_predict = get_memory_episodes(memory)

        output = reward.predict.predict(
            rd_predict, model=reward_predictor, device=device)

        episodic_rewards.extend(output)

        # TODO:
        # give episodic_rewards to MC_calculator and append the result to memory as state_values

    if episode_count % agent_update_episode_count == 0:
        agent.update(memory, path_write=agent_model_save_path)
        memory.clear_memory()

    if episode_count % reward_update_episode_count == 0:
        # train the reward predictor
        rd_train = random_data()
        reward.train.train(data=rd_train, batch_size=reward_pred_batch_size,
                           device=None, lr=3e-4, save=False, path=reward_model_save_path)

    episode_count += 1
