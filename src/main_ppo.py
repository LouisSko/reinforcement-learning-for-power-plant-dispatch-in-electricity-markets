"""
Note: Some basic understanding and Code fragments are inspired from 
- https://arxiv.org/pdf/1707.06347.pdf
- https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl
- https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch
- https://github.com/nikhilbarhate99/PPO-PyTorch 
"""

import torch
import numpy as np
from import_data import get_data, read_processed_files
from environment import market_env
from ppo_torch import PPOAgent
from torch.utils.tensorboard import SummaryWriter
import sys
import pandas as pd
import pickle
import os

# Since we don't work with large input and hidden layers (matrices like in CNN's) we rather recommend to use the cpu
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print("RUNNING ON ", device)


def rl_agent_run(hp_dict, device, train=True, model_checkpoint=None, tb_name='my_experiment'):
    # can choose between train and testing
    # setting train to true ignores model_checkpoint
    # tb name specifies the name for the summary writer

    # set hyperparameter
    lower_bound = hp_dict['lower_bound']
    upper_bound = hp_dict['upper_bound']
    batch_size = hp_dict['batch_size']
    n_episodes = hp_dict['n_episodes']
    update_timestep = hp_dict['update_timestep']
    n_epochs = hp_dict['n_epochs']
    eps_clip = hp_dict['eps_clip']
    gamma = hp_dict['gamma']
    lr_actor = hp_dict['lr_actor']
    lr_critic = hp_dict['lr_critic']

    # load data from the api (data is already downloaded in the data directory)
    # get_data()

    # read the downloaded data
    df_demand, df_demand_scaled, df_vre, df_vre_scaled, df_gen, df_gen_scaled, df_solar_cap_forecast, df_solar_cap_actual, df_mcp = read_processed_files()

    # initialize the market/gym environment
    env = market_env(demand=df_demand_scaled, re=df_vre_scaled, capacity_forecast=df_solar_cap_forecast,
                     capacity_actual=df_solar_cap_actual, prices=df_mcp, eps_length=24, capacity=200, mc=50,
                     lower_bound=lower_bound, upper_bound=upper_bound, train=True)

    # state and action space dimension
    state_dim = env.observation_space.shape[0]
    price_action_dim = env.action_space[0].n
    volume_action_dim = env.action_space[1].n

    # initialize the PPO agent
    ppo_agent = PPOAgent(state_dim, price_action_dim, volume_action_dim, lr_actor, lr_critic, gamma, n_epochs, eps_clip,
                         batch_size, device)
    if not train:
        ppo_agent.load(model_checkpoint)

    # specify paths for logging and model checkpoints
    if train:
        # create if folder for tensorboard logs is not created yet
        if not os.path.exists('runs'):
            os.makedirs('runs')

        # Create the log directory with the specific name
        log_dir = os.path.join('runs', tb_name)

        # Check if the path already exists, otherwise add a suffix
        if os.path.exists(log_dir):
            suffix = 2
            new_log_dir = log_dir
            while os.path.exists(new_log_dir):
                new_log_dir = f"{log_dir}{suffix}"
                suffix += 1
            log_dir = new_log_dir

        # init Tensorboard
        tb = SummaryWriter(log_dir)

        # Store the hp_dict in a pickle file to load it back in
        with open(os.path.join(log_dir, 'hp_dict.pkl'), 'wb') as f:
            pickle.dump(hp_dict, f)

        # Store the hp_dict in an easily readable text file for convenience
        with open(os.path.join(log_dir, 'hyperparameter.txt'), 'w') as f:
            for key, value in hp_dict.items():
                f.write(f"{key}: {value}\n")

        # specify checkpoint_path for the models
        checkpoint_path = os.path.join('../.', 'models')

    time_step = 0
    i_episode = 0
    avg_rewards = []

    # training / testing loop
    while i_episode <= n_episodes:

        state, _ = env.reset()
        current_ep_reward = 0
        done = False

        while not done:
            # convert state to Tensor
            state = torch.FloatTensor(state).to(device)
            if torch.isnan(state).any():
                state = torch.nan_to_num(state)

            # select action with policy
            price_action, volume_action, price_action_logprob, volume_action_logprob, state_val = ppo_agent.select_action(
                state)

            # perform a step in the market environment
            next_state, reward, done, _, info = env.step([price_action, volume_action])

            if train:
                # send the transition to the buffer
                ppo_agent.send_memory_to_buffer(state, price_action, volume_action, price_action_logprob,
                                                volume_action_logprob, state_val, reward, done)
                tb.add_scalars('Bid Capacity',
                               {'bid': env.bid_volume_list[-1], 'cap': env.capacity_current_list[-1]},
                               global_step=time_step)

            state = next_state
            current_ep_reward += reward
            time_step += 1

            # update PPO agent
            if time_step % update_timestep == 0:
                if train:
                    ppo_agent.update()
                    tb.add_scalar('Average Reward', np.mean(avg_rewards[-update_timestep:]), i_episode)
                    tb.add_scalar('Average Bid Price', np.mean(env.avg_bid_price[-update_timestep:]), i_episode)
                    tb.add_scalar('Average Profit', np.mean(env.profit_list[-update_timestep:]), i_episode)
                    tb.add_scalar('Average Profit Heuristic', np.mean(env.profit_heuristic_list[-update_timestep:]),
                                  i_episode)
                else:
                    df = pd.DataFrame(env.avg_bid_price)
                    print(df.value_counts())
                    sys.exit()

                print(
                    f'Episode {i_episode} out of {n_episodes}. Average Reward {np.mean(avg_rewards[-update_timestep:])}. Average Profit: {np.mean(env.profit_list[-update_timestep:])}')

        if train and (i_episode == 195000 or i_episode == 100000 or i_episode == 50000):
            print("saving model ... ")
            save_path = os.path.join(checkpoint_path, 'model_{episode}_episodes_new.pth'.format(episode=i_episode))
            ppo_agent.save(save_path)
            print("model saved")

        i_episode += 1
        avg_rewards.append(current_ep_reward)

    if train:
        tb.close()
    env.close()


if __name__ == '__main__':

    # most hyperparameters are chosen based on the default of stable baselines 3
    hp_dict = {'lower_bound': -20000,  # lower bound for the reward scaling
               'upper_bound': 20000,  # upper bound for the reward scaling
               'batch_size': 16,  # define batch size
               'n_episodes': 50000,  # number of episodes to train
               'update_timestep': 2048,  # update policy every 2048 steps
               'n_epochs': 10,  # update policy for K epochs in one PPO update
               'eps_clip': 0.22,  # clip  parameter for PPO
               'gamma': 0.99,  # discount factor
               'lr_actor': 0.0002,  # learning rate for actor network
               'lr_critic': 0.0008  # learning rate for critic network
               }

    model_checkpoint = os.path.join('../.', 'models/model_50000_episodes.pth')

    rl_agent_run(hp_dict, device, train=True, model_checkpoint=None, tb_name='my_experiments')

