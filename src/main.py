
"""
Note: Some basic understanding and Code fragments are inspired from 
- https://arxiv.org/pdf/1707.06347.pdf
- https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl
- https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch
- https://github.com/nikhilbarhate99/PPO-PyTorch 
"""

import os
from datetime import datetime
import torch
import numpy as np
from import_data import get_data, read_processed_files, get_solar_actual, get_solar_estimate
from environment import market_env
from ppo_torch import PPOAgent
from torch.utils.tensorboard import SummaryWriter
import sys
import pandas as pd


# Since we don't work with large input and hidden layers (matrices like in CNN's) we rather recommend to use the cpu
"""
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to: " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to: cpu")
"""

device = torch.device('cpu')

# create if folder for tensorboard logs if not created yet
if not os.path.exists('runs'):
    os.makedirs('runs')

TRAIN = True
if TRAIN:
    # init Tensorboard
    tb = SummaryWriter()
    checkpoint_path = os.path.join('../.', 'models')
else:
    saved_model = os.path.join('../.', 'models/model_50000_episodes.pth')


if __name__ == '__main__':
    
    #get_data()
    # get the files from the API
    df_demand, df_demand_scaled, df_vre, df_vre_scaled, df_gen, df_gen_scaled, df_solar_cap_forecast, df_solar_cap_actual, df_mcp = read_processed_files()

    # set lower and upper bound for the rescaling to -1 and 1 of the rewards
    lower_bound = -10000
    upper_bound = 10000

    # initialize the market/gym environment
    env = market_env(demand=df_demand_scaled, re=df_vre_scaled, capacity_forecast=df_solar_cap_forecast,
                     capacity_actual=df_solar_cap_actual, prices=df_mcp, eps_length=24, capacity=200, mc=50,
                     lower_bound=lower_bound, upper_bound=upper_bound)
    
    
    n_episodes = 50000   # break training loop if i_episodes > n_episodes

    # hyperparameters 
    # most hyperparameters are chosen based on the default of stable baselines 3
    update_timestep = 2048  # update policy every 2048 steps
    n_epochs = 10           # update policy for K epochs in one PPO update
    eps_clip = 0.22         # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0002       # learning rate for actor network
    lr_critic = 0.0008      # learning rate for critic network

    # state space dimension
    state_dim = env.observation_space.shape[0]
    # action space dimension
    price_action_dim = env.action_space[0].n
    volume_action_dim = env.action_space[1].n


    # initialize the PPO agent
    ppo_agent = PPOAgent(state_dim, price_action_dim, volume_action_dim, lr_actor, lr_critic, gamma, n_epochs, eps_clip, device)

    if not TRAIN:
        ppo_agent.load(saved_model)

    time_step = 0
    i_episode = 0
    avg_rewards = []


    # training / testing loop
    while i_episode <= n_episodes:

        state, _ = env.reset(TRAIN)
        current_ep_reward = 0
        done = False
        
        while not done:
            # convert state to Tensor 
            state = torch.FloatTensor(state).to(device)
            if torch.isnan(state).any():
                state = torch.nan_to_num(state)
            

            # select action with policy
            price_action, volume_action,  price_action_logprob, volume_action_logprob, state_val = ppo_agent.select_action(state)

            # perform a step in the market environment
            next_state, reward, done, _, info = env.step([price_action, volume_action], TRAIN)


            if TRAIN:
                # send the transition to the buffer
                ppo_agent.send_memory_to_buffer(state, price_action, volume_action, price_action_logprob, volume_action_logprob, state_val, reward, done)

            state = next_state
            current_ep_reward += reward

            if TRAIN:
                tb.add_scalars('Bid Capacity', {'bid': env.bid_volume_list[-1], 'cap': env.capacity_current_list[-1]}, global_step=time_step)

            time_step += 1
            # update PPO agent
            if time_step % update_timestep == 0:
                if TRAIN:
                    ppo_agent.update()
                    tb.add_scalar('Average Reward', np.mean(avg_rewards[-update_timestep:]), i_episode)
                    tb.add_scalar('Average Bid Price', np.mean(env.avg_bid_price[-update_timestep:]), i_episode)
                    tb.add_scalar('Average Profit', np.mean(env.profit_list[-update_timestep:]), i_episode)
                else:
                    df = pd.DataFrame(env.avg_bid_price)
                    print(df.value_counts())
                    sys.exit()

                print(f'Episode {i_episode} out of {n_episodes}. Average Reward {np.mean(avg_rewards[-update_timestep:])}. Average Profit: {np.mean(env.profit_list[-update_timestep:])}')

        if TRAIN and (i_episode == 195000 or i_episode == 100000 or i_episode == 50000):
            print("saving model ... ")
            save_path = os.path.join(checkpoint_path, 'model_{episode}_episodes_new.pth'.format(episode=i_episode))
            ppo_agent.save(save_path)
            print("model saved")

        i_episode += 1
        avg_rewards.append(current_ep_reward)

    env.close()
    tb.close()
