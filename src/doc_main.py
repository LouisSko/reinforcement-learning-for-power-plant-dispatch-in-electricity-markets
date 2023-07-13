"""
This script trains a Proximal Policy Optimization (PPO) agent on a market environment. The market data is obtained
from a pickle file, the environment and agent are initialized, and then the agent is trained through a series of episodes.
Training results and agent performance are logged and saved.

This implementation is based on insights and code fragments from:
- https://arxiv.org/pdf/1707.06347.pdf
- https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl
- https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch
- https://github.com/nikhilbarhate99/PPO-PyTorch
"""

# Importing required modules
import os
import torch
import numpy as np
from import_data import get_data, read_processed_files, get_solar_actual, get_solar_estimate
from environment import market_env
from ppo_torch import PPOAgent
from torch.utils.tensorboard import SummaryWriter
import sys
import pandas as pd

# Set device to CPU
device = torch.device('cpu')

# Create directory for tensorboard logs if not exist
if not os.path.exists('runs'):
    os.makedirs('runs')

TRAIN = True  # Set to True for training, False for testing
if TRAIN:
    # Initialize Tensorboard writer
    tb = SummaryWriter()
    checkpoint_path = os.path.join('../.', 'models')
else:
    # Load pre-trained model for testing
    saved_model = os.path.join('../.', 'models/model_50000_episodes.pth')

if __name__ == '__main__':
    
    # Read the processed data files
    df_demand, df_demand_scaled, df_vre, df_vre_scaled, df_gen, df_gen_scaled, df_solar_cap_forecast, df_solar_cap_actual, df_mcp = read_processed_files()

    # Set bounds for reward rescaling
    lower_bound = -20000
    upper_bound = 20000

    # Initialize market environment
    env = market_env(demand=df_demand_scaled, re=df_vre_scaled, capacity_forecast=df_solar_cap_forecast,
                     capacity_actual=df_solar_cap_actual, prices=df_mcp, eps_length=24, capacity=200, mc=50,
                     lower_bound=lower_bound, upper_bound=upper_bound)

    # Set training parameters
    n_episodes = 200_000   # Maximum number of episodes for training
    update_timestep = 2048  # Update policy every 2048 steps
    n_epochs = 10  # Update policy for K epochs in one PPO update
    eps_clip = 0.22  # Clip parameter for PPO
    gamma = 0.99  # Discount factor
    lr_actor = 0.0002  # Learning rate for actor network
    lr_critic = 0.0008  # Learning rate for critic network

    # Get state and action space dimensions
    state_dim = env.observation_space.shape[0]
    price_action_dim = env.action_space[0].n
    volume_action_dim = env.action_space[1].n

    # Initialize PPO agent
    ppo_agent = PPOAgent(state_dim, price_action_dim, volume_action_dim, lr_actor, lr_critic, gamma, n_epochs, eps_clip, device)

    # Load pre-trained model if testing
    if not TRAIN:
        ppo_agent.load(saved_model)

    # Training/testing loop
    # Initialization
    time_step = 0
    i_episode = 0
    avg_rewards = []

    # Run until maximum number of episodes
    while i_episode <= n_episodes:
        state, _ = env.reset(TRAIN)
        current_ep_reward = 0
        done = False

        # Episode loop
        while not done:
            # Prepare state for agent
            state = torch.FloatTensor(state).to(device)
            if torch.isnan(state).any():
                state = torch.nan_to_num(state)

            # Agent selects action
            price_action, volume_action,  price_action_logprob, volume_action_logprob, state_val = ppo_agent.select_action(state)

            # Execute action in environment
            next_state, reward, done, _, info = env.step([price_action, volume_action], TRAIN)

            if TRAIN:
                # Store transition for training
                ppo_agent.send_memory_to_buffer(state, price_action, volume_action, price_action_logprob, volume_action_logprob, state_val, reward, done)

            state = next_state
            current_ep_reward += reward

            # Update Tensorboard
            if TRAIN:
                tb.add_scalars('Bid Capacity', {'bid': env.bid_volume_list[-1], 'cap': env.capacity_current_list[-1]}, global_step=time_step)

            time_step += 1
            if time_step % update_timestep == 0:
                if TRAIN:
                    # Perform PPO update
                    ppo_agent.update()
                    # Log results to Tensorboard
                    tb.add_scalar('Average Reward', np.mean(avg_rewards[-update_timestep:]), i_episode)
                    tb.add_scalar('Average Bid Price', np.mean(env.avg_bid_price[-update_timestep:]), i_episode)
                    tb.add_scalar('Average Profit', np.mean(env.profit_list[-update_timestep:]), i_episode)
                    tb.add_scalar('Average Profit Heuristic', np.mean(env.profit_heuristic_list[-update_timestep:]), i_episode)

                print(f'Episode {i_episode} out of {n_episodes}. Average Reward {np.mean(avg_rewards[-update_timestep:])}. Average Profit: {np.mean(env.profit_list[-update_timestep:])}')

        # Save model periodically during training
        if TRAIN and (i_episode == 195000 or i_episode == 100000 or i_episode == 50000):
            print("saving model ... ")
            save_path = os.path.join(checkpoint_path, 'model_{episode}_episodes_new.pth'.format(episode=i_episode))
            ppo_agent.save(save_path)
            print("model saved")

        i_episode += 1
        avg_rewards.append(current_ep_reward)

    # Close environment and Tensorboard writer
    env.close()
    tb.close()