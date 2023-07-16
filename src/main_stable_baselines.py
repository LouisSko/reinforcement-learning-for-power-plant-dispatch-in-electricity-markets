"""
This script trains and evaluates the A2C model, using the stable-baselines3 library.
The model's performance is logged using TensorBoard.
"""
from import_data import get_demand, get_gen, get_mcp, get_vre, read_processed_files
from environment import market_env
import gymnasium as gym
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

models_dir = "../models/A2C"
logdir = "runs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

TRAIN = True

if __name__ == '__main__':

    # Load processed files
    df_demand, df_demand_scaled, df_vre, df_vre_scaled, df_gen, df_gen_scaled, df_solar_cap_forecast, df_solar_cap_actual, df_mcp = read_processed_files()
    # set lower and upper bound for the rescaling to -1 and 1 of the rewards
    lower_bound = -20000
    upper_bound = 20000

    # initialize the market/gym environment
    env = market_env(demand=df_demand_scaled, re=df_vre_scaled, capacity_forecast=df_solar_cap_forecast,
                     capacity_actual=df_solar_cap_actual, prices=df_mcp, eps_length=24, capacity=200, mc=50,
                     lower_bound=lower_bound, upper_bound=upper_bound, train=TRAIN)

    # Perform environment checking
    check_env(env)

    # If in training mode, train the model
    if TRAIN:
        model = A2C("MlpPolicy", env, verbose=1, ent_coef=0.01, tensorboard_log=logdir)
        model.learn(total_timesteps=10000, tb_log_name="A2C")
        model.save("../models/A2C/ac2_test")
    else:
        # If not in training mode, load the pre-trained model and evaluate it
        avg_rewards = []
        model = A2C.load("../models/A2C/a2c_test")

        # Reset the environment
        obs, _ = env.reset()

        # Perform steps in the environment and store rewards     
        for i in range(100):
            action, _states = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            avg_rewards.append(reward)
        
        print(np.mean(avg_rewards))
