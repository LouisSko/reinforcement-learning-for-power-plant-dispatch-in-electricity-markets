"""
Note: Some basic understanding and Code fragments are inspired from 
- https://arxiv.org/pdf/1707.06347.pdf
- https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl
- https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch
- https://github.com/nikhilbarhate99/PPO-PyTorch 
"""

import torch
from itertools import product
import copy
from main_ppo import rl_agent_run

# Since we don't work with large input and hidden layers (matrices like in CNN's) we rather recommend to use the cpu
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print("RUNNING ON ", device)

def hp_tuning():

    # define search space
    param_grid = {'lower_bound': [-20000],  # lower bound for the reward scaling
                  'upper_bound': [20000],  # upper bound for the reward scaling
                  'batch_size': [32, 64, 128],  # define batch size
                  'n_episodes': [50000],  # number of episodes to train
                  'update_timestep': [512, 1024, 2048],  # update policy every 2048 steps
                  'n_epochs': [10],  # update policy for K epochs in one PPO update
                  'eps_clip': [0.22],  # clip  parameter for PPO
                  'gamma': [0.99],  # discount factor
                  'lr_actor': [0.0002],  # learning rate for actor network
                  'lr_critic': [0.0008]  # learning rate for critic network
                  }

    # Generate all parameter combinations
    param_combinations = list(product(*param_grid.values()))

    # Store each parameter combination in separate dictionaries
    param_dicts = []
    for params in param_combinations:
        param_dict = copy.deepcopy(param_grid)  # Create a copy of hp_dict
        param_dict.update(dict(zip(param_grid.keys(), params)))  # Update the copy with the parameter values
        param_dicts.append(param_dict)

    # train a model for each hp combination. All results are logged in tensorboard
    for hp_dict in param_dicts:
        rl_agent_run(hp_dict, device, train=True, model_checkpoint=None, tb_name='hp_tuning')


if __name__ == '__main__':
    hp_tuning()
