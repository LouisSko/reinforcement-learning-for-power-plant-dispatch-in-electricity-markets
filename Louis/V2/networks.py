import torch.nn as nn
import os
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, input_size, n_actions_prc, n_actions_vol, n_neurons1=256, n_neurons2=128, name='actor_critic',
                 chkpt_dir=os.path.join(os.getcwd(), '../checkpoints')):
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)

        super().__init__()
        self.input_size = input_size
        self.n_actions_prc = n_actions_prc
        self.n_actions_vol = n_actions_vol
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ac')

        self.fc1 = nn.Linear(input_size, n_neurons1)
        self.fc2 = nn.Linear(n_neurons1, n_neurons2)
        self.fc_critic = nn.Linear(n_neurons2, 1)
        self.fc_actor_prc = nn.Linear(n_neurons2, self.n_actions_prc)  # outputs n_actions for price
        self.fc_actor_vol = nn.Linear(n_neurons2, self.n_actions_vol)  # outputs n_actions for volume

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        value = self.relu(self.fc1(state))
        value = self.relu(self.fc2(value))

        v = self.fc_critic(value)
        pi_prc = self.fc_actor_prc(value)
        pi_prc = self.softmax(pi_prc)
        pi_vol = self.fc_actor_vol(value)
        pi_vol = self.softmax(pi_vol)

        return v, pi_prc, pi_vol
