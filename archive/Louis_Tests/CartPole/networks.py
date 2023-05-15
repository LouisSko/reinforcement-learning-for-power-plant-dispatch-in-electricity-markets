import torch.nn as nn
import os


class ActorCritic(nn.Module):
    def __init__(self, input_size, n_actions, n_neurons1=2048, n_neurons2=1024, name='actor_critic',
                 chkpt_dir='/Users/louis.skowronek/bda-case-challenge/Louis_Tests/checkpoints'):
        super().__init__()
        self.input_size = input_size
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ac')

        self.fc1 = nn.Linear(input_size, n_neurons1)
        self.fc2 = nn.Linear(n_neurons1, n_neurons2)
        self.fc_critic = nn.Linear(n_neurons2, 1)
        self.fc_actor = nn.Linear(n_neurons2, n_actions)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        value = self.relu(self.fc1(state))
        value = self.relu(self.fc2(value))

        v = self.fc_critic(value)
        pi = self.softmax(self.fc_actor(value))

        return v, pi
