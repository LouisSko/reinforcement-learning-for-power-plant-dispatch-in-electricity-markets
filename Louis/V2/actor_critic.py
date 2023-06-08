import torch
from networks import ActorCritic
from torch import optim
import numpy as np


class ActorCriticAgent:
    def __init__(self, input_size, n_actions_prc, n_actions_vol, lr=0.00001, gamma=0.99):
        self.input_size = input_size
        self.n_actions_prc = n_actions_prc
        self.n_actions_vol = n_actions_vol
        self.lr = lr
        self.gamma = gamma
        self.action = None
        self.action_space_prc = [i for i in range(self.n_actions_prc)]
        self.action_actions_vol = [i for i in range(self.n_actions_vol)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(self.input_size, self.n_actions_prc, self.n_actions_vol).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        _, probs_prc, probs_vol = self.model(state)
        action_prc_dist = torch.distributions.Categorical(probs=probs_prc)
        action_vol_dist = torch.distributions.Categorical(probs=probs_vol)
        action_prc = action_prc_dist.sample()
        action_vol = action_vol_dist.sample()

        #self.action = torch.cat([action_prc, action_vol])

        return action_prc, action_vol

    def save_models(self):
        print('... saving models ...')
        torch.save(self.model.state_dict(), self.model.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.model.load_state_dict(torch.load(self.model.checkpoint_file))

    # mehrere actions
    def update(self, state, reward, state_next, done):
        state = torch.from_numpy(state).float().to(self.device)
        state_next = torch.from_numpy(state_next).float().to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32)  # not fed to NN

        self.optimizer.zero_grad()

        state_value, probs_prc, probs_vol = self.model(state)
        state_value_next, _, _ = self.model(state_next)

        action_prc_probs = torch.distributions.Categorical(probs=probs_prc)
        action_vol_probs = torch.distributions.Categorical(probs=probs_vol)

        log_prob_prc = action_prc_probs.log_prob(self.action[0])
        log_prob_vol = action_vol_probs.log_prob(self.action[1])

        delta = reward + self.gamma * state_value_next * (1 - int(done)) - state_value
        actor_loss = -(log_prob_prc + log_prob_vol) * delta
        critic_loss = delta.pow(2)
        total_loss = actor_loss.mean() + critic_loss.mean()

        total_loss.backward()
        self.optimizer.step()
