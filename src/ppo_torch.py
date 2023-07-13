import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import typing
from typing import List, Tuple, Union


class Buffer:
    def __init__(self):
        self.price_actions = []
        self.volume_actions = []
        self.states = []
        self.price_action_logprobs = []
        self.volume_action_logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []
    
    def clear(self):
        self.price_actions = []
        self.volume_actions = []
        self.states = []
        self.price_action_logprobs = []
        self.volume_action_logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def store_memory(self, state, price_action, volume_action, price_action_logprob, volume_action_logprob, state_value, reward, done):
        self.states.append(state)

        self.price_actions.append(price_action)
        self.volume_actions.append(volume_action)

        self.price_action_logprobs.append(price_action_logprob)
        self.volume_action_logprobs.append(volume_action_logprob)

        self.state_values.append(state_value)
        self.rewards.append(reward)
        self.dones.append(done)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, price_action_dim, volume_action_dim):
        super(ActorCritic, self).__init__()

        # note: we decided to use a ReLU() activation function for both networks other than proposed in some papers (they use tanh())
        # actor network
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        #nn.Linear(64, action_dim),
                        #nn.Softmax(dim=-1)
                    )
        
        self.price_output = nn.Sequential(
                            nn.Linear(64, price_action_dim),
                            nn.Softmax(dim=-1)
                    )
        self.volume_output = nn.Sequential(
                        nn.Linear(64, volume_action_dim),
                        nn.Softmax(dim=-1)
                    )
        # critic network
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                    )
        
    def forward(self, state):
        actor = self.actor(state)
        price_output_dist = self.price_output(actor)
        volume_output_dist = self.volume_output(actor)


        price_output_dist = Categorical(price_output_dist)
        volume_output_dist = Categorical(volume_output_dist)

        state_value = self.critic(state)
        return price_output_dist, volume_output_dist, state_value

class PPOAgent:
    def __init__(self, state_dim, price_action_dim, volume_action_dim, lr_actor, lr_critic, gamma, n_epochs, eps_clip, batch_size, device):
         
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.buffer = Buffer()

        self.policy = ActorCritic(state_dim, price_action_dim, volume_action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, price_action_dim, volume_action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def send_memory_to_buffer(self, state, price_action, volume_action,  price_action_logprob, volume_action_logprob, state_value, reward, done):
        # store transistions into the buffer
        self.buffer.store_memory(state, price_action, volume_action,  price_action_logprob, volume_action_logprob, state_value, reward, done)

    def select_action(self, state):
        with torch.no_grad():
            # forward pass
            price_dist, volume_dist, state_value = self.policy_old.forward(state)
            # sample from action distribution (sample because we want to increase exploration)

            price_action = price_dist.sample().detach()
            volume_action = volume_dist.sample().detach()
            # get the log probability of the action 
            volume_action_logprob = volume_dist.log_prob(volume_action)
            price_action_logprob = price_dist.log_prob(price_action)

        return price_action, volume_action, price_action_logprob.detach(), volume_action_logprob.detach(), state_value.detach()

    def update(self):

        # extract from the buffer
        old_rewards = self.buffer.rewards
        old_dones = self.buffer.dones
        old_state_values = self.buffer.state_values

        # calculate advantages and returns using the formula from the ppo algorithm
        advantages = []
        returns = []
        next_state_value = 0
        advantage = 0
        for reward, done, state_value in zip(reversed(old_rewards), reversed(old_dones), reversed(old_state_values)):
            delta = reward + (self.gamma * next_state_value * (int(done)) - state_value)
            advantage = delta + self.gamma * 0.95 * (int(done)) * advantage
            next_state_value = state_value
            advantages.insert(0, advantage)
            returns.insert(0, advantage + state_value)
            
        # convert advantages and returns to tensor
        advantages = torch.tensor(advantages).to(self.device)
        returns = torch.tensor(returns).to(self.device)

        # convert list of tensors (tensor states) to one tensor
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(self.device)

        old_price_actions = torch.stack(self.buffer.price_actions, dim=0).detach().to(self.device)
        old_volume_actions = torch.stack(self.buffer.volume_actions, dim=0).detach().to(self.device)

        old_price_action_logprobs = torch.stack(self.buffer.price_action_logprobs, dim=0).detach().to(self.device)
        old_volume_action_logprobs = torch.stack(self.buffer.volume_action_logprobs, dim=0).detach().to(self.device)

        # creating the batches: we generate arrays with random indices. They are used later to match the states, actions etc. 
        n_states = len(self.buffer.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        # for K epochs
        for _ in range(self.n_epochs):
            
            # iterate over all batches from the buffer and optimize the policy
            for batch in batches:
                
                price_dist, volume_dist, new_state_values = self.policy.forward(old_states[batch])

                new_price_action_logprobs = price_dist.log_prob(old_price_actions[batch])
                new_volume_action_logprobs = volume_dist.log_prob(old_volume_actions[batch])

                price_dist_entropy = price_dist.entropy()
                volume_dist_entropy = volume_dist.entropy()

                # match new_state_values tensor dimensions with rewards tensor
                new_state_values = torch.squeeze(new_state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                price_ratios = torch.exp(new_price_action_logprobs - old_price_action_logprobs[batch].detach())
                volume_ratios = torch.exp(new_volume_action_logprobs - old_volume_action_logprobs[batch].detach())

                # using the ppo formulas 
                price_ratio_advantage = price_ratios * advantages[batch]
                volume_ratio_advantage = volume_ratios * advantages[batch]

                clipped_price_ratio_advantage = torch.clamp(price_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[batch]
                clipped_volume_ratio_advantage = torch.clamp(volume_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[batch]
                
                # final loss of clipped objective PPO with entropy loss
                loss = -torch.min(price_ratio_advantage, clipped_price_ratio_advantage)  \
                       -torch.min(volume_ratio_advantage, clipped_volume_ratio_advantage) \
                        +  0.5 * self.MseLoss(returns[batch], new_state_values) - 0.009 * price_dist_entropy - 0.009 * volume_dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path))
        self.policy.load_state_dict(torch.load(checkpoint_path))


