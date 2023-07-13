import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import typing
from typing import List, Tuple, Union

# This Buffer class is used to store states, actions, and rewards for training the agent.
class Buffer:
    """
    This class acts as a memory buffer for the PPO agent.
    It stores state, action, reward, done and other values for each time step, 
    which are then used for updating the policy network.
    """
    def __init__(self):
        """
        Initializes the Buffer object with empty lists for each stored value.

        Args:
            None

        Returns:
            None
        """

        # These buffers store the agent's interactions with the environment.
        self.price_actions = []
        self.volume_actions = []
        self.states = []
        self.price_action_logprobs = []
        self.volume_action_logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []
    
    # This method clears the buffer
    def clear(self):
        """
        Clears all stored values from the buffer.

        Args:
            None

        Returns:
            None
        """

        self.price_actions = []
        self.volume_actions = []
        self.states = []
        self.price_action_logprobs = []
        self.volume_action_logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    # This method stores a new memory into the buffer
    def store_memory(self, state, price_action, volume_action, price_action_logprob, volume_action_logprob, state_value, reward, done):
        """
        Stores the given values in the buffer.

        Args:
            state (Tensor): The current state of the environment.
            price_action (Tensor): The price action chosen by the agent.
            volume_action (Tensor): The volume action chosen by the agent.
            price_action_logprob (Tensor): The log-probability of the chosen price action.
            volume_action_logprob (Tensor): The log-probability of the chosen volume action.
            state_value (Tensor): The predicted value of the current state.
            reward (Tensor): The reward received after taking the actions.
            done (bool): Whether the episode has ended.

        Returns:
            None
        """
                
        self.states.append(state)

        self.price_actions.append(price_action)
        self.volume_actions.append(volume_action)

        self.price_action_logprobs.append(price_action_logprob)
        self.volume_action_logprobs.append(volume_action_logprob)

        self.state_values.append(state_value)
        self.rewards.append(reward)
        self.dones.append(done)

# ActorCritic class defines the neural network architecture for the PPO agent
class ActorCritic(nn.Module):
    """
    This class defines the Actor-Critic model for the PPO agent.

    Args:
        state_dim (int): The dimension of the state space.
        price_action_dim (int): The dimension of the price action space.
        volume_action_dim (int): The dimension of the volume action space.

    Returns:
        None. This is a class for creating ActorCritic objects.
    """
    def __init__(self, state_dim, price_action_dim, volume_action_dim):
        """
        Initializes the ActorCritic object with an actor network and a critic network.

        Args:
            state_dim (int): The dimension of the state space.
            price_action_dim (int): The dimension of the price action space.
            volume_action_dim (int): The dimension of the volume action space.

        Returns:
            None
        """

        super(ActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                    )
        
        self.price_output = nn.Sequential(
                            nn.Linear(64, price_action_dim),
                            nn.Softmax(dim=-1)
                    )
        self.volume_output = nn.Sequential(
                        nn.Linear(64, volume_action_dim),
                        nn.Softmax(dim=-1)
                    )

        # Critic network
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                    )
        
    def forward(self, state):
        """
        Performs a forward pass through the actor and critic networks.

        Args:
            state (Tensor): The current state of the environment.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Returns the output distribution of the price and volume actions from the actor network and the state value from the critic network.
        """

        # Forward pass through actor and critic network
        actor = self.actor(state)
        price_output_dist = self.price_output(actor)
        volume_output_dist = self.volume_output(actor)

        price_output_dist = Categorical(price_output_dist)
        volume_output_dist = Categorical(volume_output_dist)

        state_value = self.critic(state)
        return price_output_dist, volume_output_dist, state_value

# PPOAgent class encapsulates the PPO learning algorithm
class PPOAgent:
    """
    The PPOAgent class creates an agent that interacts with the environment and 
    learns from it using the Proximal Policy Optimization algorithm. 
    """
    def __init__(self, state_dim, price_action_dim, volume_action_dim, lr_actor, lr_critic, gamma, n_epochs, eps_clip, device):
        """
        Initializes the PPOAgent object with an ActorCritic policy, 
        a memory Buffer and some necessary parameters.

        Args:
            state_dim (int): The dimension of the state space.
            price_action_dim (int): The dimension of the price action space.
            volume_action_dim (int): The dimension of the volume action space.
            lr_actor (float): The learning rate for the actor network.
            lr_critic (float): The learning rate for the critic network.
            gamma (float): The discount factor for future rewards.
            n_epochs (int): The number of epochs to train the agent.
            eps_clip (float): The clipping epsilon for the ratio r in PPO's objective function.
            device (str): The device (cpu or gpu) on which computations will be performed.

        Returns:
            None
        """         
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.n_epochs = n_epochs
        
        self.buffer = Buffer()

        # Initialize the policy and the optimizer
        self.policy = ActorCritic(state_dim, price_action_dim, volume_action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, price_action_dim, volume_action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    # Method to store experiences to the buffer
    def send_memory_to_buffer(self, state, price_action, volume_action,  price_action_logprob, volume_action_logprob, state_value, reward, done):
        """
        Stores the current transition into the buffer. The transition includes state, action, action log 
        probability, state value, reward, and the done flag.

        Args:
            state (torch.Tensor): The current state.
            price_action (torch.Tensor): The price action chosen by the agent.
            volume_action (torch.Tensor): The volume action chosen by the agent.
            price_action_logprob (torch.Tensor): The log probability of the price action.
            volume_action_logprob (torch.Tensor): The log probability of the volume action.
            state_value (torch.Tensor): The value of the current state estimated by the critic network.
            reward (float): The reward received after taking the action.
            done (bool): A flag indicating whether the episode is done.

        Returns:
            None
        """
        self.buffer.store_memory(state, price_action, volume_action,  price_action_logprob, volume_action_logprob, state_value, reward, done)

# 'select_action' method chooses an action given a state
def select_action(self, state):
    """
    Selects an action based on the given state.

    Args:
        state (Tensor): The current state of the environment.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Returns the selected price action, volume action, log-probability of the price action, log-probability of the volume action, and the state value, respectively.
    """


    # Use the old policy to get action probabilities and state-value
    price_dist, volume_dist, state_value = self.policy_old.forward(state)
    
    # Sample an action from the distribution and detach it from the graph
    price_action = price_dist.sample().detach()
    volume_action = volume_dist.sample().detach()
    
    # Get the log probability of the action 
    volume_action_logprob = volume_dist.log_prob(volume_action)
    price_action_logprob = price_dist.log_prob(price_action)

    return price_action, volume_action, price_action_logprob.detach(), volume_action_logprob.detach(), state_value.detach()


def update(self):
    """
    'update' method updates the policy - Performs the policy update using Proximal Policy Optimization (PPO).

    Args:
        None

    Returns:
        None. Updates the policy network parameters.
    """

    # Estimate of returns and advantage
    old_rewards = self.buffer.rewards
    old_dones = self.buffer.dones
    old_state_values = self.buffer.state_values
    
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

    advantages = torch.tensor(advantages).to(self.device)
    returns = torch.tensor(returns).to(self.device)
    old_states = torch.stack(self.buffer.states, dim=0).detach().to(self.device)

    old_price_actions = torch.stack(self.buffer.price_actions, dim=0).detach().to(self.device)
    old_volume_actions = torch.stack(self.buffer.volume_actions, dim=0).detach().to(self.device)

    old_price_action_logprobs = torch.stack(self.buffer.price_action_logprobs, dim=0).detach().to(self.device)
    old_volume_action_logprobs = torch.stack(self.buffer.volume_action_logprobs, dim=0).detach().to(self.device)

    # Creating batches from the buffers
    n_states = len(self.buffer.states)
    batch_start = np.arange(0, n_states, 64)
    indices = np.arange(n_states, dtype=np.int64)
    np.random.shuffle(indices)
    batches = [indices[i:i+64] for i in batch_start]

    # PPO Optimization step
    for _ in range(self.n_epochs):
        for batch in batches:
            price_dist, volume_dist, new_state_values = self.policy.forward(old_states[batch])

            new_price_action_logprobs = price_dist.log_prob(old_price_actions[batch])
            new_volume_action_logprobs = volume_dist.log_prob(old_volume_actions[batch])

            price_dist_entropy = price_dist.entropy()
            volume_dist_entropy = volume_dist.entropy()

            # Match new_state_values tensor dimensions with rewards tensor
            new_state_values = torch.squeeze(new_state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            price_ratios = torch.exp(new_price_action_logprobs - old_price_action_logprobs[batch].detach())
            volume_ratios = torch.exp(new_volume_action_logprobs - old_volume_action_logprobs[batch].detach())

            # Using the ppo formulas 
            price_ratio_advantage = price_ratios * advantages[batch]
            volume_ratio_advantage = volume_ratios * advantages[batch]

            clipped_price_ratio_advantage = torch.clamp(price_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[batch]
            clipped_volume_ratio_advantage = torch.clamp(volume_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages[batch]
            
            # Final loss of clipped objective PPO with entropy loss
            loss = -torch.min(price_ratio_advantage, clipped_price_ratio_advantage)  \
                   -torch.min(volume_ratio_advantage, clipped_volume_ratio_advantage) \
                    +  0.5 * self.MseLoss(returns[batch], new_state_values) - 0.009 * price_dist_entropy - 0.009 * volume_dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
    # Copy new weights into old policy
    self.policy_old.load_state_dict(self.policy.state_dict())

    # Clear buffer
    self.buffer.clear()

# 'save' method saves the model state
def save(self, checkpoint_path):
    """
    Saves the current model parameters.

    Args:
        Checkpoint_path (str): The path where the model parameters will be saved.

    Returns:
        None.
    """
    torch.save(self.policy_old.state_dict(), checkpoint_path)

# 'load' method loads the model state
def load(self, checkpoint_path):
    """
    Loads the model parameters from the given path.

    Args:
        checkpoint_path (str): The path from where the model parameters will be loaded.

    Returns:
        None. Updates the model parameters with the loaded parameters.
    """
    self.policy_old.load_state_dict(torch.load(checkpoint_path))
    self.policy.load_state_dict(torch.load(checkpoint_path))