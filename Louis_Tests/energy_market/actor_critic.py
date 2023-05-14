import torch
from networks import ActorCritic
from torch import optim


class ActorCriticAgent:
    def __init__(self, input_size, n_actions, lr=0.0003, gamma=0.99):
        self.input_size = input_size
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(self.input_size, self.n_actions).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        _, probs = self.model(state)
        action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        # action = torch.full((24,), 3)
        self.action = action
        return action  # returns 24 actions, one for each hour

    def save_models(self):
        print('... saving models ...')
        torch.save(self.model.state_dict(), self.model.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.model.load_state_dict(torch.load(self.model.checkpoint_file))

    def update(self, state, reward, state_next, done):
        state = torch.from_numpy(state).float().to(self.device)
        state_next = torch.from_numpy(state_next).float().to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32)  # not fed to NN

        self.optimizer.zero_grad()

        state_value, probs = self.model(state)
        state_value_next, _ = self.model(state_next)

        action_probs = torch.distributions.Categorical(probs=probs)
        log_prob = action_probs.log_prob(self.action)

        delta = reward + self.gamma * state_value_next * (1 - int(done)) - state_value
        actor_loss = -log_prob * delta
        critic_loss = delta.pow(2)
        total_loss = actor_loss.mean() + critic_loss

        total_loss.backward()
        self.optimizer.step()




