from actor_critic import ActorCriticAgent
import gym
import numpy as np
from utils import plot_learning_curve

env = gym.make('CartPole-v0')
# Example usage
# Assuming you have a gym environment called 'env'
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
n_games = 100

agent = ActorCriticAgent(input_size, output_size)
agent.train(env, episodes=100)

filename = 'cartpole_1e-5_1024x512_1800games.png'

figure_file = 'plots/' + filename

best_score = env.reward_range[0]
score_history = []
load_checkpoint = False

if load_checkpoint:
    agent.load_models()

for i in range(n_games):

    observation = env.reset()
    done = False
    score = 0

    action = agent.choose_action(observation)
    observation_, reward, = env.step(action)
    score += reward

    if not load_checkpoint:
        agent.update(observation, reward, observation_, done)
    observation = observation_

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_models()

    print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

if not load_checkpoint:
    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)