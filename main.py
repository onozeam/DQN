#!/usr/bin/env python3
from collections import namedtuple
import gym
from itertools import count
import math
# import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn


class DQN(nn.Module):
    """ ref: https://gym.openai.com/evaluations/eval_onwKGm96QkO9tJwdX7L0Gw/ """
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )
        self.layer3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


Transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state", "done"])


class Agent:
    def __init__(self, n_observations, n_actions):
        self.model = DQN(n_observations, n_actions, n_observations*2)
        self.optim = torch.optim.Adam(self.model.parameters())
        self.loss_fn = torch.nn.MSELoss()
        self.n_actions = n_actions

    def pred_q(self, states):
        """ get predicted Q value """
        self.model.eval()
        states = torch.autograd.Variable(torch.tensor(states, dtype=torch.float))
        pred = self.model(states)
        return pred

    def label_q(self, actions, rewards, dones, next_states, pred_q, gamma=0.99):
        """ get label Q value """
        self.model.eval()
        label = pred_q.clone().data.numpy()
        label[np.arange(label.shape[0]), actions] = rewards + gamma * np.max(self.pred_q(next_states).data.numpy(), axis=1) * ~dones
        return torch.autograd.Variable(torch.tensor(label, dtype=torch.float))

    def e_greedy_action(self, states, current_step, eps_start=0.9, eps_end=0.05, eps_decay=200):
        """
        select action by epsilon greedy policy.
        Args:
            states: 1-dim ndarray
            current_step: int
        """
        threshold = eps_end + (eps_start-eps_end) * math.exp(-1.*current_step/eps_decay)
        if random.random() > threshold:
            return self.greedy_action(states)
        return random.randrange(self.n_actions)

    def greedy_action(self, states):
        self.model.eval()
        score = self.model(torch.tensor(states, dtype=torch.float).unsqueeze(0))
        _, argmax_index = score.max(1)
        return argmax_index.item()

    def train(self, memory):
        states = np.asarray([m.state for m in memory])
        actions = np.asarray([m.action for m in memory])
        rewards = np.asarray([m.reward for m in memory])
        dones = np.asarray([m.done for m in memory])
        next_states = np.asarray([m.next_state for m in memory])
        pred_q = self.pred_q(states)
        label_q = self.label_q(actions, rewards, dones, next_states, pred_q)
        self.model.train()
        self.optim.zero_grad()
        loss = self.loss_fn(pred_q, label_q)
        loss.backward()
        self.optim.step()
        return loss


def main():
    memory = []  # List[Transition]
    env = gym.make('CartPole-v0')
    n_observations, n_actions, n_episodes = env.observation_space.shape[0], env.action_space.n, 2000
    model_file_name, episode_file_name = 'model_state', 'episode.pkl'
    agent = Agent(n_observations, n_actions)
    batch_size = 64
    if os.path.exists(model_file_name):
        print('loading model state...')
        agent.model.load_state_dict(torch.load(model_file_name))
        print('done.')
    if os.path.exists(episode_file_name):
        print('loading previous episode...')
        with open(episode_file_name, 'rb') as f:
            prev_eps = pickle.load(f)
        print('done.')
    else:
        prev_eps = 0
    for eps in range(prev_eps, prev_eps+n_episodes):
        state = env.reset()
        total_reward = 0
        for c in count():
            action = agent.e_greedy_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            reward = -1. if done else reward
            total_reward += reward
            memory.append(Transition(state, action, reward, next_state, done))
            state = next_state
            if done:
                print(f'episode: {eps}, reward: {total_reward}')
                break
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            agent.train(minibatch)
    print('saving model state and current episode...')
    torch.save(agent.model.state_dict(), model_file_name)
    with open(episode_file_name, 'wb') as f:
        pickle.dump(eps, f)
    print('done.')

    try:
        state = env.reset()
        total_reward = 0
        for c in count():
            env.render()
            action = agent.greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                print('reward:', total_reward)
                break
    finally:
        env.close()


if __name__ == '__main__':
    main()
