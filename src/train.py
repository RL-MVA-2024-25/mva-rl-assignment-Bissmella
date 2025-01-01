from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import numpy as np
import random
import os
import torch.nn as nn

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


def update_running_stats(state, state_mean, state_var, count):
    # Incrementally compute mean and variance
    count += 1
    new_mean = state_mean + (state - state_mean) / count
    new_var = state_var + (state - state_mean) * (state - new_mean)
    
    return new_mean, new_var, count

def normalize_state(state, state_mean, state_var, count):
    return (state - state_mean) / (np.sqrt(state_var / count) + 1e-8)


class ProjectAgent:
    def __init__(self, config=None, model=None):
        if config != None:
            device = "cuda" if next(model.parameters()).is_cuda else "cpu"
            self.gamma = config['gamma']
            self.batch_size = config['batch_size']
            self.nb_actions = config['nb_actions']
            self.memory = ReplayBuffer(config['buffer_size'], device)
            self.epsilon_max = config['epsilon_max']
            self.epsilon_min = config['epsilon_min']
            self.epsilon_stop = config['epsilon_decay_period']
            self.epsilon_delay = config['epsilon_delay_decay']
            self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
            self.model = model 
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        else:
            state_dim = env.observation_space.shape[0]
            n_action = env.action_space.n 
            nb_neurons=256
            self.model = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, n_action))
            self.state_mean = np.zeros(6)  # state_dim is the dimensionality of the state
            self.state_var = np.ones(6)
            self.count = 1e-5
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            done_flag = done or trunc
            self.memory.append(state, action, reward, next_state, done_flag)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            if done_flag:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return


    def act(self, observation, use_random=False):
        self.state_mean, self.state_var, self.count = update_running_stats(observation, self.state_mean, self.state_var, self.count)
        state = normalize_state(observation, self.state_mean, self.state_var, self.count)
        action = greedy_action(self.model, state)
        return action#greedy_action(self.model, observation)

    def save(self, path):
        filename = "model.pth"
        path = os.path.join(path, filename)
        torch.save(self.model.state_dict(), path)


    def load(self):
        self.model.load_state_dict(torch.load("model_v2.pth", map_location=torch.device('cpu')))

