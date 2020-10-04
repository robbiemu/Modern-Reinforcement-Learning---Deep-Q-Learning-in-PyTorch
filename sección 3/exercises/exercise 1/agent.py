import random
import numpy as np
from network import DQClassifier
from flow import Flow

# Manage epsilon_greedy selection and updating of agent


class Agent():
    # n_episodes=0 gamma=0, alpha, lambda_epsilon=lambda x: x, epsilon=1
    def __init__(self, env=None, n_episodes=0, gamma=0, lambda_epsilon=lambda x: x, alpha=0.001, epsilon=1):
        self.n_episodes = n_episodes
        self.epsilon = epsilon
        self.lambda_epsilon = lambda_epsilon
        self.env = env

        self.network = DQClassifier(
            alpha, (1,), env.action_space.n)
        self.flow = Flow(self.network, gamma)

    def select_action(self, state):
        """ apply epsilon greedy """
        if random.uniform(0, 1) < self.epsilon:  # explore - random action
            action = self.env.action_space.sample()
        else:  # exploit - a_max in Q at state
            action = self.flow.predict(state)

        self.epsilon = self.lambda_epsilon(self.epsilon)

        return action

    def update(self, state, next_state, action, reward):
        """ train for the current step in the model """
        self.flow.learn(state, next_state, action, reward)
