class Agent:
    def __init__(self, Q, alpha, gamma, n_epsiodes, n_actions, epsilon_max=1.0, epsilon_min=0, epsilon_dec_mod=1.0):
        self.Q = Q
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.n_actions = n_actions
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec_mod * (1-(1/n_epsiodes)) ** 2
        self.epsilon = epsilon_max
        self.state = None

    def __decrement_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min \
            else self.epsilon_min

    def select_action(self,):  # apply epsilon greedy
        if random.uniform(0, 1) < self.epsilon:  # explore - random action
            action = np.random.choice([i for i in range(self.n_actions)])
        else:  # exploit - a_max in Q at state
            action = np.argmax([self.Q[(state, a)]
                                for a in range(self.n_actions)])
        return action

    def update(self, state, action, reward):
        self.Q[self.state, action] += self.alpha * (reward + self.gamma *
                                                    np.max(self.Q[state]) - self.Q[self.state, action])

        self.state = state
        self.__decrement_epsilon()
