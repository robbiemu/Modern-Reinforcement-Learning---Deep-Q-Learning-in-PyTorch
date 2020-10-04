import torch as T

# helper class to make training a continuously learning agent cleaner


class Flow():
    def __init__(self, network, gamma):
        self.network = network
        self.gamma = gamma

    def learn(self, state, next_state, action, reward):
        """ at a given time step, feed-backward and step """
        self.network.optimizer.zero_grad()

        data = T.tensor(state, dtype=T.float).to(
            self.network.device).unsqueeze(dim=0)
        actions = T.tensor(action).to(self.network.device)
        reward = T.tensor(reward).to(self.network.device)
        new_data = T.tensor(next_state, dtype=T.float).to(
            self.network.device).unsqueeze(dim=0)

        prediction = self.network.forward(data)[actions]
        y_next = self.network.forward(new_data).max()

        target = reward + self.gamma * y_next

        cost = self.network.loss(
            target, prediction).to(self.network.device)
        cost.backward()

        self.network.optimizer.step()

    def predict(self, state):
        """ select from the output of a network, converting it to a discrete action """
        data = T.tensor(state, dtype=T.float).to(
            self.network.device).unsqueeze(dim=0)
        actions = self.network.forward(data)

        return T.argmax(actions).item()
