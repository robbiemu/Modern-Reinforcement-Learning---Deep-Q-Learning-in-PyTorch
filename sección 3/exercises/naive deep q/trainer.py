import numpy as np

# helper class to make training an OpenAI env clean


class Trainer():
    def __init__(self, agent, env, n_episodes=0):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes

    def train(self):
        """ train the agent in the env for n_episodes """
        state = self.env.reset()
        scores = []
        win_pct = []
        average = 0
        for i in range(self.n_episodes):
            state = self.env.reset()
            score = 0
            done = False
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.update(state, next_state, action, reward)
                score += reward

            scores.append(score)

            if (i + 1) % 100 == 0:
                average = np.mean(scores[-100:])
                win_pct.append(average)
            if (i + 1) % (self.n_episodes * 0.025) == 0:
                print(i + 1, self.agent.epsilon, average)

        return win_pct
