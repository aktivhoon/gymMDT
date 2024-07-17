import numpy as np

def softmax(Q, beta):
    return np.exp(Q * beta) / np.sum(np.exp(Q * beta), axis=1).reshape(-1, 1)

class BaseAgent(object):
    def __init__(self):
        self.policy = np.ones((5, 2)) * (1 / 2)
        self.Q = np.zeros((5, 2))
        self.action = 0

    def choose_action(self, s):
        self.action = np.random.choice(2, 1, p=self.policy[s])
        return self.action
