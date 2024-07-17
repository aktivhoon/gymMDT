import numpy as np
from gymMDT.agents.BaseAgent import BaseAgent, softmax
import os
import time

class MFAgent(BaseAgent):
    def __init__(self, lr=None, beta=None, variable=True):
        super(MFAgent, self).__init__()
        beta_loc = 2.8588
        beta_scale = 4.176
        if lr is not None and beta is not None:
            self.lr = lr
            self.beta = beta
        else:
            if variable:
                np.random.seed((os.getpid() * int(time.time() * 10000)) % 1234567)
                self.lr = np.random.uniform(0.03, 0.2, 1)
                self.beta = 1/(np.random.exponential(beta_scale) + beta_loc)
            else:
                self.lr = 0.1
                self.beta = 0.15

    def update(self, s, a, R, s_next, a_next, R_next):
        self.Q[s][a] += self.lr * (R - self.Q[s][a] + self.Q[s_next][a_next])
        self.Q[s_next][a_next] += self.lr * (R_next - self.Q[s_next][a_next])
        self.policy = softmax(self.Q, self.beta)