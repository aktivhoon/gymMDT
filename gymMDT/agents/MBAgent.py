import numpy as np
from gymMDT.agents.BaseAgent import BaseAgent, softmax
import os
import time

class MBAgent(BaseAgent):
    def __init__(self, env, lr=None, beta=None, variable=True):
        super(MBAgent, self).__init__()
        self.env = env
        self.s = 0
        self.task_setting = None

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

        # Transition Matrix: T[s][a] -> 2 x 1 transition matrix to visit two different states
        self.T = np.ones((5, 2, 2)) * (1 / 2)

        self.visited_item = {
                            "zero": False, "yellow":False, "blue": False, "red": False
                            }
        self.visited_state = np.zeros(16)
        self.Q = np.zeros((5, 2))
        self.env = env
    
    def update(self, s, a, a_o, s_next, a_next, a_o_next, task_setting=None):
        last_state = s_next * 4 + a_next * 2 + a_o_next + 1
        self.visited_item[self.env.world.states[last_state].item.item_name] = True
        self.visited_state[last_state-5] = True

        # Forward update: MB is supposed to learn with forward learning on policy.
        self.T[s][a] -= self.lr * self.T[s][a]
        self.T[s][a][a_o] += self.lr

        self.T[s_next][a_next] -= self.lr * self.T[s_next][a_next]
        self.T[s_next][a_next][a_o_next] += self.lr

        _next_states = self.env.get_next_states(s, a)
        _next_s = _next_states
        self.Q[s][a] = np.sum(self.T[s][a] * np.max(self.Q[_next_s], axis=1))

        _next_states = self.env.get_next_states(s_next, a_next)
        _next_s = _next_states
        _R = self._get_Reward(s_next, a_next)
        self.Q[s_next][a_next] = np.sum(self.T[s][a] * _R)

        if task_setting is not None:
            self.task_setting = task_setting
            # Backward upate: based on the task setting cue, agent re-evalutes Q-values
            for _step in np.array(["S1", "S0"]):
                for _state in self.env.state_list[_step]:
                    _s = _state
                    for _action in range(2):
                        if _step == "S1":
                            _R = self._get_Reward(_state, _action)
                            self.Q[_state][_action] = np.sum(self.T[_s][_action] * _R)
                        elif _step == "S0":
                            _next_states = self.env.get_next_states(_state, _action)
                            _next_s = _next_states
                            self.Q[_s][_action] = np.sum(
                                self.T[_s][_action] * np.max(self.Q[_next_s], axis=1)
                            )

        self.policy = softmax(self.Q, self.beta)
   
    # function to call the environment reward array: only used for debugging
    def _print_env_r(self):
        R_array = np.zeros(16)
        for s in range(5, 21):
            R_array[s-5] = self.env.world.states[s].item.reward
        return R_array

    def _get_Reward(self, state, action):
        R_array = np.zeros(2)
        next_states = self.env.get_next_states(state, action)
        for i, _states in enumerate(next_states):
            _s = _states
            _item = self.env.world.states[_s].item
            if _item is not None and self.visited_state[_s-5]:
                R_array[i] = _item.reward
            else:
                R_array[i] = 0
        return R_array