import numpy as np
from gymMDT.agents.BaseAgent import BaseAgent, softmax
import os
import time

class MBAgent(BaseAgent):
    def __init__(self, lr=None, beta=None, variable=True):
        super(MBAgent, self).__init__()
        self.s = 0
        self.backward_set = None

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

        self.state_list = {'S0': [0], 'S1': [1, 2, 3, 4], 
                           'S2': [ 5,  6,  7,  8, 
                                   9, 10, 11, 12, 
                                  13, 14, 15, 16,
                                  17, 18, 19, 20]}

        # self.visited_item = {
        #                     "zero": False, "yellow":False, "blue": False, "red": False
        #                     }
        # make a dictionary for visited states, for S2
        self.visited_state = {i: False for i in range(5, 21)}
        self.Q = np.zeros((5, 2))
    
    def update(self, s, a, a_o, s_next, a_next, a_o_next, env_reward, current_set, next_set):
        last_state = s_next * 4 + a_next * 2 + a_o_next + 1
        self.visited_state[last_state] = True

        if next_set is None:
            next_set = current_set
            backward_set = None
        else:
            backward_set = self.compare_set(current_set, next_set)
        
        masked_env_reward = dict(env_reward)
        
        if next_set[0] == 'g':
            target_values = {'Y': 10, 'B': 20, 'R': 40}
            color_setting = next_set[2]
            if color_setting in target_values:
                target = target_values[color_setting]
                masked_env_reward = {
                    pos: (val if val == target else 0)
                    for pos, val in masked_env_reward.items()
                }
        
        # mask the env_reward with the non-visited states
        for _state in self.state_list['S2']:
            if not self.visited_state[_state]:
                masked_env_reward[_state] = 0
    
        # Forward update: MB is supposed to learn with forward learning on policy.
        self.T[s][a] -= self.lr * self.T[s][a]
        self.T[s][a][a_o] += self.lr

        self.T[s_next][a_next] -= self.lr * self.T[s_next][a_next]
        self.T[s_next][a_next][a_o_next] += self.lr

        _next_s = self.get_next_states(s, a)
        # Q(s, a) = sum(T(s, a) * max_a(Q(s'))) -> No reward
        self.Q[s][a] = np.sum(self.T[s][a] * np.max(self.Q[_next_s], axis=1))

        _next_s = self.get_next_states(s_next, a_next)
        _R = [masked_env_reward[_next_s[0]], masked_env_reward[_next_s[1]]]
        # Q(s', a') = sum(T(s', a') * R(s'')) -> Reward
        self.Q[s_next][a_next] = np.sum(self.T[s_next][a_next] * _R)

        if backward_set is not None:
            self.backward_set = backward_set
            # Backward upate: based on the backward setting cue, agent re-evalutes Q-values
            for _step in np.array(["S1", "S0"]):
                for _state in self.state_list[_step]:
                    _s = _state
                    for _action in range(2):
                        if _step == "S1":
                            _next_s = self.get_next_states(_state, _action)
                            _R = [masked_env_reward[_next_s[0]], masked_env_reward[_next_s[1]]]
                            self.Q[_state][_action] = np.sum(self.T[_s][_action] * _R)
                        elif _step == "S0":
                            _next_s = self.get_next_states(_state, _action)
                            self.Q[_s][_action] = np.sum(
                                self.T[_s][_action] * np.max(self.Q[_next_s], axis=1)
                            )

        self.policy = softmax(self.Q, self.beta)
    

    def get_next_states(self, state, action):
        """
                           0
           1         2            3            4
        5 6 7 8  9 10 11 12  13 14 15 16  17 18 19 20
        """
        if action == 0:
            return [state * 4 + 1, state * 4 + 2]

        elif action == 1:
            return [state * 4 + 3, state * 4 + 4]
    
    def compare_set(self, set1, set2):
        if (set1[0] == set2[0]) & (set1[0] == 'g'):
            if set1[2] != set2[2]:
                return set2
            else:
                return None
        elif (set1[0] != set2[0]):
            return set2
        else:
            return None