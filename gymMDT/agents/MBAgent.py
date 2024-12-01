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

        self.reward_map = {i : 'Z' for i in range(5, 21)}

        # make a dictionary for visited states, for S2
        self.visited_state = {i: False for i in range(5, 21)}
        self.Q = np.zeros((5, 2))

        self.target_values = {'Y': 10, 'B': 20, 'R': 40}
    
    def update(self, s, a, a_o, s_next, a_next, a_o_next, env_reward, current_set, next_set):
        # Update state tracking
        last_state = self._calculate_last_state(s_next, a_next, a_o_next)
        self.visited_state[last_state] = True

        next_set, backward_set = self._handle_sets(current_set, next_set)
        self._update_rewards(env_reward, last_state)
        
        self._update_transition_probabilities(s, a, a_o, s_next, a_next, a_o_next)
        self._update_q_values_forward(s, a, s_next, a_next, next_set)

        if backward_set is not None:
            self._update_q_values_backward(backward_set)
        
        self.policy = softmax(self.Q, self.beta)
    
    def _calculate_last_state(self, s_next, a_next, a_o_next):
        return s_next * 4 + a_next * 2 + a_o_next + 1

    def _handle_sets(self, current_set, next_set):
        if next_set is None:
            next_set = current_set
            return next_set, None
        return next_set, self.compare_set(current_set, next_set)
    
    def _update_rewards(self, env_reward, last_state):
        masked_rewards = dict(env_reward)
        for state in self.state_list['S2']:
            if not self.visited_state[state]:
                masked_rewards[state] = 'Z'
        last_state_coin = masked_rewards[last_state]
        self.reward_map[last_state] = last_state_coin

    def _update_transition_probabilities(self, s, a, a_o, s_next, a_next, a_o_next):
        self.T[s][a] -= self.lr * self.T[s][a]
        self.T[s][a][a_o] += self.lr
        
        self.T[s_next][a_next] -= self.lr * self.T[s_next][a_next]
        self.T[s_next][a_next][a_o_next] += self.lr
    
    def _calculate_rewards(self, next_states, set_info):
        if set_info[0] == 'g':
            target_coin = set_info[2]
            target_value = self.target_values.get(target_coin, 0)
            return np.array([target_value if self.reward_map[state] == target_coin else 0 
                           for state in next_states])
        return np.array([self.target_values.get(self.reward_map[state], 0) for state in next_states])

    def _update_q_values_forward(self, s, a, s_next, a_next, next_set):
        next_states = self.get_next_states(s, a)
        self.Q[s][a] = np.sum(self.T[s][a] * np.max(self.Q[next_states], axis=1))
        
        next_states = self.get_next_states(s_next, a_next)
        rewards = self._calculate_rewards(next_states, next_set)
        self.Q[s_next][a_next] = np.sum(self.T[s_next][a_next] * rewards)
    
    def _update_q_values_backward(self, backward_set):
        self.backward_set = backward_set
        
        for step in ["S1", "S0"]:
            for state in self.state_list[step]:
                for action in range(2):
                    next_states = self.get_next_states(state, action)
                    
                    if step == "S1":
                        rewards = self._calculate_rewards(next_states, backward_set)
                        self.Q[state][action] = np.sum(self.T[state][action] * rewards)
                    else:  # S0
                        self.Q[state][action] = np.sum(
                            self.T[state][action] * np.max(self.Q[next_states], axis=1)
                        )

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