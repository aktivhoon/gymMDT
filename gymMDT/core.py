import numpy as np

class Action(object):
    def __init__(self):
        # physical action
        self.u = None

class State(object):
    def __init__(self, idx):
        self.state_number = idx
        self.item = None
        self.left = None
        self.right = None
        self.up = None
        self.down = None
        self.shift = None

class Agent(object):
    def __init__(self, i):
        self.number = i
        self.reward = 0
        self.action_callback = None
        self.action = Action()

class Item(object):
    def __init__(self, reward):
        self.reward = reward
        self.item_name = None

class Coin(object):
    def __init__(self, idx, handcraft = False, reward_list = None, reward_name_list = None):
        if not handcraft:
            idx = int(idx)
            possible_coins = [0, 10, 20, 40]
            coin_names = ['Z', 'Y', 'B', 'R']
            self.reward = possible_coins[idx]
            self.name = coin_names[idx]
        else:
            idx = int(idx)
            self.reward = reward_list[idx]
            self.name = reward_name_list[idx]

class World(object):
    def __init__(self, n_blocks=20):
        self.n_blocks = n_blocks

        # parameters for world setting
        self.states = [State(i) for i in range(21)]
        self.reward_code = None
        self.task_code = None
        self.code = None
        self.tasks = None
        self.block_genotypes = None

        # simulation timestep
        self.steps = 0
        self.trials = 0
        self.time = 0
        self.initial_block = True
        self.block_idx = 0

        self.s = 0
        self.action_list = []
        self.state_list = []
        self.reward = []
    
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def place_coins(self, coins):
        assert len(coins) == 21

        for i, coin in enumerate(coins):

            if coin is None:
                self.states[i].item = Item(0)
            else:
                self.states[i].item = Item(coin.reward)
                self.states[i].item.item_name = coin.name

    def set_probs(self, shift_val):
        for sa in range(10):
            self.states[sa].shift = shift_val
    
    def set_world(self, task_setting):
        if task_setting not in ['glascher']:
            self._set_settings(task_setting)
        else:
            self._set_glascher()
        self.set_coins(self.goal_setting, self.coin_setting)
    
    def _set_settings(self, task_setting):
        self.shift_setting = 'd' if task_setting[1] == 'd' else 'r'
        shift_val = 0.9 if task_setting[1] == 'd' else 0.5
        self.set_probs(shift_val)

        self.goal_setting = task_setting[0]
        self.coin_setting = '_' if self.goal_setting == 'f' else task_setting[2]
        
    def _set_glascher(self):
        self.shift_setting = 'm'
        self.set_probs(0.7)
        self.goal_setting = 'f'
        self.coin_setting = '_'
        self.reward_code = 'glascher'
    
    def set_coins(self, settings, coin_color=None):
        if self.reward_code is not None:
            if isinstance(self.reward_code, str) and self.reward_code.lower() == 'glascher':
                self._set_coins_glascher()
            elif len(self.reward_code) == 4:
                self._set_coins_restrict(settings, coin_color)
            elif len(self.reward_code) == 16:
                self._set_coins_general(settings, coin_color)
        else:
            self._set_coins_original(settings, coin_color)
    
    def _set_coins_glascher(self):
        reward_list = [0, 10, 25]
        reward_name_list = ['Z', 'L', 'H']
        distrib_list = [1, 0, 0, 1, 0, 1, 0, 2, 2, 0, 1, 0, 0, 1, 0, 2]

        coins = [None] * 5 + [Coin(idx, handcraft=True, reward_list=reward_list, reward_name_list=reward_name_list) for idx in distrib_list]
        self.place_coins(coins=coins)

    def _set_coins_restrict(self, settings, coin_color=None):
        reward_patterns = {
            0: [1, 1, 0, 3], 1: [1, 3, 3, 0], 2: [2, 2, 1, 3],
            3: [2, 3, 3, 0], 4: [3, 0, 2, 3], 5: [0, 3, 2, 0],
            6: [2, 1, 0, 3], 7: [3, 1, 2, 3], 8: [1, 3, 2, 2]
        }
        
        idx_list = [idx for code in self.reward_code for idx in reward_patterns[code]]

        if 'g' in settings:
            if coin_color == 'Y':
                idx_list = [1 if idx == 1 else 0 for idx in idx_list]
            elif coin_color == 'B':
                idx_list = [2 if idx == 2 else 0 for idx in idx_list]
            elif coin_color == 'R':
                idx_list = [3 if idx == 3 else 0 for idx in idx_list]

        coins = [None] * 5 + [Coin(idx) for idx in idx_list]
        self.place_coins(coins=coins)
    
    def _set_coins_general(self, settings, coin_color=None):
        idx_list = self.reward_code
        
        if 'g' in settings:
            if coin_color == 'Y':
                idx_list = [1 if idx == 1 else 0 for idx in idx_list]
            elif coin_color == 'B':
                idx_list = [2 if idx == 2 else 0 for idx in idx_list]
            elif coin_color == 'R':
                idx_list = [3 if idx == 3 else 0 for idx in idx_list]

        coins = [None] * 5 + [Coin(idx) for idx in idx_list]
        self.place_coins(coins=coins)

    def _set_coins_original(self, settings, coin_color=None):
        coin_color_map = {
            'Y': [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'B': [2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 0],
            'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 3]
        }
        default_list = [2, 1, 1, 0, 1, 0, 2, 0, 2, 3, 3, 0, 2, 0, 0, 3]
        
        idx_list = coin_color_map.get(coin_color, default_list) if 'g' in settings else default_list

        coins = [None] * 5 + [Coin(idx) for idx in idx_list]
        self.place_coins(coins=coins)

    # update state of the world
    def step(self):
        self._update_indices()
        self._set_scripted_agent_actions()
        self._apply_actions()
        self._update_state_and_rewards()

    def _update_indices(self):
        if self.time > 4:
            self.block_idx += 1
            self.time = 1

    def _set_scripted_agent_actions(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            sa = int(self.s * 2 + self.agents[0].action.u)
            trans_prob = self.states[sa].shift
            agent.action.u = agent.action_callback(trans_prob)
    
    def _apply_actions(self):
        actions = [int(agent.action.u) for agent in self.agents]
        action_pair = actions[0] * 2 + actions[1]
        self.s = self.s * 4 + action_pair + 1
        self.action_list.append(actions)
        self.state_list.append(self.s)
    
    def _update_state_and_rewards(self):
        if self.states[self.s].item is not None:
            self.agents[0].reward = self.states[self.s].item.reward
        
        self.reward_list.append(self.agents[0].reward)

        self.steps += 1
        if self.steps > 1:
            self.trials += 1

        if self.time == 4:
            self._prepare_next_block()
    
    def _prepare_next_block(self):
        self.next_block_idx = self.block_idx + 1
        self.initial_block = False

    def reset_param(self, initial_block=True):
        self.s = 0
        for agent in self.agents:
            agent.reward = 0
        if initial_block:
            self.set_world(self.tasks[self.block_idx])
        elif self.next_block_idx < self.n_blocks:
            self.set_world(self.tasks[self.next_block_idx])