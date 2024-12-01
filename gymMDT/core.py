import numpy as np

class Action(object):
    def __init__(self):
        self.u = None  # physical action

class State(object):
    def __init__(self, idx):
        self.state_number = idx
        self.coin = None
        self.shift = None  # transition probability

class Agent(object):
    def __init__(self, i):
        self.number = i
        self.reward = 0
        self.action_callback = None
        self.action = Action()


class Coin(object):
    POSSIBLE_REWARDS = [0, 10, 20, 40]
    COIN_NAMES = ['Z', 'Y', 'B', 'R']
    
    def __init__(self, idx, handcraft=False, reward_list=None, name_list=None):
        idx = int(idx)
        if not handcraft:
            self.reward = self.POSSIBLE_REWARDS[idx]
            self.name = self.COIN_NAMES[idx]
        else:
            self.reward = reward_list[idx]
            self.name = name_list[idx]
    
    def set_reward(self, new_reward):
        self.reward = new_reward

class World(object):
    def __init__(self, n_blocks=20):
        # Core attributes
        self.n_blocks = n_blocks
        self.states = [State(i) for i in range(21)]
        self.agents = []

        # Block settings
        self.code = None
        self.reward_code = None
        self.task_code = None
        self.tasks = None
        self.block_genotypes = None

        # Gamee state
        self.curr_state = 0
        self.time = 0 # time within a block (1-4)
        self.steps = 0 # steps within a block (1-2)
        self.trials = 0 # total trials
        self.block_idx = 0
        self.initial_block = True

        # History tracking
        self.action_list = []
        self.state_list = []
        self.reward_list = []
    
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def assign_coins_to_states(self, coins):
        # Assigns coin objects to each state in the world
        assert len(coins) == 21, "Must provide 21 coins (including None values)"
        for state, coin in zip(self.states, coins):
            state.coin = coin if coin else Coin(0)  # Empty spaces get zero-reward coins
    
    def set_transition_probabilities(self, probability):
        # Set the transition probabilities for each state in the world
        for state in self.states[:10]:
            state.shift = probability
    
    def set_world(self, task_setting):
        # Sets up the world configuration based on task settings
        if isinstance(task_setting, str) and task_setting == 'glascher':
            self._initialize_glascher_config()
        else:
            self._initialize_task_config(task_setting)
        self.initialize_reward_structure(self.goal_setting, self.coin_setting)
    
    def _initialize_task_config(self, task_setting):
        # Initializes task configuration from settings
        # Parse task settings
        self.goal_setting = task_setting[0]
        self.shift_setting = task_setting[1]
        self.coin_setting = '_' if self.goal_setting in ['f', 'a'] else task_setting[2]
        
        # Set transition probabilities
        prob = 0.9 if self.shift_setting == 'd' else 0.5
        self.set_transition_probabilities(prob)
        
    def _initialize_glascher_config(self):
        # Initializes Glascher-specific configuration
        self.shift_setting = 'm'
        self.goal_setting = 'f'
        self.coin_setting = '_'
        self.reward_code = 'glascher'
        self.set_transition_probabilities(0.7)
    
    def initialize_reward_structure(self, settings, target_coin=None):
        # Initializes the reward structure based on settings and target coin
        if isinstance(self.reward_code, str) and self.reward_code == 'glascher':
            self._initialize_glascher_rewards()
        else:
            distribution = self._get_coin_distribution(settings, target_coin)
            coins = [None] * 5 + [Coin(idx) for idx in distribution]
            self.assign_coins_to_states(coins)
        
        # Apply reward modifications based on settings
            if settings == 'g' and target_coin in ['Y', 'B', 'R']:
                self._modify_rewards_for_target(target_coin)
            elif settings == 'a':
                self._zero_all_rewards()

    def _initialize_glascher_rewards(self):
        # Initializes the special Glascher reward structure
        rewards = [0, 10, 25]
        names = ['Z', 'L', 'H']
        distribution = [1, 0, 0, 1, 0, 1, 0, 2, 2, 0, 1, 0, 0, 1, 0, 2]
        coins = [None] * 5 + [Coin(idx, True, rewards, names) for idx in distribution]
        self.assign_coins_to_states(coins)

    def _get_coin_distribution(self, settings, target_coin):
        if isinstance(self.reward_code, str) and self.reward_code == 'glascher':
            return None
        elif self.reward_code is None:
            return self._get_default_distribution(settings, target_coin)
        elif len(self.reward_code) == 4:
            return self._get_restricted_distribution(settings, target_coin)
        elif len(self.reward_code) == 16:
            return self._get_general_distribution(settings, target_coin)
            
    
    def _get_restricted_distribution(self, settings, target_coin):
        PATTERNS = {
            0: [1, 1, 0, 3], 1: [1, 3, 3, 0], 2: [2, 2, 1, 3],
            3: [2, 3, 3, 0], 4: [3, 0, 2, 3], 5: [0, 3, 2, 0],
            6: [2, 1, 0, 3], 7: [3, 1, 2, 3], 8: [1, 3, 2, 2],
            9: [0, 0, 0, 0]
        }
        return [idx for code in self.reward_code for idx in PATTERNS[code]]

    def _get_general_distribution(self, settings, target_coin):
        return self.reward_code

    def _get_default_distribution(self, settings, target_coin):
        COIN_MAPS = {
            'Y': [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'B': [2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 0],
            'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 3]
        }
        return COIN_MAPS.get(target_coin, [2, 1, 1, 0, 1, 0, 2, 0, 2, 3, 3, 0, 2, 0, 0, 3])

    def _modify_rewards_for_target(self, target_coin):
        target_type = {'Y': 1, 'B': 2, 'R': 3}[target_coin]
        for state in self.states:
            if state.coin and state.coin.name != 'Z':  # Preserve zero coins
                if state.coin.name != target_coin:
                    state.coin.set_reward(0)

    def _zero_all_rewards(self):
        for state in self.states:
            if state.coin:
                state.coin.set_reward(0)
    
    def step(self):
        # Executes one step in the environment)
        self._update_time()
        self._update_actions()
        self._update_rewards()

    def _update_time(self):
        # Updates time and block index
        if self.time > 4:
            self.block_idx += 1
            self.time = 1

    def _update_actions(self):
        # Updates agent actions and state
        # Get scripted agent actions
        player = self.agents[0]
        for agent in self.scripted_agents:
            state_action = int(self.curr_state * 2 + player.action.u)
            agent.action.u = agent.action_callback(self.states[state_action].shift)

        # Update state based on actions
        actions = [int(agent.action.u) for agent in self.agents]
        action_pair = actions[0] * 2 + actions[1]
        self.curr_state = self.curr_state * 4 + action_pair + 1
        
        # Record history
        self.action_list.append(actions)
        self.state_list.append(self.curr_state)

    def _update_rewards(self):
        # Updates rewards and game progression
        curr_state = self.states[self.curr_state]
        player = self.agents[0]
        
        # Update rewards
        player.reward = curr_state.coin.reward if curr_state.coin else 0
        self.reward_list.append(player.reward)
        
        # Update counters
        self.steps += 1
        if self.steps > 1:
            self.trials += 1
        
        if self.time == 4:
            self.next_block_idx = self.block_idx + 1
            self.initial_block = False

    def reset_param(self, initial_block=True):
        # Resets parameters for new block
        self.curr_state = 0
        for agent in self.agents:
            agent.reward = 0
            
        if initial_block:
            self.set_world(self.tasks[self.block_idx])
        elif self.next_block_idx < self.n_blocks:
            self.set_world(self.tasks[self.next_block_idx])