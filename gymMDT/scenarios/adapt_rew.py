import numpy as np
from gymMDT.environment import MDTEnv
from gymMDT.scenario import BaseScenario
from gymMDT.core import World, Agent, Coin

COIN_VALUE = {'Y': 10, 'B': 20, 'R': 40}

def transition_state(trans_prob):
    if np.random.rand() < trans_prob:
        return 0
    else:
        return 1

def gene2block(genotype):
    if genotype == 0 or genotype == 8 or genotype == 9:
        taskset = 'fr'
    elif genotype == 1 or genotype == 10 or genotype == 11:
        taskset = 'fd'
    elif genotype == 2:
        taskset = 'grY'
    elif genotype == 3:
        taskset = 'grB'
    elif genotype == 4:
        taskset = 'grR'
    elif genotype == 5:
        taskset = 'gdY'
    elif genotype == 6:
        taskset = 'gdB'
    elif genotype == 7:
        taskset = 'gdR'
    elif genotype == 12:
        taskset = 'ar'
    elif genotype == 13:
        taskset = 'ad'
    else:
        raise ValueError("genotype must be in range 0-13")
    return taskset

def code2tasks(code):
    return [gene2block(genotype) for genotype in code]

def make_reward(reward_code):
    REWARDS = {0: 'Z', 1: 'Y', 2: 'B', 3: 'R'}
    PATTERNS = {
        0: [1, 1, 0, 3], 1: [1, 3, 3, 0], 2: [2, 2, 1, 3],
        3: [2, 3, 3, 0], 4: [3, 0, 2, 3], 5: [0, 3, 2, 0],
        6: [2, 1, 0, 3], 7: [3, 1, 2, 3], 8: [1, 3, 2, 2]
    }
    indices = [idx for code in reward_code for idx in PATTERNS[code]]
    return {pos: REWARDS[idx] for pos, idx in zip(range(5, 21), indices)}

class AdaptiveWorld(World):
    def __init__(self, n_blocks=20):
        super().__init__(n_blocks)
        # For ambiguous rewards
        self.visited_coins = {'Y': 0, 'B': 0, 'R': 0}
        self.visited_unique_coins = set()
        self.valid_coin = None
        
        # For decay mechanics
        self.coin_decay = {'Y': 1.0, 'B': 1.0, 'R': 1.0}
        self.coin_visit_count = {'Y': 0, 'B': 0, 'R': 0}
        self.consecutive_non_visits = {'Y': 0, 'B': 0, 'R': 0}
        
        # Thresholds and rates
        self.VISIT_THRESHOLD = 4
        self.NON_VISIT_THRESHOLD = 3
        self.NORMAL_DECAY_RATE = 0.95
        self.DRASTIC_DECAY_RATE = 0.3
        self.NORMAL_REHAB_RATE = 1.05
        self.DRASTIC_REHAB_RATE = 1.5
    
    def step(self):
        # Update world state
        self._update_time()
        self._update_actions()
        self._update_state_and_rewards()
    
    def initialize_reward_structure(self, settings, target_coin=None):
        # Initialize reward structure with current decay rates
        # Reset visit tracking but maintain decay rates
        # check if the visited coins are all zero
        if self.valid_coin is not None and settings == 'a':
            # For a given probability, reset the visited coins and visited unique coins
            if np.random.rand() < 0.5:
                #print('resetting visited coins')
                self.visited_coins = {'Y': 0, 'B': 0, 'R': 0}
                self.visited_unique_coins = set()
                self.valid_coin = None
        
        if settings != 'a':
            self.visited_coins = {'Y': 0, 'B': 0, 'R': 0}
            self.visited_unique_coins = set()
            self.valid_coin = None

        if isinstance(self.reward_code, str) and self.reward_code == 'glascher':
            self._initialize_glascher_rewards()
        else:
            # print("Initializing reward structure")
            # print(f"Settings: {settings} | Target coin: {target_coin}")
            distribution = self._get_coin_distribution(settings, target_coin)
            coins = [None] * 5 + [Coin(idx) for idx in distribution]
            self.assign_coins_to_states(coins)
        
        # Apply reward modifications based on settings
            if settings == 'g' and target_coin in ['Y', 'B', 'R']:
                self._modify_rewards_for_target(target_coin)
            elif settings == 'a':
                if self.valid_coin is not None:
                    self._modify_rewards_for_target(self.valid_coin)
                else:
                    self._zero_all_rewards()
                
        # Apply current decay rates to rewards
        self.reward_D = make_reward(self.reward_code)
        for pos, state in enumerate(self.states[5:], start=5):
            if state.coin and state.coin.name in ['Y', 'B', 'R']:
                if state.coin.name == target_coin and settings == 'g':
                    state.coin.reward = round(COIN_VALUE[state.coin.name] * self.coin_decay[state.coin.name], 3)
                elif settings == 'f':
                    state.coin.reward = round(COIN_VALUE[state.coin.name] * self.coin_decay[state.coin.name], 3)
                #self.reward_D[pos] = base_reward * self.coin_decay[state.coin.name]
        

    def _update_state_and_rewards(self):
        # if self.goal_setting in ['a', 'f']:
        #     print(f"The current setting is {self.goal_setting}")
        # else:
        #     print(f"The current setting is {self.goal_setting}{self.coin_setting}")

        # for state in self.states[5:]:
        #     if state.coin:
        #         print(f"State {state.state_number}: {state.coin.name} | Reward: {state.coin.reward}")
        
        current_coin_type = None
        current_state = self.states[self.curr_state]
        if current_state.coin:
            # print(f"Current state: {current_state.state_number}")
            # print(f"Current coin: {current_state.coin.name}")
            # print(f"Current reward: {current_state.coin.reward}")
            current_coin_type = current_state.coin.name
            self.agents[0].reward = current_state.coin.reward

        self.reward_list.append(self.agents[0].reward)
        
        self.steps += 1
        if self.steps > 1:
            self.trials += 1

        if self.time == 4:
            self.next_block_idx = self.block_idx + 1
            self.initial_block = False

        if current_coin_type in ['Y', 'B', 'R']:
            self._handle_foraging_rewards(current_coin_type)
            if self.goal_setting == 'a':
                self._handle_ambiguous_rewards(current_coin_type)

    def _handle_ambiguous_rewards(self, visited_coin):
        # print('visited coin:', visited_coin)
        # print('self.visited_unique_coins:', self.visited_unique_coins)

        if visited_coin not in self.visited_unique_coins:
            self.visited_unique_coins.add(visited_coin)
            
            if len(self.visited_unique_coins) == 2:
                all_coins = set(['Y', 'B', 'R'])
                self.valid_coin = list(all_coins - self.visited_unique_coins)[0]
                # print('we visited coins:', self.visited_unique_coins)
                # print('valid coin:', self.valid_coin)
                for pos, state in enumerate(self.states[5:], start=5):
                    if state.coin and state.coin.name == self.valid_coin:
                        base_reward = COIN_VALUE[state.coin.name]
                        state.coin.reward = base_reward
                # print(f"decided reward_D", self.reward_D)

    def _handle_foraging_rewards(self, visited_coin):
        # Update visit counts
        self.coin_visit_count[visited_coin] += 1
        self.consecutive_non_visits[visited_coin] = 0
        
        # Reset consecutive non-visits for visited coin
        self.consecutive_non_visits[visited_coin] = 0

        # Determine decay rate based on visit count
        decay_rate = (self.DRASTIC_DECAY_RATE 
                     if self.coin_visit_count[visited_coin] >= self.VISIT_THRESHOLD 
                     else self.NORMAL_DECAY_RATE)
        self.coin_decay[visited_coin] = round(self.coin_decay[visited_coin] * decay_rate, 3)
        self._update_coin_rewards(visited_coin)
        
        # Handle non-visited coins
        for coin_type in ['Y', 'B', 'R']:
            if coin_type != visited_coin:
                self.consecutive_non_visits[coin_type] += 1
                rehab_rate = (self.DRASTIC_REHAB_RATE 
                            if self.consecutive_non_visits[coin_type] >= self.NON_VISIT_THRESHOLD 
                            else self.NORMAL_REHAB_RATE)
                self.coin_decay[coin_type] = min(1.0, round(self.coin_decay[coin_type] * rehab_rate, 3))
                self._update_coin_rewards(coin_type)
    
    def _update_coin_rewards(self, coin_type):
        # Helper method to update rewards for a specific coin type
        for pos, state in enumerate(self.states[5:], start=5):
            if state.coin and state.coin.name == coin_type:
                if (self.goal_setting == 'f' or 
                    (self.goal_setting == 'g' and state.coin.name == self.coin_setting) or
                    (self.goal_setting == 'a' and state.coin.name == self.valid_coin)):
                    base_reward = COIN_VALUE[state.coin.name]
                    state.coin.reward = round(base_reward * self.coin_decay[coin_type], 3)
        
        # for pos, state in enumerate(self.states[5:], start=5):
        #     if state.coin and state.coin.name == visited_coin:
        #         if self.goal_setting == 'f':
        #             base_reward = COIN_VALUE[state.coin.name]
        #             state.coin.reward = round(base_reward * self.coin_decay[visited_coin], 3)
        #         elif self.goal_setting == 'g' and state.coin.name == self.coin_setting:
        #             base_reward = COIN_VALUE[state.coin.name]
        #             state.coin.reward = round(base_reward * self.coin_decay[visited_coin], 3)
        #         elif self.goal_setting == 'a' and state.coin.name == self.valid_coin:
        #             base_reward = COIN_VALUE[state.coin.name]
        #             state.coin.reward = round(base_reward * self.coin_decay[visited_coin], 3)
        #         #self.reward_D[pos] *= self.coin_decay[state.coin.name]
        # # print(f'decay rate for {visited_coin}: {self.coin_decay[visited_coin]}')

        # for coin_type in ['Y', 'B', 'R']:
        #     if coin_type != visited_coin:
        #         self.coin_decay[coin_type] = min(1.0, round(self.coin_decay[coin_type] * self.REHAB_RATE, 3))
        #         for pos, state in enumerate(self.states[5:], start=5):
        #             if state.coin and state.coin.name == coin_type:
        #                 if self.goal_setting == 'f':
        #                     base_reward = COIN_VALUE[state.coin.name]
        #                     state.coin.reward = round(base_reward * self.coin_decay[coin_type], 3)
        #                 elif self.goal_setting == 'g' and state.coin.name == self.coin_setting:
        #                     base_reward = COIN_VALUE[state.coin.name]
        #                     state.coin.reward = round(base_reward * self.coin_decay[coin_type], 3)
        #                 elif self.goal_setting == 'a' and state.coin.name == self.valid_coin:
        #                     base_reward = COIN_VALUE[state.coin.name]
        #                     state.coin.reward = round(base_reward * self.coin_decay[coin_type], 3)
        #                 #self.reward_D[pos] *= self.coin_decay[state.coin.name]
        #         # print(f'decay rate for {coin_type}: {self.coin_decay[coin_type]}')

class AdaptiveRewardScenario(BaseScenario):
    def make_world(self, code, n_blocks=20):
        world = AdaptiveWorld(n_blocks)
        
        # Initialize agents
        world.agents = [Agent(i) for i in range(2)]
        world.agents[1].action_callback = transition_state
        
        # Set world parameters
        world.code = code
        world.reward_code = code[:4]
        world.task_code = code[4:]
        world.tasks = code2tasks(world.task_code)
        world.block_genotypes = 8
        world.reward_D = make_reward(world.reward_code)
        
        return world

    def reset_world(self, world):
        world.curr_state = 0
        world.steps = 0

        # print("Reset world")
        
        if (world.trials) == 0 or (world.time == 4):
            if world.initial_block:
                world.set_world(world.tasks[world.block_idx])
            elif world.next_block_idx <= world.n_blocks - 1:
                world.set_world(world.tasks[world.next_block_idx])
        
        world.action_list = []
        world.state_list = []
        world.reward_list = []
        world.time += 1

    def observation(self, world):
        # Get observation for all agents
        obs = {'state': world.curr_state}
        return obs

    def reward(self, agent):
        # Get reward for agent
        return agent.reward

    def info(self, world):
        # Get info for agent
        info = {
            'time': world.time,
            'steps': world.steps,
            'env_reward': world.reward_D,
            'coin_decay': world.coin_decay
        }
        
        if world.goal_setting == 'a':
            info['valid_coin'] = world.valid_coin if len(world.visited_unique_coins) >= 2 else None
            info['visited_coins'] = list(world.visited_unique_coins)
        
        if world.goal_setting == 'g':
            info['current_set'] = world.goal_setting + world.shift_setting + world.coin_setting
        else:
            info['current_set'] = world.goal_setting + world.shift_setting
            
        if world.steps > 1:
            info.update({
                'trials': world.trials,
                'p1 a1': world.action_list[0][0],
                'p2 a1': world.action_list[0][1],
                'p1 a2': world.action_list[1][0],
                'p2 a2': world.action_list[1][1],
                'states': world.state_list,
                'r1': world.reward_list[0],
                'r2': world.reward_list[1]
            })
            
            if world.time == 4 and world.next_block_idx <= world.n_blocks - 1:
                info['next_set'] = world.tasks[world.next_block_idx]
            
            # print(f'reward: ', world.reward_list[1])
            # print()
        
        return info

    def done(self, world):
        # Return done for agent
        return world.steps > 1

class AdaptiveRewardEnv(MDTEnv):
    def __init__(self, code, n_blocks=20):
        assert len(code) == 4 + n_blocks, f'code must be a list of {4 + n_blocks} numbers (4 reward codes + {n_blocks} task codes)'
        assert all(0 <= x <= 8 for x in code[:4]), 'first 4 numbers of code must be between 0 and 8'
        assert all(0 <= x <= 13 for x in code[4:]), f'last {n_blocks} numbers of code must be between 0 and 13'
        
        scenario = AdaptiveRewardScenario()
        world = scenario.make_world(code, n_blocks)
        super().__init__(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            done_callback=scenario.done,
            info_callback=scenario.info,
            shared_viewer=False
        )
        self.code = code
        self.task_code = code[4:]
        self.reward_code = code[:4]
        self.n_blocks = n_blocks