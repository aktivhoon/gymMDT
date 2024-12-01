import numpy as np
from gymMDT.core import World, Agent
from gymMDT.environment import MDTEnv
from gymMDT.scenario import BaseScenario
from gymMDT.core import Coin

def transition_state(trans_prob):
    if np.random.rand() < trans_prob:
        return 0
    else:
        return 1

def taskset2par(taskset):
    goal_directed = (taskset[0] == 'g')
    determine = (taskset[1] == 'd')

    return goal_directed, determine

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
    else:
        raise ValueError("genotype must be in range 0-11")
    return taskset

def code2tasks(code):
    return [gene2block(genotype) for genotype in code]

def make_reward(reward_code):
    # Define reward values
    REWARDS = {0: 'Z', 1: 'Y', 2: 'B', 3: 'R'}
    
    # Define reward patterns as a constant
    PATTERNS = {
        0: [1, 1, 0, 3],
        1: [1, 3, 3, 0],
        2: [2, 2, 1, 3],
        3: [2, 3, 3, 0],
        4: [3, 0, 2, 3],
        5: [0, 3, 2, 0],
        6: [2, 1, 0, 3],
        7: [3, 1, 2, 3],
        8: [1, 3, 2, 2]
    }
    
    # Generate reward indices
    indices = [idx for code in reward_code for idx in PATTERNS[code]]
    
    # Generate reward dictionary for positions 5-20
    return {pos: REWARDS[idx] for pos, idx in zip(range(5, 21), indices)}

class BlockRewardScenario(BaseScenario):
    def make_world(self, code, n_blocks=20):
        world = World()

        # add agents
        world.agents = [Agent(i) for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i

        # player 1 serves as static transition probability
        world.agents[1].action_callback = transition_state

        world.code = code
        world.reward_code = code[:4]
        world.task_code = code[4:]
        world.type = "coins"

        world.reward_D= make_reward(world.reward_code)

        world.tasks = code2tasks(world.task_code)
        world.block_genotypes = 8
        world.n_blocks = n_blocks

        return world

    def reset_world(self, world):
        # set random initial states
        world.curr_state = 0
        world.steps = 0

        if (world.trials) == 0 or (world.time == 4):
            if world.initial_block:
                world.set_world(world.tasks[world.block_idx])
            elif world.next_block_idx <= world.n_blocks - 1:
                world.set_world(world.tasks[world.next_block_idx])

        world.action_list = []
        world.state_list = []
        world.reward_list = []
        world.time += 1

    def reward(self, agent):
        return agent.reward

    def observation(self, world):
        # Define observation vectors
        obs = {}
        obs['state'] = world.curr_state
        return obs

    def info(self, world):
        # Information
        info = {}
        info['time'] = world.time
        info['steps'] = world.steps
        info['env_reward'] = world.reward_D
        if world.goal_setting == 'g':
            info['current_set'] = world.goal_setting + world.shift_setting + world.coin_setting
        else:
            info['current_set'] = world.goal_setting + world.shift_setting
        if world.steps > 1:
            info['trials'] = world.trials
            info['p1 a1'] = world.action_list[0][0]
            info['p2 a1'] = world.action_list[0][1]
            info['p1 a2'] = world.action_list[1][0]
            info['p2 a2'] = world.action_list[1][1]

            info['states'] = world.state_list

            info['r1'] = world.reward_list[0]
            info['r2'] = world.reward_list[1]

            if world.time == 4:
                if world.next_block_idx <= world.n_blocks - 1:
                    info['next_set'] = world.tasks[world.next_block_idx]
                    # if self.compare_set(info['current_set'], info['next_set']):
                    #     info['backward_set'] = self.compare_set(info['current_set'], info['next_set'])
                else:
                    info['next_set'] = 'end'
            
        return info

    def done(self, world):
        return world.steps > 1



class BlockRewardEnv(MDTEnv):
    def __init__(self, code, n_blocks=20):
        assert len(code) == 4 + n_blocks, f'code must be a list of {4 + n_blocks} numbers (4 reward codes + {n_blocks} task codes)'
        assert all(0 <= x <= 8 for x in code[:4]), 'first 4 numbers of code must be between 0 and 8'
        assert all(0 <= x <= 11 for x in code[4:]), f'last {n_blocks} numbers of code must be between 0 and 11'
        
        scenario = BlockRewardScenario()
        world = scenario.make_world(code, n_blocks)
        super(BlockRewardEnv, self).__init__(world, reset_callback=scenario.reset_world, 
                                        reward_callback=scenario.reward, 
                                        observation_callback=scenario.observation, 
                                        done_callback=scenario.done,
                                        info_callback=scenario.info, shared_viewer=False)
        self.code = code
        self.task_code = code[4:]
        self.reward_code = code[:4]
        self.n_blocks = n_blocks