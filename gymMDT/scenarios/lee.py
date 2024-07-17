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
    if genotype == 0:
        taskset = 'fr'
    elif genotype == 1:
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
        raise ValueError("genotype must be in range 0-7")
    return taskset

def code2blocks(code):
    blocks = []
    for i, genotype in enumerate(code):
            if i % 4 == 0:
                block_list = []
            block_list.append(gene2block(genotype))
            if i % 4 == 3:
                blocks.append(block_list)
    return blocks

def make_coin(goal_direct=False, coin_color=None):
    coin_list = []
    if goal_direct:
        if coin_color == 'Y':
            idx_list = [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif coin_color == 'B':
            idx_list = [2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 0]
        elif coin_color == 'R':
            idx_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 3]
    else:
        idx_list = [2, 1, 1, 0, 1, 0, 2, 0, 2, 3, 3, 0, 2, 0, 0, 3]
    for idx in idx_list:
        coin_list.append(Coin(idx))
    return coin_list

CODE_LIST = [
    [6, 2, 2, 1, 0, 0, 7, 4, 4, 1, 0, 0, 7, 3, 3, 1, 0, 0, 7, 2],
    [5, 3, 3, 1, 0, 0, 5, 2, 2, 1, 0, 0, 5, 4, 4, 1, 0, 0, 7, 4],
    [5, 2, 2, 1, 0, 0, 5, 4, 4, 1, 0, 0, 5, 3, 3, 1, 0, 0, 5, 2],
    [6, 4, 4, 1, 0, 0, 5, 3, 3, 1, 0, 0, 5, 2, 2, 1, 0, 0, 7, 4],
    [6, 3, 3, 1, 0, 0, 6, 4, 4, 1, 0, 0, 6, 2, 2, 1, 0, 0, 7, 3],
    [6, 2, 2, 1, 0, 0, 6, 3, 3, 1, 0, 0, 6, 4, 4, 1, 0, 0, 5, 2],
    [7, 4, 4, 1, 0, 0, 6, 2, 2, 1, 0, 0, 6, 2, 2, 1, 0, 0, 6, 2],
    [5, 4, 4, 1, 0, 0, 5, 2, 2, 1, 0, 0, 5, 3, 3, 1, 0, 0, 7, 2],
    [5, 2, 2, 1, 0, 0, 7, 3, 3, 1, 0, 0, 6, 4, 4, 1, 0, 0, 7, 4],
    [6, 4, 4, 1, 0, 0, 6, 2, 2, 1, 0, 0, 5, 3, 3, 1, 0, 0, 7, 3]
]

class LeeScenario(BaseScenario):
    def make_world(self, code_idx, n_blocks=5):
        world = World()

        # add agents
        world.agents = [Agent(i) for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i

        # player 1 serves as static transition probability
        world.agents[1].action_callback = transition_state

        world.code = CODE_LIST[code_idx]
        world.type = "coins"

        world.block_code = code2blocks(CODE_LIST[code_idx])
        world.block_genotypes = 8
        world.n_blocks = n_blocks
        print(world.block_code)

        return world

    def reset_world(self, world):
        # set random initial states
        world.s = 0
        world.steps = 0

        if (world.trials) == 0 or (world.time == 4):
            #print("BlockScenario - reset_world : ", world.block_idx, world.intra_idx)
            if world.initial_block:
                world.set_world(world.block_code[world.block_idx][world.intra_idx])
            elif world.next_block_idx <= world.n_blocks - 1:
                world.set_world(world.block_code[world.next_block_idx][world.next_intra_idx])

        world.action_list = []
        world.state_list = []
        world.reward_list = []
        world.time += 1

    def reward(self, agent):
        return agent.reward

    def observation(self, world):
        # Define observation vectors
        obs = {}
        obs['state'] = world.s
        return obs

    def info(self, world):
        # Information
        info = {}
        info['time'] = world.time
        info['steps'] = world.steps
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
                    info['next_set'] = world.block_code[world.next_block_idx][world.next_intra_idx]
                    if self.compare_set(info['current_set'], info['next_set']):
                        info['setting'] = self.compare_set(info['current_set'], info['next_set'])
                else:
                    info['next_set'] = 'end'

        return info

    def done(self, world):
        # return as np.bool
        return world.steps > 1

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


class LeeEnv(MDTEnv):
    def __init__(self, code_idx, n_blocks=5):
        assert code_idx >= 0 and code_idx < 10
        scenario = LeeScenario()
        world = scenario.make_world(code_idx, n_blocks)
        super(LeeEnv, self).__init__(world, reset_callback=scenario.reset_world, 
                                        reward_callback=scenario.reward, 
                                        observation_callback=scenario.observation, 
                                        done_callback=scenario.done,
                                        info_callback=scenario.info, shared_viewer=False)
        self.code = CODE_LIST[code_idx]
        self.n_blocks = n_blocks