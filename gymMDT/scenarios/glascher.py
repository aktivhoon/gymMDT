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

class GlascherScenario(BaseScenario):
    def make_world(self):
        world = World()

        # add agents
        world.agents = [Agent(i) for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
        # player 1 serves as static transition probability
        world.agents[1].action_callback = transition_state

        world.code = [['glascher'] * 4] * 5
        world.block_code = world.code

        world.type = "coins"

        world.n_blocks = 5

        return world

    def reset_world(self, world):
        # set random initial states
        world.s = 0
        world.steps = 0

        world.set_world('glascher')

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
                #print(world.block_code, world.next_block_idx, world.next_intra_idx)
                if world.next_block_idx <= world.n_blocks - 1:
                    info['next_set'] = world.block_code[world.next_block_idx][world.next_intra_idx]
                    if self.compare_set(info['current_set'], info['next_set']):
                        info['setting'] = self.compare_set(info['current_set'], info['next_set'])
                else:
                    info['next_set'] = 'end'

        return info

    def done(self, world):
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


class GlascherEnv(MDTEnv):
    def __init__(self):
        
        scenario = GlascherScenario()
        world = scenario.make_world()
        super(GlascherEnv, self).__init__(world, reset_callback=scenario.reset_world, 
                                        reward_callback=scenario.reward, 
                                        observation_callback=scenario.observation, 
                                        done_callback=scenario.done,
                                        info_callback=scenario.info, shared_viewer=False)