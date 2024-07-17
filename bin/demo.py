#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
from gym_MDT.environment import MDTEnv
import gym_MDT.scenarios as scenarios
import time
import random
from gym_MDT.player import PlayerAgent

class Trajectory:
    def __init__(self, info, player_num, agents):
        self.s0  = 0

        self.a1  = info['p{} a1'.format(player_num)]
        self.a1o = info['p{} a1'.format(3-player_num)]
        self.s1  = info['states'][0]
        self.r1  = info['p{} r1'.format(player_num)]

        self.a2  = info['p{} a2'.format(player_num)]
        self.a2o = info['p{} a2'.format(3-player_num)]
        self.s2  = info['states'][1]
        self.r2  = info['p{} r2'.format(player_num)]

        self.self_p = agents[player_num-1].policy_agent.policy
        self.opp_p = agents[2-player_num].policy_agent.policy


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='block_geno8.py', help='Path of the scenario Python script.')
    parser.add_argument('-p', '--players', default='bc', help='Select players strategy')
    parser.add_argument('-c', '--code', type=int, default=0, help='Select environment code')
    args = parser.parse_args()

    if args.scenario == 'block_geno8.py':
        scenario = scenarios.load(args.scenario).BlockGeno8Scenario()
    elif args.scenario == 'simple.py':
        scenario = scenarios.load(args.scenario).SimpleScenario()

    n_blocks = 5
    world = scenario.make_world(random.choices([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], k=4*n_blocks), n_blocks)

    # create multiagent environment
    env = MDTEnv(world, reset_callback=scenario.reset_world, 
                              reward_callback=scenario.reward, 
                              observation_callback=scenario.observation, 
                              done_callback=scenario.done,
                              info_callback=scenario.info, shared_viewer = False)
    # create interactive policies for each agent
    agents = []
    for i in range(2):
        if args.players[i] == 'f':
            agents.append(PlayerAgent(i, env, select_class = 0))
        elif args.players[i] == 'b':
            agents.append(PlayerAgent(i, env, select_class = 1))
        elif args.players[i] == 's':
            agents.append(PlayerAgent(i, env, select_class = 2))
        elif args.players[i] == 'a':
            agents.append(PlayerAgent(i, env, select_class = 3))
        elif args.players[i] == 'c':
            agents.append(PlayerAgent(i, env, select_class = 4))
    
    # execution loop
    n = 0
    obs_n, _ = env.reset()
    while env.world.block_idx <= n_blocks-1:
        time.sleep(0.25)
        #env.render()
        act_n = []
        for _, agent in enumerate(agents):
            act_n.append(agent.action(obs_n['state']))
        obs_n, reward_n, done_n, trunc_n, info_n = env.step(act_n)
        time.sleep(0.25)
        ###print(obs_n, reward_n, done_n, info_n)
        #env.render()
        if done_n:
            tj1 = Trajectory(info_n, 1, agents)
            tj2 = Trajectory(info_n, 2, agents)
            obs_n, _ = env.reset()
            agents[0].update(tj1, task_setting = info_n.get('setting', None))
            agents[1].update(tj2)
        print(info_n)