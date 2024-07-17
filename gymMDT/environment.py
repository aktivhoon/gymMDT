import gym
from gym import error, spaces, utils
from gym.envs.registration import EnvSpec
import numpy as np
from gymMDT.multi_discrete import MultiDiscrete

class MDTEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        self.world  = world
        self.agents = self.world.policy_agents
        self.n = len(self.agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0,...,N otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True
        self.time = 0
        self.steps = 0
        self.state_list = {'S0': [0], 'S1': [1, 2, 3, 4], 
                           'S2': [ 5,  6,  7,  8, 
                                   9, 10, 11, 12, 
                                  13, 14, 15, 16,
                                  17, 18, 19, 20]}

        # configure spaces
        self.num_agents = len(self.agents)
        self.observation_space = []

        if self.num_agents == 1:
            self.action_space = spaces.Discrete(2)
        elif self.num_agents > 1:
            self.action_space = spaces.Tuple([spaces.Discrete(2) for _ in range(self.num_agents)])

        self.observation_space = spaces.Dict({})
        self.observation_space.spaces["state"] = spaces.Discrete(21)

        # rendering
        self.shared_viewer = shared_viewer
        self.viewer = None
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            reward_n.append(self._get_reward(agent))
        info_n = self._get_info(agent)
        obs_n = self._get_obs(self.agents)
        self.time += 1
        self.steps += 1
        done_n = self._get_done()
        truncated_n = False

        return obs_n, reward_n, done_n, truncated_n, info_n

    def reset(self, **kwargs):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        obs_n = self._get_obs(self.agents)
        # record info for each agent
        info_n = []
        info_n = self._get_info(self.agents)

        return obs_n, info_n
    
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(self.world)

    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(self.world)

    def _get_done(self):
        if self.done_callback is None:
            return False
        return self.done_callback(self.world)

    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent)

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


    def _set_action(self, action, agent):
        agent.action.u = action
    
    # reset rendering assets
    def _reset_render(self):
        self.render_stimuli_geoms = None

    ## TODO: Fix rendering
    # render environment
    def render(self, mode='human', close=False):
        from gymMDT import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.world)

        #create rendering geometry
        if self.render_stimuli_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            from gymMDT import rendering
            self.render_stimuli_geoms = []
            plus_geom = rendering.make_plus(self.world)
            self.render_stimuli_geoms.append(plus_geom)

            # add geoms to viewer
            self.viewer.geoms = []
            for stimuli_geom in self.render_stimuli_geoms:
                self.viewer.add_geom(stimuli_geom)


        current_state = self.world.states[self.world.s]

        self.render_item_geoms = []

        if self.world.s in self.state_list['S1']:
            state_geom = rendering.make_state(self.world)
            self.render_item_geoms.append(state_geom)

        stimuli_geoms = None
        if not current_state.state_number in self.state_list['S2']:
            if self.world.type == "geom":
                stimuli_geoms = rendering.make_stimuli(current_state, self.world)
        elif current_state.state_number in self.state_list['S2']:
            stimuli_geoms = rendering.make_result(current_state, self.world)

        if stimuli_geoms is not None:
            for stimuli_geom in stimuli_geoms:
                self.render_item_geoms.append(stimuli_geom)
        self.viewer.items = self.render_item_geoms
        
        results = self.viewer.render(return_rgb_array = mode=='rgb_array')
        return results

    #create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'grid'
        range_min = 0.05 * 20
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))

        return dx

# vectorized wrapper for a batch of environments
# assumes all environments have the same observation and action space
class BatchEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        i = 0
        for env in self.env_batch:
            obs, reward, done, info = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
            info_n += info
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
