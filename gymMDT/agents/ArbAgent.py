import numpy as np
from gymMDT.agents.BaseAgent import BaseAgent, softmax
from gymMDT.agents.MFAgent import MFAgent
from gymMDT.agents.MBAgent import MBAgent
from collections import deque
from math import log, exp

"""Model-free and model-based arbitrator classes

Implemented based on the following publication: 
Neural Computations Underlying Arbitration between Model-Based and Model-free Learning 
http://dx.doi.org/10.1016/j.neuron.2013.11.028
"""

class ArbAgent(BaseAgent):
    def __init__(self, env, agent_lr=0.2, beta=0.3,
                 zpe_threshold=0.5, rpe_lr=0.2,
                 mf_rel_estimator=None, mb_rel_estimator=None,
                 amp_mf_to_mb=3, amp_mb_to_mf=1, p_mb=0.8,
                 mf_to_mb_bound=0.1, mb_to_mf_bound=0.01):
        
        super(ArbAgent, self).__init__()
        self.s = 0
        self.lr = agent_lr
        if beta is None:
            self.beta = 0.3
        else:
            self.beta = beta
        self.env = env

        self.mf_rel_estimator = mf_rel_estimator if mf_rel_estimator is not None else AssocRelEstimator(lr=rpe_lr)
        self.mb_rel_estimator = mb_rel_estimator if mb_rel_estimator is not None else BayesRelEstimator(threshold=zpe_threshold)

        self.A_alpha = amp_mf_to_mb
        self.A_beta = amp_mb_to_mf

        self.amp_mb_to_mf = amp_mb_to_mf
        self.amp_mf_to_mb = amp_mf_to_mb

        self.B_alpha = np.log((1 / mf_to_mb_bound) * amp_mb_to_mf - 1)
        self.B_beta = np.log((1 / mb_to_mf_bound) * amp_mf_to_mb - 1)

        self.p_mb = p_mb
        self.p_mf = 1 - self.p_mb
        self.mb_agent = MBAgent(self.env, lr=agent_lr, beta=beta)
        self.mf_agent = MFAgent(lr=agent_lr, beta=beta)

        # For two-player setting opponent agent
        self.rpe = 0
        self.spe = 0

    def update(self, s, a, a_o, R, s_next, a_next, a_o_next, R_next, task_setting=None):
        # Model free RPE
        rpe = 0.0
        rpe += abs(R - self.mf_agent.Q[s][a] + self.mf_agent.Q[s_next][a_next])
        rpe += abs(R_next - self.mf_agent.Q[s_next][a_next])
        # Model free agent update
        self.mf_agent.update(s, a, R, s_next, a_next, R_next)

        # Model based SPE
        spe = 0.0
        spe += (1 - self.mb_agent.T[s][a][a_o])
        spe += (1 - self.mb_agent.T[s_next][a_next][a_o_next])
        # Model based agent update
        self.mb_agent.update(s, a, a_o, s_next, a_next, a_o_next, task_setting)

        self.rpe = rpe
        self.spe = spe
        self._add_pe(rpe, spe)

        self.Q = self.p_mb * self.mb_agent.Q + \
                    self.p_mf * self.mf_agent.Q
        self.policy = softmax(self.Q, self.beta)

    def _get_Reward(self, state, action):
        R_array = np.zeros(2)
        next_states = self.env.get_next_states(state, action)
        for i, _states in enumerate(next_states):
            _s = _states
            _item = self.env.world.states[_s].item
            if _item is None:
                R_array[i] = 0
            else:
                R_array[i] = _item.reward

        return R_array

    def _add_pe(self, rpe, spe):
        chi_mf = self.mf_rel_estimator.add_pe(rpe)
        chi_mb = self.mb_rel_estimator.add_pe(spe)
        alpha = self.A_alpha / (1 + exp(self.B_alpha * chi_mf))
        sum_amp = self.amp_mb_to_mf + self.amp_mf_to_mb
        alpha /= sum_amp
        beta = self.A_beta / (1 + exp(self.B_beta * chi_mb))
        beta /= sum_amp
        self.p_mb += alpha * (1 - self.p_mb) - beta * self.p_mb
        self.p_mf = 1 - self.p_mb
        return chi_mf, chi_mb, self.p_mb

class BayesRelEstimator:
    DEFAULT_COND_PROB_DIST_FUNC_TEMPLATE = (lambda threshold: 
        (lambda pe:
            1 if pe < -threshold else
            0 if pe < threshold else
            2
        )
    )
    def __init__(self, memory_size=20, categories=2, threshold=0.5,
                 cond_prob_dist_func=None, target_category=0):

        """
        Args:
            memory_size (int): maximum length of memory, which is the 'T' discrete events
            appeared in the paper
            categories (int): number of categories of prediction errors 
            (negative, zero, positive by default), which is the 'K' parameter in Dirichlet Distribution
            thereshold (float): thereshold for the default three categories, no effect if customized 
            condition probability distribution function provided
            cond_prob_dist_func (closure): a function to separate continuous prediction error
            into discrete categories. the number of categories should match to the categories argument
            If given None, default function will be used
            target_category (int): when calculate reliability, we need to know the target category to
            calculate, in default case it is 0, as appeared on the paper
        
        Construct a rolling container for historic data using deque, use another counter countainer with
        size of categories to cache the number of each category
        """
        self.categories = categories
        self.pe_records_count = np.zeros(self.categories)
        self.pe_records = deque(maxlen=memory_size)
        self.target_category = target_category
        self.cond_prob_dist_func = cond_prob_dist_func
        if self.cond_prob_dist_func is None:
            self.cond_prob_dist_func = (lambda pe:
                        0 if pe < threshold else
                        1
                    )
    
    def add_pe(self, pe, rel_calc=True):
        if len(self.pe_records) == self.pe_records.maxlen:
            self.pe_records_count[self.pe_records[0]] -= 1
        pe_category = self.cond_prob_dist_func(pe)
        self.pe_records.append(pe_category)
        self.pe_records_count[pe_category] += 1
        if rel_calc:
            return self.reliability

    # cardinality of D
    @property
    def cardinal(self):
        return len(self.pe_records)

    def _dirichlet_mean(self, category):
        return (1+ self.pe_records_count[category]) / \
                (self.categories + self.cardinal)

    def _dirichlet_var(self, category):
        return ((1 + self.pe_records_count[category])) * \
                (self.categories + self.cardinal - (1 + self.pe_records_count[category])) / \
               (pow((self.categories + self.cardinal), 2) * \
                (self.categories + self.cardinal + 1))

    @property
    def reliability(self):
        chi = []
        for category in range(self.categories):
            mean = self._dirichlet_mean(category)
            var = self._dirichlet_var(category)
            chi.append(mean / var)
        return chi[self.target_category] / sum(chi)

    def set_reliability(self, input_categories):
        self.categories = input_categories

class AssocRelEstimator:
    def __init__(self, lr=0.2, pe_max=30):
        self.chi = 0
        self.lr = lr
        self.pe_max = pe_max

    def add_pe(self, pe):
        delta_chi = self.lr * ((1 - abs(pe) / self.pe_max) - self.chi)
        self.chi += delta_chi
        return self.chi

    @property
    def reliability(self):
        return self.chi
    
