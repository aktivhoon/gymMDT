import numpy as np
from gymMDT.agents.BaseAgent import BaseAgent, softmax
from gymMDT.agents.MFAgent import MFAgent
from gymMDT.agents.MBAgent import MBAgent
from collections import deque
from math import log, exp
from scipy.stats import logistic

"""Model-free and model-based arbitrator classes

Implemented based on the following publication: 
Neural Computations Underlying Arbitration between Model-Based and Model-free Learning 
http://dx.doi.org/10.1016/j.neuron.2013.11.028
"""

def logistic_cdf(x, mu=0, s=0.5):
    return 1 / (1 + np.exp(-(x - mu) / s))

def transition_rate_mf_to_mb(x, amp_mf_to_mb, mf_to_mb_bound):
        return amp_mf_to_mb / (1 + np.exp(mf_to_mb_bound * x))
    
def transition_rate_mb_to_mf(x, amp_mb_to_mf, mb_to_mf_bound):
    return amp_mb_to_mf / (1 + np.exp(mb_to_mf_bound * x))

class ArbAgent(BaseAgent):
    def __init__(self, agent_lr=0.2, beta=0.3,
                 zpe_threshold=0.5, rpe_lr=0.2,
                 mf_rel_estimator=None, mb_rel_estimator=None,
                 amp_mf_to_mb=3, amp_mb_to_mf=1, p_mb=0.8,
                 mf_to_mb_bound=0.01, mb_to_mf_bound=0.1):
        
        super(ArbAgent, self).__init__()
        self.s = 0
        self.lr = agent_lr
        self.transition_rate = 1.0
        if beta is None:
            self.beta = 0.3
        else:
            self.beta = beta

        self.mf_rel_estimator = mf_rel_estimator if mf_rel_estimator is not None else AssocRelEstimator(lr=rpe_lr)
        self.mb_rel_estimator = mb_rel_estimator if mb_rel_estimator is not None else BayesRelEstimator(threshold=zpe_threshold)

        self.A_alpha = amp_mf_to_mb
        self.A_beta = amp_mb_to_mf

        self.B_alpha = np.log((1 / mf_to_mb_bound) * self.A_alpha - 1)
        self.B_beta = np.log((1 / mb_to_mf_bound) * self.A_beta - 1)

        self.dt = 0.1

        x = np.linspace(0, 1, 1000)
        auc_mf_to_mb = np.trapz(transition_rate_mf_to_mb(x, amp_mf_to_mb, mf_to_mb_bound), x)
        auc_mb_to_mf = np.trapz(transition_rate_mb_to_mf(x, amp_mb_to_mf, mb_to_mf_bound), x)
        auc_ratio = auc_mf_to_mb / auc_mb_to_mf

        self.chi_mf_list = [0, 0]
        self.chi_mb_list = [0, 0]

        self.p_mb = logistic_cdf(np.log(auc_ratio), 0, 0.5)
        #self.p_mb = p_mb
        self.p_mf = 1 - self.p_mb
        self.mb_agent = MBAgent(lr=agent_lr, beta=beta)
        self.mf_agent = MFAgent(lr=agent_lr, beta=beta)

        #print(f"Initial p_mb: {self.p_mb}, p_mf: {self.p_mf}")

        self.Q = self.p_mb * self.mb_agent.Q + \
                    self.p_mf * self.mf_agent.Q
        self.policy = softmax(self.Q, self.beta)

    def update(self, s, a, a_o, R, s_next, a_next, a_o_next, R_next, env_reward, current_set, next_set, pretrain=False):
        if current_set is not None and next_set is not None and not pretrain:
            # print(f"Current Set: {current_set}, Next Set: {next_set}")
            # print(f"p_mb: {self.p_mb}, p_mf: {self.p_mf}")
            if current_set[0] == 'f' and next_set[0] in ['g', 'a']:
                target_p_mb = 0.8
                self.p_mb = target_p_mb
                #self.p_mb = 0.8#self.mb_rel_estimator.reliability
                #self.p_mb += (target_p_mb - self.p_mb) * self.transition_rate
                self.p_mf = 1 - self.p_mb
            elif current_set[0] in ['g', 'a'] and next_set[0] == 'f':
                target_p_mb = 0.2
                self.p_mb = target_p_mb
            #     #self.p_mb = 0.2#1 - self.mf_rel_estimator.reliability
            #     #self.p_mb += (target_p_mb - self.p_mb) * self.transition_rate
                self.p_mf = 1 - self.p_mb
            
            # print(f"-> p_mb: {self.p_mb}, p_mf: {self.p_mf}")

        # Model free RPE
        rpe1 = R - self.mf_agent.Q[s][a] + self.mf_agent.Q[s_next][a_next]
        rpe2 = R_next - self.mf_agent.Q[s_next][a_next]
        # print(f"RPE for R: {R} is {rpe1}")
        # print(f"RPE for R: {R_next} is {rpe2}")

        # Model based SPE
        spe1 = (1 - self.mb_agent.T[s][a][a_o])
        spe2 = (1 - self.mb_agent.T[s_next][a_next][a_o_next])

        # if not pretrain:
        #     print(f"T: {self.mb_agent.T[s][a][a_o]}")
        #     print(f"SPE1: {spe1}")
        #     print(f"updated T: {self.mb_agent.T[s][a][a_o] + self.lr * spe1}")
        #     print(f"T_next: {self.mb_agent.T[s_next][a_next][a_o_next]}")
        #     print(f"SPE2: {spe2}")
        #     print(f"updated T_next: {self.mb_agent.T[s_next][a_next][a_o_next] + self.lr * spe2}")

        if not pretrain:
            self._add_pe(rpe1, spe1)
            self._add_pe(rpe2, spe2)
        
        # print(f"reliability: {self.mf_rel_estimator.reliability}")
        # print('---')
        # print()

        # print(f"MFAgent: {self.mf_agent.Q}, MBAgent: {self.mb_agent.Q}")
        # print(f"Learned Transition: {self.mb_agent.T}")
        # print(f"rpe1: {abs(rpe1)/40}, spe1: {spe1}, rpe2: {abs(rpe2)/40}, spe2: {spe2}")
        # print("----")
        #self._add_pe(rpes, spes)

        # Model free agent update
        self.mf_agent.update(s, a, R, s_next, a_next, R_next)

        # Model based agent update
        self.mb_agent.update(s, a, a_o, s_next, a_next, a_o_next, env_reward, current_set, next_set)

        self.Q = self.p_mb * self.mb_agent.Q + \
                    self.p_mf * self.mf_agent.Q
        self.policy = softmax(self.Q, self.beta)

    def _add_pe(self, rpe, spe):
        chi_mf = self.mf_rel_estimator.add_pe(rpe)
        chi_mb = self.mb_rel_estimator.add_pe(spe)

        self.chi_mf_list.append(chi_mf)
        self.chi_mb_list.append(chi_mb)
        self.chi_mf_list.pop(0)
        self.chi_mb_list.pop(0)

        alpha = self.A_alpha / (1 + exp(self.B_alpha * (chi_mf)))
        sum_amp = self.A_alpha + self.A_beta
        #alpha /= sum_amp
        beta = self.A_beta / (1 + exp(self.B_beta * (chi_mb)))
        #beta /= sum_amp
        self.p_mb += (alpha * (1 - self.p_mb) - beta * self.p_mb) * self.dt
        
        self.p_mb = max(0, min(1, self.p_mb))
        self.p_mf = 1 - self.p_mb

        # print(f"A_alpha: {self.A_alpha}, A_beta: {self.A_beta}")
        # if self.A_alpha > self.A_beta:
        #     print(f"Model Based biased")
        # else:
        #     print(f"Model Free biased")
        # print(f"B_alpha: {self.B_alpha}, B_beta: {self.B_beta}")
        # print(f"chi_mf: {chi_mf}, chi_mb: {chi_mb}")
        # print(f"alpha: {alpha}, beta: {beta}, infty: {alpha/(alpha+beta)} p_mb: {self.p_mb}")

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
    def __init__(self, lr=0.2, pe_max=40):
        self.chi = 0.5
        self.lr = lr
        self.pe_max = pe_max
        self.delta_chi1 = 0
        self.delta_chi2 = 0

    def add_pe(self, pe, rel_calc=True):
        """
        if not rel_calc:
            self.delta_chi1 = self.lr * ((1 - abs(pe) / self.pe_max) - self.chi)
        else:
            self.delta_chi2 = self.lr * ((1 - abs(pe) / self.pe_max) - self.chi)
            self.chi += (self.delta_chi1 + self.delta_chi2)
        
            return self.chi
        """
        delta_chi = self.lr * ((1 - abs(pe) / self.pe_max) - self.chi)
        self.chi += delta_chi
        return self.chi

    @property
    def reliability(self):
        return self.chi
    
