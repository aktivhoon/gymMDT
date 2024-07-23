import numpy as np
from gymMDT.agents.MFAgent import MFAgent
from gymMDT.agents.MBAgent import MBAgent
from gymMDT.agents.ArbAgent import ArbAgent
from gymMDT.agents.ScriptedAgent import ScriptedAgent

class PlayerAgent(object):
    def __init__(self, env, variable=True, select_class=0,
                        agent_lr=None, beta=None,
                        zpe_threshold=0.3, rpe_lr=0.2,
                        mf_rel_estimator=None, mb_rel_estimator=None,
                        amp_mf_to_mb=3, amp_mb_to_mf=1, p_mb=0.8,
                        mf_to_mb_bound=0.1, mb_to_mf_bound=0.01):
        # name
        self.env = env
        self.action_callback = None
        self.select_class = select_class
        if self.select_class == 0:
            self.policy_agent = MFAgent(lr=agent_lr, beta=beta, variable=variable)
        elif self.select_class == 1:
            self.policy_agent = MBAgent(self.env, lr=agent_lr, beta=beta, variable=variable)
        elif self.select_class == 2:
            self.policy_agent = ArbAgent(self.env,
                                         agent_lr=agent_lr, beta=beta,
                                         zpe_threshold = zpe_threshold, rpe_lr=rpe_lr,
                                         mf_rel_estimator=mf_rel_estimator,
                                         mb_rel_estimator=mb_rel_estimator,
                                         amp_mf_to_mb=amp_mf_to_mb, amp_mb_to_mf=amp_mb_to_mf,
                                         mf_to_mb_bound=mf_to_mb_bound, mb_to_mf_bound=mb_to_mf_bound)
        elif self.select_class == 3:
            self.policy_agent = ScriptedAgent()

    def action(self, state):
        return self.policy_agent.choose_action(state)

    def update(self, trajectory, task_setting=None, rpe_prev=None, spe_prev=None, rpe=None, spe=None):
        s0 = trajectory.s0
        s1 = trajectory.s1
        if self.select_class == 0:
            a1 = trajectory.a1
            r1 = trajectory.r1
            a2 = trajectory.a2
            r2 = trajectory.r2
            self.policy_agent.update(s0, a1, r1, s1, a2, r2)
        elif self.select_class == 1:
            a1  = trajectory.a1
            a1o = trajectory.a1o
            a2  = trajectory.a2
            a2o = trajectory.a2o
            self.policy_agent.update(s0, a1, a1o, s1, a2, a2o, task_setting)
        elif self.select_class == 2:
            a1 = trajectory.a1
            a1o = trajectory.a1o
            r1 = trajectory.r1
            a2 = trajectory.a2
            a2o = trajectory.a2o
            r2 = trajectory.r2
            self.policy_agent.update(s0, a1, a1o, r1, s1, a2, a2o, r2)