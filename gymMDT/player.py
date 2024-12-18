import numpy as np
from gymMDT.agents.MFAgent import MFAgent
from gymMDT.agents.MBAgent import MBAgent
from gymMDT.agents.AdvMBAgent import AdvMBAgent
from gymMDT.agents.StaticArbAgent import StaticArbAgent
from gymMDT.agents.ArbAgent import ArbAgent
from gymMDT.agents.AdvArbAgent import AdvArbAgent
from gymMDT.agents.ScriptedAgent import ScriptedAgent

class PlayerAgent(object):
    def __init__(self, variable=True, select_class=None,
                        agent_lr=None, beta=None,
                        zpe_threshold=0.3, rpe_lr=0.2,
                        mf_rel_estimator=None, mb_rel_estimator=None,
                        amp_mf_to_mb=3, amp_mb_to_mf=1, p_mb=0.8,
                        mf_to_mb_bound=0.01, mb_to_mf_bound=0.1):
        # name
        self.action_callback = None
        self.select_class = select_class
        if select_class is None:
            raise ValueError("select_class is None")
        elif self.select_class == 'f':
            self.policy_agent = MFAgent(lr=agent_lr, beta=beta, variable=variable)
        elif self.select_class == 'b':
            self.policy_agent = MBAgent(lr=agent_lr, beta=beta, variable=variable)
        elif self.select_class == 'adv-b':
            self.policy_agent = AdvMBAgent(lr=agent_lr, beta=beta, variable=variable)
        elif self.select_class == 'static-arb':
            self.policy_agent = StaticArbAgent(agent_lr=agent_lr, beta=beta,
                                         amp_mf_to_mb=amp_mf_to_mb, amp_mb_to_mf=amp_mb_to_mf,
                                         mf_to_mb_bound=mf_to_mb_bound, mb_to_mf_bound=mb_to_mf_bound)
        elif self.select_class == 'arb':
            self.policy_agent = ArbAgent(agent_lr=agent_lr, beta=beta,
                                         zpe_threshold = zpe_threshold, rpe_lr=rpe_lr,
                                         mf_rel_estimator=mf_rel_estimator,
                                         mb_rel_estimator=mb_rel_estimator,
                                         amp_mf_to_mb=amp_mf_to_mb, amp_mb_to_mf=amp_mb_to_mf,
                                         mf_to_mb_bound=mf_to_mb_bound, mb_to_mf_bound=mb_to_mf_bound)
        elif self.select_class == 'adv-arb':
            self.policy_agent = AdvArbAgent(agent_lr=agent_lr, beta=beta,
                                         zpe_threshold = zpe_threshold, rpe_lr=rpe_lr,
                                         mf_rel_estimator=mf_rel_estimator,
                                         mb_rel_estimator=mb_rel_estimator,
                                         amp_mf_to_mb=amp_mf_to_mb, amp_mb_to_mf=amp_mb_to_mf,
                                         mf_to_mb_bound=mf_to_mb_bound, mb_to_mf_bound=mb_to_mf_bound)
        elif self.select_class == 'script':
            self.policy_agent = ScriptedAgent()
        else:
            raise ValueError("Invalid select_class")

    def action(self, state):
        return self.policy_agent.choose_action(state)

    def update(self, trajectory, pretrain=False):
        s0 = trajectory.s0
        s1 = trajectory.s1
        if self.select_class == 'f':
            a1 = trajectory.a1
            r1 = trajectory.r1
            a2 = trajectory.a2
            r2 = trajectory.r2
            self.policy_agent.update(s0, a1, r1, s1, a2, r2)
        elif self.select_class == 'b':
            a1  = trajectory.a1
            a1o = trajectory.a1o
            a2  = trajectory.a2
            a2o = trajectory.a2o
            env_reward = trajectory.env_reward
            cs = trajectory.cs
            ns = trajectory.ns
            self.policy_agent.update(s0, a1, a1o, s1, a2, a2o, env_reward, cs, ns)
        elif self.select_class == 'adv-b':
            a1 = trajectory.a1
            a1o = trajectory.a1o
            r1 = trajectory.r1
            a2 = trajectory.a2
            a2o = trajectory.a2o
            r2 = trajectory.r2
            env_reward = trajectory.env_reward
            cs = trajectory.cs
            ns = trajectory.ns
            self.policy_agent.update(s0, a1, a1o, r1, s1, a2, a2o, r2, env_reward, cs, ns)
        elif self.select_class in ['arb', 'adv-arb', 'static-arb']:
            a1 = trajectory.a1
            a1o = trajectory.a1o
            r1 = trajectory.r1
            a2 = trajectory.a2
            a2o = trajectory.a2o
            r2 = trajectory.r2
            env_reward = trajectory.env_reward
            cs = trajectory.cs
            ns = trajectory.ns
            self.policy_agent.update(s0, a1, a1o, r1, s1, a2, a2o, r2, env_reward, cs, ns, pretrain)