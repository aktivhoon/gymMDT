import numpy as np
from gymMDT.agents.BaseAgent import BaseAgent, softmax

class ScriptedAgent(BaseAgent):
    def __init__(self):
        super(ScriptedAgent, self).__init__()

    def choose_action(self, s):
        return 0
