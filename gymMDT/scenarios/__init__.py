from importlib.machinery import SourceFileLoader
import os.path as osp

from gymMDT.scenarios.general_rew import GeneralRewardEnv
from gymMDT.scenarios.block_task import BlockTaskEnv
from gymMDT.scenarios.block_rew import BlockRewardEnv
from gymMDT.scenarios.block_rew_shift import BlockRewShiftEnv
from gymMDT.scenarios.adapt_rew import AdaptiveRewardEnv
from gymMDT.scenarios.adapt_genrew import AdaptiveGeneralRewardEnv
from gymMDT.scenarios.glascher import GlascherEnv
from gymMDT.scenarios.lee import LeeEnv
from gymMDT.scenarios.lee_modified import LeeModifiedEnv

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    mymodule = SourceFileLoader('', pathname).load_module()
    return mymodule
