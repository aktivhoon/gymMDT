from importlib.machinery import SourceFileLoader
import os.path as osp

from gymMDT.scenarios.block_task import BlockTaskEnv
from gymMDT.scenarios.block_rew import BlockRewardEnv
from gymMDT.scenarios.glascher import GlascherEnv
from gymMDT.scenarios.lee import LeeEnv
from gymMDT.scenarios.lee_modified import LeeModifiedEnv

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    mymodule = SourceFileLoader('', pathname).load_module()
    return mymodule
