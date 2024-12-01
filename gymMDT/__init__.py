from gym.envs.registration import register

register(
    id='MDTGeneralReward-v0',
    entry_point='gymMDT.scenarios:GeneralRewardEnv',
)

register(
    id='MDTBlockTask-v0',
    entry_point='gymMDT.scenarios:BlockTaskEnv',
)

register(
    id='MDTBlockReward-v0',
    entry_point='gymMDT.scenarios:BlockRewardEnv',
)

register(
    id='MDTBlockRewShift-v0',
    entry_point='gymMDT.scenarios:BlockRewShiftEnv',
)

register(
    id='MDTAdaptiveReward-v0',
    entry_point='gymMDT.scenarios:AdaptiveRewardEnv',
)

register(
    id='MDTAdaptiveGeneralReward-v0',
    entry_point='gymMDT.scenarios:AdaptiveGeneralRewardEnv',
)

register(
    id='MDTGlascher-v0',
    entry_point='gymMDT.scenarios:GlascherEnv',
)

register(
    id='MDTLee-v0',
    entry_point='gymMDT.scenarios:LeeEnv',
)

register(
    id='MDTLeeModified-v0',
    entry_point='gymMDT.scenarios:LeeModifiedEnv',
)