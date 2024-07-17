from gym.envs.registration import register

register(
    id='MDTBlockReward-v0',
    entry_point='gymMDT.scenarios:BlockRewardEnv',
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