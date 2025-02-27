from gym.envs.registration import register

register(
    id='RandomNSMDP-v0',
    entry_point='dyna_gym.envs:RandomNSMDP',
)

register(
    id='NSFrozenLake-v0',
    entry_point='dyna_gym.envs:NSFrozenLakeV0',
)

register(
    id='NSFrozenLake-v1',
    entry_point='dyna_gym.envs:NSFrozenLakeV1',
)

register(
    id='NSFrozenLake-v2',
    entry_point='dyna_gym.envs:NSFrozenLakeV2',
)

register(
    id='NSCliff-v0',
    entry_point='dyna_gym.envs:NSCliffV0',
)

register(
    id='NSCliff-v1',
    entry_point='dyna_gym.envs:NSCliffV1',
)

register(
    id='NSCliff-v2',
    entry_point='dyna_gym.envs:NSCliffV2',
)

register(
    id='NSBridge-v0',
    entry_point='dyna_gym.envs:NSBridgeV0',
)

register(
    id='NSBridge-v1',
    entry_point='dyna_gym.envs:NSBridgeV1',
)

register(
    id='NSBridge-v2',
    entry_point='dyna_gym.envs:NSBridgeV2',
)

register(
    id='NSCartPole-v0',
    entry_point='dyna_gym.envs:NSCartPoleV0',
)

register(
    id='NSCartPole-v1',
    entry_point='dyna_gym.envs:NSCartPoleV1',
)

register(
    id='NSCartPole-v2',
    entry_point='dyna_gym.envs:NSCartPoleV2',
)
