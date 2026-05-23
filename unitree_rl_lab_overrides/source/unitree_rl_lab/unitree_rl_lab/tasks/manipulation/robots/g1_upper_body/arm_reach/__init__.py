import gymnasium as gym

from . import arm_reach_env_cfg
from unitree_rl_lab.tasks.manipulation.agents import rsl_rl_ppo_cfg

gym.register(
    id="Unitree-G1-ArmReach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": arm_reach_env_cfg.G1ArmReachEnvCfg,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1ArmReachPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-ArmReach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": arm_reach_env_cfg.G1ArmReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{rsl_rl_ppo_cfg.__name__}:G1ArmReachPPORunnerCfg",
    },
)
