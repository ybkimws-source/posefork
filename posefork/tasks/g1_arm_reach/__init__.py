import gymnasium as gym

from . import env_cfg, env_cfg_single
from .agents import ppo_cfg

gym.register(
    id="YB-G1-ArmReach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.G1ArmReachEnvCfg,
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ArmReachPPORunnerCfg",
    },
)

gym.register(
    id="YB-G1-ArmReach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.G1ArmReachEnvCfg_PLAY,
        "play_env_cfg_entry_point": env_cfg.G1ArmReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ArmReachPPORunnerCfg",
    },
)

gym.register(
    id="YB-G1-ArmReach-ModeHead-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.G1ArmReachEnvCfg,
        "play_env_cfg_entry_point": env_cfg.G1ArmReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ArmReachModeHeadPPORunnerCfg",
    },
)

# ── Single-mode learnability test ──────────────────────────────────────────────
gym.register(
    id="YB-G1-ArmReach-SingleA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_single.G1ArmReachSingleAEnvCfg,
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ArmReachSingleAPPORunnerCfg",
    },
)

gym.register(
    id="YB-G1-ArmReach-SingleB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_single.G1ArmReachSingleBEnvCfg,
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:G1ArmReachSingleBPPORunnerCfg",
    },
)
