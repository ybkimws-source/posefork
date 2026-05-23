"""
Single-mode arm_angle learnability ablation.

Tests whether the phi teacher can drive arm_angle to a target phi without
interference from multi-mode shared-policy conflict.

Run A (phi=-1.15): YB-G1-ArmReach-SingleA-v0
Run B (phi=-0.80): YB-G1-ArmReach-SingleB-v0
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from posefork.tasks.g1_arm_reach.env_cfg import (
    G1ArmReachEnvCfg,
    G1ArmReachEnvCfg_PLAY,
    RewardsCfg,
    ObservationsCfg,
    EE_LINK,
)
import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from posefork.tasks.g1_arm_reach import mdp


@configclass
class SingleModeObsCfg(ObservationsCfg):
    """mode_id fixed to num_modes=1 (always [1.0] — constant bias in practice)."""
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        mode_id = ObsTerm(
            func=mdp.mode_id_one_hot,
            params={"num_modes": 1},
        )
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class ModeCfg(ObservationsCfg.ModeCfg):
        mode_id = ObsTerm(
            func=mdp.mode_id_one_hot,
            params={"num_modes": 1},
        )

    mode: ModeCfg = ModeCfg()


@configclass
class SingleModeRewardsCfg_A(RewardsCfg):
    """phi_k = -1.15 (extreme negative). teacher=1.0, tracking=0.5."""
    tracking = RewTerm(
        func=mdp.tracking_ee_pos,
        weight=0.5,
        params={"command_name": "ee_target", "link_name": EE_LINK, "std": 0.2},
    )
    arm_angle_teacher = RewTerm(
        func=mdp.arm_angle_mode_teacher,
        weight=1.0,
        params={
            "command_name": "ee_target",
            "ee_link_name": EE_LINK,
            "phi_k_override": [-1.15],
        },
    )


@configclass
class SingleModeRewardsCfg_B(RewardsCfg):
    """phi_k = -0.80 (extreme negative). teacher=1.0, tracking=0.5."""
    tracking = RewTerm(
        func=mdp.tracking_ee_pos,
        weight=0.5,
        params={"command_name": "ee_target", "link_name": EE_LINK, "std": 0.2},
    )
    arm_angle_teacher = RewTerm(
        func=mdp.arm_angle_mode_teacher,
        weight=1.0,
        params={
            "command_name": "ee_target",
            "ee_link_name": EE_LINK,
            "phi_k_override": [-0.80],
        },
    )


@configclass
class G1ArmReachSingleAEnvCfg(G1ArmReachEnvCfg):
    """Single-mode, phi_k = -1.15."""
    observations: SingleModeObsCfg = SingleModeObsCfg()
    rewards: SingleModeRewardsCfg_A = SingleModeRewardsCfg_A()


@configclass
class G1ArmReachSingleBEnvCfg(G1ArmReachEnvCfg):
    """Single-mode, phi_k = -0.80."""
    observations: SingleModeObsCfg = SingleModeObsCfg()
    rewards: SingleModeRewardsCfg_B = SingleModeRewardsCfg_B()


@configclass
class G1ArmReachSingleAEnvCfg_PLAY(G1ArmReachSingleAEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0


@configclass
class G1ArmReachSingleBEnvCfg_PLAY(G1ArmReachSingleBEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0
