import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
from unitree_rl_lab.tasks.manipulation import mdp

# G1 right arm joints (7 DOF)
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
EE_LINK = "right_wrist_yaw_link"


@configclass
class ArmReachSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        self.robot.spawn.articulation_props.fix_root_link = True


@configclass
class ActionsCfg:
    arm_joints = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        scale=0.1,
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    ee_target = UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_LINK,
        resampling_time_range=(5.0, 10.0),
        ranges=UniformPoseCommandCfg.Ranges(
            # base frame relative to pelvis(root)
            # arm reach: shoulder~(0.004,-0.100,0.248), arm length~0.37m
            # x: forward, max ~0.35m (shoulder x=0.004 + arm)
            # y: negative = right side, shoulder y=-0.100
            # z: shoulder z=0.248 ± arm
            pos_x=(0.15, 0.35),
            pos_y=(-0.45, -0.05),
            pos_z=(0.05, 0.45),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # target (3)
        target = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "ee_target"},
        )
        # joint state (7 + 7)
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
            noise=GaussianNoiseCfg(std=0.01),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
            scale=0.05,
            noise=GaussianNoiseCfg(std=0.5),
        )
        # EE state (3)
        ee_pos = ObsTerm(
            func=mdp.ee_pos_w,
            params={"link_name": EE_LINK},
        )
        # relative vector target→EE (3)
        ee_to_target = ObsTerm(
            func=mdp.ee_to_target_vec,
            params={"link_name": EE_LINK, "command_name": "ee_target"},
        )
        # distance (1)
        distance = ObsTerm(
            func=mdp.ee_dist,
            params={"link_name": EE_LINK, "command_name": "ee_target"},
        )
        # last action (7)
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # ── task rewards ──
    tracking = RewTerm(
        func=mdp.tracking_ee_pos,
        weight=1.0,
        params={"command_name": "ee_target", "link_name": EE_LINK, "std": 0.05},
    )
    progress = RewTerm(
        func=mdp.progress_ee_pos,
        weight=1.0,
        params={"command_name": "ee_target", "link_name": EE_LINK, "scale": 5.0},
    )
    precision = RewTerm(
        func=mdp.precision_bonus,
        weight=1.0,
        params={"command_name": "ee_target", "link_name": EE_LINK, "std": 0.0125, "bonus": 2.0},
    )

    # ── IK teacher ──
    ik_teacher = RewTerm(
        func=mdp.ik_teacher_joint_deviation,
        weight=0.3,
        params={
            "command_name": "ee_target",
            "ee_link_name": EE_LINK,
            "joint_names": RIGHT_ARM_JOINTS,
        },
    )

    # ── regularization ──
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.05)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
    )
    joint_pos_limits = RewTerm(
        func=base_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    # velocity explosion termination
    joint_vel_explosion = DoneTerm(
        func=base_mdp.joint_vel_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
    )


@configclass
class G1ArmReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: ArmReachSceneCfg = ArmReachSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None
    events = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation


@configclass
class G1ArmReachEnvCfg_PLAY(G1ArmReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0
