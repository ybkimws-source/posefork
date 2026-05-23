from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ──────────────────────────────────────────────
# EE distance rewards
# ──────────────────────────────────────────────

def tracking_ee_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    link_name: str,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-dist/std). std=0.05 equivalent to exp(-20*dist)."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(link_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, ee_idx, :3] - robot.data.root_pos_w
    target = env.command_manager.get_command(command_name)[:, :3]
    dist = torch.norm(target - ee_pos, dim=-1)
    return torch.exp(-dist / std)


def progress_ee_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    link_name: str,
    scale: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positive when closing in, negative when moving away."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(link_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, ee_idx, :3] - robot.data.root_pos_w
    target = env.command_manager.get_command(command_name)[:, :3]
    dist = torch.norm(target - ee_pos, dim=-1)

    # lazy init for previous distance buffer
    if not hasattr(env, "_arm_reach_last_dist"):
        env._arm_reach_last_dist = dist.clone()

    progress = env._arm_reach_last_dist - dist
    env._arm_reach_last_dist[:] = dist
    return progress * scale


def precision_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    link_name: str,
    std: float = 0.0125,
    bonus: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Close-range bonus. std=0.0125 equivalent to exp(-80*dist)*2."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(link_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, ee_idx, :3] - robot.data.root_pos_w
    target = env.command_manager.get_command(command_name)[:, :3]
    dist = torch.norm(target - ee_pos, dim=-1)
    return torch.exp(-dist / std) * bonus


# ──────────────────────────────────────────────
# IK teacher reward
# ──────────────────────────────────────────────

def ik_teacher_joint_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_link_name: str,
    joint_names: list[str],
    weight: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Compute q_ik online via DifferentialIK each step and penalize deviation from it.
    State-conditioned teacher (unlike offline static lookup tables).
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # IK controller lazy init
    if not hasattr(env, "_ik_teacher"):
        cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
        )
        env._ik_teacher = DifferentialIKController(cfg, env.num_envs, env.device)
        env._ik_teacher_joint_ids, _ = robot.find_joints(joint_names)
        env._ik_teacher_ee_idx = robot.find_bodies(ee_link_name)[0][0]
        # fixed-base: root excluded, so jacobian_body_idx = body_idx - 1
        env._ik_teacher_jacobian_body_idx = env._ik_teacher_ee_idx - 1

    ee_idx = env._ik_teacher_ee_idx
    joint_ids = env._ik_teacher_joint_ids

    ee_pos = robot.data.body_pos_w[:, ee_idx, :3]
    ee_quat = robot.data.body_quat_w[:, ee_idx, :]
    joint_pos = robot.data.joint_pos[:, joint_ids]

    # Jacobian: (N, num_bodies, 6, num_dofs) — 4D in Isaac Sim 5.x
    # fixed-base: jacobian_body_idx = body_idx - 1 (root excluded)
    jacobian = robot.root_physx_view.get_jacobians()[:, env._ik_teacher_jacobian_body_idx, :, :]
    jacobian = jacobian[:, :, joint_ids]  # (N, 6, n_joints)

    # command is in base frame → convert to world frame for IK controller
    target_b = env.command_manager.get_command(command_name)[:, :3]
    target_w = target_b + robot.data.root_pos_w
    env._ik_teacher.set_command(target_w, ee_pos=ee_pos, ee_quat=ee_quat)

    # compute() returns q_teacher = joint_pos + delta_q
    q_teacher = env._ik_teacher.compute(ee_pos, ee_quat, jacobian, joint_pos)  # (N, n_joints)

    deviation = torch.norm(joint_pos - q_teacher, dim=-1)
    return -deviation * weight


# ──────────────────────────────────────────────
# regularization
# ──────────────────────────────────────────────

def action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Action smoothness penalty (stub — use base_mdp.action_rate_l2 instead)."""
    last2 = env.action_manager.prev_action if hasattr(env.action_manager, "prev_action") else torch.zeros_like(env.action_manager.action)
    last1 = env.action_manager.action
    # action history is properly accessed via observations in IsaacLab
    # replaced by action_rate_l2 from base_mdp
    return torch.zeros(env.num_envs, device=env.device)
