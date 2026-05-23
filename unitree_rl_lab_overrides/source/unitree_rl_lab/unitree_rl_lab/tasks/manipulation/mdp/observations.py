from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pos_w(env: ManagerBasedRLEnv, link_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """EE position relative to env origin."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(link_name)[0][0]
    return robot.data.body_pos_w[:, ee_idx, :3] - robot.data.root_pos_w


def ee_to_target_vec(env: ManagerBasedRLEnv, link_name: str, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vector from EE to target."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(link_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, ee_idx, :3] - robot.data.root_pos_w
    target = env.command_manager.get_command(command_name)[:, :3]
    return target - ee_pos


def ee_dist(env: ManagerBasedRLEnv, link_name: str, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """EE-to-target distance scalar, shape (N, 1)."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(link_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, ee_idx, :3] - robot.data.root_pos_w
    target = env.command_manager.get_command(command_name)[:, :3]
    return torch.norm(target - ee_pos, dim=-1, keepdim=True)
