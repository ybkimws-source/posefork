from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tcp_pos_w(
    env: ManagerBasedRLEnv,
    link_name: str,
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """TCP position in world frame from a link origin plus a fixed local-frame offset."""
    robot: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_yb_body_idx_cache"):
        env._yb_body_idx_cache = {}
    cache_key = (asset_cfg.name, link_name)
    if cache_key not in env._yb_body_idx_cache:
        env._yb_body_idx_cache[cache_key] = robot.find_bodies(link_name)[0][0]
    ee_idx = env._yb_body_idx_cache[cache_key]
    link_pos_w = robot.data.body_pos_w[:, ee_idx, :3]
    link_quat_w = robot.data.body_quat_w[:, ee_idx, :]
    offset_b = torch.tensor(tcp_offset, device=robot.device, dtype=link_pos_w.dtype).unsqueeze(0).expand_as(link_pos_w)
    return link_pos_w + quat_apply(link_quat_w, offset_b)


def ee_pos_b(
    env: ManagerBasedRLEnv,
    link_name: str,
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """TCP position in robot base (root-relative) frame. NOT world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return tcp_pos_w(env, link_name, tcp_offset, asset_cfg) - robot.data.root_pos_w


def ee_to_target_vec(
    env: ManagerBasedRLEnv,
    link_name: str,
    command_name: str,
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from EE to target (target - EE), root-relative translation. NOT orientation-corrected."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_pos = tcp_pos_w(env, link_name, tcp_offset, asset_cfg) - robot.data.root_pos_w
    target = env.command_manager.get_command(command_name)[:, :3]
    return target - ee_pos


def ee_dist(
    env: ManagerBasedRLEnv,
    link_name: str,
    command_name: str,
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Scalar EE-to-target distance in root-relative frame. Shape: (N, 1). NOT orientation-corrected."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_pos = tcp_pos_w(env, link_name, tcp_offset, asset_cfg) - robot.data.root_pos_w
    target = env.command_manager.get_command(command_name)[:, :3]
    return torch.norm(target - ee_pos, dim=-1, keepdim=True)


def mode_id_one_hot(
    env: ManagerBasedRLEnv,
    num_modes: int = 3,
) -> torch.Tensor:
    """
    Mode id as one-hot vector. Shape: (N, num_modes).
    Mode is assigned as env_index % num_modes (fixed for env lifetime).
    Must be consistent with _get_or_init_mode_ids() in rewards.py.
    """
    if (not hasattr(env, "_yb_mode_ids")) or (getattr(env, "_yb_mode_ids_num_modes", None) != num_modes):
        env._yb_mode_ids = torch.arange(env.num_envs, device=env.device) % num_modes
        env._yb_mode_ids_num_modes = num_modes
    one_hot = torch.zeros(env.num_envs, num_modes, device=env.device)
    one_hot.scatter_(1, env._yb_mode_ids.unsqueeze(1), 1.0)
    return one_hot
