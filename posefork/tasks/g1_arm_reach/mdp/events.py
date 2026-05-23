from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lock_waist_joints(env: ManagerBasedRLEnv, env_ids) -> None:
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)

    waist_ids, _ = robot.find_joints(
        ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    )
    waist_ids_tensor = torch.tensor(waist_ids, device=robot.device)

    eps = 1e-4
    limits = torch.zeros(len(env_ids), len(waist_ids), 2, device=robot.device)
    limits[..., 0] = -eps
    limits[..., 1] = eps

    robot.write_joint_position_limit_to_sim(
        limits, joint_ids=waist_ids_tensor, env_ids=env_ids
    )
