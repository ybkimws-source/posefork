"""Visualize G1 arm reach: target marker + IK solution + zero-action stepping.

Run:
  conda run -n env_isaaclab python scripts/rsl_rl/visualize_arm_reach.py \
    --task Unitree-G1-ArmReach-Play-v0

Shows:
  - Green axis frame: target EE pose (UniformPoseCommand marker)
  - Blue axis frame: current EE pose
  - Terminal: IK teacher joint solution vs current joints
"""

import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-G1-ArmReach-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.commands.ee_target.debug_vis = True

env = gym.make(args_cli.task, cfg=env_cfg)
obs, _ = env.reset()

robot = env.unwrapped.scene["robot"]
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

joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINTS)
ee_idx = robot.find_bodies(EE_LINK)[0][0]
ee_jacobi_idx = ee_idx - 1  # fixed-base: skip root

print("\n=== G1 Arm Reach Visualization ===")
print(f"Joint IDs in robot.data.joint_pos: {joint_ids}")
print(f"EE body index: {ee_idx}, Jacobian body index: {ee_jacobi_idx}")
print(f"Root pos (pelvis height from world origin): {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"EE pos world: {robot.data.body_pos_w[0, ee_idx, :3].cpu().numpy()}")
print(f"EE pos base frame: {(robot.data.body_pos_w[0, ee_idx, :3] - robot.data.root_pos_w[0]).cpu().numpy()}")
print("=" * 50)

step = 0
while simulation_app.is_running():
    with torch.inference_mode():
        # zero actions — robot stays at default pose
        actions = torch.zeros(env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim,
                              device=env.unwrapped.device)
        obs, rew, terminated, truncated, info = env.step(actions)

        if step % 50 == 0:
            target_b = env.unwrapped.command_manager.get_command("ee_target")[0, :3]
            ee_pos_b = (robot.data.body_pos_w[0, ee_idx, :3] - robot.data.root_pos_w[0]).cpu()
            dist = torch.norm(target_b.cpu() - ee_pos_b).item()

            joint_pos = robot.data.joint_pos[0, joint_ids].cpu().numpy()
            joint_names_short = ["sp", "sr", "sy", "el", "wr", "wp", "wy"]

            print(f"\n[step {step:4d}]")
            print(f"  target (base): x={target_b[0]:.3f}  y={target_b[1]:.3f}  z={target_b[2]:.3f}")
            print(f"  ee_pos (base): x={ee_pos_b[0]:.3f}  y={ee_pos_b[1]:.3f}  z={ee_pos_b[2]:.3f}")
            print(f"  dist = {dist:.4f} m")
            joint_str = "  ".join(f"{n}={v:.3f}" for n, v in zip(joint_names_short, joint_pos))
            print(f"  joints: {joint_str}")

        step += 1

env.close()
simulation_app.close()
