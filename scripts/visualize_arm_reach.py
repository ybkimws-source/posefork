"""Visualize G1 arm reach: target marker + zero-action stepping.

Run:
    cd ~/unitree_rl_lab
    python ~/teacher-without-a-human/code/posefork/scripts/visualize_arm_reach.py \
        --task YB-G1-ArmReach-Play-v0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="YB-G1-ArmReach-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import unitree_rl_lab.tasks  # noqa: F401
import posefork.tasks  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg
from posefork.tasks.g1_arm_reach.env_cfg import RIGHT_ARM_JOINTS, EE_LINK

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.commands.ee_target.debug_vis = True

env = gym.make(args_cli.task, cfg=env_cfg)
obs, _ = env.reset()

robot = env.unwrapped.scene["robot"]
joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINTS)
ee_idx = robot.find_bodies(EE_LINK)[0][0]

print("\n=== YB G1 Arm Reach Visualization ===")
print(f"Task: {args_cli.task}")
print(f"Joint IDs: {joint_ids}")
print(f"EE body index: {ee_idx}")
print(f"Root pos: {robot.data.root_pos_w[0].cpu().numpy()}")
print("=" * 40)

step = 0
while simulation_app.is_running():
    with torch.inference_mode():
        actions = torch.zeros(
            env.unwrapped.num_envs,
            env.unwrapped.action_manager.total_action_dim,
            device=env.unwrapped.device,
        )
        obs, rew, terminated, truncated, info = env.step(actions)

        if step % 50 == 0:
            target_b = env.unwrapped.command_manager.get_command("ee_target")[0, :3]
            ee_pos_b = (robot.data.body_pos_w[0, ee_idx, :3] - robot.data.root_pos_w[0]).cpu()
            dist = torch.norm(target_b.cpu() - ee_pos_b).item()
            joint_pos = robot.data.joint_pos[0, joint_ids].cpu().numpy()
            short = ["sp", "sr", "sy", "el", "wr", "wp", "wy"]

            print(f"\n[step {step:4d}]")
            print(f"  target: x={target_b[0]:.3f}  y={target_b[1]:.3f}  z={target_b[2]:.3f}")
            print(f"  ee_pos: x={ee_pos_b[0]:.3f}  y={ee_pos_b[1]:.3f}  z={ee_pos_b[2]:.3f}")
            print(f"  dist = {dist:.4f} m")
            print(f"  joints: {'  '.join(f'{n}={v:.3f}' for n, v in zip(short, joint_pos))}")

        step += 1

env.close()
simulation_app.close()
