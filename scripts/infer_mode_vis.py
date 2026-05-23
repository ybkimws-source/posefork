"""Inference visualization: 3 envs (mode 0/1/2), same target, trained policy, GUI.

Run:
    cd ~/unitree_rl_lab
    python ~/teacher-without-a-human/code/posefork/scripts/infer_mode_vis.py --checkpoint ~/unitree_rl_lab/logs/rsl_rl/yb_g1_arm_reach_mode_head/<run>/model_<iter>.pt
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
import posefork.tasks  # noqa: F401

from posefork.tasks.g1_arm_reach.env_cfg import G1ArmReachEnvCfg_PLAY, RIGHT_ARM_JOINTS, EE_LINK
from posefork.tasks.g1_arm_reach.agents.ppo_cfg import G1ArmReachPPORunnerCfg

env_cfg = G1ArmReachEnvCfg_PLAY()
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.scene.env_spacing = 3.0
env_cfg.commands.ee_target.resampling_time_range = (9999.0, 10000.0)  # prevent auto-resample

env = gym.make("YB-G1-ArmReach-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=None)

agent_cfg = G1ArmReachPPORunnerCfg()
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device="cuda:0")

robot = env.unwrapped.scene["robot"]
joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINTS)
ee_idx = robot.find_bodies(EE_LINK)[0][0]
short = ["sp", "sr", "sy", "el", "wr", "wp", "wy"]

cmd_term = env.unwrapped.command_manager._terms["ee_target"]

def sync_target_to_env0():
    """Copy env 0's target to all other envs so they all reach the same point."""
    cmd_term.command[1:] = cmd_term.command[0:1].expand(args_cli.num_envs - 1, -1).clone()

obs = env.get_observations()
sync_target_to_env0()
step = 0

# Side view: looking from Y- direction so all 3 robots appear side by side
try:
    env.unwrapped.sim.set_camera_view(
        eye=[4.5, -9.0, 2.5],     # center of 3 envs (spacing=3, so 0,3,6 → center=3), pulled back
        target=[4.5, 0.0, 1.2],   # look at roughly chest height
    )
except Exception:
    pass

print(f"\n{'='*60}")
print(f"Inference vis  |  {args_cli.checkpoint.split('/')[-1]}")
print(f"num_envs={args_cli.num_envs}  (env0=mode0, env1=mode1, env2=mode2)  same target")
print(f"{'='*60}")

while simulation_app.is_running():
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

    # Re-sync after any episode reset so modes stay on same target
    if dones.any():
        sync_target_to_env0()

    if step % 100 == 0:
        target_b = env.unwrapped.command_manager.get_command("ee_target")[:, :3]
        ee_pos_b = robot.data.body_pos_w[:, ee_idx, :3] - robot.data.root_pos_w
        dist = torch.norm(target_b - ee_pos_b, dim=-1)
        joint_pos = robot.data.joint_pos[:, joint_ids]

        print(f"\n[step {step:5d}]  target: x={target_b[0,0]:.3f} y={target_b[0,1]:.3f} z={target_b[0,2]:.3f}")
        for m in range(min(args_cli.num_envs, 3)):
            jvals = "  ".join(f"{n}={joint_pos[m, j].item():+.3f}" for j, n in enumerate(short))
            print(f"  mode{m}: dist={dist[m].item():.4f}m  |  {jvals}")

    step += 1

env.close()
simulation_app.close()
