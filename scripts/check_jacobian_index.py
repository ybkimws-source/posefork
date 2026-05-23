"""
Jacobian body index verification script.

Checks that find_bodies() index for right_wrist_yaw_link matches
the actual shape returned by get_jacobians().
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import unitree_rl_lab.tasks  # noqa
import posefork.tasks       # noqa

from posefork.tasks.g1_arm_reach.env_cfg import (
    G1ArmReachEnvCfg, RIGHT_ARM_JOINTS, EE_LINK,
)

def main():
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = 1

    env = gym.make("YB-G1-ArmReach-v0", cfg=env_cfg).unwrapped
    env.reset()
    env.step(torch.zeros(1, len(RIGHT_ARM_JOINTS), device=env.device))

    robot = env.scene["robot"]

    # ── 1. body names & indices ──────────────────────────────────────────────
    body_names = list(robot.data.body_names)
    num_bodies = len(body_names)
    print(f"\n{'='*60}")
    print(f"total bodies (articulation): {num_bodies}")
    print(f"body list:")
    for i, n in enumerate(body_names):
        marker = " <-- EE" if n == EE_LINK else ""
        print(f"  [{i:2d}] {n}{marker}")

    # ── 2. find_bodies result ─────────────────────────────────────────────────
    ee_idx_list, _ = robot.find_bodies(EE_LINK)
    ee_idx = ee_idx_list[0]
    print(f"\nfind_bodies('{EE_LINK}') → ee_idx = {ee_idx}")
    print(f"current code: jacobian_body_idx = ee_idx - 1 = {ee_idx - 1}")
    print(f"  body_names[ee_idx]     = {body_names[ee_idx]}")
    if ee_idx - 1 >= 0:
        print(f"  body_names[ee_idx - 1] = {body_names[ee_idx - 1]}")

    # ── 3. Jacobian shape ────────────────────────────────────────────────────
    jac = robot.root_physx_view.get_jacobians()
    print(f"\nget_jacobians() shape: {tuple(jac.shape)}")
    print(f"  → Jacobian body count = {jac.shape[1]}")
    print(f"  → articulation body count = {num_bodies}")
    print(f"  → difference (excluded bodies) = {num_bodies - jac.shape[1]}")

    jac_num_bodies = jac.shape[1]
    excluded = num_bodies - jac_num_bodies
    correct_jac_idx = ee_idx - excluded
    print(f"\nestimated correct jacobian_body_idx = ee_idx({ee_idx}) - excluded({excluded}) = {correct_jac_idx}")
    print(f"current code jacobian_body_idx = {ee_idx - 1}")

    if correct_jac_idx == ee_idx - 1:
        print(f"\n✅ jacobian_body_idx = ee_idx - 1 correct (excluded = 1 = root only)")
    else:
        print(f"\n❌ mismatch! current={ee_idx - 1}, correct={correct_jac_idx}")
        if 0 <= correct_jac_idx + excluded < num_bodies:
            print(f"   body at correct jac_idx: {body_names[correct_jac_idx + excluded]}")

    # ── 4. TCP offset sanity ─────────────────────────────────────────────────
    print(f"\nTCP_OFFSET = {TCP_OFFSET}")
    link_pos_w = robot.data.body_pos_w[0, ee_idx, :3]
    link_quat_w = robot.data.body_quat_w[0, ee_idx, :]
    from isaaclab.utils.math import quat_apply
    offset_t = torch.tensor(TCP_OFFSET, device=env.device, dtype=link_pos_w.dtype)
    tcp_w = link_pos_w + quat_apply(link_quat_w.unsqueeze(0), offset_t.unsqueeze(0)).squeeze(0)
    print(f"link_pos_w     = {link_pos_w.cpu().numpy().round(4)}")
    print(f"tcp_pos_w      = {tcp_w.cpu().numpy().round(4)}")
    print(f"offset_applied = {(tcp_w - link_pos_w).cpu().numpy().round(4)}")

    # ── 5. Jacobian col norms near EE ────────────────────────────────────────
    print(f"\n--- Jacobian position-row col norms near EE ---")
    joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINTS)
    for test_idx in [ee_idx - 2, ee_idx - 1, ee_idx]:
        if 0 <= test_idx < jac_num_bodies:
            j = jac[0, test_idx, :3, :][:, joint_ids]
            norms = j.norm(dim=0).cpu().numpy().round(4)
            body_name = body_names[test_idx + excluded] if (test_idx + excluded) < num_bodies else "?"
            print(f"  jac[{test_idx}] ({body_name}): {norms}")

    env.close()
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
    simulation_app.close()
