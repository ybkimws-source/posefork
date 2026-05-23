"""
Teacher-Only Probe: 3-mode posture/null-space teacher diversity check

Measures whether mode 0/1/2 null-space teachers actually diverge across random targets
without any policy training.

Core formulation:
    Δq_task = J† · (x_target - x_ee)
    Δq_null = (I - J†J) · α · (q_null_k - q_current)
    q_teacher_k = q_current + Δq_task + Δq_null

Usage:
    ./unitree_rl_lab.sh -p scripts/probe_teacher_diversity.py --headless \
        [--num_envs 256] [--num_steps 500] [--alpha 0.5] [--save_csv results/teacher_probe.csv]
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teacher diversity probe (no policy training)")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--alpha", type=float, default=0.5, help="Null-space gain α")
parser.add_argument("--collapse_thresh", type=float, default=0.05, help="Distance threshold for collapse detection (rad)")
parser.add_argument("--save_csv", type=str, default="", help="Optional path to save per-step CSV log")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports after AppLauncher ──────────────────────────────────────────────────
import torch
import gymnasium as gym

import unitree_rl_lab.tasks  # noqa: F401 — registers all tasks including manipulation

from unitree_rl_lab.tasks.manipulation.robots.g1_upper_body.arm_reach.arm_reach_env_cfg import (
    G1ArmReachEnvCfg,
    RIGHT_ARM_JOINTS,
    EE_LINK,
)

# ── 3-mode posture priors (same joint order as RIGHT_ARM_JOINTS) ───────────────
# shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
Q_NULL_PRIORS = torch.tensor([
    [ 0.1, -0.70,  0.3,  1.2, -0.15,  0.0,  0.0],  # Mode 0: elbow-out + shoulder-dominant + wrist-neutral
    [ 0.5, -0.15, -0.1,  1.8, -0.80,  0.2,  0.0],  # Mode 1: elbow-down + compact + wrist-pronated
    [ 0.3, -0.25,  0.0,  0.4,  0.50, -0.2,  0.0],  # Mode 2: arm-extended + effort-conservative + wrist-supinated
], dtype=torch.float32)  # (3, 7)

NUM_MODES = Q_NULL_PRIORS.shape[0]


def compute_null_space_teachers(
    joint_pos: torch.Tensor,  # (N, 7)
    ee_pos_w: torch.Tensor,   # (N, 3) world frame
    target_w: torch.Tensor,   # (N, 3) world frame
    jacobian: torch.Tensor,   # (N, 6, 7)
    alpha: float,
    q_null: torch.Tensor,     # (3, 7) mode priors, on correct device
) -> torch.Tensor:
    """Returns q_teacher for all 3 modes: shape (N, 3, 7)"""
    N, n_joints = joint_pos.shape
    device = joint_pos.device

    J = jacobian  # (N, 6, 7)
    J_pinv = torch.linalg.pinv(J)  # (N, 7, 6)

    # Task-space delta: position only (first 3 rows)
    delta_pos = target_w - ee_pos_w  # (N, 3)
    delta_x6 = torch.zeros(N, 6, device=device)
    delta_x6[:, :3] = delta_pos
    # Δq_task = J† · Δx
    delta_q_task = (J_pinv @ delta_x6.unsqueeze(-1)).squeeze(-1)  # (N, 7)

    # Null-space projector: (I - J†J)
    I_mat = torch.eye(n_joints, device=device).unsqueeze(0).expand(N, -1, -1)  # (N, 7, 7)
    null_proj = I_mat - (J_pinv @ J)  # (N, 7, 7)

    q_teachers = torch.zeros(N, NUM_MODES, n_joints, device=device)
    for k in range(NUM_MODES):
        q_null_k = q_null[k].unsqueeze(0).expand(N, -1)  # (N, 7)
        posture_diff = q_null_k - joint_pos               # (N, 7)
        delta_q_null = (null_proj @ (alpha * posture_diff).unsqueeze(-1)).squeeze(-1)  # (N, 7)
        q_teachers[:, k, :] = joint_pos + delta_q_task + delta_q_null

    return q_teachers  # (N, 3, 7)


def run_probe(env, args_cli):
    device = env.device
    alpha = args_cli.alpha
    collapse_thresh = args_cli.collapse_thresh

    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINTS)
    ee_idx = robot.find_bodies(EE_LINK)[0][0]
    jacobian_body_idx = ee_idx - 1  # fixed-base: root excluded → body_idx - 1

    q_null = Q_NULL_PRIORS.to(device)

    step_logs = []

    obs, _ = env.reset()
    actions = torch.zeros(env.num_envs, len(RIGHT_ARM_JOINTS), device=device)

    print(f"\n{'='*60}")
    print(f"Teacher Diversity Probe")
    print(f"  num_envs={env.num_envs}, num_steps={args_cli.num_steps}")
    print(f"  alpha={alpha}, collapse_thresh={collapse_thresh:.3f} rad")
    print(f"  Modes: {NUM_MODES}")
    print(f"  Mode 0 (q_null): {q_null[0].cpu().numpy()}")
    print(f"  Mode 1 (q_null): {q_null[1].cpu().numpy()}")
    print(f"  Mode 2 (q_null): {q_null[2].cpu().numpy()}")
    print(f"{'='*60}\n")

    for step in range(args_cli.num_steps):
        obs, rew, terminated, truncated, info = env.step(actions)

        # ── Fetch state ────────────────────────────────────────────
        ee_pos_w = robot.data.body_pos_w[:, ee_idx, :3]          # (N, 3)
        joint_pos = robot.data.joint_pos[:, joint_ids]            # (N, 7)
        target_b = env.command_manager.get_command("ee_target")[:, :3]  # (N, 3) base frame
        target_w = target_b + robot.data.root_pos_w               # (N, 3) world frame

        # ── Jacobian ───────────────────────────────────────────────
        # get_jacobians(): (N, num_bodies-1, 6, total_dofs) for fixed-base
        jac_full = robot.root_physx_view.get_jacobians()          # (N, ?, 6, total_dofs)
        jacobian = jac_full[:, jacobian_body_idx, :, :]           # (N, 6, total_dofs)
        jacobian = jacobian[:, :, joint_ids]                      # (N, 6, 7)

        # ── Compute 3-mode teachers ────────────────────────────────
        q_teachers = compute_null_space_teachers(
            joint_pos, ee_pos_w, target_w, jacobian, alpha, q_null
        )  # (N, 3, 7)

        # ── Pairwise distances ─────────────────────────────────────
        d01 = torch.norm(q_teachers[:, 0] - q_teachers[:, 1], dim=-1)  # (N,)
        d02 = torch.norm(q_teachers[:, 0] - q_teachers[:, 2], dim=-1)
        d12 = torch.norm(q_teachers[:, 1] - q_teachers[:, 2], dim=-1)

        # ── Collapse detection ─────────────────────────────────────
        any_collapse = (d01 < collapse_thresh) | (d02 < collapse_thresh) | (d12 < collapse_thresh)
        collapse_rate = any_collapse.float().mean().item()

        # ── Per-joint spread (std across 3 modes) ─────────────────
        joint_spread = q_teachers.std(dim=1).mean(0)  # (7,)

        # ── Delta-q magnitude per mode ─────────────────────────────
        delta_q_mags = torch.norm(q_teachers - joint_pos.unsqueeze(1), dim=-1).mean(0)  # (3,)

        log = {
            "step": step,
            "d01_mean": d01.mean().item(),
            "d01_min":  d01.min().item(),
            "d02_mean": d02.mean().item(),
            "d02_min":  d02.min().item(),
            "d12_mean": d12.mean().item(),
            "d12_min":  d12.min().item(),
            "collapse_rate": collapse_rate,
            "joint_spread": joint_spread.cpu().tolist(),
            "delta_q_mag_m0": delta_q_mags[0].item(),
            "delta_q_mag_m1": delta_q_mags[1].item(),
            "delta_q_mag_m2": delta_q_mags[2].item(),
        }
        step_logs.append(log)

        if step % 50 == 0:
            print(
                f"[step {step:4d}] "
                f"d(0,1)={d01.mean():.3f} d(0,2)={d02.mean():.3f} d(1,2)={d12.mean():.3f} "
                f"| collapse={collapse_rate:.1%} "
                f"| Δq_mag=[{delta_q_mags[0]:.3f},{delta_q_mags[1]:.3f},{delta_q_mags[2]:.3f}]"
            )
            js = joint_spread.cpu().numpy()
            print(f"           joint_spread: {js.round(4)}")

    return step_logs


def summarize(logs, joint_names):
    import numpy as np

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    def _arr(key):
        return np.array([l[key] for l in logs])

    for pair, key_m, key_min in [
        ("(0,1)", "d01_mean", "d01_min"),
        ("(0,2)", "d02_mean", "d02_min"),
        ("(1,2)", "d12_mean", "d12_min"),
    ]:
        m = _arr(key_m)
        mn = _arr(key_min)
        print(f"  d{pair}: mean={m.mean():.4f}  min_of_means={m.min():.4f}  abs_min={mn.min():.4f}")

    collapse = _arr("collapse_rate")
    print(f"  collapse_rate: mean={collapse.mean():.2%}  max={collapse.max():.2%}")

    spreads = np.array([l["joint_spread"] for l in logs])  # (steps, 7)
    mean_spread = spreads.mean(0)
    print(f"\n  Per-joint spread (mean over steps):")
    for i, (jn, sp) in enumerate(zip(joint_names, mean_spread)):
        print(f"    [{i}] {jn}: {sp:.4f} rad")

    print(f"\n  Min pairwise distance across ALL steps + ALL envs:")
    for pair, key in [("(0,1)", "d01_min"), ("(0,2)", "d02_min"), ("(1,2)", "d12_min")]:
        print(f"    d{pair} abs_min = {_arr(key).min():.4f} rad")

    print(f"\n  → If all min distances > {0.05:.2f} rad: teacher family is meaningfully distinct")
    print(f"  → If collapse_rate > 50%: family is too similar for most targets")


def save_csv(logs, path):
    import csv, os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(logs[0].keys())
    # flatten joint_spread list
    n_joints = len(logs[0]["joint_spread"])
    flat_fields = [f for f in fieldnames if f != "joint_spread"] + [f"spread_j{i}" for i in range(n_joints)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for log in logs:
            row = {k: v for k, v in log.items() if k != "joint_spread"}
            for i, v in enumerate(log["joint_spread"]):
                row[f"spread_j{i}"] = v
            writer.writerow(row)
    print(f"\n  CSV saved → {path}")


def main():
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # Disable IK teacher reward (probe only, no training signal needed)
    env_cfg.rewards.ik_teacher.weight = 0.0

    env = gym.make("Unitree-G1-ArmReach-v0", cfg=env_cfg)
    env = env.unwrapped

    logs = run_probe(env, args_cli)
    env.close()

    summarize(logs, RIGHT_ARM_JOINTS)

    if args_cli.save_csv:
        save_csv(logs, args_cli.save_csv)


if __name__ == "__main__":
    main()
    simulation_app.close()
