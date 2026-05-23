"""
Teacher-Only Probe: 3-mode posture/null-space teacher diversity check

Measures whether mode 0/1/2 null-space teachers actually diverge across random targets
without any policy training.

Correct formulation (position-only task):
    J_pos  = J[:, :3, :]                        # (N, 3, 7) — position rows only
    J_dls  = J_pos^T (J_pos J_pos^T + λ²I)^-1  # DLS inverse (N, 7, 3)
    Δq_task = J_dls · (x_target - x_ee)         # (N, 7)
    null_proj = I - J_dls · J_pos               # (N, 7, 7)  — 4-DOF null-space
    Δq_null = null_proj · α · (q_null_k - q_current)
    q_teacher_k = q_current + Δq_task + Δq_null

Usage:
    python scripts/probe_teacher_diversity.py --headless \
        [--num_envs 256] [--num_steps 500] [--alpha 0.5] [--dls_lambda 0.05]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teacher diversity probe (no policy training)")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--alpha", type=float, default=0.5, help="Null-space gain α")
parser.add_argument("--collapse_thresh", type=float, default=0.05, help="Distance threshold for collapse (rad)")
parser.add_argument("--dls_lambda", type=float, default=0.05, help="DLS damping factor λ")
parser.add_argument("--save_csv", type=str, default="", help="Optional path to save per-step CSV")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports after AppLauncher ──────────────────────────────────────────────────
import torch
import gymnasium as gym

import unitree_rl_lab.tasks  # noqa: F401
import posefork.tasks  # noqa: F401

from posefork.tasks.g1_arm_reach.env_cfg import (
    G1ArmReachEnvCfg,
    RIGHT_ARM_JOINTS,
    EE_LINK,
)

# ── 3-mode posture priors ─────────────────────────────────────────────────────
# joint order: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#              wrist_roll, wrist_pitch, wrist_yaw
Q_NULL_PRIORS = torch.tensor([
    [ 0.1, -0.70,  0.3,  1.2, -0.15,  0.0,  0.0],  # Mode 0: elbow-out, shoulder-dominant, wrist-neutral
    [ 0.5, -0.15, -0.1,  1.8, -0.80,  0.2,  0.0],  # Mode 1: elbow-down, compact, wrist-pronated
    [ 0.3, -0.25,  0.0,  0.4,  0.50, -0.2,  0.0],  # Mode 2: arm-extended, effort-conservative, wrist-supinated
], dtype=torch.float32)  # (3, 7)

NUM_MODES = Q_NULL_PRIORS.shape[0]


def dls_inverse(J_pos: torch.Tensor, lam: float) -> torch.Tensor:
    """DLS inverse of position Jacobian.
    J_pos: (N, 3, 7) → returns J_dls: (N, 7, 3)
    J_dls = J^T (J J^T + λ²I)^-1
    """
    JJT = J_pos @ J_pos.transpose(-1, -2)  # (N, 3, 3)
    lam2I = (lam ** 2) * torch.eye(3, device=J_pos.device, dtype=J_pos.dtype).unsqueeze(0)
    return J_pos.transpose(-1, -2) @ torch.linalg.inv(JJT + lam2I)  # (N, 7, 3)


def compute_null_space_teachers(
    joint_pos: torch.Tensor,  # (N, 7)
    ee_pos_w: torch.Tensor,   # (N, 3) world
    target_w: torch.Tensor,   # (N, 3) world
    jacobian: torch.Tensor,   # (N, 6, 7) full Jacobian
    alpha: float,
    q_null: torch.Tensor,     # (3, 7) on device
    lam: float = 0.05,
) -> tuple:                   # (q_teachers (N,3,7), dq_task (N,7), dq_nulls (N,3,7))
    N, n_joints = joint_pos.shape
    device = joint_pos.device

    # Fix 1: position-only rows (task dim = 3, null-space dim = 4)
    J_pos = jacobian[:, :3, :]           # (N, 3, 7)

    # Fix 2: DLS inverse instead of Moore-Penrose pinv
    J_dls = dls_inverse(J_pos, lam)      # (N, 7, 3)

    # Task term
    delta_x = target_w - ee_pos_w        # (N, 3)
    delta_q_task = (J_dls @ delta_x.unsqueeze(-1)).squeeze(-1)  # (N, 7)

    # Null-space projector based on position-only Jacobian
    I_mat = torch.eye(n_joints, device=device, dtype=jacobian.dtype).unsqueeze(0).expand(N, -1, -1)
    null_proj = I_mat - (J_dls @ J_pos)  # (N, 7, 7)  — 4-DOF null-space

    q_teachers = torch.zeros(N, NUM_MODES, n_joints, device=device)
    delta_q_nulls = torch.zeros(N, NUM_MODES, n_joints, device=device)

    for k in range(NUM_MODES):
        posture_diff = q_null[k].unsqueeze(0).expand(N, -1) - joint_pos
        delta_q_null = (null_proj @ (alpha * posture_diff).unsqueeze(-1)).squeeze(-1)
        delta_q_nulls[:, k] = delta_q_null
        q_teachers[:, k] = joint_pos + delta_q_task + delta_q_null

    return q_teachers, delta_q_task, delta_q_nulls


def run_probe(env, args_cli):
    device = env.device
    q_null = Q_NULL_PRIORS.to(device)

    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINTS)
    ee_idx = robot.find_bodies(EE_LINK)[0][0]
    jacobian_body_idx = ee_idx - 1  # fixed-base: root excluded

    step_logs = []
    obs, _ = env.reset()
    actions = torch.zeros(env.num_envs, len(RIGHT_ARM_JOINTS), device=device)

    print(f"\n{'='*60}")
    print(f"Teacher Diversity Probe  |  YB-G1-ArmReach")
    print(f"  num_envs={env.num_envs}  steps={args_cli.num_steps}  α={args_cli.alpha}  λ={args_cli.dls_lambda}")
    print(f"  Jacobian: position-only (3×7), null-space dim=4")
    print(f"  collapse_thresh={args_cli.collapse_thresh:.3f} rad")
    for k, q in enumerate(Q_NULL_PRIORS):
        print(f"  Mode {k} q_null: {q.numpy()}")
    print(f"{'='*60}\n")

    for step in range(args_cli.num_steps):
        obs, rew, terminated, truncated, info = env.step(actions)

        ee_pos_world = robot.data.body_pos_w[:, ee_idx, :3]
        joint_pos = robot.data.joint_pos[:, joint_ids]
        target_b = env.command_manager.get_command("ee_target")[:, :3]
        target_w = target_b + robot.data.root_pos_w

        jac_full = robot.root_physx_view.get_jacobians()
        jacobian = jac_full[:, jacobian_body_idx, :, :][:, :, joint_ids]  # (N, 6, 7)

        q_teachers, dq_task, dq_nulls = compute_null_space_teachers(
            joint_pos, ee_pos_world, target_w, jacobian,
            args_cli.alpha, q_null, args_cli.dls_lambda,
        )  # (N,3,7), (N,7), (N,3,7)

        d01 = torch.norm(q_teachers[:, 0] - q_teachers[:, 1], dim=-1)
        d02 = torch.norm(q_teachers[:, 0] - q_teachers[:, 2], dim=-1)
        d12 = torch.norm(q_teachers[:, 1] - q_teachers[:, 2], dim=-1)

        collapse = ((d01 < args_cli.collapse_thresh) |
                    (d02 < args_cli.collapse_thresh) |
                    (d12 < args_cli.collapse_thresh)).float().mean().item()

        joint_spread = q_teachers.std(dim=1).mean(0)  # (7,)

        # Fix 3: log task vs null-space contributions separately
        dq_task_mag = torch.norm(dq_task, dim=-1).mean().item()          # scalar
        dq_null_mags = torch.norm(dq_nulls, dim=-1).mean(0)              # (3,)
        null_task_ratio = dq_null_mags / (dq_task_mag + 1e-8)           # (3,)

        step_logs.append({
            "step": step,
            "d01_mean": d01.mean().item(), "d01_min": d01.min().item(),
            "d02_mean": d02.mean().item(), "d02_min": d02.min().item(),
            "d12_mean": d12.mean().item(), "d12_min": d12.min().item(),
            "collapse_rate": collapse,
            "joint_spread": joint_spread.cpu().tolist(),
            "dq_task_mag": dq_task_mag,
            "dq_null_m0": dq_null_mags[0].item(),
            "dq_null_m1": dq_null_mags[1].item(),
            "dq_null_m2": dq_null_mags[2].item(),
            "null_task_ratio_m0": null_task_ratio[0].item(),
            "null_task_ratio_m1": null_task_ratio[1].item(),
            "null_task_ratio_m2": null_task_ratio[2].item(),
        })

        if step % 50 == 0:
            print(
                f"[step {step:4d}]  "
                f"d(0,1)={d01.mean():.3f}  d(0,2)={d02.mean():.3f}  d(1,2)={d12.mean():.3f}  "
                f"collapse={collapse:.1%}"
            )
            print(
                f"           |Δq_task|={dq_task_mag:.3f}  "
                f"|Δq_null|=[{dq_null_mags[0]:.3f},{dq_null_mags[1]:.3f},{dq_null_mags[2]:.3f}]  "
                f"ratio=[{null_task_ratio[0]:.2f},{null_task_ratio[1]:.2f},{null_task_ratio[2]:.2f}]"
            )
            print(f"           spread/joint: {joint_spread.cpu().numpy().round(4)}")

    return step_logs


def summarize(logs):
    import numpy as np

    def arr(k):
        return np.array([l[k] for l in logs])

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for pair, mk, mnk in [("(0,1)", "d01_mean", "d01_min"),
                           ("(0,2)", "d02_mean", "d02_min"),
                           ("(1,2)", "d12_mean", "d12_min")]:
        m, mn = arr(mk), arr(mnk)
        print(f"  d{pair}:  mean={m.mean():.4f}  step-min={m.min():.4f}  abs-min={mn.min():.4f}")

    c = arr("collapse_rate")
    print(f"  collapse:  mean={c.mean():.2%}  max={c.max():.2%}")

    dq_task = arr("dq_task_mag")
    print(f"\n  |Δq_task| (mean over steps): {dq_task.mean():.4f} rad")
    for k in range(3):
        null_mag = arr(f"dq_null_m{k}").mean()
        ratio = arr(f"null_task_ratio_m{k}").mean()
        flag = "✓" if ratio < 0.5 else "✗ (null dominates)"
        print(f"  Mode {k}: |Δq_null|={null_mag:.4f}  ratio={ratio:.3f}  {flag}")

    spreads = np.array([l["joint_spread"] for l in logs])
    print(f"\n  Per-joint spread (mean over steps):")
    for i, (jn, sp) in enumerate(zip(RIGHT_ARM_JOINTS, spreads.mean(0))):
        print(f"    [{i}] {jn}: {sp:.4f} rad")

    print(f"\n  Verdict:")
    abs_mins = [arr("d01_min").min(), arr("d02_min").min(), arr("d12_min").min()]
    if all(v > 0.05 for v in abs_mins):
        print("    ✓ All pairwise distances > 0.05 rad → teacher family is meaningfully distinct")
    else:
        print("    ✗ Some pairs collapse to < 0.05 rad → consider adjusting q_null priors or α")


def save_csv(logs, path):
    import csv, os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n_joints = len(logs[0]["joint_spread"])
    flat_fields = [k for k in logs[0] if k != "joint_spread"] + [f"spread_j{i}" for i in range(n_joints)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for log in logs:
            row = {k: v for k, v in log.items() if k != "joint_spread"}
            for i, v in enumerate(log["joint_spread"]):
                row[f"spread_j{i}"] = v
            writer.writerow(row)
    print(f"  CSV → {path}")


def main():
    env_cfg = G1ArmReachEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("YB-G1-ArmReach-v0", cfg=env_cfg).unwrapped
    logs = run_probe(env, args_cli)
    env.close()

    summarize(logs)
    if args_cli.save_csv:
        save_csv(logs, args_cli.save_csv)


if __name__ == "__main__":
    main()
    simulation_app.close()
