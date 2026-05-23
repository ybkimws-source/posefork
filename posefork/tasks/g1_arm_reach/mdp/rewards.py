from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .observations import tcp_pos_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _watch(log: dict, name: str, values: torch.Tensor) -> None:
    with torch.no_grad():
        log[f"watch/{name}_mean"]    = values.mean().item()
        log[f"watch/{name}_max_abs"] = values.abs().max().item()
        log[f"watch/{name}_has_nan"] = float(values.isnan().any().item())
        log[f"watch/{name}_has_inf"] = float(values.isinf().any().item())


# ── EE distance rewards ──────────────────────────────────────────────────────

def tracking_ee_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    link_name: str,
    std: float = 0.05,
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    ee_pos = tcp_pos_w(env, link_name, tcp_offset, asset_cfg) - robot.data.root_pos_w
    target = env.command_manager.get_command(command_name)[:, :3]
    dist = torch.norm(target - ee_pos, dim=-1)
    reward = torch.exp(-dist / std)
    if hasattr(env, "extras") and "log" in env.extras:
        _watch(env.extras["log"], "tracking", reward)
    return reward


def progress_ee_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    link_name: str,
    scale: float = 5.0,
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    ee_pos = tcp_pos_w(env, link_name, tcp_offset, asset_cfg) - robot.data.root_pos_w
    target = env.command_manager.get_command(command_name)[:, :3]
    dist = torch.norm(target - ee_pos, dim=-1)

    if not hasattr(env, "_yb_arm_reach_last_dist"):
        env._yb_arm_reach_last_dist = dist.clone()

    reset_ids = (env.episode_length_buf == 1).nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        env._yb_arm_reach_last_dist[reset_ids] = dist[reset_ids]

    progress = env._yb_arm_reach_last_dist - dist
    env._yb_arm_reach_last_dist[:] = dist
    reward = progress * scale
    if hasattr(env, "extras") and "log" in env.extras:
        _watch(env.extras["log"], "progress", reward)
    return reward


def precision_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    link_name: str,
    std: float = 0.0125,
    bonus: float = 2.0,
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    ee_pos = tcp_pos_w(env, link_name, tcp_offset, asset_cfg) - robot.data.root_pos_w
    target = env.command_manager.get_command(command_name)[:, :3]
    dist = torch.norm(target - ee_pos, dim=-1)
    reward = torch.exp(-dist / std) * bonus
    if hasattr(env, "extras") and "log" in env.extras:
        _watch(env.extras["log"], "precision", reward)
    return reward


# ── Mode assignment ──────────────────────────────────────────────────────────

def _get_or_init_mode_ids(env: ManagerBasedRLEnv, num_modes: int) -> torch.Tensor:
    """Mode id per env: fixed as env_index % num_modes for the env lifetime."""
    if (not hasattr(env, "_yb_mode_ids")) or (getattr(env, "_yb_mode_ids_num_modes", None) != num_modes):
        env._yb_mode_ids = torch.arange(env.num_envs, device=env.device) % num_modes
        env._yb_mode_ids_num_modes = num_modes
    return env._yb_mode_ids


# ── SEW arm-angle mode teacher ───────────────────────────────────────────────
#
# Each mode k targets a specific arm angle (phi_k), parameterizing the 7-DOF
# redundancy as a scalar: the elbow-plane rotation about the shoulder->target axis.
# Computed GPU-batched every step; no IK solver needed.
#
# phi_k = [+0.80, +0.97, +1.15] rad  (iter7, positive flip resolved chicken-wing)
#
# Shoulder socket position S is fixed under fix_root_link=True + identity spawn.

_ARM_ANGLE_MODES = torch.tensor([+0.80, +0.97, +1.15], dtype=torch.float32)

# right_shoulder_pitch_joint origin in base frame (G1 29DOF, waist locked, neutral pose).
# Derived from URDF FK; constant for fixed-base robot regardless of arm configuration.
_SHOULDER_SOCKET = torch.tensor([-7.2e-6, -0.1002, 0.2918], dtype=torch.float32)


def _compute_arm_angle(
    shoulder: torch.Tensor,   # (N, 3) fixed shoulder socket world pos
    elbow:    torch.Tensor,   # (N, 3) elbow body world pos
    target:   torch.Tensor,   # (N, 3) EE target world pos
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SEW arm angle: elbow-plane rotation about the shoulder->target axis.

    Reference direction: -Z projected onto the plane perpendicular to sw_unit,
    so elbow-down neutral posture maps to ~0 rad.

    Returns:
        angle : (N,) in [-pi, pi]; 0 for degenerate envs (valid=False)
        valid : (N,) bool; False when elbow lies on the shoulder->target axis
    """
    device = shoulder.device

    sw      = target - shoulder
    sw_unit = sw / sw.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    se      = elbow - shoulder
    se_perp = se - (se * sw_unit).sum(-1, keepdim=True) * sw_unit

    valid   = se_perp.norm(dim=-1) > 1e-4

    z_down  = torch.tensor([0., 0., -1.], device=device).expand_as(sw_unit)
    ref     = z_down - (z_down * sw_unit).sum(-1, keepdim=True) * sw_unit
    ref     = ref / ref.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    cross   = torch.linalg.cross(ref, se_perp, dim=-1)
    sin_a   = (cross * sw_unit).sum(-1)
    cos_a   = (ref * se_perp).sum(-1)

    return torch.atan2(sin_a, cos_a), valid


def _angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular angle difference a - b, wrapped to [-pi, pi]."""
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def arm_angle_mode_teacher(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_link_name: str,
    phi_k_override: list[float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    SEW arm-angle mode teacher reward.

    Drives elbow plane toward phi_k for each mode.
    reward = -|arm_angle - phi_k|.clamp(max=1.0) * gate(ee_dist)
    gate = exp(-ee_dist / 0.3): teacher signal grows as EE approaches target.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "_yb_aat_elbow_idx"):
        _elbow_ids = robot.find_bodies("right_elbow_link")[0]
        assert len(_elbow_ids) > 0, "right_elbow_link not found"
        _ee_ids = robot.find_bodies(ee_link_name)[0]
        assert len(_ee_ids) > 0, f"{ee_link_name} not found"
        env._yb_aat_elbow_idx    = _elbow_ids[0]
        env._yb_aat_ee_idx       = _ee_ids[0]
        _phi = torch.tensor(phi_k_override, dtype=torch.float32) if phi_k_override is not None else _ARM_ANGLE_MODES
        env._yb_aat_phi_modes    = _phi.to(env.device)
        # fix_root_link=True + identity spawn: root_pos_w is the world origin offset,
        # shoulder_pitch_joint origin is constant relative to root regardless of joint angles.
        env._yb_aat_shoulder_off = _SHOULDER_SOCKET.to(env.device)
        _wrist_names = ["right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
        env._yb_aat_wrist_ids, _ = robot.find_joints(_wrist_names)
        env._yb_aat_sr_ids, _    = robot.find_joints(["right_shoulder_roll_joint"])

    N        = env.num_envs
    shoulder = robot.data.root_pos_w + env._yb_aat_shoulder_off
    elbow    = robot.data.body_pos_w[:, env._yb_aat_elbow_idx]
    ee_pos   = robot.data.body_pos_w[:, env._yb_aat_ee_idx]

    target_b = env.command_manager.get_command(command_name)[:, :3]
    target_w = target_b + robot.data.root_pos_w

    arm_angle, aa_valid = _compute_arm_angle(shoulder, elbow, target_w)
    mode_ids   = _get_or_init_mode_ids(env, env._yb_aat_phi_modes.shape[0])
    phi_target = env._yb_aat_phi_modes[mode_ids]

    angle_err = _angle_diff(arm_angle, phi_target).abs().clamp(max=1.0)

    ee_dist = torch.norm(target_w - ee_pos, dim=-1)
    gate    = torch.exp(-ee_dist / 0.3)
    reward  = -angle_err * gate * aa_valid.float()

    with torch.no_grad():
        if "log" not in env.extras:
            env.extras["log"] = {}
        log = env.extras["log"]
        log["arm_angle/degenerate_rate"] = (~aa_valid).float().mean().item()
        v = aa_valid
        mode_means: list[float | None] = [None] * env._yb_aat_phi_modes.shape[0]
        z_means: dict[str, float] = {}
        y_means: dict[str, float] = {}
        if v.any():
            log["arm_angle/mean"] = arm_angle[v].mean().item()
            log["arm_angle/std"]  = arm_angle[v].std().item()
        log["arm_angle/gate_mean"] = gate.mean().item()
        n_phi = env._yb_aat_phi_modes.shape[0]
        masks = [mode_ids == k for k in range(n_phi)]
        for k in range(n_phi):
            mk = masks[k] & v
            if mk.any():
                mode_means[k] = arm_angle[mk].mean().item()
                log[f"arm_angle/mode{k}_mean"] = mode_means[k]
        tz = target_b[:, 2]
        ty = target_b[:, 1]
        for zlo, zhi, zlbl in [(0.16, 0.24, "lo"), (0.24, 0.32, "hi")]:
            zm = (tz >= zlo) & (tz < zhi) & v
            if zm.sum() > 10:
                z_means[zlbl] = arm_angle[zm].mean().item()
                log[f"arm_angle/z_{zlbl}_mean"] = z_means[zlbl]
                log[f"arm_angle/z_{zlbl}_std"]  = arm_angle[zm].std().item()
        for ylo, yhi, ylbl in [(-0.38, -0.28, "lat"), (-0.28, -0.18, "fwd")]:
            ym = (ty >= ylo) & (ty < yhi) & v
            if ym.sum() > 10:
                y_means[ylbl] = arm_angle[ym].mean().item()
                log[f"arm_angle/y_{ylbl}_mean"] = y_means[ylbl]
                log[f"arm_angle/y_{ylbl}_std"]  = arm_angle[ym].std().item()
        valid_mode_means = [m for m in mode_means if m is not None]
        if len(valid_mode_means) >= 2:
            log["diag/arm_angle_mode_span"] = max(valid_mode_means) - min(valid_mode_means)
        if "lo" in z_means and "hi" in z_means:
            log["diag/arm_angle_z_span"] = abs(z_means["hi"] - z_means["lo"])
        if "lat" in y_means and "fwd" in y_means:
            log["diag/arm_angle_y_span"] = abs(y_means["fwd"] - y_means["lat"])
        wrist_pos = robot.data.joint_pos[:, env._yb_aat_wrist_ids]
        wrist_names = ["wrist_roll", "wrist_pitch", "wrist_yaw"]
        for j, wname in enumerate(wrist_names):
            log[f"wrist_drift/{wname}_mean"] = wrist_pos[:, j].mean().item()
            log[f"wrist_drift/{wname}_std"]  = wrist_pos[:, j].std().item()
        for k in range(n_phi):
            mk = masks[k]
            if mk.any():
                for j, wname in enumerate(wrist_names):
                    log[f"wrist_drift/m{k}_{wname}_mean"] = wrist_pos[mk, j].mean().item()
        sr_pos = robot.data.joint_pos[:, env._yb_aat_sr_ids[0]]
        log["posture/shoulder_roll_mean"] = sr_pos.mean().item()
        log["posture/shoulder_roll_std"]  = sr_pos.std().item()
        sr_mode_means: list[float | None] = [None] * n_phi
        for k in range(n_phi):
            mk = masks[k]
            if mk.any():
                sr_mode_means[k] = sr_pos[mk].mean().item()
                log[f"posture/m{k}_shoulder_roll_mean"] = sr_mode_means[k]
        valid_sr_mode_means = [m for m in sr_mode_means if m is not None]
        if len(valid_sr_mode_means) >= 2:
            log["diag/shoulder_roll_mode_span"] = max(valid_sr_mode_means) - min(valid_sr_mode_means)
        elbow_w = robot.data.body_pos_w[:, env._yb_aat_elbow_idx]
        elbow_b = elbow_w - robot.data.root_pos_w
        for j, ax in enumerate(["x", "y", "z"]):
            log[f"posture/elbow_b_{ax}_mean"] = elbow_b[:, j].mean().item()
            log[f"posture/elbow_b_{ax}_std"]  = elbow_b[:, j].std().item()
        elbow_mode_means: dict[str, list[float | None]] = {ax: [None] * n_phi for ax in ["x", "y", "z"]}
        for k in range(n_phi):
            mk = masks[k]
            if mk.any():
                for j, ax in enumerate(["x", "y", "z"]):
                    elbow_mode_means[ax][k] = elbow_b[mk, j].mean().item()
                    log[f"posture/m{k}_elbow_b_{ax}_mean"] = elbow_mode_means[ax][k]
        for ax in ["x", "y", "z"]:
            valid_axis_means = [m for m in elbow_mode_means[ax] if m is not None]
            if len(valid_axis_means) >= 2:
                log[f"diag/elbow_b_{ax}_mode_span"] = max(valid_axis_means) - min(valid_axis_means)

    return reward
