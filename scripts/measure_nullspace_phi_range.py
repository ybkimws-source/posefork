"""
measure_nullspace_phi_range.py

Directly measure SEW arm-angle (phi) null-space range at fixed EE targets.

Question: how much can phi vary across IK solutions for the same target?

Method:
  Collect N_RESTARTS random IK solutions per target (Adam, ee_err < threshold).
  Compute arm_angle(phi) for each valid solution.
  Report min/max/range/std of phi.
"""

import numpy as np
import torch
import pytorch_kinematics as pk
import pinocchio as pin

URDF_PATH  = "/home/nvidia/unitree_ros/robots/g1_description/g1_29dof_lock_waist_rev_1_0.urdf"
EE_LINK    = "right_wrist_yaw_link"
ELBOW_LINK = "right_elbow_link"
DEVICE     = "cpu"

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LB = torch.tensor([-3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144])
UB = torch.tensor([ 2.6704,  1.5882,  2.618,  2.0944,  1.9722,  1.6144,  1.6144])

N_RESTARTS    = 500
N_ITERS       = 120
LR            = 0.05
EE_ERR_THRESH = 0.015  # 1.5cm

SHOULDER_W = torch.tensor([-7.2e-6, -0.1002, 0.2918], dtype=torch.float32)

TARGETS = {
    "center":    [0.215, -0.280, 0.240],
    "z_lo_lat":  [0.215, -0.360, 0.170],
    "z_hi_fwd":  [0.215, -0.200, 0.310],
    "z_lo_fwd":  [0.215, -0.200, 0.170],
    "z_hi_lat":  [0.215, -0.360, 0.310],
    "far_deep":  [0.265, -0.350, 0.200],
}

print("Loading pytorch_kinematics chain (EE only for IK grad)...")
chain_ee  = pk.build_serial_chain_from_urdf(open(URDF_PATH).read(), EE_LINK)
chain_ee  = chain_ee.to(dtype=torch.float32, device=DEVICE)
joint_names = chain_ee.get_joint_parameter_names()
arm_indices = [joint_names.index(j) for j in RIGHT_ARM_JOINTS]
N_CHAIN = len(joint_names)

print("Loading pinocchio model (for elbow FK after IK)...")
pin_model = pin.buildModelFromUrdf(URDF_PATH)
pin_data  = pin.Data(pin_model)
pin_elbow_id = pin_model.getFrameId(ELBOW_LINK)
pin_arm_ids  = [pin_model.idx_qs[pin_model.getJointId(j)] for j in RIGHT_ARM_JOINTS]


def fk_ee(q_arm: torch.Tensor) -> torch.Tensor:
    """(N, 7) → ee_pos (N,3) — pytorch_kinematics, differentiable"""
    N = q_arm.shape[0]
    q_full = torch.zeros(N, N_CHAIN)
    q_full[:, arm_indices] = q_arm
    return chain_ee.forward_kinematics(q_full).get_matrix()[:, :3, 3]


def fk_elbow_pin(q_arm_np: np.ndarray) -> np.ndarray:
    """(N, 7) → elbow_pos (N,3) — pinocchio, no grad"""
    N = q_arm_np.shape[0]
    out = np.zeros((N, 3))
    q_pin = pin.neutral(pin_model)
    for i in range(N):
        for k, idx in enumerate(pin_arm_ids):
            q_pin[idx] = q_arm_np[i, k]
        pin.forwardKinematics(pin_model, pin_data, q_pin)
        pin.updateFramePlacements(pin_model, pin_data)
        out[i] = pin_data.oMf[pin_elbow_id].translation
    return out


def compute_arm_angle(shoulder, elbow, target):
    """(N,3) x3 → phi (N,), valid (N,)"""
    sw      = target - shoulder
    sw_unit = sw / sw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    se      = elbow - shoulder
    se_perp = se - (se * sw_unit).sum(-1, keepdim=True) * sw_unit
    valid   = se_perp.norm(dim=-1) > 1e-4
    z_down  = torch.tensor([0., 0., -1.]).expand_as(sw_unit)
    ref     = z_down - (z_down * sw_unit).sum(-1, keepdim=True) * sw_unit
    ref     = ref / ref.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cross   = torch.linalg.cross(ref, se_perp, dim=-1)
    sin_a   = (cross * sw_unit).sum(-1)
    cos_a   = (ref * se_perp).sum(-1)
    return torch.atan2(sin_a, cos_a), valid


PHI_TARGETS = [-1.15, -0.97, -0.80]
PHI_TOL = 0.10  # rad — tolerance for "solution exists near phi_k"

print(f"\n{'Target':<12} {'n_valid':>7} {'phi_mean':>9} {'phi_std':>8} {'phi_min':>8} {'phi_max':>8} {'phi_range':>10}  "
      + "  ".join([f"in[{p:+.2f}]" for p in PHI_TARGETS])
      + "  " + "  ".join([f"n~{p:+.2f}" for p in PHI_TARGETS]))
print("-" * 130)

for name, target_list in TARGETS.items():
    target_t = torch.tensor(target_list, dtype=torch.float32)
    t_batch  = target_t.unsqueeze(0).expand(N_RESTARTS, -1)

    q = torch.rand(N_RESTARTS, 7) * (UB - LB) + LB
    q.requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)

    for _ in range(N_ITERS):
        opt.zero_grad()
        ee = fk_ee(q)
        loss = ((ee - t_batch) ** 2).sum(dim=-1).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.data.clamp_(LB, UB)

    with torch.no_grad():
        q_det    = q.detach()
        ee_final = fk_ee(q_det)
        err      = torch.norm(ee_final - t_batch, dim=-1)
        valid_ee = err < EE_ERR_THRESH

        n_valid = valid_ee.sum().item()
        if n_valid < 3:
            print(f"{name:<12} {'<3 valid':>7}")
            continue

        q_valid_np   = q_det[valid_ee].numpy()
        elbow_np     = fk_elbow_pin(q_valid_np)
        elbow_v      = torch.tensor(elbow_np, dtype=torch.float32)
        target_v     = t_batch[valid_ee]
        shoulder_batch = SHOULDER_W.unsqueeze(0).expand(n_valid, -1)

        phi, phi_valid = compute_arm_angle(shoulder_batch, elbow_v, target_v)
        phi = phi[phi_valid].numpy()

        if len(phi) < 2:
            print(f"{name:<12} {'<2 phi_valid':>7}")
            continue

        p_min, p_max = float(phi.min()), float(phi.max())
        in_range = ["  Y  " if p_min <= pt <= p_max else "  .  " for pt in PHI_TARGETS]
        near_counts = [int(np.sum(np.abs(phi - pt) <= PHI_TOL)) for pt in PHI_TARGETS]
        print(f"{name:<12} {len(phi):>7} {phi.mean():>9.3f} {phi.std():>8.3f} "
              f"{p_min:>8.3f} {p_max:>8.3f} {p_max-p_min:>10.3f}  "
              + "  ".join([f"{s:>7}" for s in in_range])
              + "  " + "  ".join([f"{c:>7d}" for c in near_counts]))

print("\ninterpretation:")
print("  phi_range > 0.3 rad -> null-space allows mode separation")
print("  phi_range < 0.15 rad -> target geometry nearly fixes phi -> mode separation unlikely")
