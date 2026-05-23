"""
probe_elbow_pose_by_phi.py

Measure elbow 3D position when phi is forced to various values (negative / positive)
at each EE target — chicken-wing diagnostic.

Question: does flipping the sign of phi_k (option B) actually lower the elbow naturally?

Method:
  Collect N_RESTARTS random IK solutions per target.
  Compute arm_angle(phi) for each valid solution.
  Report mean elbow position for solutions within ±0.10 rad of each phi candidate.

Key metrics:
  - elbow_z - shoulder_z  : positive = elbow above shoulder (chicken-wing risk)
  - |elbow_y - shoulder_y|: larger = more lateral elbow flare (chicken-wing risk)
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

N_RESTARTS    = 1500   # increased to adequately sample positive phi region
N_ITERS       = 120
LR            = 0.05
EE_ERR_THRESH = 0.015

SHOULDER_W = torch.tensor([-7.2e-6, -0.1002, 0.2918], dtype=torch.float32)

# 4 representative targets — z_lo is critical for chicken-wing diagnosis
TARGETS = {
    "z_lo_fwd":  [0.215, -0.200, 0.170],   # low + forward
    "z_lo_lat":  [0.215, -0.360, 0.170],   # low + lateral
    "z_hi_fwd":  [0.215, -0.200, 0.310],   # high + forward
    "z_hi_lat":  [0.215, -0.360, 0.310],   # high + lateral
}

# negative phi_k (prior) + positive phi_k (option B flip)
PHI_CANDIDATES = [-1.15, -0.97, -0.80, +0.80, +0.97, +1.15]
PHI_TOL = 0.10  # rad

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
    N = q_arm.shape[0]
    q_full = torch.zeros(N, N_CHAIN)
    q_full[:, arm_indices] = q_arm
    return chain_ee.forward_kinematics(q_full).get_matrix()[:, :3, 3]


def fk_elbow_pin(q_arm_np: np.ndarray) -> np.ndarray:
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


print(f"\nShoulder world: x={SHOULDER_W[0]:+.3f}, y={SHOULDER_W[1]:+.3f}, z={SHOULDER_W[2]:+.3f}")
print("chicken-wing metric: elbow_z - shoulder_z (positive = elbow above shoulder = risk)")
print("                     |elbow_y - shoulder_y| (larger = more lateral flare = risk)")
print()

for tname, target_list in TARGETS.items():
    target_t = torch.tensor(target_list, dtype=torch.float32)
    t_batch  = target_t.unsqueeze(0).expand(N_RESTARTS, -1)

    # IK with random restarts
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
        if n_valid < 5:
            print(f"== {tname}: target=({target_list}): <5 valid IK ({n_valid}) ==")
            continue

        q_valid_np  = q_det[valid_ee].numpy()
        elbow_np    = fk_elbow_pin(q_valid_np)
        elbow_v     = torch.tensor(elbow_np, dtype=torch.float32)
        target_v    = t_batch[valid_ee]
        shoulder_batch = SHOULDER_W.unsqueeze(0).expand(n_valid, -1)

        phi, phi_valid = compute_arm_angle(shoulder_batch, elbow_v, target_v)
        phi_np = phi[phi_valid].numpy()
        elbow_v_np = elbow_v[phi_valid].numpy()

        print(f"== {tname}: target=({target_list[0]:+.2f}, {target_list[1]:+.2f}, {target_list[2]:+.2f}), "
              f"n_valid_phi={len(phi_np)} ==")
        print(f"   {'phi_k':>7}  {'n_near':>7}  "
              f"{'elbow_x':>9}  {'elbow_y':>9}  {'elbow_z':>9}  "
              f"{'e_z-s_z':>9}  {'|ey-sy|':>9}  {'verdict':>14}")

        for phi_k in PHI_CANDIDATES:
            mask = np.abs(phi_np - phi_k) <= PHI_TOL
            n = int(mask.sum())
            if n < 2:
                print(f"   {phi_k:>+7.2f}  {n:>7}  (insufficient samples)")
                continue
            e_mean = elbow_v_np[mask].mean(axis=0)
            dz = e_mean[2] - SHOULDER_W[2].item()
            dy = abs(e_mean[1] - SHOULDER_W[1].item())

            # chicken-wing verdict — lateral flare (dy) and elbow height (dz) both matter
            # lateral flare is primary: elbow splaying sideways forces forearm inward = chicken-wing
            if dy > 0.15:
                wing = "WING-LATERAL"
            elif dz > 0.0:
                wing = "WING-HIGH"
            elif dy > 0.10:
                wing = "borderline"
            else:
                wing = "natural-tucked"
            print(f"   {phi_k:>+7.2f}  {n:>7}  "
                  f"{e_mean[0]:>+9.3f}  {e_mean[1]:>+9.3f}  {e_mean[2]:>+9.3f}  "
                  f"{dz:>+9.3f}  {dy:>9.3f}  {wing:>14}")
        print()

print("=" * 90)
print("verdict:")
print("  negative phi: CHICKEN-WING  +  positive phi: natural-down -> option B (sign flip) justified")
print("  positive phi also CHICKEN-WING -> option C (remap) needed, B alone insufficient")
print("  negative phi: natural-down -> root cause elsewhere (workspace, wrist, etc.) — re-examine")
