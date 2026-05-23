"""
visualize_posture_freedom.py

G1 right arm — posture freedom heatmap by target position.

Method:
  Collect N random IK solutions per target position.
  Use shoulder_roll std among valid solutions as posture freedom metric.

  shoulder_roll std ≈ 0   -> solutions converge to a single posture (low freedom)
  shoulder_roll std ≈ large -> multiple posture families reachable (high freedom)
"""

import numpy as np
import torch
import pytorch_kinematics as pk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── config ────────────────────────────────────────────────────────────────────
URDF_PATH = "/home/nvidia/unitree_ros/robots/g1_description/g1_29dof_lock_waist_rev_1_0.urdf"
EE_LINK   = "right_wrist_yaw_link"
DEVICE    = "cpu"

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# joint limits (extracted from URDF)
LB = torch.tensor([-3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144])
UB = torch.tensor([ 2.6704,  1.5882,  2.618,  2.0944,  1.9722,  1.6144,  1.6144])

# IK parameters
N_RESTARTS    = 200   # random restarts per target
N_ITERS       = 80    # Adam iterations
LR            = 0.05
EE_ERR_THRESH = 0.04  # valid if EE error < 4cm

# target grid (3 slices)
X_RANGE = np.linspace(0.10, 0.38, 12)
Y_RANGE = np.linspace(-0.45, -0.02, 12)
Z_RANGE = np.linspace(0.02, 0.45, 12)

# proposed workspace range for overlay
PROP = dict(x=(0.22, 0.28), y=(-0.28, -0.15), z=(0.12, 0.28))

# ── pytorch_kinematics init ───────────────────────────────────────────────────
print("Loading chain...")
chain_full = pk.build_serial_chain_from_urdf(open(URDF_PATH).read(), EE_LINK)
chain_full = chain_full.to(dtype=torch.float32, device=DEVICE)
joint_names = chain_full.get_joint_parameter_names()
arm_indices = [joint_names.index(j) for j in RIGHT_ARM_JOINTS]
N_CHAIN = len(joint_names)
ROLL_IDX = 1  # shoulder_roll is index 1 in RIGHT_ARM_JOINTS

print(f"Chain joints ({N_CHAIN}): {joint_names}")
print(f"Arm indices: {arm_indices}")


def fk_batch(q_arm: torch.Tensor) -> torch.Tensor:
    """(N, 7) → (N, 3) EE position"""
    N = q_arm.shape[0]
    q_full = torch.zeros(N, N_CHAIN)
    q_full[:, arm_indices] = q_arm
    ret = chain_full.forward_kinematics(q_full)
    return ret.get_matrix()[:, :3, 3]


def compute_freedom(target: np.ndarray) -> dict:
    """
    Collect N_RESTARTS IK solutions for one target and compute posture freedom.

    Returns dict with:
      valid_frac   : fraction of valid solutions
      roll_std     : shoulder_roll std (primary freedom metric)
      roll_mean    : shoulder_roll mean
      n_valid      : number of valid solutions
    """
    t = torch.tensor(target, dtype=torch.float32).unsqueeze(0).expand(N_RESTARTS, -1)

    # uniform random initialization within joint limits
    q = torch.rand(N_RESTARTS, 7) * (UB - LB) + LB
    q.requires_grad_(True)
    opt = torch.optim.Adam([q], lr=LR)

    for _ in range(N_ITERS):
        opt.zero_grad()
        ee = fk_batch(q)
        loss = ((ee - t) ** 2).sum(dim=-1).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.data.clamp_(LB, UB)

    with torch.no_grad():
        ee_final = fk_batch(q.detach())
        err = torch.norm(ee_final - t, dim=-1)
        q_det = q.detach()

        # EE error threshold + shoulder_roll <= 0 (negative-only constraint)
        valid_task = err < EE_ERR_THRESH
        valid_roll = q_det[:, ROLL_IDX] <= 0.0
        valid = valid_task & valid_roll
        n_valid = valid.sum().item()
        n_task_only = valid_task.sum().item()

        if n_valid < 2:
            return dict(valid_frac=n_valid/N_RESTARTS, roll_std=0.0,
                        roll_mean=0.0, n_valid=n_valid,
                        n_task_only=n_task_only)

        valid_q = q_det[valid]
        roll_vals = valid_q[:, ROLL_IDX].numpy()
        return dict(
            valid_frac  = n_valid / N_RESTARTS,
            roll_std    = float(np.std(roll_vals)),
            roll_mean   = float(np.mean(roll_vals)),
            n_valid     = n_valid,
            n_task_only = n_task_only,
        )


# ── compute 3 slices ──────────────────────────────────────────────────────────
# 1. XZ slice at y = -0.20 (center of proposed range)
# 2. XY slice at z = 0.20
# 3. YZ slice at x = 0.25

slices = [
    dict(name="XZ (y=-0.20)", fixed_ax="y", fixed_val=-0.20,
         ax1="x", ax1_vals=X_RANGE, ax2="z", ax2_vals=Z_RANGE),
    dict(name="XY (z=0.20)", fixed_ax="z", fixed_val=0.20,
         ax1="x", ax1_vals=X_RANGE, ax2="y", ax2_vals=Y_RANGE),
    dict(name="YZ (x=0.25)", fixed_ax="x", fixed_val=0.25,
         ax1="y", ax1_vals=Y_RANGE, ax2="z", ax2_vals=Z_RANGE),
]

all_results = []
for sl in slices:
    print(f"\nComputing slice: {sl['name']}")
    results = []
    ax1v, ax2v = sl['ax1_vals'], sl['ax2_vals']
    total = len(ax1v) * len(ax2v)
    for i, a1 in enumerate(ax1v):
        for j, a2 in enumerate(ax2v):
            target = {sl['fixed_ax']: sl['fixed_val'], sl['ax1']: a1, sl['ax2']: a2}
            pt = np.array([target['x'], target['y'], target['z']])
            res = compute_freedom(pt)
            results.append((a1, a2, res))
            done = i * len(ax2v) + j + 1
            if done % 20 == 0:
                print(f"  {done}/{total}  ({a1:.2f}, {a2:.2f}) "
                      f"valid(roll≤0)={res['valid_frac']:.2f} "
                      f"task_only={res['n_task_only']/N_RESTARTS:.2f} "
                      f"roll_std={res['roll_std']:.3f}")
    all_results.append(results)
    print(f"  Done.")

# ── visualization ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("G1 Right Arm — Posture Freedom by Target Position  [roll ≤ 0 only]\n"
             "(color = shoulder_roll std among valid IK solutions with roll ≤ 0; "
             "higher = more diversity in safe region)", fontsize=13)

for col, (sl, results) in enumerate(zip(slices, all_results)):
    ax1_vals_u = sl['ax1_vals']
    ax2_vals_u = sl['ax2_vals']

    # grids
    roll_std_grid  = np.zeros((len(ax2_vals_u), len(ax1_vals_u)))
    valid_frac_grid = np.zeros_like(roll_std_grid)

    for (a1, a2, res) in results:
        i = np.argmin(np.abs(ax1_vals_u - a1))
        j = np.argmin(np.abs(ax2_vals_u - a2))
        roll_std_grid[j, i]   = res['roll_std']
        valid_frac_grid[j, i] = res['valid_frac']

    # top row: roll_std (posture freedom)
    ax_top = axes[0, col]
    im = ax_top.imshow(roll_std_grid, origin='lower',
                       extent=[ax1_vals_u[0], ax1_vals_u[-1],
                                ax2_vals_u[0], ax2_vals_u[-1]],
                       aspect='auto', cmap='viridis',
                       vmin=0, vmax=roll_std_grid.max())
    plt.colorbar(im, ax=ax_top, label='roll std (rad)')
    ax_top.set_xlabel(sl['ax1'])
    ax_top.set_ylabel(sl['ax2'])
    ax_top.set_title(f"{sl['name']}\nshoulder_roll std (diversity)")

    # proposed range overlay
    px = PROP[sl['ax1']]
    pz = PROP[sl['ax2']]
    rect = plt.Rectangle((px[0], pz[0]), px[1]-px[0], pz[1]-pz[0],
                          fill=False, edgecolor='red', linewidth=2, linestyle='--',
                          label='proposed range')
    ax_top.add_patch(rect)
    ax_top.legend(fontsize=8)

    # bottom row: valid_frac (reachability)
    ax_bot = axes[1, col]
    im2 = ax_bot.imshow(valid_frac_grid, origin='lower',
                        extent=[ax1_vals_u[0], ax1_vals_u[-1],
                                 ax2_vals_u[0], ax2_vals_u[-1]],
                        aspect='auto', cmap='plasma',
                        vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax_bot, label='valid frac')
    ax_bot.set_xlabel(sl['ax1'])
    ax_bot.set_ylabel(sl['ax2'])
    ax_bot.set_title(f"{sl['name']}\nreachability (valid IK fraction, roll≤0)")

    rect2 = plt.Rectangle((px[0], pz[0]), px[1]-px[0], pz[1]-pz[0],
                           fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax_bot.add_patch(rect2)

plt.tight_layout()
out_path = "/home/nvidia/yb_g1_rl_notes/2026_05_20/posture_freedom_map.png"
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.show()
