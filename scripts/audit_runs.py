"""
audit_runs.py

Scan multiple runs and print a comparison table of key metrics.

Usage:
  python scripts/audit_runs.py
  python scripts/audit_runs.py --pattern iter
  python scripts/audit_runs.py --pattern iter5 --full
  python scripts/audit_runs.py --pattern iter6 --last_n 50

Options:
  --logdir:   rsl_rl log root (default: unitree_rl_lab/.../yb_g1_arm_reach_mode_head)
  --pattern:  run name substring filter
  --last_n:   tail window size (default 200)
  --full:     also show per-mode and diagnostic columns
"""
from __future__ import annotations
import argparse
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


DEFAULT_LOGDIR = str(Path.home() / "unitree_rl_lab/logs/rsl_rl/yb_g1_arm_reach_mode_head")


CORE_TAGS = [
    "Train/mean_reward",
    "diag/arm_angle_mode_span",
    "diag/arm_angle_z_span",
    "diag/arm_angle_y_span",
    "arm_angle/mode0_mean",
    "arm_angle/mode1_mean",
    "arm_angle/mode2_mean",
    "arm_angle/gate_mean",
    "Loss/entropy",
    "Loss/value_function",
    "Loss/learning_rate",
]

FULL_TAGS = CORE_TAGS + [
    "arm_angle/z_lo_mean",
    "arm_angle/z_hi_mean",
    "arm_angle/y_fwd_mean",
    "arm_angle/y_lat_mean",
    "arm_angle/mean",
    "arm_angle/std",
    "Loss/mode_0/advantage_std",
    "Loss/mode_1/advantage_std",
    "Loss/mode_2/advantage_std",
    "Loss/mode_0/clip_fraction",
    "Loss/mode_1/clip_fraction",
    "Loss/mode_2/clip_fraction",
    "Loss/mode_0/worst_pick_rate",
    "Loss/mode_1/worst_pick_rate",
    "Loss/mode_2/worst_pick_rate",
    "Loss/worst_mode_alpha",
]


def load_run(run_dir: Path, tags: list[str]) -> dict[str, list[tuple[int, float]]]:
    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    try:
        ea.Reload()
    except Exception as e:
        return {"_error": str(e)}
    scalars = set(ea.Tags().get("scalars", []))
    out = {}
    for tag in tags:
        if tag in scalars:
            out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
        else:
            out[tag] = []
    out["_all_scalars"] = scalars  # for first-time inspection
    return out


def tail_mean(series: list[tuple[int, float]], last_n: int) -> float | None:
    if not series:
        return None
    return sum(v for _, v in series[-last_n:]) / min(len(series), last_n)


def max_iter(series: list[tuple[int, float]]) -> int | None:
    return series[-1][0] if series else None


def peak(series: list[tuple[int, float]]) -> float | None:
    if not series:
        return None
    return max(v for _, v in series)


def fmt(v: float | None, w: int = 8, prec: int = 3) -> str:
    if v is None:
        return f"{'—':>{w}}"
    return f"{v:>+{w}.{prec}f}" if v < 0 else f"{v:>{w}.{prec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=DEFAULT_LOGDIR)
    ap.add_argument("--pattern", default="", help="run name substring filter")
    ap.add_argument("--last_n", type=int, default=200)
    ap.add_argument("--full", action="store_true", help="also show per-mode and diagnostic columns")
    ap.add_argument("--list_tags", action="store_true", help="print all scalar tags from the first run and exit")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    runs = sorted(p for p in logdir.iterdir() if p.is_dir() and args.pattern in p.name)
    if not runs:
        print(f"no runs match pattern '{args.pattern}' under {logdir}")
        return

    tags = FULL_TAGS if args.full else CORE_TAGS

    if args.list_tags:
        first = load_run(runs[0], [])
        if "_error" in first:
            print(first["_error"])
            return
        print(f"== all scalar tags in {runs[0].name} ==")
        for t in sorted(first["_all_scalars"]):
            print(f"  {t}")
        return

    print(f"# audit_runs — {len(runs)} runs, last {args.last_n} iter mean\n")

    # header
    if args.full:
        header_groups = [
            ("run", 38),
            ("iters", 6),
            ("reward", 8),
            ("span", 7),
            ("z_sp", 7),
            ("y_sp", 7),
            ("mode0", 8),
            ("mode1", 8),
            ("mode2", 8),
            ("gate", 6),
            ("ent", 6),
            ("vf", 8),
            ("z_lo", 8),
            ("z_hi", 8),
            ("y_fwd", 8),
            ("y_lat", 8),
            ("adv0", 7),
            ("adv1", 7),
            ("adv2", 7),
            ("clip0", 6),
            ("clip1", 6),
            ("clip2", 6),
            ("alpha", 6),
            ("vf_max", 8),
        ]
    else:
        header_groups = [
            ("run", 38),
            ("iters", 6),
            ("reward", 8),
            ("span", 7),
            ("z_sp", 7),
            ("y_sp", 7),
            ("mode0", 8),
            ("mode1", 8),
            ("mode2", 8),
            ("gate", 6),
            ("ent", 6),
            ("vf", 8),
            ("vf_max", 8),
        ]

    header = "  ".join(f"{name:>{w}}" for name, w in header_groups)
    print(header)
    print("-" * len(header))

    for run in runs:
        data = load_run(run, tags)
        if "_error" in data:
            print(f"{run.name[:38]:>38}  ERROR: {data['_error']}")
            continue

        N = args.last_n
        cells = [run.name[:38]]
        # iters (max step across any tag)
        max_it = None
        for t in tags:
            mi = max_iter(data.get(t, []))
            if mi is not None and (max_it is None or mi > max_it):
                max_it = mi
        cells.append(f"{max_it if max_it is not None else '—':>6}")
        cells.append(fmt(tail_mean(data.get("Train/mean_reward", []), N), 8, 3))
        cells.append(fmt(tail_mean(data.get("diag/arm_angle_mode_span", []), N), 7, 3))
        cells.append(fmt(tail_mean(data.get("diag/arm_angle_z_span", []), N), 7, 3))
        cells.append(fmt(tail_mean(data.get("diag/arm_angle_y_span", []), N), 7, 3))
        cells.append(fmt(tail_mean(data.get("arm_angle/mode0_mean", []), N), 8, 3))
        cells.append(fmt(tail_mean(data.get("arm_angle/mode1_mean", []), N), 8, 3))
        cells.append(fmt(tail_mean(data.get("arm_angle/mode2_mean", []), N), 8, 3))
        cells.append(fmt(tail_mean(data.get("arm_angle/gate_mean", []), N), 6, 2))
        cells.append(fmt(tail_mean(data.get("Loss/entropy", []), N), 6, 2))
        cells.append(fmt(tail_mean(data.get("Loss/value_function", []), N), 8, 5))

        if args.full:
            cells.append(fmt(tail_mean(data.get("arm_angle/z_lo_mean", []), N), 8, 3))
            cells.append(fmt(tail_mean(data.get("arm_angle/z_hi_mean", []), N), 8, 3))
            cells.append(fmt(tail_mean(data.get("arm_angle/y_fwd_mean", []), N), 8, 3))
            cells.append(fmt(tail_mean(data.get("arm_angle/y_lat_mean", []), N), 8, 3))
            cells.append(fmt(tail_mean(data.get("Loss/mode_0/advantage_std", []), N), 7, 4))
            cells.append(fmt(tail_mean(data.get("Loss/mode_1/advantage_std", []), N), 7, 4))
            cells.append(fmt(tail_mean(data.get("Loss/mode_2/advantage_std", []), N), 7, 4))
            cells.append(fmt(tail_mean(data.get("Loss/mode_0/clip_fraction", []), N), 6, 3))
            cells.append(fmt(tail_mean(data.get("Loss/mode_1/clip_fraction", []), N), 6, 3))
            cells.append(fmt(tail_mean(data.get("Loss/mode_2/clip_fraction", []), N), 6, 3))
            cells.append(fmt(tail_mean(data.get("Loss/worst_mode_alpha", []), N), 6, 2))

        # vf_max over entire run: divergence signal
        cells.append(fmt(peak(data.get("Loss/value_function", [])), 8, 5))

        print("  ".join(cells))

    print()
    print("legend:")
    print("  span  = diag/arm_angle_mode_span (target >= 0.30)")
    print("  z_sp  = z_lo vs z_hi gap (chicken-wing diagnostic)")
    print("  y_sp  = y_lat vs y_fwd gap")
    print("  gate  = arm_angle/gate_mean (tracking proxy, higher is better)")
    print("  ent   = Loss/entropy (collapse signal: < 1)")
    print("  vf    = Loss/value_function (last_n mean)")
    print("  vf_max= Loss/value_function peak over full run (divergence: > 0.01)")
    if args.full:
        print("  adv_k = mode_k/advantage_std (< 0.02 risks gradient amplification)")
        print("  clip_k= mode_k/clip_fraction (> 0.3 means frequent PPO clip)")
        print("  alpha = worst_mode_alpha (worst-mode aggregation weight)")


if __name__ == "__main__":
    main()
